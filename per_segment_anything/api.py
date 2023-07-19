import typing as t
from pathlib import Path

import appdirs
import numpy as np
import torch
import torch.nn as nn
import wget

# from PIL import Image
from torch.nn import functional as F

from per_segment_anything import SamPredictor, sam_model_registry

# from ..show import show_box, show_mask, show_points


class Mask_Weights(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2, 1, requires_grad=True) / 3)


def point_selection(mask_sim, topk=1):
    # Top-1 point selection
    w, h = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = topk_xy - topk_x * h
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = np.array([1] * topk)
    topk_xy = topk_xy.cpu().numpy()

    # Top-last point selection
    last_xy = mask_sim.flatten(0).topk(topk, largest=False)[1]
    last_x = (last_xy // h).unsqueeze(0)
    last_y = last_xy - last_x * h
    last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
    last_label = np.array([0] * topk)
    last_xy = last_xy.cpu().numpy()

    return topk_xy, topk_label, last_xy, last_label


def calculate_dice_loss(inputs, targets, num_masks=1):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def calculate_sigmoid_focal_loss(
    inputs, targets, num_masks=1, alpha: float = 0.25, gamma: float = 2
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks


class PersamPredictor:
    def __init__(
        self, model_type: t.Literal["vit_h", "vit_b", "vit_l", "vit_t"]
    ) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if model_type == "vit_t":
            model_ckpt = "weights/mobile_sam.pt"
        else:
            model_ckpt = Path(appdirs.user_cache_dir("sam")) / f"{model_type}.pth"
            url_map = {
                "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            }
            model_ckpt.parent.mkdir(parents=True, exist_ok=True)
            if not model_ckpt.exists():
                wget.download(url_map[model_type], model_ckpt.as_posix())

        sam = sam_model_registry[model_type](checkpoint=model_ckpt).to(device=device)
        sam.eval()
        self.predictor = SamPredictor(sam)
        self.weights = None
        self.mask_weights = None

    def finetune(self, ic_image, ic_mask, epochs=1000, lr=1e-3):
        gt_mask = torch.tensor(ic_mask)[:, :, 0] > 0
        gt_mask = gt_mask.float().unsqueeze(0).flatten(1).cuda()

        print("======> Obtain Self Location Prior")
        # Image features encoding
        ref_mask = self.predictor.set_image(ic_image, ic_mask)
        ref_feat = self.predictor.features.squeeze().permute(1, 2, 0)

        ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0:2], mode="bilinear")
        ref_mask = ref_mask.squeeze()[0]

        # Target feature extraction
        self.target_feat = ref_feat[ref_mask > 0]
        target_feat_mean = self.target_feat.mean(0)
        target_feat_max = torch.max(self.target_feat, dim=0)[0]
        self.target_feat = (target_feat_max / 2 + target_feat_mean / 2).unsqueeze(0)

        # Cosine similarity
        h, w, C = ref_feat.shape
        self.target_feat = self.target_feat / self.target_feat.norm(
            dim=-1, keepdim=True
        )
        ref_feat = ref_feat / ref_feat.norm(dim=-1, keepdim=True)
        ref_feat = ref_feat.permute(2, 0, 1).reshape(C, h * w)
        sim = self.target_feat @ ref_feat

        sim = sim.reshape(1, 1, h, w)
        sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
        sim = self.predictor.model.postprocess_masks(
            sim,
            input_size=self.predictor.input_size,
            original_size=self.predictor.original_size,
        ).squeeze()

        # Positive location prior
        topk_xy, topk_label, _, _ = point_selection(sim, topk=1)

        print("======> Start Training")
        # Learnable mask weights
        if self.mask_weights is None:
            self.mask_weights = Mask_Weights().cuda()

        self.mask_weights.train()
        optimizer = torch.optim.AdamW(self.mask_weights.parameters(), lr=lr, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        for train_idx in range(epochs):
            # Run the decoder
            masks, scores, logits, logits_high = self.predictor.predict(
                point_coords=topk_xy, point_labels=topk_label, multimask_output=True
            )
            logits_high = logits_high.flatten(1)

            # Weighted sum three-scale masks
            weights = torch.cat(
                (
                    1 - self.mask_weights.weights.sum(0).unsqueeze(0),
                    self.mask_weights.weights,
                ),
                dim=0,
            )
            logits_high = logits_high * weights
            logits_high = logits_high.sum(0).unsqueeze(0)

            dice_loss = calculate_dice_loss(logits_high, gt_mask)
            focal_loss = calculate_sigmoid_focal_loss(logits_high, gt_mask)
            loss = dice_loss + focal_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if train_idx % 10 == 0:
                print("Train Epoch: {:} / {:}".format(train_idx, epochs))
                current_lr = scheduler.get_last_lr()[0]
                print(
                    "LR: {:.6f}, Dice_Loss: {:.4f}, Focal_Loss: {:.4f}".format(
                        current_lr, dice_loss.item(), focal_loss.item()
                    )
                )

        self.mask_weights.eval()
        self.weights = torch.cat(
            (
                1 - self.mask_weights.weights.sum(0).unsqueeze(0),
                self.mask_weights.weights,
            ),
            dim=0,
        )
        self.weights_np = self.weights.detach().cpu().numpy()
        print("======> Mask weights:\n", self.weights_np)

    def predict(self, test_image):
        if self.target_feat is None:
            raise ValueError("Please finetune first!")

        # Image feature encoding
        self.predictor.set_image(test_image)
        test_feat = self.predictor.features.squeeze()

        # Cosine similarity
        C, h, w = test_feat.shape
        test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
        test_feat = test_feat.reshape(C, h * w)
        sim = self.target_feat @ test_feat

        sim = sim.reshape(1, 1, h, w)
        sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
        sim = self.predictor.model.postprocess_masks(
            sim,
            input_size=self.predictor.input_size,
            original_size=self.predictor.original_size,
        ).squeeze()

        # Positive location prior
        topk_xy, topk_label, _, _ = point_selection(sim, topk=1)

        # First-step prediction
        masks, scores, logits, logits_high = self.predictor.predict(
            point_coords=topk_xy, point_labels=topk_label, multimask_output=True
        )

        # Weighted sum three-scale masks
        logits_high = logits_high * self.weights.unsqueeze(-1)
        logit_high = logits_high.sum(0)
        mask = (logit_high > 0).detach().cpu().numpy()

        logits = logits * self.weights_np[..., None]
        logit = logits.sum(0)

        # Cascaded Post-refinement-1
        y, x = np.nonzero(mask)
        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()
        input_box = np.array([x_min, y_min, x_max, y_max])
        masks, scores, logits, _ = self.predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            box=input_box[None, :],
            mask_input=logit[None, :, :],
            multimask_output=True,
        )
        best_idx = np.argmax(scores)

        # Cascaded Post-refinement-2
        y, x = np.nonzero(masks[best_idx])
        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()
        input_box = np.array([x_min, y_min, x_max, y_max])
        masks, scores, logits, _ = self.predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            box=input_box[None, :],
            mask_input=logits[best_idx : best_idx + 1, :, :],
            multimask_output=True,
        )
        best_idx = np.argmax(scores)

        final_mask = masks[best_idx]
        mask_colors = np.zeros(
            (final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8
        )
        mask_colors[final_mask, :] = np.array([[128, 0, 0]])

        return final_mask


class ZeroShotPersamPredictor:
    def __init__(
        self, model_type: t.Literal["vit_h", "vit_b", "vit_l", "vit_t"]
    ) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if model_type == "vit_t":
            model_ckpt = "weights/mobile_sam.pt"
        else:
            model_ckpt = Path(appdirs.user_cache_dir("sam")) / f"{model_type}.pth"
            url_map = {
                "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            }
            model_ckpt.parent.mkdir(parents=True, exist_ok=True)
            if not model_ckpt.exists():
                wget.download(url_map[model_type], model_ckpt.as_posix())

        sam = sam_model_registry[model_type](checkpoint=model_ckpt).to(device=device)
        sam.eval()
        self.predictor = SamPredictor(sam)
        self.target_embedding = None
        self.target_feat = None

    def finetune(self, ic_image, ic_mask):
        # Image features encoding
        ref_mask = self.predictor.set_image(ic_image, ic_mask)
        ref_feat = self.predictor.features.squeeze().permute(1, 2, 0)

        ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0:2], mode="bilinear")
        ref_mask = ref_mask.squeeze()[0]

        # Target feature extraction
        print("======> Obtain Location Prior")
        self.target_feat = ref_feat[ref_mask > 0]
        target_embedding = self.target_feat.mean(0).unsqueeze(0)
        self.target_feat = target_embedding / target_embedding.norm(
            dim=-1, keepdim=True
        )
        target_embedding = target_embedding.unsqueeze(0)

    def predict(self, test_image):
        if self.target_feat is None:
            raise ValueError("Please finetune first!")

        # Image feature encoding
        self.predictor.set_image(test_image)
        test_feat = self.predictor.features.squeeze()

        # Cosine similarity
        C, h, w = test_feat.shape
        test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
        test_feat = test_feat.reshape(C, h * w)
        sim = self.target_feat @ test_feat

        sim = sim.reshape(1, 1, h, w)
        sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
        sim = self.predictor.model.postprocess_masks(
            sim,
            input_size=self.predictor.input_size,
            original_size=self.predictor.original_size,
        ).squeeze()

        # Positive-negative location prior
        topk_xy_i, topk_label_i, last_xy_i, last_label_i = point_selection(sim, topk=1)
        topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
        topk_label = np.concatenate([topk_label_i, last_label_i], axis=0)

        # Obtain the target guidance for cross-attention layers
        sim = (sim - sim.mean()) / torch.std(sim)
        sim = F.interpolate(
            sim.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear"
        )
        attn_sim = sim.sigmoid_().unsqueeze(0).flatten(3)

        # First-step prediction
        masks, scores, logits, _ = self.predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            multimask_output=False,
            attn_sim=attn_sim,  # Target-guided Attention
            target_embedding=self.target_embedding,  # Target-semantic Prompting
        )
        best_idx = 0

        # Cascaded Post-refinement-1
        masks, scores, logits, _ = self.predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            mask_input=logits[best_idx : best_idx + 1, :, :],
            multimask_output=True,
        )
        best_idx = np.argmax(scores)

        # Cascaded Post-refinement-2
        y, x = np.nonzero(masks[best_idx])
        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()
        input_box = np.array([x_min, y_min, x_max, y_max])
        masks, scores, logits, _ = self.predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            box=input_box[None, :],
            mask_input=logits[best_idx : best_idx + 1, :, :],
            multimask_output=True,
        )
        best_idx = np.argmax(scores)

        final_mask = masks[best_idx]
        mask_colors = np.zeros(
            (final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8
        )
        mask_colors[final_mask, :] = np.array([[128, 0, 0]])

        return final_mask
