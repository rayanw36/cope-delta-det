import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou

class DetectionLoss(nn.Module):
    """
    Combined loss function for CoPE-Δ-Det fine-tuning.
    L_total = λ1 * L_box(L1) + λ2 * L_GIoU + λ3 * L_cls(Focal) + λ4 * L_conf(BCE)
    """
    def __init__(self, lambda_box=5.0, lambda_giou=2.0, lambda_cls=2.0, lambda_conf=1.0):
        super().__init__()
        self.lambda_box = lambda_box
        self.lambda_giou = lambda_giou
        self.lambda_cls = lambda_cls
        self.lambda_conf = lambda_conf

    def focal_loss(self, inputs, targets, alpha=0.25, gamma=2.0):
        """
        Compute focal loss for classification.
        inputs: [N, num_classes] raw logits
        targets: [N] class indices
        """
        # Cross entropy loss with no reduction
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        f_loss = alpha * (1 - pt)**gamma * ce_loss
        return f_loss.mean()

    def forward(self, pred_boxes, pred_cls, target_boxes, target_cls,
                pred_conf_matched=None, pred_conf_unmatched=None,
                img_h=720, img_w=1280):
        """
        Assumes predictions and targets are matched (e.g. 1-to-1 matching done prior)
        pred_boxes: [N, 4] format [x1, y1, x2, y2] in pixel coordinates
        pred_cls: [N, num_classes] raw logits
        target_boxes: [N, 4] format [x1, y1, x2, y2] in pixel coordinates
        target_cls: [N] target class indices
        img_h, img_w: image dimensions for normalizing L1 loss
        """
        zero_terms = []
        for tensor in (pred_boxes, pred_cls, pred_conf_matched, pred_conf_unmatched):
            if tensor is not None:
                zero_terms.append(tensor.sum() * 0)
        zero = sum(zero_terms) if zero_terms else torch.tensor(0.0, device=target_boxes.device if target_boxes.numel() > 0 else 'cpu')

        if pred_boxes.shape[0] > 0:
            # 1. L1 Box Loss — normalize to [0, 1] so loss scale is independent of resolution
            norm = torch.tensor([img_w, img_h, img_w, img_h],
                               device=pred_boxes.device, dtype=pred_boxes.dtype)
            loss_l1 = F.l1_loss(pred_boxes / norm, target_boxes / norm, reduction='mean')

            # 2. GIoU Loss (scale-invariant, no normalization needed)
            giou_matrix = generalized_box_iou(pred_boxes, target_boxes)
            giou = torch.diag(giou_matrix)
            loss_giou = 1 - giou.mean()

            # 3. Focal Classification Loss
            loss_cls = self.focal_loss(pred_cls, target_cls)
        else:
            loss_l1 = zero
            loss_giou = zero
            loss_cls = zero

        conf_losses = []
        if pred_conf_matched is not None and pred_conf_matched.numel() > 0:
            pos_targets = torch.ones_like(pred_conf_matched)
            conf_losses.append(F.binary_cross_entropy(pred_conf_matched.clamp(1e-6, 1 - 1e-6), pos_targets))
        if pred_conf_unmatched is not None and pred_conf_unmatched.numel() > 0:
            neg_targets = torch.zeros_like(pred_conf_unmatched)
            conf_losses.append(F.binary_cross_entropy(pred_conf_unmatched.clamp(1e-6, 1 - 1e-6), neg_targets))
        loss_conf = torch.stack(conf_losses).mean() if conf_losses else zero

        total_loss = (
            self.lambda_box * loss_l1 +
            self.lambda_giou * loss_giou +
            self.lambda_cls * loss_cls +
            self.lambda_conf * loss_conf
        )

        return total_loss, {
            'l1': loss_l1.item(),
            'giou': loss_giou.item(),
            'cls': loss_cls.item(),
            'conf': loss_conf.item(),
            'total': total_loss.item()
        }

if __name__ == '__main__':
    loss_fn = DetectionLoss()
    pred_b = torch.tensor([[10., 10., 20., 20.]])
    tgt_b = torch.tensor([[10., 11., 20., 21.]])
    pred_c = torch.randn(1, 10)
    tgt_c = torch.tensor([3])
    
    loss, metrics = loss_fn(pred_b, pred_c, tgt_b, tgt_c)
    print(metrics)
