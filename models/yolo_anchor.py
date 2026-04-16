import torch
import torch.nn as nn
from ultralytics import YOLO

class YOLOAnchor(nn.Module):
    """
    YOLO wrapper for I-frame anchor detection and ground-truth feature extraction.

    Parameters
    ----------
    model_size : str
        Path to the YOLO checkpoint. ``yolov8m.pt`` is the COCO-pretrained
        release; pass a fine-tuned path (e.g. ``runs/detect/yolov8m_vid/weights/best.pt``)
        to use a custom detector.
    class_mapping : dict | str | None
        How to remap detector class IDs to downstream task class IDs.
        * None or 'identity'  — pass classes through as-is.
        * 'coco_to_bdd'       — use the legacy BDD100K mapping.
        * dict                — explicit {detector_cls: target_cls}.
    """
    def __init__(self, model_size='yolov8m.pt', device='cuda' if torch.cuda.is_available() else 'cpu',
                 class_mapping='coco_to_bdd'):
        super().__init__()
        # Load YOLOv8 model
        yolo_instance = YOLO(model_size)
        self._yolo_wrap = [yolo_instance] # Hide from PyTorch nn.Module recursive train()
        self.device = device

        # We want to use the PyTorch model inside YOLO
        self.model = yolo_instance.model.to(device)
        self.model.eval()

        # Freeze all YOLO weights (will only be used for feature extraction and anchors)
        for param in self.model.parameters():
            param.requires_grad = False

        if class_mapping == 'coco_to_bdd':
            self.class_mapping = {0: 0, 1: 7, 2: 2, 3: 6, 5: 4, 6: 5, 7: 3, 9: 8, 11: 9}
        elif class_mapping in (None, 'identity'):
            self.class_mapping = None
        elif isinstance(class_mapping, dict):
            self.class_mapping = dict(class_mapping)
        else:
            raise ValueError(f"Unknown class_mapping: {class_mapping}")

    def train(self, mode=True):
        """Override train to ensure YOLO backbone NEVER switches to train mode which breaks ultralytics NMS."""
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def get_anchor_boxes(self, iframe_rgb):
        """
        Runs YOLO inference on the fully decoded I-frame RGB to get anchor bounding boxes.
        Returns boxes in [x, y, w, h] format.
        """
        # ultralytics expects frames usually as numpy or lists, but can take tensors [B, 3, H, W]
        # values should be normalized to [0, 1] or [0, 255] depending on preprocessing step.
        # usually model directly handles preprocessing if we use the top-level predictor,
        # but here we use the raw model if passing tensors.
        import torch.nn.functional as F
        
        # Ensure dimensions are multiples of 32 (YOLO max stride constraint)
        _, _, h, w = iframe_rgb.shape
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h > 0 or pad_w > 0:
            iframe_rgb = F.pad(iframe_rgb, (0, pad_w, 0, pad_h))
            
        results = self._yolo_wrap[0](iframe_rgb, verbose=False)
        
        anchors = []
        for r in results:
            boxes = r.boxes
            if len(boxes) > 0:
                xyxy = boxes.xyxy
                confs = boxes.conf.unsqueeze(1)
                det_cls = boxes.cls.long()

                if self.class_mapping is None:
                    # Identity mapping — use detector class IDs directly.
                    out_cls = det_cls
                    valid_mask = torch.ones_like(det_cls, dtype=torch.bool)
                else:
                    out_cls = torch.full_like(det_cls, -1)
                    for src_id, tgt_id in self.class_mapping.items():
                        out_cls[det_cls == src_id] = tgt_id
                    valid_mask = out_cls >= 0

                if not valid_mask.any():
                    anchors.append(torch.empty((0, 6), device=self.device))
                else:
                    xyxy = xyxy[valid_mask]
                    confs = confs[valid_mask]
                    out_cls = out_cls[valid_mask].float().unsqueeze(1)

                    anchor = torch.cat([xyxy, confs, out_cls], dim=1)
                    anchors.append(anchor)
            else:
                anchors.append(torch.empty((0, 6), device=self.device))

        return anchors

    @torch.no_grad()
    def extract_backbone_features(self, rgb_tensor):
        """
        Used during Stage 1 pre-training.
        Extracts multi-scale features from YOLO backbone for RoI-Align based GT generation.
        """
        # Forward pass through the backbone
        # YOLOv8 returns a list of features from P3 to P5
        import torch.nn.functional as F
        
        # Ensure dimensions are multiples of 32 (YOLO max stride constraint)
        _, _, h, w = rgb_tensor.shape
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h > 0 or pad_w > 0:
            rgb_tensor = F.pad(rgb_tensor, (0, pad_w, 0, pad_h))
            
        preds = self.model(rgb_tensor)
        
        # preds will typically be a tuple where preds[0] are detection head outputs, 
        # and preds[1] is a tuple of intermediate feature maps (e.g. at stride 8, 16, 32).
        
        # Return the feature maps
        if isinstance(preds, dict):
            features = preds.get('feats', preds)
        elif isinstance(preds, tuple) and len(preds) > 1:
            if isinstance(preds[1], dict):
                features = preds[1].get('feats', preds[1])
            else:
                features = preds[1]
        else:
            features = preds
            
        return features

if __name__ == '__main__':
    # Simple test
    detector = YOLOAnchor('yolov8n.pt', device='cpu')
    dummy_input = torch.rand(1, 3, 640, 640)
    print("Testing YOLO Anchor component...")
    
    # We can test extracting features
    features = detector.extract_backbone_features(dummy_input)
    if isinstance(features, (list, tuple)):
        for i, f in enumerate(features):
            print(f"Feature path {i}: {f.shape}")
