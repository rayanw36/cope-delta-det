# CoPE-Delta-Det: Compressed-Domain Object Detection via Delta-Tokens in HEVC Video

> **Codec-aware temporal object detection that avoids full pixel reconstruction on P-frames.**

CoPE-Delta-Det adapts the delta-token concept from [CoPE-VideoLM](https://arxiv.org/abs/xxx) to spatially localized, per-object detection updates using HEVC compressed-domain primitives (motion vectors, residual energy, partition depth, prediction modes). Only I-frames are fully decoded; P-frames are processed directly from codec statistics at 1/16 resolution, yielding significant decode savings with minimal accuracy loss.

## Architecture

```
I-frame --> YOLO Anchor Detector --> Anchor Boxes + Embeddings (queries)
                                            |
P-frame --> HEVC Parser --> [MV, Residual, Depth, Mode] (45x80 grid)
                                  |
                         Delta-Det Encoder
                        /                  \
                Motion Branch          Appearance Branch
              (MV -> cross-attn)    (Res+Depth+Mode -> conv -> cross-attn)
                        \                  /
                     8 Delta-tokens per object (key/value)
                                  |
                        Temporal Fusion Head
                      (3-layer Transformer Decoder)
                                  |
                   Box Refinement + Confidence + Classification
```

### Key Components

| Module | File | Description |
|--------|------|-------------|
| YOLO Anchor | `models/yolo_anchor.py` | Frozen YOLOv8 for I-frame detection and feature extraction |
| Delta Encoder | `models/delta_encoder.py` | Dual-branch encoder: motion (MVs) + appearance (residual/depth/mode) |
| Temporal Fusion | `models/temporal_fusion.py` | Transformer decoder fusing anchor queries with delta-tokens |
| Refresh Policy | `models/refresh_policy.py` | Saliency-based policy deciding when to re-run full detection |
| Full Pipeline | `models/cope_delta_det.py` | End-to-end CoPE-Delta-Det forward pass |

## Supported Datasets

### ImageNet VID (ILSVRC 2015)
- **30 object classes**, 3862 train + 555 val video snippets
- Real per-frame bounding box annotations with tracking IDs
- Primary evaluation dataset with dense temporal ground truth

### BDD100K
- **10 object classes**, driving video at 30fps
- Single annotated keyframe per video with MV-based box propagation
- Legacy support maintained

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

- Python >= 3.10
- PyTorch >= 2.0 with CUDA
- FFmpeg with HEVC encoder (`libx265` or `hevc_nvenc` for GPU)
- [Ultralytics](https://github.com/ultralytics/ultralytics) YOLOv8

### Dataset Preparation (ImageNet VID)

1. **Download** from [HuggingFace mirror](https://huggingface.co/datasets/guanxiongsun/imagenetvid):

```bash
# Download video data (~92 GB in two parts)
huggingface-cli download guanxiongsun/imagenetvid ILSVRC2015_VID.tar.gz.aa --repo-type dataset --local-dir ./data/imagenetvid
huggingface-cli download guanxiongsun/imagenetvid ILSVRC2015_VID.tar.gz.ab --repo-type dataset --local-dir ./data/imagenetvid

# Download pre-processed COCO-style annotations (~60 MB)
huggingface-cli download guanxiongsun/imagenetvid annotations.tar.gz --repo-type dataset --local-dir ./data/imagenetvid
```

2. **Extract**:

```bash
cd data/imagenetvid
cat ILSVRC2015_VID.tar.gz.aa ILSVRC2015_VID.tar.gz.ab > ILSVRC2015_VID.tar.gz
tar -xzf ILSVRC2015_VID.tar.gz
tar -xzf annotations.tar.gz
```

3. **Build GOP index** (groups frames into GOPs of 16 with per-frame annotations scaled to 720x1280):

```bash
python data/prepare_imagenetvid.py --vid_root ./data/imagenetvid --gop_length 16 --max_gops_per_video_train 6
```

4. **Verify annotations visually** (optional but recommended):

```bash
python visualize_imagenetvid.py --split train --num_gops 10 --random
# Opens vid_annotation_check_train.avi with overlaid bounding boxes
```

## Training Pipeline

### Phase 1: HEVC Encoding

Re-encode JPEG frame sequences into HEVC with controlled parameters (QP=22, GOP=16, no B-frames):

```bash
python data/encode_imagenetvid_hevc.py --vid_root ./data/imagenetvid --qp 22 --gop 16 --workers 4
```

### Phase 2: Feature Extraction

Extract compressed-domain primitives (MVs, residual energy, partition depth, prediction modes) from HEVC files:

```bash
python data/extract_features.py --hevc_dir ./data/imagenetvid/hevc --output_dir ./data/imagenetvid/features --device cuda --workers 3
```

### Phase 3: Fine-tune YOLO on VID Classes

Fine-tune YOLOv8m on all 30 ImageNet VID classes (uses ~57k official sparse training frames):

```bash
python training/finetune_yolo_vid.py --vid_root ./data/imagenetvid --epochs 30 --batch 16 --imgsz 640 --device 0
```

Output: `runs/detect/yolov8m_vid/weights/best.pt`

### Phase 4: Stage 1 - Delta Encoder Pre-training (optional)

Pre-train the delta encoder with reconstruction loss:

```bash
python training/pretrain.py --dataset imagenetvid --root ./data/imagenetvid \
    --yolo_weights runs/detect/yolov8m_vid/weights/best.pt --num_classes 30 --epochs 1
```

### Phase 5: Stage 2 - End-to-End Fine-tuning

Train the full CoPE-Delta-Det pipeline with detection loss:

```bash
python training/finetune.py \
    --dataset imagenetvid \
    --root ./data/imagenetvid \
    --num_classes 30 \
    --yolo_weights runs/detect/yolov8m_vid/weights/best.pt \
    --class_mapping identity \
    --epochs 30 \
    --prefix vid_
```

Output: `checkpoints/vid_finetuned_best.pt`

## Evaluation

### Main Evaluation

```bash
python evaluation/evaluate.py \
    --dataset imagenetvid \
    --root ./data/imagenetvid \
    --num_classes 30 \
    --yolo_weights runs/detect/yolov8m_vid/weights/best.pt \
    --class_mapping identity \
    --split val \
    --checkpoint checkpoints/vid_finetuned_best.pt \
    --output results/vid_eval.json
```

### Baselines

```bash
python evaluation/evaluate_baselines.py \
    --dataset imagenetvid \
    --root ./data/imagenetvid \
    --num_classes 30 \
    --yolo_weights runs/detect/yolov8m_vid/weights/best.pt \
    --mode yolo_full          # Per-frame YOLO (accuracy ceiling)
    # Also: yolo_iframe_only, cope, mean_mv
```

### Metrics

| Metric | Description |
|--------|-------------|
| mAP@0.5 | COCO-style mean Average Precision at IoU=0.5 |
| mAP@[.5:.95] | COCO-style mAP averaged over IoU thresholds 0.5 to 0.95 |
| Decode Budget | Percentage of frames requiring full pixel decode |
| Latency/FPS | Per-frame inference time |

### Additional Evaluation Modes

- **Codec robustness sweep**: `evaluation/codec_robustness.py` — QP x GOP grid (4x3 = 12 configs)
- **Ablation studies**: `evaluation/ablations.py` — branch, fusion head, refresh policy, token count
- **Pareto analysis**: `evaluation/pareto.py` — mAP vs decode budget curves

## Baselines

| # | Method | File | Description |
|---|--------|------|-------------|
| 1 | Per-frame YOLO | `baselines/full_decode_yolo.py` | Full decode every frame (accuracy ceiling) |
| 2 | Mean-MV Propagation | `baselines/mean_mv.py` | Average MVs to shift boxes (no learning) |
| 3 | MMNet | `baselines/mmnet.py` | Feature-level MV warping + residual correction |
| 4 | Chen et al. HEVC Intra | `baselines/chen_hevc_intra.py` | Compressed-domain I-frame detection |
| 5 | BAFE + BiLSTM | `baselines/bafe.py` | Box-aligned features with recurrent temporal head |

## Project Structure

```
cope-delta-det2/
├── models/                     # Core model architecture
│   ├── cope_delta_det.py       # End-to-end pipeline
│   ├── delta_encoder.py        # Dual-branch delta encoder
│   ├── temporal_fusion.py      # Transformer fusion head
│   ├── yolo_anchor.py          # Frozen YOLO anchor detector
│   └── refresh_policy.py       # Saliency refresh policy
├── training/                   # Training scripts
│   ├── pretrain.py             # Stage 1: delta encoder pre-training
│   ├── finetune.py             # Stage 2: end-to-end fine-tuning
│   ├── finetune_yolo_vid.py    # YOLO fine-tune on VID 30 classes
│   └── losses.py               # Detection losses (GIoU + focal + L1)
├── evaluation/                 # Evaluation and analysis
│   ├── evaluate.py             # Main CoPE-Delta-Det evaluation
│   ├── evaluate_baselines.py   # Baseline comparisons
│   ├── ablations.py            # Component ablation studies
│   ├── codec_robustness.py     # QP x GOP robustness sweep
│   └── pareto.py               # Pareto frontier analysis
├── baselines/                  # Baseline implementations
├── data/                       # Data pipeline
│   ├── prepare_imagenetvid.py  # Build GOP index with per-frame GT
│   ├── encode_imagenetvid_hevc.py  # JPEG sequences -> HEVC encoding
│   ├── extract_features.py     # HEVC -> compressed-domain features
│   ├── dataset.py              # PyTorch dataset (VID + BDD100K)
│   └── prepare_bdd100k.py      # BDD100K preparation (legacy)
├── utils/                      # Utilities
│   ├── metrics.py              # COCO mAP, latency, decode tracking
│   ├── box_utils.py            # Box format conversions
│   ├── hevc_parser.py          # HEVC bitstream parsing
│   └── visualization.py        # Detection visualization
├── configs/                    # YAML configuration files
├── scripts/                    # Shell scripts for full pipeline runs
├── visualize_imagenetvid.py    # Visual QA for VID annotations
├── visualize_dataset.py        # Visual QA for BDD100K annotations
├── requirements.txt
└── README.md
```

## Citation

If you use this code, please cite:

```bibtex
@misc{cope-delta-det,
  title={CoPE-Delta-Det: Compressed-Domain Object Detection via Delta-Tokens in HEVC Video},
  year={2025},
}
```

## License

This project is for academic research purposes.
