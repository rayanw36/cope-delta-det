"""Visualization utilities: draw detections, feature maps, Pareto curves."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# BDD100K color palette
CLASS_COLORS = [
    (1.0, 0.0, 0.0),       # pedestrian - red
    (0.0, 1.0, 0.0),       # rider - green
    (0.0, 0.0, 1.0),       # car - blue
    (1.0, 1.0, 0.0),       # truck - yellow
    (1.0, 0.0, 1.0),       # bus - magenta
    (0.0, 1.0, 1.0),       # train - cyan
    (1.0, 0.5, 0.0),       # motorcycle - orange
    (0.5, 0.0, 1.0),       # bicycle - purple
    (0.5, 1.0, 0.0),       # traffic light - lime
    (0.0, 0.5, 1.0),       # traffic sign - sky blue
]

CLASS_NAMES = [
    'pedestrian', 'rider', 'car', 'truck', 'bus',
    'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign'
]


def draw_detections(image, boxes, scores, labels, save_path=None, title=None):
    """Draw bounding boxes on an image.

    Args:
        image: np.array (H, W, 3) in RGB, values [0, 255]
        boxes: np.array (N, 4) in xyxy format
        scores: np.array (N,) confidence scores
        labels: np.array (N,) class indices
        save_path: optional path to save figure
        title: optional title string
    """
    fig, ax = plt.subplots(1, figsize=(16, 9))
    ax.imshow(image.astype(np.uint8))

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        cls_idx = int(labels[i])
        color = CLASS_COLORS[cls_idx % len(CLASS_COLORS)]
        score = scores[i]

        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 5,
            f'{CLASS_NAMES[cls_idx]} {score:.2f}',
            color='white', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7)
        )

    if title:
        ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_motion_vectors(mv_tensor, save_path=None, title='Motion Vectors'):
    """Visualize motion vector field as quiver plot.

    Args:
        mv_tensor: np.array (H, W, 2) with (mvx, mvy)
        save_path: optional save path
        title: plot title
    """
    h, w = mv_tensor.shape[:2]
    Y, X = np.mgrid[0:h, 0:w]
    U = mv_tensor[:, :, 0]
    V = mv_tensor[:, :, 1]

    fig, ax = plt.subplots(1, figsize=(12, 8))
    magnitude = np.sqrt(U**2 + V**2)
    ax.quiver(X, Y, U, -V, magnitude, cmap='jet', scale=50)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_title(title)
    ax.set_aspect('equal')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_residual_energy(energy_map, save_path=None, title='Residual Energy'):
    """Visualize residual energy heatmap.

    Args:
        energy_map: np.array (H, W) or (H, W, 1)
        save_path: optional save path
        title: plot title
    """
    if energy_map.ndim == 3:
        energy_map = energy_map.squeeze(-1)

    fig, ax = plt.subplots(1, figsize=(12, 8))
    im = ax.imshow(energy_map, cmap='hot', interpolation='nearest')
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_partition_depth(depth_map, save_path=None, title='CU Partition Depth'):
    """Visualize CU partition depth map.

    Args:
        depth_map: np.array (H, W) or (H, W, 1) with values 0-3
    """
    if depth_map.ndim == 3:
        depth_map = depth_map.squeeze(-1)

    fig, ax = plt.subplots(1, figsize=(12, 8))
    im = ax.imshow(depth_map, cmap='viridis', vmin=0, vmax=3, interpolation='nearest')
    plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_pareto_curve(decode_budgets, mAPs, method_names=None, save_path=None):
    """Plot mAP vs decode budget Pareto curve.

    Args:
        decode_budgets: list of lists, decode % per method
        mAPs: list of lists, mAP per method
        method_names: list of str
        save_path: optional save path
    """
    fig, ax = plt.subplots(1, figsize=(10, 7))
    markers = ['o', 's', '^', 'D', 'v', 'p']

    for i in range(len(decode_budgets)):
        label = method_names[i] if method_names else f'Method {i}'
        ax.plot(
            decode_budgets[i], mAPs[i],
            marker=markers[i % len(markers)],
            label=label, linewidth=2, markersize=8
        )

    ax.set_xlabel('Decode Budget (%)', fontsize=14)
    ax.set_ylabel('mAP@0.5', fontsize=14)
    ax.set_title('Detection Accuracy vs Decode Budget', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_codec_robustness_heatmap(qp_values, gop_lengths, mAP_grid, save_path=None):
    """Plot QP x GOP heatmap of mAP values.

    Args:
        qp_values: list of QP values
        gop_lengths: list of GOP lengths
        mAP_grid: np.array (len(qp), len(gop)) of mAP values
        save_path: optional save path
    """
    fig, ax = plt.subplots(1, figsize=(8, 6))
    im = ax.imshow(mAP_grid, cmap='YlOrRd_r', aspect='auto')

    ax.set_xticks(range(len(gop_lengths)))
    ax.set_xticklabels([str(g) for g in gop_lengths])
    ax.set_yticks(range(len(qp_values)))
    ax.set_yticklabels([str(q) for q in qp_values])
    ax.set_xlabel('GOP Length')
    ax.set_ylabel('QP Value')
    ax.set_title('mAP@0.5 — Codec Robustness')

    # Annotate cells
    for i in range(len(qp_values)):
        for j in range(len(gop_lengths)):
            ax.text(j, i, f'{mAP_grid[i, j]:.1f}',
                    ha='center', va='center', fontsize=12, fontweight='bold')

    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
