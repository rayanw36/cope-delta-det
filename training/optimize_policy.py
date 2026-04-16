import os
import sys
import torch
import json
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import BDD100KCoPEDataset
from models.cope_delta_det import CoPEDeltaDet
from evaluation.evaluate import evaluate_cope_delta_det

def run_policy_grid_search(device='cuda'):
    print(f"Executing Stage 3: Saliency Policy Evaluation Sweep on {device}...")

    # Load dataset
    base_dir = Path(__file__).resolve().parent.parent / "data" / "bdd100k"
    dataset = BDD100KCoPEDataset(
        root_dir=str(base_dir),
        split='val', # Use val set for evaluation tuning!
        gop_length=16
    )

    # Initialize model
    model = CoPEDeltaDet(yolo_size='yolov8n.pt', embed_dim=256, num_classes=10, device=device).to(device)

    # Load previously trained finetuned weights (Stage 2 output)
    stage2_weights = "D:/cope-delta-det2/checkpoints/finetuned_model_epoch_10.pt"
    if os.path.exists(stage2_weights):
        print(f"Loading Stage 2 tracked weights from {stage2_weights}...")
        model.load_state_dict(torch.load(stage2_weights, map_location=device))
    else:
        print("WARNING: Stage 2 weights not found. Using untrained primitives.")

    # Define hyperparameter grid bounds
    w1_vals = [0.8, 1.0, 1.2]
    w2_vals = [0.8, 1.0, 1.2]
    thresh_vals = [0.3, 0.5, 0.7]

    best_mAP = 0.0
    best_params = {}
    all_results = []

    print("\nStarting Hyperparameter Grid Search...")
    total_iters = len(w1_vals) * len(w2_vals) * len(thresh_vals)
    curr_iter = 1

    for w1 in w1_vals:
        for w2 in w2_vals:
            for thresh in thresh_vals:
                print(f"\n[Iteration {curr_iter}/{total_iters}] Testing config -> w1: {w1}, w2: {w2}, thresh: {thresh}")
                
                # Evaluate full validation set under this policy configuration
                results = evaluate_cope_delta_det(
                    model, 
                    dataset, 
                    device, 
                    measure_latency=False,
                    policy_w1=w1,
                    policy_w2=w2,
                    policy_thresh=thresh
                )
                
                map_50 = results['mAP_50']
                decode_budget = results['decode_budget'] # Lower is faster!
                
                print(f"==> Result: mAP@0.5: {map_50*100:.2f}%, Decode Budget: {decode_budget:.1f}%")
                
                config_res = {
                    'w1': w1, 'w2': w2, 'thresh': thresh,
                    'mAP_50': map_50,
                    'decode_budget': decode_budget
                }
                all_results.append(config_res)

                # Simple selection criteria: maximize mAP primarily. 
                # Ideally, you'd maximize (mAP - lambda * decode_budget) to penalize frequent YOLO decodings.
                if map_50 > best_mAP:
                    best_mAP = map_50
                    best_params = config_res

                curr_iter += 1

    print("\n" + "="*50)
    print("Stage 3 Grid Search Complete!")
    print(f"Best Configuration Found: {best_params}")
    
    with open('D:/cope-delta-det2/checkpoints/stage3_policy_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)
        print("Exported full sweep metrics to stage3_policy_results.json")

if __name__ == '__main__':
    run_policy_grid_search(device='cuda' if torch.cuda.is_available() else 'cpu')
