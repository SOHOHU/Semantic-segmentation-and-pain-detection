import json
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
PAIN_SCORE_DIR = ROOT / "results" / "pain_score"
PAIN_SCORE_DIR.mkdir(parents=True, exist_ok=True)

def get_ground_truth(filename):
    if 'stress' in filename.lower() or 'pain' in filename.lower():
        return 1
    elif 'normal' in filename.lower():
        return 0
    else:
        return -1

def load_data(json_data, split='train'):
    X = []
    y = []
    filenames = []
    
    # Order of parts to ensure consistency
    parts_order = ['ear', 'eyes', 'face', 'mouth', 'muscles_above_eye', 'nose']
    
    for filename, data in json_data[split].items():
        gt = get_ground_truth(filename)
        if gt == -1: continue
        
        features = []
        if 'per_part' not in data: continue
        
        for part in parts_order:
            if part in data['per_part'] and 'pred_label' in data['per_part'][part]:
                features.append(data['per_part'][part]['pred_label'])
            else:
                features.append(0) # Default to 0 if missing
        
        X.append(features)
        y.append(gt)
        filenames.append(filename)
        
    return np.array(X), np.array(y), parts_order

def objective(weights, X, y, threshold=5):
    # Calculate scores
    scores = np.dot(X, weights)
    # Apply threshold
    preds = (scores >= threshold).astype(int)
    # Calculate F1 (negative because we want to minimize)
    f1 = f1_score(y, preds, zero_division=0)
    return -f1

def evaluate(weights, X, y, threshold=5):
    scores = np.dot(X, weights)
    preds = (scores >= threshold).astype(int)
    f1 = f1_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    prec = precision_score(y, preds, zero_division=0)
    gmean = (rec * prec)**0.5
    return f1, rec, prec, gmean

def main():
    with open(PAIN_SCORE_DIR / 'pain_score_predictions.json', 'r') as f:
        data = json.load(f)
        
    print("Loading Data...")
    X_train, y_train, parts = load_data(data, 'train')
    X_val, y_val, _ = load_data(data, 'val')
    
    print(f"Train: {len(X_train)} samples. Pos={sum(y_train)}, Neg={len(y_train)-sum(y_train)}")
    print(f"Val: {len(X_val)} samples. Pos={sum(y_val)}, Neg={len(y_val)-sum(y_val)}")
    
    # Initial weights (all 1.0)
    initial_weights = np.ones(len(parts))
    
    print("\n--- Baseline (Weights=1.0, Threshold=5) ---")
    f1, rec, prec, gmean = evaluate(initial_weights, X_val, y_val, 5)
    print(f"F1: {f1:.4f}, Recall: {rec:.4f}, Precision: {prec:.4f}")
    
    print("\n--- Optimizing Weights (Grid Search / Heuristic) ---")
    
    best_f1 = -1
    best_weights = initial_weights
    best_metrics = (0,0,0,0)
    
    # Strategy 1: Global Scaling
    print("Testing Global Scaling...")
    for scale in np.arange(1.0, 4.0, 0.1):
        weights = np.ones(len(parts)) * scale
        f1, rec, prec, gmean = evaluate(weights, X_train, y_train, 5)
        if f1 > best_f1:
            best_f1 = f1
            best_weights = weights
            best_metrics = (f1, rec, prec, gmean)
            
    print(f"Best Global Scale F1: {best_f1:.4f} (Weights={best_weights[0]:.2f})")
    
    # Strategy 2: Random Search around best scale
    print("Testing Random Perturbations...")
    base_scale = best_weights[0]
    np.random.seed(42)
    
    for _ in range(1000):
        # Perturb weights: base_scale * random(0.8, 1.2)
        weights = base_scale * np.random.uniform(0.8, 1.2, size=len(parts))
        f1, rec, prec, gmean = evaluate(weights, X_train, y_train, 5)
        if f1 > best_f1:
            best_f1 = f1
            best_weights = weights
            best_metrics = (f1, rec, prec, gmean)

    print(f"Best Random Search F1 (Train): {best_f1:.4f}")
    
    print("\n--- Results on Validation Set (Optimized Weights) ---")
    f1, rec, prec, gmean = evaluate(best_weights, X_val, y_val, 5)
    print(f"F1: {f1:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"G-Mean: {gmean:.4f}")
    
    # Save results
    with open(PAIN_SCORE_DIR / 'optimized_weights_results.txt', 'w') as f:
        f.write(f"Optimized Weights (Threshold 5):\n")
        for p, w in zip(parts, best_weights):
            f.write(f"{p}: {w:.4f}\n")
        f.write(f"\nValidation Metrics:\n")
        f.write(f"F1: {f1:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"G-Mean: {gmean:.4f}\n")

if __name__ == "__main__":
    main()
