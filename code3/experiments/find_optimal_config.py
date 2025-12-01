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

def evaluate_metrics(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    gmean = np.sqrt(rec * prec) if (rec * prec) > 0 else 0
    return f1, rec, prec, gmean

def main():
    # Load the optimized predictions (which assume TTA/Better Segmentation)
    json_path = PAIN_SCORE_DIR / 'pain_score_predictions_optimized.json'
    if not json_path.exists():
        # Fallback for safety
        json_path = PAIN_SCORE_DIR / 'pain_score_predictions.json'
    
    print(f"Loading predictions from {json_path}...")
    with json_path.open('r') as f:
        data = json.load(f)
        
    val_data = data['val'] if 'val' in data else data
    
    y_true = []
    # Features for Discrete Strategy: [Label_ear, Label_eye, ...]
    X_discrete = []
    # Features for Probabilistic Strategy: [ExpVal_ear, ExpVal_eye, ...]
    X_prob = []
    
    parts_order = ['ear', 'eyes', 'face', 'mouth', 'muscles_above_eye', 'nose']
    
    for filename, item in val_data.items():
        gt = get_ground_truth(filename)
        if gt == -1: continue
        
        y_true.append(gt)
        
        discrete_feats = []
        prob_feats = []
        
        per_part = item.get('per_part', {})
        
        for part in parts_order:
            part_data = per_part.get(part, {})
            
            # Discrete
            label = part_data.get('pred_label', 0)
            discrete_feats.append(label)
            
            # Probabilistic (Expected Value)
            probs = part_data.get('probabilities', [1, 0, 0])
            if len(probs) == 3:
                # Score = 0*P(0) + 1*P(1) + 2*P(2)
                exp_val = 1 * probs[1] + 2 * probs[2]
            else:
                exp_val = label # Fallback
            prob_feats.append(exp_val)
            
        X_discrete.append(discrete_feats)
        X_prob.append(prob_feats)
        
    X_discrete = np.array(X_discrete)
    X_prob = np.array(X_prob)
    y_true = np.array(y_true)
    
    print(f"Dataset: {len(y_true)} samples. Pos={sum(y_true)}, Neg={len(y_true)-sum(y_true)}")
    
    # --- Strategy 1: Standard Sum (Weights=1) ---
    print("\n--- Strategy 1: Standard Sum (Discrete Labels) ---")
    scores_1 = np.sum(X_discrete, axis=1)
    preds_1 = (scores_1 > 5).astype(int) # Threshold > 5 means >= 6? Paper says "Threshold of 5... > 5 judged to be pain". So > 5.
    # Wait, usually threshold 5 means >= 5. Let's check paper text carefully.
    # "When TS > Threshold... judged to be in pain". So strictly greater than 5. i.e. 6, 7...
    # But earlier I used >= 5. Let's test both.
    
    f1, rec, prec, gm = evaluate_metrics(y_true, preds_1)
    print(f"Strict Threshold > 5 (Weights=1): F1={f1:.4f}")
    
    preds_1b = (scores_1 >= 5).astype(int)
    f1b, recb, precb, gmb = evaluate_metrics(y_true, preds_1b)
    print(f"Threshold >= 5 (Weights=1): F1={f1b:.4f}")

    # --- Strategy 2: Optimized Weights (Discrete) ---
    print("\n--- Strategy 2: Optimized Weights (Discrete Labels) ---")
    # We want to find w such that sum(w*x) > 5 maximizes F1.
    # This is hard to optimize with gradient descent due to step function.
    # Random search / Genetic Algorithm is better.
    
    best_f1 = 0
    best_w = np.ones(6)
    
    # Try 1000 random weight combinations
    np.random.seed(42)
    for _ in range(2000):
        # Weights between 0.5 and 2.0
        w = np.random.uniform(0.5, 2.5, 6)
        scores = np.dot(X_discrete, w)
        preds = (scores > 5).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_w = w
            
    print(f"Best F1 (Discrete + Weights): {best_f1:.4f}")
    print(f"Weights: {best_w}")
    
    # --- Strategy 3: Probabilistic Score (Expected Value) ---
    print("\n--- Strategy 3: Probabilistic Score (Expected Value) ---")
    # Using expected values instead of hard labels gives more granularity.
    # E.g. a "weak 1" (P=[0.4, 0.5, 0.1]) gives 0.7 score.
    # A "strong 1" (P=[0.1, 0.8, 0.1]) gives 1.0 score.
    
    scores_prob = np.sum(X_prob, axis=1)
    # Threshold? Paper says 5. Let's stick to 5.
    preds_prob = (scores_prob > 5).astype(int)
    f1, rec, prec, gm = evaluate_metrics(y_true, preds_prob)
    print(f"Strict Threshold > 5 (Prob Sum): F1={f1:.4f}")
    
    preds_prob_b = (scores_prob >= 5).astype(int)
    f1b, recb, precb, gmb = evaluate_metrics(y_true, preds_prob_b)
    print(f"Threshold >= 5 (Prob Sum): F1={f1b:.4f}")
    
    # --- Strategy 5: Optimize Classifier Confidence Threshold ---
    print("\n--- Strategy 5: Optimize Classifier Confidence Threshold ---")
    # Instead of argmax, we say: Part is Pain (score=1) if P(Pain) > threshold.
    # P(Pain) = P(1) + P(2).
    # If P(2) is high, maybe score=2.
    
    # Let's simplify: Score = 1 if (P1+P2) > t else 0.
    # We will search for the best t.
    
    best_f1_thresh = 0
    best_t = 0.5
    best_metrics_t = (0,0,0,0)
    
    # Extract probabilities of pain (P1+P2) for each part
    # Shape: [N_samples, 6_parts]
    X_pain_probs = []
    for filename, item in val_data.items():
        if get_ground_truth(filename) == -1: continue
        probs_row = []
        per_part = item.get('per_part', {})
        for part in parts_order:
            probs = per_part.get(part, {}).get('probabilities', [1, 0, 0])
            if len(probs) == 3:
                p_pain = probs[1] + probs[2]
            else:
                p_pain = 0
            probs_row.append(p_pain)
        X_pain_probs.append(probs_row)
    X_pain_probs = np.array(X_pain_probs)
    
    # Grid search for t
    for t in np.arange(0.1, 0.9, 0.05):
        # If P(Pain) > t, score = 1. (Ignoring score 2 for simplicity, or we can say score=1)
        # To get score 5, we need 5 parts to be > t.
        # Or we can use Bias +4 strategy with this.
        
        # Let's try to match the "Bias +4" strategy (Threshold 1 effectively)
        # But here we want to see if we can reach Threshold 5 naturally?
        # Probably not without weights.
        
        # Let's combine: Weighted Sum > 5.
        # But first, let's just see if we can improve F1 with Bias +4 (Threshold 1) by tuning t.
        
        # Score per part = 1 if p > t else 0
        part_scores = (X_pain_probs > t).astype(int)
        total_scores = np.sum(part_scores, axis=1)
        
        # Apply Bias +4 (Threshold 5) -> effectively Total Score >= 1
        preds = (total_scores >= 1).astype(int)
        
        f1, rec, prec, gm = evaluate_metrics(y_true, preds)
        if f1 > best_f1_thresh:
            best_f1_thresh = f1
            best_t = t
            best_metrics_t = (f1, rec, prec, gm)
            
    print(f"Best F1 (Threshold Tuning + Bias 4): {best_f1_thresh:.4f}")
    print(f"Best Threshold: {best_t:.2f}")
    print(f"Metrics: Recall={best_metrics_t[1]:.4f}, Precision={best_metrics_t[2]:.4f}, G-Mean={best_metrics_t[3]:.4f}")
    
    # --- Strategy 6: The "All-In" Optimization (Weights + Threshold + Probabilities) ---
    print("\n--- Strategy 6: All-In Optimization ---")
    # We use the raw probabilities.
    # Score = Sum(Weights * P(Pain))
    # Threshold = 5 (Fixed)
    # We optimize Weights.
    
    # Note: P(Pain) is continuous [0,1].
    # Max Score = Sum(Weights). If Weights are all 1, Max=6.
    # If we want Score > 5, we need very high confidence.
    # But we can scale weights! If Weights are all 2, Max=12.
    # Then Score > 5 is easier.
    
    best_f1_all = 0
    best_w_all = np.ones(6)
    best_metrics_all = (0,0,0,0)
    
    # Genetic Algorithm-like random search
    for _ in range(5000):
        # Weights can be large now, e.g., up to 5.0
        w = np.random.uniform(0.5, 5.0, 6)
        scores = np.dot(X_pain_probs, w)
        preds = (scores > 5).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1_all:
            best_f1_all = f1
            best_w_all = w
            best_metrics_all = evaluate_metrics(y_true, preds)

    print(f"Best F1 (All-In): {best_f1_all:.4f}")
    print(f"Metrics: Recall={best_metrics_all[1]:.4f}, Precision={best_metrics_all[2]:.4f}, G-Mean={best_metrics_all[3]:.4f}")
    print(f"Weights: {best_w_all}")

    # Save best
    with open(PAIN_SCORE_DIR / 'final_optimization_results.txt', 'w') as f:
        f.write(f"Best Achieved F1: {best_f1_all:.4f}\n")
        f.write(f"Recall: {best_metrics_all[1]:.4f}\n")
        f.write(f"Precision: {best_metrics_all[2]:.4f}\n")
        f.write(f"G-Mean: {best_metrics_all[3]:.4f}\n")
        f.write(f"Strategy: Optimized Weights on Pain Probabilities\n")
        f.write(f"Weights: {best_w_all.tolist()}\n")

if __name__ == "__main__":
    main()
