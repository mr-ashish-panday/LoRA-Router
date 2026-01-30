"""Full evaluation script comparing all methods."""
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm

from src.config import LABELED_DATA_DIR, RESULTS_DIR, CHECKPOINT_DIR
from src.data.labeler import load_model_and_tokenizer
from src.baselines.routers import get_all_baselines
from src.eval.metrics import (
    compute_router_metrics, compute_task_metrics, compute_speedup,
    compute_calibration, plot_calibration_diagram, plot_threshold_sweep
)


def load_test_data() -> tuple:
    """Load test data."""
    with open(f"{LABELED_DATA_DIR}/test.json", "r") as f:
        data = json.load(f)
    
    return data


def evaluate_baselines(data: List[Dict], model=None, tokenizer=None) -> Dict:
    """Evaluate all baseline routers."""
    questions = [d["question"] for d in data]
    labels = [d["label"] for d in data]
    direct_correct = [d["direct_correct"] for d in data]
    cot_correct = [d["cot_correct"] for d in data]
    
    # Create oracle labels dict
    oracle_labels = {d["question"]: d["label"] for d in data}
    
    # Get baselines
    baselines = get_all_baselines(model, tokenizer, oracle_labels)
    
    results = {}
    
    for name, router in tqdm(baselines.items(), desc="Evaluating baselines"):
        # Get predictions
        probs = []
        for q in questions:
            try:
                prob = router.predict(q)
            except Exception as e:
                prob = 0.5  # Default on error
            probs.append(prob)
        
        decisions = [1 if p > 0.5 else 0 for p in probs]
        
        # Router metrics
        router_metrics = compute_router_metrics(labels, probs)
        
        # Task metrics
        task_metrics = compute_task_metrics(direct_correct, cot_correct, decisions)
        
        # Speedup
        speed_metrics = compute_speedup(decisions)
        
        results[name] = {
            **router_metrics,
            **task_metrics,
            **speed_metrics,
        }
    
    return results


def evaluate_lora_router(data: List[Dict]) -> Dict:
    """Evaluate trained LoRA router."""
    from src.inference import RoutingInference
    
    questions = [d["question"] for d in data]
    labels = [d["label"] for d in data]
    direct_correct = [d["direct_correct"] for d in data]
    cot_correct = [d["cot_correct"] for d in data]
    
    # Load router
    pipeline = RoutingInference(router_path=CHECKPOINT_DIR)
    
    # Get predictions
    probs = []
    for q in tqdm(questions, desc="LoRA Router"):
        probs.append(pipeline.predict_route(q))
    
    decisions = [1 if p > 0.5 else 0 for p in probs]
    
    # All metrics
    router_metrics = compute_router_metrics(labels, probs)
    task_metrics = compute_task_metrics(direct_correct, cot_correct, decisions)
    speed_metrics = compute_speedup(decisions)
    ece, bin_stats = compute_calibration(probs, labels)
    
    return {
        **router_metrics,
        **task_metrics,
        **speed_metrics,
        "ece": ece,
        "calibration_bins": bin_stats,
        "probs": probs,  # For plotting
        "labels": labels,
        "direct_correct": direct_correct,
        "cot_correct": cot_correct,
    }


def run_full_evaluation():
    """Run complete evaluation suite."""
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    
    print("Loading test data...")
    data = load_test_data()
    print(f"Loaded {len(data)} test samples")
    
    # Load model for model-based baselines
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer()
    
    # Evaluate baselines
    print("\n=== Evaluating Baselines ===")
    baseline_results = evaluate_baselines(data, model, tokenizer)
    
    # Evaluate LoRA router
    print("\n=== Evaluating LoRA Router ===")
    lora_results = evaluate_lora_router(data)
    
    # Combine results
    all_results = {
        "baselines": baseline_results,
        "lora_router": {k: v for k, v in lora_results.items() 
                       if k not in ["probs", "labels", "direct_correct", "cot_correct"]},
    }
    
    # Save results
    with open(f"{RESULTS_DIR}/evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Generate plots
    print("\n=== Generating Plots ===")
    
    # Calibration
    plot_calibration_diagram(
        lora_results["probs"],
        lora_results["labels"],
        f"{RESULTS_DIR}/calibration_diagram.png"
    )
    
    # Threshold sweep
    plot_threshold_sweep(
        lora_results["probs"],
        lora_results["labels"],
        lora_results["direct_correct"],
        lora_results["cot_correct"],
        f"{RESULTS_DIR}/threshold_sweep.png"
    )
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    print("\n--- Task Accuracy ---")
    print(f"{'Method':<20} {'Accuracy':>10} {'CoT %':>10} {'Speedup':>10}")
    print("-"*50)
    
    for name, res in baseline_results.items():
        print(f"{name:<20} {res['routed_acc']:>10.1%} {res['cot_usage_rate']:>10.1%} {res['speedup']:>10.2f}x")
    
    lora = all_results["lora_router"]
    print(f"{'lora_router':<20} {lora['routed_acc']:>10.1%} {lora['cot_usage_rate']:>10.1%} {lora['speedup']:>10.2f}x")
    
    print("\n--- Router Metrics ---")
    print(f"LoRA Router F1:      {lora['f1']:.3f}")
    print(f"LoRA Router AUROC:   {lora['auroc']:.3f}")
    print(f"LoRA Router ECE:     {lora['ece']:.3f}")
    
    print(f"\nResults saved to {RESULTS_DIR}/")
    
    return all_results


if __name__ == "__main__":
    run_full_evaluation()
