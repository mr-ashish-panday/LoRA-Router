"""Ablation study runner."""
import gc
import json
from pathlib import Path
from typing import List, Dict
import torch
from tqdm import tqdm

from src.config import (
    LORA_RANK, LORA_ALPHA, LORA_TARGET_MODULES,
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS,
    RESULTS_DIR, CHECKPOINT_DIR
)


def cleanup_gpu():
    """Force GPU memory cleanup between runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_rank_ablation(ranks: List[int] = [2, 4, 8]) -> Dict:
    """Ablate LoRA rank."""
    from src.train_router import train
    from src import config
    
    results = {}
    original_rank = config.LORA_RANK
    
    for rank in ranks:
        print(f"\n{'='*50}")
        print(f"Training with rank={rank}")
        print(f"{'='*50}")
        
        config.LORA_RANK = rank
        
        # Train and get results
        metrics = train()
        results[f"rank_{rank}"] = metrics
        
        # Cleanup GPU memory before next iteration
        cleanup_gpu()
        
        # Save checkpoint with rank suffix
        Path(f"{CHECKPOINT_DIR}_rank{rank}").mkdir(exist_ok=True)
    
    config.LORA_RANK = original_rank
    
    # Save ablation results
    with open(f"{RESULTS_DIR}/ablation_rank.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def run_target_ablation(
    configs: Dict[str, List[str]] = {
        "qv": ["q_proj", "v_proj"],
        "qkvo": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }
) -> Dict:
    """Ablate LoRA target modules."""
    from src.train_router import train
    from src import config
    
    results = {}
    original_targets = config.LORA_TARGET_MODULES
    
    for name, targets in configs.items():
        print(f"\n{'='*50}")
        print(f"Training with targets={name}: {targets}")
        print(f"{'='*50}")
        
        config.LORA_TARGET_MODULES = targets
        
        metrics = train()
        results[name] = metrics
        
        # Cleanup GPU memory before next iteration
        cleanup_gpu()
    
    config.LORA_TARGET_MODULES = original_targets
    
    with open(f"{RESULTS_DIR}/ablation_targets.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def run_data_ablation(sizes: List[int] = [100, 250, 500, 1000]) -> Dict:
    """Ablate training data size."""
    from src.data.labeler import label_dataset
    from src.train_router import train
    
    results = {}
    
    for size in sizes:
        print(f"\n{'='*50}")
        print(f"Training with {size} samples")
        print(f"{'='*50}")
        
        # Generate labels with limit
        label_dataset(split="train", limit=size)
        
        # Train
        metrics = train()
        results[f"n_{size}"] = metrics
        
        # Cleanup GPU memory before next iteration
        cleanup_gpu()
    
    with open(f"{RESULTS_DIR}/ablation_data.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def run_threshold_ablation(
    thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7],
    test_data_path: str = "data/labeled/test.json",
) -> Dict:
    """Ablate decision threshold."""
    import json
    from src.inference import RoutingInference
    
    with open(test_data_path, "r") as f:
        data = json.load(f)
    
    questions = [d["question"] for d in data]
    ground_truths = [d["ground_truth"] for d in data]
    direct_correct = [d["direct_correct"] for d in data]
    cot_correct = [d["cot_correct"] for d in data]
    
    # Load router
    pipeline = RoutingInference(router_path=CHECKPOINT_DIR)
    
    # Get all route probabilities
    probs = []
    for q in tqdm(questions, desc="Getting route probs"):
        probs.append(pipeline.predict_route(q))
    
    results = {}
    
    for thresh in thresholds:
        decisions = [1 if p > thresh else 0 for p in probs]
        
        # Task accuracy with routing
        correct = 0
        for i in range(len(decisions)):
            if decisions[i] == 1:  # Used CoT
                correct += cot_correct[i]
            else:  # Used direct
                correct += direct_correct[i]
        
        acc = correct / len(decisions)
        cot_rate = sum(decisions) / len(decisions)
        
        # Speedup (assuming 10:1 token ratio)
        avg_cost = cot_rate * 1.0 + (1 - cot_rate) * 0.1
        speedup = 1.0 / avg_cost
        
        results[f"thresh_{thresh}"] = {
            "accuracy": acc,
            "cot_usage_rate": cot_rate,
            "speedup": speedup,
        }
    
    with open(f"{RESULTS_DIR}/ablation_threshold.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def run_all_ablations():
    """Run all ablation studies."""
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("RUNNING ALL ABLATION STUDIES")
    print("="*60)
    
    # 1. Rank ablation
    print("\n[1/4] Rank Ablation")
    rank_results = run_rank_ablation()
    
    # 2. Target ablation
    print("\n[2/4] Target Ablation")
    target_results = run_target_ablation()
    
    # 3. Data ablation
    print("\n[3/4] Data Size Ablation")
    data_results = run_data_ablation()
    
    # 4. Threshold ablation
    print("\n[4/4] Threshold Ablation")
    thresh_results = run_threshold_ablation()
    
    # Combine all
    all_results = {
        "rank": rank_results,
        "targets": target_results,
        "data_size": data_results,
        "threshold": thresh_results,
    }
    
    with open(f"{RESULTS_DIR}/all_ablations.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*60)
    print(f"All ablations complete. Results saved to {RESULTS_DIR}/")
    print("="*60)
    
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", choices=["rank", "target", "data", "threshold", "all"], default="all")
    args = parser.parse_args()
    
    if args.study == "rank":
        run_rank_ablation()
    elif args.study == "target":
        run_target_ablation()
    elif args.study == "data":
        run_data_ablation()
    elif args.study == "threshold":
        run_threshold_ablation()
    else:
        run_all_ablations()
