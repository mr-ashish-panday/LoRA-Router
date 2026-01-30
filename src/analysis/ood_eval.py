"""Out-of-distribution evaluation on AQUA-RAT and SVAMP."""
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from datasets import load_dataset
from tqdm import tqdm

from src.config import RESULTS_DIR, CHECKPOINT_DIR
from src.inference import RoutingInference
from src.data.labeler import extract_number


def load_aqua_rat(n_samples: int = 50) -> List[Dict]:
    """Load AQUA-RAT dataset."""
    print("Loading AQUA-RAT...")
    dataset = load_dataset("aqua_rat", "raw", split="test")
    
    samples = []
    for item in dataset:
        if len(samples) >= n_samples:
            break
        
        # Parse answer
        options = item["options"]
        correct_option = item["correct"]
        
        # Find correct answer value
        for opt in options:
            if opt.startswith(correct_option + ")"):
                answer_text = opt.split(")")[1].strip()
                answer = extract_number(answer_text)
                if answer is not None:
                    samples.append({
                        "question": item["question"],
                        "ground_truth": answer,
                        "source": "aqua_rat",
                    })
                break
    
    print(f"Loaded {len(samples)} AQUA-RAT samples")
    return samples


def load_svamp(n_samples: int = 50) -> List[Dict]:
    """Load SVAMP dataset."""
    print("Loading SVAMP...")
    dataset = load_dataset("ChilleD/SVAMP", split="test")
    
    samples = []
    for item in dataset:
        if len(samples) >= n_samples:
            break
        
        answer = extract_number(str(item["Answer"]))
        if answer is not None:
            samples.append({
                "question": item["Body"] + " " + item["Question"],
                "ground_truth": answer,
                "source": "svamp",
            })
    
    print(f"Loaded {len(samples)} SVAMP samples")
    return samples


def evaluate_ood(
    pipeline: RoutingInference,
    data: List[Dict],
    dataset_name: str,
) -> Dict:
    """Evaluate router on OOD data."""
    results = {
        "direct_correct": 0,
        "cot_correct": 0,
        "routed_correct": 0,
        "cot_usage": 0,
        "total": len(data),
        "samples": [],
    }
    
    for item in tqdm(data, desc=f"Evaluating {dataset_name}"):
        question = item["question"]
        gt = item["ground_truth"]
        
        # Get route probability
        prob = pipeline.predict_route(question)
        use_cot = prob > 0.5
        
        # Get direct answer
        direct_result = pipeline.direct_answer(question)
        direct_correct = (
            direct_result["answer"] is not None and
            abs(direct_result["answer"] - gt) < 0.01
        )
        
        # Get CoT answer
        cot_result = pipeline.cot_answer(question)
        cot_correct = (
            cot_result["answer"] is not None and
            abs(cot_result["answer"] - gt) < 0.01
        )
        
        # Routed answer
        routed_correct = cot_correct if use_cot else direct_correct
        
        results["direct_correct"] += direct_correct
        results["cot_correct"] += cot_correct
        results["routed_correct"] += routed_correct
        results["cot_usage"] += use_cot
        
        results["samples"].append({
            "question": question[:100],
            "ground_truth": gt,
            "route_prob": prob,
            "direct_correct": direct_correct,
            "cot_correct": cot_correct,
            "routed_correct": routed_correct,
        })
    
    # Compute metrics
    n = results["total"]
    results["direct_acc"] = results["direct_correct"] / n
    results["cot_acc"] = results["cot_correct"] / n
    results["routed_acc"] = results["routed_correct"] / n
    results["cot_rate"] = results["cot_usage"] / n
    
    return results


def run_ood_evaluation(n_samples: int = 50):
    """Run OOD evaluation on AQUA-RAT and SVAMP."""
    output_dir = f"{RESULTS_DIR}/ood"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load router
    pipeline = RoutingInference(router_path=CHECKPOINT_DIR)
    
    # Load OOD datasets
    aqua_data = load_aqua_rat(n_samples)
    svamp_data = load_svamp(n_samples)
    
    # Evaluate
    print("\n=== AQUA-RAT Evaluation ===")
    aqua_results = evaluate_ood(pipeline, aqua_data, "AQUA-RAT")
    
    print("\n=== SVAMP Evaluation ===")
    svamp_results = evaluate_ood(pipeline, svamp_data, "SVAMP")
    
    # Combine results
    all_results = {
        "aqua_rat": {k: v for k, v in aqua_results.items() if k != "samples"},
        "svamp": {k: v for k, v in svamp_results.items() if k != "samples"},
    }
    
    # Save
    with open(f"{output_dir}/ood_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    with open(f"{output_dir}/ood_samples.json", "w") as f:
        json.dump({
            "aqua_rat": aqua_results["samples"],
            "svamp": svamp_results["samples"],
        }, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("OOD EVALUATION SUMMARY")
    print("="*60)
    print(f"\n{'Dataset':<15} {'Direct':>10} {'CoT':>10} {'Routed':>10} {'CoT %':>10}")
    print("-"*55)
    
    for name, res in all_results.items():
        print(f"{name:<15} {res['direct_acc']:>10.1%} {res['cot_acc']:>10.1%} "
              f"{res['routed_acc']:>10.1%} {res['cot_rate']:>10.1%}")
    
    print(f"\nResults saved to {output_dir}/")
    
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()
    
    run_ood_evaluation(args.samples)
