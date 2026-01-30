"""Error analysis by problem type."""
import json
import re
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
from tqdm import tqdm

from src.config import LABELED_DATA_DIR, RESULTS_DIR, CHECKPOINT_DIR
from src.inference import RoutingInference


# Keywords for problem categorization
PROBLEM_CATEGORIES = {
    "arithmetic": ["add", "subtract", "multiply", "divide", "sum", "difference", "product"],
    "algebra": ["solve", "equation", "variable", "x", "unknown"],
    "percentage": ["percent", "%", "fraction", "ratio"],
    "geometry": ["area", "perimeter", "length", "width", "radius", "circle", "square", "rectangle"],
    "time": ["hour", "minute", "second", "day", "week", "month", "year", "time"],
    "money": ["dollar", "$", "cent", "price", "cost", "buy", "sell", "profit", "loss"],
    "rate": ["speed", "rate", "per hour", "per minute", "mph", "km/h"],
    "comparison": ["more than", "less than", "difference", "how many more", "how many fewer"],
}


def categorize_problem(question: str) -> List[str]:
    """Categorize problem based on keywords."""
    question_lower = question.lower()
    categories = []
    
    for cat, keywords in PROBLEM_CATEGORIES.items():
        if any(kw in question_lower for kw in keywords):
            categories.append(cat)
    
    if not categories:
        categories.append("other")
    
    return categories


def analyze_errors(
    pipeline: RoutingInference,
    data: List[Dict],
) -> Dict:
    """Analyze errors by problem category."""
    category_stats = defaultdict(lambda: {
        "total": 0,
        "direct_correct": 0,
        "cot_correct": 0,
        "routed_correct": 0,
        "cot_needed": 0,
        "router_correct": 0,
        "false_positives": 0,
        "false_negatives": 0,
    })
    
    error_examples = {
        "false_positives": [],
        "false_negatives": [],
    }
    
    for item in tqdm(data, desc="Analyzing errors"):
        question = item["question"]
        label = item["label"]
        direct_correct = item["direct_correct"]
        cot_correct = item["cot_correct"]
        
        # Get router prediction
        prob = pipeline.predict_route(question)
        pred = 1 if prob > 0.5 else 0
        routed_correct = cot_correct if pred == 1 else direct_correct
        
        # Categorize
        categories = categorize_problem(question)
        
        for cat in categories:
            stats = category_stats[cat]
            stats["total"] += 1
            stats["direct_correct"] += direct_correct
            stats["cot_correct"] += cot_correct
            stats["routed_correct"] += routed_correct
            stats["cot_needed"] += label
            stats["router_correct"] += (pred == label)
            
            if pred == 1 and label == 0:
                stats["false_positives"] += 1
                if len(error_examples["false_positives"]) < 5:
                    error_examples["false_positives"].append({
                        "question": question[:150],
                        "category": cat,
                        "prob": prob,
                    })
            
            if pred == 0 and label == 1:
                stats["false_negatives"] += 1
                if len(error_examples["false_negatives"]) < 5:
                    error_examples["false_negatives"].append({
                        "question": question[:150],
                        "category": cat,
                        "prob": prob,
                    })
    
    # Compute rates
    for cat, stats in category_stats.items():
        n = stats["total"]
        if n > 0:
            stats["direct_acc"] = stats["direct_correct"] / n
            stats["cot_acc"] = stats["cot_correct"] / n
            stats["routed_acc"] = stats["routed_correct"] / n
            stats["cot_need_rate"] = stats["cot_needed"] / n
            stats["router_acc"] = stats["router_correct"] / n
            stats["fp_rate"] = stats["false_positives"] / n
            stats["fn_rate"] = stats["false_negatives"] / n
    
    return dict(category_stats), error_examples


def run_error_analysis():
    """Run full error analysis."""
    output_dir = f"{RESULTS_DIR}/error_analysis"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    with open(f"{LABELED_DATA_DIR}/test.json", "r") as f:
        data = json.load(f)
    
    # Load router
    pipeline = RoutingInference(router_path=CHECKPOINT_DIR)
    
    # Analyze
    category_stats, error_examples = analyze_errors(pipeline, data)
    
    # Save
    with open(f"{output_dir}/category_stats.json", "w") as f:
        json.dump(category_stats, f, indent=2)
    
    with open(f"{output_dir}/error_examples.json", "w") as f:
        json.dump(error_examples, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("ERROR ANALYSIS BY PROBLEM TYPE")
    print("="*80)
    
    print(f"\n{'Category':<15} {'N':>6} {'Direct':>8} {'CoT':>8} {'Routed':>8} {'CoT%':>8} {'FP%':>8} {'FN%':>8}")
    print("-"*75)
    
    for cat in sorted(category_stats.keys()):
        s = category_stats[cat]
        print(f"{cat:<15} {s['total']:>6} {s['direct_acc']:>8.1%} {s['cot_acc']:>8.1%} "
              f"{s['routed_acc']:>8.1%} {s['cot_need_rate']:>8.1%} "
              f"{s['fp_rate']:>8.1%} {s['fn_rate']:>8.1%}")
    
    print(f"\nResults saved to {output_dir}/")
    
    return category_stats, error_examples


if __name__ == "__main__":
    run_error_analysis()
