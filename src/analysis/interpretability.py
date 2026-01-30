"""Interpretability analysis for router decisions."""
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm

from src.config import LABELED_DATA_DIR, RESULTS_DIR, CHECKPOINT_DIR
from src.inference import RoutingInference


def get_attention_weights(pipeline: RoutingInference, question: str) -> np.ndarray:
    """Extract attention weights from router."""
    inputs = pipeline.tokenizer(
        question, return_tensors="pt", truncation=True, max_length=512
    ).to(pipeline.model.device)
    
    with torch.no_grad():
        outputs = pipeline.model(
            **inputs,
            output_attentions=True,
            return_dict=True,
        )
    
    # Average attention across heads and layers
    # Shape: [batch, heads, seq, seq] -> [seq, seq]
    attentions = outputs.attentions
    avg_attention = torch.stack(attentions).mean(dim=(0, 1, 2))
    
    # Get attention from all tokens to each token (importance)
    importance = avg_attention.mean(dim=0).cpu().numpy()
    
    return importance


def visualize_attention(
    pipeline: RoutingInference,
    question: str,
    save_path: str,
    title: str = "Attention Heatmap",
):
    """Create attention heatmap visualization."""
    inputs = pipeline.tokenizer(
        question, return_tensors="pt", truncation=True, max_length=128
    ).to(pipeline.model.device)
    
    tokens = pipeline.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    importance = get_attention_weights(pipeline, question)[:len(tokens)]
    
    # Normalize
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
    
    # Get routing decision
    prob = pipeline.predict_route(question)
    decision = "CoT" if prob > 0.5 else "Direct"
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 3))
    
    # Create color map
    colors = plt.cm.YlOrRd(importance)
    
    # Plot tokens with colored backgrounds
    for i, (token, imp) in enumerate(zip(tokens, importance)):
        ax.text(i, 0, token, ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle='round', facecolor=colors[i], alpha=0.8))
    
    ax.set_xlim(-0.5, len(tokens) - 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.axis('off')
    ax.set_title(f"{title}\nRoute: {decision} (p={prob:.2f})")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def analyze_token_patterns(
    pipeline: RoutingInference,
    data: List[Dict],
    n_samples: int = 100,
) -> Dict:
    """Analyze which token patterns trigger CoT routing."""
    results = {
        "high_attention_tokens": {},
        "cot_trigger_words": {},
        "direct_trigger_words": {},
    }
    
    for item in tqdm(data[:n_samples], desc="Analyzing patterns"):
        question = item["question"]
        label = item["label"]
        
        inputs = pipeline.tokenizer(
            question, return_tensors="pt", truncation=True, max_length=128
        ).to(pipeline.model.device)
        
        tokens = pipeline.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        importance = get_attention_weights(pipeline, question)[:len(tokens)]
        
        # Track high-attention tokens
        threshold = np.percentile(importance, 80)
        for token, imp in zip(tokens, importance):
            if imp > threshold:
                clean_token = token.replace("▁", "").lower()
                if len(clean_token) > 2:
                    results["high_attention_tokens"][clean_token] = \
                        results["high_attention_tokens"].get(clean_token, 0) + 1
                    
                    if label == 1:
                        results["cot_trigger_words"][clean_token] = \
                            results["cot_trigger_words"].get(clean_token, 0) + 1
                    else:
                        results["direct_trigger_words"][clean_token] = \
                            results["direct_trigger_words"].get(clean_token, 0) + 1
    
    # Sort by frequency
    for key in results:
        results[key] = dict(sorted(
            results[key].items(), key=lambda x: -x[1]
        )[:20])
    
    return results


def generate_case_studies(
    pipeline: RoutingInference,
    data: List[Dict],
    output_dir: str,
    n_each: int = 2,
):
    """Generate case study visualizations."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Find examples
    true_positives = []
    true_negatives = []
    false_positives = []
    false_negatives = []
    
    for item in data:
        prob = pipeline.predict_route(item["question"])
        pred = 1 if prob > 0.5 else 0
        label = item["label"]
        
        item["prob"] = prob
        
        if pred == 1 and label == 1:
            true_positives.append(item)
        elif pred == 0 and label == 0:
            true_negatives.append(item)
        elif pred == 1 and label == 0:
            false_positives.append(item)
        elif pred == 0 and label == 1:
            false_negatives.append(item)
    
    # Generate visualizations
    cases = []
    
    for i, item in enumerate(true_positives[:n_each]):
        visualize_attention(
            pipeline, item["question"],
            f"{output_dir}/true_positive_{i}.png",
            f"True Positive (CoT needed & predicted)"
        )
        cases.append({"type": "true_positive", "question": item["question"][:100], "prob": item["prob"]})
    
    for i, item in enumerate(true_negatives[:n_each]):
        visualize_attention(
            pipeline, item["question"],
            f"{output_dir}/true_negative_{i}.png",
            f"True Negative (Direct OK & predicted)"
        )
        cases.append({"type": "true_negative", "question": item["question"][:100], "prob": item["prob"]})
    
    for i, item in enumerate(false_negatives[:n_each]):
        visualize_attention(
            pipeline, item["question"],
            f"{output_dir}/false_negative_{i}.png",
            f"False Negative (CoT needed but missed)"
        )
        cases.append({"type": "false_negative", "question": item["question"][:100], "prob": item["prob"]})
    
    # Save case study summary
    with open(f"{output_dir}/case_studies.json", "w") as f:
        json.dump(cases, f, indent=2)
    
    return cases


def run_interpretability_analysis():
    """Run full interpretability analysis."""
    output_dir = f"{RESULTS_DIR}/interpretability"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    with open(f"{LABELED_DATA_DIR}/test.json", "r") as f:
        data = json.load(f)
    
    # Load router
    pipeline = RoutingInference(router_path=CHECKPOINT_DIR)
    
    # Analyze patterns
    print("Analyzing token patterns...")
    patterns = analyze_token_patterns(pipeline, data)
    
    with open(f"{output_dir}/token_patterns.json", "w") as f:
        json.dump(patterns, f, indent=2)
    
    # Generate case studies
    print("Generating case studies...")
    cases = generate_case_studies(pipeline, data, f"{output_dir}/cases")
    
    print(f"\nInterpretability analysis saved to {output_dir}/")
    
    return patterns, cases


if __name__ == "__main__":
    run_interpretability_analysis()
