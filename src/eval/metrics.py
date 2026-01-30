"""Evaluation metrics and utilities."""
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    roc_auc_score, accuracy_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


def compute_router_metrics(labels: List[int], probs: List[float], threshold: float = 0.5) -> Dict:
    """Compute all router metrics."""
    preds = [1 if p > threshold else 0 for p in probs]
    
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "auroc": roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0,
    }
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["true_positives"] = int(tp)
        metrics["false_positives"] = int(fp)
        metrics["true_negatives"] = int(tn)
        metrics["false_negatives"] = int(fn)
    
    return metrics


def compute_task_metrics(
    direct_correct: List[bool],
    cot_correct: List[bool],
    router_decisions: List[int],  # 0 = direct, 1 = CoT
) -> Dict:
    """Compute end-to-end task accuracy with routing."""
    n = len(direct_correct)
    
    # With routing
    routed_correct = 0
    for i in range(n):
        if router_decisions[i] == 0:  # Used direct
            routed_correct += direct_correct[i]
        else:  # Used CoT
            routed_correct += cot_correct[i]
    
    return {
        "always_direct_acc": sum(direct_correct) / n,
        "always_cot_acc": sum(cot_correct) / n,
        "routed_acc": routed_correct / n,
        "cot_usage_rate": sum(router_decisions) / n,
    }


def compute_speedup(
    router_decisions: List[int],
    direct_tokens: float = 15,
    cot_tokens: float = 150,
    tokens_per_sec: float = 12,
) -> Dict:
    """Compute token-based speedup metrics."""
    n = len(router_decisions)
    cot_rate = sum(router_decisions) / n
    
    # Expected tokens
    direct_only_tokens = n * direct_tokens
    cot_only_tokens = n * cot_tokens
    routed_tokens = sum(
        cot_tokens if d == 1 else direct_tokens for d in router_decisions
    )
    
    # Times
    direct_only_time = direct_only_tokens / tokens_per_sec
    cot_only_time = cot_only_tokens / tokens_per_sec
    routed_time = routed_tokens / tokens_per_sec
    
    return {
        "cot_usage_rate": cot_rate,
        "routed_total_tokens": routed_tokens,
        "cot_only_tokens": cot_only_tokens,
        "token_savings": (cot_only_tokens - routed_tokens) / cot_only_tokens,
        "routed_time_sec": routed_time,
        "cot_only_time_sec": cot_only_time,
        "speedup": cot_only_time / routed_time if routed_time > 0 else 0,
    }


def compute_calibration(
    probs: List[float], 
    labels: List[int], 
    n_bins: int = 10
) -> Tuple[float, Dict]:
    """Compute Expected Calibration Error (ECE) and per-bin stats."""
    probs = np.array(probs)
    labels = np.array(labels)
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_stats = []
    
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        
        bin_conf = probs[mask].mean()
        bin_acc = labels[mask].mean()
        bin_size = mask.sum()
        
        ece += (bin_size / len(probs)) * abs(bin_acc - bin_conf)
        bin_stats.append({
            "bin": i,
            "confidence": float(bin_conf),
            "accuracy": float(bin_acc),
            "count": int(bin_size),
        })
    
    return ece, bin_stats


def plot_calibration_diagram(probs: List[float], labels: List[int], save_path: str):
    """Plot reliability diagram."""
    ece, bin_stats = compute_calibration(probs, labels)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
    
    # Bar chart
    confs = [b["confidence"] for b in bin_stats]
    accs = [b["accuracy"] for b in bin_stats]
    ax.bar(confs, accs, width=0.08, alpha=0.7, label='Model')
    
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Calibration Diagram (ECE = {ece:.3f})')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    return ece


def plot_threshold_sweep(
    probs: List[float],
    labels: List[int],
    direct_correct: List[bool],
    cot_correct: List[bool],
    save_path: str,
):
    """Plot accuracy vs speedup for different thresholds."""
    thresholds = np.linspace(0.1, 0.9, 17)
    
    accs, speedups, f1s = [], [], []
    
    for thresh in thresholds:
        decisions = [1 if p > thresh else 0 for p in probs]
        
        # Task accuracy
        correct = sum(
            cot_correct[i] if decisions[i] else direct_correct[i]
            for i in range(len(decisions))
        )
        acc = correct / len(decisions)
        accs.append(acc)
        
        # Speedup
        cot_rate = sum(decisions) / len(decisions)
        speedup = 1 / (cot_rate * 1.0 + (1 - cot_rate) * 0.1)  # Simplified
        speedups.append(speedup)
        
        # Router F1
        preds = [1 if p > thresh else 0 for p in probs]
        f1s.append(f1_score(labels, preds))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy vs Threshold
    ax1.plot(thresholds, accs, 'b-o', label='Task Accuracy')
    ax1.plot(thresholds, f1s, 'r-s', label='Router F1')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Performance vs Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Pareto: Accuracy vs Speedup
    ax2.scatter(speedups, accs, c=thresholds, cmap='viridis', s=100)
    ax2.set_xlabel('Speedup')
    ax2.set_ylabel('Task Accuracy')
    ax2.set_title('Accuracy-Speedup Tradeoff')
    
    cbar = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar.set_label('Threshold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
