"""Generate visualizations and tables from experiment results."""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Results parsed from the experiment output
RESULTS = {
    "data_ablation": {
        "n_100": {"f1": 0.7273, "precision": 0.5714, "recall": 1.0, "auroc": 0.3571},
        "n_250": {"f1": 0.5000, "precision": 0.3529, "recall": 0.8571, "auroc": 0.4107},
        "n_500": {"f1": 0.6353, "precision": 0.5294, "recall": 0.7941, "auroc": 0.6578},
        "n_1000": {"f1": 0.5955, "precision": 0.4818, "recall": 0.7794, "auroc": 0.5662},
    },
    "rank_ablation": {
        "rank_2": {"f1": 0.4722, "precision": 0.4474, "recall": 0.50, "auroc": 0.4964},
        "rank_4": {"f1": 0.6214, "precision": 0.4638, "recall": 0.9412, "auroc": 0.4799},
        "rank_8": {"f1": 0.4691, "precision": 0.4043, "recall": 0.5588, "auroc": 0.4053},
    },
    "target_ablation": {
        "qv": {"f1": 0.6292, "precision": 0.5091, "recall": 0.8235, "auroc": 0.5904},
        "qkvo": {"f1": 0.2791, "precision": 0.6667, "recall": 0.1765, "auroc": 0.5287},
    },
    "threshold_ablation": {
        0.3: {"accuracy": 0.0533, "cot_usage_rate": 0.0467, "speedup": 7.04},
        0.4: {"accuracy": 0.0467, "cot_usage_rate": 0.0267, "speedup": 8.06},
        0.5: {"accuracy": 0.0467, "cot_usage_rate": 0.0133, "speedup": 8.93},
        0.6: {"accuracy": 0.04, "cot_usage_rate": 0.0, "speedup": 10.0},
        0.7: {"accuracy": 0.04, "cot_usage_rate": 0.0, "speedup": 10.0},
    },
    "baselines": {
        "always_direct": {"routed_acc": 0.04, "speedup": 10.0, "f1": 0.0},
        "always_cot": {"routed_acc": 0.48, "speedup": 1.0, "f1": 0.6239},
        "random": {"routed_acc": 0.24, "speedup": 1.80, "f1": 0.4444},
        "length": {"routed_acc": 0.1867, "speedup": 2.58, "f1": 0.4138},
        "keyword": {"routed_acc": 0.3067, "speedup": 1.62, "f1": 0.5455},
        "entropy": {"routed_acc": 0.44, "speedup": 1.08, "f1": 0.6019},
        "self_consistency": {"routed_acc": 0.3867, "speedup": 1.34, "f1": 0.5909},
        "oracle": {"routed_acc": 0.4933, "speedup": 1.97, "f1": 1.0},
    },
    "lora_router": {
        "accuracy": 0.5333,
        "routed_acc": 0.0933,
        "speedup": 4.81,
        "f1": 0.1860,
        "token_savings": 0.792,
    }
}

# Create output directory
Path("figures").mkdir(exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16

# ============================================================
# Figure 1: Data Size Ablation
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
sizes = [100, 250, 500, 1000]
f1_scores = [RESULTS["data_ablation"][f"n_{s}"]["f1"] for s in sizes]
auroc_scores = [RESULTS["data_ablation"][f"n_{s}"]["auroc"] for s in sizes]

x = np.arange(len(sizes))
width = 0.35
bars1 = ax.bar(x - width/2, f1_scores, width, label='F1 Score', color='#2ecc71')
bars2 = ax.bar(x + width/2, auroc_scores, width, label='AUROC', color='#3498db')

ax.set_xlabel('Training Data Size')
ax.set_ylabel('Score')
ax.set_title('Data Size Ablation: Impact on Router Performance')
ax.set_xticks(x)
ax.set_xticklabels(sizes)
ax.legend()
ax.set_ylim(0, 1)

# Add value labels
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('figures/data_ablation.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# Figure 2: LoRA Rank Ablation
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
ranks = [2, 4, 8]
f1_scores = [RESULTS["rank_ablation"][f"rank_{r}"]["f1"] for r in ranks]
recall_scores = [RESULTS["rank_ablation"][f"rank_{r}"]["recall"] for r in ranks]

x = np.arange(len(ranks))
width = 0.35
bars1 = ax.bar(x - width/2, f1_scores, width, label='F1 Score', color='#9b59b6')
bars2 = ax.bar(x + width/2, recall_scores, width, label='Recall', color='#e74c3c')

ax.set_xlabel('LoRA Rank')
ax.set_ylabel('Score')
ax.set_title('LoRA Rank Ablation')
ax.set_xticks(x)
ax.set_xticklabels(ranks)
ax.legend()
ax.set_ylim(0, 1)

for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('figures/rank_ablation.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# Figure 3: Accuracy vs Speedup Comparison
# ============================================================
fig, ax = plt.subplots(figsize=(10, 8))

methods = ['Always Direct', 'Always CoT', 'Random', 'Length', 'Keyword', 
           'Entropy', 'Self-Consistency', 'LoRA Router', 'Oracle']
accuracies = [0.04, 0.48, 0.24, 0.1867, 0.3067, 0.44, 0.3867, 0.0933, 0.4933]
speedups = [10.0, 1.0, 1.80, 2.58, 1.62, 1.08, 1.34, 4.81, 1.97]

colors = ['#95a5a6', '#95a5a6', '#3498db', '#3498db', '#3498db', 
          '#3498db', '#3498db', '#e74c3c', '#2ecc71']

for i, (method, acc, speed) in enumerate(zip(methods, accuracies, speedups)):
    ax.scatter(speed, acc, s=200, c=colors[i], alpha=0.7, edgecolors='black', linewidth=1.5)
    offset = (10, 10) if method not in ['Oracle', 'LoRA Router'] else (10, -15)
    ax.annotate(method, (speed, acc), xytext=offset, 
                textcoords='offset points', fontsize=10, fontweight='bold')

ax.set_xlabel('Speedup (×)', fontsize=14)
ax.set_ylabel('Routing Accuracy', fontsize=14)
ax.set_title('Accuracy vs Speedup Trade-off', fontsize=16)
ax.set_xlim(0, 12)
ax.set_ylim(0, 0.6)
ax.axhline(y=0.48, color='gray', linestyle='--', alpha=0.5, label='Always CoT baseline')

plt.tight_layout()
plt.savefig('figures/accuracy_speedup.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# Figure 4: Target Module Comparison
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
targets = ['Q,V only', 'Q,K,V,O']
f1_scores = [RESULTS["target_ablation"]["qv"]["f1"], RESULTS["target_ablation"]["qkvo"]["f1"]]
recall_scores = [RESULTS["target_ablation"]["qv"]["recall"], RESULTS["target_ablation"]["qkvo"]["recall"]]

x = np.arange(len(targets))
width = 0.35
bars1 = ax.bar(x - width/2, f1_scores, width, label='F1 Score', color='#f39c12')
bars2 = ax.bar(x + width/2, recall_scores, width, label='Recall', color='#1abc9c')

ax.set_xlabel('Target Modules')
ax.set_ylabel('Score')
ax.set_title('LoRA Target Module Ablation')
ax.set_xticks(x)
ax.set_xticklabels(targets)
ax.legend()
ax.set_ylim(0, 1)

for bar in bars1 + bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('figures/target_ablation.png', dpi=300, bbox_inches='tight')
plt.close()

print("All figures generated in figures/ directory!")
print("\nKey Findings:")
print("=" * 60)
print(f"Best Data Size: n=500 (AUROC: 0.6578, F1: 0.6353)")
print(f"Best LoRA Rank: rank=4 (F1: 0.6214, Recall: 0.9412)")
print(f"Best Target Modules: Q,V only (F1: 0.6292)")
print(f"LoRA Router Speedup: 4.81× with 79.2% token savings")
print("=" * 60)
