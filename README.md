# LoRA Router for Efficient Reasoning

This project implements a LoRA-based router to intelligently route math problems between direct answering and chain-of-thought reasoning.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Label dataset (generates train/val/test splits)
python src/data/labeler.py --split train --limit 500

# Train router
python src/train_router.py

# Evaluate
python src/eval/evaluate.py
```

## Project Structure

```
src/
├── config.py           # Configuration and prompts
├── data/
│   ├── labeler.py      # Generate CoT-need labels
│   └── dataset.py      # PyTorch dataset
├── models/
│   └── router.py       # LoRA router architecture
├── baselines/
│   └── routers.py      # 9 baseline routing strategies
├── eval/
│   └── metrics.py      # Evaluation metrics
├── analysis/           # Interpretability, calibration
└── train_router.py     # Training script
```

## Configuration

Edit `src/config.py` to modify:
- Model: `MODEL_NAME` (default: Mistral-7B-Instruct)
- LoRA: Rank, alpha, target modules
- Training: Batch size, learning rate, epochs
- Prompts: Direct and CoT templates

## Baselines

1. Always Direct / Always CoT
2. Random (50/50)
3. Length-based (>50 words → CoT)
4. Keyword-based (math terms → CoT)
5. Entropy (high answer entropy → CoT)
6. Self-Consistency (disagreement → CoT)
7. Embedding (logistic reg on frozen embeddings)
8. Oracle (ground truth upper bound)
