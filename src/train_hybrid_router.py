"""Training script for the Hybrid LoRA Router.

Trains a router that combines text embeddings with uncertainty features
for routing decisions between direct and chain-of-thought prompting.
"""
import json
import gc
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

from src.config import (
    MODEL_NAME, BATCH_SIZE, GRADIENT_ACCUMULATION, LEARNING_RATE,
    NUM_EPOCHS, MAX_LENGTH, LABELED_DATA_DIR, RESULTS_DIR, CHECKPOINT_DIR
)
from src.data.labeler import load_model_and_tokenizer
from src.models.hybrid_router import HybridLoRARouter
from src.models.uncertainty import uncertainty_features_to_tensor


class HybridRouterDataset(Dataset):
    """Dataset with text and uncertainty features."""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = MAX_LENGTH):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize question
        encoding = self.tokenizer(
            item["question"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Get uncertainty features (with defaults if missing)
        uncertainty = item.get("uncertainty", {
            "entropy": 0.0,
            "margin": 0.0,
            "max_logprob": 0.0,
            "entropy_max": 0.0,
        })
        uncertainty_tensor = uncertainty_features_to_tensor(uncertainty)
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "uncertainty": uncertainty_tensor,
            "label": torch.tensor(item["label"], dtype=torch.float32),
        }


def load_data(split: str) -> List[Dict]:
    """Load labeled data from JSON."""
    path = f"{LABELED_DATA_DIR}/{split}.json"
    with open(path, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {split}")
    return data


def compute_class_weight(data: List[Dict]) -> float:
    """Compute positive class weight for imbalanced data."""
    n_pos = sum(1 for d in data if d["label"] == 1)
    n_neg = len(data) - n_pos
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


def compute_uncertainty_stats(data: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute mean and std of uncertainty features for normalization."""
    features = []
    for item in data:
        uncertainty = item.get("uncertainty", {
            "entropy": 0.0,
            "margin": 0.0,
            "max_logprob": 0.0,
            "entropy_max": 0.0,
        })
        features.append(uncertainty_features_to_tensor(uncertainty))
    
    features = torch.stack(features)
    mean = features.mean(dim=0)
    std = features.std(dim=0)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    
    return mean, std


def train_epoch(
    model, train_loader, optimizer, scheduler, criterion, device, grad_accum
) -> Dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    optimizer.zero_grad()
    pbar = tqdm(train_loader, desc="Training")
    
    for i, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        uncertainty = batch["uncertainty"].to(device)
        labels = batch["label"].to(device)
        
        logits = model(input_ids, attention_mask, uncertainty)
        loss = criterion(logits, labels) / grad_accum
        
        loss.backward()
        total_loss += loss.item() * grad_accum
        
        if (i + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        preds = torch.sigmoid(logits).detach().cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().tolist())
        
        pbar.set_postfix(loss=f"{loss.item() * grad_accum:.4f}")
    
    # Compute metrics
    from sklearn.metrics import f1_score
    binary_preds = [1 if p > 0.5 else 0 for p in all_preds]
    f1 = f1_score(all_labels, binary_preds, zero_division=0)
    
    avg_loss = total_loss / len(train_loader)
    return {"loss": avg_loss, "f1": f1}


def evaluate(model, val_loader, criterion, device) -> Dict:
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            uncertainty = batch["uncertainty"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(input_ids, attention_mask, uncertainty)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.sigmoid(logits).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().tolist())
    
    # Metrics
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
    
    binary_preds = [1 if p > 0.5 else 0 for p in all_preds]
    
    metrics = {
        "loss": total_loss / len(val_loader),
        "f1": f1_score(all_labels, binary_preds, zero_division=0),
        "precision": precision_score(all_labels, binary_preds, zero_division=0),
        "recall": recall_score(all_labels, binary_preds, zero_division=0),
    }
    
    if len(set(all_labels)) > 1:
        metrics["auroc"] = roc_auc_score(all_labels, all_preds)
    else:
        metrics["auroc"] = 0.5
    
    return metrics


def train_hybrid():
    """Main training function for hybrid router."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    train_data = load_data("train")
    val_data = load_data("val")
    test_data = load_data("test")
    
    # Class weight
    pos_weight = compute_class_weight(train_data)
    print(f"Class weight for positive class: {pos_weight:.2f}")
    
    # Uncertainty normalization stats
    uncertainty_mean, uncertainty_std = compute_uncertainty_stats(train_data)
    print(f"Uncertainty stats - Mean: {uncertainty_mean}, Std: {uncertainty_std}")
    
    # Load model
    print("Initializing model...")
    base_model, tokenizer = load_model_and_tokenizer()
    
    model = HybridLoRARouter(base_model)
    model.set_uncertainty_normalization(uncertainty_mean, uncertainty_std)
    model.to(device)
    
    # Print trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {100*trainable/total:.4f}")
    
    # Datasets
    train_dataset = HybridRouterDataset(train_data, tokenizer)
    val_dataset = HybridRouterDataset(val_data, tokenizer)
    test_dataset = HybridRouterDataset(test_data, tokenizer)
    
    # Use smaller batch size for hybrid (full forward pass needs more memory)
    hybrid_batch_size = 1
    hybrid_grad_accum = GRADIENT_ACCUMULATION * BATCH_SIZE  # Keep effective batch size same
    
    train_loader = DataLoader(train_dataset, batch_size=hybrid_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hybrid_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=hybrid_batch_size)
    
    # Loss with class weighting
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    
    # Optimizer
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )
    
    total_steps = len(train_loader) * NUM_EPOCHS // GRADIENT_ACCUMULATION
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # Training loop
    best_f1 = 0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print("="*50)
        
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, hybrid_grad_accum
        )
        print(f"Train Loss: {train_metrics['loss']:.4f}, Train F1: {train_metrics['f1']:.4f}")
        
        val_metrics = evaluate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}")
        print(f"Val AUROC: {val_metrics['auroc']:.4f}")
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            model.save(f"{CHECKPOINT_DIR}/hybrid")
            print(f"Saved best model (F1: {best_f1:.4f})")
    
    # Final test evaluation
    print(f"\n{'='*50}")
    print("Final Test Evaluation")
    print("="*50)
    
    test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test AUROC: {test_metrics['auroc']:.4f}")
    
    # Save results
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    with open(f"{RESULTS_DIR}/hybrid_training_results.json", "w") as f:
        json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=2)
    
    print(f"\nResults saved to {RESULTS_DIR}/hybrid_training_results.json")
    
    # Cleanup
    del model
    del optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    train_hybrid()
