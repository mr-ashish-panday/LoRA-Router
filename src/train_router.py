"""Training script for LoRA router."""
import os
import gc
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.config import (
    BATCH_SIZE, GRADIENT_ACCUMULATION, LEARNING_RATE, NUM_EPOCHS,
    CHECKPOINT_DIR, RESULTS_DIR
)
from src.data.dataset import get_dataloaders
from src.models.router import LoRARouter


def train_epoch(model, loader, optimizer, criterion, device, accumulation_steps):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    optimizer.zero_grad()
    pbar = tqdm(loader, desc="Training")
    
    for i, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        # Forward (model returns logits, not probs)
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss = loss / accumulation_steps
        
        # Backward
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        probs = torch.sigmoid(logits.detach())  # Apply sigmoid for predictions
        all_preds.extend((probs > 0.5).cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        
        pbar.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})
    
    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, all_preds)
    
    return avg_loss, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0
    all_probs, all_preds, all_labels = [], [], []
    
    for batch in tqdm(loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        # Model returns logits, apply sigmoid for probs
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)
        
        total_loss += loss.item()
        all_probs.extend(probs.cpu().numpy().tolist())
        all_preds.extend((probs > 0.5).cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    
    avg_loss = total_loss / len(loader)
    
    # Diagnostic: Check for probability inversion
    import numpy as np
    probs_np = np.array(all_probs)
    labels_np = np.array(all_labels)
    
    mean_prob_pos = probs_np[labels_np == 1].mean() if (labels_np == 1).sum() > 0 else 0
    mean_prob_neg = probs_np[labels_np == 0].mean() if (labels_np == 0).sum() > 0 else 0
    
    print(f"\n[DIAGNOSTIC] Mean prob for label=1 (CoT needed): {mean_prob_pos:.4f}")
    print(f"[DIAGNOSTIC] Mean prob for label=0 (direct ok): {mean_prob_neg:.4f}")
    
    # Check if predictions are inverted (label=1 should have HIGHER probs)
    is_inverted = mean_prob_pos < mean_prob_neg
    if is_inverted:
        print("[WARNING] Predictions appear INVERTED! Label=1 has lower mean prob than label=0")
        print("[FIX] Using (1 - prob) for AUROC computation")
        auroc_probs = [1 - p for p in all_probs]
    else:
        auroc_probs = all_probs
    
    metrics = {
        "loss": avg_loss,
        "f1": f1_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "auroc": roc_auc_score(all_labels, auroc_probs) if len(set(all_labels)) > 1 else 0,
        "mean_prob_positive": float(mean_prob_pos),
        "mean_prob_negative": float(mean_prob_neg),
        "predictions_inverted": is_inverted,
    }
    
    return metrics, all_probs, all_labels


def find_optimal_threshold(probs, labels, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7]):
    """Find threshold that maximizes F1."""
    best_f1, best_thresh = 0, 0.5
    
    for thresh in thresholds:
        preds = [1 if p > thresh else 0 for p in probs]
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    return best_thresh, best_f1


def train():
    """Main training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get data
    print("Loading data...")
    train_loader, val_loader, test_loader, pos_weight = get_dataloaders(BATCH_SIZE)
    print(f"Class weight for positive class: {pos_weight:.2f}")
    
    # Initialize model
    print("Initializing model...")
    model = LoRARouter()
    model.classifier.to(device)
    
    # Loss with class weighting to handle imbalance
    pos_weight_tensor = torch.tensor([pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    # Note: For weighted, use BCEWithLogitsLoss with pos_weight
    
    # Optimizer (only LoRA + classifier params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Training loop
    best_f1 = 0
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"{'='*50}")
        
        # Train
        train_loss, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, device, GRADIENT_ACCUMULATION
        )
        print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
        
        # Validate
        val_metrics, val_probs, val_labels = evaluate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}")
        print(f"Val AUROC: {val_metrics['auroc']:.4f}")
        
        # Find optimal threshold
        opt_thresh, opt_f1 = find_optimal_threshold(val_probs, val_labels)
        print(f"Optimal threshold: {opt_thresh} (F1: {opt_f1:.4f})")
        
        # Save best
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            model.save(CHECKPOINT_DIR)
            print(f"Saved best model (F1: {best_f1:.4f})")
        
        scheduler.step()
    
    # Final evaluation on test set
    print(f"\n{'='*50}")
    print("Final Test Evaluation")
    print(f"{'='*50}")
    
    model.load(CHECKPOINT_DIR)
    test_metrics, test_probs, test_labels = evaluate(model, test_loader, criterion, device)
    
    opt_thresh, opt_f1 = find_optimal_threshold(test_probs, test_labels)
    test_metrics["optimal_threshold"] = opt_thresh
    test_metrics["optimal_f1"] = opt_f1
    
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test AUROC: {test_metrics['auroc']:.4f}")
    print(f"Optimal threshold: {opt_thresh}")
    
    # Save results
    with open(f"{RESULTS_DIR}/training_results.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    
    print(f"\nResults saved to {RESULTS_DIR}/training_results.json")
    
    # Cleanup GPU memory
    del model
    del optimizer
    del scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return test_metrics


if __name__ == "__main__":
    train()
