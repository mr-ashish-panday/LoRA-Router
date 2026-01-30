"""PyTorch Dataset for labeled router data."""
import json
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.config import MODEL_NAME, MAX_LENGTH, LABELED_DATA_DIR


class RouterDataset(Dataset):
    """Dataset for training the LoRA router."""
    
    def __init__(
        self,
        split: str = "train",
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = MAX_LENGTH,
    ):
        self.split = split
        self.max_length = max_length
        
        # Load tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer
        
        # Load data
        path = Path(LABELED_DATA_DIR) / f"{split}.json"
        with open(path, "r") as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} samples from {split}")
    
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
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(item["label"], dtype=torch.float),
        }
    
    def get_class_weights(self) -> float:
        """Compute positive class weight for imbalanced data."""
        labels = [d["label"] for d in self.data]
        num_pos = sum(labels)
        num_neg = len(labels) - num_pos
        
        if num_pos == 0:
            return 1.0
        
        return num_neg / num_pos


def get_dataloaders(batch_size: int = 4, num_workers: int = 0):
    """Get train/val/test dataloaders."""
    from torch.utils.data import DataLoader
    
    train_ds = RouterDataset("train")
    val_ds = RouterDataset("val")
    test_ds = RouterDataset("test")
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader, train_ds.get_class_weights()
