"""Hybrid LoRA Router combining text embeddings with uncertainty features.

This router uses both:
1. Text embeddings from the base model's [CLS] representation
2. Uncertainty features from direct answer generation (entropy, margin, logprob)

The combination provides stronger routing signals than either alone.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
from peft import get_peft_model, LoraConfig, TaskType

from src.config import (
    LORA_RANK, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES, CHECKPOINT_DIR
)


class HybridClassifier(nn.Module):
    """MLP classifier combining embeddings and uncertainty features."""
    
    def __init__(
        self,
        embedding_dim: int,
        uncertainty_dim: int = 4,  # entropy, margin, max_logprob, entropy_max
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.uncertainty_dim = uncertainty_dim
        
        # Project embedding to smaller dim
        self.embed_proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Project uncertainty features
        self.uncertainty_proj = nn.Sequential(
            nn.Linear(uncertainty_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Combined classifier
        combined_dim = hidden_dim + hidden_dim // 4
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(
        self,
        embeddings: torch.Tensor,
        uncertainty_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            embeddings: (batch, embedding_dim)
            uncertainty_features: (batch, uncertainty_dim)
            
        Returns:
            logits: (batch,)
        """
        embed_proj = self.embed_proj(embeddings)
        uncertainty_proj = self.uncertainty_proj(uncertainty_features)
        
        combined = torch.cat([embed_proj, uncertainty_proj], dim=-1)
        logits = self.classifier(combined).squeeze(-1)
        
        return logits


class HybridLoRARouter(nn.Module):
    """
    Hybrid router using LoRA-adapted embeddings + uncertainty features.
    
    This combines the semantic understanding from text with the model's
    own confidence signals for more accurate routing decisions.
    """
    
    def __init__(
        self,
        base_model,
        lora_rank: int = LORA_RANK,
        lora_alpha: int = LORA_ALPHA,
        lora_dropout: float = LORA_DROPOUT,
        target_modules: list = None,
        uncertainty_dim: int = 4,
    ):
        super().__init__()
        
        if target_modules is None:
            target_modules = LORA_TARGET_MODULES
        
        # Setup LoRA
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        
        self.base_model = get_peft_model(base_model, peft_config)
        
        # Freeze base model, only train LoRA + classifier
        for name, param in self.base_model.named_parameters():
            if "lora_" not in name:
                param.requires_grad = False
        
        # Get embedding dimension
        if hasattr(base_model.config, 'hidden_size'):
            self.hidden_size = base_model.config.hidden_size
        else:
            self.hidden_size = 4096  # Default for Mistral
        
        # Hybrid classifier
        self.classifier = HybridClassifier(
            embedding_dim=self.hidden_size,
            uncertainty_dim=uncertainty_dim,
            hidden_dim=256,
            dropout=0.1,
        )
        
        # Normalization stats for uncertainty features
        self.register_buffer('uncertainty_mean', torch.zeros(uncertainty_dim))
        self.register_buffer('uncertainty_std', torch.ones(uncertainty_dim))
    
    def set_uncertainty_normalization(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
    ):
        """Set normalization stats computed from training data."""
        self.uncertainty_mean = mean.to(self.uncertainty_mean.device)
        self.uncertainty_std = std.to(self.uncertainty_std.device)
    
    def normalize_uncertainty(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize uncertainty features using stored stats."""
        return (features - self.uncertainty_mean) / (self.uncertainty_std + 1e-6)
    
    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get text embeddings from LoRA-adapted model."""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        # Use last hidden state, mean pool over sequence
        last_hidden = outputs.hidden_states[-1]
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_hidden = torch.sum(last_hidden * mask_expanded, 1)
        sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
        embeddings = sum_hidden / sum_mask
        
        return embeddings
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        uncertainty_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass returning logits.
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            uncertainty_features: (batch, uncertainty_dim)
            
        Returns:
            logits: (batch,)
        """
        embeddings = self.get_embeddings(input_ids, attention_mask)
        normalized_uncertainty = self.normalize_uncertainty(uncertainty_features)
        logits = self.classifier(embeddings, normalized_uncertainty)
        return logits
    
    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        uncertainty_features: torch.Tensor,
    ) -> torch.Tensor:
        """Get probability for routing decision."""
        logits = self.forward(input_ids, attention_mask, uncertainty_features)
        return torch.sigmoid(logits)
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        uncertainty_features: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Get binary routing decision."""
        proba = self.predict_proba(input_ids, attention_mask, uncertainty_features)
        return (proba > threshold).float()
    
    def save(self, path: str):
        """Save model state."""
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save LoRA weights
        self.base_model.save_pretrained(f"{path}/lora")
        
        # Save classifier and normalization stats
        torch.save({
            'classifier': self.classifier.state_dict(),
            'uncertainty_mean': self.uncertainty_mean,
            'uncertainty_std': self.uncertainty_std,
        }, f"{path}/hybrid_classifier.pt")
    
    def load(self, path: str):
        """Load model state."""
        from peft import PeftModel
        
        # Load LoRA weights
        self.base_model = PeftModel.from_pretrained(
            self.base_model.base_model,
            f"{path}/lora",
        )
        
        # Load classifier and normalization stats
        state = torch.load(f"{path}/hybrid_classifier.pt", map_location='cpu')
        self.classifier.load_state_dict(state['classifier'])
        self.uncertainty_mean = state['uncertainty_mean']
        self.uncertainty_std = state['uncertainty_std']
