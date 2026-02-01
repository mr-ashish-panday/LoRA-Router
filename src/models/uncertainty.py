"""Uncertainty computation module for hybrid routing.

Computes model confidence features from logits:
- Entropy: measure of output distribution spread
- Top-k Margin: difference between top-1 and top-2 logits
- Max Logprob: confidence of highest probability token
"""
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


def compute_token_uncertainty(
    logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute uncertainty features from token logits.
    
    Args:
        logits: Shape (batch, seq_len, vocab_size) or (seq_len, vocab_size)
        attention_mask: Optional mask for valid tokens
        
    Returns:
        Dictionary with entropy, margin, max_logprob, mean_logprob
    """
    # Handle batch dimension
    if logits.dim() == 3:
        logits = logits[0]  # Take first batch item
    
    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Entropy per token
    entropy = -torch.sum(probs * log_probs, dim=-1)  # (seq_len,)
    
    # Top-2 margin per token
    top2_logits, _ = torch.topk(logits, k=2, dim=-1)
    margin = top2_logits[:, 0] - top2_logits[:, 1]  # (seq_len,)
    
    # Max log probability per token
    max_logprob = log_probs.max(dim=-1).values  # (seq_len,)
    
    # Apply mask if provided
    if attention_mask is not None:
        if attention_mask.dim() == 2:
            attention_mask = attention_mask[0]
        mask = attention_mask.bool()
        entropy = entropy[mask]
        margin = margin[mask]
        max_logprob = max_logprob[mask]
    
    # Aggregate to scalars (mean over sequence)
    return {
        "entropy": float(entropy.mean().item()),
        "entropy_std": float(entropy.std().item()) if len(entropy) > 1 else 0.0,
        "entropy_max": float(entropy.max().item()),
        "margin": float(margin.mean().item()),
        "margin_min": float(margin.min().item()),
        "max_logprob": float(max_logprob.mean().item()),
        "max_logprob_min": float(max_logprob.min().item()),
    }


def compute_generation_uncertainty(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 50,
) -> Tuple[str, Dict[str, float]]:
    """
    Generate response and compute uncertainty features.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_tokens: Max tokens to generate
        
    Returns:
        (response_text, uncertainty_features)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )
    
    # Decode response
    full_response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    response = full_response[len(prompt):].strip()
    
    # Compute uncertainty from generation scores
    if outputs.scores:
        # Stack all generation step logits
        all_logits = torch.stack(outputs.scores, dim=1)  # (batch, gen_len, vocab)
        uncertainty = compute_token_uncertainty(all_logits)
    else:
        uncertainty = {
            "entropy": 0.0,
            "entropy_std": 0.0,
            "entropy_max": 0.0,
            "margin": 0.0,
            "margin_min": 0.0,
            "max_logprob": 0.0,
            "max_logprob_min": 0.0,
        }
    
    return response, uncertainty


def uncertainty_features_to_tensor(features: Dict[str, float]) -> torch.Tensor:
    """Convert uncertainty dict to tensor for model input."""
    return torch.tensor([
        features.get("entropy", 0.0),
        features.get("margin", 0.0),
        features.get("max_logprob", 0.0),
        features.get("entropy_max", 0.0),
    ], dtype=torch.float32)


def normalize_uncertainty_features(
    features: torch.Tensor,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize uncertainty features for stable training.
    
    If mean/std not provided, computes from features (for training).
    Returns normalized features, mean, std.
    """
    if mean is None:
        mean = features.mean(dim=0)
    if std is None:
        std = features.std(dim=0)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
    
    normalized = (features - mean) / std
    return normalized, mean, std
