"""LoRA Router model architecture."""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

from src.config import (
    MODEL_NAME, QUANTIZATION_BITS,
    LORA_RANK, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES
)


class AttentionWeightedPooling(nn.Module):
    """Attention-weighted mean pooling + last token."""
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        # Attention-weighted mean
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        mean_pooled = sum_hidden / sum_mask
        
        # Last token (before padding)
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_token = hidden_states[batch_indices, seq_lengths.long()]
        
        # Concatenate
        return torch.cat([mean_pooled, last_token], dim=-1)


class LoRARouter(nn.Module):
    """Binary classifier using LoRA adapter for routing decisions."""
    
    def __init__(self, base_model=None, hidden_dim: int = 4096):
        super().__init__()
        
        if base_model is None:
            base_model = self._load_base_model()
        
        self.base = base_model
        self.hidden_dim = hidden_dim
        
        # Freeze base model
        for param in self.base.parameters():
            param.requires_grad = False
        
        # Pooling
        self.pooling = AttentionWeightedPooling()
        
        # Classification head (input: 2 * hidden_dim from concat)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(2 * hidden_dim, 1)
        )
    
    def _load_base_model(self):
        """Load quantized base model with LoRA."""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Add LoRA
        lora_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        # Get hidden states from base model
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Use last layer hidden states
        hidden_states = outputs.hidden_states[-1]
        
        # Pool and classify
        pooled = self.pooling(hidden_states, attention_mask)
        logits = self.classifier(pooled)
        
        return torch.sigmoid(logits.squeeze(-1))
    
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get probability that CoT is needed."""
        self.eval()
        with torch.no_grad():
            return self.forward(input_ids, attention_mask)
    
    def save(self, path: str):
        """Save LoRA adapter and classifier."""
        self.base.save_pretrained(f"{path}/lora_adapter")
        torch.save(self.classifier.state_dict(), f"{path}/classifier.pt")
    
    def load(self, path: str):
        """Load LoRA adapter and classifier."""
        from peft import PeftModel
        self.base = PeftModel.from_pretrained(self.base, f"{path}/lora_adapter")
        self.classifier.load_state_dict(torch.load(f"{path}/classifier.pt"))
