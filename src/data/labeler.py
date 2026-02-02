"""Data labeling module for generating routing labels.

This module contains functions to:
1. Load the base model
2. Generate direct and CoT responses
3. Extract numerical answers
4. Create binary labels for router training
"""
import json
import re
from pathlib import Path
from typing import Optional, List, Dict
import torch
from datasets import load_dataset
from tqdm import tqdm

from src.config import (
    MODEL_NAME, QUANTIZATION_BITS, DIRECT_PROMPT, COT_PROMPT,
    DIRECT_MAX_TOKENS, COT_MAX_TOKENS, LABELED_DATA_DIR
)


def extract_answer(text: str) -> Optional[float]:
    """
    Extract answer from model output with multi-tier fallback.
    
    Priority:
    1. #### pattern (GSM8K standard format)
    2. Number at the very start (when model just outputs the answer)
    3. Last number on last non-empty line
    4. Any last number (least reliable fallback)
    """
    if not text:
        return None
    
    text = text.strip()
    
    # Tier 1: Look for #### pattern (GSM8K standard format)
    hash_matches = re.findall(r'####\s*(-?[\d,]+\.?\d*)', text)
    if hash_matches:
        try:
            return float(hash_matches[-1].replace(',', ''))
        except ValueError:
            pass
    
    # Tier 2: Number at start of response (model just outputs answer)
    start_match = re.match(r'^\s*(-?[\d,]+\.?\d*)\s*$', text.split('\n')[0])
    if start_match:
        try:
            return float(start_match.group(1).replace(',', ''))
        except ValueError:
            pass
    
    # Tier 3: Last number on last non-empty line
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        last_line = lines[-1]
        numbers = re.findall(r'-?[\d,]+\.?\d*', last_line)
        numbers = [n for n in numbers if n and n not in ['-', '.', '-.']]
        if numbers:
            try:
                return float(numbers[-1].replace(',', ''))
            except ValueError:
                pass
    
    # Tier 4: Any last number in entire text (fallback)
    all_numbers = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    all_numbers = [n for n in all_numbers if n and n not in ['-', '.', '-.']]
    if all_numbers:
        try:
            return float(all_numbers[-1])
        except ValueError:
            pass
    
    return None


def parse_answer(answer_text: str) -> Optional[float]:
    """Parse GSM8K gold answer format: '#### 42'. Wrapper for extract_answer."""
    return extract_answer(answer_text)


def load_model_and_tokenizer():
    """Load base model with quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    print(f"Loading {MODEL_NAME} with {QUANTIZATION_BITS}-bit quantization...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=(QUANTIZATION_BITS == 4),
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_tokens: int) -> str:
    """Generate model response."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(prompt):].strip()
    return response


def generate_response_with_uncertainty(
    model, tokenizer, prompt: str, max_tokens: int
) -> tuple:
    """
    Generate response and compute uncertainty features from logits.
    
    Returns:
        (response_text, uncertainty_dict)
    """
    import torch.nn.functional as F
    
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
    
    full_response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    response = full_response[len(prompt):].strip()
    
    uncertainty = {
        "entropy": 0.0,
        "margin": 0.0,
        "max_logprob": 0.0,
        "entropy_max": 0.0,
    }
    
    if outputs.scores:
        all_logits = torch.stack(outputs.scores, dim=0)
        probs = F.softmax(all_logits, dim=-1)
        log_probs = F.log_softmax(all_logits, dim=-1)
        
        entropy = -torch.sum(probs * log_probs, dim=-1).squeeze()
        top2, _ = torch.topk(all_logits, k=2, dim=-1)
        margin = (top2[:, :, 0] - top2[:, :, 1]).squeeze()
        max_lp = log_probs.max(dim=-1).values.squeeze()
        
        uncertainty = {
            "entropy": float(entropy.mean().item()) if entropy.numel() > 0 else 0.0,
            "margin": float(margin.mean().item()) if margin.numel() > 0 else 0.0,
            "max_logprob": float(max_lp.mean().item()) if max_lp.numel() > 0 else 0.0,
            "entropy_max": float(entropy.max().item()) if entropy.numel() > 0 else 0.0,
        }
    
    return response, uncertainty


def generate_label(
    model, tokenizer, question: str, ground_truth: float,
    capture_uncertainty: bool = False
) -> dict:
    """Generate label for a single problem."""
    direct_prompt = DIRECT_PROMPT.format(question=question)
    
    if capture_uncertainty:
        direct_response, uncertainty = generate_response_with_uncertainty(
            model, tokenizer, direct_prompt, DIRECT_MAX_TOKENS
        )
    else:
        direct_response = generate_response(model, tokenizer, direct_prompt, DIRECT_MAX_TOKENS)
        uncertainty = None
    
    direct_answer = extract_answer(direct_response)
    direct_correct = (direct_answer is not None and abs(direct_answer - ground_truth) < 0.01)
    
    cot_prompt = COT_PROMPT.format(question=question)
    cot_response = generate_response(model, tokenizer, cot_prompt, COT_MAX_TOKENS)
    cot_answer = extract_answer(cot_response)
    cot_correct = (cot_answer is not None and abs(cot_answer - ground_truth) < 0.01)
    
    label = 1 if (cot_correct and not direct_correct) else 0
    
    result = {
        "question": question,
        "ground_truth": ground_truth,
        "direct_response": direct_response,
        "direct_answer": direct_answer,
        "direct_correct": direct_correct,
        "cot_response": cot_response,
        "cot_answer": cot_answer,
        "cot_correct": cot_correct,
        "label": label,
    }
    
    if uncertainty:
        result["uncertainty"] = uncertainty
    
    return result


def split_by_index(data: list, seed: int = 42) -> tuple:
    """Split data deterministically by index."""
    import numpy as np
    np.random.seed(seed)
    
    n = len(data)
    indices = np.random.permutation(n)
    
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    return (
        [data[i] for i in train_idx],
        [data[i] for i in val_idx],
        [data[i] for i in test_idx],
    )


def label_dataset(n_samples: int = 500, capture_uncertainty: bool = False):
    """Label GSM8K dataset and save splits."""
    print("Loading GSM8K train split...")
    dataset = load_dataset("gsm8k", "main", split="train")
    
    model, tokenizer = load_model_and_tokenizer()
    
    print(f"Processing {n_samples} problems...")
    labeled_data = []
    
    for i, item in enumerate(tqdm(dataset, total=n_samples, desc="Labeling")):
        if i >= n_samples:
            break
        
        question = item["question"]
        answer_text = item["answer"]
        ground_truth = parse_answer(answer_text)
        
        if ground_truth is None:
            continue
        
        result = generate_label(
            model, tokenizer, question, ground_truth,
            capture_uncertainty=capture_uncertainty
        )
        labeled_data.append(result)
    
    # Split data
    train_data, val_data, test_data = split_by_index(labeled_data)
    
    # Save
    Path(LABELED_DATA_DIR).mkdir(parents=True, exist_ok=True)
    
    for name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        path = f"{LABELED_DATA_DIR}/{name}.json"
        with open(path, "w") as f:
            json.dump(split_data, f, indent=2)
        print(f"Saved {len(split_data)} samples to {path}")
    
    # Print statistics
    total = len(labeled_data)
    cot_needed = sum(1 for d in labeled_data if d["label"] == 1)
    direct_correct = sum(1 for d in labeled_data if d["direct_correct"])
    cot_correct = sum(1 for d in labeled_data if d["cot_correct"])
    
    print(f"\nStatistics:")
    print(f"  Total: {total}")
    print(f"  Direct correct: {direct_correct} ({100*direct_correct/total:.1f}%)")
    print(f"  CoT correct: {cot_correct} ({100*cot_correct/total:.1f}%)")
    print(f"  CoT needed (label=1): {cot_needed} ({100*cot_needed/total:.1f}%)")
    print(f"  Direct sufficient (label=0): {total - cot_needed} ({100*(total-cot_needed)/total:.1f}%)")
    
    return labeled_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--uncertainty", action="store_true", help="Capture uncertainty features")
    args = parser.parse_args()
    
    label_dataset(args.samples, capture_uncertainty=args.uncertainty)
