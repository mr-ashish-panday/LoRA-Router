"""Data loading and labeling utilities."""
import json
import re
from pathlib import Path
from typing import Optional
from datasets import load_dataset
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.config import (
    MODEL_NAME, QUANTIZATION_BITS, DIRECT_PROMPT, COT_PROMPT,
    TEMPERATURE, DIRECT_MAX_TOKENS, COT_MAX_TOKENS,
    RAW_DATA_DIR, LABELED_DATA_DIR
)


def extract_number(text: str) -> Optional[float]:
    """Extract the last number from text."""
    # Find all numbers (including decimals and negatives)
    numbers = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            return None
    return None


def parse_answer(answer_text: str) -> Optional[float]:
    """Parse GSM8K answer format: '#### 42'."""
    if '####' in answer_text:
        after_hash = answer_text.split('####')[-1].strip()
        return extract_number(after_hash)
    return extract_number(answer_text)


def load_model_and_tokenizer():
    """Load quantized Mistral model."""
    print(f"Loading {MODEL_NAME} with {QUANTIZATION_BITS}-bit quantization...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
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
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_tokens: int) -> str:
    """Generate model response."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=TEMPERATURE,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from response
    response = response[len(prompt):].strip()
    return response


def generate_label(
    model, tokenizer, question: str, ground_truth: float
) -> dict:
    """Generate label for a single problem."""
    # Run direct
    direct_prompt = DIRECT_PROMPT.format(question=question)
    direct_response = generate_response(model, tokenizer, direct_prompt, DIRECT_MAX_TOKENS)
    direct_answer = extract_number(direct_response)
    direct_correct = (direct_answer is not None and abs(direct_answer - ground_truth) < 0.01)
    
    # Run CoT
    cot_prompt = COT_PROMPT.format(question=question)
    cot_response = generate_response(model, tokenizer, cot_prompt, COT_MAX_TOKENS)
    cot_answer = extract_number(cot_response)
    cot_correct = (cot_answer is not None and abs(cot_answer - ground_truth) < 0.01)
    
    # Label: 1 if CoT is NECESSARY (CoT correct AND direct wrong)
    label = 1 if (cot_correct and not direct_correct) else 0
    
    return {
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


def label_dataset(split: str = "train", limit: Optional[int] = None):
    """Label GSM8K dataset and save."""
    print(f"Loading GSM8K {split} split...")
    dataset = load_dataset("gsm8k", "main", split=split)
    
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    
    print(f"Processing {len(dataset)} problems...")
    model, tokenizer = load_model_and_tokenizer()
    
    labeled_data = []
    for item in tqdm(dataset, desc="Labeling"):
        question = item["question"]
        ground_truth = parse_answer(item["answer"])
        
        if ground_truth is None:
            continue
        
        result = generate_label(model, tokenizer, question, ground_truth)
        labeled_data.append(result)
    
    # Split and save
    train_data, val_data, test_data = split_by_index(labeled_data)
    
    Path(LABELED_DATA_DIR).mkdir(parents=True, exist_ok=True)
    
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        path = Path(LABELED_DATA_DIR) / f"{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data)} samples to {path}")
    
    # Print statistics
    total = len(labeled_data)
    cot_needed = sum(1 for d in labeled_data if d["label"] == 1)
    print(f"\nStatistics:")
    print(f"  Total: {total}")
    print(f"  CoT needed (label=1): {cot_needed} ({100*cot_needed/total:.1f}%)")
    print(f"  Direct sufficient (label=0): {total - cot_needed} ({100*(total-cot_needed)/total:.1f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples")
    args = parser.parse_args()
    
    label_dataset(split=args.split, limit=args.limit)
