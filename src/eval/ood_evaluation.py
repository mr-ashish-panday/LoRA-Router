"""Out-of-Distribution (OOD) Transfer Evaluation.

Tests if the router trained on GSM8K can generalize to other reasoning tasks.
This tests the hypothesis that the router learns a general "difficulty detector".
"""
import json
from pathlib import Path
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from src.config import MODEL_NAME, QUANTIZATION_BITS, CHECKPOINT_DIR, RESULTS_DIR
from src.models.router import LoRARouter
from src.data.labeler import extract_answer


# OOD Datasets to evaluate
OOD_DATASETS = {
    "strategyqa": {
        "name": "wics/strategy-qa",
        "split": "test",
        "question_field": "question", 
        "answer_field": "answer",
        "task_type": "boolean",  # yes/no answers
    },
    "arc_easy": {
        "name": "allenai/ai2_arc",
        "config": "ARC-Easy",
        "split": "test",
        "question_field": "question",
        "answer_field": "answerKey",
        "task_type": "multiple_choice",
    },
    "arc_challenge": {
        "name": "allenai/ai2_arc",
        "config": "ARC-Challenge",
        "split": "test",
        "question_field": "question",
        "answer_field": "answerKey",
        "task_type": "multiple_choice",
    },
}


def load_model_and_tokenizer():
    """Load quantized model for inference."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    return model, tokenizer


def format_direct_prompt(question: str, task_type: str, choices: list = None) -> str:
    """Format prompt for direct answering based on task type."""
    if task_type == "boolean":
        return f"""Answer this question with just 'yes' or 'no'.

Question: {question}

Answer:"""
    elif task_type == "multiple_choice":
        choice_str = "\n".join([f"{c['label']}: {c['text']}" for c in choices])
        return f"""Answer this multiple choice question. Give only the letter of the correct answer.

Question: {question}
{choice_str}

Answer:"""
    else:
        return f"""Answer this question directly.

Question: {question}

Answer:"""


def format_cot_prompt(question: str, task_type: str, choices: list = None) -> str:
    """Format prompt for CoT reasoning based on task type."""
    if task_type == "boolean":
        return f"""Think through this step by step, then answer 'yes' or 'no'.

Question: {question}

Let's think step by step:"""
    elif task_type == "multiple_choice":
        choice_str = "\n".join([f"{c['label']}: {c['text']}" for c in choices])
        return f"""Think through this step by step, then give the letter of the correct answer.

Question: {question}
{choice_str}

Let's think step by step:"""
    else:
        return f"""Think through this step by step.

Question: {question}

Let's think step by step:"""


def extract_boolean_answer(text: str) -> str:
    """Extract yes/no answer from text."""
    text_lower = text.lower().strip()
    if text_lower.startswith("yes"):
        return "yes"
    elif text_lower.startswith("no"):
        return "no"
    # Check for yes/no anywhere
    if "yes" in text_lower and "no" not in text_lower:
        return "yes"
    if "no" in text_lower and "yes" not in text_lower:
        return "no"
    return None


def extract_mc_answer(text: str) -> str:
    """Extract multiple choice letter from text."""
    text = text.strip().upper()
    # Check first character
    if text and text[0] in "ABCDE":
        return text[0]
    # Look for pattern like "Answer: A"
    import re
    match = re.search(r'\b([A-E])\b', text)
    if match:
        return match.group(1)
    return None


def evaluate_ood_dataset(
    dataset_key: str,
    router: LoRARouter,
    model,
    tokenizer,
    n_samples: int = 100,
    device: str = "cuda"
) -> dict:
    """Evaluate router transfer on an OOD dataset."""
    config = OOD_DATASETS[dataset_key]
    
    # Load dataset
    if "config" in config:
        dataset = load_dataset(config["name"], config["config"], split=config["split"])
    else:
        dataset = load_dataset(config["name"], split=config["split"])
    
    # Sample if needed
    if len(dataset) > n_samples:
        dataset = dataset.shuffle(seed=42).select(range(n_samples))
    
    results = {
        "direct_correct": 0,
        "cot_correct": 0,
        "router_cot_predictions": [],
        "router_correct": 0,
        "oracle_correct": 0,
    }
    
    task_type = config["task_type"]
    
    for item in tqdm(dataset, desc=f"Evaluating {dataset_key}"):
        question = item[config["question_field"]]
        true_answer = item[config["answer_field"]]
        choices = item.get("choices", None)
        
        # Format as list of dicts if needed
        if choices and isinstance(choices, dict):
            # ARC format: {"label": [...], "text": [...]}
            choices = [{"label": l, "text": t} for l, t in zip(choices["label"], choices["text"])]
        
        # Get router prediction
        inputs = tokenizer(
            question,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            router_prob = router.predict_proba(inputs["input_ids"], inputs["attention_mask"])
            use_cot = router_prob.item() > 0.5
            results["router_cot_predictions"].append(use_cot)
        
        # Generate direct answer
        direct_prompt = format_direct_prompt(question, task_type, choices)
        direct_inputs = tokenizer(direct_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            direct_output = model.generate(
                **direct_inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        direct_response = tokenizer.decode(
            direct_output[0][direct_inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        # Generate CoT answer
        cot_prompt = format_cot_prompt(question, task_type, choices)
        cot_inputs = tokenizer(cot_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            cot_output = model.generate(
                **cot_inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        cot_response = tokenizer.decode(
            cot_output[0][cot_inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        # Extract answers based on task type
        if task_type == "boolean":
            direct_ans = extract_boolean_answer(direct_response)
            cot_ans = extract_boolean_answer(cot_response)
            true_ans = "yes" if true_answer else "no"
        else:  # multiple_choice
            direct_ans = extract_mc_answer(direct_response)
            cot_ans = extract_mc_answer(cot_response)
            true_ans = true_answer
        
        # Check correctness
        direct_correct = direct_ans == true_ans if direct_ans else False
        cot_correct = cot_ans == true_ans if cot_ans else False
        
        results["direct_correct"] += int(direct_correct)
        results["cot_correct"] += int(cot_correct)
        
        # Router choice
        if use_cot:
            results["router_correct"] += int(cot_correct)
        else:
            results["router_correct"] += int(direct_correct)
        
        # Oracle (best of both)
        results["oracle_correct"] += int(direct_correct or cot_correct)
    
    # Compute percentages
    n = len(dataset)
    results["n_samples"] = n
    results["direct_accuracy"] = results["direct_correct"] / n * 100
    results["cot_accuracy"] = results["cot_correct"] / n * 100
    results["router_accuracy"] = results["router_correct"] / n * 100
    results["oracle_accuracy"] = results["oracle_correct"] / n * 100
    results["router_cot_rate"] = sum(results["router_cot_predictions"]) / n * 100
    
    return results


def run_ood_evaluation(n_samples: int = 100):
    """Run full OOD transfer evaluation."""
    print("=" * 60)
    print("OOD TRANSFER EVALUATION")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load base model
    print("\nLoading base model...")
    model, tokenizer = load_model_and_tokenizer()
    
    # Load trained router (from GSM8K training)
    print("Loading trained router...")
    router = LoRARouter()
    router.load(str(CHECKPOINT_DIR))
    router.eval()
    
    all_results = {}
    
    for dataset_key in OOD_DATASETS:
        print(f"\n{'=' * 40}")
        print(f"Evaluating: {dataset_key}")
        print("=" * 40)
        
        try:
            results = evaluate_ood_dataset(
                dataset_key, router, model, tokenizer, n_samples, device
            )
            all_results[dataset_key] = results
            
            print(f"\n{dataset_key} Results:")
            print(f"  Direct Accuracy:  {results['direct_accuracy']:.1f}%")
            print(f"  CoT Accuracy:     {results['cot_accuracy']:.1f}%")
            print(f"  Router Accuracy:  {results['router_accuracy']:.1f}%")
            print(f"  Oracle Accuracy:  {results['oracle_accuracy']:.1f}%")
            print(f"  Router CoT Rate:  {results['router_cot_rate']:.1f}%")
            
        except Exception as e:
            print(f"Error evaluating {dataset_key}: {e}")
            all_results[dataset_key] = {"error": str(e)}
    
    # Save results
    output_path = Path(RESULTS_DIR) / "ood_transfer_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_path}")
    
    # Summary table
    print("\n" + "=" * 60)
    print("OOD TRANSFER SUMMARY")
    print("=" * 60)
    print(f"{'Dataset':<20} {'Direct':<10} {'CoT':<10} {'Router':<10} {'Oracle':<10}")
    print("-" * 60)
    
    for dataset_key, results in all_results.items():
        if "error" not in results:
            print(f"{dataset_key:<20} {results['direct_accuracy']:>6.1f}%   {results['cot_accuracy']:>6.1f}%   {results['router_accuracy']:>6.1f}%   {results['oracle_accuracy']:>6.1f}%")
    
    return all_results


if __name__ == "__main__":
    run_ood_evaluation(n_samples=100)
