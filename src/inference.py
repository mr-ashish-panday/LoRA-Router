"""Inference pipeline with routing."""
import json
import time
from pathlib import Path
from typing import Optional, List, Dict
import torch
from tqdm import tqdm

from src.config import (
    DIRECT_PROMPT, COT_PROMPT, ROUTER_THRESHOLD,
    DIRECT_MAX_TOKENS, COT_MAX_TOKENS, CHECKPOINT_DIR
)
from src.data.labeler import load_model_and_tokenizer, generate_response, extract_answer
from src.models.router import LoRARouter


class RoutingInference:
    """Inference pipeline with intelligent routing."""
    
    def __init__(
        self,
        router_path: Optional[str] = None,
        threshold: float = ROUTER_THRESHOLD,
    ):
        self.threshold = threshold
        
        # Load model
        print("Loading base model...")
        self.model, self.tokenizer = load_model_and_tokenizer()
        
        # Load router
        if router_path:
            print(f"Loading router from {router_path}...")
            self.router = LoRARouter(base_model=self.model)
            self.router.load(router_path)
        else:
            self.router = None
    
    def predict_route(self, question: str) -> float:
        """Get routing probability (needs CoT)."""
        if self.router is None:
            raise ValueError("Router not loaded")
        
        inputs = self.tokenizer(
            question, return_tensors="pt", truncation=True, max_length=512
        ).to(self.model.device)
        
        return self.router.predict(
            inputs["input_ids"], inputs["attention_mask"]
        ).item()
    
    def direct_answer(self, question: str) -> Dict:
        """Get direct answer."""
        start = time.time()
        prompt = DIRECT_PROMPT.format(question=question)
        response = generate_response(
            self.model, self.tokenizer, prompt, DIRECT_MAX_TOKENS
        )
        elapsed = time.time() - start
        
        return {
            "response": response,
            "answer": extract_answer(response),
            "time_sec": elapsed,
            "tokens": len(self.tokenizer.encode(response)),
            "method": "direct",
        }
    
    def cot_answer(self, question: str) -> Dict:
        """Get chain-of-thought answer."""
        start = time.time()
        prompt = COT_PROMPT.format(question=question)
        response = generate_response(
            self.model, self.tokenizer, prompt, COT_MAX_TOKENS
        )
        elapsed = time.time() - start
        
        return {
            "response": response,
            "answer": extract_answer(response),
            "time_sec": elapsed,
            "tokens": len(self.tokenizer.encode(response)),
            "method": "cot",
        }
    
    def routed_answer(self, question: str) -> Dict:
        """Get answer using intelligent routing."""
        # Get routing decision
        route_start = time.time()
        route_prob = self.predict_route(question)
        route_time = time.time() - route_start
        
        use_cot = route_prob > self.threshold
        
        # Get answer
        if use_cot:
            result = self.cot_answer(question)
        else:
            result = self.direct_answer(question)
        
        result["route_prob"] = route_prob
        result["route_time_sec"] = route_time
        result["routed_to"] = "cot" if use_cot else "direct"
        
        return result
    
    def batch_inference(
        self,
        questions: List[str],
        ground_truths: Optional[List[float]] = None,
        use_routing: bool = True,
    ) -> List[Dict]:
        """Run inference on batch of questions."""
        results = []
        
        for i, question in enumerate(tqdm(questions, desc="Inference")):
            if use_routing and self.router is not None:
                result = self.routed_answer(question)
            else:
                # Default to CoT for comparison
                result = self.cot_answer(question)
            
            result["question"] = question
            
            if ground_truths is not None:
                gt = ground_truths[i]
                result["ground_truth"] = gt
                result["correct"] = (
                    result["answer"] is not None and 
                    abs(result["answer"] - gt) < 0.01
                )
            
            results.append(result)
        
        return results


def run_comparison(data_path: str, output_path: str):
    """Compare routed vs always-CoT inference."""
    # Load test data
    with open(data_path, "r") as f:
        data = json.load(f)
    
    questions = [d["question"] for d in data]
    ground_truths = [d["ground_truth"] for d in data]
    
    # Initialize
    pipeline = RoutingInference(router_path=CHECKPOINT_DIR)
    
    # Run routed
    print("\n=== Routed Inference ===")
    routed_results = pipeline.batch_inference(questions, ground_truths, use_routing=True)
    
    # Run always-CoT
    print("\n=== Always-CoT Inference ===")
    cot_results = pipeline.batch_inference(questions, ground_truths, use_routing=False)
    
    # Compute stats
    routed_acc = sum(r["correct"] for r in routed_results) / len(routed_results)
    routed_time = sum(r["time_sec"] for r in routed_results)
    routed_tokens = sum(r["tokens"] for r in routed_results)
    cot_rate = sum(1 for r in routed_results if r["routed_to"] == "cot") / len(routed_results)
    
    cot_acc = sum(r["correct"] for r in cot_results) / len(cot_results)
    cot_time = sum(r["time_sec"] for r in cot_results)
    cot_tokens = sum(r["tokens"] for r in cot_results)
    
    comparison = {
        "routed": {
            "accuracy": routed_acc,
            "total_time_sec": routed_time,
            "total_tokens": routed_tokens,
            "cot_usage_rate": cot_rate,
        },
        "always_cot": {
            "accuracy": cot_acc,
            "total_time_sec": cot_time,
            "total_tokens": cot_tokens,
        },
        "speedup": cot_time / routed_time if routed_time > 0 else 0,
        "token_savings": (cot_tokens - routed_tokens) / cot_tokens,
        "accuracy_drop": cot_acc - routed_acc,
    }
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n=== Results ===")
    print(f"Routed:     {routed_acc:.1%} acc, {routed_time:.1f}s, {cot_rate:.1%} CoT")
    print(f"Always-CoT: {cot_acc:.1%} acc, {cot_time:.1f}s")
    print(f"Speedup:    {comparison['speedup']:.2f}x")
    print(f"Saved to:   {output_path}")
    
    return comparison


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/labeled/test.json")
    parser.add_argument("--output", default="data/results/comparison.json")
    args = parser.parse_args()
    
    run_comparison(args.data, args.output)
