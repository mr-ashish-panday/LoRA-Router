"""Baseline routing strategies."""
import re
import random
from abc import ABC, abstractmethod
from typing import List, Optional
import torch
import numpy as np


class BaseRouter(ABC):
    """Abstract base class for routing strategies."""
    
    @abstractmethod
    def predict(self, question: str) -> float:
        """Return probability of needing CoT (0-1)."""
        pass
    
    def route(self, question: str, threshold: float = 0.5) -> bool:
        """Return True if should use CoT."""
        return self.predict(question) > threshold


class AlwaysDirectRouter(BaseRouter):
    """Always route to direct answer."""
    def predict(self, question: str) -> float:
        return 0.0


class AlwaysCoTRouter(BaseRouter):
    """Always route to chain-of-thought."""
    def predict(self, question: str) -> float:
        return 1.0


class RandomRouter(BaseRouter):
    """Random 50/50 routing."""
    def __init__(self, seed: int = 42):
        random.seed(seed)
    
    def predict(self, question: str) -> float:
        return random.random()


class LengthRouter(BaseRouter):
    """Route based on question length."""
    def __init__(self, threshold_words: int = 50):
        self.threshold = threshold_words
    
    def predict(self, question: str) -> float:
        word_count = len(question.split())
        # Sigmoid-like scaling
        return 1 / (1 + np.exp(-(word_count - self.threshold) / 10))


class KeywordRouter(BaseRouter):
    """Route based on keywords indicating complexity."""
    
    COMPLEX_KEYWORDS = [
        "calculate", "compute", "total", "sum", "difference",
        "how many", "how much", "if", "then", "after", "before",
        "remaining", "left", "more than", "less than", "times",
        "each", "per", "average", "percent", "fraction",
    ]
    
    def predict(self, question: str) -> float:
        question_lower = question.lower()
        matches = sum(1 for kw in self.COMPLEX_KEYWORDS if kw in question_lower)
        # More matches = higher probability
        return min(1.0, matches / 5)


class EntropyRouter(BaseRouter):
    """Route based on model answer entropy."""
    
    def __init__(self, model, tokenizer, num_samples: int = 3):
        self.model = model
        self.tokenizer = tokenizer
        self.num_samples = num_samples
    
    def _get_answer_entropy(self, question: str) -> float:
        """Compute entropy over sampled answers."""
        from src.config import DIRECT_PROMPT, DIRECT_MAX_TOKENS
        
        prompt = DIRECT_PROMPT.format(question=question)
        answers = []
        
        for _ in range(self.num_samples):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=DIRECT_MAX_TOKENS,
                    temperature=0.7,  # Need sampling for entropy
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract number
            numbers = re.findall(r'-?\d+\.?\d*', response.replace(',', ''))
            if numbers:
                answers.append(numbers[-1])
        
        # Compute entropy
        if len(answers) == 0:
            return 1.0  # High uncertainty
        
        unique = len(set(answers))
        return unique / len(answers)  # 1 = all different, 0 = all same
    
    def predict(self, question: str) -> float:
        return self._get_answer_entropy(question)


class SelfConsistencyRouter(BaseRouter):
    """Route if two direct answers disagree."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def _get_answer(self, question: str, temperature: float) -> Optional[str]:
        """Get single answer."""
        from src.config import DIRECT_PROMPT, DIRECT_MAX_TOKENS
        
        prompt = DIRECT_PROMPT.format(question=question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=DIRECT_MAX_TOKENS,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        numbers = re.findall(r'-?\d+\.?\d*', response.replace(',', ''))
        return numbers[-1] if numbers else None
    
    def predict(self, question: str) -> float:
        # Get two answers with slight temperature
        answer1 = self._get_answer(question, temperature=0.3)
        answer2 = self._get_answer(question, temperature=0.3)
        
        if answer1 is None or answer2 is None:
            return 1.0  # Uncertain, use CoT
        
        return 0.0 if answer1 == answer2 else 1.0


class OracleRouter(BaseRouter):
    """Oracle router using ground truth labels."""
    
    def __init__(self, labels: dict):
        """labels: dict mapping question -> 0/1"""
        self.labels = labels
    
    def predict(self, question: str) -> float:
        return float(self.labels.get(question, 0))


class EmbeddingRouter(BaseRouter):
    """Simple logistic regression on frozen embeddings."""
    
    def __init__(self, model, tokenizer, classifier=None):
        self.model = model
        self.tokenizer = tokenizer
        self.classifier = classifier  # sklearn LogisticRegression
    
    def _get_embedding(self, question: str) -> np.ndarray:
        """Get mean-pooled embedding."""
        inputs = self.tokenizer(
            question, return_tensors="pt", truncation=True, max_length=512
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
        
        # Mean pool last layer
        hidden = outputs.hidden_states[-1]
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        
        return pooled.cpu().numpy().flatten()
    
    def fit(self, questions: List[str], labels: List[int]):
        """Train logistic regression."""
        from sklearn.linear_model import LogisticRegression
        
        embeddings = np.array([self._get_embedding(q) for q in questions])
        self.classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.classifier.fit(embeddings, labels)
    
    def predict(self, question: str) -> float:
        if self.classifier is None:
            raise ValueError("Classifier not trained. Call fit() first.")
        
        embedding = self._get_embedding(question).reshape(1, -1)
        return self.classifier.predict_proba(embedding)[0, 1]


def get_all_baselines(model=None, tokenizer=None, labels=None) -> dict:
    """Get all baseline routers."""
    baselines = {
        "always_direct": AlwaysDirectRouter(),
        "always_cot": AlwaysCoTRouter(),
        "random": RandomRouter(),
        "length": LengthRouter(),
        "keyword": KeywordRouter(),
    }
    
    if model is not None and tokenizer is not None:
        baselines["entropy"] = EntropyRouter(model, tokenizer)
        baselines["self_consistency"] = SelfConsistencyRouter(model, tokenizer)
        baselines["embedding"] = EmbeddingRouter(model, tokenizer)
    
    if labels is not None:
        baselines["oracle"] = OracleRouter(labels)
    
    return baselines
