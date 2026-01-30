"""Configuration constants for the LoRA Router project."""

# Model config
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
QUANTIZATION_BITS = 4

# LoRA config
LORA_RANK = 4
LORA_ALPHA = 8
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# Training config
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
MAX_LENGTH = 512

# Prompts
DIRECT_PROMPT = """Solve this math problem. Give only the final numerical answer.

Problem: {question}

Answer:"""

COT_PROMPT = """Solve this math problem step by step. Show your reasoning, then give the final answer.

Problem: {question}

Let's solve this step by step:"""

# Paths
DATA_DIR = "data"
RAW_DATA_DIR = f"{DATA_DIR}/raw"
LABELED_DATA_DIR = f"{DATA_DIR}/labeled"
RESULTS_DIR = f"{DATA_DIR}/results"
CHECKPOINT_DIR = "checkpoints"

# Inference
TEMPERATURE = 0.0
DIRECT_MAX_TOKENS = 50
COT_MAX_TOKENS = 512
ROUTER_THRESHOLD = 0.5
