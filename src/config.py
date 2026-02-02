"""Configuration constants for the LoRA Router project."""

# Model config - Qwen2 is strong at math and doesn't require license approval
MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
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

# Prompts - MUST enforce #### format for reliable parsing
DIRECT_PROMPT = """Solve this math problem. Give only the final numerical answer.
Format your final answer as: #### <number>

Problem: {question}

Answer:"""

COT_PROMPT = """Solve this math problem step by step.
After your reasoning, format your final answer as: #### <number>

Problem: {question}

Solution:"""

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
