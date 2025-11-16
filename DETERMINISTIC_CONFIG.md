# Deterministic Configuration Guide

## Overview
This guide explains how to get consistent/reproducible outputs from Ollama and PyTorch-based models.

## Key Settings for Deterministic Outputs

### 1. Ollama Parameters

For **completely deterministic** output from Ollama models:

```python
ollama_chat_api(
    model_name="model-name",
    system_prompt="...",
    user_prompt="...",
    temperature=0.0,  # MUST be 0 for deterministic
    top_p=1.0,        # MUST be 1.0 for deterministic
    top_k=1,          # MUST be 1 for deterministic (only top token)
    seed=42           # Fixed seed (any integer)
)
```

**Parameter Explanations:**

- **`temperature=0.0`**: Disables randomness in token selection. Always picks the most probable token.
- **`top_k=1`**: Only considers the single most likely token at each step.
- **`top_p=1.0`**: Disables nucleus sampling (considers all tokens, but with temperature=0, only top matters).
- **`seed=42`**: Ensures the model's internal random state is consistent across runs.

### 2. Python Random Seeds

At the top of your main script (`vlmtest.py`), set all random seeds:

```python
import random
import numpy as np
import torch

RANDOM_SEED = 42

def set_random_seeds(seed=RANDOM_SEED):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)                      # Python random
    np.random.seed(seed)                   # NumPy
    torch.manual_seed(seed)                # PyTorch CPU
    torch.cuda.manual_seed(seed)           # PyTorch GPU (single GPU)
    torch.cuda.manual_seed_all(seed)       # PyTorch GPU (multi-GPU)
    torch.backends.cudnn.deterministic = True  # cuDNN deterministic
    torch.backends.cudnn.benchmark = False     # Disable cuDNN auto-tuner
    print(f"[info] Set random seed to {seed} for reproducibility")

# Call this at the start of your script
set_random_seeds(RANDOM_SEED)
```

### 3. PyTorch Model Generation (for VLM)

In your `generate()` function for the vision-language model:

```python
with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,      # Disable sampling for deterministic output
        temperature=0.0,      # (optional when do_sample=False)
    )
```

## What Each Seed Controls

| Seed | What It Controls |
|------|------------------|
| `random.seed()` | Python's built-in `random` module |
| `np.random.seed()` | NumPy random number generation |
| `torch.manual_seed()` | PyTorch CPU operations |
| `torch.cuda.manual_seed()` | PyTorch single GPU operations |
| `torch.cuda.manual_seed_all()` | PyTorch multi-GPU operations |
| `torch.backends.cudnn.deterministic` | Forces cuDNN to use deterministic algorithms |
| `torch.backends.cudnn.benchmark` | Disables cuDNN auto-tuner (which can introduce variability) |
| Ollama `seed` parameter | Ollama model's internal random state |

## Important Notes

### Trade-offs of Deterministic Settings

1. **Performance**: Deterministic mode may be slightly slower than normal mode
   - `cudnn.deterministic = True` disables some optimizations
   - `cudnn.benchmark = False` prevents auto-tuning for your specific hardware

2. **Creativity**: With `temperature=0.0`, outputs will be:
   - More repetitive
   - Less diverse
   - Always the same for the same input

### When to Use Deterministic Settings

✅ **Use deterministic settings when:**
- Running experiments that need to be reproducible
- Debugging model behavior
- Comparing different prompts/approaches fairly
- Creating benchmarks or evaluations

❌ **Don't use deterministic settings when:**
- You want creative/diverse outputs
- Running in production where variety is desired
- Generating multiple responses to the same prompt

## Verification

To verify deterministic behavior, run the same prompt multiple times:

```python
# Should produce identical results every time
for i in range(3):
    result = ollama_chat_api(
        model_name="deepseek-r1:14b",
        system_prompt="You are helpful.",
        user_prompt="What is 2+2?",
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        seed=42
    )
    print(f"Run {i+1}: {result}")
```

All three runs should produce **exactly** the same output.

## Current Implementation

The following files have been updated with deterministic settings:

1. **`vlmtest.py`**: 
   - Sets all random seeds at startup
   - Uses deterministic parameters for uncensored model generation

2. **`ollama_api.py`**: 
   - `ollama_generate_api()` now accepts deterministic parameters
   - `ollama_chat_api()` already had these parameters

3. **`reward_utils.py`**: 
   - `llm_as_judge_reward_score()` uses deterministic parameters (temperature=0.0, top_k=1, top_p=1.0)

## Quick Reference

```python
# Deterministic Configuration
RANDOM_SEED = 42

# Ollama Deterministic Parameters
{
    'temperature': 0.0,
    'top_k': 1,
    'top_p': 1.0,
    'seed': RANDOM_SEED
}

# PyTorch Deterministic Parameters
{
    'do_sample': False,
    'temperature': 0.0
}
```
