# Project Structure

## Directory Tree

\`\`\`
cs593rl-project/
│
├── Core RL Training & Testing
│   ├── train_query_mutator.py (16K)      # Main training script
│   ├── test_query_mutator.py (6.9K)       # Testing/evaluation script
│   └── rl_query_mutator_env.py (15K)      # Gymnasium environment
│
├── RL Core Package (Policy Components)
│   └── rl_core/
│       ├── __init__.py (242B)             # Package exports
│       ├── policy_network.py (8.0K)       # Neural network policy
│       ├── ppo_algorithm.py (3.8K)        # PPO algorithm
│       └── rollout_storage.py (7.9K)      # Experience replay buffer
│
├── API & Client Classes
│   ├── ollama_client.py (14K)             # Ollama API wrapper class
│   ├── query_mutation_prompts.py (2.6K)   # Mutation operators & prompts
│   └── dataset_loader.py (8.5K)           # Dataset loading & management
│
├── Utilities & Helpers
│   ├── ollama_utils.py (13K)              # Backward compatibility wrappers
│   ├── pregenerate_unaligned_responses.py (3.1K)  # Response pregeneration
│   ├── image_prompt_generator.py (11K)    # Text-to-image for VLM attacks
│   ├── reward_utils.py (5.7K)             # Keyword-based rewards
│   ├── reward_utils_llm_judge.py (5.5K)   # LLM judge rewards
│   └── benchmark_batching.py (5.4K)       # Performance benchmarking
│
├── Legacy/Compatibility Files
│   ├── policy_network.py (8.0K)           # Old location (use rl_core/)
│   ├── ppo_algorithm.py (3.8K)            # Old location (use rl_core/)
│   └── rollout_storage.py (7.9K)          # Old location (use rl_core/)
│
├── Experimental/Other
│   ├── pipeline.py (5.7K)                 # Alternative pipeline
│   ├── target.py (4.2K)                   # Target model utilities
│   ├── TextSimilaryReward.py (13K)        # Text similarity scoring
│   ├── ollama_api.py (2.2K)               # Alternative API interface
│   └── demo_frac_samples.py (2.2K)        # Dataset sampling demo
│
├── Dataset Files
│   └── dataset/
│       ├── prompts_harmful.csv            # 1020 harmful queries
│       ├── unaligned_responses.csv        # Pregenerated responses
│       ├── prompts_harmful_all.csv        # All harmful prompts
│       ├── SafeBench-Tiny.csv             # SafeBench dataset
│       └── [other CSV files]
│
├── Trained Models
│   └── trained_models_query_mutator/
│       ├── final_model.pt                 # Final trained model
│       ├── checkpoint_*.pt                # Training checkpoints
│       └── training_log_*.csv             # Training logs
│
└── Documentation
    ├── README.md                          # Main documentation
    ├── REORGANIZATION_SUMMARY.md          # Code reorganization details
    ├── QUICK_REFERENCE.md                 # Import cheat sheet
    ├── BATCHING_IMPROVEMENTS.md           # Batching documentation
    ├── BATCH_SIZE_CONTROL.md              # Batch size tuning
    ├── DATASET_SAMPLING.md                # Dataset sampling guide
    └── PREGENERATION_GUIDE.md             # Response pregeneration guide
\`\`\`

---

## Files Used in Training (train_query_mutator.py)

### Direct Imports
\`\`\`python
# Core RL Components
from rl_core import PPO, Policy, RolloutStorage

# Environment
from rl_query_mutator_env import QueryMutationEnv, BatchedQueryMutationEnv

# Standard libraries
import torch, numpy, argparse, csv, tqdm
\`\`\`

### Indirect Dependencies (used by environment)
\`\`\`python
# Environment depends on:
from ollama_client import OllamaClient
from query_mutation_prompts import QueryMutator, QueryMutationPrompts
from dataset_loader import DatasetLoader
from image_prompt_generator import TextToImageConverter, ImagePromptStyle
\`\`\`

### Complete Training Dependency Tree
\`\`\`
train_query_mutator.py
├── rl_core/
│   ├── policy_network.py
│   ├── ppo_algorithm.py
│   └── rollout_storage.py
├── rl_query_mutator_env.py
│   ├── ollama_client.py
│   ├── query_mutation_prompts.py
│   ├── dataset_loader.py
│   └── image_prompt_generator.py (optional, for VLM)
└── Standard libraries (torch, numpy, pandas, gymnasium)
\`\`\`

### Required Dataset Files
- \`dataset/prompts_harmful.csv\` (queries 0-799 for training)
- \`dataset/unaligned_responses.csv\` (optional, for LLM judge)

---

## Files Used in Testing (test_query_mutator.py)

### Direct Imports
\`\`\`python
# Core RL Components
from rl_core import Policy

# Environment & Mutations
from rl_query_mutator_env import QueryMutationEnv
from query_mutation_prompts import QueryMutator

# Standard libraries
import torch, numpy, argparse, json
\`\`\`

### Indirect Dependencies
\`\`\`python
# Environment depends on:
from ollama_client import OllamaClient
from query_mutation_prompts import QueryMutationPrompts
from dataset_loader import DatasetLoader
from image_prompt_generator import TextToImageConverter (optional)
\`\`\`

### Complete Testing Dependency Tree
\`\`\`
test_query_mutator.py
├── rl_core/
│   └── policy_network.py
├── rl_query_mutator_env.py
│   ├── ollama_client.py
│   ├── query_mutation_prompts.py
│   └── dataset_loader.py
├── query_mutation_prompts.py (for QueryMutator enum)
└── Standard libraries (torch, numpy, pandas, gymnasium)
\`\`\`

### Required Files
- \`trained_models_query_mutator/final_model.pt\` (or checkpoint)
- \`dataset/prompts_harmful.csv\` (queries 800-1019 for testing)
- \`dataset/unaligned_responses.csv\` (optional, for LLM judge)

---

## Optional/Utility Files

### Pregeneration (speeds up LLM judge training)
\`\`\`
pregenerate_unaligned_responses.py
├── ollama_client.py
└── dataset_loader.py
\`\`\`

### Benchmarking
\`\`\`
benchmark_batching.py
├── ollama_client.py
└── Standard timing libraries
\`\`\`

---

## File Size Summary

| Category | Files | Total Size |
|----------|-------|------------|
| **Core Training/Test** | 3 files | 38K |
| **RL Core Package** | 4 files | 24K |
| **API & Classes** | 3 files | 25K |
| **Utilities** | 6 files | 50K |
| **Legacy Files** | 3 files | 24K |
| **Documentation** | 7+ files | ~50K |
| **Total** | ~26 Python files | ~211K |

---

## Minimal Training Setup

To train, you only need these files:

✅ **Essential (8 files):**
1. \`train_query_mutator.py\`
2. \`rl_query_mutator_env.py\`
3. \`rl_core/__init__.py\`
4. \`rl_core/policy_network.py\`
5. \`rl_core/ppo_algorithm.py\`
6. \`rl_core/rollout_storage.py\`
7. \`ollama_client.py\`
8. \`query_mutation_prompts.py\`
9. \`dataset_loader.py\`

📁 **Dataset:**
- \`dataset/prompts_harmful.csv\`

⚙️ **Optional (for faster LLM judge):**
- \`dataset/unaligned_responses.csv\`
- \`pregenerate_unaligned_responses.py\` (to create it)

---

## Minimal Testing Setup

To test, you need:

✅ **Essential (7 files):**
1. \`test_query_mutator.py\`
2. \`rl_query_mutator_env.py\`
3. \`rl_core/__init__.py\`
4. \`rl_core/policy_network.py\`
5. \`ollama_client.py\`
6. \`query_mutation_prompts.py\`
7. \`dataset_loader.py\`

📁 **Dataset:**
- \`dataset/prompts_harmful.csv\`

🤖 **Trained Model:**
- \`trained_models_query_mutator/final_model.pt\`

---

## Files NOT Needed for Training/Testing

❌ **Not required:**
- \`pipeline.py\` (alternative pipeline)
- \`target.py\` (alternative utilities)
- \`TextSimilaryReward.py\` (alternative reward)
- \`ollama_api.py\` (alternative API)
- \`demo_frac_samples.py\` (demo script)
- \`benchmark_batching.py\` (benchmarking only)
- \`reward_utils.py\` (integrated in environment)
- \`reward_utils_llm_judge.py\` (integrated in environment)
- \`image_prompt_generator.py\` (only for VLM attacks)
- \`ollama_utils.py\` (backward compatibility only)
- Legacy files in root (use \`rl_core/\` versions)

---

## Quick Command Reference

### Train with minimal setup:
\`\`\`bash
python train_query_mutator.py --target-model llama3.1:8b --num-processes 16
\`\`\`

### Test trained model:
\`\`\`bash
python test_query_mutator.py --model-path trained_models_query_mutator/final_model.pt
\`\`\`

### Pregenerate responses (optional):
\`\`\`bash
python pregenerate_unaligned_responses.py
\`\`\`

