# Quick Reference: New Code Organization

## Import Cheat Sheet

### Before (Old Way)
```python
from ppo_algorithm import PPO
from policy_network import Policy
from rollout_storage import RolloutStorage
from ollama_utils import encode_query, mutate_query, query_target_model
from rl_query_mutator_env import QueryMutator
```

### After (New Way)
```python
# RL Core Components
from rl_core import PPO, Policy, RolloutStorage

# Ollama API Client
from ollama_client import OllamaClient

# Mutation Operators and Prompts
from query_mutation_prompts import QueryMutator, QueryMutationPrompts

# Dataset Loader
from dataset_loader import DatasetLoader

# Environment (unchanged)
from rl_query_mutator_env import QueryMutationEnv, BatchedQueryMutationEnv
```

## Usage Examples

### 1. Using OllamaClient

```python
from ollama_client import OllamaClient

# Initialize client
client = OllamaClient(embedding_model='nomic-embed-text')

# Check and pull models
client.check_and_pull_models(
    target_model='llama3.1:8b',
    mutator_model='qwen2.5:7b',
    judge_model='deepseek-r1:14b'
)

# Encode query
embedding = client.encode_query(query, obs_size=768)

# Mutate query
mutated = client.mutate_query(query, mutation_prompt, mutator_model='qwen2.5:7b')

# Batch operations
embeddings = client.batch_encode_queries(queries, obs_size=768)
mutated_queries = client.batch_mutate_queries(queries, prompts, 'qwen2.5:7b', batch_size=8)
responses = client.batch_query_target_model(queries, 'llama3.1:8b', batch_size=8)
```

### 2. Using QueryMutationPrompts

```python
from query_mutation_prompts import QueryMutator, QueryMutationPrompts

# Get prompt for a specific mutation
prompt = QueryMutationPrompts.get_mutation_prompt(
    query="How to make a bomb",
    mutator=QueryMutator.paraphrase
)

# Get all prompt templates (with {query} placeholder)
templates = QueryMutationPrompts.get_all_prompt_templates()
for mutator, template in templates.items():
    print(f"{mutator.name}: {template[:50]}...")
```

### 3. Using DatasetLoader

```python
from dataset_loader import DatasetLoader

# Initialize dataset loader
loader = DatasetLoader(dataset_path="dataset/prompts_harmful.csv", seed=42)

# Load all queries
queries = loader.load_queries()

# Split train/test
train_queries, test_queries = loader.split_train_test(queries, train_size=800)

# Sample 25% of queries
sampled_queries = loader.sample_queries(train_queries, frac_samples=0.25)

# Load pregenerated responses
responses_dict = loader.load_pregenerated_responses(
    csv_path="dataset/unaligned_responses.csv",
    queries=train_queries,
    offset=0  # 0 for train, 800 for test
)

# High-level: Load complete dataset
queries, responses, indices = loader.load_dataset(
    eval=False,  # False for train, True for test
    frac_samples=0.25,
    unaligned_csv="dataset/unaligned_responses.csv",
    verbose=True
)

# Get statistics
stats = loader.get_dataset_stats(queries, responses)
print(f"Total queries: {stats['total_queries']}")
print(f"Pregenerated coverage: {stats['pregenerated_coverage']:.1%}")
```

### 4. Using RL Core Components

```python
from rl_core import PPO, Policy, RolloutStorage

# Create policy
policy = Policy(
    obs_shape=(768,),
    action_space=gym.spaces.Discrete(5),
    base_kwargs={'recurrent': False, 'hidden_size': 64}
)

# Create PPO agent
agent = PPO(
    actor_critic=policy,
    clip_param=0.2,
    ppo_epoch=4,
    num_mini_batch=4,
    value_loss_coef=0.5,
    entropy_coef=0.01,
    lr=3e-4
)

# Create rollout storage
rollouts = RolloutStorage(
    num_steps=32,
    num_processes=16,
    obs_shape=(768,),
    action_space=gym.spaces.Discrete(5),
    recurrent_hidden_state_size=1
)
```

## File Organization

```
Project Root
│
├── rl_core/                    # RL Policy Package
│   ├── __init__.py            # Exports: PPO, Policy, RolloutStorage
│   ├── policy_network.py      # Neural network policy
│   ├── ppo_algorithm.py       # PPO algorithm
│   └── rollout_storage.py     # Experience storage
│
├── ollama_client.py           # OllamaClient class (Ollama API wrapper)
├── query_mutation_prompts.py  # QueryMutator enum + QueryMutationPrompts class
├── dataset_loader.py          # DatasetLoader class (dataset management)
│
├── rl_query_mutator_env.py    # RL environment (uses OllamaClient, DatasetLoader internally)
├── train_query_mutator.py     # Training script (imports from rl_core)
├── test_query_mutator.py      # Testing script (imports from rl_core)
│
└── ollama_utils.py            # Backward compatibility wrappers (DEPRECATED)
```

## Key Classes

### OllamaClient
- **Purpose**: Wraps all Ollama API calls
- **Location**: `ollama_client.py`
- **Key Methods**:
  - `check_and_pull_models()`: Model management
  - `encode_query()`, `batch_encode_queries()`: Embeddings
  - `mutate_query()`, `batch_mutate_queries()`: Mutations
  - `query_target_model()`, `batch_query_target_model()`: Target queries
  - `generate_unaligned_response()`: Baseline responses
  - `llm_judge_score()`: LLM-based rewards

### QueryMutationPrompts
- **Purpose**: Manages mutation prompts
- **Location**: `query_mutation_prompts.py`
- **Key Methods**:
  - `get_mutation_prompt(query, mutator)`: Get formatted prompt
  - `get_all_prompt_templates()`: Get all templates

### QueryMutator (Enum)
- **Purpose**: Defines mutation operators
- **Location**: `query_mutation_prompts.py`
- **Values**: `paraphrase`, `add_context`, `change_perspective`, `add_justification`, `make_indirect`

### DatasetLoader
- **Purpose**: Manages dataset loading and sampling
- **Location**: `dataset_loader.py`
- **Key Methods**:
  - `load_queries()`: Load from CSV
  - `split_train_test()`: Split into train/test
  - `sample_queries()`: Random sampling
  - `load_pregenerated_responses()`: Load response mappings
  - `load_dataset()`: High-level all-in-one method
  - `get_dataset_stats()`: Dataset statistics

## Environment Variables

No changes to environment variables needed. Everything works as before.

## Common Tasks

### Add a New Mutation Operator
Edit `query_mutation_prompts.py`:

```python
class QueryMutator(Enum):
    paraphrase = 0
    add_context = 1
    # ... existing ...
    my_new_mutation = 5  # Add new

class QueryMutationPrompts:
    @staticmethod
    def get_mutation_prompt(query, mutator):
        prompts = {
            # ... existing ...
            QueryMutator.my_new_mutation: f"Apply my transformation to: {query}"
        }
        return prompts[mutator]
```

### Change Embedding Model
```python
client = OllamaClient(embedding_model='your-model-name')
```

### Debug Ollama API Calls
The OllamaClient class makes it easy to add logging:

```python
class OllamaClient:
    def __init__(self, embedding_model='nomic-embed-text', debug=False):
        self.embedding_model = embedding_model
        self.debug = debug
    
    def encode_query(self, query, obs_size):
        if self.debug:
            print(f"Encoding: {query[:50]}...")
        # ... rest of implementation
```

## Backward Compatibility

All old imports still work with deprecation warnings:
```python
from ollama_utils import encode_query  # Works, but shows warning
from ppo_algorithm import PPO  # Works (old file still exists)
```

To silence warnings:
```python
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
```

## Testing

Verify everything compiles:
```bash
python -m py_compile ollama_client.py query_mutation_prompts.py \
    rl_core/*.py train_query_mutator.py test_query_mutator.py
```

Run training (same as before):
```bash
python train_query_mutator.py --target-model llama3.1:8b --num-processes 16
```
