# Dataset Sampling Feature

## Overview

The `--frac-samples` parameter allows you to randomly sample a fraction of the training dataset instead of using all queries. This is useful for rapid prototyping, hyperparameter tuning, and resource-constrained environments.

## Usage

### Basic Syntax

```bash
python train_query_mutator.py --frac-samples <fraction>
```

Where `<fraction>` is a float between 0.0 and 1.0:
- `1.0` = Use all data (default)
- `0.5` = Use 50% of data
- `0.25` = Use 25% of data
- `0.1` = Use 10% of data

### Examples

**Fast Prototyping (25% of data)**
```bash
python train_query_mutator.py \
    --frac-samples 0.25 \
    --num-env-steps 5000
```
- Train set: 200 queries (instead of 800)
- Test set: 55 queries (instead of 220)
- Time: ~75% faster

**Hyperparameter Testing (10% of data)**
```bash
python train_query_mutator.py \
    --frac-samples 0.1 \
    --num-env-steps 2000 \
    --lr 5e-4
```
- Train set: 80 queries
- Test set: 22 queries
- Time: ~90% faster

**Medium Training (50% of data)**
```bash
python train_query_mutator.py \
    --frac-samples 0.5 \
    --num-env-steps 15000
```
- Train set: 400 queries
- Test set: 110 queries
- Time: ~50% faster

**Full Training (all data, default)**
```bash
python train_query_mutator.py \
    --frac-samples 1.0 \
    --num-env-steps 50000
```
- Train set: 800 queries
- Test set: 220 queries

## How It Works

1. **Load Dataset**: Loads all queries from `dataset/prompts_harmful.csv`
2. **Split Train/Test**: Splits into train (0-799) and test (800-1019)
3. **Random Sampling**: Randomly samples `frac_samples * size` queries (keeping track of indices)
4. **Load Pregenerated Responses**: Loads responses from CSV (same sequence as dataset)
5. **Index-based Mapping**: Maps sampled queries to responses using their original indices
6. **Training**: Uses sampled subset with correctly mapped pregenerated responses

**Important:** 
- Sampling is random on each run. Use `--seed` for reproducibility.
- Pregenerated responses CSV must have the same sequence as the dataset CSV
- Index-based mapping ensures O(1) lookup instead of O(n) dictionary search
- Much faster and more memory efficient than dictionary-based matching

## Use Cases

### 1. Rapid Prototyping
Test your ideas quickly with a small dataset:
```bash
python train_query_mutator.py --frac-samples 0.1 --num-env-steps 1000
```

### 2. Hyperparameter Search
Iterate faster when tuning hyperparameters:
```bash
# Test different learning rates
for lr in 1e-4 3e-4 5e-4 1e-3; do
    python train_query_mutator.py \
        --lr $lr \
        --frac-samples 0.2 \
        --num-env-steps 3000 \
        --save-dir models_lr_$lr
done
```

### 3. Model Comparison
Compare different models on smaller datasets:
```bash
# Compare mutator models
python train_query_mutator.py --mutator-model qwen2.5:7b --frac-samples 0.3
python train_query_mutator.py --mutator-model llama3.1:8b --frac-samples 0.3
python train_query_mutator.py --mutator-model gemma2:9b --frac-samples 0.3
```

### 4. Resource-Constrained Environments
Train on machines with limited memory:
```bash
python train_query_mutator.py \
    --frac-samples 0.5 \
    --num-processes 4 \
    --batch-size 4
```

### 5. Development Testing
Quickly verify code changes work:
```bash
python train_query_mutator.py \
    --frac-samples 0.05 \
    --num-env-steps 500 \
    --log-interval 5
```

## Performance Impact

| frac_samples | Train Queries | Time Reduction | Use Case |
|--------------|---------------|----------------|----------|
| 1.0 | 800 | 0% (baseline) | Production training |
| 0.5 | 400 | ~50% | Medium-scale experiments |
| 0.25 | 200 | ~75% | Fast prototyping |
| 0.1 | 80 | ~90% | Hyperparameter testing |
| 0.05 | 40 | ~95% | Code verification |

**Note:** Time reduction is approximate and depends on other factors (batching, models, etc.)

## Reproducibility

Random sampling is controlled by `--seed`:

```bash
# These will use the same random sample
python train_query_mutator.py --frac-samples 0.5 --seed 42
python train_query_mutator.py --frac-samples 0.5 --seed 42

# These will use different random samples
python train_query_mutator.py --frac-samples 0.5 --seed 42
python train_query_mutator.py --frac-samples 0.5 --seed 123
```

## Combining with Other Parameters

### Fast Development Mode
```bash
python train_query_mutator.py \
    --frac-samples 0.1 \
    --num-processes 4 \
    --batch-size 4 \
    --num-env-steps 1000 \
    --mutator-model qwen2.5:3b
```

### Medium Experimentation
```bash
python train_query_mutator.py \
    --frac-samples 0.3 \
    --num-processes 8 \
    --batch-size 8 \
    --num-env-steps 5000 \
    --mutator-model qwen2.5:7b
```

### Production Training
```bash
python train_query_mutator.py \
    --frac-samples 1.0 \
    --num-processes 16 \
    --batch-size 8 \
    --num-env-steps 50000 \
    --use-llm-judge \
    --unaligned-csv dataset/unaligned_responses.csv
```

## Testing with frac-samples

The test script also supports this parameter:

```bash
python test_query_mutator.py \
    --model-path trained_models/final_model.pt \
    --frac-samples 0.5 \
    --num-episodes 50
```

This will test on 50% of the test set (110 queries instead of 220).

## Tips

1. **Start small**: Use `--frac-samples 0.1` for initial development
2. **Scale up gradually**: 0.1 → 0.25 → 0.5 → 1.0
3. **Use with --seed**: Ensure reproducible experiments
4. **Adjust env-steps**: Reduce `--num-env-steps` proportionally
5. **Combine with batching**: Still get batching speedup with smaller datasets
6. **Log frequently**: Use `--log-interval 5` to monitor progress closely

## Validation

The environment will print the sampling information:
```
Randomly sampled 200/800 queries (25.0%)
Loading pregenerated unaligned responses from dataset/unaligned_responses.csv...
Loaded 1020 total pregenerated responses
Mapped 200/200 queries to pregenerated responses (100.0%)
```

This shows:
- 200 queries were sampled from the 800 available
- All 200 have pregenerated unaligned responses (100% match via index mapping)
- No on-the-fly generation needed

If invalid values are provided:
```
Warning: frac_samples=1.5 invalid, using all data (1.0)
```

## Synchronization with Pregenerated Responses

When using `--frac-samples` with `--use-llm-judge` and `--unaligned-csv`:

1. **Queries are randomly sampled** based on `frac_samples` (with indices tracked)
2. **All pregenerated responses are loaded** from CSV
3. **Index-based mapping** matches sampled queries to responses by position
4. **Mapped responses are reported** (e.g., "Mapped 200/200 queries")

This ensures:
- ✅ **O(1) index-based lookup** instead of O(n) dictionary search
- ✅ **Memory efficient** - only stores responses for sampled queries
- ✅ **100% match guaranteed** - uses sequential index mapping
- ✅ **Fast loading** - no string comparison needed

**Example output with sampling:**
```
Randomly sampled 200/800 queries (25.0%)
Loading pregenerated unaligned responses from dataset/unaligned_responses.csv...
Loaded 1020 total pregenerated responses
Mapped 200/200 queries to pregenerated responses (100.0%)
```

**How index-based mapping works:**
1. Sample query at index 150 from train set → Maps to response at index 150 in CSV
2. Sample query at index 830 from test set → Maps to response at index 830 in CSV (offset handled automatically)
3. No string comparison needed - direct index lookup

**Requirements:**
- ✅ Pregenerated responses CSV must have **same sequence** as dataset CSV
- ✅ One response per line in same order as queries
- ✅ Automatically handles train/test split offsets

## Common Workflows

### Workflow 1: Prototype → Production
```bash
# 1. Quick prototype (5 minutes)
python train_query_mutator.py --frac-samples 0.1 --num-env-steps 1000

# 2. Verify approach (20 minutes)
python train_query_mutator.py --frac-samples 0.25 --num-env-steps 5000

# 3. Full training (4 hours)
python train_query_mutator.py --frac-samples 1.0 --num-env-steps 50000
```

### Workflow 2: Hyperparameter Sweep
```bash
# Test configurations on 20% of data
for lr in 1e-4 3e-4 5e-4; do
  for entropy in 0.005 0.01 0.02; do
    python train_query_mutator.py \
      --lr $lr \
      --entropy-coef $entropy \
      --frac-samples 0.2 \
      --num-env-steps 3000 \
      --save-dir sweep_lr${lr}_ent${entropy}
  done
done

# Train best config on full data
python train_query_mutator.py --lr 3e-4 --entropy-coef 0.01 --frac-samples 1.0
```

### Workflow 3: Model Selection
```bash
# Compare models on 30% of data
for model in qwen2.5:7b llama3.1:8b gemma2:9b; do
    python train_query_mutator.py \
        --mutator-model $model \
        --frac-samples 0.3 \
        --num-env-steps 5000 \
        --save-dir models_$model
done
```

## Summary

The `--frac-samples` parameter is a powerful tool for:
- ✅ Faster iteration during development
- ✅ Efficient hyperparameter tuning
- ✅ Resource management on limited hardware
- ✅ Quick validation of code changes
- ✅ Reproducible experiments with --seed

Use it to speed up your development workflow, then scale to full dataset for production training!
