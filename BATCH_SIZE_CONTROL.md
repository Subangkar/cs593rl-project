# Controlling Batch Size

## Overview

The batch size controls how many concurrent API calls are made to Ollama during training. This affects both speed and resource usage.

## Command-Line Usage

### Set Batch Size

Use the `--batch-size` argument to control concurrent API calls:

```bash
# Use 16 concurrent API calls (faster if your server can handle it)
python train_query_mutator.py --batch-size 16

# Use 4 concurrent API calls (more conservative)
python train_query_mutator.py --batch-size 4

# Default is 8
python train_query_mutator.py  # Uses batch_size=8
```

### Complete Example

```bash
python train_query_mutator.py \
    --num-processes 16 \
    --batch-size 8 \
    --num-env-steps 10000 \
    --use-batching
```

## How It Works

### What is Batch Size?

The batch size determines how many API calls are processed **concurrently** (in parallel) at the same time:

- **Batch size = 1**: Sequential processing (one at a time)
- **Batch size = 4**: Process 4 API calls simultaneously
- **Batch size = 8**: Process 8 API calls simultaneously (default)
- **Batch size = 16**: Process 16 API calls simultaneously

### Relationship with `--num-processes`

These are two different concepts:

- **`--num-processes`**: How many RL environments to run in parallel (affects rollout collection)
- **`--batch-size`**: How many concurrent API calls within each step

Example with `--num-processes 16` and `--batch-size 8`:
- You have 16 environments
- Each step requires 16 mutations + 16 target queries = 32 API calls
- With batch_size=8, these 32 calls are processed in 4 batches of 8
- Time per step: ~4 × (average API latency)

## Choosing the Right Batch Size

### Considerations

1. **Ollama Server Capacity**: Can your server handle concurrent requests?
2. **RAM/VRAM**: Larger batches need more memory
3. **API Response Time**: Faster models benefit more from larger batches
4. **Number of Processes**: Match or divide evenly into `--num-processes`

### Recommended Settings

| Server Setup | Batch Size | Notes |
|-------------|-----------|-------|
| Local laptop, 8GB RAM | 2-4 | Conservative |
| Desktop, 16GB RAM | 4-8 | Default/balanced |
| Server, 32GB+ RAM | 8-16 | Aggressive |
| High-end GPU server | 16-32 | Maximum speed |

### Rule of Thumb

```
batch_size ≤ num_processes ≤ 2 × batch_size
```

This ensures efficient batching without too many small batches.

## Examples

### Conservative (Slower but Safer)

```bash
python train_query_mutator.py \
    --num-processes 8 \
    --batch-size 4
```

- 8 environments
- Process 4 API calls at a time
- 2 batches per step

### Balanced (Default)

```bash
python train_query_mutator.py \
    --num-processes 16 \
    --batch-size 8
```

- 16 environments
- Process 8 API calls at a time
- 2 batches per step

### Aggressive (Faster if Hardware Permits)

```bash
python train_query_mutator.py \
    --num-processes 32 \
    --batch-size 16
```

- 32 environments
- Process 16 API calls at a time
- 2 batches per step

## Disabling Batching

If you encounter issues or want sequential processing:

```bash
python train_query_mutator.py --no-batching
```

This ignores `--batch-size` and processes everything sequentially.

## Performance Testing

Test different batch sizes with the benchmark script:

```bash
# Test with batch_size=4
python benchmark_batching.py

# Edit NUM_QUERIES in the script to test larger batches
```

Or time actual training runs:

```bash
# Test batch_size=4
time python train_query_mutator.py --num-processes 8 --batch-size 4 --num-env-steps 500

# Test batch_size=8
time python train_query_mutator.py --num-processes 8 --batch-size 8 --num-env-steps 500

# Test batch_size=16
time python train_query_mutator.py --num-processes 8 --batch-size 16 --num-env-steps 500
```

Compare wall-clock times to find the optimal setting for your hardware.

## Troubleshooting

### "Connection refused" or timeout errors

**Problem**: Batch size too high for your Ollama server

**Solution**: Reduce batch size
```bash
python train_query_mutator.py --batch-size 2
```

### "Out of memory" errors

**Problem**: Too many concurrent model loads

**Solution**: Reduce batch size and/or num-processes
```bash
python train_query_mutator.py --num-processes 8 --batch-size 4
```

### Ollama server configuration

Increase Ollama's concurrent request limit:
```bash
# Set environment variable before starting Ollama
export OLLAMA_NUM_PARALLEL=16
ollama serve
```

### Still too slow?

1. **Increase batch size** if hardware allows:
   ```bash
   python train_query_mutator.py --batch-size 16
   ```

2. **Use faster models** for mutations:
   ```bash
   python train_query_mutator.py --mutator-model qwen2.5:3b
   ```

3. **Reduce max tokens** (edit ollama_utils.py):
   - Lower `max_tokens` in mutation/query functions

## Summary

- **Default batch size**: 8 concurrent API calls
- **Adjust with**: `--batch-size N`
- **Disable batching**: `--no-batching`
- **Optimal range**: 4-16 depending on hardware
- **Test with**: `benchmark_batching.py` or timing training runs
