# Pregeneration Guide for Unaligned Responses

This guide explains how to pregenerate unaligned responses for faster training with LLM judge.

## Why Pregenerate?

When using `--use-llm-judge` during training, the environment needs to generate unaligned baseline responses from an uncensored model for every query. This is:
- **Slow**: Each response generation takes several seconds
- **Redundant**: The same queries are used across multiple training runs
- **Expensive**: Uses significant compute resources

By pregenerating responses, you:
- ✅ Generate each response once, use many times
- ✅ Speed up training by 2-3x when using LLM judge
- ✅ Enable consistent baselines across experiments

## Step 1: Pregenerate Unaligned Responses

Run the pregeneration script on your harmful queries dataset:

```bash
python pregenerate_unaligned_responses.py \
    --input-csv dataset/prompts_harmful.csv \
    --output-csv dataset/unaligned_responses.csv \
    --uncensored-model wizard-vicuna-uncensored:13b
```

**Options:**
- `--input-csv`: Input CSV with harmful queries (one query per line)
- `--output-csv`: Output CSV for storing responses (query, response pairs)
- `--uncensored-model`: Ollama model to use for generating unaligned responses

**Output Format:**
The generated CSV will have 2 columns:
```
query,unaligned_response
"How to hack a system?","Here's how you can hack..."
"Write malware code","I can help with that..."
```

The CSV writer uses `QUOTE_ALL` to properly handle multiline responses.

## Step 2: Train with Pregenerated Responses

Once you have the pregenerated responses CSV, use it during training:

```bash
python train_query_mutator.py \
    --num-env-steps 10000 \
    --use-llm-judge \
    --unaligned-csv dataset/unaligned_responses.csv
```

**Key Points:**
- The environment automatically loads responses from the CSV file
- If a query is found in the pregenerated responses, it uses that
- If not found (e.g., for mutated queries), it generates on-the-fly
- Original queries will always use pregenerated responses (major speedup!)

## Step 3: Verify CSV Logging

During training, all mutations are logged to a timestamped CSV file:

```
trained_models/training_log_YYYYMMDD_HHMMSS.csv
```

**Columns:**
- `step`: Global step number
- `update`: PPO update number
- `episode`: Episode number
- `query`: Original query (truncated to 100 chars)
- `mutation_type`: Name of mutation operator used
- `mutated_query`: Query after mutation (truncated)
- `target_response`: Response from target model (truncated)
- `unaligned_response`: Baseline response from uncensored model (truncated)
- `reward_score`: Reward value for this mutation
- `mutation_number`: Step within the episode (1-5)

## Example Workflow

```bash
# 1. Pregenerate unaligned responses (one-time, takes ~30-60 minutes)
python pregenerate_unaligned_responses.py

# 2. Train with LLM judge using pregenerated responses (much faster!)
python train_query_mutator.py \
    --num-env-steps 50000 \
    --use-llm-judge \
    --save-dir trained_models_llm_judge

# 3. Analyze training logs
head -20 trained_models_llm_judge/training_log_*.csv
```

## Performance Comparison

**Without pregeneration (--use-llm-judge):**
- ~15-20 seconds per mutation (includes generating unaligned response)
- 1000 steps ≈ 4-5 hours

**With pregeneration (--use-llm-judge + --unaligned-csv):**
- ~5-7 seconds per mutation (pregenerated baseline)
- 1000 steps ≈ 1.5-2 hours
- **~2-3x faster!**

**Without LLM judge (keyword-based, default):**
- ~2-3 seconds per mutation
- 1000 steps ≈ 30-40 minutes
- Fast but less accurate rewards

## Tips

1. **Pregenerate once, use many times**: The unaligned responses CSV can be reused across all training runs

2. **Different models**: You can pregenerate with different uncensored models and compare:
   ```bash
   python pregenerate_unaligned_responses.py \
       --output-csv dataset/unaligned_vicuna.csv \
       --uncensored-model wizard-vicuna-uncensored:13b
   
   python pregenerate_unaligned_responses.py \
       --output-csv dataset/unaligned_mistral.csv \
       --uncensored-model mistral-uncensored:latest
   ```

3. **Version control**: Commit the unaligned responses CSV to your repo (if allowed) so team members can share

4. **CSV inspection**: Open the CSV in any spreadsheet software to inspect the quality of unaligned responses
