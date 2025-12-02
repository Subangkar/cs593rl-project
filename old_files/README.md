# Old Files Archive

This directory contains files that are **not used** in the current training/testing pipeline.

## Files Moved Here

### Legacy Files (Superseded by rl_core/ package)
- `policy_network.py` - Use `rl_core/policy_network.py` instead
- `ppo_algorithm.py` - Use `rl_core/ppo_algorithm.py` instead
- `rollout_storage.py` - Use `rl_core/rollout_storage.py` instead

### Experimental/Alternative Implementations
- `pipeline.py` - Alternative pipeline implementation
- `target.py` - Alternative target model utilities
- `TextSimilaryReward.py` - Alternative text similarity scoring
- `ollama_api.py` - Alternative API interface
- `demo_frac_samples.py` - Dataset sampling demo script

### Utilities (Integrated into Environment)
- `reward_utils.py` - Keyword-based rewards (integrated into `rl_query_mutator_env.py`)
- `reward_utils_llm_judge.py` - LLM judge rewards (integrated into `rl_query_mutator_env.py`)

### Backward Compatibility
- `ollama_utils.py` - Wrapper providing backward compatibility (uses `OllamaClient` internally)

### Benchmarking
- `benchmark_batching.py` - Performance benchmarking tool

## Why These Files Are Here

These files were moved to keep the root directory clean and focused on the essential training/testing pipeline. They are:
- **Not imported** by `train_query_mutator.py` or `test_query_mutator.py`
- **Not required** for the core RL training loop
- Either superseded by newer implementations or serve auxiliary purposes

## Can I Delete These?

**Not recommended** - These files may be useful for:
- Historical reference
- Alternative approaches
- Backward compatibility with old code
- Benchmarking and experimentation

If disk space is not a concern, keep them archived here.

---

**Date archived:** December 2, 2025
