CUDA_VISIBLE_DEVICES=1 python test_query_mutator.py \
  --model-path logs/run_20251206_223829/checkpoints/final_model.pt \
  --target-model llava:latest \
  --mutator-model gemma3:latest \
  --judge-model deepseek-r1:14b \
  --dataset /home/makil/cs593rl-project/dataset/prompts_harmful_responses_test.csv \
  --output-file test_results_rl_llava_target.json \