# Test grid (3x3x3 = 27 runs, ~1-2 hours total with reduced steps)
for entropy in 0.01 0.03 0.05; do
  CUDA_VISIBLE_DEVICES=1,2,3 python train_query_mutator.py \
    --target-model gemma3:4b \
    --judge-model deepseek-r1:14b \
    --use-llm-judge \
    --num-processes 3 \
    --num-env-steps 1000 \
    --entropy-coef $entropy \
    --num-steps 32 \
    --ppo-epoch 2 \
    --batch-size 128
done
