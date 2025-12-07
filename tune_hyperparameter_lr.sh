# Test grid (3x3x3 = 27 runs, ~1-2 hours total with reduced steps)
for lr in 0.0003 0.0005 0.001; do
      CUDA_VISIBLE_DEVICES=0,3 python train_query_mutator.py \
        --target-model gemma3:4b \
        --judge-model deepseek-r1:14b \
        --use-llm-judge \
        --num-processes 2 \
        --num-env-steps 1000 \
        --lr $lr \
        --num-steps 32 \
        --ppo-epoch 2 \
        --batch-size 128
done