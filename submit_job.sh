#!/bin/bash
#SBATCH --job-name=train_32      # train for 32 steps
#SBATCH --output=slurm-%j.out     # Standard output log
#SBATCH --error=slurm-%j.err      # Error log
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1         # Request 1 GPU
#SBATCH --cpus-per-task=8         # 8 CPU cores for data processing
#SBATCH --mem=48G                 # 48 GB RAM
#SBATCH --time=01:00:00           # 1 Hour
#SBATCH --partition=scholar-j       # Request GPU partition
#SBATCH --account=gpu         # sacctmgr show association user=arko format=Account
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=arko@purdue.edu

# 1. Load Environment
source $CLUSTER_SCRATCH/anaconda3/bin/activate
# source activate my_env  # Uncomment and add your env name if you use one

# 2. Set Environment Variables
# Point Ollama to the Scratch directory where we downloaded models in Phase 1
export OLLAMA_MODELS="$CLUSTER_SCRATCH/.ollama/models"
export OLLAMA_HOST="127.0.0.1:11435"

# Print debug info
echo "Job started on $(hostname) at $(date)"
echo "Scratch directory: $CLUSTER_SCRATCH"

# 3. Start Ollama Server in Background with memory-optimized settings
cd $CLUSTER_SCRATCH
echo "Starting Ollama server..."
# Set memory management to prevent buffer failures while keeping models loaded
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_MAX_QUEUE=4
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_KEEP_ALIVE="48h"
./ollama/bin/ollama serve > ollama_runtime.log 2>&1 &
OLLAMA_PID=$!

# 4. Wait for Ollama to Initialize
# We try to connect to localhost:11434 until it succeeds
echo "Waiting for Ollama..."
MAX_RETRIES=20
COUNT=0
until curl -s $OLLAMA_HOST > /dev/null; do
    sleep 5
    COUNT=$((COUNT+1))
    if [ $COUNT -ge $MAX_RETRIES ]; then
        echo "Error: Ollama failed to start."
        exit 1
    fi
done
echo "Ollama is active!"

# 5. Run the Python Training Script
cd cs593rl-project

echo "Starting training..."
python train_query_mutator.py \
    --target-model llava:latest \
    --judge-model deepseek-r1:14b \
    --use-llm-judge \
    --num-processes 1 \
    --batch-size 4 \
    --frac-samples 0.5 \
    --num-env-steps 32 \
    --save-images

# 6. Cleanup
echo "Training finished. Killing Ollama server..."
kill $OLLAMA_PID