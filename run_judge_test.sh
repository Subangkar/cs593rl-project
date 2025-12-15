#!/bin/bash
#SBATCH --job-name=test_judge
#SBATCH --output=slurm_judge_test-%j.out
#SBATCH --error=slurm_judge_test-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --time=02:00:00
#SBATCH --partition=a100-80gb
#SBATCH --account=dgoldwas

# 1. Load Environment
source $CLUSTER_SCRATCH/anaconda3/bin/activate

# 2. Set Environment Variables
export OLLAMA_MODELS="$CLUSTER_SCRATCH/.ollama/models"
export OLLAMA_HOST="127.0.0.1:11435"

echo "=================================="
echo "Testing Improved Judge on Jailbreak Samples"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "Scratch directory: $CLUSTER_SCRATCH"
echo "=================================="

# 3. Start Ollama Server in Background
cd $CLUSTER_SCRATCH
echo "Starting Ollama server..."
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_MAX_QUEUE=4
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_KEEP_ALIVE="48h"
./ollama/bin/ollama serve > ollama_judge_test.log 2>&1 &
OLLAMA_PID=$!

# 4. Wait for Ollama to Initialize
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

# 5. Run the test script
cd cs593rl-project
echo "Starting judge test..."
python test_judge_on_jailbreaks.py --limit 100 --judge deepseek-r1:14b

# 6. Cleanup
echo "=================================="
echo "Test Complete"
echo "End Time: $(date)"
echo "=================================="
echo "Stopping Ollama server..."
kill $OLLAMA_PID
