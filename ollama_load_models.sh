# 1. Navigate to Scratch
cd $CLUSTER_SCRATCH

# 2. Set the models directory to Scratch (Crucial to save Home quota)
export OLLAMA_MODELS="$CLUSTER_SCRATCH/.ollama/models"

# 3. Start a temporary background Ollama server with correct models directory
echo "Starting temporary server for downloading..."
OLLAMA_MODELS="$CLUSTER_SCRATCH/.ollama/models" ./ollama/bin/ollama serve > download_log.txt 2>&1 &
SERVER_PID=$!

# 4. Wait 10 seconds for the server to initialize
sleep 10

# 5. Pull all required models
echo "Pulling llama3.1:8b (Target)..."
OLLAMA_MODELS="$CLUSTER_SCRATCH/.ollama/models" ./ollama/bin/ollama pull llama3.1:8b

echo "Pulling qwen2.5:7b (Mutator)..."
OLLAMA_MODELS="$CLUSTER_SCRATCH/.ollama/models" ./ollama/bin/ollama pull qwen2.5:7b

echo "Pulling nomic-embed-text (Embedding)..."
OLLAMA_MODELS="$CLUSTER_SCRATCH/.ollama/models" ./ollama/bin/ollama pull nomic-embed-text

echo "Pulling deepseek-r1:14b (Judge)..."
OLLAMA_MODELS="$CLUSTER_SCRATCH/.ollama/models" ./ollama/bin/ollama pull deepseek-r1:14b

echo "Pulling wizard-vicuna-uncensored (Uncensored Judge)..."
OLLAMA_MODELS="$CLUSTER_SCRATCH/.ollama/models" ./ollama/bin/ollama pull wizard-vicuna-uncensored

# 6. Kill the temporary server
echo "Downloads complete. Stopping server."
kill $SERVER_PID