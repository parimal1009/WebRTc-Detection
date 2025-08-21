#!/bin/bash

# Ultra-Advanced WebRTC Object Detection Server Startup Script

echo "Starting Ultra-Advanced WebRTC Object Detection Server..."

# Activate Python environment (optional - adjust path as needed)
if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
fi

# Export environment variables if set externally (optional)
# export GROQ_API_KEY="your_groq_api_key"
# export HF_TOKEN="your_huggingface_token"

# Run the FastAPI server on port 8000 with one worker
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --log-level info

echo "Server stopped."
