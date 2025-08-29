#!/bin/bash

set -e # Exit on any error

echo "🚀 AI Research Copilot - Local Setup"
echo "===================================="

# Check if Python 3.11+ is available
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1-2)
echo "📍 Python version: $python_version"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
  echo "❌ Python 3.11+ required. Current version: $python_version"
  exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
  echo "📦 Creating virtual environment..."
  python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing requirements..."
pip install -r requirements.txt

# Check for .env file
if [ ! -f ".env" ]; then
  echo "⚠️  .env file not found. Please copy .env.example to .env and configure:"
  echo "   cp .env.example .env"
  echo "   # Edit .env with your API keys"
  read -p "Press Enter to continue after setting up .env file..."
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data vectordb uploads

# Check if sample files exist
if [ "$(ls -A data/ 2>/dev/null)" ]; then
  echo "📄 Sample files found in data/. Starting ingestion..."
  python -m app.ingest data/*
else
  echo "📄 No sample files in data/. You can:"
  echo "   - Add PDF/TXT files to the data/ directory"
  echo "   - Use the web interface to upload files"
  echo "   - Run: python -m app.ingest path/to/your/file.pdf"
fi

# Function to check if port is available
check_port() {
  if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1; then
    return 1
  else
    return 0
  fi
}

# Start API server in background
if check_port 8000; then
  echo "🖥️  Starting API server on port 8000..."
  python -m uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload &
  API_PID=$!
  echo "   API PID: $API_PID"
  sleep 3 # Wait for server to start
else
  echo "⚠️  Port 8000 is busy. Please stop the existing service or use a different port."
fi

# Test API health
echo "🔍 Testing API health..."
if curl -s http://localhost:8000/health >/dev/null; then
  echo "✅ API server is healthy"
else
  echo "❌ API server health check failed"
  if [ ! -z "$API_PID" ]; then
    kill $API_PID 2>/dev/null || true
  fi
  exit 1
fi

# Start Streamlit UI
if check_port 8501; then
  echo "🎨 Starting Streamlit UI on port 8501..."
  export API_BASE_URL=http://localhost:8000
  streamlit run ui/streamlit_app.py --server.port 8501 &
  UI_PID=$!
  echo "   UI PID: $UI_PID"
else
  echo "⚠️  Port 8501 is busy. Please stop the existing service or use a different port."
fi

# Wait a moment for services to start
sleep 5

echo ""
echo "🎉 Setup complete!"
echo "==================="
echo "📊 API Server: http://localhost:8000"
echo "📖 API Docs:   http://localhost:8000/docs"
echo "🎨 Web UI:     http://localhost:8501"
echo ""
echo "📝 Quick test commands:"
echo "   # Upload file:"
echo "   curl -X POST \"http://localhost:8000/ingest\" -H \"Authorization: Bearer dev-api-key\" -F \"file=@data/sample.pdf\""
echo ""
echo "   # Query:"
echo "   curl -X POST \"http://localhost:8000/query\" -H \"Authorization: Bearer dev-api-key\" -H \"Content-Type: application/json\" -d '{\"question\": \"What is this document about?\"}'"
echo ""
echo "   # Stop services:"
echo "   kill $API_PID $UI_PID"
echo ""
echo "🎯 Next steps:"
echo "   1. Open http://localhost:8501 in your browser"
echo "   2. Upload some research papers"
echo "   3. Ask questions about your documents"
echo ""

# Keep script running and handle cleanup
cleanup() {
  echo ""
  echo "🛑 Shutting down services..."
  if [ ! -z "$API_PID" ]; then
    kill $API_PID 2>/dev/null || true
    echo "   Stopped API server"
  fi
  if [ ! -z "$UI_PID" ]; then
    kill $UI_PID 2>/dev/null || true
    echo "   Stopped UI server"
  fi
  echo "👋 Goodbye!"
  exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for user input to stop
echo "Press Ctrl+C to stop all services..."
wait
