#!/usr/bin/env bash

# Directory containing your PDFs
DATA_DIR="data"

# Loop over each PDF in data/
for file in "$DATA_DIR"/*.pdf; do
  echo "🚀 Ingesting $file …"
  
  # Submit ingestion request and capture task ID
  response=$(curl -s -X POST http://localhost:8000/ingest \
    -H "Authorization: Bearer dev-api-key" \
    -F "file=@${file}")
  
  task_id=$(echo "$response" | jq -r '.task_id')
  if [[ "$task_id" == "null" ]]; then
    echo "❌ Failed to start ingestion for $file"
    echo "Response: $response"
    exit 1
  fi
  echo "   ↳ Task ID: $task_id"

  # Poll ingestion status until completion or failure
  status=""
  until [[ "$status" == "completed" || "$status" == "failed" ]]; do
    sleep 3
    status=$(curl -s http://localhost:8000/ingest/status/$task_id \
      -H "Authorization: Bearer dev-api-key" \
      | jq -r '.status')
    progress=$(curl -s http://localhost:8000/ingest/status/$task_id \
      -H "Authorization: Bearer dev-api-key" \
      | jq -r '.progress // empty')
    echo "   • Status: $status${progress:+ ($progress%)}"
  done

  if [[ "$status" == "completed" ]]; then
    echo "✅ Successfully ingested $file"
  else
    echo "❌ Ingestion failed for $file"
    exit 1
  fi

done

echo "🎉 All PDFs ingested. Now rebuilding FAISS index."
# Verify index stats
python - << 'EOF'
from app.vectorstore import vector_store
print("📊 Vector store stats:", vector_store.get_stats())
EOF

