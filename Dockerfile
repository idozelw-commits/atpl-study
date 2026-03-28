FROM python:3.11-slim

WORKDIR /app

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download ChromaDB's ONNX embedding model so first query is fast
RUN python -c "import chromadb; c = chromadb.Client(); col = c.get_or_create_collection('warmup'); col.add(ids=['1'], documents=['warmup']); print('Model cached')"

# Copy app
COPY . .

# Create data directory
RUN mkdir -p data/uploads

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
