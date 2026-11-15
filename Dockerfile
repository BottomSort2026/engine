FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image, at the revision corpus/manifest.json resolves to, so start-up
# never touches the network. Only the files embedder.load() needs are copied here, so code changes
# don't re-download the model.
ENV HF_HOME=/opt/huggingface
COPY app/__init__.py app/embedder.py app/embedding.py app/
COPY corpus/manifest.json corpus/
RUN python -c "from app import embedder; embedder.load()"
ENV HF_HUB_OFFLINE=1

# The code and the FLEX models in models/ (baked into the image: no volumes).
COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
