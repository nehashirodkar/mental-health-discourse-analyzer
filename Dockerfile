# Container for HuggingFace Spaces (Docker SDK).
# Runs FastAPI on internal :8000 and Streamlit on the public :7860.
# Models are pulled from HF Hub at first request (see src/config.py).

FROM python:3.11-slim

# Build deps for hdbscan / numba (compiled C extensions).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs containers as a non-root user (uid 1000); set up a
# matching home dir so HF/torch caches don't try to write to root-only paths.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/user/.cache/huggingface \
    TORCH_HOME=/home/user/.cache/torch \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

WORKDIR /app

# Install Python deps first so this layer caches across code edits.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the source.
COPY --chown=user . .

RUN chmod +x start.sh

USER user
EXPOSE 7860

CMD ["./start.sh"]
