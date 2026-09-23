FROM python:3.11-slim AS backend

WORKDIR /app

# libgl1/libglib2.0-0: required by opencv-python at import time even
# headless; ffmpeg: PyAV's runtime codec dependency for video decoding.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# CPU-only torch/torchvision for the video-ASD classifier (backend/app/ml/video_asd).
# Swap for a CUDA index URL at build time if deploying to a GPU host:
#   docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 .
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch torchvision --index-url "$TORCH_INDEX_URL"

COPY backend ./backend
COPY scripts ./scripts
COPY models ./models
COPY data ./data
COPY knowledge_base ./knowledge_base
COPY alembic.ini ./

# Bake in network-fetched model weights at build time rather than on
# first request -- keeps runtime free of outbound-internet dependence
# and avoids the first real user request paying (or failing) a
# multi-hundred-MB download.
RUN python -c "from backend.app.video.analyzer import _ensure_model_downloaded; _ensure_model_downloaded()" \
    && python -c "import torchvision; torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)"

ENV NEUROSCREEN_ENV=production \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["sh", "-c", "python -m alembic upgrade head && uvicorn backend.app.main:app --host 0.0.0.0 --port 8000"]
