# API image. CPU torch is installed before this package so pip does not
# pull a CUDA wheel. The install is editable: PROJECT_ROOT is parents[3]
# of src/rag/core/config.py, which stays /app. A copy into site-packages
# would point that root at the wrong directory.
#
# Secrets are not copied. Compose injects .env at run time.
# The process stays root so a Windows bind mount of ./data is writable.

FROM python:3.12-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache \
    HUGGINGFACE_HUB_CACHE=/cache \
    SENTENCE_TRANSFORMERS_HOME=/cache

COPY pyproject.toml IMPLEMENTATION_PLAN.md ./
COPY src ./src
COPY configs ./configs
COPY data/raw ./data/raw

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e . --upgrade-strategy only-if-needed

EXPOSE 8000

CMD ["python", "-m", "rag.serving"]
