FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy the editable path dependency first so uv can resolve it during install.
# Build context must be the repo root for this COPY to work.
COPY ml-pipeline/ ./ml-pipeline/

# Copy backend source and install all deps.
COPY backend/ ./backend/
WORKDIR /app/backend
RUN uv sync --no-dev --frozen

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
