FROM python:3.12

# All application files live under /app inside the container.
WORKDIR /app


RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install application dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, evaluation scripts, and data.
COPY src/ ./src/
COPY evaluation/ ./evaluation/
COPY data/ ./data/

# Keep the separate DeepEval requirements available
# without installing them in the application environment.
COPY requirements-deepeval.txt .

# Allow imports such as `from src.retrieval import ...`.
ENV PYTHONPATH=/app

# Start the command-line RAG application by default.
CMD ["python", "-m", "src.main"]