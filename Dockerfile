# Hugging Face Spaces: Docker image for CLV Predictor (Flask + Gunicorn)
# Port 7860 is required by Hugging Face Spaces.

FROM python:3.11-slim

# Create non-root user (UID 1000 required by Hugging Face Spaces)
RUN useradd -m -u 1000 user

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Install dependencies first (better layer caching)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code (models, outputs, templates, static are included)
COPY --chown=user . .

# Hugging Face Spaces expect the app on port 7860
ENV PORT=7860
EXPOSE 7860

# Run with Gunicorn (production WSGI server)
CMD gunicorn --bind 0.0.0.0:7860 --workers 1 --threads 2 --timeout 120 app:app
