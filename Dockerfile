FROM python:3.11-slim 

WORKDIR /app

# Install inference dependencies
COPY inference/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app code
COPY inference/ /app/inference/

# Copy registry + production artifacts (needed at runtime)
COPY model_registry/ /app/model_registry/
COPY artifacts/production/ /app/artifacts/production/

# Default envs (can override in Cloud Run)
ENV REGISTRY_PATH=/app/model_registry/metadata.json
ENV PORT=8080

EXPOSE 8080

CMD ["uvicorn", "inference.main:app", "--host=0.0.0.0", "--port=8080"]


# This is Cloud Run compatible (listens on 8080 and respects PORT convention; we still set PORT=8080 as default).