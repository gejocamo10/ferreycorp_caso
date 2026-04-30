# Imagen del contenedor agnostica del proveedor cloud.
# Corre en DigitalOcean App Platform, AWS ECS o App Runner, GCP Cloud Run
# o cualquier cluster de Kubernetes sin cambios.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# build-essential y libgomp1 son necesarios para compilar e invocar LightGBM.
# curl se usa en health checks de ECS/App Runner.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/
COPY docs/ ./docs/

EXPOSE 8501

# Por defecto arranca la UI de Streamlit. Para correr trabajos batch
# (entrenar, scorear, etc.) se sobrescribe el CMD al lanzar el contenedor.
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
