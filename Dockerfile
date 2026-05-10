# Usamos a imagem slim para manter o tamanho reduzido na VPS
FROM python:3.12-slim

# Instala dependências do sistema para áudio (librosa/whisper)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia o requirements.txt primeiro para aproveitar o cache do Docker
COPY requirements.txt .

# Instala as dependências (incluindo o Torch para CPU conforme configuramos)
RUN pip install --no-cache-dir -r requirements.txt

# Copia o resto do código e as pastas dos modelos
# Certifique-se de que as pastas dos modelos estão no mesmo nível do Dockerfile
COPY . .

# Expõe a porta que o FastAPI vai rodar
EXPOSE 8000

# Comando para rodar com Gunicorn para suportar os 5 usuários de forma estável
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--timeout", "300", "--bind", "0.0.0.0:8000", "main:app"]