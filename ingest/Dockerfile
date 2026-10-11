FROM python:3.11-slim

ARG UID=1000
ARG GID=1000

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ingest_fastf1.py .

# Pliki w ./data maja nalezec do uzytkownika hosta.
RUN groupadd -g ${GID} app && useradd -u ${UID} -g ${GID} -m app
USER app

ENTRYPOINT ["python", "ingest_fastf1.py"]
