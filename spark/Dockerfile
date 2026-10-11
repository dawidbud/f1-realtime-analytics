FROM apache/spark:3.5.9-python3

# UID/GID uzytkownika na hoscie - pliki zapisywane do ./data (bind mount)
# beda nalezaly do niego, a nie do roota / uid 185.
ARG SPARK_UID=1000
ARG SPARK_GID=1000

USER root

# numpy jest wymagany przez pyspark.ml
RUN pip install --no-cache-dir numpy==2.2.6

# Uzytkownik spark z UID hosta i prawdziwym katalogiem domowym
# (potrzebnym m.in. dla cache Ivy przy spark-submit --packages).
RUN groupmod -g ${SPARK_GID} spark \
    && usermod -u ${SPARK_UID} -g ${SPARK_GID} -d /home/spark spark \
    && mkdir -p /home/spark \
    && chown -R spark:spark /home/spark /opt/spark

ENV HOME=/home/spark
ENV PATH="/opt/spark/bin:${PATH}"

USER spark
