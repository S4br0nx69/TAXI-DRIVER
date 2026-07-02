FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-tk && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-mlops.txt ./
RUN pip install --no-cache-dir -r requirements-mlops.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg
ENV GIT_PYTHON_REFRESH=quiet

ENTRYPOINT ["python3"]