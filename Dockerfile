FROM public.ecr.aws/docker/library/python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY src/convexlance ./convexlance

ENTRYPOINT ["python", "-m", "convexlance.cli"]
