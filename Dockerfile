# Dashboard server image for AWS App Runner (or any container runtime).
# The server needs only the deterministic stack + boto3; the Strands/AgentCore
# packages are deployment-time dependencies of the AgentCore app, not of this
# server, so they stay out of the image.
FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir "pandas>=2.0" "numpy>=1.24" "scipy>=1.10" "boto3>=1.34"

COPY src/ src/
COPY scripts/serve_dashboard.py scripts/build_dashboard.py scripts/
COPY dashboard/ dashboard/
COPY data/raw/ data/raw/
COPY outputs/2024_spanish/ outputs/2024_spanish/

ENV HOST=0.0.0.0 \
    PORT=8080 \
    PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["python", "scripts/serve_dashboard.py"]
