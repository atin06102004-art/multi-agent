FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# uv/uvx is required at runtime: mcp_client.py launches the AviationStack
# MCP server via `uvx --with mcp<2 aviationstack-mcp`. Without this, the
# flight agent fails on every request.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Streamlit is the primary UI (streamlit_app.py talks to backend.py
# directly). app.py's JSON API is the alternative surface for
# programmatic/HTTP clients — run it instead with:
#   docker run -p 8000:8000 <image> uvicorn app:app --host 0.0.0.0 --port 8000
EXPOSE 8501

# Azure App Service for Containers routes traffic to the port set in the
# WEBSITES_PORT app setting; keep this in sync with that value (8501).
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]