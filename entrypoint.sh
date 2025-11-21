#!/bin/sh
# Entrypoint for Cloud Run — starts Streamlit on the port provided by the platform
PORT="${PORT:-8080}"

echo "Starting Streamlit on port ${PORT}..."
exec streamlit run app.py --server.port ${PORT} --server.address 0.0.0.0 --server.enableCORS false
