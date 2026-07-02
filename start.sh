#!/bin/bash
# Spin up FastAPI on port 8000
python -m uvicorn app:app --host 0.0.0.0 --port 8000 &

# Spin up Streamlit UI on port 8501
python -m streamlit run frontend.py --server.port 8501 --server.address 0.0.0.0