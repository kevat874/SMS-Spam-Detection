#!/bin/bash
# Quick launcher for the Streamlit app
cd "$(dirname "$0")"
streamlit run app.py --server.headless true
