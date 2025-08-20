#!/bin/bash
set -e

echo "Setting up environment..."

python3 -m venv .venv
source .venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Running training script..."
python src/model_training.py

echo "Pipeline completed."
