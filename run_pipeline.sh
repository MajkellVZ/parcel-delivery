#!/bin/bash
set -e

echo "Setting up environment..."

sudo apt install python3.12-venv -y

python3 -m venv .venv
source .venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Running training script..."
python src/train_model.py

echo "Pipeline completed."
