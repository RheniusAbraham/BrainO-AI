#!/bin/bash
# Trains both models for three seeds. Usage: bash run_all.sh
cd "$(dirname "$0")"
for seed in 42 43 44; do
  python -I src/train.py --model cnn --seed $seed
  python -I src/train.py --model mobilenet --seed $seed
done
echo ALL_DONE
