#!/usr/bin/env bash

set -e

python main.py \
  --config config_full.yaml \
  # --train-path /data/train.parquet \
  # --val-path /data/val.parquet \
  # --output-dir runs/discovery_net \
  # --n-epochs 10 \
  # --batch-size 1024 \ 
  # --lr 0.001 \
  # --embedding-lr 0.0001 \
  # --weight-decay 0.00001 \
  # --device cuda \
  # --user-embedding-path embeddings/user_embeddings.npy \
  # --item-embedding-path embeddings/item_embeddings.npy \
  # --freeze-user-embeddings \
  # --freeze-item-embeddings



