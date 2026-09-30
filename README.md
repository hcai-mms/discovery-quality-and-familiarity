
# Music Discovery Quality and the Value of Familiarity
RecSys '26: Proceedings of the 20th ACM Conference on Recommender Systems

https://doi.org/10.1145/3773078.3841296
### Abstract
 > Arguably, driving music discovery is one of the most important functions of recommender systems; however, defining what characterizes discovery and determining what constitutes a successful one are both nontrivial problems. Here, we argue that repetition offers a valuable lens: when users discover a track they like, they typically listen to it repeatedly several times. In this work, we use this pattern as an indicator to characterise the quality of successful discoveries. We also argue that familiarity is a fundamental part of the discovery process. We therefore use different familiarity measures to construct user profiles from long-term consumption logs and assess their ability to predict discovery success. Our experiments on listening data from a well-established music streaming platform show that familiarity provides a useful signal for discovery prediction, with performance improving further when user and item information is incorporated.

### Citation
```bib
@inproceedings{escobedo2026discoveryquality,
author = {Escobedo, Gustavo and Bonnin, Geoffray and Schedl, Markus and Sguerra, Bruno},
title = {Music Discovery Quality and the Value of Familiarity},
year = {2026},
isbn = {9798400722844},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3773078.3841296},
doi = {10.1145/3773078.3841296},
booktitle = {Proceedings of the 20th ACM Conference on Recommender Systems},
pages = {1731–1733},
numpages = {3},
keywords = {Discovery Modeling, Discovery Quality, Familiarity, Recommender Systems}
}
```

## Usage
Run the training script using the default configuration (see `config_full.yaml`):
 
```bash
python main.py \
  --config config_full.yaml
```

### Custom Configuration

To customize the training process, specify additional command-line arguments:

```bash
python main.py \
  --config config_full.yaml \
  --train-path /data/train.parquet \
  --val-path /data/val.parquet \
  --output-dir runs/discovery_net \
  --n-epochs 10 \
  --batch-size 1024 \
  --lr 0.001 \
  --embedding-lr 0.0001 \
  --weight-decay 0.00001 \
  --device cuda \
  --user-embedding-path embeddings/user_embeddings.npy \
  --item-embedding-path embeddings/item_embeddings.npy \
  --freeze-user-embeddings \
  --freeze-item-embeddings
```

Remove optional arguments that are not needed for your experiment.

## Command-Line Arguments

| Argument | Description |
|---|---|
| `--config` | Path to the YAML configuration file. |
| `--train-path` | Path to the training dataset in Parquet format. |
| `--val-path` | Path to the validation dataset in Parquet format. |
| `--output-dir` | Directory for saving training outputs. |
| `--n-epochs` | Number of training epochs. |
| `--batch-size` | Training batch size. |
| `--lr` | Learning rate for the main model. |
| `--embedding-lr` | Learning rate for embeddings. |
| `--weight-decay` | Weight decay coefficient. |
| `--device` | Training device, such as `cuda` or `cpu`. |
| `--user-embedding-path` | Path to the user embeddings in NumPy format. |
| `--item-embedding-path` | Path to the item embeddings in NumPy format. |
| `--freeze-user-embeddings` | Freeze user embeddings during training. |
| `--freeze-item-embeddings` | Freeze item embeddings during training. |

