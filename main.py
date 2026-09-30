import argparse
import copy
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Any
from copy import copy, deepcopy

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (
    accuracy_score,balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    confusion_matrix,
    classification_report
)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import Dataset, DataLoader
from utils import *
from dataset import *
from modules import ShallowDiscoveryNet

from index_mapping import load_mappings,load_stat_data
from logging_utils import *

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    grad_clip_norm: Optional[float] = None,
):
    model.train()

    total_loss = 0.0
    total_examples = 0

    for batch in dataloader:
        user_ids = batch["user_id"].to(device)
        item_ids = batch["item_id"].to(device)

        event_features = {
            key: value.to(device)
            for key, value in batch["event_features"].items()
        }

        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits = model(
            user_ids=user_ids,
            item_ids=item_ids,
            event_features=event_features,
        )
        # print(logits.shape,labels.shape)
        labels = labels.float()
        loss = criterion(logits, labels)
        loss.backward()

        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / total_examples



@torch.no_grad()
def predict_with_ids(
    model,
    dataloader,
    device,
    threshold=0.5,
):
    """
    Generate predictions and return user_id and item_id together with predictions.
    """

    model.eval()

    all_user_ids = []
    all_item_ids = []
    all_probs = []

    for batch in dataloader:
        user_ids = batch["user_id"].to(device)
        item_ids = batch["item_id"].to(device)

        event_features = {
            key: value.to(device)
            for key, value in batch["event_features"].items()
        }

        logits = model(
            user_ids=user_ids,
            item_ids=item_ids,
            event_features=event_features,
        )

        logits = logits.view(-1)
        probs = torch.sigmoid(logits)

        all_user_ids.append(user_ids.detach().cpu())
        all_item_ids.append(item_ids.detach().cpu())
        all_probs.append(probs.detach().cpu())

    user_ids = torch.cat(all_user_ids)
    item_ids = torch.cat(all_item_ids)
    probs = torch.cat(all_probs)
    preds = (probs >= threshold).long()

    return {
        "user_id": user_ids,
        "item_id": item_ids,
        "probs": probs,
        "preds": preds,
    }

@torch.no_grad()
def test_model( model, config, device, threshold=0.5):
    data_cfg = config["data"]
    
    training_cfg = config["training"]

    test_df = load_dataframe(
        path=data_cfg['dataset_dir']+data_cfg["test_path"],
        file_format=data_cfg.get("file_format"),
    )

    event_feature_sizes = data_cfg["event_features"]
    event_feature_names = list(event_feature_sizes.keys())

    test_rows= dataframe_to_rows(test_df)

    test_dataset = DiscoveryDataset(
        rows=test_rows,
        num_classes=config["model"]["target_classes"],
        event_feature_names=event_feature_names,
        label_col=data_cfg.get("label_col", "label"),
        user_col=data_cfg.get("user_col", "user_id"),
        item_col=data_cfg.get("item_col", "item_id"),
    ) 
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=training_cfg.get("batch_size", 1024),
        shuffle=False,
        num_workers=training_cfg.get("num_workers", 0),
        pin_memory=device == "cuda",
    )
    if model is None:
        checkpoint_path= config['output']['model_path']
      
        model,checkpoint = load_model(
            ShallowDiscoveryNet,
            checkpoint_path, 
            device,

        
        ) 
    criterion= nn.BCEWithLogitsLoss()
    criterion= nn.CrossEntropyLoss(weight=torch.tensor([1,4,15,18]).to(device))
    criterion  = nn.CrossEntropyLoss()

    output_dir= Path(config['output']['output_dir'])
    predictions = evaluate(model, test_dataloader, criterion, device, threshold=threshold)
    save_metrics(predictions["metrics"], output_dir, "test_metrics.json")
   

    return predictions

@torch.no_grad()
def evaluate(
    model,
    dataloader,
    criterion,
    device,
    threshold=0.5,
):
    """
    Evaluate a binary classification model.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.

    dataloader : torch.utils.data.DataLoader
        Evaluation dataloader.

    criterion : torch loss
        Loss function, usually BCEWithLogitsLoss.

    device : torch.device
        Device used for inference.

    threshold : float
        Probability threshold used to convert probabilities into class predictions.

    Returns
    -------
    dict
        Dictionary containing loss, probabilities, labels, predictions, and metrics.
    """

    model.eval()

    total_loss = 0.0
    total_examples = 0

    all_probs = []
    all_preds = []
    all_labels = []
    all_user_ids=[]
    all_item_ids=[] 
    for batch in dataloader:
        all_item_ids.append(batch["item_id"])
        all_user_ids.append(batch["user_id"])
        user_ids = batch["user_id"].to(device)
        item_ids = batch["item_id"].to(device)

        event_features = {
            key: value.to(device)
            for key, value in batch["event_features"].items()
        }

        labels = batch["label"].to(device).float()

        logits = model(
            user_ids=user_ids,
            item_ids=item_ids,
            event_features=event_features,
        )

        # logits = logits.view(-1)
        # labels = labels.view(-1)
        
       
        loss = criterion(logits, labels)

        # probs = torch.argmax(torch.sigmoid(logits),dim=1)
        # labels = torch.argmax(labels,dim=1) 

        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_examples += batch_size

        all_probs.append(probs.detach().cpu())
        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())

    avg_loss = total_loss / total_examples

    y_proba = torch.cat(all_probs).numpy()
    y_pred = torch.cat(all_preds).numpy()
    
    y_true_onehot = np.concatenate(all_labels) 
    y_true = np.argmax(y_true_onehot, axis=1)
    # preds = (probs >= threshold).astype(int)
    classes = np.arange(y_proba.shape[1])  # [0, 1, 2, 3]

    # --- Core metrics ---
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    loss = log_loss(y_true, y_proba, labels=classes)

    roc_auc_macro = roc_auc_score(y_true_onehot, y_proba, multi_class="ovr", average="macro")
    roc_auc_weighted = roc_auc_score(y_true_onehot, y_proba, multi_class="ovr", average="weighted")

    pr_auc_macro = average_precision_score(y_true_onehot, y_proba, average="macro")
    pr_auc_weighted = average_precision_score(y_true_onehot, y_proba, average="weighted")

    # --- Per-class breakdown ---
    per_class_roc_auc = {}
    per_class_pr_auc = {}
    for i, cls in enumerate(classes):
        per_class_roc_auc[i] = roc_auc_score(y_true_onehot[:, i], y_proba[:, i])
        per_class_pr_auc[i] = average_precision_score(y_true_onehot[:, i], y_proba[:, i])

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    target_names =[str(c) for c in classes]
    report = classification_report(y_true, y_pred, labels=classes, target_names=target_names, zero_division=0)

    metrics = {
        "accuracy": acc,
        "balanced_accuracy":bacc,
        "loss":avg_loss,
        "log_loss": loss,
        "roc_auc_macro": roc_auc_macro,
        "roc_auc_weighted": roc_auc_weighted,
        "pr_auc_macro": pr_auc_macro,
        "pr_auc_weighted": pr_auc_weighted,
        "per_class_roc_auc": per_class_roc_auc,
        "per_class_pr_auc": per_class_pr_auc,
        "confusion_matrix": cm,
        "classification_report": report,
        # "y_true": y_true,
        # "y_true_onehot": y_true_onehot,
        # "y_pred": y_pred,
        # "y_proba": y_proba,
    }
    return {
        "loss": avg_loss,
        "probs": y_proba,
        "labels":y_true,
        "preds": y_pred,
        "metrics": metrics,
        "user_ids": torch.cat(all_user_ids).flatten().numpy(), 
        "items_ids":torch.cat(all_item_ids).flatten().numpy()
    }
    # return metrics
    # metrics = {
    #     "loss": avg_loss,
    #     "accuracy": accuracy_score(labels, preds),
    #     "precision": precision_score(labels, preds, zero_division=0,average=None),
    #     "recall": recall_score(labels, preds, zero_division=0,average=None),
    #     "f1": f1_score(labels, preds, zero_division=0,average=None),
    #     "confusion_matrix": confusion_matrix(labels, preds),
    # }
    # classes = np.arange(probs.shape[1]) 

    # # ROC AUC and PR AUC require both classes to be present.
    # if len(np.unique(labels)) == 2:
    #     metrics["roc_auc"] = roc_auc_score(labels, probs)
    #     metrics["pr_auc"] = average_precision_score(labels, probs)
    #     metrics["log_loss"] = log_loss(labels, probs)
    # else:
    #     metrics["roc_auc"] = roc_auc_score(labels, preds,multi_class='ovr',average='weighted')
    #     metrics["pr_auc"] = average_precision_score(labels, probs,average='weighted')
    #     metrics["log_loss"] = log_loss(labels, probs,labels=classes)
    
    

def apply_cli_overrides(config: Dict[str, Any], args):
    if args.train_path is not None:
        config["data"]["train_path"] = args.train_path

    if args.val_path is not None:
        config["data"]["val_path"] = args.val_path

    if args.output_dir is not None:
        config["output"]["output_dir"] = args.output_dir

    if args.n_epochs is not None:
        config["training"]["n_epochs"] = args.n_epochs

    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size

    if args.lr is not None:
        config["training"]["lr"] = args.lr

    if args.embedding_lr is not None:
        config["training"]["embedding_lr"] = args.embedding_lr

    if args.weight_decay is not None:
        config["training"]["weight_decay"] = args.weight_decay

    if args.device is not None:
        config["training"]["device"] = args.device

    if args.user_embedding_path is not None:
        config["embeddings"]["user_embedding_path"] = args.user_embedding_path

    if args.item_embedding_path is not None:
        config["embeddings"]["item_embedding_path"] = args.item_embedding_path

    if args.freeze_user_embeddings:
        config["embeddings"]["freeze_user_embeddings"] = True

    if args.freeze_item_embeddings:
        config["embeddings"]["freeze_item_embeddings"] = True

    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DiscoveryNet from a YAML configuration file."
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )

    parser.add_argument(
        "--train-path",
        type=str,
        default=None,
        help="Override training dataset path.",
    )

    parser.add_argument(
        "--val-path",
        type=str,
        default=None,
        help="Override validation dataset path.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory.",
    )

    parser.add_argument(
        "--n-epochs",
        type=int,
        default=None,
        help="Override number of training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate.",
    )

    parser.add_argument(
        "--embedding-lr",
        type=float,
        default=None,
        help="Override embedding learning rate.",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Override weight decay.",
    )

    def device_type(value):
        """Validate device string: 'cpu', 'mps', 'cuda', or 'cuda:N'."""
        if value in ("cpu", "mps", "cuda"):
            return value
        if value.startswith("cuda:"):
            try:
                idx = int(value.split(":", 1)[1])
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"Invalid CUDA device index in '{value}'. Expected format 'cuda:N'."
                )
            if idx < 0:
                raise argparse.ArgumentTypeError(f"CUDA device index must be >= 0, got {idx}.")
            return value
        raise argparse.ArgumentTypeError(
            f"Invalid device '{value}'. Expected 'cpu', 'mps', 'cuda', or 'cuda:N'."
        )

    parser.add_argument(
        "--device",
        type=device_type,
        default=None,
        help="Override training device. Use 'cpu', 'mps', 'cuda', or 'cuda:N' to select a specific GPU (e.g., 'cuda:0', 'cuda:1').",
    )

    parser.add_argument(
        "--user-embedding-path",
        type=str,
        default=None,
        help="Path to custom user embeddings.",
    )

    parser.add_argument(
        "--item-embedding-path",
        type=str,
        default=None,
        help="Path to custom item embeddings.",
    )

    parser.add_argument(
        "--freeze-user-embeddings",
        action="store_true",
        help="Freeze user embeddings during training.",
    )

    parser.add_argument(
        "--freeze-item-embeddings",
        action="store_true",
        help="Freeze item embeddings during training.",
    )

    return parser.parse_args()
 

def train_from_config(config: Dict[str, Any]):
    data_cfg = config["data"]
    dataset_dir = data_cfg["dataset_dir"]
    model_cfg = config["model"]
    embedding_cfg = config.get("embeddings", {})
    training_cfg = config["training"]
    output_cfg = config["output"]

    stat_data = load_stat_data(dataset_dir)
    model_cfg['n_users'] = stat_data['user_idx']
    model_cfg['n_items'] = stat_data['item_idx']
    model_cfg['target_classes'] = stat_data[data_cfg['label_col']]
    seed = training_cfg.get("seed", 42)
    set_seed(seed)

    device = training_cfg.get("device")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    output_dir = Path(output_cfg.get("output_dir", "runs/discovery_net"))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")
    print(f"Output directory: {output_dir}")
    
    tensorboard_dir = output_dir / "tensorboard"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    tb_writer = SummaryWriter(log_dir=str(tensorboard_dir))

    print(f"TensorBoard logs: {tensorboard_dir}")

    train_df = load_dataframe(
        path=dataset_dir+data_cfg["train_path"],
        file_format=data_cfg.get("file_format"),
    )
    
    val_df = load_dataframe(
        path=dataset_dir+data_cfg["val_path"],
        file_format=data_cfg.get("file_format"),
    )
    

    event_feature_sizes = data_cfg["event_features"]
    event_feature_names = list(event_feature_sizes.keys())

    train_rows = dataframe_to_rows(train_df)
    val_rows = dataframe_to_rows(val_df)
    

    train_dataset = DiscoveryDataset(
        rows=train_rows,
        num_classes=model_cfg['target_classes'],
        event_feature_names=event_feature_names,
        label_col=data_cfg.get("label_col", "label"),
        user_col=data_cfg.get("user_col", "user_id"),
        item_col=data_cfg.get("item_col", "item_id"),
    )

    val_dataset = DiscoveryDataset(
        rows=val_rows,
        num_classes=model_cfg['target_classes'],
        event_feature_names=event_feature_names,
        label_col=data_cfg.get("label_col", "label"),
        user_col=data_cfg.get("user_col", "user_id"),
        item_col=data_cfg.get("item_col", "item_id"),
    )
     

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg.get("batch_size", 1024),
        shuffle=True,
        num_workers=training_cfg.get("num_workers", 0),
        pin_memory=device == "cuda",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg.get("batch_size", 1024),
        shuffle=False,
        num_workers=training_cfg.get("num_workers", 0),
        pin_memory=device == "cuda",
    )
    
    user_embedding_weights = None
    item_embedding_weights = None

    if "user" in model_cfg["inputs"]:
        user_embedding_weights = load_embedding_weights(
            data_cfg["dataset_dir"],
            embedding_cfg.get("user_embedding_path")
        )

        if user_embedding_weights is not None:
            user_embedding_weights = user_embedding_weights.to(device)

    if "item" in model_cfg["inputs"]:
        item_embedding_weights = load_embedding_weights(
            data_cfg["dataset_dir"],
            embedding_cfg.get("item_embedding_path")
        )

    if item_embedding_weights is not None:
        item_embedding_weights = item_embedding_weights.to(device)
    
    
    model_kwargs = {
        "hidden_dims": model_cfg.get("hidden_dims", [256, 128, 64]),
        "emb_size": model_cfg["emb_size"],
        "target_classes": model_cfg.get("target_classes"),
        "event_feature_sizes": event_feature_sizes,
        "n_users": model_cfg["n_users"],
        "n_items": model_cfg["n_items"],
        "user_embedding_weights": user_embedding_weights,
        "item_embedding_weights": item_embedding_weights,
        "freeze_user_embeddings": embedding_cfg.get("freeze_user_embeddings", False),
        "freeze_item_embeddings": embedding_cfg.get("freeze_item_embeddings", False),
        "dropout": model_cfg.get("dropout", 0.0),
        "inputs": model_cfg["inputs"],
        } 
    config['model'] = model_kwargs  
    model = ShallowDiscoveryNet(
        **model_kwargs
    ).to(device)

    #criterion = nn.BCEWithLogitsLoss()
    #criterion  = nn.CrossEntropyLoss(weight=torch.tensor([1,4,15,18]).to(device))
    criterion  = nn.CrossEntropyLoss()

    #counts = np.array([513068, 180939, 50319, 42473])

    #weights = counts.sum() / (len(counts) * counts)
    #weights = torch.tensor(weights, dtype=torch.float32, device=device)
    #criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = build_optimizer(
        model=model,
        lr=training_cfg.get("lr", 1e-3),
        embedding_lr=training_cfg.get("embedding_lr"),
        weight_decay=training_cfg.get("weight_decay", 1e-5),
    )

    n_epochs = training_cfg.get("n_epochs", 10)
    grad_clip_norm = training_cfg.get("grad_clip_norm", 1.0)

    best_val_loss = float("inf")
    best_state_dict = None

    history = {
        "train_loss": [],
        "val_loss": [],
    }
    count_epoch=0
    checkpoint_every_n_epochs= training_cfg.get("checkpoint_every_n_epochs", 10)
    
    early_stop_state = {"best_score": None, "counter": 0, "best_epoch": 0}
    patience = training_cfg.get("patience", n_epochs)
    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_clip_norm=grad_clip_norm,
        )

        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        val_loss = val_metrics["loss"]

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))

        log_to_tensorboard(
                writer=tb_writer,
                metrics={'loss':float(train_loss)},
                epoch=epoch,
                prefix="train",
            )
        
        tb_val_metrics= {"loss":float(val_loss), **val_metrics["metrics"]}
        
        log_to_tensorboard(
                writer=tb_writer,
                metrics=tb_val_metrics,
                epoch=epoch,
                prefix="val",
            )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.5f} | "
            f"val_loss={val_loss:.5f}"
        )
        count_epoch+=1
        if count_epoch == checkpoint_every_n_epochs:   
            epoch_ckpt_path = output_dir / f"checkpoint_epoch_{epoch:03d}.pt"
                

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "history": history,
                },
                epoch_ckpt_path,
            )
            count_epoch=0
            
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())

            best_model_path = output_dir / output_cfg.get(
                "model_name",
                "discovery_net.pt",
            )
            save_metrics(val_metrics["metrics"], output_dir, "best_val_metrics.json") 

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "history": history,
                    "best_val_loss": best_val_loss,
                },
                best_model_path,
            )

            print(f"Saved best model to: {best_model_path}")
            
        
        should_stop = check_early_stopping(
            val_loss, early_stop_state, patience=patience, min_delta=1e-6, mode="min", epoch=epoch
        )

        if should_stop:
            print(f"Early stopping triggered at epoch {epoch}. "
                f"Best val_loss={early_stop_state['best_score']:.4f} at epoch {early_stop_state['best_epoch']}.")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    history_path = output_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    resolved_config_path = output_dir / "resolved_config.yaml"
    config_to_save = deepcopy(config)
    config_to_save['model']["user_embedding_weights"] = None
    config_to_save['model']["item_embedding_weights"] = None
    save_yaml_config(config_to_save, str(resolved_config_path))

    print(f"Saved history to: {history_path}")
    print(f"Saved resolved config to: {resolved_config_path}")
    print(f"Best validation loss: {best_val_loss:.5f}")

    return model, history

def predict_from_model(
    model,
    dataloader,
    device,
):
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            user_ids = batch["user_id"].to(device)
            item_ids = batch["item_id"].to(device)

            event_features = {
                key: value.to(device)
                for key, value in batch["event_features"].items()
            }

            labels = batch["label"].to(device)

            logits = model(
                user_ids=user_ids,
                item_ids=item_ids,
                event_features=event_features,
            )

            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    return {
        "probs": torch.cat(all_probs),
        "labels": torch.cat(all_labels),
    }
def main():

    
    args = parse_args()

    config = load_yaml_config(args.config)

    config = apply_cli_overrides(config, args)
    run_dir, config = create_unique_run_dir(config)
    train_from_config(config)
    
    test_results = test_model(config=config, model=None, device=config["training"]["device"], threshold=0.5)
    save_pickle(test_results, (Path(config['output']['output_dir'])/"test_predictions.pkl"))
if __name__ == "__main__":
    main()