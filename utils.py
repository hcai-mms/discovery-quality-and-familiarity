
import numpy as np
import torch
from torch import nn
from typing import Optional, Sequence, Tuple, Union, Dict, Any,List
import yaml 
import random 
from pathlib import Path
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from uuid import uuid4
import yaml
import math

def create_unique_run_dir(config):
    """
    Create a unique run directory from a config dictionary.

    Example output:
    runs/discovery_net/20260702_113614_seed42_a8f3c2

    Parameters
    ----------
    config : dict
        Loaded YAML config.

    Returns
    -------
    run_dir : pathlib.Path
        Unique run directory.
    config : dict
        Updated config with output.run_dir.
    """

    base_output_dir = Path(config["output"]["output_dir"])

    seed = config.get("training", {}).get("seed", "na")
    run_name = config.get("output", {}).get("run_name")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid4().hex[:6]

    if run_name is None:
        run_name = f"{timestamp}_seed{seed}_{short_uuid}"
    else:
        run_name = f"{timestamp}_{run_name}_seed{seed}_{short_uuid}"

    run_dir = base_output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    config["output"]["run_dir"] = str(run_dir)
    config["output"]["model_path"] = str(run_dir / config["output"]["model_name"])
    config["output"]["output_dir"] = str(run_dir)
    if config["output"].get("save_config", True):
        config_path = run_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        config["output"]["config_path"] = str(config_path)

    return run_dir, config



def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml_config(config: Dict[str, Any], path: str):
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataframe(path: str, file_format: Optional[str] = None) -> pd.DataFrame:
    path_obj = Path(path)

    if file_format is None:
        suffix = path_obj.suffix.lower()
        if suffix == ".csv":
            file_format = "csv"
        elif suffix in [".parquet", ".pq"]:
            file_format = "parquet"
        else:
            raise ValueError(
                f"Could not infer file format from extension: {suffix}. "
                f"Please specify data.file_format."
            )

    if file_format == "csv":
        return pd.read_csv(path)

    if file_format == "parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported file format: {file_format}")


def dataframe_to_rows(df: pd.DataFrame):
    return df.to_dict(orient="records")


def load_embedding_weights(root_dir:Optional[str] ,emb_file: Optional[str]) -> Optional[torch.Tensor]:
    if emb_file is None:
        return None

    if str(emb_file).lower() in ["none", "null", ""]:
        return None
    emb_file= root_dir+emb_file
    path_obj = Path(emb_file)
    suffix = path_obj.suffix.lower()

    if suffix in [".pt", ".pth"]:
        weights = torch.load(emb_file, map_location="cpu")

        if isinstance(weights, dict):
            if "weight" in weights:
                weights = weights["weight"]
            elif "embeddings" in weights:
                weights = weights["embeddings"]
            elif "state_dict" in weights:
                raise ValueError(
                    f"{emb_file} looks like a checkpoint with state_dict. "
                    f"Please provide a raw embedding tensor or a dict with key 'weight' or 'embeddings'."
                )
            else:
                raise ValueError(
                    f"Could not find embedding tensor in dict. Available keys: {list(weights.keys())}"
                )

        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights)

        return weights.float()

    if suffix == ".npy":
        weights = np.load(emb_file)
        return torch.tensor(weights, dtype=torch.float32)

    raise ValueError(
        f"Unsupported embedding format: {suffix}. "
        f"Supported formats are .pt, .pth, and .npy."
    )


def build_optimizer(
    model,
    lr: float,
    weight_decay: float,
    embedding_lr: Optional[float] = None,
):
    if embedding_lr is None:
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

    embedding_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("user_emb") or name.startswith("item_emb"):
            embedding_params.append(param)
        else:
            other_params.append(param)

    param_groups = []

    if embedding_params:
        param_groups.append(
            {
                "params": embedding_params,
                "lr": embedding_lr,
                "weight_decay": weight_decay,
            }
        )

    if other_params:
        param_groups.append(
            {
                "params": other_params,
                "lr": lr,
                "weight_decay": weight_decay,
            }
        )

    return torch.optim.AdamW(param_groups)


def load_model(
    model_class,
    checkpoint_path,
    device,
    model_kwargs=None,
    strict=True,
):
    """
    Load a saved PyTorch model from a checkpoint.

    Parameters
    ----------
    model_class : type
        The model class to instantiate.

    checkpoint_path : str
        Path to the saved checkpoint file.

    device : torch.device or str
        Device where the model should be loaded.

    model_kwargs : dict, optional
        Arguments used to initialize the model.

    strict : bool
        Whether to strictly enforce that checkpoint keys match model keys.

    Returns
    -------
    model : torch.nn.Module
        Loaded model in eval mode.
    checkpoint : dict
        Loaded checkpoint object.
    """

    if model_kwargs is None:
        model_kwargs = {}

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )
    model_kwargs= checkpoint.get("config", model_kwargs).get('model',model_kwargs)
    model = model_class(**model_kwargs).to(device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=strict
        )
    else:
        model.load_state_dict(
            checkpoint,
            strict=strict
        )

    model.eval()

    return model, checkpoint




def make_json_serializable(obj):
    """
    Convert common ML metric objects into JSON-serializable Python objects.
    """

    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu()

        if obj.numel() == 1:
            return make_json_serializable(obj.item())

        return obj.tolist()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return make_json_serializable(obj.item())

    if isinstance(obj, dict):
        return {
            str(key): make_json_serializable(value)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [
            make_json_serializable(value)
            for value in obj
        ]

    if isinstance(obj, tuple):
        return [
            make_json_serializable(value)
            for value in obj
        ]

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None

        return obj

    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj

    return str(obj)


def save_metrics(
    metrics,
    run_dir,
    filename="metrics.json",
):
    """
    Save metrics as JSON inside the run directory.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary. Can contain torch tensors, numpy arrays,
        numpy scalars, nested dictionaries, lists, tuples, etc.

    run_dir : str or pathlib.Path
        Directory where the metrics file will be saved.

    filename : str
        Name of the JSON file.

    Returns
    -------
    pathlib.Path
        Path to the saved metrics file.
    """

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / filename

    serializable_metrics = make_json_serializable(metrics)

    with open(metrics_path, "w") as f:
        json.dump(
            serializable_metrics,
            f,
            indent=2,
            allow_nan=False,
        )

    return metrics_path

def check_early_stopping(current_score, state, patience=10, min_delta=0.0, mode="min", verbose=True, epoch=None):
    """
    Checks whether training should stop early based on validation loss (or any monitored metric).

    Args:
        current_score (float): The metric value for the current epoch (e.g., val_loss).
        state (dict): Mutable dict tracking early stopping state across calls.
                       Initialize as: {"best_score": None, "counter": 0, "best_epoch": 0}
        patience (int): Number of epochs to tolerate no improvement before stopping.
        min_delta (float): Minimum change to qualify as an improvement.
        mode (str): "min" for loss, "max" for metrics like accuracy/F1.
        verbose (bool): Whether to print status messages.
        epoch (int, optional): Current epoch number, for logging.

    Returns:
        bool: True if training should stop, False otherwise.
    """
    if state["best_score"] is None:
        state["best_score"] = current_score
        state["best_epoch"] = epoch
        return False

    improved = (
        current_score < state["best_score"] - min_delta
        if mode == "min"
        else current_score > state["best_score"] + min_delta
    )

    if improved:
        state["best_score"] = current_score
        state["best_epoch"] = epoch
        state["counter"] = 0
        if verbose:
            print(f"[EarlyStopping] Improved at epoch {epoch}: {current_score:.6f}")
    else:
        state["counter"] += 1
        if verbose:
            print(f"[EarlyStopping] No improvement for {state['counter']}/{patience} epochs "
                  f"(best={state['best_score']:.6f} at epoch {state['best_epoch']})")
        if state["counter"] >= patience:
            return True

    return False



import pandas as pd
import numpy as np


def generate_recsys_report_brief(
    parquet_path: str,
    user_col: str = "user_id",
    item_col: str = "item_id",
    dataset_name: str = "Dataset",
) -> dict:
    """
    Brief recommender-systems dataset statistics, suitable for a compact
    table in an ACM paper (e.g., Table 1 dataset description).
    """
    df = pd.read_parquet(parquet_path, columns=[user_col, item_col])

    n_interactions = len(df)
    n_users = df[user_col].nunique()
    n_items = df[item_col].nunique()
    density = n_interactions / (n_users * n_items)

    interactions_per_user = df.groupby(user_col).size()

    table = {
        "Dataset": dataset_name,
        "#Users": n_users,
        "#Items": n_items,
        "#Interactions": n_interactions,
        "Avg. actions/user": round(interactions_per_user.mean(), 2),
        "Density (\\%)": round(density * 100, 4),
    }
    return table


def to_acm_latex_table(tables: list[dict], caption: str = "Dataset statistics.", label: str = "tab:dataset_stats") -> str:
    """
    Convert one or more dataset stat dicts (e.g., train/val/test, or multiple
    datasets) into a booktabs-style LaTeX table for the ACM template.
    """
    metrics = [k for k in tables[0].keys() if k != "Dataset"]
    header = " & ".join(["Dataset"] + metrics) + " \\\\"

    rows = []
    for t in tables:
        row = [str(t["Dataset"])] + [f"{t[m]:,}" if isinstance(t[m], int) else f"{t[m]}" for m in metrics]
        rows.append(" & ".join(row) + " \\\\")

    col_spec = "l" + "r" * len(metrics)

    latex = (
        "\\begin{table}\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{{label}}}\n"
        f"  \\begin{{tabular}}{{{col_spec}}}\n"
        "    \\toprule\n"
        f"    {header}\n"
        "    \\midrule\n"
        + "\n".join(f"    {r}" for r in rows) + "\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}"
    )
    return latex


import pickle


def save_pickle(data: dict, output_path: str) -> None:
    """
    Save a dictionary of numpy arrays (or any picklable object) to a pickle file.

    Parameters
    ----------
    data : dict
        Dictionary to save, e.g. {key: np.ndarray, ...}
    output_path : str
        Path to write the .pkl file to.
    """
    with open(output_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved {len(data)} entries to {output_path}")

def save_column_to_npy(df: pd.DataFrame, column: str, output_path: str) -> None:
    """
    Save a pandas column of numpy arrays (e.g. embeddings) to a .npy file
    as a single stacked 2D array.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the column.
    column : str
        Name of the column whose values are numpy arrays (all same length).
    output_path : str
        Path to write the .npy file to.
    """
    matrix = np.stack(df[column].values)  # shape (n_rows, embedding_dim)
    np.save(output_path, matrix)
    print(f"Saved array of shape {matrix.shape} to {output_path}")