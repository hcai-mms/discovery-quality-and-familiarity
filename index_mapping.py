from pathlib import Path
import json

import numpy as np
import pandas as pd


def fit_user_item_mappings(
    df,
    user_col="user_id",
    item_col="item_id",
    sort_values=True,
):
    """
    Create user/item ID to index mappings from a dataframe.

    Known IDs start from index 0.
    No unknown index is reserved.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing user and item columns.

    user_col : str
        User ID column name.

    item_col : str
        Item ID column name.

    sort_values : bool
        If True, unique IDs are sorted for deterministic mappings.
        If False, order of first appearance is used.

    Returns
    -------
    dict
        Dictionary containing user_to_idx, item_to_idx, idx_to_user,
        idx_to_item, num_users, and num_items.
    """

    if user_col not in df.columns:
        raise ValueError(f"user_col '{user_col}' not found in dataframe.")

    if item_col not in df.columns:
        raise ValueError(f"item_col '{item_col}' not found in dataframe.")

    if sort_values:
        users = sorted(df[user_col].dropna().unique().tolist())
        items = sorted(df[item_col].dropna().unique().tolist())
    else:
        users = df[user_col].dropna().drop_duplicates().tolist()
        items = df[item_col].dropna().drop_duplicates().tolist()

    user_to_idx = {
        user_id: idx
        for idx, user_id in enumerate(users)
    }

    item_to_idx = {
        item_id: idx
        for idx, item_id in enumerate(items)
    }

    idx_to_user = {
        idx: user_id
        for user_id, idx in user_to_idx.items()
    }

    idx_to_item = {
        idx: item_id
        for item_id, idx in item_to_idx.items()
    }

    mappings = {
        "user_to_idx": user_to_idx,
        "item_to_idx": item_to_idx,
        "idx_to_user": idx_to_user,
        "idx_to_item": idx_to_item,
        "num_users": len(user_to_idx),
        "num_items": len(item_to_idx),
    }

    return mappings


def transform_user_ids(
    user_ids,
    user_to_idx,
):
    """
    Convert raw user IDs to integer indexes.

    Raises
    ------
    KeyError
        If an unknown user ID is found.
    """

    return np.array(
        [
            user_to_idx[user_id]
            for user_id in user_ids
        ],
        dtype=np.int64,
    )


def transform_item_ids(
    item_ids,
    item_to_idx,
):
    """
    Convert raw item IDs to integer indexes.

    Raises
    ------
    KeyError
        If an unknown item ID is found.
    """

    return np.array(
        [
            item_to_idx[item_id]
            for item_id in item_ids
        ],
        dtype=np.int64,
    )


def inverse_transform_user_idxs(
    user_idxs,
    idx_to_user,
):
    """
    Convert user indexes back to raw user IDs.
    """

    return [
        idx_to_user[int(idx)]
        for idx in user_idxs
    ]


def inverse_transform_item_idxs(
    item_idxs,
    idx_to_item,
):
    """
    Convert item indexes back to raw item IDs.
    """

    return [
        idx_to_item[int(idx)]
        for idx in item_idxs
    ]


def transform_dataframe(
    df,
    mappings,
    user_col="user_id",
    item_col="item_id",
    user_idx_col="user_idx",
    item_idx_col="item_idx",
    copy=True,
):
    """
    Add user/item index columns to a dataframe.

    Raises
    ------
    KeyError
        If unknown user or item IDs are found.
    """

    if copy:
        df = df.copy()

    df[user_idx_col] = transform_user_ids(
        user_ids=df[user_col].values,
        user_to_idx=mappings["user_to_idx"],
    )

    df[item_idx_col] = transform_item_ids(
        item_ids=df[item_col].values,
        item_to_idx=mappings["item_to_idx"],
    )

    return df


def inverse_transform_dataframe(
    df,
    mappings,
    user_idx_col="user_idx",
    item_idx_col="item_idx",
    user_col="user_id",
    item_col="item_id",
    copy=True,
):
    """
    Add raw user/item ID columns from index columns.
    """

    if copy:
        df = df.copy()

    df[user_col] = inverse_transform_user_idxs(
        user_idxs=df[user_idx_col].values,
        idx_to_user=mappings["idx_to_user"],
    )

    df[item_col] = inverse_transform_item_idxs(
        item_idxs=df[item_idx_col].values,
        idx_to_item=mappings["idx_to_item"],
    )

    return df


def save_mappings(
    mappings,
    path,
):
    """
    Save mappings to JSON.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "user_to_idx": mappings["user_to_idx"],
        "item_to_idx": mappings["item_to_idx"],
    }

    with open(path, "w") as f:
        json.dump(data, f)

    return path


def load_mappings(path):
    """
    Load mappings from JSON.
    """

    path = Path(path)

    with open(path, "r") as f:
        data = json.load(f)

    user_to_idx = {
        user_id: int(idx)
        for user_id, idx in data["user_to_idx"].items()
    }

    item_to_idx = {
        item_id: int(idx)
        for item_id, idx in data["item_to_idx"].items()
    }

    idx_to_user = {
        idx: user_id
        for user_id, idx in user_to_idx.items()
    }

    idx_to_item = {
        idx: item_id
        for item_id, idx in item_to_idx.items()
    }

    mappings = {
        "user_to_idx": user_to_idx,
        "item_to_idx": item_to_idx,
        "idx_to_user": idx_to_user,
        "idx_to_item": idx_to_item,
        "num_users": len(user_to_idx),
        "num_items": len(item_to_idx),
    }

    return mappings

def load_stat_data(path):
    """
    Load stat_data from JSON.
    """

    path = Path(path)
    path = path / "stat_info.json"
    with open(path, "r") as f:
        data = json.load(f)
    return data