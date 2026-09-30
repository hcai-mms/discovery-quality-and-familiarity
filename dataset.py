import torch
import torch.nn as nn
import yaml
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class DiscoveryDataset(Dataset):
    def __init__(
        self,
        rows,
        event_feature_names,
        num_classes,
        label_col: str = "label",
        user_col: str = "user_id",
        item_col: str = "item_id",
    ):
        self.rows = rows
        self.event_feature_names = event_feature_names
        self.label_col = label_col
        self.user_col = user_col
        self.item_col = item_col
        self.num_classes = num_classes

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        user_id = torch.tensor(row[self.user_col], dtype=torch.long)
        item_id = torch.tensor(row[self.item_col], dtype=torch.long)

        event_features = {
            feature: torch.tensor(row[feature], dtype=torch.long)
            for feature in self.event_feature_names
        }

        label = F.one_hot(torch.tensor(row[self.label_col], dtype=torch.long), num_classes=self.num_classes)

        return {
            "user_id": user_id,
            "item_id": item_id,
            "event_features": event_features,
            "label": label,
        }
