import numpy as np
import torch
from torch import nn
from typing import Optional, Sequence, Tuple, Union, Dict, Any,List


class ShallowDiscoveryNet(nn.Module):
    def __init__(
        self,
        hidden_dims: List[int],
        emb_size: int,
        target_classes: int,
        event_feature_sizes: Dict[str, int],
        n_users: int,
        n_items: int,
        inputs: List[str],
        user_embedding_weights: Optional[torch.Tensor] = None,
        item_embedding_weights: Optional[torch.Tensor] = None,
        freeze_user_embeddings: bool = False,
        freeze_item_embeddings: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.event_feature_sizes = event_feature_sizes
        self.n_event_features = len(event_feature_sizes)
        self.inputs = inputs

        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer size")
        
        if len(inputs) == 0:
            raise ValueError("inputs must contain at least one feature")
        
        # Check that all requested inputs are valid
        valid_inputs = {"user", "item"} | set(event_feature_sizes.keys())

        unknown_inputs = set(inputs) - valid_inputs
        if unknown_inputs:
            raise ValueError(
                f"Unknown model inputs: {unknown_inputs}. "
                f"Valid inputs are: {valid_inputs}"
            )
        # ---------------------------------------------------------
        #   User embedding
        # ---------------------------------------------------------

        if "user" in self.inputs:
            if user_embedding_weights is not None:
                if user_embedding_weights.shape[0] != n_users:
                    raise ValueError(
                        f"user_embedding_weights has {user_embedding_weights.shape[0]} rows, "
                        f"but n_users={n_users}"
                    )
                if user_embedding_weights.shape[1] != emb_size:
                    raise ValueError(
                        f"user_embedding_weights has embedding size {user_embedding_weights.shape[1]}, "
                        f"but emb_size={emb_size}"
                    )

                self.user_emb = nn.Embedding.from_pretrained(
                    user_embedding_weights.float(),
                    freeze=freeze_user_embeddings,
                )
            else:
                self.user_emb = nn.Embedding(n_users, emb_size)
                self.user_emb.weight.requires_grad = not freeze_user_embeddings

        # ---------------------------------------------------------
        #   Item embedding
        # ---------------------------------------------------------
        if "item" in self.inputs:
            if item_embedding_weights is not None:
                if item_embedding_weights.shape[0] != n_items:
                    raise ValueError(
                        f"item_embedding_weights has {item_embedding_weights.shape[0]} rows, "
                        f"but n_items={n_items}"
                    )
                if item_embedding_weights.shape[1] != emb_size:
                    raise ValueError(
                        f"item_embedding_weights has embedding size {item_embedding_weights.shape[1]}, "
                        f"but emb_size={emb_size}"
                    )

                self.item_emb = nn.Embedding.from_pretrained(
                    item_embedding_weights.float(),
                    freeze=freeze_item_embeddings,
                )
            else:
                self.item_emb = nn.Embedding(n_items, emb_size)
                self.item_emb.weight.requires_grad = not freeze_item_embeddings

        # ---------------------------------------------------------
        # Event feature embeddings
        # ---------------------------------------------------------

        self.event_encoder = nn.ModuleDict()
        for feature, n_classes in event_feature_sizes.items():
            if feature in self.inputs:
                self.event_encoder[feature] = nn.Embedding(n_classes, emb_size)

        input_dim = len(self.inputs) * emb_size
        # input_dim = (self.n_event_features ) * emb_size
        layers = []
        dims = [input_dim] + hidden_dims

        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dims[-1], target_classes)
        print(self)

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        event_features: Dict[str, torch.Tensor],
    ):
        input_vecs = []

        for input_name in self.inputs:
            if input_name =="user":
                input_vecs.append(self.user_emb(user_ids))
            
            elif input_name =="item":
                input_vecs.append(self.item_emb(item_ids))

            else:
                if input_name not in event_features:
                    raise KeyError(f"Missing event feature: {input_name}")
                
                if input_name not in self.event_encoder:
                    raise KeyError(f"No encoder defined for event feature: {input_name}")
                

                feature_values = event_features[input_name]
                input_vecs.append(self.event_encoder[input_name](feature_values))

        x = torch.cat(input_vecs, dim=-1)

        # x = torch.cat(event_vecs, dim=-1)

        x = self.hidden_layers(x)
        logits = self.output_layer(x)

        return logits.squeeze(-1)
