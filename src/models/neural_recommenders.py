"""Neural Collaborative Filtering and Two-Tower recommendation models."""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=32):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.mlp = nn.Sequential(nn.Linear(embedding_dim * 2, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, users, items):
        return self.mlp(torch.cat([self.user_embedding(users), self.item_embedding(items)], dim=1)).squeeze(1)


class TwoTowerModel(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=32):
        super().__init__()
        self.user_tower = nn.Sequential(nn.Embedding(n_users, embedding_dim), nn.Linear(embedding_dim, embedding_dim), nn.ReLU())
        self.item_tower = nn.Sequential(nn.Embedding(n_items, embedding_dim), nn.Linear(embedding_dim, embedding_dim), nn.ReLU())

    def forward(self, users, items):
        return (self.user_tower(users) * self.item_tower(items)).sum(dim=1)


def train_model(model, users, items, labels, epochs=3, batch_size=2048, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(TensorDataset(torch.tensor(users), torch.tensor(items), torch.tensor(labels, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        for batch_users, batch_items, batch_labels in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_users.to(device), batch_items.to(device)), batch_labels.to(device))
            loss.backward()
            optimizer.step()
    return model.cpu()


def save_checkpoint(model, path, user_map, item_map, model_name):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "n_users": len(user_map), "n_items": len(item_map)}, path)
    path.with_suffix(".json").write_text(json.dumps({"model": model_name, "user_map": user_map, "item_map": item_map}), encoding="utf-8")
