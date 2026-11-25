"""ResNet-18 model loading and defect prediction."""

import torch
import numpy as np
import torchvision
from torch import nn

from .data import loader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model():
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model()
    model.load_state_dict(checkpoint["state"])
    return model.to(DEVICE).eval(), checkpoint["threshold"]


@torch.no_grad()
def predict(model, dataset, return_features=False):
    model.eval()
    features, hook = {}, None
    if return_features:
        hook = model.avgpool.register_forward_hook(
            lambda module, inputs, output: features.update(pooled=output.flatten(1))
        )
    scores, labels, embeddings = [], [], []
    try:
        for images, targets in loader(dataset):
            output = model(images.to(DEVICE))
            scores.append(torch.softmax(output.float(), 1)[:, 1].cpu().numpy())
            labels.append(targets.numpy())
            if return_features:
                embeddings.append(features["pooled"].float().cpu().numpy())
    finally:
        if hook is not None:
            hook.remove()
    result = np.concatenate(scores), np.concatenate(labels)
    return (*result, np.concatenate(embeddings)) if return_features else result
