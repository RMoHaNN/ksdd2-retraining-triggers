"""KolektorSDD2 labels, validation split and camera conditions."""

import io
from pathlib import Path

import torch
import numpy as np
import yaml
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
IMG_H, IMG_W = CONFIG["data"]["image_size"]
MEAN = np.array(CONFIG["data"]["mean"], dtype=np.float32)
STD = np.array(CONFIG["data"]["std"], dtype=np.float32)


def image_item(path):
    mask = path.with_name(path.stem + "_GT.png")
    with Image.open(mask) as image:
        label = int(np.asarray(image).max() > 0)
    return str(path), str(mask), label


def get_splits(data_path):
    items = [image_item(p) for p in sorted((Path(data_path) / "train").glob("*.png"))
             if p.stem.isdigit()]
    rng = np.random.RandomState(CONFIG["data"]["split_seed"])
    positive = [item for item in items if item[2] == 1]
    negative = [item for item in items if item[2] == 0]
    rng.shuffle(positive)
    rng.shuffle(negative)
    fraction = CONFIG["data"]["validation_fraction"]
    npv, nnv = round(len(positive) * fraction), round(len(negative) * fraction)
    test = [image_item(p) for p in sorted((Path(data_path) / "test").glob("*.png"))
            if p.stem.isdigit()]
    return positive[npv:] + negative[nnv:], positive[:npv] + negative[:nnv], test


def figure_items(data_path):
    path = Path(data_path) / "test" / (CONFIG["figures"]["condition_image"] + ".png")
    if path.is_file() and path.with_name(path.stem + "_GT.png").is_file():
        return [image_item(path)]
    return []


def noise_seed(path):
    image_id = int(Path(path).stem)
    return CONFIG.get("noise_seed_overrides", {}).get(image_id, image_id)


def apply_condition(image, condition, rng):
    settings = CONFIG["conditions"][condition or "nominal"]
    if "brightness" in settings:
        image = ImageEnhance.Brightness(image).enhance(settings["brightness"])
    if "contrast" in settings:
        image = ImageEnhance.Contrast(image).enhance(settings["contrast"])
    if "noise_sigma" in settings:
        array = np.asarray(image).astype(np.float32)
        array = array + rng.normal(0, settings["noise_sigma"], array.shape)
        image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    if "jpeg_quality" in settings:
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=settings["jpeg_quality"])
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")
    if "blur_radius" in settings:
        image = image.filter(ImageFilter.GaussianBlur(radius=settings["blur_radius"]))
    return image


class KSDD2(torch.utils.data.Dataset):
    def __init__(self, items, condition=None):
        self.items, self.condition = items, condition

    def __len__(self):
        return len(self.items)

    def load(self, index):
        path, _, label = self.items[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        seed = noise_seed(path) if self.condition == "lighting_noise" else 0
        return apply_condition(image, self.condition, np.random.RandomState(seed)), label

    def __getitem__(self, index):
        image, label = self.load(index)
        array = np.asarray(image.resize((IMG_W, IMG_H), Image.BILINEAR)).astype(np.float32) / 255
        array = (array - MEAN) / STD
        return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))), torch.tensor(label)


def loader(dataset):
    return torch.utils.data.DataLoader(
        dataset, batch_size=CONFIG["model"]["prediction_batch_size"],
        num_workers=CONFIG["model"]["workers"], shuffle=False,
    )
