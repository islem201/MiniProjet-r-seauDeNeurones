"""Entrainement et evaluation avec les boites originales des fichiers TXT."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.detector import (
    ANCHORS,
    NMS_IOU_THRESHOLD,
    RATIO_MAX,
    SCORE_THRESHOLD,
    STRIDE,
    Detector,
    ShapeDataset,
    compute_loss,
    iou,
    predict,
)


def train_model(
    model: Detector,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Entraine une epoque avec les targets encodees necessaires a la loss."""
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = compute_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.shape[0]
    return total_loss / len(loader.dataset)


def read_boxes(label_path: Path) -> torch.Tensor:
    """Lit toutes les boites originales d'un fichier TXT, en pixels."""
    lines = label_path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return torch.zeros((0, 4), dtype=torch.float32)
    return torch.tensor(
        [[float(value) for value in line.split()[:4]] for line in lines],
        dtype=torch.float32,
    )


def _to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convertit les boites originales cx,cy,w,h vers x1,y1,x2,y2."""
    cx, cy, width, height = boxes.unbind(dim=1)
    return torch.stack(
        (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2),
        dim=1,
    )


def _match_boxes(
    predicted: torch.Tensor,
    truth: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[list[float], int]:
    """Apparie les boites au plus une fois selon le meilleur IoU disponible."""
    if len(truth) == 0:
        return [0.0] * len(predicted), 0
    if len(predicted) == 0:
        return [0.0] * len(truth), 0

    pairs = []
    overlaps = torch.stack([iou(box, truth) for box in predicted])
    for prediction_index in range(len(predicted)):
        for truth_index in range(len(truth)):
            pairs.append(
                (
                    float(overlaps[prediction_index, truth_index]),
                    prediction_index,
                    truth_index,
                )
            )
    pairs.sort(reverse=True)

    used_predictions = set()
    used_truth = set()
    matched_ious = []
    for overlap, prediction_index, truth_index in pairs:
        if prediction_index in used_predictions or truth_index in used_truth:
            continue
        used_predictions.add(prediction_index)
        used_truth.add(truth_index)
        matched_ious.append(overlap)

    true_positives = sum(overlap > threshold for overlap in matched_ious)
    unmatched_truth = len(truth) - len(used_truth)
    return matched_ious + [0.0] * unmatched_truth, true_positives


def evaluate_model(
    model: Detector,
    dataset: Dataset,
    size: int,
    batch_size: int = 64,
) -> dict[str, float]:
    """Evalue les predictions finales contre les boites originales des TXT."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    all_ious = []
    matched_predictions = 0
    total_predictions = 0
    total_truth = 0

    with torch.no_grad():
        offset = 0
        for images, _ in loader:
            logits = model(images)
            for batch_index, logit in enumerate(logits):
                image_path = Path(dataset.paths[offset + batch_index])
                truth = _to_xyxy(read_boxes(image_path.with_suffix(".txt")))
                predicted, _ = predict(
                    logit,
                    size,
                    model.anchors,
                    score_threshold=SCORE_THRESHOLD,
                    iou_threshold=NMS_IOU_THRESHOLD,
                )
                matched_ious, true_positives = _match_boxes(predicted, truth)
                all_ious.extend(matched_ious)
                matched_predictions += true_positives
                total_predictions += len(predicted)
                total_truth += len(truth)
            offset += len(logits)

    iou_values = torch.tensor(all_ious) if all_ious else torch.tensor([0.0])
    return {
        "mean_iou": float(iou_values.mean()),
        "recall": matched_predictions / max(total_truth, 1),
        "precision": matched_predictions / max(total_predictions, 1),
    }


def create_datasets(
    root: Path,
    size: int = 64,
    anchors=ANCHORS,
) -> tuple[ShapeDataset, ShapeDataset]:
    """Cree les datasets train et validation a partir des PNG/TXT existants."""
    return (
        ShapeDataset(root / "train", size=size, anchors=anchors),
        ShapeDataset(root / "val", size=size, anchors=anchors),
    )
