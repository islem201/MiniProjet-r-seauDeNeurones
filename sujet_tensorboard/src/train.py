"""Entraîne le détecteur, évalue (IoU moyen, Rappel, Précision) et sauvegarde une figure."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# On importe RATIO_MAX et ANCHORS pour nos calculs
from src.detector import (
    Detector, ShapeDataset, compute_loss, iou, predict, STRIDE, ANCHORS,
    RATIO_MAX, SCORE_THRESHOLD, NMS_IOU_THRESHOLD,
)

def train(model: Detector, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    """Entraîne le modèle sur une époque et retourne la perte moyenne."""
    model.train() 
    total_loss = 0.0
    for images, targets in loader:
        optimizer.zero_grad() 
        outputs = model(images) 
        loss = compute_loss(outputs, targets) 
        loss.backward() 
        optimizer.step() 
        total_loss += loss.item() 
    return total_loss / len(loader.dataset)

# --- CORRECTION DE LA LIMITE 5 (Vraie évaluation Multi-figures) ---
def evaluate(model: Detector, loader: DataLoader, size: int) -> dict:
    """Évalue un détecteur multi-figures sur 3 métriques : IoU moyen, Rappel, Précision."""
    model.eval()
    ious = []
    matched_preds = 0  # Compte les bonnes prédictions (pour la précision)
    n_preds = 0        # Compte le total de prédictions (pour la précision)
    
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images)
            grid = targets.shape[-1]
            
            for logit, target in zip(logits, targets):
                # 1. On récupère TOUTES les boîtes prédites valides
                pred_boxes, _ = predict(
                    logit,
                    size,
                    model.anchors,
                    score_threshold=SCORE_THRESHOLD,
                    iou_threshold=NMS_IOU_THRESHOLD,
                )
                n_preds += len(pred_boxes)
                
                # 2. Reconstruction de TOUTES les vraies boîtes depuis les 15 canaux
                n_anchors = len(model.anchors)
                target = target.view(n_anchors, 5, grid, grid)
                true_boxes = []
                
                for k in range(n_anchors):
                    for row in range(grid):
                        for col in range(grid):
                            if target[k, 0, row, col] > 0.5:
                                dx = target[k, 1, row, col]
                                dy = target[k, 2, row, col]
                                tw = target[k, 3, row, col]
                                th = target[k, 4, row, col]
                                
                                cx = (col + dx) * STRIDE
                                cy = (row + dy) * STRIDE
                                
                                # On n'oublie pas le RATIO_MAX ici !
                                w = tw * RATIO_MAX * model.anchors[k][0] * size
                                h = th * RATIO_MAX * model.anchors[k][1] * size
                                
                                true_boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
                
                # 3. Calcul des scores sur TOUTES les boîtes
                if true_boxes:
                    truth_tensor = torch.tensor(true_boxes, device=logits.device)
                    
                    # A. Pour le Rappel et l'IoU moyen : On regarde si chaque vrai cercle a été trouvé
                    for t_box in truth_tensor:
                        if len(pred_boxes) > 0:
                            ious.append(float(iou(t_box, pred_boxes).max()))
                        else:
                            ious.append(0.0)
                            
                    # B. Pour la Précision : On regarde si chaque prédiction correspond à un vrai cercle
                    for p_box in pred_boxes:
                        if float(iou(p_box, truth_tensor).max()) > 0.5:
                            matched_preds += 1
                else:
                    # Si aucune vraie boîte (rare), on ajoute des zéros si le modèle en a prédit
                    for _ in pred_boxes:
                        ious.append(0.0)

    # 4. Synthèse des résultats
    t = torch.tensor(ious) if ious else torch.tensor([0.0])
    recall = (t > 0.5).float().mean().item()
    precision = matched_preds / max(n_preds, 1) # max(..., 1) évite la division par zéro
    
    return {
        "mean_iou": t.mean().item(), 
        "recall": recall, 
        "precision": precision
    }