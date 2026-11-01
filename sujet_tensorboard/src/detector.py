"""Détecteur avec Ancres : feature maps -> K prédictions par cellule -> NMS -> plusieurs boîtes.

Image 64x64 -> 3 blocs conv stride 2 -> feature map 32x8x8 (stride 8).
Une conv 1x1 prédit pour chaque cellule K * 5 valeurs (K ancres, 5 valeurs par ancre) :
(objectness, dx, dy, w, h)
  - dx, dy in [0,1] : décalage du centre dans la cellule
  - w, h in [0,1]   : rapport (taille de la boîte / taille de l'ancre) divisé par RATIO_MAX
"""

from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

STRIDE = 8

# Nos 3 ancres (largeur, hauteur) normalisées. 
# C'est la base de "l'Option A" : un carré, un rectangle horizontal, un vertical.
ANCHORS = ((0.25, 0.25), (0.33, 0.33), (0.15, 0.15))

# --- CORRECTION DE LA LIMITE 1 ---
# Le réseau utilise une sigmoïde en sortie, donc la valeur max prédite est 1.0.
# Si on ne met pas ce ratio, une boîte ne pourra jamais être plus grande que son ancre.
# Avec RATIO_MAX = 3.0, la sigmoïde peut "gonfler" la boîte jusqu'à 3 fois la taille de l'ancre.
RATIO_MAX = 3.0
SCORE_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.45

def iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU entre une boîte a (4,) et des boîtes b (N, 4). (inchangé)"""
    inter_w = (torch.minimum(a[2], b[:, 2]) - torch.maximum(a[0], b[:, 0])).clamp(min=0)
    inter_h = (torch.minimum(a[3], b[:, 3]) - torch.maximum(a[1], b[:, 1])).clamp(min=0)
    inter = inter_w * inter_h
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a + area_b - inter).clamp(min=1e-8)

def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> list[int]:
    """Garde la meilleure boîte, supprime celles qui la chevauchent, recommence et retourne les indices des boîtes conservées.
    
    ATTENTION : J'ai remis le seuil par défaut à 0.5. Ton 1.1 d'avant désactivait 
    complètement le NMS (car l'IoU ne dépasse jamais 1).
    """
    order = scores.argsort(descending=True)
    kept = []
    while len(order):
        best = order[0]
        kept.append(int(best))
        order = order[1:]
        order = order[iou(boxes[best], boxes[order]) < iou_threshold]
    return kept

def anchor_iou(w: float, h: float, anchors=ANCHORS) -> torch.Tensor:
    """NOUVELLE FONCTION : Calcule l'IoU "de forme" entre un cercle et nos 3 ancres.
    
    On l'utilise pour savoir dans quel "tiroir" (quelle ancre) on va ranger notre cercle 
    pendant l'encodage. On centre tout sur 0 pour ne comparer que les tailles.
    """
    a = torch.tensor([-w / 2, -h / 2, w / 2, h / 2])
    b = torch.tensor([[-aw / 2, -ah / 2, aw / 2, ah / 2] for aw, ah in anchors])
    return iou(a, b)

# --- CORRECTION DU MULTI-FIGURES (Limite 3) ---
def code(boxes, size: int = 64, anchors=ANCHORS) -> torch.Tensor:
    """Encode une LISTE de boîtes en un tenseur cible de 15 canaux (3 ancres * 5).
    
    Si deux cercles tombent dans la même cellule, on les met dans des ancres différentes
    pour éviter qu'ils ne s'écrasent.
    """
    grid = size // STRIDE
    n_anchors = len(anchors)
    
    # Notre cible fait maintenant 15 canaux (K*5)
    target = torch.zeros(n_anchors * 5, grid, grid)
    
    # On trie les cercles du plus grand au plus petit pour l'encodage
    for cx, cy, w, h in sorted(boxes, key=lambda b: (-b[2] * b[3], b[0], b[1])):
        
        # 1. Dans quelle cellule 8x8 tombe ce cercle ?
        gx, gy = cx / size * grid, cy / size * grid
        col, row = min(int(gx), grid - 1), min(int(gy), grid - 1)
        
        # 2. Quelle est la meilleure ancre pour sa forme ?
        shape_iou = anchor_iou(w / size, h / size, anchors).tolist()
        
        # 3. On essaie de le ranger dans la meilleure ancre. 
        # Si la place est déjà prise par un autre cercle, on prend la 2ème meilleure ancre.
        for k in sorted(range(n_anchors), key=lambda k: (-round(shape_iou[k], 6), k)):
            if target[5 * k, row, col] == 0:  # Vérifie si le canal "objectness" est vide
                aw, ah = anchors[k]
                
                # On remplit les 5 canaux de cette ancre spécifique
                # C'est ici qu'on divise w et h par RATIO_MAX (Limite 1)
                target[5 * k:5 * k + 5, row, col] = torch.tensor([
                    1.0, 
                    gx - col, 
                    gy - row, 
                    w / size / aw / RATIO_MAX, 
                    h / size / ah / RATIO_MAX
                ])
                break # Le cercle est rangé, on passe au suivant
                
    return target
def decode(logits: torch.Tensor, size: int = 64, anchors=ANCHORS) -> tuple[torch.Tensor, torch.Tensor]:
    """Décode les prédictions (15, 8, 8) en boîtes (K*G*G, 4) et scores (K*G*G,)."""
    grid = logits.shape[-1]
    n_anchors = len(anchors)
    
    # 1. On sépare les 3 ancres. Le tenseur passe de [15, 8, 8] à [3, 5, 8, 8].
    # On applique la sigmoïde pour tout ramener entre 0 et 1.
    p = torch.sigmoid(logits.reshape(n_anchors, 5, grid, grid))
    
    # 2. Création de la grille (8x8) pour retrouver la position de chaque cellule.
    rows, cols = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    
    # 3. Préparation des ancres pour le calcul matriciel. 
    # On isole les largeurs et hauteurs des 3 ancres.
    anchor_w = torch.tensor([a[0] for a in anchors]).view(n_anchors, 1, 1).to(logits.device)
    anchor_h = torch.tensor([a[1] for a in anchors]).view(n_anchors, 1, 1).to(logits.device)
    
    # 4. Calcul du centre (x, y) en pixels.
    cx = (cols + p[:, 1]) * STRIDE
    cy = (rows + p[:, 2]) * STRIDE
    
    # 5. Calcul de la taille. On multiplie la valeur prédite par la taille de l'ancre, 
    # par la taille de l'image (64), ET par notre nouveau RATIO_MAX (3.0).
    w = p[:, 3] * RATIO_MAX * anchor_w * size
    h = p[:, 4] * RATIO_MAX * anchor_h * size
    
    # 6. Conversion du centre (cx, cy) vers les coins [x1, y1, x2, y2].
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
    
    # 7. On renvoie une longue liste plate de toutes les boîtes et de tous les scores.
    return boxes.reshape(-1, 4), p[:, 0].reshape(-1)

# --- CORRECTION MULTI-FIGURES (Limite 3) ---
class ShapeDataset(Dataset):
    """Prépare les images et génère la cible 15 canaux."""

    def __init__(self, folder: Path, size: int = 64, anchors=ANCHORS) -> None:
        self.paths = sorted(Path(folder).glob("*.png"))
        if not self.paths:
            raise FileNotFoundError(f"Aucune image dans {folder}.")
        self.grid = size // STRIDE
        self.anchors = anchors # On passe les ancres au dataset

    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[i]
        
        # L'image est chargée et convertie pour PyTorch (format NCHW)
        image = torch.from_numpy(
            np.asarray(Image.open(path).convert("RGB"), dtype=np.float32).copy()
        ).permute(2, 0, 1) / 255.0
        
        # CORRECTION : Lecture de TOUTES les figures (cercles) du fichier texte.
        lines = path.with_suffix(".txt").read_text().strip().splitlines()
        boxes = [tuple(map(float, line.split())) for line in lines]
        
        # On envoie la LISTE de cercles à notre nouvelle fonction code()
        target = code(boxes, size=image.shape[-1], anchors=self.anchors)
        return image, target

class Detector(nn.Module):
    def __init__(self, anchors=ANCHORS) -> None:
        super().__init__()
        self.anchors = anchors
        
        # Backbone (Extracteur de caractéristiques)
        blocks = [nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
                  nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                  nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU()]
        self.backbone = nn.Sequential(*blocks)
        
        # Tête de prédiction : Passe de 32 à 15 canaux (K*5).
        self.head = nn.Conv2d(32, 5 * len(anchors), kernel_size=1, stride=1)
        
    def feature_maps(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.feature_maps(images))

def compute_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calcule l'erreur du modèle. (Presque parfait dans ta version, j'ai juste nettoyé K)."""
    batch, channels, grid, _ = logits.shape
    n_anchors = channels // 5 # On déduit le nombre d'ancres (3) à partir des 15 canaux.
    
    # On découpe en [Batch, 3 ancres, 5 valeurs, 8, 8]
    logits = logits.reshape(batch, n_anchors, 5, grid, grid)
    target = target.reshape(batch, n_anchors, 5, grid, grid)
    
    # 1. Erreur sur la présence (BCE) calculée pour toutes les cases de toutes les ancres.
    obj_loss = nn.functional.binary_cross_entropy_with_logits(
        logits[:, :, 0], target[:, :, 0], pos_weight=torch.tensor(15.0))
        
    # 2. Masque : on isole les rares endroits où il y a vraiment un cercle.
    positive = target[:, :, 0] > 0.5 
    
    # Sécurité au cas où une image serait complètement vide.
    if not positive.any():
        return obj_loss
        
    # 3. On extrait les prédictions (après sigmoïde) et les cibles (canaux 1 à 4).
    # permute(0, 1, 3, 4, 2) sert juste à mettre les 4 valeurs à la fin pour la MSE.
    pred_box = torch.sigmoid(logits[:, :, 1:]).permute(0, 1, 3, 4, 2)[positive]
    true_box = target[:, :, 1:].permute(0, 1, 3, 4, 2)[positive]
    
    return obj_loss + 10.0 * nn.functional.mse_loss(pred_box, true_box)

def predict(logits: torch.Tensor, size: int = 64, anchors=ANCHORS,
            score_threshold: float = SCORE_THRESHOLD,
            iou_threshold: float = NMS_IOU_THRESHOLD) -> tuple[torch.Tensor, torch.Tensor]:
    """Renvoie les boîtes validées après score minimal et NMS."""
    boxes, scores = decode(logits, size, anchors)
    
    # 1. On vire tout ce qui a un score de confiance en dessous de notre seuil (50%).
    keep = scores > score_threshold
    boxes, scores = boxes[keep], scores[keep]
    
    if len(boxes) == 0:
        return torch.zeros((0, 4), device=logits.device), torch.zeros(0, device=logits.device)
        
    # 2. Le tri final : le NMS.
    # Un seuil de 0.45 limite les doublons tout en conservant les objets proches.
    kept = torch.tensor(nms(boxes, scores, iou_threshold), dtype=torch.long)
    
    return boxes[kept], scores[kept]