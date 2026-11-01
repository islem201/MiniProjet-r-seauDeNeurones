"""Générateur de données : plusieurs figures par image, avec chevauchements.

Étiquette : une ligne "cx cy w h" par figure (coordonnées en pixels).
Les figures sont dessinées dans l'ordre de leurs annotations.
"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

SHAPES = ("circle", "square", "triangle")
COLOR = (230, 75, 60)
OUTLINE = (25, 25, 25)


def draw_shape(draw, shape, box):
    x1, y1, x2, y2 = box

    if shape == "circle":
        draw.ellipse(box, fill=COLOR, outline=OUTLINE)
    elif shape == "square":
        draw.rectangle(box, fill=COLOR, outline=OUTLINE)
    else:
        draw.polygon(
            ((x1 + (x2 - x1) // 2, y1),
             (x2, y2),
             (x1, y2)),
            fill=COLOR,
            outline=OUTLINE,
        )


def make_example(size, shape, rng, min_objects = 2, max_objects = 4):
    background = rng.randint(225, 250)
    image = Image.new("RGB", (size, size), (background,) * 3)
    draw = ImageDraw.Draw(image)

    n_objects = rng.randint(min_objects, max_objects)
    boxes = []

    for i in range(n_objects):
        side = rng.randint(size // 6, size // 3)

        if i == 0:
            # Première figure : position aléatoire
            x1 = rng.randint(0, size - side - 1)
            y1 = rng.randint(0, size - side - 1)
        else:
            # Figures suivantes : placement proche de la précédente
            # pour favoriser le chevauchement
            px1, py1, px2, py2 = boxes[-1]
            prev_cx = (px1 + px2) / 2
            prev_cy = (py1 + py2) / 2

            jitter_x = rng.randint(
                -max(1, side // 3), max(1, side // 3)
            )
            jitter_y = rng.randint(
                -max(1, side // 3), max(1, side // 3)
            )

            x1 = round(prev_cx - side / 2 + jitter_x)
            y1 = round(prev_cy - side / 2 + jitter_y)

            x1 = max(0, min(x1, size - side - 1))
            y1 = max(0, min(y1, size - side - 1))

        box = (x1, y1, x1 + side, y1 + side)
        boxes.append(box)

        # Les figures dessinées plus tard peuvent masquer les précédentes
        draw_shape(draw, shape, box)

    labels = []
    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        labels.append(f"{cx} {cy} {w} {h}")

    return image, "\n".join(labels)


def generate(
    output: Path,
    size: int = 64,
    train: int = 400,
    val: int = 100,
    shape: str = "circle",
    min_objects: int = 2,
    max_objects: int = 4,
    seed: int = 7,
):
    if size % 8:
        raise ValueError("size doit être un multiple de 8.")
    if size < 24:
        raise ValueError("size doit être supérieur ou égal à 24.")
    if min_objects < 2 or max_objects < min_objects:
        raise ValueError("Il faut au moins 2 objets et max >= min.")

    rng = random.Random(seed)

    for split, total in (("train", train), ("val", val)):
        folder = output / split
        folder.mkdir(parents=True, exist_ok=True)

        for i in range(total):
            image, label = make_example(
                size, shape, rng, min_objects, max_objects
            )
            image.save(folder / f"{i:05d}.png")
            (folder / f"{i:05d}.txt").write_text(
                label + "\n", encoding="utf-8"
            )

    print(
        f"{train + val} images générées dans {output.resolve()}"
    )
    print(
        f"Figures : {shape}, de {min_objects} à {max_objects} par image"
    )


