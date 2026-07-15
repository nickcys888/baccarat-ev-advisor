import json
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "tg-cards"


def connected_components(mask):
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=np.uint8)
    components = []
    for start_y in range(height):
        for start_x in range(width):
            if not mask[start_y, start_x] or seen[start_y, start_x]:
                continue
            queue = deque([(start_y, start_x)])
            seen[start_y, start_x] = 1
            area = 0
            min_x = max_x = start_x
            min_y = max_y = start_y
            while queue:
                y, x = queue.pop()
                area += 1
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
                for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= next_y < height and 0 <= next_x < width:
                        if mask[next_y, next_x] and not seen[next_y, next_x]:
                            seen[next_y, next_x] = 1
                            queue.append((next_y, next_x))
            components.append({
                "x": min_x,
                "y": min_y,
                "width": max_x - min_x + 1,
                "height": max_y - min_y + 1,
                "area": area,
            })
    return components


def card_face_evidence(image, box, side):
    crop = image.crop((box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"]))
    rotations = [0] if box["width"] <= box["height"] * 0.95 else ([90, -90] if side == "player" else [-90, 90])
    best = None
    for rotation in rotations:
        card = crop.rotate(rotation, expand=True) if rotation else crop
        portrait_width = max(80, card.width)
        portrait_height = max(110, card.height)
        card = card.resize((portrait_width, portrait_height))
        pixels = np.asarray(card)
        low = pixels.min(axis=2)
        high = pixels.max(axis=2)
        pale_ratio = float(((low > 145) & ((high - low) < 95)).mean())
        luminance = pixels @ np.array([0.299, 0.587, 0.114])
        red_ink = (
            (pixels[:, :, 0] > 115)
            & (pixels[:, :, 0] > pixels[:, :, 1] * 1.3)
            & (pixels[:, :, 0] > pixels[:, :, 2] * 1.15)
        )
        ink = (luminance < 145) | red_ink
        corner_right = max(1, int(portrait_width * 0.38))
        rank_ink = int(ink[int(portrait_height * 0.02):int(portrait_height * 0.21), 2:corner_right].sum())
        suit_ink = int(ink[int(portrait_height * 0.24):int(portrait_height * 0.49), 2:corner_right].sum())
        valid = pale_ratio >= 0.32 and rank_ink >= 40 and suit_ink >= 40
        score = pale_ratio + min(rank_ink, 240) / 1200 + min(suit_ink, 240) / 1200
        candidate = {"valid": valid, "score": score, "rotation": rotation}
        if valid and (best is None or score > best["score"]):
            best = candidate
    return best


def detect_cards(source):
    image = source.convert("RGB") if isinstance(source, Image.Image) else Image.open(source).convert("RGB")
    source_width, source_height = image.size
    target_width = min(700, max(280, source_width))
    scale = target_width / source_width
    target_height = max(1, round(source_height * scale))
    analysis = np.asarray(image.resize((target_width, target_height)))
    low = analysis.min(axis=2)
    high = analysis.max(axis=2)
    mask = (low > 158) & ((high - low) < 82)
    boxes = []
    for component in connected_components(mask):
        aspect = component["width"] / component["height"]
        fill = component["area"] / (component["width"] * component["height"])
        if not (
            component["height"] >= target_height * 0.14
            and component["height"] <= target_height * 0.92
            and component["width"] >= target_height * 0.10
            and 0.34 <= aspect <= 1.55
            and fill >= 0.32
        ):
            continue
        boxes.append({
            "x": round(component["x"] / scale),
            "y": round(component["y"] / scale),
            "width": round(component["width"] / scale),
            "height": round(component["height"] / scale),
            "area": component["area"] / (scale * scale),
        })

    middle = source_width / 2
    result = {"player": [], "banker": []}
    for side in result:
        side_boxes = [box for box in boxes if (box["x"] + box["width"] / 2 < middle) == (side == "player")]
        candidates = []
        for box in side_boxes:
            if box["y"] + box["height"] / 2 <= source_height * 0.52:
                continue
            evidence = card_face_evidence(image, box, side)
            if evidence:
                candidates.append({**box, **evidence})
        result[side] = sorted(sorted(candidates, key=lambda box: box["area"], reverse=True)[:3], key=lambda box: box["x"])
    return result


def main():
    labels = json.loads((FIXTURES / "labels.json").read_text(encoding="utf-8"))
    failed = False
    for case in labels:
        detected = detect_cards(FIXTURES / case["file"])
        counts = {side: len(detected[side]) for side in ("player", "banker")}
        expected = {side: len(case[side]) for side in ("player", "banker")}
        passed = counts == expected
        failed |= not passed
        print(json.dumps({"file": case["file"], "passed": passed, "detected": counts, "expected": expected}, ensure_ascii=False))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
