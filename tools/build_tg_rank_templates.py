import argparse
import base64
import json

import numpy as np
from PIL import Image

from check_tg_card_boxes import FIXTURES, ROOT, card_face_evidence, detect_cards


HTML = ROOT / "index.html"
SCRIPT_START = '  <script id="tg-card-patterns">\n'
SCRIPT_END = "\n  </script>\n  <script>\n    const ranks = ["
TARGET_WIDTH = 32
TARGET_HEIGHT = 24


def portrait_card(image, box):
    card = image.crop((
        box["x"],
        box["y"],
        box["x"] + box["width"],
        box["y"] + box["height"],
    ))
    if box["rotation"]:
        # Pillow and Canvas use opposite visual rotation directions.
        card = card.rotate(-box["rotation"], expand=True)
    size = (max(80, card.width), max(110, card.height))
    return card.resize(size, Image.Resampling.BILINEAR)


def rank_signature(card):
    left = card.width * 0.02
    top = card.height * 0.01
    right = left + card.width * 0.44
    bottom = top + card.height * 0.31
    crop = card.crop((left, top, right, bottom)).resize((244, 154), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (260, 170), "white")
    canvas.paste(crop, (8, 8))
    pixels = np.asarray(canvas)
    red = pixels[:, :, 0].astype(float)
    green = pixels[:, :, 1].astype(float)
    blue = pixels[:, :, 2].astype(float)
    luminance = red * 0.299 + green * 0.587 + blue * 0.114
    ink = (luminance < 185) | ((red > 120) & (red > green * 1.35) & (red > blue * 1.2))
    ys, xs = np.where(ink)
    if len(xs) < 80:
        return None
    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    signature = []
    for target_y in range(TARGET_HEIGHT):
        source_y = min(max_y, int(min_y + (target_y + 0.5) * height / TARGET_HEIGHT))
        for target_x in range(TARGET_WIDTH):
            source_x = min(max_x, int(min_x + (target_x + 0.5) * width / TARGET_WIDTH))
            signature.append(1 if ink[source_y, source_x] else 0)
    filled = sum(signature)
    if filled < 6 or filled > len(signature) * 0.62:
        return None
    return signature


def distance(left, right):
    return sum(a != b for a, b in zip(left, right)) / len(left)


def encode_signature(signature):
    packed = bytearray((len(signature) + 7) // 8)
    for index, value in enumerate(signature):
        if value:
            packed[index >> 3] |= 1 << (7 - (index & 7))
    return base64.b64encode(packed).decode("ascii")


def collect_templates():
    labels = json.loads((FIXTURES / "labels.json").read_text(encoding="utf-8"))
    templates = {}
    samples = []
    for case in labels:
        path = FIXTURES / case["file"]
        image = Image.open(path).convert("RGB")
        detected = detect_cards(image)
        scale = min(1, 1400 / max(image.size))
        if scale < 1:
            image = image.resize(
                (round(image.width * scale), round(image.height * scale)),
                Image.Resampling.BILINEAR,
            )
            detected = {
                side: [{
                    **box,
                    "x": round(box["x"] * scale),
                    "y": round(box["y"] * scale),
                    "width": round(box["width"] * scale),
                    "height": round(box["height"] * scale),
                } for box in boxes]
                for side, boxes in detected.items()
            }
        for side in ("player", "banker"):
            if len(detected[side]) != len(case[side]):
                raise RuntimeError(f"{case['file']} {side}: card count does not match labels")
            for box, key in zip(detected[side], case[side]):
                signature = rank_signature(portrait_card(image, box))
                if not signature:
                    raise RuntimeError(f"{case['file']} {side} {key}: rank signature is empty")
                samples.append((case["file"], side, key, signature))
                rank_samples = templates.setdefault(key, [])
                if not any(distance(signature, existing) < 0.01 for existing in rank_samples):
                    rank_samples.append(signature)
    return templates, samples


def classify(signature, templates):
    ranked = []
    for key, rank_samples in templates.items():
        ranked.append((min(distance(signature, sample) for sample in rank_samples), key))
    ranked.sort()
    best_distance, best_key = ranked[0]
    next_distance = next((value for value, key in ranked[1:] if key != best_key), 1.0)
    return best_key, best_distance, next_distance - best_distance


def fallback_boxes(image):
    width, height = image.size

    def make(fraction, horizontal=False):
        card_width = width * (0.073 if horizontal else 0.052)
        card_height = height * (0.18 if horizontal else 0.25)
        return {
            "x": round(width * fraction - card_width / 2),
            "y": round(height * (0.76 if horizontal else 0.685)),
            "width": round(card_width),
            "height": round(card_height),
            "area": card_width * card_height,
        }

    return {
        "player": [make(0.116, True), make(0.203), make(0.274)],
        "banker": [make(0.728), make(0.799), make(0.881, True)],
    }


def map_boxes_to_slots(image, boxes):
    result = {"player": {}, "banker": {}}
    expected = {
        "player": [0.116, 0.203, 0.274],
        "banker": [0.728, 0.799, 0.881],
    }
    for side in result:
        candidates = []
        for box in boxes[side]:
            if box["y"] + box["height"] / 2 <= image.height * 0.52:
                continue
            evidence = card_face_evidence(image, box, side)
            if evidence:
                candidates.append({**box, **evidence})
        candidates = sorted(candidates, key=lambda box: box["area"], reverse=True)[:3]
        candidates.sort(key=lambda box: box["x"])
        has_horizontal = any(box["width"] > box["height"] * 0.95 for box in candidates)
        available = {0, 1, 2}
        for position, box in enumerate(candidates):
            index = position
            if has_horizontal:
                center = (box["x"] + box["width"] / 2) / image.width
                index = min(available, key=lambda slot: abs(center - expected[side][slot]))
            available.discard(index)
            result[side][index] = box
    return result


def runtime_image(path):
    image = Image.open(path).convert("RGB")
    scale = min(1, 1400 / max(image.size))
    if scale < 1:
        image = image.resize(
            (round(image.width * scale), round(image.height * scale)),
            Image.Resampling.BILINEAR,
        )
    return image


def check_runtime_cases(templates):
    labels = json.loads((FIXTURES / "labels.json").read_text(encoding="utf-8"))
    failed = False
    for case in labels:
        path = FIXTURES / case["file"]
        image = runtime_image(path)
        detected = map_boxes_to_slots(image, detect_cards(image))
        fallback = map_boxes_to_slots(image, fallback_boxes(image))
        actual = {"player": [], "banker": []}
        for side in actual:
            for index in range(3):
                candidates = []
                if index in detected[side]:
                    candidates.append(detected[side][index])
                if index in fallback[side]:
                    candidates.append(fallback[side][index])
                ranked = []
                for box in candidates:
                    signature = rank_signature(portrait_card(image, box))
                    if not signature:
                        continue
                    key, best_distance, margin = classify(signature, templates)
                    if best_distance <= 0.05 and margin >= 0.035:
                        ranked.append((best_distance, key))
                actual[side].append(min(ranked)[1] if ranked else "")
            while actual[side] and not actual[side][-1]:
                actual[side].pop()
        expected = {side: case[side] for side in ("player", "banker")}
        passed = actual == expected
        failed |= not passed
        print(json.dumps({
            "runtime_file": case["file"],
            "expected": expected,
            "recognized": actual,
            "passed": passed,
        }, ensure_ascii=False))
    return failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    templates, samples = collect_templates()
    failed = False
    for file_name, side, expected, signature in samples:
        predicted, best_distance, margin = classify(signature, templates)
        passed = predicted == expected and best_distance <= 0.18 and margin >= 0.035
        failed |= not passed
        print(json.dumps({
            "file": file_name,
            "side": side,
            "expected": expected,
            "predicted": predicted,
            "distance": round(best_distance, 4),
            "margin": round(margin, 4),
            "passed": passed,
        }, ensure_ascii=False))
    missing = sorted(set("A23456789TJQK") - set(templates))
    if missing:
        print(json.dumps({"missing_ranks": missing}, ensure_ascii=False))
        failed = True
    failed |= check_runtime_cases(templates)
    if args.write and not failed:
        encoded = {
            key: [encode_signature(signature) for signature in signatures]
            for key, signatures in sorted(templates.items())
        }
        assignment = (
            "    window.TG_PRESET_RANK_TEMPLATES = "
            + json.dumps(encoded, ensure_ascii=True, separators=(",", ":"))
            + ";"
        )
        html = HTML.read_text(encoding="utf-8")
        start = html.index(SCRIPT_START) + len(SCRIPT_START)
        end = html.index(SCRIPT_END, start)
        HTML.write_text(html[:start] + assignment + html[end:], encoding="utf-8", newline="\n")
        print(f"updated {HTML} ({sum(map(len, encoded.values()))} templates)")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
