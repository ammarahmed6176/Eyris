#!/usr/bin/env python3
"""Box the doors, chairs and tables in photos that were collected for other
objects, so the hazard model is not taught that they are nothing.

    python furniture_drafts.py draft <weights.pt>     # drafts, contact sheets of the ESP32 frames
    python furniture_drafts.py write [fixes.json]     # hazard/furniture_review.json

Photos: the ESP32 room frames of 249.mp4 that show furniture (left out of
the first 9-class training, see ROOM_FRAMES_EMPTY in build_hazard_dataset.py)
and the Objects365 / Wikimedia photos of the standing-AC set (rooms and
showrooms). Drafts come from the first 9-class model (hazard_i) at 0.40.
The ESP32 frames are checked by eye on the contact sheets: fixes.json maps a
frame name to the boxes to use instead ([[class, cx, cy, w, h], ...], may be
empty) or to "drop" to leave the frame out. The AC photos keep the drafts at
0.50 and above without a check.
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_hazard_dataset import H, ROOM_FRAMES_EMPTY  # noqa: E402

DRAFTS = f'{H}/furniture_drafts'
REVIEW = f'{H}/furniture_review.json'
ROOM_CLASSES = {6: 'door', 7: 'chair', 8: 'table'}


def norm(path):
    return os.path.abspath(path).replace('\\', '/')


def photos():
    rooms = f'{H}/client_249/crops'
    esp = [norm(f'{rooms}/{n}') for n in sorted(os.listdir(rooms))
           if int(n.rsplit('_', 1)[1][:-4]) // 25 not in ROOM_FRAMES_EMPTY]
    review = json.load(open(f'{H}/ac_review.json'))
    ac = [norm(p) for p in review['commons']]
    ac += [norm(f'{H}/o365/images/{n}') for n in review['o365_keep'] + review['o365_negative']]
    return esp, ac


def draft(weights):
    from ultralytics import YOLO
    model = YOLO(weights)
    esp, ac = photos()
    os.makedirs(f'{DRAFTS}/labels', exist_ok=True)
    out = {}
    for k in range(0, len(esp + ac), 32):
        batch = (esp + ac)[k:k + 32]
        for path, result in zip(batch, model.predict(batch, imgsz=416, conf=0.40, verbose=False,
                                                     classes=list(ROOM_CLASSES))):
            rows = []
            for (cx, cy, w, h), c, s in zip(result.boxes.xywhn.tolist(), result.boxes.cls.tolist(),
                                            result.boxes.conf.tolist()):
                rows.append([int(c), cx, cy, w, h, round(s, 2)])
            out[path] = rows
    json.dump(out, open(f'{DRAFTS}/drafts.json', 'w'))
    # Contact sheets of the ESP32 frames with their drafts (class id and confidence).
    for p in esp:
        with open(f'{DRAFTS}/labels/{os.path.basename(p)[:-4]}.txt', 'w') as f:
            for c, cx, cy, w, h, s in out[p]:
                # Shown on the sheet: D/C/T for door/chair/table and the confidence.
                f.write(f'{c} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f} {"DCT"[c - 6]}{round(s * 100)}\n')
    for i in range(0, len(esp), 40):
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'contact_sheet.py'),
                        '--labels', f'{DRAFTS}/labels', f'{DRAFTS}/esp_{i // 40}.jpg', '8', '240',
                        *esp[i:i + 40]], check=True)
    print(len(esp), 'ESP32 frames,', len(ac), 'AC photos;',
          sum(len(v) for v in out.values()), 'boxes drafted')


def write(fixes_path=None):
    drafts = json.load(open(f'{DRAFTS}/drafts.json'))
    fixes = json.load(open(fixes_path)) if fixes_path else {}
    esp, ac = photos()
    review = {}
    for p in esp:
        fix = fixes.get(os.path.basename(p))
        if fix == 'drop':
            continue
        review[p] = fix if fix is not None else [r[:5] for r in drafts[p]]
    for p in ac:
        review[p] = [r[:5] for r in drafts[p] if r[5] >= 0.50]
    json.dump(review, open(REVIEW, 'w'), indent=0)
    print(len(review), 'photos,', sum(len(v) for v in review.values()), 'boxes ->', REVIEW)


if __name__ == '__main__':
    if sys.argv[1] == 'draft':
        draft(sys.argv[2])
    else:
        write(*sys.argv[2:3])
