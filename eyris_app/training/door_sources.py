#!/usr/bin/env python3
"""Door, chair and table photos for the hazard model (1.0.9), read from the
three Kaggle sets in D:/dev/eyris_work/doors:

  - thepbordin/indoor-object-detection: rooms and corridors seen from about
    waist height (door, cabinetDoor, refrigeratorDoor, window, chair, table,
    cabinet, couch, openedDoor, pole). Its valid and test folders are test.
  - cubeai/door-detection-for-yolov8: doors, one class. Its valid folder is
    test.
  - muftirestumahesa/obstacles-in-public-spaces-for-dist-yolo: obstacles for
    blind people in public places (Indonesia), boxes in _annotations.txt as
    "x1,y1,x2,y2,class,distance". Test by a hash of the name.

    python door_sources.py count        # photos and boxes per class
    python door_sources.py sheet <kind> <out.jpg> [n]   # look at them

Every source yields (image path, rows, split) with rows [class, cx, cy, w, h]
in the hazard model's numbering (DOOR, CHAIR, TABLE below).
"""
import hashlib
import os
import sys
from collections import Counter

from PIL import Image

ROOT = 'D:/dev/eyris_work/doors'
DOOR, CHAIR, TABLE = 6, 7, 8
INDOOR = {'0': DOOR, '8': DOOR, '4': CHAIR, '5': TABLE}     # door, openedDoor, chair, table
# As trained for 1.0.9 (run hazard_k): doors and chairs only. 13 of its
# training photos also show poles, 1 a pothole; boxing those too (8: 0,
# 20: 4) would be right for a later run.
MUFTI = {15: DOOR, 7: CHAIR}
MUFTI_WANTED = {15, 7}


def test_share(name, share):
    base = os.path.basename(name).split('.rf.')[0].split('_jpg')[0].split('_png')[0]
    return int(hashlib.md5(base.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < share


def read(path):
    if not os.path.exists(path):
        return []
    return [l.split() for l in open(path) if l.strip()]


def image_for(folder, label_name):
    stem = label_name[:-4]
    for ext in ('.jpg', '.jpeg', '.png', '.JPG'):
        p = f'{folder}/{stem}{ext}'
        if os.path.exists(p):
            return p
    return None


def indoor():
    root = f'{ROOT}/indoor-object-detection'
    for split in ('train', 'valid', 'test'):
        seen = set()
        for n in sorted(os.listdir(f'{root}/{split}/labels')):
            base = n.split('.rf.')[0]
            image = image_for(f'{root}/{split}/images', n)
            if base in seen or image is None:
                continue
            seen.add(base)
            rows = [[INDOOR[r[0]]] + r[1:5] for r in read(f'{root}/{split}/labels/{n}') if r[0] in INDOOR]
            yield image, rows, 'train' if split == 'train' else 'test', 'di_'


def cube():
    root = f'{ROOT}/door-detection-for-yolov8'
    for split in ('train', 'valid'):
        seen = set()
        for n in sorted(os.listdir(f'{root}/{split}/labels')):
            base = n.split('.rf.')[0]
            image = image_for(f'{root}/{split}/images', n)
            if base in seen or image is None:
                continue
            seen.add(base)
            rows = [[DOOR] + r[1:5] for r in read(f'{root}/{split}/labels/{n}') if r[0] == '0']
            yield image, rows, 'train' if split == 'train' else 'test', 'dc_'


def mufti():
    root = f'{ROOT}/obstacles-in-public-spaces-for-dist-yolo'
    seen = set()
    for line in open(f'{root}/_annotations.txt', encoding='utf-8'):
        parts = line.split()
        if not parts:
            continue
        name, boxes = parts[0], parts[1:]
        base = name.split('.rf.')[0]
        image = f'{root}/{name}'
        if base in seen or not os.path.exists(image):
            continue
        seen.add(base)
        rows = []
        mine = [b.split(',') for b in boxes]
        if not any(int(b[4]) in MUFTI_WANTED for b in mine):
            continue        # only photos with a door or a chair are of use here
        w, h = Image.open(image).size
        for b in mine:
            c = int(b[4])
            if c in MUFTI:
                x1, y1, x2, y2 = map(float, b[:4])
                rows.append([MUFTI[c], (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h])
        yield image, rows, 'test' if test_share(name, 0.2) else 'train', 'dm_'


SOURCES = {'indoor': indoor, 'cube': cube, 'mufti': mufti}


def main():
    if sys.argv[1] == 'count':
        for kind, source in SOURCES.items():
            photos, boxes = Counter(), Counter()
            for _image, rows, split in ((i, r, s) for i, r, s, _ in source()):
                photos[split] += 1
                for r in rows:
                    boxes[(split, r[0])] += 1
            print(kind, dict(photos), dict(sorted(boxes.items())))
    elif sys.argv[1] == 'sheet':
        import random
        import subprocess
        kind, out = sys.argv[2], sys.argv[3]
        n = int(sys.argv[4]) if len(sys.argv) > 4 else 48
        items = list(SOURCES[kind]())
        random.Random(1).shuffle(items)
        tmp = f'{ROOT}/_sheet_{kind}'
        os.makedirs(tmp, exist_ok=True)
        paths = []
        for k, (image, rows, _split, _p) in enumerate(items[:n]):
            dst = f'{tmp}/{k:03d}.jpg'
            Image.open(image).convert('RGB').save(dst)
            with open(f'{tmp}/{k:03d}.txt', 'w') as f:
                for c, cx, cy, w, h in rows:
                    f.write(f'{c} {cx} {cy} {w} {h}\n')
            paths.append(dst)
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'contact_sheet.py'),
                        '--labels', tmp, out, '8', '200', *paths], check=True)


if __name__ == '__main__':
    main()
