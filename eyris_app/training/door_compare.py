#!/usr/bin/env python3
"""Doors, chairs and tables as the whole app reports them, before and after
the hazard model learned them (1.0.9).

    python door_compare.py collect                  # scene/world/road on the test photos
    python door_compare.py report <door> <chair> <table>   # hazard thresholds to try

"Before" is the app of 1.0.8: scene, world and road models with their
measured confidences (app_models.json), agreement between models, scene
rules, merging. "After" adds the hazard model's doors, chairs and tables
(cache of hazard_eval.py collect, HAZARD_TAG) at the given confidences, and
drops a wall box that holds a named door (preferDoorToWall in the app).

Test photos: the test half of hazard/ds that holds door, chair or table
boxes or comes from the door sets, ADE rooms and streets (every object
boxed), and the Open Images test photos checked for those objects. A
detection is right when it overlaps a labelled object of its name (IoU >=
0.3, or 70% inside it), as in hazard_eval.py.
"""
import os
import pickle
import sys
from collections import defaultdict

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app_sim import CONFIG, app_detections, esp32_frame, frame_for_models, load_models, run_model, to_labels  # noqa: E402
import hazard_eval  # noqa: E402

WORK = 'D:/dev/eyris_work'
CACHE = f'{WORK}/sim'
LABELS = ['Door', 'Chair', 'Table']
SOURCES = ('di_', 'dc_', 'dm_', 'neg_', 'oi_')


def items():
    out = []
    for name, path, truth in hazard_eval.test_items():
        if not name.startswith(SOURCES):
            continue
        scored = set(LABELS)
        if name.startswith('oi_'):
            scored = set(LABELS) & hazard_eval.oi_scored(name)
            if not scored:
                continue
        out.append((name, path, {l: b for l, b in truth.items() if l in LABELS}, scored))
    return out


def collect():
    models = load_models(['scene', 'world', 'road'])
    todo = items()
    for camera in ('phone', 'esp32'):
        out = {}
        for k, (name, path, _t, _s) in enumerate(todo):
            photo = Image.open(path).convert('RGB')
            if camera == 'esp32':
                photo = esp32_frame(photo, seed=k)
            small = frame_for_models(photo)
            out[name] = {m['spec']['tag']: to_labels(run_model(m, small), m['spec']) for m in models}
            if k % 300 == 0:
                print(camera, k, len(todo), flush=True)
        pickle.dump(out, open(f'{CACHE}/door_cmp_{camera}.pkl', 'wb'))


def door_over_wall(dets):
    doors = [b for _s, l, b in dets if l == 'Door']

    def inside_share(inner, outer):
        w = min(inner[2], outer[2]) - max(inner[0], outer[0])
        h = min(inner[3], outer[3]) - max(inner[1], outer[1])
        if w <= 0 or h <= 0:
            return 0.0
        return w * h / max((inner[2] - inner[0]) * (inner[3] - inner[1]), 1e-9)

    return [d for d in dets if d[1] != 'Wall' or not any(inside_share(b, d[2]) >= 0.5 for b in doors)]


def report(thresholds):
    specs = {s['tag']: s for s in CONFIG['models']}
    todo = items()
    for camera in ('phone', 'esp32'):
        cache = pickle.load(open(f'{CACHE}/door_cmp_{camera}.pkl', 'rb'))
        hazard = pickle.load(open(hazard_eval.cache_file(camera), 'rb'))
        res = {k: (defaultdict(int), defaultdict(int)) for k in ('before', 'after')}
        totals = defaultdict(int)
        walls = defaultdict(int)
        for name, _path, truth, scored in todo:
            for label in scored:
                totals[label] += len(truth.get(label, []))
            before = app_detections([(specs[t], cache[name][t]) for t in ('scene', 'world', 'road')])
            extra = [(s, l, b) for s, l, b in hazard.get(name, ((1, 1), []))[1]
                     if l in thresholds and s >= thresholds[l]]
            # DROP_OLD=Door: that label only from the hazard model afterwards.
            drop = set(filter(None, os.environ.get('DROP_OLD', '').split(',')))
            merged = sorted([d for d in before if d[1] not in drop] + extra, key=lambda d: -d[0])
            kept = []
            for s, l, b in merged:
                if not any(l == l2 and hazard_eval.iou(b, b2) > 0.5 for _, l2, b2 in kept):
                    kept.append((s, l, b))
            after = door_over_wall(kept)
            walls['before'] += sum(1 for d in before if d[1] == 'Wall')
            walls['after'] += sum(1 for d in after if d[1] == 'Wall')
            for key, dets in (('before', before), ('after', after)):
                rows = hazard_eval.judge([d for d in dets if d[1] in scored], truth)
                right, wrong = res[key]
                for label, _s, ok in rows:
                    (right if ok else wrong)[label] += 1
        print(f'\n{camera}: walls reported {walls["before"]} before, {walls["after"]} after')
        for label in LABELS:
            cells = []
            for key in ('before', 'after'):
                right, wrong = res[key]
                r, w = right[label], wrong[label]
                p = r / (r + w) if r + w else float('nan')
                cells.append(f'{key}: {r} right, {w} wrong ({p:.0%}), found {r / max(totals[label], 1):.0%}')
            print(f'  {label:6s} of {totals[label]:5d} | ' + ' | '.join(cells))


if __name__ == '__main__':
    if sys.argv[1] == 'collect':
        collect()
    else:
        report(dict(zip(LABELS, map(float, sys.argv[2:5]))))
