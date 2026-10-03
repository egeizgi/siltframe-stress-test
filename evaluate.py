"""Siltframe stress-test pack — score your model and print its failure map.

1. Run your model on every image in the pack and save one PNG per image, same relative path,
   same file name (.png), pixel value = predicted class id (see classes.csv, column `id`).
       clean/images/017.jpg              ->  <pred>/clean/images/017.png
       degraded/night/s2/images/017.jpg  ->  <pred>/degraded/night/s2/images/017.png
   Use --index if your PNGs hold class *indices* 0..18 (row order of classes.csv) instead of ids.
2. python evaluate.py --pred <pred>            (needs numpy + Pillow only)

Output: a table of mIoU per condition and severity, the relative drop against clean frames, traversable-ground IoU (dirt, grass, asphalt, concrete),
the classes that fail first — and failure_map.json with all of it. Void pixels (id 0) are never scored: these are
unlabelled pixels, or pixels a dense veil makes physically invisible.

MIT licence. https://siltframe.com
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="folder with your predicted PNGs, mirroring the pack layout")
    ap.add_argument("--pack", default=str(HERE), help="pack folder (default: where this script is)")
    ap.add_argument("--index", action="store_true", help="predictions hold class indices 0..18 instead of ids")
    ap.add_argument("--out", default="failure_map.json")
    a = ap.parse_args()
    pack, pred = Path(a.pack), Path(a.pred)

    classes = list(csv.DictReader(open(pack / "classes.csv", encoding="utf-8")))
    ids = [int(c["id"]) for c in classes]
    names = [c["name"] for c in classes]
    trav = [i for i, c in enumerate(classes) if c["traversable"] == "1"]
    n = len(ids)
    lut = np.full(256, 255, np.uint8)               # label id -> index, void -> 255
    for i, k in enumerate(ids):
        lut[k] = i

    rows = list(csv.DictReader(open(pack / "manifest.csv", encoding="utf-8")))
    cms, missing = defaultdict(lambda: np.zeros((n, n), np.int64)), 0
    for r in rows:
        p = pred / Path(r["image"]).with_suffix(".png")
        if not p.exists():
            missing += 1
            continue
        gt = lut[np.array(Image.open(pack / r["label"]))]
        pr = np.array(Image.open(p))
        if pr.ndim == 3:
            pr = pr[..., 0]
        pr = pr if a.index else lut[pr]
        if pr.shape != gt.shape:
            raise SystemExit(f"{p}: prediction is {pr.shape[::-1]}, label is {gt.shape[::-1]} (W x H) — predict at full size")
        m = (gt != 255) & (pr < n)
        key = r["condition"] if r["condition"] in ("clean", "real_rain") else f'{r["condition"]}/s{r["severity"]}'
        cms[key] += np.bincount(gt[m].astype(np.int64) * n + pr[m], minlength=n * n).reshape(n, n)
    if missing:
        print(f"warning: {missing} of {len(rows)} predictions missing — those images are skipped")
    if "clean" not in cms:
        raise SystemExit("no predictions for clean/ — they are the reference for every drop")

    present = cms["clean"].sum(1) > 0

    def score(cm):
        tp = np.diag(cm).astype(float)
        iou = tp / np.maximum(cm.sum(0) + cm.sum(1) - tp, 1)
        d = np.zeros(n, bool); d[trav] = True
        dtp = cm[np.ix_(d, d)].sum()
        div = dtp / max(cm[d].sum() + cm[:, d].sum() - dtp, 1)
        return {"miou": float(iou[present].mean() * 100), "traversable_iou": float(div * 100),
                "iou": {names[i]: float(iou[i] * 100) for i in range(n) if present[i]}}

    res = {k: score(cm) for k, cm in cms.items()}
    base = res["clean"]
    for k, v in res.items():
        v["drop_pct"] = (base["miou"] - v["miou"]) / max(base["miou"], 1e-6) * 100
        v["worst_classes"] = sorted(((c, base["iou"][c] - x) for c, x in v["iou"].items()), key=lambda t: -t[1])[:3]

    order = ["clean"] + sorted(k for k in res if "/" in k) + (["real_rain"] if "real_rain" in res else [])
    def pct(d):   # a mild corruption can land slightly above clean; show that as a gain, not as a negative drop
        return ("−%.0f%%" % d) if d >= 0 else ("+%.0f%%" % -d)

    print(f"\n{'set':16s} {'mIoU':>6s} {'drop':>7s} {'traversable':>11s}   classes that fail first")
    for k in order:
        v = res[k]
        worst = ", ".join(f"{c} −{d:.0f}" for c, d in v["worst_classes"] if d > 0) if k != "clean" else ""
        drop = "—" if k == "clean" else pct(v["drop_pct"])
        print(f"{k:16s} {v['miou']:6.1f} {drop:>7s} {v['traversable_iou']:11.1f}   {worst}")
    conds = sorted({k.split("/")[0] for k in res if "/" in k})
    summary = {c: float(np.mean([res[k]["drop_pct"] for k in res if k.startswith(c + "/")])) for c in conds}
    print("\nmean drop per condition: " + "  ".join(f"{c} {pct(d)}" for c, d in sorted(summary.items(), key=lambda t: -t[1])))
    json.dump({"per_set": res, "mean_drop_per_condition": summary}, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
