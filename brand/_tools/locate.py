"""元素自动定位：连通域 → 包围盒 → 裁切参考图"""
import json
import os
from collections import deque

import numpy as np
from PIL import Image

SRC = r"D:/Javastudy/web-frontend/webmail/brand/_src/key-visual.jpg"
OUT = r"D:/Javastudy/web-frontend/webmail/brand/_src/crops"
os.makedirs(OUT, exist_ok=True)

img = Image.open(SRC).convert("RGB")
W, H = img.size
a = np.asarray(img).astype(np.int16)

r, g, b = a[..., 0], a[..., 1], a[..., 2]
mx, mn = a.max(2).astype(float), a.min(2).astype(float)
sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1.0), 0.0)

MASK = {
    "yellow": (sat > 0.42) & (r > 150) & (b < 150) & (g > 120),
    "ink": (mx < 120),
    "gray": (sat < 0.10) & (mx >= 120) & (mx < 205),   # 引导圆等中灰
}


def label(mask, downsample=6, min_cells=3):
    """降采样 + BFS 连通域，返回原图尺度的包围盒列表"""
    yr, xr = mask.shape
    h, w = yr // downsample, xr // downsample
    small = mask[: h * downsample, : w * downsample].reshape(h, downsample, w, downsample).mean((1, 3)) > 0.28
    seen = np.zeros_like(small, dtype=bool)
    out = []
    for sy in range(h):
        for sx in range(w):
            if not small[sy, sx] or seen[sy, sx]:
                continue
            q = deque([(sy, sx)])
            seen[sy, sx] = True
            cells = []
            while q:
                cy, cx = q.popleft()
                cells.append((cy, cx))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and small[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            q.append((ny, nx))
            if len(cells) < min_cells:
                continue
            ys_ = [c[0] for c in cells]
            xs_ = [c[1] for c in cells]
            x0, x1 = min(xs_) * downsample, (max(xs_) + 1) * downsample
            y0, y1 = min(ys_) * downsample, (max(ys_) + 1) * downsample
            out.append({
                "bbox": [int(x0), int(y0), int(x1), int(y1)],
                "w": int(x1 - x0), "h": int(y1 - y0),
                "cells": len(cells),
                "px_area": int(mask[y0:y1, x0:x1].sum()),
            })
    out.sort(key=lambda d: -d["px_area"])
    return out


result = {}
for name, m in MASK.items():
    comps = label(m)
    result[name] = comps
    print(f"\n=== {name}  共 {len(comps)} 个连通域，占画布 {m.sum()/(W*H)*100:.2f}% ===")
    for c in comps[:22]:
        x0, y0, x1, y1 = c["bbox"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        print(f"  bbox({x0:>4},{y0:>4})-({x1:>4},{y1:>4})  {c['w']:>4}x{c['h']:<4} "
              f"中心({cx:>6.1f},{cy:>6.1f})  占比{ c['w']*c['h']/(W*H)*100:5.2f}%  实心{c['px_area']:>7}")

# ---- 精确底色（众数） ----
flat = a.reshape(-1, 3)
key = (flat // 2).astype(np.int32)
key = key[:, 0] * 100000 + key[:, 1] * 200 + key[:, 2]
vals, counts = np.unique(key, return_counts=True)
top = vals[np.argsort(-counts)[:6]]
print("\n=== 底色众数 (量化步长2) ===")
for v in top:
    m = key == v
    c = flat[m].mean(0)
    print("  #%02X%02X%02X  %5.2f%%" % (*[int(round(x)) for x in c], m.sum() / len(flat) * 100))

# ---- 黄像素精修（只取高饱和核心） ----
ym = MASK["yellow"]
core = a[ym]
core = core[core.max(1) - core.min(1) > 150]
print("\n=== 黄核心均值 ===")
cm = core.mean(0)
print("  #%02X%02X%02X  rgb(%s)  n=%d" % (*[int(round(x)) for x in cm],
                                            ", ".join(f"{x:.1f}" for x in cm), len(core)))
mxq, mnq = cm.max(), cm.min()
print(f"  HSV: H={(60*((cm[1]-mnq)/max(mxq-mnq,1e-6)) if cm[0]==mxq else 0):.1f}°  "
      f"V={mxq/255*100:.1f}%  S={(mxq-mnq)/mxq*100:.1f}%")

# ---- 墨色精修 ----
ik = a[MASK["ink"]]
ikd = ik[ik.sum(1) <= np.percentile(ik.sum(1), 12)]
c = ikd.mean(0)
print("\n=== 墨核心均值 ===")
print("  #%02X%02X%02X  rgb(%s)   G-R=%+.2f  B-R=%+.2f" % (*[int(round(x)) for x in c],
      ", ".join(f"{x:.1f}" for x in c), c[1] - c[0], c[2] - c[0]))

json.dump(result, open(r"D:/Javastudy/web-frontend/webmail/brand/_src/components.json", "w"),
          ensure_ascii=False, indent=1)

# ---- 裁切参考图 ----
print("\n=== 裁切参考 ===")
for name, m in MASK.items():
    for i, c in enumerate(result[name][:6]):
        x0, y0, x1, y1 = c["bbox"]
        if c["px_area"] < 400:
            continue
        pad = 10
        box = (max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad))
        fn = os.path.join(OUT, f"{name}-{i:02d}-{c['w']}x{c['h']}.png")
        img.crop(box).save(fn)
        print("  ", os.path.basename(fn))
