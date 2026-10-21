"""把追踪结果回渲染成 PNG，与原图并排比对（左=原图 右=矢量回渲染）"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracer import trace  # noqa: E402

import extract as E  # noqa: E402  复用同一套掩膜逻辑

OUT = r"D:/Javastudy/web-frontend/webmail/brand/_src/verify"
os.makedirs(OUT, exist_ok=True)

src = Image.open(E.SRC).convert("RGB")

jobs = []
box = (230, 170, 1040, 700)
m, _ = E.filter_mask(E.INK, box, 18, 0.35, (612, 637))
jobs.append(("monogram-ink", m, E.TOK_INK, box))
m, _ = E.filter_mask(E.YEL, box, 8, 0.5, (612, 637))
jobs.append(("monogram-yel", m, E.TOK_YEL, box))
box2 = (200, 718, 1060, 900)
m, _ = E.filter_mask(E.INK, box2, 18, 0.35)
jobs.append(("wordmark-ink", m, E.TOK_INK, box2))
m, _ = E.filter_mask(E.YEL, box2, 6, 0.4)
jobs.append(("wordmark-yel", m, E.TOK_YEL, box2))

PAPER = (231, 232, 229)
pad = 16
for name, mask, color, box in jobs:
    x0, y0, x1, y1 = box
    cw, ch = x1 - x0, y1 - y0
    orig = src.crop(box)
    polys = trace(mask, eps=1.0, min_area=60)
    canvas = Image.new("RGB", (cw, ch), PAPER)
    d = ImageDraw.Draw(canvas)
    items = sorted(polys, key=lambda p: -(abs(sum(p[i][0] * p[(i + 1) % len(p)][1] - p[(i + 1) % len(p)][0] * p[i][1] for i in range(len(p))) / 2)))
    for i, p in enumerate(items):
        pt = (p[0][0] - x0, p[0][1] - y0)
        dep = 0
        for j in range(i):
            q = [((v[0] - x0), (v[1] - y0)) for v in items[j]]
            c = False
            for k in range(len(q)):
                ax, ay = q[k]
                bx, by = q[(k + 1) % len(q)]
                if (ay > pt[1]) != (by > pt[1]) and pt[0] < (bx - ax) * (pt[1] - ay) / (by - ay + 1e-12) + ax:
                    c = not c
            dep += 1 if c else 0
        coords = [((v[0] - x0), (v[1] - y0)) for v in p]
        d.polygon(coords, fill=(PAPER if dep % 2 else tuple(int(color[1:][k:k + 2], 16) for k in (0, 2, 4))))
    sheet = Image.new("RGB", (cw * 2 + pad * 3, ch + pad * 2 + 22), "#FFFFFF")
    sheet.paste(orig, (pad, pad + 22))
    sheet.paste(canvas, (cw + pad * 2, pad + 22))
    dd = ImageDraw.Draw(sheet)
    dd.text((pad, 6), f"{name}  左=KV原图   右=矢量回渲染   ({len(polys)} 环)")
    sheet.save(os.path.join(OUT, f"{name}.png"))
    print("saved", name, sheet.size)
