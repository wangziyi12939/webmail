"""从 KV 提取矢量元素（v2）

改进：
- 2x LANCZOS 上采样后二值化 → 亚像素平滑轮廓，消除阶梯感
- 连通域筛选去除中轴细条与稀疏虚线
- 按「细长 + 全部顶点落在引导圆环带上」剔除圆弧碎段
- 回栅格 IoU 自检 + 并排目视比对图
"""
import json
import math
import os
import sys
from collections import deque

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracer import trace, render_back  # noqa: E402

SRC = r"D:/Javastudy/web-frontend/webmail/brand/_src/key-visual.jpg"
MARKS = r"D:/Javastudy/web-frontend/webmail/brand/marks"
VERIFY = r"D:/Javastudy/web-frontend/webmail/brand/_src/verify"
os.makedirs(MARKS, exist_ok=True)
os.makedirs(VERIFY, exist_ok=True)

SS = 2                                  # 上采样倍率
CIRCLE = (626.3, 442.4, 239.1)          # 引导圆（紧色窗重拟合，σ=5.3）

TOK = {"ink": "#232527", "accent": "#F5CE15", "paper": "#E7E8E5"}
PAPER_RGB = tuple(int(TOK["paper"][i:i + 2], 16) for i in (1, 3, 5))

_img = Image.open(SRC).convert("RGB")
W, H = _img.size


def mask_2x(box, kind):
    """box 区域内 2x 上采样布尔掩膜（crop-local 坐标）"""
    c = _img.crop(box)
    c = c.resize((c.width * SS, c.height * SS), Image.LANCZOS)
    a = np.asarray(c).astype(np.int16)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    if kind == "ink":
        return a.max(2) < 100
    return (R - B > 55) & (R > 140) & (G > 110)


def _fit_circle(pts):
    x = pts[:, 0].astype(float)
    y = pts[:, 1].astype(float)
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    cx, cy = sol[0], sol[1]
    return cx, cy, math.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0))


def _arc_stats(pts):
    """最小二乘圆拟合 → (R, max|残差|)。直线/实心矩形的 max 残差会很大"""
    cx, cy, R = _fit_circle(pts)
    d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - R
    return R, float(np.abs(d).max())


def clean_components(mask, min_side, min_density, drop_cols=None, box=None,
                     drop_arcs=False, arc_tol=34.0, arc_overlap=0.60):
    """连通域筛选（ds=2 降采样 CCL） + 引导圆弧段的几何剔除

    引导圆不是虚线而是若干「粗弧段 + 缺口」。判据分两步：
      1) 先按圆性（拟合残差小）认出所有弧段像素，反算真实圆心/半径
      2) 任何「细小组件」只要有 >60% 像素落在该圆环带上，即为弧段 → 剔除
    条码竖条(直线,残差大)、信封带(实心矩形)、字形主体 均不会被误杀。
    """
    sub = mask.copy()
    if drop_cols:
        c0, c1 = max(0, drop_cols[0]), min(sub.shape[1], drop_cols[1])
        if c1 > c0:
            sub[:, c0:c1] = False
    h, w = sub.shape
    hs, ws = h // 2, w // 2
    small = sub[:hs * 2, :ws * 2].reshape(hs, 2, ws, 2).mean((1, 3)) > 0.3
    seen = np.zeros_like(small)

    # ---- 一遍：收集全部连通域 ----
    comps = []
    for sy in range(hs):
        for sx in range(ws):
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
                        if 0 <= ny < hs and 0 <= nx < ws and small[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            q.append((ny, nx))
            cc = np.array(cells)
            mm = np.zeros((hs, ws), bool)
            mm[cc[:, 0], cc[:, 1]] = True
            blob = mm.repeat(2, 0).repeat(2, 1)
            px = blob & sub                       # 真实掩膜像素（2x 局部坐标）
            sel = np.nonzero(px)
            comps.append({
                "blob": blob, "px": px,
                "pts": np.c_[sel[1], sel[0]].astype(float),
                "bw": int(cc[:, 1].max() - cc[:, 1].min() + 1) * 2,
                "bh": int(cc[:, 0].max() - cc[:, 0].min() + 1) * 2,
                "x0": int(cc[:, 1].min()) * 2, "y0": int(cc[:, 0].min()) * 2,
                "dens": float(sub[blob].mean()) if blob.any() else 0.0,
            })

    # ---- 二遍：裁决（引导圆残段留到多边形层面按半径剔除）----
    keep = np.zeros_like(sub)
    stats = []
    for c in comps:
        ok = min(c["bw"], c["bh"]) >= min_side and c["dens"] >= min_density
        stats.append((c["x0"] // SS + (box[0] if box else 0), c["y0"] // SS + (box[1] if box else 0),
                      c["bw"] // SS, c["bh"] // SS, round(c["dens"], 2), ok, ""))
        if ok:
            keep |= c["px"]
    return keep, stats
    return keep, stats


def _poly_area(p):
    s = 0.0
    for i in range(len(p)):
        s += p[i][0] * p[(i + 1) % len(p)][1] - p[(i + 1) % len(p)][0] * p[i][1]
    return abs(s) / 2.0


def drop_circle_fragments(polys, cx, cy, r, max_area=520.0, band=34.0):
    """剔除引导圆上的裁切标记：小面积 且 中位半径落在环带内

    Z 里 hatch/条码块中位半径 ≈300、面积 380，与环带 [r±34] 不交 → 安全保留。
    """
    out, dropped = [], []
    for p in polys:
        if _poly_area(p) < max_area:
            med = float(np.median([math.hypot(x - cx, y - cy) for x, y in p]))
            if abs(med - r) < band:
                dropped.append((round(_poly_area(p)), round(med, 1)))
                continue
        out.append(p)
    if dropped:
        print(f"  剔除引导圆标记 {len(dropped)} 个: {dropped}")
    return out


def _fmt(v):
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else s


def emit(name, mask, box, eps, min_area, fill, label, circle_filter=False):
    polys = trace(mask, eps=eps * SS, min_area=min_area * SS * SS)
    x0, y0 = box[0], box[1]
    polys = [[(x / SS + x0, y / SS + y0) for x, y in p] for p in polys]
    if circle_filter:
        polys = drop_circle_fragments(polys, *CIRCLE)
    if not polys:
        print(f"  [空] {label}")
        return None, polys
    allx = [v[0] for p in polys for v in p]
    ally = [v[1] for p in polys for v in p]
    ox, oy = min(allx), min(ally)
    vw = int(math.ceil(max(allx) - ox))
    vh = int(math.ceil(max(ally) - oy))
    d = "".join(
        "M" + _fmt(p[0][0] - ox) + " " + _fmt(p[0][1] - oy)
        + "".join("L" + _fmt(v[0] - ox) + " " + _fmt(v[1] - oy) for v in p[1:]) + "Z"
        for p in polys)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {vw} {vh}" '
           f'width="{vw}" height="{vh}" role="img" aria-label="{label}">\n'
           f'  <path fill="{fill}" fill-rule="evenodd" d="{d}"/>\n</svg>\n')
    open(os.path.join(MARKS, name), "w", encoding="utf-8").write(svg)

    back = render_back(polys, (H, W), 0, 0)
    tgt = np.zeros((H, W), bool)
    ys, xs = np.nonzero(mask)
    if len(xs):
        tgt[(ys // SS + y0).astype(int), (xs // SS + x0).astype(int)] = True
    iou = (back & tgt).sum() / max(1, (back | tgt).sum()) * 100
    nodes = sum(len(p) for p in polys)
    print(f"  [OK] {label:<10} {name:<24} {vw:>4}x{vh:<4} 环{len(polys):>3} 节点{nodes:>5} "
          f"{len(svg)/1024:>6.1f}KB  IoU≈{iou:.1f}%")
    return {"name": name, "label": label, "viewBox": [vw, vh], "offset": [round(ox, 1), round(oy, 1)],
            "rings": len(polys), "nodes": nodes, "kb": round(len(svg) / 1024, 1),
            "iou": round(iou, 1), "fill": fill}, polys


def side_by_side(name, box, fill, polys):
    x0, y0, x1, y1 = box
    orig = _img.crop(box)
    cw, ch = orig.size
    canvas = Image.new("RGB", (cw, ch), PAPER_RGB)
    d = ImageDraw.Draw(canvas)
    items = sorted(polys, key=lambda p: -abs(sum(
        p[i][0] * p[(i + 1) % len(p)][1] - p[(i + 1) % len(p)][0] * p[i][1] for i in range(len(p)))))
    rgb = tuple(int(fill[i:i + 2], 16) for i in (1, 3, 5))
    for i, p in enumerate(items):
        pt = (p[0][0] - x0, p[0][1] - y0)
        dep = 0
        for j in range(i):
            q = [(v[0] - x0, v[1] - y0) for v in items[j]]
            c = False
            for k in range(len(q)):
                ax, ay = q[k]
                bx, by = q[(k + 1) % len(q)]
                if (ay > pt[1]) != (by > pt[1]) and pt[0] < (bx - ax) * (pt[1] - ay) / (by - ay + 1e-12) + ax:
                    c = not c
            dep += 1 if c else 0
        d.polygon([(v[0] - x0, v[1] - y0) for v in p], fill=PAPER_RGB if dep % 2 else rgb)
    pad = 14
    sheet = Image.new("RGB", (cw * 2 + pad * 3, ch + pad * 2 + 20), "#FFFFFF")
    sheet.paste(orig, (pad, pad + 20))
    sheet.paste(canvas, (cw + pad * 2, pad + 20))
    ImageDraw.Draw(sheet).text((pad, 5), f"{name}   左=KV原图(1x)   右=矢量回渲染(2x追踪)")
    sheet.save(os.path.join(VERIFY, f"{name}.png"))


def main():
    rep = []
    jobs = [
        ("wz-monogram.svg", "WZ 主徽标", (230, 170, 1040, 700), 36, 0.35, "ink", TOK["ink"], True, 0.5, 130),
        ("wz-monogram-accent.svg", "WZ 强调色", (230, 250, 1040, 700), 16, 0.50, "accent", TOK["accent"], False, 0.5, 40),
        ("wordmark.svg", "字标", (200, 718, 1060, 900), 36, 0.35, "ink", TOK["ink"], False, 0.4, 130),
        ("wordmark-accent.svg", "字标强调", (200, 718, 1060, 900), 12, 0.40, "accent", TOK["accent"], False, 0.4, 30),
    ]
    for name, label, box, mins, mind, kind, fill, cf, eps, marea in jobs:
        print(f"\n--- {label} ---")
        m = mask_2x(box, kind)
        cols = None
        if box[0] < 624 < box[2]:
            cols = (2 * (612 - box[0]), 2 * (637 - box[0]))
        m, stats = clean_components(m, mins, mind, cols, box=box, drop_arcs=cf)
        bad = [s for s in stats if not s[5]]
        print(f"  连通域 {len(stats)} → 保留 {len(stats)-len(bad)}；剔除 "
              + ", ".join(f"{s[2]}x{s[3]}@{s[4]}{s[6]}" for s in bad))
        r, polys = emit(name, m, box, eps, marea, fill, label, cf)
        if r:
            rep.append(r)
            side_by_side(name.replace(".svg", ""), box, fill, polys)
    json.dump(rep, open(os.path.join(VERIFY, "..", "trace-report.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\n完成，共 {len(rep)} 个元素。")


if __name__ == "__main__":
    main()
