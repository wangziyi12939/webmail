"""像素级分析 key-visual：取色 + 分区 + 几何量测"""
import json
import numpy as np
from PIL import Image

SRC = r"D:/Javastudy/web-frontend/webmail/brand/_src/key-visual.jpg"
img = Image.open(SRC).convert("RGB")
W, H = img.size
a = np.asarray(img).astype(np.int16)
print(f"size = {W} x {H}")


def patch(cx, cy, r=6):
    """取以 (cx,cy) 为中心 r 半径方块的平均色（避开 JPEG 噪点）"""
    x0, x1 = max(0, cx - r), min(W, cx + r)
    y0, y1 = max(0, cy - r), min(H, cy + r)
    p = a[y0:y1, x0:x1].reshape(-1, 3)
    m = p.mean(0)
    return "#%02X%02X%02X" % tuple(int(round(v)) for v in m), tuple(round(float(v), 1) for v in m)


print("\n=== 关键点位取样 ===")
probes = {
    "纸张底 (中部留白)": (540, 640),
    "纸张底 (底部留白)": (300, 990),
    "斜切面 左上": (120, 300),
    "斜切面 右下": (1000, 760),
    "斜切面 右侧": (1010, 400),
    "墨色 (字身)": (300, 690),
    "墨色 (W 主干)": (270, 430),
    "黄色 (信封盖)": (540, 315),
    "黄色 (Z 斜条)": (830, 380),
    "黄点 (中轴线)": (540, 890),
    "引导圆 (虚线)": (770, 300),
    "角标文字": (60, 46),
}
for k, (x, y) in probes.items():
    hx, rgb = patch(x, y)
    print(f"  {k:<20} {hx}  rgb{rgb}")

# ---- 全图色簇 ----
print("\n=== 全图主色簇 (KMeans-lite / 量化=6bit) ===")
q = (a // 8 * 8).reshape(-1, 3)
uniq, cnt = np.unique(q, axis=0, return_counts=True)
order = np.argsort(-cnt)[:14]
total = q.shape[0]
for i in order:
    rgb = uniq[i]
    hx = "#%02X%02X%02X" % tuple(int(v) for v in rgb)
    print(f"  {hx}  {cnt[i] / total * 100:6.2f}%   rgb{tuple(int(v) for v in rgb)}")

# ---- 黄/黑 像素占比 ----
r, g, b = a[..., 0].astype(float), a[..., 1].astype(float), a[..., 2].astype(float)
mx, mn = a.max(2).astype(float), a.min(2).astype(float)
sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0)
yellow = (sat > 0.45) & (r > 150) & (b < 140) & (g > 130)
ink = (mx < 110)
paper = (mx > 200) & (sat < 0.12)
print("\n=== 面积占比 ===")
for name, mask in (("黄 (强调色)", yellow), ("墨 (深色)", ink), ("纸 (浅底)", paper)):
    print(f"  {name:<12} {mask.sum() / (W * H) * 100:6.2f}%")

# 纯黄均值
if yellow.sum():
    ymean = a[yellow].mean(0)
    print("  黄像素均值    #%02X%02X%02X  rgb%s" % (*[int(round(v)) for v in ymean], tuple(round(float(v), 1) for v in ymean)))
if ink.sum():
    # 只取最暗的一批，避开反锯齿边
    dark = a[ink]
    d15 = dark[dark.sum(1) <= np.percentile(dark.sum(1), 15)]
    print("  墨像素最暗15%% #%02X%02X%02X" % tuple(int(round(v)) for v in d15.mean(0)))

# ---- 灰阶倾向检查 ----
print("\n=== 灰阶倾向 (B<=R<G 留绿?) ===")
for label, m in (("墨色区", ink), ("纸底区", paper)):
    if m.sum():
        c = a[m].mean(0)
        print(f"  {label}: R={c[0]:.1f} G={c[1]:.1f} B={c[2]:.1f}   G-R={c[1]-c[0]:+.1f}  B-R={c[2]-c[0]:+.1f}")
        hm = a[m]
        dark_half = hm[hm.sum(1) <= np.median(hm.sum(1))]
        c2 = dark_half.mean(0)
        print(f"  {label}(偏暗半): R={c2[0]:.1f} G={c2[1]:.1f} B={c2[2]:.1f}   G-R={c2[1]-c2[0]:+.1f}  B-R={c2[2]-c2[0]:+.1f}")

# ---- 墨色区域的包围盒（定位字形群） ----
ys, xs = np.nonzero(ink)
print(f"\n墨色总包围盒: x {xs.min()}–{xs.max()}  y {ys.min()}–{ys.max()}")

# 逐行墨色密度，找水平分带
row = ink.sum(1) / W
print("\n=== 水平分带 (每 40px 一行，>3% 视为有内容) ===")
for y0 in range(0, H, 40):
    seg = row[y0:y0 + 40].mean() * 100
    bar = "#" * int(seg * 2)
    print(f"  y{y0:>4}–{y0+40:<4} {seg:5.1f}% {bar}")

col = ink.sum(0) / H
print("\n=== 垂直分带 (每 60px 一列) ===")
for x0 in range(0, W, 60):
    seg = col[x0:x0 + 60].mean() * 100
    bar = "#" * int(seg * 2)
    print(f"  x{x0:>4}–{x0+60:<4} {seg:5.1f}% {bar}")
