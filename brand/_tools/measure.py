"""测量构造件：引导圆线宽/虚线节奏、中轴分段、四角标几何、EMAIL 行"""
import math

import numpy as np
from PIL import Image

a = np.asarray(Image.open(r"D:/Javastudy/web-frontend/webmail/brand/_src/key-visual.jpg").convert("RGB")).astype(np.int16)
H, W = a.shape[:2]
mx = a.max(2)
CX, CY, R = 626.3, 442.4, 239.1

print("=== 1. 引导圆：极坐标展开找线段/缺口 ===")
marks = []
for i in range(2880):                     # 0.125° 步长
    th = i * 2 * math.pi / 2880
    x = int(round(CX + R * math.cos(th)))
    y = int(round(CY + R * math.sin(th)))
    if not (0 <= x < W and 0 <= y < H):
        marks.append(False)
        continue
    marks.append(mx[y, x] < 190)
runs, cur, start = [], marks[0], 0
for i in range(1, 2880):
    if marks[i] != cur:
        runs.append((cur, start, i))
        cur, start = marks[i], i
runs.append((cur, start, 2880))
arc = [(i2 - i1) * (R * 2 * math.pi / 2880) for on, i1, i2 in runs if on]
gap = [(i2 - i1) * (R * 2 * math.pi / 2880) for on, i1, i2 in runs if not on]
print(f"  线段 {len(arc)} 段：长度 {[round(v) for v in arc]}")
print(f"  缺口 {len(gap)} 段：长度 {[round(v) for v in gap]}")

print("\n=== 2. 引导圆线宽（径向断面）===")
for deg in (200, 250, 290, 340, 20, 70, 110, 160):
    th = math.radians(deg)
    on = []
    for dr in np.arange(-30, 30, 0.5):
        x = int(round(CX + (R + dr) * math.cos(th)))
        y = int(round(CY + (R + dr) * math.sin(th)))
        if 0 <= x < W and 0 <= y < H and mx[y, x] < 190:
            on.append(dr)
    if on:
        print(f"  {deg}°: dr {min(on)}~{max(on)}  宽 {max(on)-min(on)+0.5}px  墨色均值 {int(mx[int(CY+(R+np.mean(on))*math.sin(th)), int(CX+(R+np.mean(on))*math.cos(th))])}")

print("\n=== 3. 中轴 x=618..630 分段（含灰）===")
kinds = []
for y in range(H):
    seg = a[y, 618:631]
    Mm = seg.max(1)
    if (seg[:, 0] - seg[:, 2] > 60)[Mm.argmax()] and seg[:, 0].max() > 150:
        k = "黄"
    elif Mm.min() < 100:
        k = "墨"
    else:
        k = None
    kinds.append(k)
runs, cur, st = [], kinds[0], 0
for i in range(1, H):
    if kinds[i] != cur:
        if cur:
            runs.append((cur, st, i - 1))
        cur, st = kinds[i], i
if cur:
    runs.append((cur, st, H - 1))
print(f"  轴宽带 x=618..630 (12px)。共 {len(runs)} 段：")
for k, y0, y1 in runs:
    if y1 - y0 >= 3:
        print(f"    {k}  y{y0:>4}–{y1:<4} 长 {y1-y0+1}")
print(f"  轴总跨度 y{runs[0][1]} – y{runs[-1][2]}  = {runs[-1][2]-runs[0][1]+1}px")

print("\n=== 4. 四角标与 EMAIL 行 ===")
for name, box in (("左上 ARKNIGHTS/ENDFIELD", (30, 30, 260, 200)),
                  ("右上 三角+TALENT", (1130, 30, 1230, 200)),
                  ("左下 WZ+rule+hatch", (30, 1150, 320, 1210)),
                  ("右下 条码++", (1080, 1140, 1230, 1210)),
                  ("EMAIL 行", (150, 990, 1110, 1060))):
    x0, y0, x1, y1 = box
    sub = mx[y0:y1, x0:x1] < 190
    ys, xs = np.nonzero(sub)
    if len(xs) == 0:
        print(f"  {name}: 空")
        continue
    print(f"  {name}: bbox({x0+xs.min()},{y0+ys.min()})-({x0+xs.max()},{y0+ys.max()}) "
          f"尺寸 {xs.max()-xs.min()+1}x{ys.max()-ys.min()+1}")

print("\n=== 5. EMAIL 文字与两侧横线 ===")
band = mx[1020:1050, :] < 190
cols = band.sum(0)
runs, cur, st = [], cols[0] > 0, 0
for i in range(1, W):
    v = cols[i] > 0
    if v != cur:
        runs.append((cur, st, i - 1))
        cur, st = v, i
runs.append((cur, st, W - 1))
print("  y1020-1050 横向游程:", [(("有" if o else "空"), x0, x1, x1 - x0 + 1) for o, x0, x1 in runs if x1 - x0 > 3])

print("\n=== 6. 字标/EMAil 精确定位 ===")
ink = mx < 120
for name, box in (("W 字标", (220, 730, 420, 870)), ("g 字标", (610, 760, 730, 890)),
                  ("y 字标", (875, 730, 1030, 890)), ("黄三角", (255, 725, 320, 785))):
    x0, y0, x1, y1 = box
    sub = ink[y0:y1, x0:x1]
    ys, xs = np.nonzero(sub)
    if len(xs):
        print(f"  {name}: ({x0+xs.min()},{y0+ys.min()})-({x0+xs.max()},{y0+ys.max()}) "
              f"{xs.max()-xs.min()+1}x{ys.max()-ys.min()+1}")
