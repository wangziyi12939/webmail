"""定位残留碎段 + 用紧色窗重拟合引导圆"""
import numpy as np
from PIL import Image

SRC = r"D:/Javastudy/web-frontend/webmail/brand/_src/key-visual.jpg"
img = Image.open(SRC).convert("RGB")
a = np.asarray(img).astype(np.int16)
H, W = a.shape[:2]
mx = a.max(2)
mn = a.min(2)

# 1) 残留碎段附近像素
print("=== 碎段区域 (430..480, 200..250) 暗像素 ===")
sub = a[200:250, 430:480]
sm = sub.max(2)
ys, xs = np.nonzero(sm < 110)
print(f"  mx<110 数量 {len(ys)}")
for y, x in list(zip(ys.tolist(), xs.tolist()))[:12]:
    print(f"   ({430+x},{200+y}) rgb{tuple(int(v) for v in sub[y, x])} mx={int(sm[y,x])}")

# 2) 紧色窗拟合圆：只取「中灰、低饱和」像素
r0, g0, b0 = a[..., 0], a[..., 1], a[..., 2]
sat = (mx - mn) / np.maximum(mx, 1)
gray = (mx > 120) & (mx < 185) & (sat < 0.10)
yy, xx = np.mgrid[0:H, 0:W]
# 排除 monogram 内部（用大致的 W/Z 包围盒外扩）
inmono = (xx > 250) & (xx < 1010) & (yy > 280) & (yy < 640)
for ann in ((210, 300), (230, 280)):
    band = gray & ~inmono & (np.hypot(xx - 629.5, yy - 442.5) > ann[0]) & (np.hypot(xx - 629.5, yy - 442.5) < ann[1])
    if band.sum() < 30:
        print(f"  环带 {ann}: 样本不足 ({band.sum()})")
        continue
    px, py = xx[band].astype(float), yy[band].astype(float)
    A = np.c_[2 * px, 2 * py, np.ones(len(px))]
    sol, *_ = np.linalg.lstsq(A, px ** 2 + py ** 2, rcond=None)
    cx, cy = sol[0], sol[1]
    R = np.sqrt(sol[2] + cx ** 2 + cy ** 2)
    d = np.hypot(px - cx, py - cy) - R
    print(f"  环带 {ann}: n={band.sum():>5} 圆心({cx:.1f},{cy:.1f}) R={R:.1f} "
          f"残差 σ={d.std():.2f} max|d|={np.abs(d).max():.1f}")

# 3) 列出「细长小多边形」候选：在 2x 蒙版上找孤立小组件
print("\n=== 孤立小组件（monogram 区内，墨色）===")
ink = mx < 100
box = (230, 170, 1040, 700)
m = np.zeros_like(ink)
m[box[1]:box[3], box[0]:box[2]] = ink[box[1]:box[3], box[0]:box[2]]
m[:, 612:637] = False
from collections import deque
h, w = m.shape
hs, ws = h, w
small = m
seen = np.zeros_like(small)
for sy in range(hs):
    for sx in range(ws):
        if not small[sy, sx] or seen[sy, sx]:
            continue
        q = deque([(sy, sx)])
        seen[sy, sx] = True
        n = 0
        xs_, ys_ = [], []
        while q:
            cy2, cx2 = q.popleft()
            n += 1
            xs_.append(cx2)
            ys_.append(cy2)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = cy2 + dy, cx2 + dx
                    if 0 <= ny < hs and 0 <= nx < ws and small[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
        bw = max(xs_) - min(xs_) + 1
        bh = max(ys_) - min(ys_) + 1
        if min(bw, bh) < 26 and n > 40:
            cx2, cy2 = (min(xs_) + max(xs_)) / 2, (min(ys_) + max(ys_)) / 2
            dist = np.hypot(cx2 - 629.5, cy2 - 442.5)
            dens = n / (bw * bh)
            print(f"  bbox({min(xs_)},{min(ys_)})-({max(xs_)},{max(ys_)}) {bw}x{bh} "
                  f"n={n} dens={dens:.2f} 到圆心 {dist:.1f}  r-{dist:+.0f}")
