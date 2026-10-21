"""细节确认：暗部直方图 / 引导圆量测 / 元素接触表"""
import numpy as np
from PIL import Image, ImageDraw

SRC = r"D:/Javastudy/web-frontend/webmail/brand/_src/key-visual.jpg"
img = Image.open(SRC).convert("RGB")
W, H = img.size
a = np.asarray(img).astype(np.int16)
mx = a.max(2)

# ---- 暗部直方图：确定墨 / 引导圆的阈值分界 ----
print("=== 暗部像素分布 (mx<215) ===")
dark = mx[mx < 215]
hist, edges = np.histogram(dark, bins=21, range=(5, 215))
for i, c in enumerate(hist):
    lo, hi = int(edges[i]), int(edges[i + 1])
    print(f"  {lo:>3}-{hi:<3} {c:>8} {c / len(dark) * 100:5.2f}% {'#' * int(c / len(dark) * 300)}")
print(f"  (全图 {W*H} px, 暗部 {len(dark)} px = {len(dark)/(W*H)*100:.2f}%)")

# ---- 引导圆：沿中轴 / 水平中线扫描 ----
print("\n=== 引导圆量测 ===")


def scan(axis, fixed, lo, hi, thr=200):
    hits = []
    for v in range(lo, hi):
        px = a[v, fixed] if axis == "y" else a[fixed, v]
        if px.max() < thr:
            hits.append(v)
    return hits


# 沿 x=624（中轴）看被圆打断的位置 —— 圆在轴上的交点
col_dark = [y for y in range(0, H) if a[y, 624].max() < 200]
print("  中轴 x=624 暗像素 y 区间:", col_dark[:6], "...", col_dark[-6:] if col_dark else "")
row_dark = [x for x in range(0, W) if a[441, x].max() < 200]
print("  水平 y=441 暗像素 x:", row_dark[:8], "...", row_dark[-8:] if row_dark else "")

# 取「只在灰区间(110<mx<200)」的像素，拟合圆
g = a[..., 0].astype(float)
m = np.zeros(a.shape[:2], bool)
m |= (a[..., 0] > 108) & (a[..., 0] < 205) & (np.abs(a[..., 0] - a[..., 2]) < 30) & (np.abs(a[..., 0] - a[..., 1]) < 30)
# 只在 monogram 外环区域找
yy, xx = np.mgrid[0:H, 0:W]
ring = m & (((xx - 606) ** 2 + (yy - 441) ** 2) > 200 ** 2) & (((xx - 606) ** 2 + (yy - 441) ** 2) < 320 ** 2)
if ring.sum() > 50:
    px, py = xx[ring], yy[ring]
    # 代数拟圆
    A = np.c_[2 * px, 2 * py, np.ones(len(px))]
    bb = px ** 2 + py ** 2
    sol, *_ = np.linalg.lstsq(A, bb, rcond=None)
    cx, cy = sol[0], sol[1]
    r = np.sqrt(sol[2] + cx ** 2 + cy ** 2)
    print(f"  拟合圆: 圆心({cx:.1f}, {cy:.1f})  半径 {r:.1f}  (画布 {W}px → r={r/W*100:.1f}%)")
    print(f"  弧像素均值色 #%02X%02X%02X" % tuple(int(v) for v in a[ring].mean(0)))
    print(f"  弧像素 n={ring.sum()}")

# ---- 中轴线：按中轴单列 x=624 分类 ----
print("\n=== 中轴线 (单列 x=624) 分段 ===")
seg_kind, seg_start = None, 0
runs = []
for y in range(H):
    R, G, B = a[y, 624]
    if R > 150 and B < 160 and R - B > 60:
        k = "黄"
    elif max(R, G, B) < 160:
        k = "墨"
    elif max(R, G, B) < 212:
        k = "灰"
    else:
        k = None
    if k != seg_kind:
        if seg_kind:
            runs.append((seg_kind, seg_start, y - 1))
        seg_kind, seg_start = k, y
if seg_kind:
    runs.append((seg_kind, seg_start, H - 1))
for k, y0, y1 in runs:
    if y1 - y0 >= 2:
        print(f"  {k:<2}  y {y0:>4}–{y1:<4} 长 {y1 - y0 + 1:<4} 宽 12px")

# ---- 墨色真值：腐蚀后取内部 ----
print("\n=== 墨色真值（腐蚀 6px 后的内部平坦区）===")
core = (mx < 120).copy()
er = core.copy()
for _ in range(6):
    er = er & np.roll(er, 1, 0) & np.roll(er, -1, 0) & np.roll(er, 1, 1) & np.roll(er, -1, 1)
v = a[er]
print(f"  内部像素 n={er.sum()}   均值 #%02X%02X%02X" % tuple(int(x) for x in v.mean(0)))
print("  众数:", end=" ")
k = (v // 3).astype(np.int32)
k = k[:, 0] * 100000 + k[:, 1] * 200 + k[:, 2]
vals, cnts = np.unique(k, return_counts=True)
for i in np.argsort(-cnts)[:3]:
    sel = v[k == vals[i]]
    print("#%02X%02X%02X(%.1f%%)" % (*[int(round(x)) for x in sel.mean(0)], len(sel) / len(v) * 100), end=" ")
print()

# ---- 黄真值：腐蚀后取内部 ----
core = ((a[..., 0] > 150) & (a[..., 2] < 160) & (a[..., 0] - a[..., 2] > 60)).copy()
er = core.copy()
for _ in range(4):
    er = er & np.roll(er, 1, 0) & np.roll(er, -1, 0) & np.roll(er, 1, 1) & np.roll(er, -1, 1)
v = a[er]
print(f"  黄内部 n={er.sum()}   均值 #%02X%02X%02X  rgb(%s)" % (*[int(round(x)) for x in v.mean(0)],
                                                       ", ".join(f"{x:.1f}" for x in v.mean(0))))

mv=v.mean(0).tolist(); print("  HSV: H=%.1f deg  S=%.1f%%  V=%.1f%%" % (60*(mv[1]-mv[2])/(mv[0]-mv[2]), (mv[0]-mv[2])/mv[0]*100, mv[0]/255*100))

# ---- 接触表 ----
regions = [
    ("monogram-W", (255, 275, 575, 640)),
    ("monogram-Z", (655, 285, 1010, 625)),
    ("envelope", (480, 285, 775, 640)),
    ("Z-bars", (835, 320, 1010, 540)),
    ("wordmark", (220, 725, 1040, 900)),
    ("email-row", (200, 930, 1050, 1000)),
    ("tl-corner", (30, 30, 120, 90)),
    ("tr-corner", (1060, 30, 1230, 170)),
    ("bl-corner", (30, 1150, 300, 1210)),
    ("br-corner", (1080, 1140, 1230, 1210)),
    ("axis-top", (595, 140, 655, 210)),
    ("hatch-in-Z", (880, 480, 960, 560)),
    ("barcode-in-Z", (890, 570, 1000, 620)),
]
scale = 2
cw = max((b[2] - b[0]) for _, b in regions) * scale + 20
sheet = Image.new("RGB", (cw + 220, sum((b[3] - b[1]) * scale + 24 for _, b in regions) + 20), "#FFFFFF")
d = ImageDraw.Draw(sheet)
y = 10
for name, box in regions:
    c = img.crop(box)
    c = c.resize((c.width * scale, c.height * scale), Image.LANCZOS)
    sheet.paste(c, (215, y))
    d.text((8, y + c.height // 2), name, fill="#000000")
    d.line([(200, y), (212, y + c.height)], fill="#CCCCCC")
    y += c.height + 24
sheet.save(r"D:/Javastudy/web-frontend/webmail/brand/_src/contact-sheet.png")
print(f"\n接触表已生成 {sheet.size}")
