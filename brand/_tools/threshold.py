"""阈值扫描：找出既保字形又切掉引导圆残段的墨色阈值"""
import numpy as np
from PIL import Image, ImageDraw

SRC = r"D:/Javastudy/web-frontend/webmail/brand/_src/key-visual.jpg"
img = Image.open(SRC).convert("RGB")
W, H = img.size
a = np.asarray(img).astype(np.int16)
mx = a.max(2)

# 拟合圆
CX, CY, RAD = 629.5, 442.5, 248.7
yy, xx = np.mgrid[0:H, 0:W]
dist = np.sqrt((xx - CX) ** 2 + (yy - CY) ** 2)
annulus = np.abs(dist - RAD) < 16          # 圆周带
box = (230, 170, 1040, 700)
x0, y0, x1, y1 = box
inbox = np.zeros_like(mx, bool)
inbox[y0:y1, x0:x1] = True

print("阈值  区内墨像素   圆周带内    圆周带占比   说明")
for t in (55, 65, 75, 85, 95, 105, 115):
    m = (mx < t) & inbox
    tot = m.sum()
    ring = (m & annulus).sum()
    print(f"  {t:>3}  {tot:>9}  {ring:>9}      {ring / max(tot,1) * 100:6.2f}%")

# 用极暗阈值看圆周带的像素到底多暗
ring_dark = (mx < 140) & annulus & inbox
vals = mx[ring_dark & (mx < 200)]
print(f"\n圆周带内暗像素 (mx<140) 数量 {ring_dark.sum()}")
if ring_dark.sum():
    print("  亮度分位:", {q: int(np.percentile(mx[annulus & inbox], q)) for q in (1, 5, 25, 50, 75, 95)})

# 出图：三个阈值并排
tiles = []
for t in (75, 95, 105):
    m = (mx < t) & inbox
    im = Image.fromarray(np.where(m, 0, 255).astype(np.uint8)).convert("RGB").crop(box)
    im = im.resize((im.width // 2, im.height // 2), Image.NEAREST)
    tiles.append((t, im))
tw, th = tiles[0][1].size
sheet = Image.new("RGB", (tw * 3 + 40, th + 26), "#FFFFFF")
d = ImageDraw.Draw(sheet)
for i, (t, im) in enumerate(tiles):
    sheet.paste(im, (10 + i * (tw + 10), 22))
    d.text((10 + i * (tw + 10), 6), f"mx < {t}")
sheet.save(r"D:/Javastudy/web-frontend/webmail/brand/_src/verify/thresholds.png")
print("saved thresholds.png", sheet.size)
