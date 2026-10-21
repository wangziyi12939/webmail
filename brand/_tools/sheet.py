"""紧凑接触表：网格排布，便于一次目视核对"""
from PIL import Image, ImageDraw

SRC = r"D:/Javastudy/web-frontend/webmail/brand/_src/key-visual.jpg"
img = Image.open(SRC).convert("RGB")

regions = [
    ("W (左字)", (255, 275, 575, 640), 1.5),
    ("Z (右字)", (655, 285, 1010, 625), 1.5),
    ("信封", (480, 285, 775, 645), 2.0),
    ("Z 斜条组", (830, 315, 1010, 545), 2.6),
    ("Z 内 hatch+条码", (870, 470, 1005, 625), 3.0),
    ("wordmark", (225, 730, 1035, 895), 1.2),
    ("W 上的黄块", (235, 725, 410, 865), 3.0),
    ("EMAIL 行", (200, 928, 1050, 1000), 1.2),
    ("左上角标", (28, 32, 200, 92), 4.0),
    ("右上角标", (1060, 32, 1235, 165), 4.0),
    ("左下角标", (28, 1150, 300, 1210), 4.0),
    ("右下角标", (1075, 1130, 1235, 1210), 4.0),
    ("中轴顶段", (596, 130, 652, 300), 4.0),
    ("引导圆弧", (330, 180, 930, 320), 1.4),
]

COLS, CELL = 3, 470
rows = (len(regions) + COLS - 1) // COLS
sheet = Image.new("RGB", (COLS * CELL, rows * (CELL + 26) + 8), "#FFFFFF")
d = ImageDraw.Draw(sheet)
for i, (name, box, sc) in enumerate(regions):
    c = img.crop(box)
    c = c.resize((max(1, int(c.width * sc)), max(1, int(c.height * sc))), Image.LANCZOS)
    if c.width > CELL - 12:
        c = c.resize((CELL - 12, int(c.height * (CELL - 12) / c.width)), Image.LANCZOS)
    if c.height > CELL - 12:
        c = c.resize((int(c.width * (CELL - 12) / c.height), CELL - 12), Image.LANCZOS)
    cx, cy = (i % COLS) * CELL, (i // COLS) * (CELL + 26)
    sheet.paste(c, (cx + 6, cy + 22))
    d.text((cx + 6, cy + 6), f"{name}  {box[2]-box[0]}x{box[3]-box[1]}", fill="#000000")
    d.rectangle([cx + 1, cy + 1, cx + CELL - 2, cy + CELL + 22], outline="#BBBBBB")
sheet.save(r"D:/Javastudy/web-frontend/webmail/brand/_src/contact-sheet.png")
print("contact-sheet", sheet.size)
