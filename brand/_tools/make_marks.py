"""生成 brand/marks 矢量件与完整 lockup（定稿）

实测参数（原画布 1254×1254）：
  引导圆   圆心(626.3,442.4) R=239.1 描边7.5 极坐标实测节奏
  中轴     x618 宽12  y80–1023  四段黄 156-175/323-411/711-751/957-991
  主徽标   (268,296.5) 726×329  · 强调色 (507,309.5) 484×204
  字标     (233,743)   786×138  · 强调三角 (266,741.5) 36×34
  EMAIL    字 x445–805 高30(y909–938)；横线 y921 厚4  x239–413 / 841–1014
  四角     构造件灰 #828282–#888888（三角标为墨色）
"""
import json
import math
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tracer import trace  # noqa: E402

ROOT = r"D:/Javastudy/web-frontend/webmail/brand"
SRC = os.path.join(ROOT, "_src", "key-visual.jpg")
MARKS = os.path.join(ROOT, "marks")
os.makedirs(MARKS, exist_ok=True)

TOK = {"ink": "#232527", "accent": "#F5CE15", "paper": "#E7E8E5",
       "gray": "#6B6D6E", "gray2": "#828282"}
CV = 1254
CIRCLE = (626.3, 442.4)
R_CIRCLE, STROKE_CIRCLE = 239.1, 7.5
AXIS = dict(x=618, w=12, y0=80, y1=1023, yellow=[(156, 175), (323, 411), (711, 751), (957, 991)])
EMAIL = dict(rules=[(239, 413), (841, 1014)], y=921, h=4, text_x=445, text_w=360, cap=30, baseline=938)

img = Image.open(SRC).convert("RGB")
_arr = np.asarray(img).astype(np.int16)
FONT = "Inter, 'Helvetica Neue', Arial, sans-serif"


def _f(v, p=1):
    s = f"{v:.{p}f}"
    return s[:-2] if s.endswith(".0") else s


def trace_part(name, box, thr, eps=0.35, min_area=8, ss=3, fill=None, label="", keep=None):
    c = img.crop(box).resize(((box[2] - box[0]) * ss, (box[3] - box[1]) * ss), Image.LANCZOS)
    a = np.asarray(c).astype(np.int16)
    m = a.max(2) < thr
    polys = trace(m, eps=eps * ss, min_area=min_area * ss * ss)
    if keep is not None:
        polys = [p for p in polys if _area(p) >= keep * ss * ss]
    if not polys:
        print(f"  [空] {label or name}")
        return None
    polys = [[(x / ss + box[0], y / ss + box[1]) for x, y in p] for p in polys]
    xs = [v[0] for p in polys for v in p]
    ys = [v[1] for p in polys for v in p]
    ox, oy = min(xs), min(ys)
    vw, vh = int(math.ceil(max(xs) - ox)), int(math.ceil(max(ys) - oy))
    d = "".join("M" + _f(p[0][0] - ox) + " " + _f(p[0][1] - oy)
                + "".join("L" + _f(v[0] - ox) + " " + _f(v[1] - oy) for v in p[1:]) + "Z"
                for p in polys)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {vw} {vh}" '
           f'width="{vw}" height="{vh}" role="img" aria-label="{label or name}">\n'
           f'  <path fill="{fill or TOK["ink"]}" fill-rule="evenodd" d="{d}"/>\n</svg>\n')
    open(os.path.join(MARKS, name), "w", encoding="utf-8").write(svg)
    print(f"  [OK] {label or name:<12} {name:<26} {vw:>3}x{vh:<3} 环{len(polys):>3} "
          f"节点{sum(len(p) for p in polys):>4}  {len(svg)/1024:.1f}KB")
    return {"ox": ox, "oy": oy, "w": vw, "h": vh, "d": d, "fill": fill or TOK["ink"]}


def _area(p):
    s = 0.0
    for i in range(len(p)):
        s += p[i][0] * p[(i + 1) % len(p)][1] - p[(i + 1) % len(p)][0] * p[i][1]
    return abs(s) / 2


def polar_dasharray():
    mx = _arr.max(2)
    N = 2880
    marks = []
    for i in range(N):
        th = i * 2 * math.pi / N
        x = int(round(CIRCLE[0] + R_CIRCLE * math.cos(th)))
        y = int(round(CIRCLE[1] + R_CIRCLE * math.sin(th)))
        marks.append(0 <= x < CV and 0 <= y < CV and mx[y, x] < 190)
    runs, cur, st = [], marks[0], 0
    for i in range(1, N):
        if marks[i] != cur:
            runs.append((cur, st, i))
            cur, st = marks[i], i
    runs.append((cur, st, N))
    unit = R_CIRCLE * 2 * math.pi / N
    arr = [round((i2 - i1) * unit) for _, i1, i2 in runs]
    if len(arr) > 2 and marks[0] == marks[-1]:
        arr = [arr[0] + arr[-1]] + arr[1:-1]
    return arr


def main():
    report = {}
    print("=== 1. 小件描摹 ===")
    report["triangle"] = trace_part("part-talent-triangle.svg", (1130, 34, 1184, 78), 140,
                                    label="三角标", fill=TOK["ink"])
    report["barcode"] = trace_part("part-barcode.svg", (1086, 1150, 1184, 1200), 190,
                                   label="条码", fill=TOK["gray2"])
    # "++" 用参数化重建（极轻的灰、描摹会失真）
    cr = 'M0 4.5H13M6.5 -2V11'
    open(os.path.join(MARKS, "part-plus-plus.svg"), "w", encoding="utf-8").write(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 30 14" width="30" height="14" '
        f'role="img" aria-label="加号组">\n'
        + "".join(f'  <path transform="translate({i*17},1.5)" d="{cr}" stroke="{TOK["gray2"]}" '
                  f'stroke-width="3" fill="none"/>\n' for i in range(2))
        + '</svg>\n')
    print("  [OK] 加号（参数化）  part-plus-plus.svg     30x14  2 十字  0.3KB")
    report["plus"] = {"ox": 1180.0, "oy": 1164.0, "w": 30, "h": 14, "fill": TOK["gray2"],
                      "d": ""}
    report["hatch"] = trace_part("part-hatch.svg", (194, 1158, 294, 1202), 190,
                                 label="hatch", fill=TOK["gray2"])
    report["rule-ticks"] = trace_part("part-rule-ticks.svg", (200, 1166, 266, 1196), 200,
                                      label="短横线组", fill=TOK["gray2"])
    report["z-hatch"] = trace_part("part-hatch-z.svg", (852, 556, 926, 620), 150,
                                   label="Z 内 hatch", fill=TOK["ink"])
    report["z-bars"] = trace_part("part-barcode-z.svg", (876, 556, 930, 620), 150,
                                  label="Z 内竖条", fill=TOK["ink"])

    print("\n=== 2. 引导圆 ===")
    da = polar_dasharray()
    print(f"  实测弧/缺口节奏 {da}")
    sz = 2 * R_CIRCLE
    open(os.path.join(MARKS, "guide-circle.svg"), "w", encoding="utf-8").write(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_f(sz)} {_f(sz)}" '
        f'width="{_f(sz)}" height="{_f(sz)}" role="img" aria-label="引导圆">\n'
        f'  <circle cx="{R_CIRCLE}" cy="{R_CIRCLE}" r="{R_CIRCLE}" fill="none" '
        f'stroke="{TOK["gray"]}" stroke-width="{STROKE_CIRCLE}" '
        f'stroke-dasharray="{",".join(map(str, da))}"/>\n</svg>\n')
    report["guide-circle"] = {"dasharray": da, "r": R_CIRCLE, "stroke": STROKE_CIRCLE, "center": list(CIRCLE)}

    print("\n=== 3. 中轴 ===")
    h = AXIS["y1"] - AXIS["y0"]
    body = [f'  <rect x="0" y="0" width="{AXIS["w"]}" height="{h}" fill="{TOK["ink"]}"/>']
    for y0, y1 in AXIS["yellow"]:
        body.append(f'  <rect x="0" y="{y0-AXIS["y0"]}" width="{AXIS["w"]}" '
                    f'height="{y1-y0+1}" fill="{TOK["accent"]}"/>')
    open(os.path.join(MARKS, "axis-rule.svg"), "w", encoding="utf-8").write(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {AXIS["w"]} {h}" '
        f'width="{AXIS["w"]}" height="{h}" role="img" aria-label="中轴">\n'
        + "\n".join(body) + "\n</svg>\n")
    print(f"  [OK] axis-rule.svg  {AXIS['w']}x{h}  黄段 {len(AXIS['yellow'])}")
    report["axis"] = AXIS

    print("\n=== 4. lockup 组装 ===")
    rep = {r["name"]: r for r in json.load(open(os.path.join(ROOT, "_src", "trace-report.json"), encoding="utf-8"))}

    def path_of(fn):
        return open(os.path.join(MARKS, fn), encoding="utf-8").read().split('d="')[1].split('"')[0]

    def grp(fn, fill):
        r = rep[fn]
        return (f'  <g transform="translate({_f(r["offset"][0])},{_f(r["offset"][1])})">'
                f'<path fill="{fill}" fill-rule="evenodd" d="{path_of(fn)}"/></g>\n')

    def part(p, dx, dy, fill=None):
        if not p:
            return ""
        return (f'  <g transform="translate({_f(p["ox"]+dx)},{_f(p["oy"]+dy)})">'
                f'<path fill="{fill or p["fill"]}" fill-rule="evenodd" d="{p["d"]}"/></g>\n')

    L = []
    L.append('  <title>WangZiyi Email — 标志锁定 (elements extracted from key visual)</title>\n')
    L.append('  <g id="guide-circle">\n'
             f'    <circle cx="{_f(CIRCLE[0])}" cy="{_f(CIRCLE[1])}" r="{R_CIRCLE}" fill="none" '
             f'stroke="{TOK["gray"]}" stroke-width="{STROKE_CIRCLE}" '
             f'stroke-dasharray="{",".join(map(str, da))}"/>\n  </g>\n')
    L.append('  <g id="center-axis">\n'
             + "".join(f'    <rect x="{AXIS["x"]}" y="{AXIS["y0"]}" width="{AXIS["w"]}" height="{h}" fill="{TOK["ink"]}"/>\n' for _ in (0,))
             + "".join(f'    <rect x="{AXIS["x"]}" y="{y0}" width="{AXIS["w"]}" height="{y1-y0+1}" fill="{TOK["accent"]}"/>\n'
                       for y0, y1 in AXIS["yellow"]) + '  </g>\n')
    L.append('  <g id="monogram">\n' + grp("wz-monogram.svg", TOK["ink"])
             + grp("wz-monogram-accent.svg", TOK["accent"]) + '  </g>\n')
    L.append('  <g id="wordmark">\n' + grp("wordmark.svg", TOK["ink"])
             + grp("wordmark-accent.svg", TOK["accent"]) + '  </g>\n')
    L.append('  <g id="email-rule">\n'
             + "".join(f'    <rect x="{x0}" y="{EMAIL["y"]}" width="{x1-x0}" height="{EMAIL["h"]}" fill="{TOK["ink"]}"/>\n'
                       for x0, x1 in EMAIL["rules"])
             + f'    <text x="{EMAIL["text_x"]}" y="{EMAIL["baseline"]}" fill="{TOK["ink"]}" '
               f'font-family="{FONT}" font-size="{EMAIL["cap"]}" font-weight="500" '
               f'letter-spacing="{EMAIL["text_w"]/5 - EMAIL["cap"]*0.62:.1f}" '
               f'textLength="{EMAIL["text_w"]}" lengthAdjust="spacing">EMAIL</text>\n  </g>\n')

    # 四角构造件（文字度量按实测bbox近似：左上 9字/140px、右上 8字/86px）
    C = []
    C.append('  <g id="corner-tl">\n'
             f'    <text x="47" y="66" fill="{TOK["ink"]}" font-family="{FONT}" font-size="21" '
             f'font-weight="500" letter-spacing="2.6">ARKNIGHTS</text>\n'
             f'    <text x="47" y="92" fill="{TOK["ink"]}" font-family="{FONT}" font-size="21" '
             f'font-weight="500" letter-spacing="2.6">ENDFIELD</text>\n'
             f'    <rect x="47" y="104" width="73" height="8" fill="{TOK["ink"]}"/>\n  </g>\n')
    C.append('  <g id="corner-tr">\n' + part(report["triangle"], 0, 0)
             + "".join(f'    <text x="1138" y="{108+19*i}" fill="{TOK["ink"]}" font-family="{FONT}" '
                       f'font-size="16" font-weight="500" letter-spacing="1.35">{t}</text>\n'
                       for i, t in enumerate(("TALENT", "CONNECTS", "THE FUTURE")))
             + f'    <rect x="1138" y="152" width="34" height="5" fill="{TOK["gray2"]}"/>\n  </g>\n')
    C.append('  <g id="corner-bl">\n'
             f'    <text x="47" y="1178" fill="{TOK["ink"]}" font-family="{FONT}" font-size="34" '
             f'font-weight="700" letter-spacing="1.5">WZ</text>\n'
             + "".join(f'    <rect x="{x}" y="1188" width="{w}" height="6" fill="{TOK["gray2"]}"/>\n'
                       for x, w in ((47, 42), (102, 96)))
             + part(report["hatch"], 0, 0) + '  </g>\n')
    C.append('  <g id="corner-br">\n' + part(report["barcode"], 0, 0)
             + "".join(f'    <path transform="translate({1180 + i * 17},1165.5)" d="M0 4.5H13M6.5 -2V11" '
                       f'stroke="{TOK["gray2"]}" stroke-width="3" fill="none"/>\n' for i in range(2))
             + '  </g>\n')
    L.extend(C)

    lockup = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {CV} {CV}" '
              f'width="{CV}" height="{CV}" role="img" aria-label="WangZiyi Email 标志锁定">\n'
              + "".join(L) + '</svg>\n')
    open(os.path.join(MARKS, "lockup.svg"), "w", encoding="utf-8").write(lockup)
    print(f"  [OK] lockup.svg  {CV}x{CV}  {len(lockup)/1024:.1f}KB  {len(L)} 组")

    mono = (lockup.replace(TOK["ink"], "currentColor")
                  .replace(TOK["accent"], "currentColor")
                  .replace(TOK["gray2"], "currentColor")
                  .replace(TOK["gray"], "currentColor"))
    open(os.path.join(MARKS, "lockup-mono.svg"), "w", encoding="utf-8").write(mono)
    print(f"  [OK] lockup-mono.svg  {len(mono)/1024:.1f}KB")

    json.dump(report, open(os.path.join(ROOT, "_src", "marks-report.json"), "w"),
              ensure_ascii=False, indent=1, default=str)
    print("\n完成 → brand/marks/")


if __name__ == "__main__":
    main()
