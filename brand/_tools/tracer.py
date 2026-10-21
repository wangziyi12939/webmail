"""位图 → 矢量：裂缝追踪(crack following) + Douglas-Peucker，输出干净 SVG path

不依赖 potrace。思路：
1. 对二值掩膜提取「前景像素与外部的交界边」→ 有向边集合
2. 顶点处按固定转向规则串联 → 闭合环（外轮廓 + 内孔自动分离）
3. 共线合并 → DP 简化 → SVG path（evenodd 填充）
"""
import math
from collections import defaultdict

import numpy as np

# 顺时针绕前景的 4 条有向边（像素坐标，y 向下）
# 顶点编号用 (x, y) 整数角点


def _edges(mask):
    fg = mask
    h, w = fg.shape
    pad = np.zeros((h + 2, w + 2), bool)
    pad[1:-1, 1:-1] = fg
    up = pad[0:h, 1:w + 1]
    dn = pad[2:h + 2, 1:w + 1]
    lf = pad[1:h + 1, 0:w]
    rt = pad[1:h + 1, 2:w + 2]
    out = defaultdict(list)
    ys, xs = np.nonzero(fg)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if not up[y, x]:
            out[(x, y)].append((x + 1, y))
        if not rt[y, x]:
            out[(x + 1, y)].append((x + 1, y + 1))
        if not dn[y, x]:
            out[(x + 1, y + 1)].append((x, y + 1))
        if not lf[y, x]:
            out[(x, y + 1)].append((x, y))
    return out


# 转向优先级：进入方向 → 优先尝试的方向（右转优先，保证对角相接不粘连）
DIRS = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}
_ROT = {  # 右转优先顺序：直行、右转… 实际取 (dx,dy) 的取值表
}


def _turn_priority(din):
    """给定进入方向，返回候选离开方向的优先级（先直、再右、再左、最后掉头由环路自然处理）"""
    dx, dy = din
    straight = (dx, dy)
    right = (-dy, dx)      # 顺时针 90°
    left = (dy, -dx)
    back = (-dx, -dy)
    return [straight, right, left, back]


def trace(mask, eps=1.1, min_area=60.0):
    """返回多边形列表 [(x0,y0), (x1,y1), ...]，坐标为像素角点"""
    out = _edges(mask)
    polys = []
    for start_v in [v for v, l in out.items() if l]:
        while out.get(start_v):
            first_to = out[start_v].pop()
            pts = [start_v, first_to]
            cur_v = first_to
            din = (cur_v[0] - start_v[0], cur_v[1] - start_v[1])
            closed = False
            guard = 0
            while not closed:
                guard += 1
                if guard > 3_000_000:
                    break
                if cur_v == start_v:
                    closed = True
                    break
                cands = out.get(cur_v)
                if not cands:
                    break
                pick = None
                for d in _turn_priority(din):
                    nxt = (cur_v[0] + d[0], cur_v[1] + d[1])
                    if nxt in cands:
                        cands.remove(nxt)
                        pick = nxt
                        break
                if pick is None:
                    break
                din = (pick[0] - cur_v[0], pick[1] - cur_v[1])
                cur_v = pick
                pts.append(cur_v)
            if not closed:
                continue
            if len(pts) > 4:
                pts = pts[:-1]          # 去掉重复的起点
                poly = _simplify(_collinear(pts), eps)
                if len(poly) >= 3 and _area(poly) >= min_area:
                    polys.append(poly)
    return polys


def _collinear(pts):
    """去掉共线中间点（含 90° 阶梯合并）"""
    n = len(pts)
    out = []
    for i in range(n):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if cross != 0:
            out.append(b)
    return out if len(out) >= 3 else pts


def _area(poly):
    s = 0.0
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        s += x0 * y1 - x1 * y0
    return abs(s) / 2.0


def _dp(pts, eps):
    if len(pts) < 3:
        return pts
    a, b = pts[0], pts[-1]
    dx, dy = b[0] - a[0], b[1] - a[1]
    norm = math.hypot(dx, dy)
    dmax, idx = 0.0, 0
    for i in range(1, len(pts) - 1):
        p = pts[i]
        if norm == 0:
            d = math.hypot(p[0] - a[0], p[1] - a[1])
        else:
            d = abs(dy * (p[0] - a[0]) - dx * (p[1] - a[1])) / norm
        if d > dmax:
            dmax, idx = d, i
    if dmax > eps:
        left = _dp(pts[:idx + 1], eps)
        right = _dp(pts[idx:], eps)
        return left[:-1] + right
    return [a, b]


def _simplify(poly, eps):
    """闭环 DP：以最远点为界拆成两段"""
    if len(poly) <= 4:
        return poly
    p0 = poly[0]
    far, fd = 1, -1
    for i, p in enumerate(poly):
        d = (p[0] - p0[0]) ** 2 + (p[1] - p0[1]) ** 2
        if d > fd:
            far, fd = i, d
    a = _dp(poly[:far + 1], eps)
    b = _dp(poly[far:] + [poly[0]], eps)
    return a[:-1] + b[:-1]


def to_path(polys, ox, oy, prec=1):
    """输出 SVG path d，坐标平移到 (ox,oy)"""
    fmt = f"{{:.{prec}f}}"
    segs = []
    for poly in polys:
        cmds = []
        for i, (x, y) in enumerate(poly):
            X = fmt.format(x - ox).rstrip("0").rstrip(".") or "0"
            Y = fmt.format(y - oy).rstrip("0").rstrip(".") or "0"
            cmds.append(("M" if i == 0 else "L") + X + " " + Y)
        segs.append("".join(cmds) + "Z")
    return "".join(segs)


def render_back(polys, shape, ox, oy, ss=1):
    """按奇偶规则回栅格化，用于与原掩膜比对"""
    from PIL import Image, ImageDraw
    h, w = shape
    im = Image.new("1", (w, h), 0)
    d = ImageDraw.Draw(im)
    # 按 |面积| 降序，判断嵌套深度
    items = sorted(polys, key=lambda p: -_area(p))
    depth = []
    for i, p in enumerate(items):
        pt = ((p[0][0] - ox) * ss, (p[0][1] - oy) * ss)
        dep = 0
        for j in range(i):
            if _inside(pt, [((q[0] - ox) * ss, (q[1] - oy) * ss) for q in items[j]]):
                dep += 1
        depth.append(dep)
    for p, dep in zip(items, depth):
        coords = [((x - ox) * ss, (y - oy) * ss) for x, y in p]
        d.polygon(coords, fill=(0 if dep % 2 else 1))
    return np.asarray(im, bool)


def _inside(pt, poly):
    x, y = pt
    n = len(poly)
    c = False
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y) and x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-12) + x0:
            c = not c
    return c
