"""Coordinate assignment for the two cones of Fig. 1a.

Levels (y) are fixed by the data: level = longest path from a PI. Only the horizontal positions are
chosen here, by simulated annealing on a 0.05 cm grid, so that the drawing is geometrically honest:
  * no edge passes over a node it does not connect (clearance >= CLEAR to every other node),
  * every edge leaves a visible shaft between its two end nodes (>= MIN_SHAFT after the arrow head),
  * the two fanin edges of a gate, and edges leaving one node, are separated by an angle >= MIN_ANG,
    so arrow heads and shafts never lie on top of each other,
  * as few crossings as possible, then the narrowest drawing.
The result is written to fig1_layout.json (cm, relative to the cone's own origin) and is read by gen_e.py;
verify_geom() is the same check gen_e.py asserts at build time.
"""
import json, math, random, os, sys

R, R_GRAY, PI_S = 0.12, 0.0897, 0.16   # R raised from 0.115 (gate numbers clear the ring); R_GRAY kept
LEVEL = 0.36
CLEAR = 0.07          # min gap between an edge and a node it does not touch (cm)
MIN_SHAFT = 0.10      # min visible shaft, after the arrow head (cm)
HEAD = 0.092          # arrow head length (2.6pt)
MIN_ANG = 24.0        # (reported only)
HEADW = 2.4 / 28.4528 # arrow head width (cm)
MIN_HEADGAP = 0.015   # min gap between the backs of two arrow heads entering one node (cm)
MIN_XANG = 30.0       # crossings must be at >= 30 degrees ...
XNODE = 0.06          # ... and at least this far from any node outline and from any other crossing (cm)
MIN_OANG = 15.0       # degrees between two edges leaving one node (shafts stay apart at the outline)
MIN_SEP = 2 * R + 0.10  # min centre distance of two nodes on one level
GRID = 0.05
NSEED, ITERS = 16, 60000


def radius(kind):
    return {'pi': PI_S / 2, 'gray': R_GRAY, 'gate': R}[kind]


def boundary(kind, dx, dy):
    """distance from the centre to the node outline along direction (dx, dy) (unit vector)"""
    if kind == 'pi':                       # square
        h = PI_S / 2
        return h / max(abs(dx), abs(dy))
    return radius(kind)


def seg_dist(p, a, b):
    ax, ay = a; bx, by = b; px, py = p
    vx, vy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy)))
    return math.hypot(ax + t * vx - px, ay + t * vy - py)


def cross(a, b, c, d):
    def o(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    return (o(a, b, c) * o(a, b, d) < 0) and (o(c, d, a) * o(c, d, b) < 0)


def ang(u, v):
    c = (u[0] * v[0] + u[1] * v[1]) / (math.hypot(*u) * math.hypot(*v))
    return math.degrees(math.acos(max(-1, min(1, c))))


def metrics(P, kinds, edges):
    """returns dict of the geometric quantities that the figure must satisfy"""
    worst_clear, worst_shaft, worst_ang, ncross = 9.0, 9.0, 180.0, 0
    worst_hgap, xbad = 9.0, 0
    for a, b, _ in edges:
        for c in P:
            if c in (a, b):
                continue
            d = seg_dist(P[c], P[a], P[b]) - radius(kinds[c]) * (1.0 if kinds[c] != 'pi' else math.sqrt(2))
            worst_clear = min(worst_clear, d)
        dx, dy = P[b][0] - P[a][0], P[b][1] - P[a][1]
        L = math.hypot(dx, dy)
        ux, uy = dx / L, dy / L
        shaft = L - boundary(kinds[a], ux, uy) - boundary(kinds[b], ux, uy) - HEAD
        worst_shaft = min(worst_shaft, shaft)
    ins, outs = {}, {}
    for a, b, _ in edges:
        ins.setdefault(b, []).append(a)
        outs.setdefault(a, []).append(b)
    for b, fs in ins.items():
        for i in range(len(fs)):
            for j in range(i + 1, len(fs)):
                u = (P[fs[i]][0] - P[b][0], P[fs[i]][1] - P[b][1])
                v = (P[fs[j]][0] - P[b][0], P[fs[j]][1] - P[b][1])
                a_ = ang(u, v)
                worst_ang = min(worst_ang, a_)
                rb = radius(kinds[b])
                worst_hgap = min(worst_hgap, 2 * (rb + HEAD) * math.sin(math.radians(a_) / 2) - HEADW)
    worst_oang = 180.0
    for a, gs in outs.items():
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                u = (P[gs[i]][0] - P[a][0], P[gs[i]][1] - P[a][1])
                v = (P[gs[j]][0] - P[a][0], P[gs[j]][1] - P[a][1])
                worst_oang = min(worst_oang, ang(u, v))
    E = [(P[a], P[b], a, b) for a, b, _ in edges]
    xpts = []
    for i in range(len(E)):
        for j in range(i + 1, len(E)):
            if len({E[i][2], E[i][3], E[j][2], E[j][3]}) == 4 and cross(E[i][0], E[i][1], E[j][0], E[j][1]):
                ncross += 1
                (ax, ay), (bx, by) = E[i][0], E[i][1]
                (cx, cy), (dx_, dy_) = E[j][0], E[j][1]
                r_ = (bx - ax, by - ay); s_ = (dx_ - cx, dy_ - cy)
                den = r_[0] * s_[1] - r_[1] * s_[0]
                t = ((cx - ax) * s_[1] - (cy - ay) * s_[0]) / den
                X_ = (ax + t * r_[0], ay + t * r_[1])
                xa = ang(r_, s_); xa = min(xa, 180 - xa)
                near = min(math.hypot(X_[0] - P[c][0], X_[1] - P[c][1]) - radius(kinds[c]) * (math.sqrt(2) if kinds[c] == 'pi' else 1) for c in P)
                if xa < MIN_XANG or near < XNODE:
                    xbad += 1
                xpts.append(X_)
    for i in range(len(xpts)):
        for j in range(i + 1, len(xpts)):
            if math.hypot(xpts[i][0] - xpts[j][0], xpts[i][1] - xpts[j][1]) < XNODE + 0.04:
                xbad += 1
    worst_sep = 9.0
    nodes = list(P)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if abs(P[nodes[i]][1] - P[nodes[j]][1]) < 1e-6:
                worst_sep = min(worst_sep, abs(P[nodes[i]][0] - P[nodes[j]][0]))
    xs = [p[0] for p in P.values()]
    return dict(clear=worst_clear, shaft=worst_shaft, ang=worst_ang, hgap=worst_hgap, oang=worst_oang, cross=ncross, xbad=xbad, sep=worst_sep,
                width=max(xs) - min(xs))


def cost(m, root_x, wmax):
    c = 0.0
    c += 400 * max(0, CLEAR - m['clear'])
    c += 400 * max(0, MIN_SHAFT - m['shaft'])
    c += 400 * max(0, MIN_HEADGAP - m['hgap'])
    c += 4.0 * m['xbad']
    c += 4 * max(0, MIN_OANG - m['oang'])
    c += 400 * max(0, MIN_SEP - m['sep'])
    c += 30 * max(0, m['width'] - wmax)
    c += 6.0 * m["cross"]
    c += 0.3 * m['width'] + 0.2 * abs(root_x)
    return c


def solve(levels, kinds, edges, root, wmax, init, seed=1, iters=30000):
    """annealing over x only; moves: shift one node, shift a whole level, swap two neighbours"""
    rnd = random.Random(seed)
    nodes = sorted(levels)
    byl = {}
    for v in nodes:
        byl.setdefault(levels[v], []).append(v)
    X = {v: round(init[v] / GRID) * GRID for v in nodes}
    def P_of(X):
        return {v: (X[v], levels[v] * LEVEL) for v in nodes}
    def total(X):
        m = metrics(P_of(X), kinds, edges)
        xs = list(X.values())
        mid = (max(xs) + min(xs)) / 2
        return cost(m, X[root] - mid, wmax), m
    cur, m = total(X)
    best, bestX = cur, dict(X)
    T0 = 0.6
    for it in range(iters):
        T = T0 * (1 - it / iters) + 1e-3
        old = dict(X)
        r = rnd.random()
        if r < 0.7:
            v = rnd.choice(nodes)
            X[v] = round((X[v] + rnd.choice([-2, -1, 1, 2]) * GRID) / GRID) * GRID
        elif r < 0.85:
            Lv = rnd.choice(list(byl))
            d = rnd.choice([-1, 1]) * GRID
            for v in byl[Lv]:
                X[v] = round((X[v] + d) / GRID) * GRID
        else:
            Lv = rnd.choice([l for l in byl if len(byl[l]) > 1] or [0])
            vs = sorted(byl[Lv], key=lambda v: X[v])
            if len(vs) > 1:
                i = rnd.randrange(len(vs) - 1)
                X[vs[i]], X[vs[i + 1]] = X[vs[i + 1]], X[vs[i]]
        new, _ = total(X)
        if new <= cur or rnd.random() < math.exp((cur - new) / T):
            cur = new
            if new < best:
                best, bestX = new, dict(X)
        else:
            X = old
    xs = list(bestX.values())
    mid = (max(xs) + min(xs)) / 2
    bestX = {v: round(x - mid, 3) for v, x in bestX.items()}
    return bestX, metrics(P_of(bestX), kinds, edges)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    __file__ = 'gen_e.py'
    src = open('gen_e.py').read().split('W, H = 18.00')[0]
    exec(src)
    out = {}
    for name, cone, wmax, labelled in (('orig', O, 2.25, None), ('var', Vc, 2.75, None)):
        pis = set(cone['pis'])
        levels = {int(k): round(p[1] * cone['maxl']) for k, p in cone['pos'].items()}
        if name == 'var':
            proven = {b for _, b, _ in PAIRS}
            kinds = {v: 'pi' if v in pis else ('gate' if v in proven else 'gray') for v in levels}
        else:
            kinds = {v: 'pi' if v in pis else 'gate' for v in levels}
        best = None
        init = {int(k): p[0] * 2.55 for k, p in cone['pos'].items()}
        from multiprocessing import Pool
        with Pool(min(NSEED, os.cpu_count())) as pool:
            res = pool.starmap(solve, [(levels, kinds, cone['edges'], cone['root'], wmax, init, seed, ITERS)
                                       for seed in range(1, NSEED + 1)])
        for seed, (X, m) in enumerate(res, 1):
            c = cost(m, 0, wmax)
            print(name, seed, {k: round(v, 3) for k, v in m.items()}, round(c, 3), flush=True)
            if best is None or c < best[0]:
                best = (c, X, m)
        out[name + '_all'] = [dict(x={str(k): v for k, v in X.items()}, metrics=m) for X, m in res]
        out[name] = dict(x={str(k): v for k, v in best[1].items()}, metrics=best[2],
                         params=dict(R=R, R_GRAY=R_GRAY, PI_S=PI_S, LEVEL=LEVEL))
        print('BEST', name, best[2])
    json.dump(out, open('fig1_layout.json', 'w'), indent=1)
