"""Independent check for questions/electrostatics/coulomb-null-point.

* Null point: scans E_x on the x axis (vector Coulomb sum in SI units) over [-12d, 13d], brackets every sign change
  that is not a charge singularity, refines by bisection, and requires exactly one zero; its location fixes the
  region key of (a) and the value of (b) (server: closed-form square-root formula).
* (c): E = -grad V at P by central finite differences of the scalar potential (server: vector Coulomb sum).
* (d): rescales both charges by the variant's factor and re-finds the zero.
* (e): every displayed statement's truth is recomputed from numbers (fields at the null point, at the probe point,
  at large distance, along the segment between the charges, a released test charge versus its field line).
"""
import math

import numpy as np

EPS0 = 8.854e-12
K = 1.0 / (4.0 * math.pi * EPS0)


def _E(charges, pts):
    """Field (V/m) at points pts (N x 2, metres) of point charges [(q, x, y)]."""
    pts = np.atleast_2d(pts).astype(float)
    E = np.zeros_like(pts)
    for q, x, y in charges:
        R = pts - np.array([x, y])
        r = np.linalg.norm(R, axis=1)[:, None]
        E += K * q * R / r ** 3
    return E


def _V(charges, pt):
    return sum(K * q / math.hypot(pt[0] - x, pt[1] - y) for q, x, y in charges)


def _zeros_on_axis(charges, d):
    """All zeros of E_x on the x axis away from the charges (metres)."""
    xs_charge = sorted(x for _, x, _ in charges)
    edges = [-12 * d] + xs_charge + [13 * d]
    roots = []
    for a, b in zip(edges[:-1], edges[1:]):
        lo, hi = a + 1e-6 * d, b - 1e-6 * d
        xs = np.linspace(lo, hi, 20001)
        ex = _E(charges, np.c_[xs, np.zeros_like(xs)])[:, 0]
        idx = np.nonzero(np.sign(ex[:-1]) * np.sign(ex[1:]) < 0)[0]
        for i in idx:
            u, w = xs[i], xs[i + 1]
            fu = _E(charges, [[u, 0]])[0, 0]
            for _ in range(200):
                mid = 0.5 * (u + w)
                fm = _E(charges, [[mid, 0]])[0, 0]
                if fm == 0:
                    u = w = mid
                    break
                if np.sign(fm) == np.sign(fu):
                    u, fu = mid, fm
                else:
                    w = mid
            roots.append(0.5 * (u + w))
    return roots


def _released_charge_leaves_field_line():
    """A + test charge released from rest near two charges drifts off the (curved) field line through its start."""
    ch = [(1e-9, 0.0, 0.0), (1e-9, 0.1, 0.0)]
    start = np.array([0.03, 0.04])
    qm = 1e-3                                       # q/m of the test charge (C/kg); only the path shape matters
    # trajectory: x'' = (q/m) E, RK4
    def acc(p):
        return qm * _E(ch, p)[0]
    p, v, h = start.copy(), np.zeros(2), 2e-4
    path = [p.copy()]
    for _ in range(4000):
        k1v = acc(p); k1p = v
        k2v = acc(p + 0.5 * h * k1p); k2p = v + 0.5 * h * k1v
        k3v = acc(p + 0.5 * h * k2p); k3p = v + 0.5 * h * k2v
        k4v = acc(p + h * k3p); k4p = v + h * k3v
        p = p + h * (k1p + 2 * k2p + 2 * k3p + k4p) / 6
        v = v + h * (k1v + 2 * k2v + 2 * k3v + k4v) / 6
        path.append(p.copy())
        if np.linalg.norm(p - start) > 0.15:
            break
    path = np.array(path)
    # field line through the start: dp/ds = E/|E|
    q, s_step = start.copy(), 1e-4
    line = [q.copy()]
    for _ in range(3000):
        e = _E(ch, q)[0]; q = q + s_step * e / np.linalg.norm(e)
        line.append(q.copy())
    line = np.array(line)
    # distance from the trajectory's last point to the field line
    dist = np.min(np.linalg.norm(line - path[-1], axis=1))
    return dist > 1e-3                              # metres: clearly off the line


_LEAVES_LINE = _released_charge_leaves_field_line()


def phys(d, check):
    p, c = d["params"], d["correct_answers"]
    q1, q2, dd = p["q1"] * 1e-9, p["q2"] * 1e-9, p["d"] * 1e-2
    ch = [(q1, 0.0, 0.0), (q2, dd, 0.0)]
    check(abs(abs(q1) - abs(q2)) > 1e-15, "null: equal magnitudes")
    check(abs(abs(q2 / q1) - (p["n"] / p["m"]) ** 2) < 1e-12, "null: ratio is not the stated perfect square")
    # ---------------------------------------------------------------- (a), (b)
    roots = _zeros_on_axis(ch, dd)
    check(len(roots) == 1, f"null: {len(roots)} zeros of E_x on the axis, expected 1 ({roots})")
    if len(roots) != 1:
        return
    x0 = roots[0]
    check(abs(x0 * 100 - c["x0"]) <= 1e-7 * max(1, abs(x0 * 100)), f"null: x0 {c['x0']} vs scan {x0*100}")
    region = "left" if x0 < 0 else ("between" if x0 < dd else "right")
    keyed = [x["id"] for x in p["region_choices"] if x["correct"] == "true"]
    check(keyed == [region], f"null: region keyed {keyed}, scan says {region}")
    check(len(p["region_choices"]) == 5, "null: region choices")
    # the 'two points' choice: the squared balance has two roots but only one is a field zero
    # |q1| (x-d)^2 = |q2| x^2  ->  quadratic in x; one root is x0, the other is the spurious one shown in the feedback
    rts = np.roots([abs(q1) - abs(q2), -2 * abs(q1) * dd, abs(q1) * dd * dd]).real
    xo = rts[np.argmax(np.abs(rts - x0))]
    check(np.min(np.abs(rts - x0)) < 1e-9, "null: x0 is not a root of the squared balance")
    E1o, E2o = _E(ch[:1], [[xo, 0]])[0, 0], _E(ch[1:], [[xo, 0]])[0, 0]
    check(abs(abs(E1o) - abs(E2o)) < 1e-9 * abs(E1o) and E1o * E2o > 0, "null: 'other root' is not an equal-magnitude same-direction point")
    check(abs(float(p["x_other_str"]) - xo * 100) <= 6e-4 * abs(xo * 100), f"null: displayed other root {p['x_other_str']} vs {xo*100}")
    # ---------------------------------------------------------------- (c) by -grad V
    P = np.array([p["xp"], p["yp"]]) * 1e-2
    h = 1e-7
    Ex = -(_V(ch, P + [h, 0]) - _V(ch, P - [h, 0])) / (2 * h)
    Ey = -(_V(ch, P + [0, h]) - _V(ch, P - [0, h])) / (2 * h)
    check(abs(Ex / 1e3 - c["Ex"]) <= 1e-5 * math.hypot(Ex, Ey) / 1e3, f"null: Ex {c['Ex']} vs -dV/dx {Ex/1e3}")
    check(abs(Ey / 1e3 - c["Ey"]) <= 1e-5 * math.hypot(Ex, Ey) / 1e3, f"null: Ey {c['Ey']} vs -dV/dy {Ey/1e3}")
    # ---------------------------------------------------------------- (d)
    k = p["kfac"]
    r2 = _zeros_on_axis([(k * q1, 0.0, 0.0), (k * q2, dd, 0.0)], dd)
    moved = not (len(r2) == 1 and abs(r2[0] - x0) < 1e-9)
    keyed = [x["id"] for x in p["scale_choices"] if x["correct"] == "true"]
    check(keyed == ["same"] and not moved, f"null: scaling key {keyed}, null moved: {moved}")
    check(len(p["scale_choices"]) == 5 and len({x["text"] for x in p["scale_choices"]}) == 5, "null: scale choices")
    # ---------------------------------------------------------------- (e)
    E1P, E2P = _E(ch[:1], [P])[0], _E(ch[1:], [P])[0]
    E1n, E2n = _E(ch[:1], [[x0, 0]])[0], _E(ch[1:], [[x0, 0]])[0]
    far, expo = [], []
    for ang in (0.3, 1.9, 4.0):
        R = 1e4 * dd
        u = np.array([math.cos(ang), math.sin(ang)])
        e_R, e_2R = _E(ch, [R * u])[0], _E(ch, [2 * R * u])[0]
        far.append(np.linalg.norm(e_R - K * (q1 + q2) * u / R ** 2) / np.linalg.norm(e_R))   # vs a point charge q1+q2
        expo.append(-math.log2(np.linalg.norm(e_2R) / np.linalg.norm(e_R)))                  # fall-off exponent
    xs = np.linspace(0.001 * dd, 0.999 * dd, 4001)
    exb = _E(ch, np.c_[xs, np.zeros_like(xs)])[:, 0]
    truth = {
        "vector_sum": bool(np.allclose(_E(ch, [P])[0], E1P + E2P)),
        "mag_sum": bool(abs(np.linalg.norm(E1P + E2P) - (np.linalg.norm(E1P) + np.linalg.norm(E2P))) < 1e-9 * np.linalg.norm(E1P)),
        "begin_end": True,                     # drawing convention (Lecture 2)
        "lines_both": q1 > 0 and q2 > 0,
        "density": True,                       # drawing convention (Lecture 2)
        "path": not _LEAVES_LINE,
        "cross_null": True,                    # E has one direction wherever it is nonzero
        "test_sign": False,                    # F/q is the same for test charges of either sign:
        "null_equal": bool(abs(np.linalg.norm(E1n) - np.linalg.norm(E2n)) < 1e-6 * np.linalg.norm(E1n) and np.dot(E1n, E2n) < 0),
        "both_zero": bool(np.linalg.norm(E1n) < 1e-12),
        "closer_smaller": (abs(x0) < abs(x0 - dd)) == (abs(q1) < abs(q2)),
        "far": bool(all(r < 1e-2 for r in far)),
        "far_dipole": bool(all(abs(e - 3) < 0.1 for e in expo)),
        "Ex_between": bool(np.all(exb > 0) or np.all(exb < 0)),
    }
    for qt in (1.6e-19, -1.6e-19):                # E = F/q identical for either sign of test charge
        F = qt * _E(ch, [P])[0]
        check(np.allclose(F / qt, _E(ch, [P])[0]), "null: F/q depends on the test charge")
    facts = p["facts"]
    check(len(facts) == 5 and 1 <= sum(f["correct"] == "true" for f in facts) <= 4, "null: statement count")
    check(len({f["text"] for f in facts}) == 5, "null: duplicate statements")
    for f in facts:
        check(f["id"] in truth, f"null: unknown statement {f['id']}")
        if f["id"] in truth:
            check((f["correct"] == "true") == truth[f["id"]], f"null: statement {f['id']} keyed {f['correct']}, physics {truth[f['id']]}")
            check(f["verdict"] == ("True" if truth[f["id"]] else "False"), f"null: verdict text {f['id']}")
