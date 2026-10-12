"""Independent check for questions/electrostatics/gauss-symmetry.

No Gauss's law anywhere in the field computations: every field is a Coulomb superposition evaluated by quadrature.
  * spherical objects: thin shells as sums of rings (Gauss-Legendre in the polar angle), balls as sums of shells;
  * cylindrical objects: infinite lines (Coulomb integral along the line), cylindrical shells as rings of lines,
    solid cylinders as sums of cylindrical shells;
  * planar objects: sheets as very large disks (sum of rings), slabs as stacks of sheets.
(a) every matching key is re-derived by measuring the distance exponent of |E| at two field points (or E = 0).
(b) the correct choice must be the one non-symmetric distribution (sets written independently here).
(c) E_r at r1, r2, r3 from the brute-force ball + shell, and again from Gauss's law with a numerically integrated
    enclosed charge (midpoint rule) as a second route.
(d) each displayed statement's truth is recomputed from brute-force fields (changing Q_s, continuity at a and b,
    flux through spheres, signs and sizes across the shell).
"""
import math

import numpy as np
from numpy.polynomial.legendre import leggauss

EPS0 = 8.854e-12
K = 1.0 / (4.0 * math.pi * EPS0)

_xt, _wt = leggauss(400)
_TH = (_xt + 1) * math.pi / 2                    # polar angle on [0, pi], nodes cluster near the nearest point
_WTH = _wt * math.pi / 2
_CT, _ST = np.cos(_TH), np.sin(_TH)
_XR, _WR = leggauss(100)


# ------------------------------------------------------------------ spherical: rings -> shells -> balls (field on the +z axis)
def shell_E(R, Q, s):
    R = np.atleast_1d(np.asarray(R, float))[:, None]
    Q = np.atleast_1d(np.asarray(Q, float))[:, None]
    z, rho = R * _CT, R * _ST
    dQ = 0.5 * Q * _ST * _WTH
    return float((K * dQ * (s - z) / ((s - z) ** 2 + rho ** 2) ** 1.5).sum())


def ball_E(a, Q, s):
    tot = 0.0
    for lo, hi in ([(0.0, s), (s, a)] if s < a else [(0.0, a)]):
        R = lo + (_XR + 1) * (hi - lo) / 2
        w = _WR * (hi - lo) / 2
        tot += shell_E(R, Q * 3 * R ** 2 * w / a ** 3, s)
    return tot


# ------------------------------------------------------------------ cylindrical (2-D): lines -> cylindrical shells -> cylinders
def _line_coeff():
    """r E(r) / (k lambda) for an infinite line, from the Coulomb integral with z = r tan(alpha)."""
    x, w = leggauss(200)
    al = x * math.pi / 2
    return float((np.cos(al) * w * math.pi / 2).sum())        # integral of cos(alpha) over (-pi/2, pi/2)


_CLINE = _line_coeff()


def line_E(lam, r):
    return _CLINE * K * lam / r


def cylshell_E(R, lam, s):
    """Radial field at distance s from the axis of a cylindrical shell (radius R, charge lam per length)."""
    R = np.atleast_1d(np.asarray(R, float))[:, None]
    lam = np.atleast_1d(np.asarray(lam, float))[:, None]
    ph, wph = _TH, _WTH                                         # phi on [0, pi], doubled by symmetry
    dl = lam * wph / math.pi
    dx, dy = s - R * np.cos(ph), -R * np.sin(ph)
    r2 = dx ** 2 + dy ** 2
    return float((_CLINE * K * dl * dx / r2).sum())


def cyl_E(a, lam, s):
    tot = 0.0
    for lo, hi in ([(0.0, s), (s, a)] if s < a else [(0.0, a)]):
        R = lo + (_XR + 1) * (hi - lo) / 2
        w = _WR * (hi - lo) / 2
        tot += cylshell_E(R, lam * 2 * R * w / a ** 2, s)
    return tot


# ------------------------------------------------------------------ planar: sheets as huge disks (sum of rings), slabs as stacks
def sheet_E(sigma, x):
    """Normal field at signed distance x from a uniformly charged disk of radius 1e6 |x| (an 'infinite' sheet)."""
    amax = math.atan(1e6)
    al = (_xt + 1) * amax / 2
    return (1.0 if x > 0 else -1.0) * float(2 * math.pi * K * sigma * (np.sin(al) * _wt * amax / 2).sum())


def slab_E(rho, W, x):
    tot = 0.0
    cuts = sorted({-W / 2, W / 2} | ({x} if -W / 2 < x < W / 2 else set()))
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        xs = lo + (_XR + 1) * (hi - lo) / 2
        w = _WR * (hi - lo) / 2
        tot += sum(sheet_E(rho * wi, x - xi) for xi, wi in zip(xs, w))
    return tot


# ------------------------------------------------------------------ (a): measure how |E| depends on the distance
def _profile(kind):
    """(|E| at s1, |E| at s2, s1, s2, reference scale) for a matching statement id, unit sources."""
    q = 1e-9
    f = {
        "point": (lambda s: shell_E(1e-4, q, s), 1.0, 2.0),
        "ball_out": (lambda s: ball_E(1.0, q, s), 1.5, 3.0),
        "shell_out": (lambda s: shell_E(1.0, q, s), 1.5, 3.0),
        "ball_in": (lambda s: ball_E(1.0, q, s), 0.3, 0.6),
        "shell_in": (lambda s: shell_E(1.0, q, s), 0.3, 0.6),
        "shells_pm_out": (lambda s: shell_E(0.5, q, s) + shell_E(1.0, -q, s), 1.5, 3.0),
        "line": (lambda s: line_E(q, s), 1.0, 2.0),
        "cyl_out": (lambda s: cyl_E(1.0, q, s), 1.5, 3.0),
        "cyl_in": (lambda s: cyl_E(1.0, q, s), 0.3, 0.6),
        "coax_mid": (lambda s: cylshell_E(0.2, q, s) + cylshell_E(1.0, -q, s), 0.4, 0.8),
        "coax_out": (lambda s: cylshell_E(0.2, q, s) + cylshell_E(1.0, -q, s), 1.5, 3.0),
        "sheet": (lambda s: sheet_E(q, s), 1.0, 2.0),
        "slab_out": (lambda s: slab_E(q, 1.0, s), 1.0, 2.0),
        "slab_in": (lambda s: slab_E(q, 1.0, s), 0.1, 0.2),
        "sheets_pm_in": (lambda s: sheet_E(q, s) + sheet_E(-q, s - 1.0), 0.25, 0.5),
        "sheets_pm_out": (lambda s: sheet_E(q, -s) + sheet_E(-q, -s - 1.0), 1.0, 2.0),
        "sheets_pp_in": (lambda s: sheet_E(q, s) + sheet_E(q, s - 1.0), 0.25, 0.4),
    }[kind]
    fn, s1, s2 = f
    scale = K * q / max(s1, 1e-3) ** 2 + 2 * math.pi * K * q          # generous size of one constituent's field
    return abs(fn(s1)), abs(fn(s2)), s1, s2, scale


def _classify(kind):
    e1, e2, s1, s2, scale = _profile(kind)
    if max(e1, e2) < 1e-7 * scale:
        return "zero"
    p = math.log(e2 / e1) / math.log(s2 / s1)
    for key, expo in (("inv2", -2), ("inv1", -1), ("const", 0), ("lin", 1)):
        if abs(p - expo) < 0.01:
            return key
    return f"exponent {p:.3f}"


_MATCH_TRUTH = {k: _classify(k) for k in (
    "point", "ball_out", "shell_out", "ball_in", "shell_in", "shells_pm_out", "line", "cyl_out", "cyl_in",
    "coax_mid", "coax_out", "sheet", "slab_out", "slab_in", "sheets_pm_in", "sheets_pm_out", "sheets_pp_in")}

# (b) the classification of the distributions in the multiple choice (written independently of the server)
_NONSYM = {"rod", "cube", "disk", "ring", "hemi"}
_SYM = {"inf_line", "inf_sheet", "ball_rho_r", "shell", "cyl", "slab_rho_x", "coax"}


def phys(d, check):
    p, c = d["params"], d["correct_answers"]
    # ------------------------------------------------ (a)
    st = p["match_st"]
    check(len(st) == 5 and len({x["text"] for x in st}) == 5, "gauss: matching needs 5 distinct statements")
    check(len({x["key"] for x in st}) >= 3, "gauss: matching shows fewer than 3 different answers")
    for x in st:
        want = _MATCH_TRUTH.get(x["id"])
        check(x["key"] == want, f"gauss: matching {x['id']} keyed {x['key']}, brute force says {want}")
    # ------------------------------------------------ (b)
    ch = p["gauss_choices"]
    corr = [x["id"] for x in ch if x["correct"] == "true"]
    check(len(corr) == 1 and corr[0] in _NONSYM, f"gauss: (b) correct choice {corr}")
    check(all(x["id"] in _SYM for x in ch if x["correct"] == "false" and x["id"] != "none"), "gauss: (b) a distractor is not symmetric")
    check(any(x["id"] == "none" and x["correct"] == "false" for x in ch) and len(ch) == 5, "gauss: (b) 'none' choice")
    # ------------------------------------------------ (c) brute-force Coulomb
    a, b = p["a"] * 1e-2, p["b"] * 1e-2
    r1, r2, r3 = float(p["r1"]) * 1e-2, float(p["r2"]) * 1e-2, p["r3"] * 1e-2
    Qb, Qs = p["Qb"] * 1e-9, p["Qs"] * 1e-9
    check(0 < r1 < a < r2 < b < r3, "gauss: radii out of order")
    E = lambda s, qs=Qs: ball_E(a, Qb, s) + shell_E(b, qs, s)
    scale = K * abs(Qb) / b ** 2
    for key, r in (("E1", r1), ("E2", r2), ("E3", r3)):
        val = E(r) / 1e3
        check(abs(val - c[key]) <= 2e-5 * max(abs(val), 1e-3 * scale / 1e3), f"gauss: {key} {c[key]} vs Coulomb sum {val}")
        # second route: Gauss with a midpoint-rule enclosed charge
        n = 20000
        rr = (np.arange(n) + 0.5) * (r / n)
        rho = np.where(rr < a, Qb / (4 / 3 * math.pi * a ** 3), 0.0)
        qenc = float(np.sum(rho * 4 * math.pi * rr ** 2) * (r / n)) + (Qs if r > b else 0.0)
        if r > a:
            qenc = Qb + (Qs if r > b else 0.0)      # the midpoint grid does not resolve the ball edge exactly
        val2 = qenc / (4 * math.pi * EPS0 * r ** 2) / 1e3
        check(abs(val2 - c[key]) <= 1e-6 * max(abs(val2), 1e-3), f"gauss: {key} {c[key]} vs Gauss/quadrature {val2}")
    # ------------------------------------------------ (d)
    e1, e2, e3 = E(r1), E(r2), E(r3)
    dq = 1e-9
    eps = 1e-3
    Ea_in, Ea_out = E(a * (1 - eps)), E(a * (1 + eps))
    Eb_in, Eb_out = E(b * (1 - 0.01)), E(b * (1 + 0.01))
    flux = lambda r: EPS0 * E(r) * 4 * math.pi * r ** 2
    truth = {
        "qs_r1": abs(E(r1, Qs + dq) - e1) > 1e-6 * scale,
        "qs_r2": abs(E(r2, Qs + dq) - e2) > 1e-6 * scale,
        "qs_r3": abs(E(r3, Qs + dq) - e3) > 1e-6 * scale,
        "cont_a": abs(Ea_out - Ea_in) < 1e-2 * abs(Ea_in),
        "cont_b": abs(Eb_out - Eb_in) < 0.05 * max(abs(Eb_in), abs(Eb_out)),
        "flux_out": abs(flux(r3) - (Qb + Qs)) < 1e-4 * abs(Qb) and abs(flux(1.7 * r3) - (Qb + Qs)) < 1e-4 * abs(Qb),
        "flux_r1": abs(flux(r1) - Qb) < 1e-4 * abs(Qb),
        "point_r2": abs(e2 - K * Qb / r2 ** 2) < 1e-6 * abs(e2),
        "point_r1": abs(e1 - K * Qb / r1 ** 2) < 1e-6 * abs(e1),
        "louder": abs(Eb_out) > abs(Eb_in),
        "flip": abs(e3) > 1e-6 * scale and e2 * e3 < 0,
        "out_zero": abs(e3) < 1e-6 * scale and abs(E(2 * r3)) < 1e-6 * scale,
    }
    facts = p["facts"]
    check(len(facts) == 5 and 1 <= sum(f["correct"] == "true" for f in facts) <= 4, "gauss: statement count")
    check(len({f["text"] for f in facts}) == 5, "gauss: duplicate statements")
    for f in facts:
        check(f["id"] in truth, f"gauss: unknown statement {f['id']}")
        if f["id"] in truth:
            check((f["correct"] == "true") == bool(truth[f["id"]]), f"gauss: statement {f['id']} keyed {f['correct']}, brute force {truth[f['id']]}")
            check(f["verdict"] == ("True" if truth[f["id"]] else "False"), f"gauss: verdict text {f['id']}")
