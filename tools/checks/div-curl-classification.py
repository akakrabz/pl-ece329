"""Independent check for questions/electrostatics/div-curl-classification.

Works only with the field F(r) = M r evaluated numerically (no formulas in terms of M entries):
  * div and curl by central finite differences at random points (and checked to be position independent);
  * (c) the circulation by a periodic trapezoid rule around the stated circle (counterclockwise seen from +k);
  * (d) every statement re-derived from experiments: circulation around random triangles and around a small loop
    facing the curl, line integrals from the origin to (1,1,1) along straight, staircase and random detour paths,
    outward flux through random boxes (Gauss-Legendre on each face), -grad V of the potential the solution
    quotes, values of F at different points, counterexample fields for the generic claims;
  * (e) -div J by finite differences.
"""
import math

import numpy as np
from numpy.polynomial.legendre import leggauss

_rng = np.random.default_rng(2026)
_GX, _GW = leggauss(12)
_LX, _LW = leggauss(64)


def _fd_jacobian(F, r, h=1e-3):
    J = np.zeros((3, 3))
    for j in range(3):
        e = np.zeros(3); e[j] = h
        J[:, j] = (F(r + e) - F(r - e)) / (2 * h)
    return J


def _div_curl(F, r):
    J = _fd_jacobian(F, r)
    return np.trace(J), np.array([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]])


def _line_integral(F, pts):
    """Integral of F . dl along the polyline through pts (Gauss-Legendre per segment); F takes (N, 3) arrays."""
    tot = 0.0
    t = (_LX + 1) / 2
    for a, b in zip(pts[:-1], pts[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        P = a + np.outer(t, b - a)
        tot += float(np.sum(_LW / 2 * (F(P) @ (b - a))))
    return tot


def _box_flux(F, lo, hi):
    """Outward flux of F through the box [lo, hi] (Gauss-Legendre on each face)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    tot = 0.0
    for ax in range(3):
        u, v = [a for a in range(3) if a != ax]
        us = lo[u] + (_GX + 1) / 2 * (hi[u] - lo[u]); wu = _GW / 2 * (hi[u] - lo[u])
        vs = lo[v] + (_GX + 1) / 2 * (hi[v] - lo[v]); wv = _GW / 2 * (hi[v] - lo[v])
        U, Vv = np.meshgrid(us, vs, indexing="ij")
        W = np.outer(wu, wv)
        for side, sgn in ((hi[ax], 1.0), (lo[ax], -1.0)):
            P = np.zeros((U.size, 3)); P[:, ax] = side; P[:, u] = U.ravel(); P[:, v] = Vv.ravel()
            tot += sgn * float(np.sum(F(P)[:, ax] * W.ravel()))
    return tot


def _generic_facts():
    """Truths of the field-independent statements, from counterexamples."""
    t = {}
    Fx = lambda r: np.stack([np.asarray(r)[..., 0], 0 * np.asarray(r)[..., 0], 0 * np.asarray(r)[..., 0]], axis=-1)   # x x-hat
    Fy = lambda r: np.stack([np.asarray(r)[..., 1], 0 * np.asarray(r)[..., 0], 0 * np.asarray(r)[..., 0]], axis=-1)   # y x-hat
    d1, c1 = _div_curl(Fx, _rng.normal(size=3))
    d2, c2 = _div_curl(Fy, _rng.normal(size=3))
    t["curlfree_divfree"] = not (np.allclose(c1, 0) and abs(d1) > 0.5)     # a curl-free field with divergence exists
    t["divfree_curlfree"] = not (abs(d2) < 1e-9 and np.linalg.norm(c2) > 0.5)
    return t


_GENERIC = _generic_facts()


def phys(d, check):
    p, c = d["params"], d["correct_answers"]
    M = np.array(p["M"], float)
    F = lambda r: np.asarray(r, float) @ M.T
    # ------------------------------------------------ (a), (b), (e) by finite differences at two random points
    pts = _rng.uniform(-3, 3, size=(2, 3))
    (dv, cu), (dv2, cu2) = _div_curl(F, pts[0]), _div_curl(F, pts[1])
    check(abs(dv - c["div"]) < 1e-6, f"divcurl: div {c['div']} vs FD {dv}")
    for i, k in enumerate(("curlx", "curly", "curlz")):
        check(abs(cu[i] - c[k]) < 1e-6, f"divcurl: {k} {c[k]} vs FD {cu[i]}")
    check(abs(-dv - c["drho"]) < 1e-6, f"divcurl: drho {c['drho']} vs -div {-dv}")
    curl0 = np.linalg.norm(cu) < 1e-6
    div0 = abs(dv) < 1e-6
    case = {(True, True): "sym_traceless", (True, False): "sym_trace", (False, True): "curl_traceless", (False, False): "neither"}[(curl0, div0)]
    check(case == p["case"], f"divcurl: case label {p['case']} but the field is {case}")
    # ------------------------------------------------ (c) circulation around the circle, counterclockwise seen from +k
    k = p["kidx"]; i, j = (k + 1) % 3, (k + 2) % 3
    R = p["R"]
    n = 2000
    t = np.arange(n) * 2 * math.pi / n
    ei, ej = np.eye(3)[i], np.eye(3)[j]
    check(np.allclose(np.cross(ei, ej), np.eye(3)[k]), "divcurl: in-plane axes not right-handed")
    r = R * (np.outer(np.cos(t), ei) + np.outer(np.sin(t), ej))
    dr = R * (np.outer(-np.sin(t), ei) + np.outer(np.cos(t), ej))
    gam = float(np.sum(np.einsum("ij,ij->i", F(r), dr))) * 2 * math.pi / n
    ch = p["circ_choices"]
    check(len(ch) == 5 and len({x["text"] for x in ch}) == 5, "divcurl: circulation choices")
    good = [x for x in ch if abs(x["value"] * math.pi - gam) < 1e-6 * max(1, abs(gam))]
    check(len(good) == 1 and good[0]["correct"] == "true", f"divcurl: circulation {gam} V; keyed {[x['value'] for x in ch if x['correct'] == 'true']} pi")
    for x in ch:   # each text shows its own value
        v = x["value"]
        want = "$0$" if v == 0 else ("$" + ("-" if v < 0 else "") + ("" if abs(v) == 1 else str(abs(v))) + r"\pi")
        check(x["text"].startswith(want), f"divcurl: choice text {x['text']} does not show {v} pi")
    # ------------------------------------------------ (d) statements from experiments
    circs = [_line_integral(F, [a, b, cc, a]) for a, b, cc in (_rng.uniform(-2, 2, size=(3, 3)) for _ in range(4))]
    if np.linalg.norm(cu) > 1e-6:                     # a small square facing the curl
        nhat = cu / np.linalg.norm(cu)
        u = np.cross(nhat, [1.0, 0.3, -0.2]); u /= np.linalg.norm(u); v = np.cross(nhat, u)
        circs.append(_line_integral(F, [0.2 * (u + v), 0.2 * (-u + v), 0.2 * (-u - v), 0.2 * (u - v), 0.2 * (u + v)]))
    circ_zero = all(abs(g) < 1e-9 for g in circs)
    A, B = np.zeros(3), np.ones(3)
    paths = [[A, B], [A, [1, 0, 0], [1, 1, 0], B], [A, [0, 0, 1], [0, 1, 1], B]]
    for _ in range(4):                                # detours A -> A+delta -> B+delta -> B
        delta = _rng.normal(size=3)
        paths.append([A, A + delta, B + delta, B])
    vals = [_line_integral(F, pth) for pth in paths]
    path_ind = max(vals) - min(vals) < 1e-9
    boxes = [np.sort(_rng.uniform(-2, 2, size=(2, 3)), axis=0) for _ in range(3)]
    fluxes = [_box_flux(F, b[0], b[1]) for b in boxes]
    flux_zero = all(abs(f) < 1e-9 for f in fluxes)
    for b, f in zip(boxes, fluxes):                   # divergence theorem consistency
        check(abs(f - dv * np.prod(b[1] - b[0])) < 1e-8, "divcurl: box flux != div x volume")
    potential_ok = False
    if p["Vc"]:
        a_, b_, c_, dxy, dxz, dyz = p["Vc"]
        V = lambda r: a_ * r[0] ** 2 + b_ * r[1] ** 2 + c_ * r[2] ** 2 + dxy * r[0] * r[1] + dxz * r[0] * r[2] + dyz * r[1] * r[2]
        h = 1e-4
        potential_ok = True
        for rr in pts:
            g = np.array([(V(rr + e) - V(rr - e)) / (2 * h) for e in np.eye(3) * h])
            potential_ok = potential_ok and bool(np.allclose(-g, F(rr), atol=1e-7))
        check(potential_ok, "divcurl: the quoted potential V does not give F = -grad V")
    dshift, cshift = _div_curl(lambda r: F(r) + np.array([3.0, -1.0, 2.0]), pts[0])
    truth = {
        "static_E": circ_zero and path_ind,
        "potential": path_ind and circ_zero,
        "circ_zero": circ_zero,
        "path_indep": path_ind,
        "changing_B": not circ_zero,                  # Faraday: needs curl E = -dB/dt != 0
        "could_be_B": flux_zero,
        "flux_zero": flux_zero,
        "rho_uniform": (not flux_zero) and abs(dv - dv2) < 1e-6,
        "charge_free": flux_zero,
        "uniform_field": bool(np.allclose(F(pts[0]), F(pts[1]))),
        "derivs_const": abs(dv - dv2) < 1e-6 and bool(np.allclose(cu, cu2, atol=1e-6)),
        "curlfree_divfree": _GENERIC["curlfree_divfree"],
        "divfree_curlfree": _GENERIC["divfree_curlfree"],
        "add_const": not (abs(dshift - dv) < 1e-6 and np.allclose(cshift, cu, atol=1e-6)),
    }
    check(truth["potential"] == potential_ok, "divcurl: potential quoted iff the field is conservative")
    check(truth["circ_zero"] == curl0 and truth["flux_zero"] == div0, "divcurl: experiments disagree with div/curl")
    facts = p["facts"]
    check(len(facts) == 5 and 1 <= sum(f["correct"] == "true" for f in facts) <= 4, "divcurl: statement count")
    check(len({f["text"] for f in facts}) == 5, "divcurl: duplicate statements")
    for f in facts:
        check(f["id"] in truth, f"divcurl: unknown statement {f['id']}")
        if f["id"] in truth:
            check((f["correct"] == "true") == bool(truth[f["id"]]), f"divcurl: statement {f['id']} keyed {f['correct']}, experiment {truth[f['id']]}")
            check(f["verdict"] == ("True" if truth[f["id"]] else "False"), f"divcurl: verdict text {f['id']}")
