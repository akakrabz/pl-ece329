"""Independent check for ampere-symmetry-toroid (Lecture 13).

Routes (none of them uses Ampere's law in closed form):
  * toroid: Biot-Savart sum over the N discrete rectangular turns (closed-form straight-segment fields)
    -> H_phi at r0, the 1/r profile, and H = 0 at the point of part (c); flux by numerical quadrature of that
    profile over the rectangle (not the log formula);
  * matching: every statement is classified by computing |H| at two distances from a numerical model of the
    distribution (Biot-Savart sums over filaments, loops or turns) and reading off the power law;
  * vector potential: the Coulomb integral sum mu0 I dl / (4 pi R) over a closed rectangular circuit;
  * checkbox: each statement's truth is recomputed (finite-difference curls/divergences, Stokes on a square,
    Biot-Savart for the toroid statements).
"""
import math

import numpy as np

MU0 = 4e-7 * math.pi


# ---------------------------------------------------------------- Biot-Savart / Coulomb-integral primitives
def seg_H(P, A, B, I):
    """H at P from straight segments A[i] -> B[i] carrying I[i] (closed form of the Biot-Savart integral)."""
    P = np.asarray(P, float)
    u = B - A
    L = np.linalg.norm(u, axis=1)
    u = u / L[:, None]
    a1, a2 = P - A, P - B
    c1 = np.einsum("ij,ij->i", a1, u)
    c2 = np.einsum("ij,ij->i", a2, u)
    rho = a1 - c1[:, None] * u
    d = np.linalg.norm(rho, axis=1)
    fac = c1 / np.linalg.norm(a1, axis=1) - c2 / np.linalg.norm(a2, axis=1)
    dirv = np.cross(u, rho) / d[:, None]
    return np.sum((I * fac / (4 * np.pi * d))[:, None] * dirv, axis=0)


def seg_A(P, A, B, I):
    """Coulomb-gauge vector potential at P: sum of mu0 I/(4 pi) * integral dl/|P - s| over straight segments."""
    P = np.asarray(P, float)
    u = B - A
    L = np.linalg.norm(u, axis=1)
    u = u / L[:, None]
    a1 = P - A
    c1 = np.einsum("ij,ij->i", a1, u)
    d = np.linalg.norm(a1 - c1[:, None] * u, axis=1)
    integ = np.arcsinh(c1 / d) - np.arcsinh((c1 - L) / d)
    return np.sum((MU0 * I / (4 * np.pi) * integ)[:, None] * u, axis=0)


def filaments_H(px, py, fx, fy, fI):
    """2-D: H (Hx, Hy) at (px, py) from infinite filaments along +z (Biot-Savart for an infinite line)."""
    dx, dy = px - fx, py - fy
    r2 = dx * dx + dy * dy
    return np.array([np.sum(-fI * dy / (2 * np.pi * r2)), np.sum(fI * dx / (2 * np.pi * r2))])


def toroid(N, a, b, h, s, I):
    """N rectangular turns in the planes phi_k = 2 pi k / N: up the inner wall (s = +1), out along the top,
    down the outer wall, in along the bottom."""
    phi = 2 * np.pi * np.arange(N) / N
    c, sn = np.cos(phi), np.sin(phi)

    def pt(r, z):
        return np.stack([r * c, r * sn, np.full(N, z)], axis=1)

    p1, p2, p3, p4 = pt(a, -h / 2), pt(a, h / 2), pt(b, h / 2), pt(b, -h / 2)
    return np.concatenate([p1, p2, p3, p4]), np.concatenate([p2, p3, p4, p1]), np.full(4 * N, float(s * I))


def polygon_loops(R, zs, M, I):
    """Circular loops of radius R (M-gons, counterclockwise seen from +z) at heights zs."""
    th = 2 * np.pi * np.arange(M + 1) / M
    xs, ys = R * np.cos(th), R * np.sin(th)
    A, B = [], []
    for z in zs:
        A.append(np.stack([xs[:-1], ys[:-1], np.full(M, z)], axis=1))
        B.append(np.stack([xs[1:], ys[1:], np.full(M, z)], axis=1))
    A, B = np.concatenate(A), np.concatenate(B)
    return A, B, np.full(len(A), float(I))


def H_phi_at(Ab, r, z, phi):
    A, B, I = Ab
    H = seg_H([r * math.cos(phi), r * math.sin(phi), z], A, B, I)
    return float(np.dot(H, [-math.sin(phi), math.cos(phi), 0.0])), H


# ---------------------------------------------------------------- matching: classify each statement numerically
def _classify(h1, h2, d1, d2, scale):
    if max(abs(h1), abs(h2)) < 1e-3 * scale:
        return "zero"
    p = math.log(abs(h2) / abs(h1)) / math.log(d2 / d1)
    for key, pe in (("lin_r", 1), ("uniform", 0), ("inv_r", -1), ("inv_r2", -2), ("inv_r3", -3)):
        if abs(p - pe) < 0.1:
            return key
    return f"unclassified(p={p:.3f})"


def _polar_filaments(r_in, r_out, n_r, n_phi, total):
    rr = r_in + (np.arange(n_r) + 0.5) * (r_out - r_in) / n_r
    pp = 2 * np.pi * (np.arange(n_phi) + 0.5) / n_phi
    R, PH = np.meshgrid(rr, pp)
    w = R.ravel()
    return R.ravel() * np.cos(PH.ravel()), R.ravel() * np.sin(PH.ravel()), total * w / w.sum()


def _match_table():
    out = {}
    mag = lambda v: float(np.linalg.norm(v))  # noqa: E731
    # long straight wire (finite segment 2000 long, field points near its middle)
    A, B, I = np.array([[0, 0, -1000.0]]), np.array([[0, 0, 1000.0]]), np.array([1.0])
    out["wire_out"] = _classify(mag(seg_H([1, 0, 0], A, B, I)), mag(seg_H([2, 0, 0], A, B, I)), 1, 2, 1 / (2 * np.pi))
    # uniform round wire of radius 1 (filaments on a polar grid; field points between grid nodes)
    fx, fy, fI = _polar_filaments(0, 1, 120, 240, 1.0)
    h1, h2 = mag(filaments_H(0.3, 0, fx, fy, fI)), mag(filaments_H(0.6, 0, fx, fy, fI))
    out["wire_in"] = _classify(h1, h2, 0.3, 0.6, 1 / (2 * np.pi))
    # coax: core radius 1 (+1 A), shell 2..2.5 (-1 A)
    sx, sy, sI = _polar_filaments(2, 2.5, 30, 240, -1.0)
    cx, cy, cI = np.concatenate([fx, sx]), np.concatenate([fy, sy]), np.concatenate([fI, sI])
    out["coax_mid"] = _classify(mag(filaments_H(1.2, 0, cx, cy, cI)), mag(filaments_H(1.8, 0, cx, cy, cI)), 1.2, 1.8,
                                1 / (2 * np.pi))
    out["coax_out"] = _classify(mag(filaments_H(3, 0, cx, cy, cI)), mag(filaments_H(6, 0, cx, cy, cI)), 3, 6,
                                1 / (2 * np.pi * 3))
    # sheet on x = 0 (J_s = 1 A/m), 4000 m wide
    ys = -2000 + (np.arange(8000) + 0.5) * 0.5
    zeros = np.zeros_like(ys)
    sh = (zeros, ys, np.full_like(ys, 0.5))
    out["sheet"] = _classify(mag(filaments_H(1, 0, *sh)), mag(filaments_H(2, 0, *sh)), 1, 2, 0.5)
    # two sheets: +J_s on x = 0, -J_s on x = 2
    two = (np.concatenate([zeros, zeros + 2]), np.concatenate([ys, ys]),
           np.concatenate([np.full_like(ys, 0.5), np.full_like(ys, -0.5)]))
    out["two_in"] = _classify(mag(filaments_H(0.5, 0, *two)), mag(filaments_H(1.0, 0, *two)), 0.5, 1.0, 1.0)
    out["two_out"] = _classify(mag(filaments_H(3, 0, *two)), mag(filaments_H(4, 0, *two)), 3, 4, 1.0)
    # slab |x| < 1 with J0 = 1 A/m^2: lattice of filaments, field points at interstitial sites
    xs = -1 + (np.arange(50) + 0.5) * 0.04
    yy = -400 + (np.arange(4000) + 0.5) * 0.2
    X, Y = np.meshgrid(xs, yy)
    sl = (X.ravel(), Y.ravel(), np.full(X.size, 0.04 * 0.2))
    out["slab_in"] = _classify(mag(filaments_H(0.24, 0, *sl)), mag(filaments_H(0.48, 0, *sl)), 0.24, 0.48, 1.0)
    out["slab_out"] = _classify(mag(filaments_H(2, 0, *sl)), mag(filaments_H(4, 0, *sl)), 2, 4, 1.0)
    # solenoid: radius 1, 10 turns per unit length, 200 long, 1 A (n I = 10)
    zs = -100 + (np.arange(2000) + 0.5) * 0.1
    sol = polygon_loops(1.0, zs, 96, 1.0)
    out["sol_in"] = _classify(mag(seg_H([0.2, 0, 0], *sol)), mag(seg_H([0.6, 0, 0], *sol)), 0.2, 0.6, 10.0)
    out["sol_out"] = _classify(mag(seg_H([1.5, 0, 0], *sol)), mag(seg_H([3.0, 0, 0], *sol)), 1.5, 3.0, 10.0)
    # toroid: N = 400, a = 1, b = 2, h = 1
    tor = toroid(400, 1.0, 2.0, 1.0, 1, 1.0)
    ph = math.pi / 400
    out["tor_in"] = _classify(abs(H_phi_at(tor, 1.25, 0, ph)[0]), abs(H_phi_at(tor, 1.75, 0, ph)[0]), 1.25, 1.75,
                              400 / (2 * np.pi))
    hole = mag(H_phi_at(tor, 0.5, 0, ph)[1])
    beyond = mag(H_phi_at(tor, 3.0, 0, ph)[1])
    out["tor_out"] = "zero" if max(hole, beyond) < 1e-6 * 400 / (2 * np.pi * 2) else "nonzero"
    # small loop (radius 1) far along its axis
    loop = polygon_loops(1.0, [0.0], 720, 1.0)
    out["loop_far"] = _classify(mag(seg_H([0, 0, 20], *loop)), mag(seg_H([0, 0, 40], *loop)), 20, 40, 1e-9)
    return out


MATCH = _match_table()


# ---------------------------------------------------------------- finite-difference helpers for the checkbox
def jac(F, x, h=1e-3):
    J = np.zeros((3, 3))
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        J[:, k] = (F(x + e) - F(x - e)) / (2 * h)
    return J


def curl(F, x, h=1e-3):
    J = jac(F, x, h)
    return np.array([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]])


def div(F, x, h=1e-3):
    return float(np.trace(jac(F, x, h)))


def grad(f, x, h=1e-3):
    return np.array([(f(x + e) - f(x - e)) / (2 * h) for e in np.eye(3) * h])


_rng = np.random.default_rng(329)
_C = _rng.normal(size=(3, 10))


def A_rand(x):        # a cubic vector field with no special structure
    X, Y, Z = x
    mons = np.array([1, X, Y, Z, X * Y, Y * Z, Z * X, X * X * Y, Y * Y * Z, X * Z * Z])
    return _C @ mons


def lam(x):
    X, Y, Z = x
    return X * Y * Z + X * X - 2 * Y * Z


_CACHE = {}


def statement_truth(key, d):
    if key in _CACHE:
        return _CACHE[key]
    out = _statement_truth(key, d)
    if key in ("div_curl", "gauge", "poisson", "units", "stokes", "unique", "gauge_law", "parallel",
               "ampere_alone", "a_zero"):
        _CACHE[key] = out
    return out


def _statement_truth(key, d):
    p = d["params"]
    x0 = np.array([0.3, -0.7, 0.5])
    if key == "div_curl":
        return abs(div(lambda x: curl(A_rand, x, 1e-4), x0, 1e-2)) < 1e-6
    if key == "gauge":
        A2 = lambda x: A_rand(x) + grad(lam, x, 1e-4)  # noqa: E731
        return np.allclose(curl(A2, x0), curl(A_rand, x0), atol=1e-6)
    if key == "poisson":   # divergence-free A = curl(F): curl curl A = -laplacian A, componentwise
        Af = lambda x: curl(A_rand, x, 1e-4)  # noqa: E731
        cc = curl(lambda x: curl(Af, x, 1e-3), x0, 1e-3)
        lap = np.array([sum((Af(x0 + e)[i] - 2 * Af(x0)[i] + Af(x0 - e)[i]) / 1e-4 for e in np.eye(3) * 1e-2)
                        for i in range(3)])
        return abs(div(Af, x0)) < 1e-6 and np.allclose(cc, -lap, atol=1e-4)
    if key == "units":     # [A] = [B][m] = (Wb/m^2) m = Wb/m: exponents of (Wb, m)
        return tuple(np.add((1, -2), (0, 1))) == (1, -1)
    if key == "stokes":    # square loop in the plane z = 0.2, side 0.6, counterclockwise from +z
        n = 4000
        t = (np.arange(n) + 0.5) / n
        corners = np.array([[0, 0], [0.6, 0], [0.6, 0.6], [0, 0.6], [0, 0]]) + np.array([-0.1, 0.2])
        circ = 0.0
        for k in range(4):
            p0, p1 = corners[k], corners[k + 1]
            for tt in t:
                pt = np.array([*(p0 + tt * (p1 - p0)), 0.2])
                circ += np.dot(A_rand(pt), np.array([*(p1 - p0), 0.0])) / n
        g = (np.arange(40) + 0.5) / 40 * 0.6
        flux = sum(curl(A_rand, np.array([-0.1 + u, 0.2 + v, 0.2]))[2] for u in g for v in g) * (0.6 / 40) ** 2
        return abs(circ - flux) < 1e-3 * max(1, abs(flux))
    N, I, s = p["N"], float(p["I"]), p["s"]
    a, b, h = float(p["a_cm"]) * 1e-2, float(p["b_cm"]) * 1e-2, float(p["h_cm"]) * 1e-2
    r0 = float(p["r0_cm"]) * 1e-2
    ph = math.pi / N
    if key == "toroid_h":
        H1 = H_phi_at(toroid(N, a, b, h, s, I), r0, 0, ph)[0]
        H2 = H_phi_at(toroid(N, a, b, 2 * h, s, I), r0, 0, ph)[0]
        f1 = flux_quad(H1, r0, a, b, h)
        f2 = flux_quad(H2, r0, a, b, 2 * h)
        return abs(H2 - H1) < 1e-6 * abs(H1) and abs(f2 / f1 - 2) < 1e-6
    if key in ("ratio", "toroid_uniform"):
        tor = toroid(N, a, b, h, s, I)
        r1, r2 = a + 0.3 * (b - a), a + 0.7 * (b - a)
        H1, H2 = H_phi_at(tor, r1, 0, ph)[0], H_phi_at(tor, r2, 0, ph)[0]
        if key == "toroid_uniform":
            return abs(H1 - H2) < 1e-3 * abs(H1)
        pexp = math.log(abs(H2 / H1)) / math.log(r2 / r1)
        ratio = (a / b) ** pexp          # |H(a)| / |H(b)|
        stated = [f for f in p["facts"] if f["key"] == "ratio"][0]["ratio_stated"]
        return abs(stated - ratio) < 1e-4 * ratio
    if key == "unique":    # two different potentials with the same curl -> not unique
        A2 = lambda x: A_rand(x) + grad(lam, x, 1e-4)  # noqa: E731
        same_B = np.allclose(curl(A2, x0), curl(A_rand, x0), atol=1e-6)
        return not (same_B and np.linalg.norm(A2(x0) - A_rand(x0)) > 1e-3)
    if key == "gauge_law":  # an A with nonzero divergence (and nonzero curl) still gives div B = 0
        A2 = lambda x: np.array([x[0] * x[1], x[1] * x[2] ** 2, x[2] * x[0] ** 2])  # noqa: E731
        divA = div(A2, x0)
        Bc = curl(A2, x0)
        divB = div(lambda x: curl(A2, x, 1e-4), x0, 1e-2)
        return not (abs(divA) > 1e-2 and np.linalg.norm(Bc) > 1e-2 and abs(divB) < 1e-6)
    if key == "parallel":  # wire along z: A along the wire, B around it
        Aw = seg_A([0.05, 0, 0], np.array([[0, 0, -50.0]]), np.array([[0, 0, 50.0]]), np.array([1.0]))
        Bw = MU0 * seg_H([0.05, 0, 0], np.array([[0, 0, -50.0]]), np.array([[0, 0, 50.0]]), np.array([1.0]))
        return abs(np.dot(Aw, Bw)) > 0.99 * np.linalg.norm(Aw) * np.linalg.norm(Bw)
    if key == "ampere_alone":   # two different fields with the same circulation on the unit circle
        th = 2 * np.pi * (np.arange(2000) + 0.5) / 2000
        dl = np.stack([-np.sin(th), np.cos(th)], axis=1) * (2 * np.pi / 2000)
        H_a = np.stack([-np.sin(th), np.cos(th)], axis=1) / (2 * np.pi)                 # line current at the centre
        H_b = H_a + np.stack([np.cos(th) ** 2, np.zeros_like(th)], axis=1)              # plus a curl-carrying extra
        same = abs(np.sum(H_a * dl) - np.sum(H_b * dl)) < 1e-9
        return not same    # the same I_enc does not single out one H
    if key == "a_zero":    # a constant A has zero curl but is not zero
        Ac = lambda x: np.array([1.0, -2.0, 0.5])  # noqa: E731
        return not (np.allclose(curl(Ac, x0), 0) and np.linalg.norm(Ac(x0)) > 0)
    raise KeyError(key)


def flux_quad(H_ref, r_ref, a, b, h, n=200000):
    """Flux through one turn by midpoint quadrature of mu0 H(r) h, with H(r) = H_ref * r_ref / r (the profile
    the Biot-Savart sums establish)."""
    r = a + (np.arange(n) + 0.5) * (b - a) / n
    return float(np.sum(MU0 * H_ref * r_ref / r) * (b - a) / n * h)


EXPECTED = {"div_curl": True, "gauge": True, "poisson": True, "units": True, "stokes": True, "toroid_h": True,
            "unique": False, "gauge_law": False, "parallel": False, "toroid_uniform": False, "ampere_alone": False,
            "a_zero": False}


def phys(d, check):
    p, c = d["params"], d["correct_answers"]
    N, I, s = p["N"], float(p["I"]), p["s"]
    a, b, h = float(p["a_cm"]) * 1e-2, float(p["b_cm"]) * 1e-2, float(p["h_cm"]) * 1e-2
    r0 = float(p["r0_cm"]) * 1e-2
    tor = toroid(N, a, b, h, s, I)
    ph = math.pi / N                       # midway between two turns

    # (b) H_phi at r0 by Biot-Savart, and the profile exponent
    H0 = H_phi_at(tor, r0, 0.0, ph)[0]
    check(abs(H0 - c["H"]) <= 1e-5 * abs(H0), f"toroid: H_phi Biot-Savart {H0} vs {c['H']}")
    Hz = H_phi_at(tor, r0, 0.05 * h, ph)[0]
    check(abs(Hz - H0) <= 1e-5 * abs(H0), "toroid: H_phi should not depend on z inside the core")
    r1, r2 = a + 0.3 * (b - a), a + 0.7 * (b - a)
    pexp = math.log(abs(H_phi_at(tor, r2, 0, ph)[0] / H_phi_at(tor, r1, 0, ph)[0])) / math.log(r2 / r1)
    check(abs(pexp + 1) < 1e-4, f"toroid: profile exponent {pexp}")
    flux = flux_quad(H0, r0, a, b, h)
    check(abs(flux * 1e6 - c["Psi"]) <= 1e-6 * abs(flux * 1e6), f"toroid: flux quadrature {flux * 1e6} vs {c['Psi']}")
    check(("+" in p["inner_dir"]) == (s > 0) and ("outward across the top" in p["faces"]) == (s > 0),
          "toroid: winding description does not match s")

    # (c) H at P vanishes; the correct choice gives the right reason
    rP, zP = p["rP_cm"] * 1e-2, p["zP_cm"] * 1e-2
    HP = H_phi_at(tor, rP, zP, ph)[1]
    check(np.linalg.norm(HP) < 1e-5 * N * I / (2 * math.pi * rP), f"toroid: |H| at P = {np.linalg.norm(HP)} not ~0")
    inner = (rP > a) and abs(zP) < h / 2
    outer = (rP > b) and abs(zP) < h / 2
    pierce = N * inner + N * outer
    reason = "none" if pierce == 0 else "cancel"
    ch = p["out_choices"]
    check(sum(x["correct"] == "true" for x in ch) == 1, "toroid MC: exactly one correct")
    corr = [x for x in ch if x["correct"] == "true"][0]
    check(corr["key"] == "zero_right" and corr["reason"] == reason, f"toroid MC: correct reason should be {reason}")
    wrong = [x for x in ch if x["key"] == "zero_wrong"][0]
    check(wrong["reason"] != reason, "toroid MC: the wrong-reason choice must give the other reason")

    # (d) A near the wire: Coulomb integral over a closed rectangular circuit
    e = np.array({"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[p["wire_axis"]], float)
    perp = np.roll(e, 1)
    sg = p["wire_sgn"]
    Lw, D = 50.0, 20.0
    corners = [-Lw * e, Lw * e, Lw * e + D * perp, -Lw * e + D * perp]
    A0 = np.array([corners[0], corners[1], corners[2], corners[3]])
    B0 = np.array([corners[1], corners[2], corners[3], corners[0]])
    Avec = seg_A(-0.05 * perp, A0, B0, np.full(4, float(sg)))
    Ahat = Avec / np.linalg.norm(Avec)
    check(np.dot(Ahat, sg * e) > 0.99, f"vector potential: numerical A {Ahat} not along the current")
    chA = p["A_choices"]
    check(sum(x["correct"] == "true" for x in chA) == 1, "A MC: exactly one correct")
    for x in chA:
        along = x["vec"] is not None and np.dot(np.array(x["vec"], float), Ahat) > 0.99
        check(along == (x["correct"] == "true"), f"A MC: choice {x['key']} marked {x['correct']}")

    # (a) matching keys against the numerical classification
    for m in p["match"]:
        check(MATCH[m["id"]] == m["key"], f"matching: {m['id']} keyed {m['key']} but numerics say {MATCH[m['id']]}")
    check(len({m["key"] for m in p["match"]}) >= 4, "matching: fewer than 4 different answers")

    # (e) checkbox truths
    facts = p["facts"]
    nt = sum(f["correct"] == "true" for f in facts)
    check(len(facts) == 5 and 1 <= nt <= 4, f"checkbox: {len(facts)} statements, {nt} true")
    for f in facts:
        truth = statement_truth(f["key"], d)
        check(truth == (f["correct"] == "true"), f"checkbox: {f['key']} marked {f['correct']}, recomputed {truth}")
        if f["key"] in EXPECTED:
            check(EXPECTED[f["key"]] == truth, f"checkbox: {f['key']} expected {EXPECTED[f['key']]}")
