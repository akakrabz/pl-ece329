"""Independent check for midterm2/voltmeters-changing-flux.

Route: potentials instead of Faraday-per-meter-loop. With E = -grad(Phi) - dA/dt and, for the uniform field,
A = 1/2 B(t) z x (r - r0) about an arbitrary (deliberately off-centre) origin r0, every branch from P to Q obeys
    integral_P^Q E.dl = Phi(P) - Phi(Q) - integral_P^Q dA/dt . dl  =  (current P->Q) x R   (0 for an ideal wire).
The four sides give four equations for Phi_B, Phi_C, Phi_D (Phi_A = 0) and the loop current I, solved with numpy.
A meter reads integral_+^- E.dl along its own route = Phi(X) - Phi(Y) - integral (dA/dt).dl along that route, the
last integral done by quadrature along the actual polyline (offset slightly outside the sides for V1, V2).
Flux from Stokes: Psi = closed integral of A.dl around the square; emf = -dPsi/dt by a centred difference.
Part (f): thin solenoid at the centroid of the stated triangle, dA/dt = (dPhi_s/dt) phi-hat / (2 pi rho), same solve.
Second route for I and the readings: mesh analysis of circuit + meters (1 GOhm each) with face emfs -dBz/dt x area.
"""
import math
import re

import numpy as np

CORNERS = "ABCD"
_GL = np.polynomial.legendre.leggauss(160)       # smooth integrands along every segment: 160 nodes is plenty


def _seg_integral(fun, p, q):
    """Integral of fun(r).dl along the straight segment p -> q (Gauss-Legendre)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    x, w = _GL
    t = 0.5 * (x + 1.0)
    pts = p[None, :] + t[:, None] * (q - p)[None, :]
    vals = fun(pts)                                   # (n, 3)
    return 0.5 * float(np.sum(w * (vals @ (q - p))))


def _poly_integral(fun, pts):
    return sum(_seg_integral(fun, pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def _solve(R, s, dAdt):
    """Nodal solve. Returns node potentials (A..D) and the counterclockwise current I."""
    P = [np.array([0.0, 0.0, 0.0]), np.array([s, 0.0, 0.0]), np.array([s, s, 0.0]), np.array([0.0, s, 0.0])]
    M = np.zeros((4, 4))
    rhs = np.zeros(4)
    # unknowns: Phi_B, Phi_C, Phi_D, I   (Phi_A = 0)
    for k in range(4):
        a, b = k, (k + 1) % 4
        G = _seg_integral(dAdt, P[a], P[b])
        # Phi(a) - Phi(b) - R_k I = G
        if a > 0:
            M[k, a - 1] += 1.0
        if b > 0:
            M[k, b - 1] -= 1.0
        M[k, 3] = -R[k]
        rhs[k] = G
    sol = np.linalg.solve(M, rhs)
    Phi = np.array([0.0, sol[0], sol[1], sol[2]])
    return P, Phi, sol[3]


def _readings(R, s, X, dAdt, eps):
    P, Phi, I = _solve(R, s, dAdt)
    Y, W1, W2 = (X + 2) % 4, (X + 1) % 4, (X + 3) % 4
    ctr = np.array([s / 2, s / 2, 0.0])

    def outward(k):                      # outward normal of side k
        m = 0.5 * (P[k] + P[(k + 1) % 4]) - ctr
        return m / np.linalg.norm(m)

    def route(W, sX, sY):
        n1, n2 = outward(sX), outward(sY)
        return [P[X], P[X] + eps * n1, P[W] + eps * (n1 + n2), P[Y] + eps * n2, P[Y]]

    out = {}
    for name, pts in (("V1", route(W1, X, W1)), ("V2", route(W2, W2, Y)), ("V3", [P[X], P[Y]])):
        out[name] = Phi[X] - Phi[Y] - _poly_integral(dAdt, pts)
    return out, I


def _mesh(R, s, X, beta, Rm=1e9):
    """Second route: mesh analysis of the planar network circuit + three meters (finite resistance Rm).
    Faces, all traversed counterclockwise, with emf = -dBz/dt x area:
      F1 strip between V1's leads and sides X-W1-Y (area 0)    F2 triangle X W1 Y (s^2/2)
      F3 triangle X Y W2 (s^2/2)                                F4 strip between sides X-W2-Y and V2's leads (area 0)
    Returns the three readings (current from + to - through each meter times Rm) and the circuit current along X->W1."""
    Y, W1, W2 = (X + 2) % 4, (X + 1) % 4, (X + 3) % 4
    Ra = R[X] + R[W1]                    # sides X-W1 and W1-Y
    Rb = R[W2] + R[Y]                    # sides W2-X and Y-W2
    e_tri = -beta * s * s / 2
    M = np.array([
        [Rm + Ra, -Ra, 0.0, 0.0],                    # F1: Rm I1 + Ra (I1 - I2) = 0
        [-Ra, Ra + Rm, -Rm, 0.0],                    # F2: Ra (I2 - I1) + Rm (I2 - I3) = e_tri
        [0.0, -Rm, Rm + Rb, -Rb],                    # F3: Rm (I3 - I2) + Rb (I3 - I4) = e_tri
        [0.0, 0.0, -Rb, Rb + Rm],                    # F4: Rb (I4 - I3) + Rm I4 = 0
    ])
    I1, I2, I3, I4 = np.linalg.solve(M, np.array([0.0, e_tri, e_tri, 0.0]))
    return {"V1": Rm * I1, "V3": Rm * (I3 - I2), "V2": -Rm * I4}, I2 - I1


def phys(d, check):
    p, c = d["params"], d["correct_answers"]
    R = [float(v) for v in p["R_sides"]]
    s, X, beta, B0, t1 = p["s"], p["Xi"], p["beta"], p["B0_val"], p["t1_val"]
    Y, W1, W2 = (X + 2) % 4, (X + 1) % 4, (X + 3) % 4

    # --- route 2: loops with emfs proportional to the enclosed area (meters as 1 GOhm branches)
    rm, Im = _mesh(R, s, X, beta)
    check(abs(Im * 1e3 - c["I"]) <= 1e-6 * max(1, abs(c["I"])), f"voltmeters (mesh): I {Im*1e3} vs {c['I']}")
    for k in ("V1", "V2", "V3"):
        check(abs(rm[k] * 1e3 - c[k]) <= 1e-6 * max(1, abs(c[k])), f"voltmeters (mesh): {k} {rm[k]*1e3} vs {c[k]}")

    # --- the statement and the figure describe the same circuit
    for k in range(4):
        side = CORNERS[k] + CORNERS[(k + 1) % 4]
        if R[k] > 0:
            check(re.search(r"= %d\\ \\Omega\$ on side %s" % (int(R[k]), side), p["res_sentence"]) is not None,
                  f"voltmeters: statement does not put {int(R[k])} ohm on side {side}")
            check(f" = {int(R[k])} \u03a9" in p["svg"], "voltmeters: figure legend misses a resistor value")
        else:
            check(side in p["res_sentence"].split(". ")[-1], f"voltmeters: plain side {side} not stated")
    check(p["X"] == CORNERS[X] and p["Y"] == CORNERS[Y] and p["W1"] == CORNERS[W1] and p["W2"] == CORNERS[W2],
          "voltmeters: corner names")
    check(len(p["svg"]) <= 2000, f"voltmeters: figure is {len(p['svg'])} bytes")

    # --- flux by Stokes with an off-centre gauge origin, emf by a centred time difference
    r0 = np.array([0.37 * s, -0.21 * s, 0.0])
    zhat = np.array([0.0, 0.0, 1.0])

    def A_at(t):
        Bz = B0 + beta * t
        return lambda pts: 0.5 * Bz * np.cross(np.broadcast_to(zhat, pts.shape), pts - r0)

    sq = [np.array(v, float) for v in ([0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0], [0, 0, 0])]
    Psi = lambda t: _poly_integral(A_at(t), sq)
    check(abs(Psi(t1) * 1e3 - c["Psi"]) <= 1e-7 * max(1, abs(c["Psi"])), f"voltmeters: Psi {Psi(t1)*1e3} vs {c['Psi']}")
    h = 1e-3
    emf = -(Psi(t1 + h) - Psi(t1 - h)) / (2 * h)
    check(abs(emf * 1e3 - c["emf"]) <= 1e-6 * max(1, abs(c["emf"])), f"voltmeters: emf {emf*1e3} vs {c['emf']}")

    # --- circuit and meters from the potentials
    dAdt = lambda pts: 0.5 * beta * np.cross(np.broadcast_to(zhat, pts.shape), pts - r0)
    rd, I = _readings(R, s, X, dAdt, eps=1e-9 * s)
    check(abs(I * 1e3 - c["I"]) <= 1e-6 * max(1, abs(c["I"])), f"voltmeters: I {I*1e3} vs {c['I']}")
    for k in ("V1", "V2", "V3"):
        check(abs(rd[k] * 1e3 - c[k]) <= 1e-5 * max(1, abs(c[k])), f"voltmeters: {k} {rd[k]*1e3} vs {c[k]}")
    check(abs((c["V1"] - c["V2"]) - c["emf"]) <= 1e-6 * max(1, abs(c["emf"])), "voltmeters: V1 - V2 != emf")
    check(len({round(c[k], 6) for k in ("V1", "V2", "V3")}) == 3 and abs(c["V3"]) > 1e-6, "voltmeters: readings not distinct")
    # review R7: V3's per-variant atol (1% of the larger reading) must stay well inside |V3| (rejects 0 and -V3)
    check(0 < float(p["V3_atol"]) <= 0.45 * abs(c["V3"]), f"voltmeters: V3 atol {p['V3_atol']} vs V3 {c['V3']}")
    check((c["I"] > 0) == (beta < 0), "voltmeters: Lenz - counterclockwise current iff the +z flux decreases")

    # --- part (f): thin solenoid at the centroid of the stated triangle, same dPsi/dt
    tri = p["sol_tri"]
    corner_xy = {"A": (0, 0), "B": (s, 0), "C": (s, s), "D": (0, s)}
    rs = np.array([sum(corner_xy[ch][0] for ch in tri) / 3, sum(corner_xy[ch][1] for ch in tri) / 3, 0.0])
    check(set(tri) == {CORNERS[X], CORNERS[Y], CORNERS[W1 if p["sol_half"] == 1 else W2]}, "voltmeters: sol_tri")
    dPhis = beta * s * s

    def dAdt_sol(pts):
        rel = pts - rs
        rho2 = rel[:, 0] ** 2 + rel[:, 1] ** 2
        phi_hat = np.stack([-rel[:, 1], rel[:, 0], np.zeros(len(rel))], axis=1) / np.sqrt(rho2)[:, None]
        return dPhis / (2 * math.pi * np.sqrt(rho2))[:, None] * phi_hat

    rdf, If = _readings(R, s, X, dAdt_sol, eps=1e-9 * s)
    check(abs(If - I) <= 1e-6 * max(1e-3, abs(I)), f"voltmeters (f): current changed {If} vs {I}")
    for k in ("V1", "V2"):
        check(abs(rdf[k] - rd[k]) <= 1e-5 * max(1e-3, abs(rd[k])), f"voltmeters (f): {k} changed {rdf[k]} vs {rd[k]}")
    V3f = rdf["V3"] * 1e3
    ch = p["f_choices"]
    check(sum(x["correct"] == "true" for x in ch) == 1 and len({x["text"] for x in ch}) == len(ch), "voltmeters (f): MC")

    def num_in(text):
        m = re.search(r"= (-?\d+(?:\.\d+)?)\\ \\mathrm\{mV\}", text)
        return float(m.group(1)) if m else None

    for x in ch:
        v = num_in(x["text"])
        hit = v is not None and abs(v - V3f) <= 6e-4 * max(1, abs(V3f))
        check(hit == (x["correct"] == "true"), f"voltmeters (f): choice {x['text'][:40]} vs V3' = {V3f}")
