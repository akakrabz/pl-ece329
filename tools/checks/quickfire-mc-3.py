"""Independent check for midterm3/quickfire-mc-3 (six items from fourteen generators).

Each choice carries a machine-readable val/key. For every item the truth is recomputed from the item's data by a
route different from server.py, and every choice is judged against it:
  phasor    each offered phasor is turned back into a time function and compared with the stated field
  losstan   exact gamma of the medium (complex root); loss tangent recovered as 2 alpha beta / (beta^2 - alpha^2)
  skin      exact skin depth of copper at the two frequencies
  depth     |E|^2 ratio of E = exp(-(1 + j) z / delta)
  polname   field tip traced over a period at a fixed point (signed area about the travel direction, max/min)
  standing  incident plus reflected phasors in front of a mirror on a fine grid; nulls located numerically
  gamma     tangential E and H continuity solved as a 2x2 system
  backward  telegrapher's equation -dV/dz = L dI/dt tested by finite differences on a Gaussian pulse for each choice
  inject, gammai, steady, echotime, junction   exact time-domain simulation (tools/checks/_tlsim.py)
  ppline    Z0 = sqrt(L'/C') with L' = mu d/w, C' = eps w/d before and after the change
"""
import cmath
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tlsim import Cascade  # noqa: E402

MU0 = 4e-7 * math.pi
EPS0 = 8.8541878128e-12
IDX = {"x": 0, "y": 1, "z": 2}


def close(a, b, rel=1e-3):
    return abs(a - b) <= rel * max(abs(a), abs(b)) + 1e-12


def unit(ax):
    v = np.zeros(3); v[IDX[ax]] = 1.0; return v


def t_phasor(dd, c):
    ang, ks = c["val"]
    A, phi, s = dd["A"], math.radians(dd["phi"]), dd["s"]
    pts = [(0.3 * k, 0.7 * k + 0.1) for k in range(1, 7)]        # (beta z, omega t)
    return all(abs((A * cmath.exp(1j * math.radians(ang)) * cmath.exp(-1j * ks * bz) * cmath.exp(1j * wt)).real
                   - A * math.sin(wt - s * bz + phi)) < 1e-9 for bz, wt in pts)


def gamma_of(f, sigma, er):
    om = 2 * math.pi * f
    return cmath.sqrt(1j * om * MU0 * (sigma + 1j * om * er * EPS0))


def t_losstan(dd, c):
    g = gamma_of(dd["f"], dd["sigma"], dd["er"])
    t = 2 * g.real * g.imag / (g.imag ** 2 - g.real ** 2)
    assert not (0.05 <= t <= 0.2 or 5 <= t <= 20), f"loss tangent {t} too close to a threshold"
    return c["key"] == ("good" if t > 10 else "diel" if t < 0.1 else "mid")


def t_skin(dd, c):
    d = lambda f: 1 / gamma_of(f * 1e3, 5.8e7, 1).real
    return close(c["val"], dd["d1"] * d(dd["f2"]) / d(dd["f1"]))


def t_depth(dd, c):
    return close(c["val"], abs(cmath.exp(-(1 + 1j) * dd["m"])) ** 2)


def t_polname(dd, c):
    coef = {"+j": 1j, "-j": -1j, "+": 1, "-": -1, "+2j": 2j, "-2j": -2j}[dd["coef"]]
    ea, eb, u = unit(dd["a"]), unit(dd["b"]), dd["s"] * unit(dd["w"])
    th = np.linspace(0, 2 * math.pi, 2001)
    pts = np.array([np.real((ea + coef * eb) * cmath.exp(1j * t)) for t in th])
    mags = np.linalg.norm(pts, axis=1)
    area = 0.5 * np.sum(np.cross(pts[:-1], pts[1:]) @ u)
    kd = "lin" if abs(area) < 1e-6 else ("rc" if area > 0 else "lc") if mags.min() > 0.999 * mags.max() else "ell"
    return c["key"] == kd


def t_standing(dd, c):
    lam = 3e8 / (dd["fG"] * 1e9) * 100          # cm
    beta = 2 * math.pi / lam
    z = np.linspace(0, 1.3 * lam, 260001)       # distance from the mirror
    E = np.abs(np.exp(1j * beta * z) - np.exp(-1j * beta * z))        # Gamma = -1 at the mirror
    H = np.abs(np.exp(1j * beta * z) + np.exp(-1j * beta * z))
    def nulls(a):
        i = np.where((a[1:-1] < a[:-2]) & (a[1:-1] <= a[2:]) & (a[1:-1] < 1e-3))[0] + 1
        out = list(z[i])
        return ([0.0] if a[0] < 1e-9 else []) + out
    truth = (nulls(E)[1] - nulls(E)[0]) if dd["mode"] == "Enull" else nulls(H)[0]
    return close(c["val"], truth)


def t_gamma(dd, c):
    n1, n2 = (1.0, dd["n"]) if dd["into"] else (dd["n"], 1.0)
    e1, e2 = 1 / n1, 1 / n2                     # impedances up to a common factor
    G, _ = np.linalg.solve(np.array([[1.0, -1.0], [-1.0 / e1, -1.0 / e2]]), np.array([-1.0, -1.0 / e1]))
    return close(c["val"], G)


def t_backward(dd, c):
    Z0, V, s, v = dd["Z0"], dd["V"], dd["s"], 2e8
    L = Z0 / v
    g = lambda x: math.exp(-(x / 1e-9) ** 2)
    Vf = lambda z, t: V * g(t - s * z / v)
    If = lambda z, t: c["val"] * 1e-3 * g(t - s * z / v)
    z0, t0, hz, ht = 0.05, 0.6e-9, 1e-5, 1e-14
    lhs = -(Vf(z0 + hz, t0) - Vf(z0 - hz, t0)) / (2 * hz)
    rhs = L * (If(z0, t0 + ht) - If(z0, t0 - ht)) / (2 * ht)
    return close(lhs, rhs, 1e-4)


def t_inject(dd, c):
    sim = Cascade([dd["Z0"]], [8], dd["Rg"], dd["RL"], np.full(4, float(dd["V0"])))
    return close(c["val"], sim.f[0][0])


def t_gammai(dd, c):
    sim = Cascade([float(dd["Z0"])], [8], float(dd["Z0"]), dd["RL"], np.full(20, 1.0))
    return close(c["val"], (-sim.g[0][8] / dd["Z0"]) / (sim.f[0][0] / dd["Z0"]))


def t_steady(dd, c):
    sim = Cascade([float(dd["Z0"])], [4], dd["Rg"], dd["RL"], np.full(600, float(dd["V0"])))
    return close(c["val"], sim.V(0, 4, 599), 1e-6)


def t_echotime(dd, c):
    N = 120
    k = N * dd["frac"][0] // dd["frac"][1]
    src = np.zeros(3 * N); src[0] = 1.0
    sim = Cascade([50.0], [N], 50.0, 150.0, src)
    hits = [n for n in range(3 * N) if abs(sim.V(0, k, n)) > 1e-12]
    return close(c["val"], hits[1] * (dd["ell"] / dd["v"]) / N)


def t_junction(dd, c):
    Z1, Z2 = float(dd["Z1"]), float(dd["Z2"])
    src = np.zeros(40); src[0] = 2.0 * dd["Vi"]
    sim = Cascade([Z1, Z2], [5, 5], Z1, Z2, src)
    assert close(sim.f[0][0], dd["Vi"], 1e-9)
    return close(c["val"], sim.f[1][5])


def t_ppline(dd, c):
    def Z0(w, d, er, length):
        return math.sqrt((MU0 * d / w) / (er * EPS0 * w / d))
    base = dict(w=0.1, d=0.002, er=2.0, length=1.0)
    new = dict(base)
    what = dd["what"]
    if "width is doubled" in what: new["w"] *= 2
    elif "width is halved" in what: new["w"] /= 2
    elif "separation is doubled" in what: new["d"] *= 2
    elif "separation is halved" in what: new["d"] /= 2
    elif "four times the permittivity" in what: new["er"] *= 4
    elif "twice as long" in what: new["length"] *= 2
    else: raise AssertionError("unknown change: " + what)
    return close(c["val"], Z0(**new) / Z0(**base))


TRUTH = {"phasor": t_phasor, "losstan": t_losstan, "skin": t_skin, "depth": t_depth, "polname": t_polname, "standing": t_standing,
         "gamma": t_gamma, "backward": t_backward, "inject": t_inject, "gammai": t_gammai, "steady": t_steady, "echotime": t_echotime,
         "junction": t_junction, "ppline": t_ppline}


def phys(d, check):
    items = d["params"]["items"]
    check(len(items) == 6 and len({it["gen"] for it in items}) == 6, "quickfire-mc-3: needs six items from six different generators")
    check([it["name"] for it in items] == [f"q{i}" for i in range(1, 7)], "quickfire-mc-3: item names")
    for it in items:
        fn = TRUTH[it["gen"]]
        good = 0
        for c in it["choices"]:
            ok = bool(fn(it["data"], c))
            check((c["correct"] == "true") == ok, f"quickfire-mc-3/{it['gen']}: choice {c['text']} mis-marked (data {it['data']})")
            good += ok
        check(good == 1 and len(it["choices"]) == 4, f"quickfire-mc-3/{it['gen']}: {good} true choices")
        check(all(c["feedback"] for c in it["choices"]) and it["solution"], f"quickfire-mc-3/{it['gen']}: missing feedback or solution")
