"""Independent check for midterm3/two-component-wave.

Route: everything is done on sampled fields, without the server's sign rules.
  (a) each phasor choice is turned back into a time function Re{E~ e^{jwt}} and compared with the given field on a
      grid of (w, t);
  (b), (c) the phasor of the given field is built from two samples a quarter period apart, H follows from Faraday's
      law by central differences, and <S> = Re(E x H*)/2;
  (d), (e) polarization is found by tracing the field tip over one period at a fixed point: the signed area swept
      (about the travel direction) gives the turning sense, max|F|/min|F| separates linear, circular, elliptical.
"""
import cmath
import math

import numpy as np

MU0 = 4e-7 * math.pi
C0 = 3e8
IDX = {"x": 0, "y": 1, "z": 2}
OM = 2 * math.pi * 1e8          # any frequency: the answers do not depend on it


def unit(ax):
    v = np.zeros(3); v[IDX[ax]] = 1.0; return v


def trace_kind(F, u):
    """F(th) -> real 3-vector at phase th = omega t; u = travel direction. Returns lin / rc / lc / ell."""
    th = np.linspace(0, 2 * math.pi, 2001)
    pts = np.array([F(t) for t in th])
    mags = np.linalg.norm(pts, axis=1)
    area = 0.5 * np.sum(np.cross(pts[:-1], pts[1:]) @ u)          # signed area, positive = right-handed about u
    if abs(area) < 1e-6 * mags.max() ** 2:
        return "lin"
    if mags.min() > 0.999 * mags.max():
        return "rc" if area > 0 else "lc"
    return "ell"


def phys(d, check):
    p, ca = d["params"], d["correct_answers"]
    q = "two-component-wave"
    a, b, w, s, n = p["a"], p["b"], p["w"], p["s"], p["n"]
    A, B, phi = p["kA"] * math.pi, p["kB"] * math.pi, math.radians(p["phi"])
    beta = OM * n / C0
    ea, eb, ew = unit(a), unit(b), unit(w)

    def E(r, t, Bx=B, phx=phi):
        th = OM * t - s * beta * (r @ ew)
        return ea * A * math.cos(th) + eb * Bx * math.sin(th + phx)

    # (a) phasor choices
    pts = [(np.array([0.13, -0.2, 0.31]) * k, 1e-9 * k * 0.37) for k in range(1, 6)]
    ncorrect = 0
    for c in p["ph_choices"]:
        cb = complex(*c["cb"])
        ok = all(np.allclose(np.real((ea * A + eb * cb) * cmath.exp(-1j * c["ksign"] * beta * (r @ ew)) * cmath.exp(1j * OM * t)),
                             E(r, t), rtol=1e-9, atol=1e-9) for r, t in pts)
        check((c["correct"] == "true") == ok, f"{q}: phasor choice mis-marked: {c['text']}")
        ncorrect += ok
    check(ncorrect == 1 and len(p["ph_choices"]) == 4, f"{q}: phasor MC needs exactly one true choice of four")

    # (b), (c): phasor from two time samples, H from Faraday
    T4 = (math.pi / 2) / OM

    def Eph(r):                                  # Re{E~} = E(t=0), Im{E~} = -E(t=T/4)
        return E(r, 0.0) - 1j * E(r, T4)

    def Hph(r):
        h = 1e-4 / beta
        J = np.zeros((3, 3), dtype=complex)
        for j in range(3):
            dr = np.zeros(3); dr[j] = h
            J[:, j] = (Eph(r + dr) - Eph(r - dr)) / (2 * h)
        curl = np.array([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]])
        return -curl / (1j * OM * MU0)

    r0 = np.zeros(3)
    H0 = np.real(Hph(r0))                        # H at the origin at t = 0
    check(abs(H0 @ ea - ca["Ha"]) < 1e-5 + 1e-5 * abs(ca["Ha"]), f"{q}: H_a {H0 @ ea} vs {ca['Ha']}")
    check(abs(H0 @ eb - ca["Hb"]) < 1e-5 + 1e-5 * abs(ca["Hb"]), f"{q}: H_b {H0 @ eb} vs {ca['Hb']}")
    S = 0.5 * np.real(np.cross(Eph(r0), np.conj(Hph(r0))))
    check(abs(S @ ew - ca["Sw"]) < 1e-5 * abs(ca["Sw"]) and abs(S @ ew) > 0.9999 * np.linalg.norm(S), f"{q}: <S> {S} vs {ca['Sw']}")

    # (d) circular condition
    u = s * ew
    hits = 0
    for c in p["circ_choices"]:
        kd = trace_kind(lambda th, c=c: ea * A * math.cos(th) + eb * c["ratio"] * A * math.sin(th + math.radians(c["phi2"])), u)
        check((c["correct"] == "true") == (kd == p["target"]), f"{q}: circular choice {c['text']} is {kd}, target {p['target']}")
        hits += kd == p["target"]
    check(hits == 1, f"{q}: circular MC has {hits} true choices")
    # (e) rows given through H; E = eta H x u turns the same way, checked on E itself
    u3 = p["s3"] * unit("z")
    for it in p["match_items"]:
        def Hrow(th, it=it):
            return unit("x") * it["A"] * math.cos(th) + unit("y") * it["B"] * math.cos(th + math.radians(it["phi"]))
        kd = trace_kind(lambda th: np.cross(Hrow(th), u3), u3)
        check(kd == it["key"], f"{q}: row {it['text']} is {kd}, keyed {it['key']}")
    check(len({it["text"] for it in p["match_items"]}) == 3, f"{q}: duplicate matching rows")
