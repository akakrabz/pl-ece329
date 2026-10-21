"""Independent check for transmission-lines/line-parameters-waves (Lecture 27).

Geometry and filling are parsed from the displayed text. Line parameters come from field integrals:
C' = charge per metre / V with V the quadrature of E from Gauss's law, L' = 2 W'_m / I^2 with W'_m the quadrature
of mu H^2/2 over the cross-section; then L'C' = mu eps is checked, v = 1/sqrt(L'C'), Z0 = sqrt(L'/C').
The pulse V(z,t) is parsed from its LaTeX; its direction is found numerically, and I(z1, t1) by integrating the
telegrapher equation dI/dz = -C' dV/dt from a point the pulse has not reached (I = 0 there); the other telegrapher
equation is checked by finite differences. p = V I. The scaling part is re-derived from field integrals of the
rebuilt line, its text parsed. Checkbox statements are re-decided by computations: field integrals for several
geometries, and a leapfrog simulation of the L'dz-series / C'dz-shunt ladder (speeds, V/I of each wave, energies).
"""
import math
import re

import numpy as np

MU0 = 4e-7 * math.pi
EPS0 = 1e-9 / (36 * math.pi)          # as stated in the problem
ETA0 = math.sqrt(MU0 / EPS0)


def coax_LC(a, b, eps, mu=MU0):
    r = np.geomspace(a, b, 20001)
    q = 1.0
    E = q / (2 * math.pi * eps * r)            # Gauss
    V = np.trapezoid(E, r)
    I = 1.0
    Hh = I / (2 * math.pi * r)                 # Ampere
    Wm = np.trapezoid(0.5 * mu * Hh ** 2 * 2 * math.pi * r, r)
    return 2 * Wm / I ** 2, q / V


def plates_LC(W, d, eps, mu=MU0):
    x = np.linspace(0, d, 2001)
    rho = 1.0
    E = np.full_like(x, rho / eps)             # Gauss, W >> d
    V = np.trapezoid(E, x)
    I = 1.0
    Hh = I / W                                  # J_s = H on the facing surface
    Wm = np.trapezoid(np.full_like(x, 0.5 * mu * Hh ** 2 * W), x)
    return 2 * Wm / I ** 2, rho * W / V


def LC(geom, d1, d2, eps):
    return coax_LC(d1, d2, eps) if geom == "coax" else plates_LC(d1, d2, eps)


def vz(Lp, Cp):
    return 1 / math.sqrt(Lp * Cp), math.sqrt(Lp / Cp)


def parse_geom(tex):
    m = re.fullmatch(r"a coaxial cable with inner-conductor radius \$a = ([\d.]+)\\ \\mathrm\{mm\}\$ and outer-conductor "
                     r"inner radius \$b = ([\d.]+)\\ \\mathrm\{mm\}\$", tex)
    if m:
        return "coax", float(m.group(1)) * 1e-3, float(m.group(2)) * 1e-3
    m = re.fullmatch(r"two perfectly conducting plates of width \$W = ([\d.]+)\\ \\mathrm\{mm\}\$, at \$x = 0\$ and "
                     r"\$x = d = ([\d.]+)\\ \\mathrm\{mm\}\$ \(neglect fringing\)", tex)
    assert m, "cannot parse geometry: " + tex
    return "plates", float(m.group(1)) * 1e-3, float(m.group(2)) * 1e-3


def parse_V(tex, v):
    m = re.fullmatch(r"(-?\d+)\\,\\exp\\!\\left\[-\\left\(\\frac\{(.+?)\}\{([\d.]+)\\ \\mathrm\{(ns|m)\}\}\\right\)\^\{2\}\\right\]\\ \\mathrm\{V\}", tex)
    assert m, "cannot parse V: " + tex
    V0 = int(m.group(1))
    den = float(m.group(3)) * (1e-9 if m.group(4) == "ns" else 1.0)
    n = re.fullmatch(r"([tz]) ([+-]) (z/v|vt)(?: - ([\d.]+)\\ \\mathrm\{(ns|m)\})?", m.group(2))
    assert n, "cannot parse numerator: " + m.group(2)
    first, sg, term = n.group(1), (1 if n.group(2) == "+" else -1), n.group(3)
    off = float(n.group(4)) * (1e-9 if n.group(5) == "ns" else 1.0) if n.group(4) else 0.0
    assert (first == "t") == (term == "z/v") and (first == "t") == (m.group(4) == "ns") and (not n.group(5) or n.group(5) == m.group(4))

    def V(z, t):
        if first == "t":
            u = (t + sg * z / v - off) / den
        else:
            u = (z + sg * v * t - off) / den
        return V0 * np.exp(-u * u)
    width = den * (v if first == "t" else 1.0)          # spatial width in m
    return V, width, V0


def fac_parse(s):
    if s == "unchanged":
        return 1.0
    if s == "doubled":
        return 2.0
    if s == "halved":
        return 0.5
    m = re.fullmatch(r"multiplied by ([\d.]+)", s)
    assert m, s
    return float(m.group(1))


def ladder(Lp, Cp, n=1600, courant=0.5, steps=1200):
    """Leapfrog on series L'dz / shunt C'dz sections. Initial V = Gaussian, I = 0 -> two pulses."""
    v = 1 / math.sqrt(Lp * Cp)
    dz = 1.0 / n
    dt = courant * dz / v
    zN = np.arange(n + 1) * dz
    sig = 30 * dz
    V = np.exp(-((zN - 0.5) / sig) ** 2)
    I = np.zeros(n)                                     # at (k + 1/2) dz
    snaps = {}
    E_tot = []
    for k in range(1, steps + 1):
        I -= dt / (Lp * dz) * (V[1:] - V[:-1])
        V[1:-1] -= dt / (Cp * dz) * (I[1:] - I[:-1])
        if k in (20, 600, 1200):
            Ic = np.concatenate([[0.0], 0.5 * (I[1:] + I[:-1]), [0.0]])    # I at the nodes
            snaps[k] = (V.copy(), Ic, k * dt)
        if k % 200 == 0:
            E_tot.append(np.sum(0.5 * Cp * V ** 2) * dz + np.sum(0.5 * Lp * I ** 2) * dz)
    return zN, snaps, E_tot, v, dz, dt


def phys(d, check):
    p, ca = d["params"], d["correct_answers"]
    geom, d1, d2 = parse_geom(p["geom_txt"])
    check(geom == p["geom"], "line: geometry text vs param")
    er = float(p["er_str"])
    check(abs(er - p["eps_r"]) < 1e-12, "line: eps_r text")
    eps = er * EPS0
    Lp, Cp = LC(geom, d1, d2, eps)
    check(abs(Lp * Cp - MU0 * eps) < 1e-6 * MU0 * eps, "line: L'C' != mu eps")
    v, Z0 = vz(Lp, Cp)
    check(abs(Lp * 1e9 - ca["Lp"]) < 1e-5 * ca["Lp"], f"line: L' {Lp*1e9} vs {ca['Lp']}")
    check(abs(Cp * 1e12 - ca["Cp"]) < 1e-5 * ca["Cp"], f"line: C' {Cp*1e12} vs {ca['Cp']}")
    check(abs(v / 1e8 - ca["v"]) < 1e-5 * ca["v"], f"line: v {v/1e8} vs {ca['v']}")
    check(abs(Z0 - ca["Z0"]) < 1e-5 * ca["Z0"], f"line: Z0 {Z0} vs {ca['Z0']}")
    if geom == "plates":
        check(d1 / d2 >= 8, "line: plates not wide compared with the gap")

    # ---- (c), (d): the pulse
    V, width, V0 = parse_V(p["V_tex"], v)
    z1, t1 = float(p["z1_str"]), float(p["t1_str"]) * 1e-9
    dirs = []
    for s in (1, -1):
        ok = all(abs(V(z + s * v * dtt, t + dtt) - V(z, t)) < 1e-9 * 20
                 for z, t, dtt in ((z1, t1, 3e-9), (z1 + 0.3 * width, t1 - 2e-9, 7e-9)))
        if ok:
            dirs.append(s)
    check(len(dirs) == 1, f"line: pulse direction ambiguous {dirs}")
    s = dirs[0]
    V1 = V(z1, t1)
    check(abs(V1 - ca["V1"]) < 1e-6 * max(1, abs(V1)), f"line: V(z1,t1) {V1} vs {ca['V1']}")

    def I_at(t):
        zs = np.linspace(z1 - 14 * width, z1, 40001)
        dt_ = 1e-4 * width / v
        dVdt = (V(zs, t + dt_) - V(zs, t - dt_)) / (2 * dt_)
        return -Cp * np.trapezoid(dVdt, zs)               # I(z_far) = 0
    I1 = I_at(t1)
    check(abs(V(z1 - 14 * width, t1)) < 1e-30 + 1e-12 * 20, "line: integration did not start outside the pulse")
    check(abs(I1 * 1e3 - ca["I1"]) < 1e-5 * max(1, abs(I1 * 1e3)), f"line: I(z1,t1) {I1*1e3} vs {ca['I1']}")
    dt_ = 1e-3 * width / v
    dz_ = 1e-3 * width
    dVdz = (V(z1 + dz_, t1) - V(z1 - dz_, t1)) / (2 * dz_)
    dIdt = (I_at(t1 + dt_) - I_at(t1 - dt_)) / (2 * dt_)
    check(abs(-dVdz - Lp * dIdt) < 1e-3 * max(abs(dVdz), abs(V0) / width), "line: first telegrapher equation fails")
    p1 = V1 * I1
    check(abs(p1 - ca["p1"]) < 1e-5 * max(1e-3, abs(p1)), f"line: power {p1} vs {ca['p1']}")
    check((p1 > 0) == (s > 0) or abs(V1) < 1e-12, "line: power sign vs direction")
    dc = [c for c in p["dirs_list"] if c["correct"] == "true"]
    check(len(dc) == 1 and dc[0]["id"] == ("+z" if s > 0 else "-z"), "line: direction MC key")

    # ---- (e) scaling
    mt = p["mod_txt"]
    er2, g1, g2 = er, d1, d2
    m = re.fullmatch(r"(?:both radii are|the width and the gap are both) multiplied by (\d+), with the same filling", mt)
    if m:
        k = int(m.group(1)); g1, g2 = d1 * k, d2 * k
        check(("radii" in mt) == (geom == "coax"), "line: scale text vs geometry")
    elif mt.startswith("the filling is replaced"):
        m = re.fullmatch(r"the filling is replaced by .+ \(\$\\epsilon_r = ([\d.]+)\$, \$\\mu_r = 1\$\), with the same conductors", mt)
        check(m is not None, "line: filling text")
        er2 = float(m.group(1))
    elif mt.startswith("the outer radius"):
        m = re.fullmatch(r"the outer radius is increased to \$b' = b\^2/a = ([\d.]+)\\ \\mathrm\{mm\}\$, with the same inner conductor and filling", mt)
        check(m is not None and geom == "coax", "line: outer-radius text")
        g2 = d2 ** 2 / d1
        check(abs(float(m.group(1)) * 1e-3 - g2) < 1e-4 * g2, "line: displayed b'")
    elif mt == "the plate width is doubled, with the same gap and filling":
        check(geom == "plates", "line: width text"); g1 = 2 * d1
    elif mt == "the gap is doubled, with the same plate width and filling":
        check(geom == "plates", "line: gap text"); g2 = 2 * d2
    else:
        check(False, "line: unknown modification " + mt)
    v2, Z02 = vz(*LC(geom, g1, g2, er2 * EPS0))
    fv, fz = v2 / v, Z02 / Z0
    ncorr = 0
    for c in p["scl_list"]:
        mm = re.fullmatch(r"\$v\$ (.+); \$Z_0\$ (.+)", c["text"])
        check(mm is not None, "line: scaling choice text")
        a1, a2 = fac_parse(mm.group(1)), fac_parse(mm.group(2))
        match = abs(a1 - fv) < 6e-3 * fv and abs(a2 - fz) < 6e-3 * fz
        ncorr += match
        check(match == (c["correct"] == "true"), f"line: scaling choice {c['text']} marked {c['correct']} (computed v x{fv:.4g}, Z0 x{fz:.4g})")
    check(ncorr == 1 and len(p["scl_list"]) >= 4, "line: scaling MC needs exactly one matching choice")
    check(len({c["text"] for c in p["scl_list"]}) == len(p["scl_list"]), "line: duplicate scaling choices")

    # ---- (b) statements
    truth = {}
    samples = [("coax", 0.4e-3, 0.8e-3), ("coax", 1e-3, 7e-3), ("plates", 10e-3, 1e-3), ("plates", 30e-3, 0.6e-3)]
    truth["LC_mue"] = all(abs(np.prod(LC(g, a, b, e * EPS0)) - MU0 * e * EPS0) < 1e-6 * MU0 * e * EPS0
                          for g, a, b in samples for e in (1.0, 2.25, 9.0))
    vg = [vz(*LC(g, a, b, eps)) for g, a, b in samples]
    v_same = max(x[0] for x in vg) - min(x[0] for x in vg) < 1e-6 * v
    z_diff = max(x[1] for x in vg) - min(x[1] for x in vg) > 0.1 * Z0
    vf = [vz(*LC(geom, d1, d2, e * EPS0)) for e in (1.0, 4.0)]
    truth["geom_med"] = v_same and z_diff and abs(vf[0][0] / vf[1][0] - 2) < 1e-6 and abs(vf[0][1] / vf[1][1] - 2) < 1e-6
    Ltot = [vz(Lp * ell, Cp * ell)[1] for ell in (1.0, 7.0)]
    truth["Z0_length"] = Ltot[1] > Ltot[0] * (1 + 1e-9)
    truth["double_W"] = abs(vz(*plates_LC(2 * 12e-3, 1e-3, eps))[1] / vz(*plates_LC(12e-3, 1e-3, eps))[1] - 2) < 1e-6
    truth["coax_speed"] = vz(*coax_LC(1e-3, 8e-3, eps))[0] > vz(*coax_LC(1e-3, 2e-3, eps))[0] * (1 + 1e-9)
    truth["shrink_C"] = abs(coax_LC(0.3e-3, 0.9e-3, eps)[1] / coax_LC(0.6e-3, 1.8e-3, eps)[1] - 0.5) < 1e-3
    eta = math.sqrt(MU0 / eps)
    truth["Z0_eta"] = all(abs(x[1] - math.sqrt(MU0 / eps)) < 1e-3 * eta for x in vg)

    zN, snaps, E_tot, vl, dz, dtl = ladder(Lp, Cp)
    Vb, Ib, tb = snaps[600]
    Vc, Icc, tc = snaps[1200]
    half = len(zN) // 2
    iR = half + int(np.argmax(Vb[half:])); iL = int(np.argmax(Vb[:half]))
    iR2 = half + int(np.argmax(Vc[half:]))
    iRI = half + int(np.argmax(np.abs(Ib[half:])))
    speedV = (iR2 - iR) * dz / (tc - tb)
    jR2 = half + int(np.argmax(np.abs(Icc[half:])))
    speedI = (jR2 - iRI) * dz / (tc - tb)
    rR, rL = Ib[iR] / Vb[iR], Ib[iL] / Vb[iL]
    check(abs(speedV - v) < 0.01 * v, f"line: ladder pulse speed {speedV} vs {v}")
    truth["VI_speed"] = abs(speedV - speedI) < 0.01 * v and abs(iRI - iR) <= 1
    truth["I_sign"] = abs(rR * Z0 - 1) < 0.01 and abs(rL * Z0 + 1) < 0.01
    truth["ladder"] = abs(speedV - v) < 0.01 * v and abs(1 / rR - Z0) < 0.01 * Z0
    EC = np.sum(0.5 * Cp * Vb[half:] ** 2) * dz
    EL = np.sum(0.5 * Lp * Ib[half:] ** 2) * dz
    truth["equipart"] = abs(EC - EL) < 0.02 * EC
    truth["back_Ipos"] = Ib[iL] > 0
    drop = E_tot[0] - E_tot[-1]
    predicted = Z0 * np.sum(Ib ** 2) * dz * (len(E_tot) - 1) * 200 * dtl     # what an I^2 Z0 per metre loss would remove
    truth["Z0_resist"] = abs(drop - predicted) < 0.1 * predicted and abs(drop) > 0.01 * E_tot[0]
    check(abs(drop) < 1e-2 * E_tot[0], "line: lossless ladder lost energy")
    V0s, I0s, _ = snaps[20]
    i_off = half + 8                                # overlap region early on
    ratio = V0s[i_off] / I0s[i_off] if abs(I0s[i_off]) > 1e-15 else math.inf
    truth["sum_ratio"] = abs(abs(ratio) - Z0) > 0.05 * Z0
    truth["power_cross"] = abs((rR + rL) - 2 / Z0) < 0.05 * (2 / Z0)
    ids = {c["id"] for c in p["facts_list"]}
    check(ids == set(truth), f"line: statement ids differ {ids ^ set(truth)}")
    for c in p["facts_list"]:
        check((c["correct"] == "true") == bool(truth.get(c["id"])),
              f"line: statement {c['id']} marked {c['correct']}, computed {truth.get(c['id'])}")
    check(len({c["text"] for c in p["facts_list"]}) == len(p["facts_list"]), "line: duplicate statements")

    # ---- answer-panel numbers
    check(abs(float(p["V1_str"]) - V1) < 1e-3 * max(1, abs(V1)) and abs(float(p["I1_str"]) - I1 * 1e3) < 1e-3 * abs(I1 * 1e3)
          and abs(float(p["Z0_str"]) - Z0) < 1e-3 * Z0, "line: answer-panel numbers")
