"""Independent check for transmission-lines/bounce-step-response (Lecture 28).

Every bounce-diagram value is re-derived by an exact time-domain simulation of the line (method of
characteristics: forward and backward wave samples shift one cell per time step dt = dz/v). At each end the
terminal voltage and current are solved directly from the circuit law there (KVL through R_g at z = 0, Ohm's law
or I = 0 at z = ell) together with V = f + g, Z0 I = f - g; no reflection or injection coefficient is used. The
coefficients asked in (a) are then read off the simulation as wave ratios. The what-if MC is checked by simulating
the modified circuit and testing every choice's claim.
"""
import math
import re

import numpy as np


def fnum(s):
    s = str(s)
    m = re.fullmatch(r"(-?[\d.]+)\\times10\^\{(-?\d+)\}", s)
    return float(m.group(1)) * 10 ** int(m.group(2)) if m else float(s)


class Line:
    """Step source Vg u(t) behind Rg at z = 0, load RL at z = ell (None = open). Exact characteristic stepping:
    f0[n] is the forward wave leaving z = 0 at step n, gN[n] the backward wave leaving z = ell; with dt = T/N the
    forward wave at node k and step n is f0[n - k], the backward one gN[n - (N - k)]."""

    def __init__(self, Z0, Rg, RL, Vg, N, nsteps):
        self.Z0, self.N = Z0, N
        # z = 0: unknowns (V, I) from  V + Rg I = Vs  (KVL through Rg)  and  V - Z0 I = 2 g_in
        Ag = np.linalg.inv(np.array([[1.0, Rg], [1.0, -Z0]]))
        # z = ell: unknowns (V, I) from  V + Z0 I = 2 f_in  and the load law
        Al = np.linalg.inv(np.array([[1.0, Z0], [0.0, 1.0]]) if RL is None else np.array([[1.0, Z0], [1.0, -RL]]))
        self.f0 = np.zeros(nsteps + 1)
        self.gN = np.zeros(nsteps + 1)
        for n in range(nsteps + 1):
            g_in = self.gN[n - N] if n >= N else 0.0
            V0, I0 = Ag @ np.array([Vg, 2 * g_in])
            self.f0[n] = (V0 + Z0 * I0) / 2
            f_in = self.f0[n - N] if n >= N else 0.0
            VL, IL = Al @ np.array([2 * f_in, 0.0])
            self.gN[n] = (VL - Z0 * IL) / 2

    def F(self, k, n):
        return self.f0[n - k] if n >= k else 0.0

    def G(self, k, n):
        m = n - (self.N - k)
        return self.gN[m] if m >= 0 else 0.0

    def V(self, k, n):
        return self.F(k, n) + self.G(k, n)

    def I(self, k, n):
        return (self.F(k, n) - self.G(k, n)) / self.Z0

    def Vrow(self, n):
        return np.array([self.V(k, n) for k in range(self.N + 1)])

    def Irow(self, n):
        return np.array([self.I(k, n) for k in range(self.N + 1)])


def gen_reflection(Rg, Z0):
    """Unit backward wave arriving at a dead source (KVL with Vs = 0): outgoing forward wave."""
    V0, I0 = np.linalg.solve(np.array([[1.0, Rg], [1.0, -Z0]]), [0.0, 2.0])
    return (V0 + Z0 * I0) / 2


def phys(d, check):
    p, c = d["params"], d["correct_answers"]
    Z0 = float(p["Z0"]); Vg = float(p["Vg"])
    Rg = float(p["Rg"])
    RL = None if p["RL"] == -1 else float(p["RL"])
    # statement text agrees with the numbers
    if p["gk"] == "ideal":
        check(Rg == 0 and "R_g = 0" in p["Rg_txt"], "bsr: ideal-source text")
    else:
        check(f"R_g = {p['Rg_s']}\\ " in p["Rg_txt"] and abs(fnum(p["Rg_s"]) - Rg) < 1e-9, "bsr: Rg text")
    if RL is None:
        check("open" in p["load_txt"], "bsr: open-load text")
    elif RL == 0:
        check("short" in p["load_txt"], "bsr: short text")
    else:
        check(f"R_L = {p['RL_sym']}" in p["load_txt"], "bsr: RL text")
    # line timing from the displayed givens
    us = "mu" in p["tu"]
    check((p["lu"] == r"\mathrm{m}") == us, "bsr: unit pair")
    c_disp = 300.0 if us else 30.0                      # c = 3e8 m/s in m/us or cm/ns
    er = float(p["er"])
    if er == 1:
        check(p["er_txt"] == "air", "bsr: air text")
    else:
        check(f"\\epsilon_r = {er:g}" in p["er_txt"], "bsr: eps_r text")
    v = c_disp / math.sqrt(er)
    ell = fnum(p["ell_s"])
    T = ell / v
    N = 48
    dt = T / N

    def kz(zs):
        k = fnum(zs) / ell * N
        check(abs(k - round(k)) < 1e-9 and 0 <= round(k) <= N, f"bsr: point {zs} not on the grid")
        return int(round(k))

    def nt(ts):
        n = fnum(ts) / dt
        check(abs(n - round(n)) < 1e-9, f"bsr: time {ts} not on the grid")
        return int(round(n))

    tmax = max(fnum(p[k]) for k in ("t1_s", "t2_s", "t3_s", "tq_s"))
    nsteps = int(round(max(tmax / dt + N, 200 * N)))
    L = Line(Z0, Rg, RL, Vg, N, nsteps)

    # (a) the first wave and the two reflection coefficients, as wave ratios in the simulation
    Vplus = L.F(0, N // 2)                             # forward wave leaving z = 0 before any echo returns
    check(abs(L.G(0, N // 2)) < 1e-12, "bsr: echo at z = 0 before 2T")
    check(abs(Vplus - c["Vplus"]) < 1e-9 * Vg, f"bsr: V+ {Vplus} vs {c['Vplus']}")
    nL = N + N // 4
    GL = L.G(N, nL) / L.F(N, nL)
    check(abs(GL - c["GL"]) < 1e-9, f"bsr: Gamma_L {GL} vs {c['GL']}")
    Gg = gen_reflection(Rg, Z0)
    check(abs(Gg - c["Gg"]) < 1e-9, f"bsr: Gamma_g {Gg} vs {c['Gg']}")

    # (b) voltages at z_p; times must not sit on a step (value unchanged T/16 before and after)
    kp = kz(p["zp_s"])
    for name, key in (("V1", "t1_s"), ("V2", "t2_s"), ("V3", "t3_s")):
        n = nt(p[key])
        val = L.V(kp, n)
        check(abs(L.V(kp, n - 3) - val) < 1e-9 * Vg and abs(L.V(kp, n + 3) - val) < 1e-9 * Vg, f"bsr: {name} time too close to a step")
        check(abs(val - c[name]) < 1e-9 * Vg, f"bsr: {name} {val} vs {c[name]}")
    tb = [fnum(p[k]) for k in ("t1_s", "t2_s", "t3_s")]
    check(tb[0] < tb[1] < tb[2], "bsr: times not increasing")

    # (c) current at z_q
    kq = kz(p["zq_s"])
    n = nt(p["tq_s"])
    iq = L.I(kq, n) * 1e3
    check(abs(L.I(kq, n - 3) - L.I(kq, n)) < 1e-12 and abs(L.I(kq, n + 3) - L.I(kq, n)) < 1e-12, "bsr: tq too close to a step")
    check(abs(iq - c["Iq"]) < 1e-9 * Vg / Z0 * 1e3, f"bsr: Iq {iq} vs {c['Iq']}")
    check(kq != kp, "bsr: z_q equals z_p")

    # (d) late-time values, uniform along the line
    Vinf, Iinf = L.Vrow(nsteps), L.Irow(nsteps) * 1e3
    check(np.ptp(Vinf) < 1e-9 * Vg and np.ptp(Iinf) < 1e-9 * Vg / Z0 * 1e3, "bsr: late state not uniform / not settled")
    check(np.max(np.abs(Vinf - L.Vrow(nsteps - 2 * N))) < 1e-9 * Vg, "bsr: not settled")
    check(abs(Vinf[0] - c["Vdc"]) < 1e-8 * Vg, f"bsr: Vdc {Vinf[0]} vs {c['Vdc']}")
    check(abs(Iinf[0] - c["Idc"]) < 1e-8 * Vg / Z0 * 1e3, f"bsr: Idc {Iinf[0]} vs {c['Idc']}")

    # tolerances: atol nonzero and small; nonzero answers not tiny (so rtol stays meaningful)
    aV, aI = float(p["atolV"]), float(p["atolI"])
    check(0 < aV <= 3e-4 * Vg and 0 < aI <= 3e-4 * Vg / Z0 * 1e3, "bsr: atol out of range")
    # running-sum answers: the separate wave steps seen at the point (forward and backward components) give the
    # expected 3-s.f. rounding noise; the atol must absorb it, and dropping any one step must still be wrong.
    def wave_steps(k, n_end, current):
        out = []
        for comp, sg in ((L.F, 1.0), (L.G, -1.0 if current else 1.0)):
            prev = 0.0
            for n in range(n_end + 1):
                val = comp(k, n)
                if abs(val - prev) > 1e-12 * Vg:
                    out.append(sg * (val - prev) * (1e3 / Z0 if current else 1.0))
                prev = val
        return out

    def r3(x):
        return float(f"{x:.3g}")
    for name, key, akey, k, cur in (("V1", "t1_s", "atolV1", kp, False), ("V2", "t2_s", "atolV2", kp, False),
                                    ("V3", "t3_s", "atolV3", kp, False), ("Iq", "tq_s", "atolIq", kq, True)):
        steps = wave_steps(k, nt(p[key]), cur)
        check(abs(sum(steps) - c[name]) < 1e-9 * max(1, abs(c[name])), f"bsr: {name} steps do not add up")
        floor = 2e-4 * Vg * (1e3 / Z0 if cur else 1.0)
        expect = max(floor, 0.003 * sum(abs(x) for x in steps))
        a = float(p[akey])
        check(abs(a - expect) <= 0.06 * expect, f"bsr: {akey} {a} vs expected {expect}")
        tol = a + 0.01 * abs(c[name])
        check(abs(sum(r3(x) for x in steps) - c[name]) <= tol, f"bsr: 3-s.f. chain rejected for {name}")
        check(all(abs(x) > 1.2 * tol for x in steps), f"bsr: dropping one step of {name} would be accepted")
        # no credit for the DC value or for one crossing too many (unless they equal the answer)
        val = (lambda n: L.I(k, n) * 1e3) if cur else (lambda n: L.V(k, n))
        n0 = nt(p[key])
        n1 = next((n for n in range(n0 + 1, nsteps) if abs(val(n) - val(n0)) > 1e-12 * Vg), None)   # None: settled
        for other in [c["Idc"] if cur else c["Vdc"]] + ([] if n1 is None else [val(n1 + 1)]):
            check(abs(other - c[name]) < 1e-9 * max(1, abs(c[name])) or abs(other - c[name]) > 1.2 * tol,
                  f"bsr: {name}: DC value or next level {other} would be accepted")
    for k in ("Vplus", "V1", "V2", "V3", "Vdc"):
        check(c[k] == 0 or abs(c[k]) >= 0.0199 * Vg, f"bsr: {k} tiny but nonzero")
    check(c["Vdc"] == 0 or aV < 0.02 * abs(c["Vdc"]), "bsr: atol vs Vdc")
    for k in ("Iq", "Idc"):
        check(c[k] == 0 or abs(c[k]) >= 0.0199 * Vg / Z0 * 1e3, f"bsr: {k} tiny but nonzero")
    # panel strings agree with the keys
    for s, k in (("Vb1_s", "V1"), ("Vb2_s", "V2"), ("Vb3_s", "V3"), ("Iq_s", "Iq"), ("Vdc_s", "Vdc"), ("Idc_s", "Idc"), ("Vplus_s", "Vplus")):
        check(abs(fnum(p[s]) - c[k]) <= 5e-4 * max(1, abs(c[k])), f"bsr: panel {s} vs {k}")

    # (e) what-if MC, by simulating the modified circuit
    ch = p["e_choices"]
    check(sum(x["correct"] == "true" for x in ch) == 1, "bsr: (e) needs one key")
    sc = p["scen"]
    if sc == "matchgen":
        check(p["gk"] != "matched", "bsr: matchgen offered for a matched generator")
        L2 = Line(Z0, Z0, RL, Vg, N, 8 * N)
        rows = [L2.Vrow(n) for n in range(8 * N + 1)]
        ch_steps = [n for n in range(1, len(rows)) if np.max(np.abs(rows[n] - rows[n - 1])) > 1e-12 * Vg]
        last = max(ch_steps)
        check(last < len(rows) - 2 * N, "bsr: matched-generator line still changing")
        t_settle = last * dt
        for x in ch:
            truth = (x["claim"] == "settle" and abs(x["tT"] * T - t_settle) < 1e-9 * T)
            check((x["correct"] == "true") == truth, f"bsr: (e) choice {x['id']} key wrong (settles at {t_settle})")
            if x["claim"] == "settle":
                m = re.search(r"\$t = (\d*)T\$", x["text"])            # symbolic: T, 2T, 4T
                check(m is not None and int(m.group(1) or 1) == x["tT"], "bsr: (e) time text")
    else:
        check(p["gk"] != "ideal", "bsr: open/short what-if with an ideal source")
        RL2 = None if sc == "open" else 0.0
        ns2 = 300 * N
        L2 = Line(Z0, Rg, RL2, Vg, N, ns2)
        f_arr = L2.F(N, nL)
        VT, IT, gT = L2.V(N, nL), L2.I(N, nL), L2.G(N, nL)
        Vlate, Ilate = L2.Vrow(ns2), L2.Irow(ns2)
        VL_all = np.array([L2.V(N, n) for n in range(ns2 + 1)])
        IL_all = np.array([L2.I(N, n) for n in range(ns2 + 1)])
        Vlater = np.array([L2.Vrow(n) for n in range(N, 6 * N)])
        tol = 1e-9 * Vg
        claims = {}
        if sc == "open":
            claims["ok"] = (abs(VT - 2 * f_arr) < tol and np.max(np.abs(IL_all)) < tol / Z0
                            and np.max(np.abs(Vlate - Vg)) < 1e-8 * Vg and np.max(np.abs(Ilate)) < 1e-8 * Vg / Z0)
            claims["zeroV"] = abs(VT) < tol
            claims["noecho"] = abs(gT) < tol and np.max(np.abs(Vlater - Vplus)) < tol
            claims["dc"] = abs(VT - 2 * f_arr) < tol and np.max(np.abs(Vlate - Vplus)) < 1e-8 * Vg
            claims["Idouble"] = abs(IT - 2 * f_arr / Z0) < tol / Z0
        else:
            claims["ok"] = (np.max(np.abs(VL_all)) < tol and abs(IT - 2 * f_arr / Z0) < tol / Z0
                            and np.max(np.abs(Ilate - Vg / Rg)) < 1e-8 * Vg / Z0)
            claims["Vdouble"] = abs(VT - 2 * f_arr) < tol
            claims["sign"] = gT * f_arr > 0
            claims["dc"] = (np.max(np.abs(VL_all)) < tol and abs(IT - 2 * f_arr / Z0) < tol / Z0
                            and np.max(np.abs(Ilate - Vg / (Rg + Z0))) < 1e-8 * Vg / Z0)
            claims["noI"] = abs(IT) < tol / Z0
        check(set(claims) == {x["id"] for x in ch}, "bsr: (e) choice ids")
        for x in ch:
            check((x["correct"] == "true") == bool(claims.get(x["id"])), f"bsr: (e) {sc} choice {x['id']} key wrong")
