"""Independent check for transmission-lines/two-line-junction (Lecture 30).

The circuit is read from the displayed statement. Two lines are simulated exactly by the method of characteristics
(dt = 0.25 us). The generator end is solved from KVL through R_g, the load end from its Ohm's law (or I = 0), and the
junction from continuity of V plus KCL including the shunt resistor (V_J + Z1 I1 = 2 f_in, V_J - Z2 I2 = 2 g_in,
I1 = V_J/R + I2); no reflection or transmission coefficient is used. The coefficients asked in (a) are read off a
single junction solve with a unit wave from each side; (b)-(d) come from the simulation; the matching-resistor MC is
decided by solving the junction with each candidate resistor; every checkbox statement is re-decided by computation.
"""
import re

import numpy as np

DT = 0.25


def junction(Z1, Z2, R, f_in, g_in, Rser=0.0):
    """Unknowns V1 (line-1 side voltage), I1, I2. Line 1: V1 + Z1 I1 = 2 f_in. Line 2 side voltage V2 = V1 - Rser I2
    (series resistor in the upper conductor): V2 - Z2 I2 = 2 g_in. KCL: I1 = V1/R + I2 (shunt on the line-1 side node)."""
    gR = 0.0 if R is None else 1.0 / R
    A = np.array([[1.0, Z1, 0.0], [1.0, 0.0, -Rser - Z2], [-gR, 1.0, -1.0]])
    V1, I1, I2 = np.linalg.solve(A, [2 * f_in, 2 * g_in, 0.0])
    V2 = V1 - Rser * I2
    return V1, I1, I2, (V1 - Z1 * I1) / 2, (V2 + Z2 * I2) / 2      # ..., outgoing g on line 1, outgoing f on line 2


class Cascade:
    def __init__(self, Z1, Z2, Rg, R, RL, V0, N1, N2, nsteps):
        self.N1, self.N2 = N1, N2
        self.f1 = [0.0] * (nsteps + 1)    # forward wave leaving z = 0
        self.g1 = [0.0] * (nsteps + 1)    # backward wave leaving the junction on line 1
        self.f2 = [0.0] * (nsteps + 1)    # forward wave leaving the junction on line 2
        self.g2 = [0.0] * (nsteps + 1)    # backward wave leaving the load
        self.VJ = [0.0] * (nsteps + 1)
        self.VJa = [0.0] * (nsteps + 1)   # part of V_J due to the wave arriving from line 1 (linearity)
        self.VJb = [0.0] * (nsteps + 1)   # part due to the wave arriving from line 2
        self.VL = [0.0] * (nsteps + 1)
        Ag = np.linalg.inv(np.array([[1.0, Rg], [1.0, -Z1]]))                   # V + Rg I = V0, V - Z1 I = 2 g_in
        Al = np.linalg.inv(np.array([[1.0, Z2], [0.0, 1.0]]) if RL is None else np.array([[1.0, Z2], [1.0, -RL]]))
        gR = 0.0 if R is None else 1.0 / R
        Aj = np.linalg.inv(np.array([[1.0, Z1, 0.0], [1.0, 0.0, -Z2], [-gR, 1.0, -1.0]]))   # same system as junction()
        ag, al, aj = Ag.tolist(), Al.tolist(), Aj.tolist()
        for n in range(nsteps + 1):
            gin0 = self.g1[n - N1] if n >= N1 else 0.0
            V_ = ag[0][0] * V0 + ag[0][1] * 2 * gin0
            I_ = ag[1][0] * V0 + ag[1][1] * 2 * gin0
            self.f1[n] = (V_ + Z1 * I_) / 2
            fJ = self.f1[n - N1] if n >= N1 else 0.0
            gJ = self.g2[n - N2] if n >= N2 else 0.0
            VJ = aj[0][0] * 2 * fJ + aj[0][1] * 2 * gJ
            self.VJa[n], self.VJb[n] = aj[0][0] * 2 * fJ, aj[0][1] * 2 * gJ
            I1 = aj[1][0] * 2 * fJ + aj[1][1] * 2 * gJ
            I2 = aj[2][0] * 2 * fJ + aj[2][1] * 2 * gJ
            self.g1[n], self.f2[n], self.VJ[n] = (VJ - Z1 * I1) / 2, (VJ + Z2 * I2) / 2, VJ
            fL = self.f2[n - N2] if n >= N2 else 0.0
            VL = al[0][0] * 2 * fL
            IL = al[1][0] * 2 * fL
            self.g2[n], self.VL[n] = (VL - Z2 * IL) / 2, VL


def phys(d, check):
    p, c = d["params"], d["correct_answers"]
    stm = "{V0} {Rg_s} {Z1} {v1} {ell1} {Z2_s} {v2} {ell2}".format(**p)
    V0, Rg, Z1, v1, ell1, Z2, v2, ell2 = [float(x) for x in stm.split()]
    if p["load_txt"] == "is left open":
        RL = None
    else:
        RL = float(re.fullmatch(r"ends in a resistor \$R_L = ([\d.]+)\\ \\Omega\$", p["load_txt"]).group(1))
    if p["shunt_txt"]:
        R = float(re.match(r"A resistor \$R = ([\d.]+)\\ \\Omega\$ is connected across the junction", p["shunt_txt"]).group(1))
    else:
        R = None
    check((R is None) == (not p["shunt"]) and (RL is None) == (p["RL"] == -1), "tlj: parsed circuit vs params")
    check(abs(float(p["ellsum"]) - ell1 - ell2) < 1e-9, "tlj: total length")
    T1, T2 = ell1 / v1, ell2 / v2
    N1, N2 = int(round(T1 / DT)), int(round(T2 / DT))
    check(abs(N1 * DT - T1) < 1e-12 and abs(N2 * DT - T2) < 1e-12, "tlj: delays off the grid")
    tol = 1e-9 * V0

    # (a) one junction solve per side, unit incident wave, nothing else arriving
    _, _, _, gref, ftr = junction(Z1, Z2, R, 1.0, 0.0)
    G12, t12 = gref, ftr
    # from line 2: incoming g2 = 1; reflected = outgoing f2, transmitted = outgoing g1
    _, _, _, gtr, fref = junction(Z1, Z2, R, 0.0, 1.0)
    G21, t21 = fref, gtr
    for nm, val in (("G12", G12), ("t12", t12), ("G21", G21), ("t21", t21)):
        check(abs(val - c[nm]) < 1e-9, f"tlj: {nm} {val} vs {c[nm]}")
    check(abs(G12) > 1e-9, "tlj: matched junction slipped through")

    # (b)-(d) simulation
    tc = float(p["tc_s"])
    nc = int(round(tc / DT))
    check(abs(nc * DT - tc) < 1e-12, "tlj: tc off the grid")
    nsteps = max(nc + 8, int(round(4000 / DT)))
    S = Cascade(Z1, Z2, Rg, R, RL, V0, N1, N2, nsteps)
    n1 = N1 + 1                                   # just after the first arrival at the junction
    check(abs(S.f2[n1] - c["V2f"]) < tol, f"tlj: first transmitted {S.f2[n1]} vs {c['V2f']}")
    check(abs(S.g1[n1] - c["V1r"]) < tol, f"tlj: first reflected {S.g1[n1]} vs {c['V1r']}")
    check(all(x == 0 for x in S.f2[:N1]) and S.f2[N1] != 0, "tlj: first transmission not at T1")
    check(abs(S.VJ[nc] - c["VJ"]) < tol, f"tlj: V_J(tc) {S.VJ[nc]} vs {c['VJ']}")
    check(abs(S.VJ[nc - 1] - S.VJ[nc]) < tol and abs(S.VJ[nc + 1] - S.VJ[nc]) < tol, "tlj: tc too close to a junction step")
    # tc lies after both second-order echoes (3 T1 from the generator, T1 + 2 T2 from the load)
    check(tc > 3 * T1 and tc > T1 + 2 * T2, "tlj: tc before the second-order echoes")
    check(c["VJ"] == 0 or abs(c["VJ"]) >= 0.0199 * V0, "tlj: tiny V_J")
    VLinf = S.VL[-1]
    check(abs(S.VL[-1] - S.VL[-1 - int(round(20 / DT))]) < 1e-9 * V0, "tlj: not settled")
    check(abs(VLinf - c["VL"]) < 1e-8 * V0, f"tlj: V_L(inf) {VLinf} vs {c['VL']}")
    check(abs(S.VJ[-1] - VLinf) < 1e-8 * V0, "tlj: DC junction != load")
    aV = float(p["atolV"])
    check(0 < aV <= 3e-4 * V0, "tlj: atol range")
    for k in ("V2f", "V1r", "VL"):
        check(c[k] == 0 or abs(c[k]) > 25 * aV, f"tlj: atol not small vs {k}")
    # V_J is a running sum: its jumps (from the simulation) give the 3-s.f. rounding noise
    def r3(x):
        return float(f"{x:.3g}")
    jumps = [arr[n] - arr[n - 1] for arr in (S.VJa, S.VJb) for n in range(1, nc + 1) if abs(arr[n] - arr[n - 1]) > 1e-12 * V0]
    check(abs(sum(jumps) - c["VJ"]) < tol, "tlj: junction jumps do not add up")
    aJ = float(p["atolJ"])
    expect = max(2e-4 * V0, 0.003 * sum(abs(x) for x in jumps))
    check(abs(aJ - expect) <= 0.06 * expect, f"tlj: atolJ {aJ} vs expected {expect}")
    tolJ = aJ + 0.01 * abs(c["VJ"])
    check(abs(sum(r3(x) for x in jumps) - c["VJ"]) <= tolJ, "tlj: 3-s.f. chain rejected for V_J")
    check(all(abs(x) > 1.2 * tolJ for x in jumps), "tlj: dropping one junction step would be accepted")
    # no credit for the DC value or for one junction arrival too many (unless they equal the answer)
    n_nx = next((n for n in range(nc + 1, len(S.VJ) - 1) if abs(S.VJ[n] - S.VJ[nc]) > 1e-12 * V0), None)
    for other in [VLinf] + ([] if n_nx is None else [S.VJ[n_nx + 1]]):
        check(abs(other - c["VJ"]) < 1e-9 * V0 or abs(other - c["VJ"]) > 1.2 * tolJ, f"tlj: DC or next level {other} would be accepted")

    # (e) matching resistor: solve the junction with each candidate (no other shunt) and test Gamma_12 = 0
    nok = 0
    for ch in p["e_choices"]:
        txt = ch["text"]
        m = re.match(r"a resistor of ([\d.]+) Ω (connected across the junction|inserted in series)", txt)
        if m:
            Rc = float(m.group(1))
            if m.group(2).startswith("connected"):
                _, _, _, gref_c, _ = junction(Z1, Z2, Rc, 1.0, 0.0)
            else:
                _, _, _, gref_c, _ = junction(Z1, Z2, None, 1.0, 0.0, Rser=Rc)
            truth = abs(gref_c) < 1e-12
        else:
            check(txt.startswith("none:"), "tlj: unknown (e) choice")
            truth = False                                     # some choice is a valid resistor (tested below)
        nok += truth
        check((ch["correct"] == "true") == truth, f"tlj: (e) choice {ch['id']} key wrong")
    check(nok == 1, "tlj: (e) needs exactly one valid resistor")
    check(4 <= len(p["e_choices"]) <= 5, "tlj: (e) choice count")

    # (f) statements
    truth = {}
    truth["inv"] = abs(G21 + G12) < 1e-12
    truth["tau1pG"] = abs(t12 - (1 + G12)) < 1e-12
    # power: incident V+^2/Z1 against reflected and transmitted
    truth["tau_gt1"] = True
    pin, pref, ptr = 1 / Z1, G12 ** 2 / Z1, t12 ** 2 / Z2
    truth["power"] = abs(pin - pref - ptr) < 1e-12 / Z1
    if t12 > 1:
        check(ptr < pin, "tlj: tau > 1 but more power transmitted")
    # counterexample for the 'every junction' claim: Z2 = 3 Z1, no resistor
    _, _, _, _, ta = junction(1.0, 3.0, None, 1.0, 0.0)
    _, _, _, tb, _ = junction(1.0, 3.0, None, 0.0, 1.0)
    truth["tautau"] = abs(ta * tb - 1) < 1e-12
    # speeds: rerun the cascade with line 2 twice as slow; the first reflection must be unchanged
    S2 = Cascade(Z1, Z2, Rg, R, RL, V0, N1, 2 * N2, N1 + 2)
    truth["speeds"] = abs(S2.g1[n1] - S.g1[n1]) > tol
    S3 = Cascade(Z1, Z1, Rg, None, RL, V0, N1, N2 + 3, N1 + 2)
    truth["equalZ"] = abs(S3.g1[n1]) < tol
    tg_sim = S.f1[0] / V0
    gen_ok = abs(tg_sim - Z1 / (Rg + Z1)) < 1e-12
    nL = N1 + N2 + 1
    GL_sim = (S.g2[nL] / S.f2[nL - N2]) if abs(S.f2[nL - N2]) > 0 else None
    load_ok = GL_sim is not None and abs(GL_sim - (1.0 if RL is None else (RL - Z2) / (RL + Z2))) < 1e-12
    truth["gen_Z1"] = gen_ok and load_ok
    truth["back_tau12"] = abs(t21 - t12) < 1e-12
    I1, I2 = junction(Z1, Z2, R, 1.0, 0.0)[1:3]
    truth["Itrans"] = abs(I2 / (1 / Z1) - t12) < 1e-12
    truth["dc_equal"] = abs(S.VJ[-1] - S.VL[-1]) < 1e-8 * V0
    # reflected current over incident current, from the junction solve
    Iinc = 1 / Z1
    Iref = I1 - Iinc
    truth["Irefl"] = abs(Iref / Iinc + G12) < 1e-12
    for x in p["facts"]:
        check(x["id"] in truth, f"tlj: statement {x['id']} not re-decided")
        if x["id"] in truth:
            check((x["correct"] == "true") == bool(truth[x["id"]]), f"tlj: statement {x['id']} keyed {x['correct']}, computed {truth[x['id']]}")
    nt_ = sum(x["correct"] == "true" for x in p["facts"])
    check(nt_ >= 4 and len(p["facts"]) - nt_ >= 4, "tlj: pool cannot fill 5 with 1-4 correct")
    # panel strings
    for s, k in (("V2f_s", "V2f"), ("V1r_s", "V1r"), ("VJ_s", "VJ"), ("VL_s", "VL")):
        check(abs(float(p[s]) - c[k]) <= 5e-4 * max(1, abs(c[k])), f"tlj: panel {s} vs {k}")
