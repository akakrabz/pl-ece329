"""Independent check for waves/reflection-interface (Lecture 25).

Geometry, media and the incident field are parsed from the displayed text. Each wave's H is built with np.cross
(H = u x E/eta, eta = sqrt(mu/eps)) and validated against Faraday's law by finite differences. The reflected and
transmitted amplitudes come from a direct 4 x 4 solve of the boundary conditions (both tangential components of
E and H continuous), not from the Gamma formula. Power fractions come from quadrature of E x H over a period.
The mirror case is solved from tangential E = 0, and J_s = n x H with n built from the geometry (pointing into
region 1). Every checkbox statement is re-decided by a computation.
"""
import math
import re

import numpy as np

IDX = {"x": 0, "y": 1, "z": 2}
E3 = np.eye(3)
MU0 = 4e-7 * math.pi
EPS0 = MU0 / (120 * math.pi) ** 2      # consistent with eta0 = 120 pi
OM = 2 * math.pi * 1e8                  # any frequency; nothing asked depends on it


def parse_medium(tex):
    if tex.startswith("air"):
        m = re.fullmatch(r"air \(\$\\epsilon_r = 1\$, \$\\mu_r = 1\$\)", tex)
        assert m, tex
        return 1.0, 1.0
    m = re.fullmatch(r"a lossless medium with \$\\epsilon_r = ([\d.]+)\$, \$\\mu_r = ([\d.]+)\$", tex)
    assert m, tex
    return float(m.group(1)), float(m.group(2))


def solve_interface(nvec, uinc, Einc, m1, m2):
    """Direct boundary-condition solve at the plane through the origin with normal nvec.
    Returns reflected and transmitted E phasors at the interface (3-vectors)."""
    eta1 = math.sqrt(MU0 * m1[1] / (EPS0 * m1[0]))
    eta2 = math.sqrt(MU0 * m2[1] / (EPS0 * m2[0]))
    t1 = E3[[i for i in range(3) if abs(nvec[i]) < 0.5][0]]
    t2 = np.cross(nvec, t1)
    tang = [t1, t2]
    def residual(R, T):
        Hi = np.cross(uinc, Einc) / eta1
        Hr = np.cross(-uinc, R) / eta1
        Ht = np.cross(uinc, T) / eta2
        eE = Einc + R - T
        eH = Hi + Hr - Ht
        return np.array([eE @ tang[0], eE @ tang[1], eH @ tang[0], eH @ tang[1]])
    r0 = residual(np.zeros(3), np.zeros(3))
    A = np.zeros((4, 4))
    basis = [(tang[0], 0 * t1), (tang[1], 0 * t1), (0 * t1, tang[0]), (0 * t1, tang[1])]
    for k, (R, T) in enumerate(basis):
        A[:, k] = residual(R, T) - r0
    x = np.linalg.solve(A, -r0)
    R = x[0] * tang[0] + x[1] * tang[1]
    T = x[2] * tang[0] + x[3] * tang[1]
    return R, T, eta1, eta2


def faraday_ok(E0, H0, u, m, r):
    """Real plane wave E0 cos(OM t - beta u.r), H0 likewise: check curl E = -mu dH/dt at r."""
    eps, mu = EPS0 * m[0], MU0 * m[1]
    beta = OM * math.sqrt(mu * eps)
    E = lambda rr, t: E0 * math.cos(OM * t - beta * np.dot(u, rr))
    H = lambda rr, t: H0 * math.cos(OM * t - beta * np.dot(u, rr))
    t0, h, ht = 0.37 / OM, 1e-6, 1e-4 / OM
    J = np.zeros((3, 3))
    for j in range(3):
        dr = np.zeros(3); dr[j] = h
        J[:, j] = (E(r + dr, t0) - E(r - dr, t0)) / (2 * h)
    curl = np.array([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]])
    dHdt = (H(r, t0 + ht) - H(r, t0 - ht)) / (2 * ht)
    sc = max(np.linalg.norm(curl), 1e-30)
    return np.linalg.norm(curl + mu * dHdt) < 1e-5 * sc


def avg_S(E0, H0, n=2000):
    ts = np.linspace(0, 2 * math.pi, n + 1)[:-1]
    cs = np.cos(ts)
    return np.mean(np.cross(np.outer(cs, E0), np.outer(cs, H0)), axis=0)


def gamma_tau(m1, m2):
    n = E3[2]
    R, T, e1, e2 = solve_interface(n, n, E3[0], m1, m2)
    return R[0], T[0], e1, e2


def phys(d, check):
    p, c = d["params"], d["correct_answers"]
    m = re.fullmatch(r"([xyz]) \\(lt|gt) 0", p["reg1_side"])
    w = m.group(1)
    s1 = 1 if m.group(2) == "gt" else -1
    m2_ = re.fullmatch(r"([xyz]) \\(lt|gt) 0", p["reg2_side"])
    check(m2_.group(1) == w and (1 if m2_.group(2) == "gt" else -1) == -s1, "refl: region sides")
    me = re.fullmatch(r"(-?)(\d+)\\cos\(\\omega t ([+-]) \\beta_1 ([xyz])\)\\,\\hat\{([xyz])\}", p["E_tex"])
    check(me is not None, "refl: cannot parse E_tex")
    sig = -1 if me.group(1) == "-" else 1
    E0 = float(me.group(2))
    argsg = 1 if me.group(3) == "+" else -1
    check(me.group(4) == w, "refl: argument axis is not the normal")
    a = me.group(5)
    wv = E3[IDX[w]]
    # travel from the argument: f(w, t) = cos(t + argsg w) is unchanged by (w + d, t + d) iff it travels toward +w
    f = lambda ww, t: math.cos(t + argsg * ww)
    trav = 1 if abs(f(0.3 + 0.5, 1.1 + 0.5) - f(0.3, 1.1)) < 1e-12 else -1
    check(abs(f(0.3 - 0.5 * trav, 1.1 + 0.5) - f(0.3, 1.1)) > 1e-3, "refl: travel test inconclusive")
    ui = trav * wv
    check(trav == -s1, "refl: incident wave does not travel from region 1 toward region 2")
    m1 = parse_medium(p["medium1"])
    m2 = parse_medium(p["medium2"])
    check(m1 == (p["er1"], p["mr1"]) and m2 == (p["er2"], p["mr2"]), "refl: medium text vs params")
    Ei = sig * E0 * E3[IDX[a]]
    check(abs(np.dot(Ei, wv)) == 0, "refl: incident E not tangential")
    nvec = wv
    R, T, eta1, eta2 = solve_interface(nvec, ui, Ei, m1, m2)
    Hi = np.cross(ui, Ei) / eta1
    Hr = np.cross(-ui, R) / eta1
    Ht = np.cross(ui, T) / eta2
    b = [ax for ax in "xyz" if ax not in (w, a)][0]
    bv, av = E3[IDX[b]], E3[IDX[a]]
    # Faraday validates the H construction of each wave, in its own region
    r1 = 0.4 * s1 * wv + np.array([0.11, -0.23, 0.17]) * (1 - wv)
    r2 = -0.4 * s1 * wv + np.array([0.11, -0.23, 0.17]) * (1 - wv)
    check(faraday_ok(Ei, Hi, ui, m1, r1) and faraday_ok(T, Ht, ui, m2, r2), "refl: Faraday fails (incident/transmitted)")
    if np.linalg.norm(R) > 1e-12:
        check(faraday_ok(R, Hr, -ui, m1, r1), "refl: Faraday fails (reflected)")
    Gam = np.dot(R, Ei) / np.dot(Ei, Ei)
    tau = np.dot(T, Ei) / np.dot(Ei, Ei)
    check(np.linalg.norm(R - Gam * Ei) < 1e-9 * E0 and np.linalg.norm(T - tau * Ei) < 1e-9 * E0, "refl: R or T not parallel to E_i")
    check(abs(Gam - c["Gam"]) < 1e-9 and abs(tau - c["tau"]) < 1e-9, f"refl: Gamma/tau {Gam},{tau} vs {c['Gam']},{c['tau']}")
    check(abs(np.dot(R, av) - c["Er"]) < 1e-9 * max(1, E0) and abs(np.dot(T, av) - c["Et"]) < 1e-9 * max(1, E0), "refl: E_r/E_t")
    check(abs(np.dot(Hr, bv) * 1e3 - c["Hr"]) < 1e-9 * max(1, abs(c["Hr"])), f"refl: H_r {np.dot(Hr, bv) * 1e3} vs {c['Hr']}")
    # power by quadrature
    Si = avg_S(Ei, Hi) @ ui
    St = avg_S(T, Ht) @ ui
    Sr = avg_S(R, Hr) @ (-ui) if np.linalg.norm(R) > 0 else 0.0
    check(abs(St / Si * 100 - c["frac"]) < 1e-6 * 100, f"refl: transmitted % {St / Si * 100} vs {c['frac']}")
    check(abs(Si - Sr - St) < 1e-9 * Si, "refl: power does not balance")
    check(abs(avg_S(T, Ht) @ wv * 1e3 - c["St"]) < 1e-6 * abs(c["St"]), f"refl: <S_t> {avg_S(T, Ht) @ wv * 1e3} vs {c['St']}")
    # MC (d): compare reflected and incident E at the interface over time
    ts = np.linspace(0, 2 * math.pi, 37)
    dots = [np.dot(R * math.cos(t), Ei * math.cos(t)) for t in ts]
    if np.linalg.norm(R) < 1e-9 * E0:
        okd = "none"
    elif all(x >= -1e-12 for x in dots) and max(dots) > 0:
        okd = "same"
    elif all(x <= 1e-12 for x in dots) and min(dots) < 0:
        okd = "opp"
    else:
        okd = "quad"
    check((okd == "none") == bool(p["matched"]), "refl: matched flag")
    for ch in p["refl"]:
        check((ch["correct"] == "true") == (ch["id"] == okd), f"refl: (d) choice {ch['id']} key (expected {okd})")
    # mirror: tangential E = 0 at the surface
    Rp = -Ei
    Hp = np.cross(ui, Ei) / eta1 + np.cross(-ui, Rp) / eta1
    n_out = s1 * wv                           # out of the conductor (region 2) into region 1
    Js = np.cross(n_out, Hp)
    check(abs(np.linalg.norm(Js) * 1e3 - c["Js"]) < 1e-9 * c["Js"], f"refl: |J_s| {np.linalg.norm(Js) * 1e3} vs {c['Js']}")
    def unit_of(text):
        mm = re.match(r"\$([+-])\\hat\{([xyz])\}\$", text)
        return (1 if mm.group(1) == "+" else -1) * E3[IDX[mm.group(2)]]
    jd = Js / np.linalg.norm(Js)
    for ch in p["js"]:
        is_dir = np.allclose(unit_of(ch["text"]), jd)
        check((ch["correct"] == "true") == is_dir, f"refl: (e) choice {ch['id']} key")
    check(np.allclose(jd, Ei / np.linalg.norm(Ei)), "refl: J_s should be along the incident E")

    # checkbox statements
    ts2 = np.linspace(0, 2 * math.pi, 13)
    E_cont = all(np.allclose((Ei + R) * math.cos(t) - np.dot((Ei + R) * math.cos(t), wv) * wv,
                             T * math.cos(t) - np.dot(T * math.cos(t), wv) * wv, atol=1e-9) for t in ts2)
    H_cont = all(np.allclose((Hi + Hr) * math.cos(t), Ht * math.cos(t), atol=1e-12) for t in ts2)
    g_m, t_m, _, _ = gamma_tau((1, 1), (3, 3))
    match = abs(g_m) < 1e-12 and abs(1 / math.sqrt(MU0 * 3 * EPS0 * 3) - 1 / math.sqrt(MU0 * EPS0) / 3) < 1e-3
    g2, t2, e1, e2 = gamma_tau((4, 1), (1, 1))          # eta2 = 2 eta1
    n = E3[2]
    Rx, Tx, _, _ = solve_interface(n, n, E3[0], (4, 1), (1, 1))
    fr = (avg_S(Tx, np.cross(n, Tx) / e2) @ n) / (avg_S(E3[0], np.cross(n, E3[0]) / e1) @ n)
    tau_gt1 = t2 > 1 and fr < 1
    Rs, Ts, _, _ = solve_interface(nvec, -ui, Ei, m2, m1)
    swap = abs(np.dot(Rs, Ei) / np.dot(Ei, Ei) + Gam) < 1e-9
    H_ratio = abs(np.dot(Hr, bv) / np.dot(Hi, bv) + Gam) < 1e-9
    tau_1mg = all(abs(gamma_tau(x, y)[1] - (1 - gamma_tau(x, y)[0])) < 1e-9 for x, y in [((1, 1), (4, 1)), ((9, 1), (1, 4))])
    inv_hi = g2 < 0                                      # eta2 > eta1 here
    g_s = gamma_tau((1, 4), (4, 1))[0]
    same_speed = abs(MU0 * 4 * EPS0 - MU0 * EPS0 * 4) < 1e-30
    speed = same_speed and abs(g_s) < 1e-12
    pec_both = np.linalg.norm(Hp - np.dot(Hp, wv) * wv) < 1e-12
    tau_le1 = not (t2 > 1)
    g3, t3, e13, e23 = gamma_tau((1, 1), (4, 1))
    R3, T3, _, _ = solve_interface(n, n, E3[0], (1, 1), (4, 1))
    fr3 = (avg_S(T3, np.cross(n, T3) / e23) @ n) / (avg_S(E3[0], np.cross(n, E3[0]) / e13) @ n)
    frac_tau2 = abs(fr3 - t3 ** 2) < 1e-9 and abs(fr - t2 ** 2) < 1e-9
    amp = abs(np.dot(T, av)) > abs(np.dot(Ei, av)) * (1 + 1e-12)
    truth = {"E_cont": E_cont, "H_cont": H_cont, "match": match, "tau_gt1": tau_gt1, "swap": swap, "H_ratio": H_ratio,
             "tau_1mg": tau_1mg, "inv_hi": inv_hi, "speed": speed, "pec_both": pec_both, "tau_le1": tau_le1,
             "frac_tau2": frac_tau2, "amp": amp}
    expect = {"E_cont": True, "H_cont": True, "match": True, "tau_gt1": True, "swap": True, "H_ratio": True,
              "tau_1mg": False, "inv_hi": False, "speed": False, "pec_both": False, "tau_le1": False, "frac_tau2": False}
    if p["matched"]:
        check(abs(np.dot(Hr, bv)) < 1e-15, "refl: matched case should have no reflected H")
    for k, val in expect.items():
        check(bool(truth[k]) == val, f"refl: computed truth of {k} is {truth[k]}, expected {val} (self-test)")
    facts = p["facts"]
    check({f_["id"] for f_ in facts} == set(truth) and len(facts) == len(truth), "refl: statement ids")
    for f_ in facts:
        check((f_["correct"] == "true") == bool(truth[f_["id"]]), f"refl: statement {f_['id']} key")
        check(f_["verdict"] == f_["correct"], "refl: verdict text")
