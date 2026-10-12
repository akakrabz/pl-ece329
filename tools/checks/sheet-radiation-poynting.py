"""Independent check for waves/sheet-radiation-poynting (Lecture 19).

The sheet current is parsed from the displayed LaTeX; the medium from the displayed text. Fields are built as vectors
with np.cross from the sheet formulas E = -(eta/2) J_s(t - d/v), H = (1/2) J_s(t - d/v) x n_hat, with
v = 1/sqrt(mu eps) and eta = sqrt(mu/eps) from mu0 and eps0 = 1/(mu0 c^2) (c = 3e8, as the statement says).
Then:
  * wavelength from the numerical spacing of zero crossings of E along w at a fixed instant; beta = 2 pi/lambda;
    v_p from tracking a zero crossing in time;
  * Faraday's and Ampere-Maxwell's laws by central finite differences at the point of part (b) (validates the signs);
  * the boundary conditions at the sheet (E_t continuous, n x (H1 - H2) = J_s);
  * S = E x H with np.cross; the mirror point; every checkbox statement re-decided by a computation.
"""
import math
import re

import numpy as np

C = 3e8
MU0 = 4e-7 * math.pi
EPS0 = 1.0 / (MU0 * C * C)
IDX = {"x": 0, "y": 1, "z": 2}
E3 = np.eye(3)


def parse_unit(tex):
    m = re.search(r"([+-])\\hat\{([xyz])\}", tex)
    return (1 if m.group(1) == "+" else -1) * E3[IDX[m.group(2)]]


def curl_fd(F, r, h):
    J = np.zeros((3, 3))
    for j in range(3):
        dr = np.zeros(3); dr[j] = h
        J[:, j] = (F(r + dr) - F(r - dr)) / (2 * h)
    return np.array([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]])


def phys(d, check):
    p, c = d["params"], d["correct_answers"]
    m = re.fullmatch(r"(-?)([\d.]+)\\cos\(\\omega t\)\\,\\hat\{([xyz])\}", p["J_tex"])
    check(m is not None, "sheet: cannot parse J_tex")
    J0 = float(m.group(2)) * (-1 if m.group(1) == "-" else 1)
    ahat = E3[IDX[m.group(3)]]
    if p["medium"] == "vacuum":
        eps_r = 1.0
    else:
        eps_r = float(re.search(r"\\epsilon_r = ([\d.]+)", p["medium"]).group(1))
    check(abs(eps_r - p["eps_r"]) < 1e-12, "sheet: medium text vs eps_r")
    eps, mu = EPS0 * eps_r, MU0
    v = 1 / math.sqrt(mu * eps)
    eta = math.sqrt(mu / eps)
    om = 2 * math.pi * p["f_MHz"] * 1e6
    what = E3[IDX[p["w"]]]
    check(abs(np.dot(ahat, what)) == 0, "sheet: current not tangential")

    def Js(t):
        return J0 * math.cos(om * t) * ahat

    def E(r, t):
        dd = abs(np.dot(r, what))
        return -(eta / 2) * Js(t - dd / v)

    def H(r, t):
        wc = np.dot(r, what)
        nh = what if wc > 0 else -what
        return 0.5 * np.cross(Js(t - abs(wc) / v), nh)

    # ---- (a): wavelength from zero-crossing spacing, v_p from a moving zero crossing
    def Ea_line(xs, t):        # a-component of E(x w_hat, t) for x > 0, vectorized version of E()
        return -(eta / 2) * J0 * np.cos(om * (t - np.abs(xs) / v))
    check(abs(Ea_line(np.array([0.37]), 1e-9)[0] - np.dot(E(0.37 * what, 1e-9), ahat)) < 1e-9, "sheet: line helper")
    xs = np.linspace(1e-9, 40.0, 8001)
    Ew = Ea_line(xs, 0.0)
    zc = xs[:-1][np.sign(Ew[:-1]) != np.sign(Ew[1:])]
    lam_num = 2 * float(np.mean(np.diff(zc)))
    check(abs(lam_num - c["lam"]) < 2e-3 * c["lam"], f"sheet: lambda {lam_num} vs {c['lam']}")
    beta_num = 2 * math.pi / lam_num
    check(abs(beta_num - c["beta"]) < 2e-3 * c["beta"], f"sheet: beta {beta_num} vs {c['beta']}")
    dt = 0.01 / p["f_MHz"] / 1e6
    def first_zero(t):
        x2 = np.linspace(0.01, 0.6 * lam_num + 0.01, 3001)
        vals = Ea_line(x2, t)
        i = int(np.nonzero(np.sign(vals[:-1]) != np.sign(vals[1:]))[0][0])
        return x2[i] - vals[i] * (x2[i + 1] - x2[i]) / (vals[i + 1] - vals[i])
    z0 = first_zero(0.0)
    z1 = first_zero(dt)
    vp_num = (z1 - z0) / dt
    check(abs(vp_num / 1e8 - c["vp8"]) < 2e-3 * c["vp8"], f"sheet: v_p {vp_num/1e8} vs {c['vp8']}")
    check(abs(c["lam"] - v / (p["f_MHz"] * 1e6)) < 1e-9 * c["lam"], "sheet: lambda exact")

    # ---- (b), (c) at the point
    w1, t1 = float(p["w1_str"]), float(p["t1_str"]) * 1e-9
    r1 = w1 * what + np.array([0.3, -0.8, 0.45]) * (1 - what)
    E1, H1 = E(r1, t1), H(r1, t1)
    a_ax, b_ax = E3[IDX[p["a"]]], E3[IDX[p["b"]]]
    check(np.linalg.norm(E1 - np.dot(E1, a_ax) * a_ax) < 1e-12 and np.linalg.norm(H1 - np.dot(H1, b_ax) * b_ax) < 1e-12,
          "sheet: E or H has components other than the ones asked for")
    check(abs(np.dot(E1, a_ax) - c["Ea"]) < 1e-9 * max(1, abs(c["Ea"])), f"sheet: E_a {np.dot(E1, a_ax)} vs {c['Ea']}")
    check(abs(np.dot(H1, b_ax) - c["Hb"]) < 1e-9 * max(1, abs(c["Hb"])), f"sheet: H_b {np.dot(H1, b_ax)} vs {c['Hb']}")
    S1 = np.cross(E1, H1)
    check(abs(np.dot(S1, what) - c["Sw"]) < 1e-9 * max(1, abs(c["Sw"])), f"sheet: S_w {np.dot(S1, what)} vs {c['Sw']}")
    check(np.dot(S1, what) * w1 > 0, "sheet: S does not point away from the sheet")
    check(abs(np.dot(E1, a_ax)) > 0.5 * eta * abs(J0) / 2 * 0.99, "sheet: chosen point has a near-zero field")

    # Faraday and Ampere-Maxwell at the point
    hx = lam_num * 1e-6
    ht = 1e-6 / (p["f_MHz"] * 1e6)
    cE = curl_fd(lambda rr: E(rr, t1), r1, hx)
    cH = curl_fd(lambda rr: H(rr, t1), r1, hx)
    dH = (H(r1, t1 + ht) - H(r1, t1 - ht)) / (2 * ht)
    dE = (E(r1, t1 + ht) - E(r1, t1 - ht)) / (2 * ht)
    scaleE = max(np.linalg.norm(cE), mu * np.linalg.norm(dH), 1e-30)
    scaleH = max(np.linalg.norm(cH), eps * np.linalg.norm(dE), 1e-30)
    check(np.linalg.norm(cE + mu * dH) < 1e-5 * scaleE, "sheet: Faraday's law fails")
    check(np.linalg.norm(cH - eps * dE) < 1e-5 * scaleH, "sheet: Ampere-Maxwell fails")

    # boundary conditions at the sheet, at several instants
    tsamp = np.linspace(0, 1 / (p["f_MHz"] * 1e6), 7)
    bc_E = all(np.allclose(E(1e-12 * what, t), E(-1e-12 * what, t), atol=1e-9) for t in tsamp)
    bc_H = all(np.allclose(np.cross(what, H(1e-12 * what, t) - H(-1e-12 * what, t)), Js(t), atol=1e-9) for t in tsamp)
    check(bc_E and bc_H, "sheet: boundary conditions at the sheet fail")

    # ---- (d) mirror point and parity
    r2 = -w1 * what + (r1 - w1 * what)
    E2, H2 = E(r2, t1), H(r2, t1)
    S2 = np.cross(E2, H2)
    sch = {x["id"]: x for x in p["s_choices"]}
    check(sum(x["correct"] == "true" for x in p["s_choices"]) == 1 and sch["away"]["correct"] == "true", "sheet: S MC key")
    check(np.allclose(parse_unit(sch["away"]["text"]), S2 / np.linalg.norm(S2)), "sheet: S direction at the mirror point")
    check(np.allclose(parse_unit(sch["same"]["text"]), -S2 / np.linalg.norm(S2)), "sheet: 'same' distractor")
    check(abs(np.linalg.norm(S2) - np.linalg.norm(S1)) < 1e-9 * np.linalg.norm(S1), "sheet: |S| differs at the mirror point")
    par = {x["id"]: x for x in p["parity"]}
    E_even = np.allclose(E1, E2)
    H_odd = np.allclose(H1, -H2)
    check(E_even and H_odd and par["Eeven"]["correct"] == "true" and sum(x["correct"] == "true" for x in p["parity"]) == 1,
          "sheet: parity MC key")

    # ---- (e) statements (time series vectorized; helpers cross-checked against E() and H())
    def E_arr(r, ts):
        dd = abs(np.dot(r, what))
        return -(eta / 2) * J0 * np.cos(om * (ts - dd / v))[:, None] * ahat[None, :]

    def H_arr(r, ts):
        wc = np.dot(r, what)
        nh = what if wc > 0 else -what
        Jt = J0 * np.cos(om * (ts - abs(wc) / v))[:, None] * ahat[None, :]
        return 0.5 * np.cross(Jt, np.broadcast_to(nh, Jt.shape))

    check(np.allclose(E_arr(r1, np.array([t1]))[0], E1) and np.allclose(H_arr(r1, np.array([t1]))[0], H1), "sheet: array helpers")
    T = 1 / (p["f_MHz"] * 1e6)
    tt = np.linspace(0, T, 2001)
    Sline = np.cross(E_arr(r1, tt), H_arr(r1, tt)) @ what * np.sign(w1)
    interior = (Sline[1:-1] > Sline[:-2]) & (Sline[1:-1] >= Sline[2:])
    n_max = int(np.sum(interior)) + (1 if Sline[0] >= Sline[1] and Sline[0] >= Sline[-2] else 0)
    amp_E = float(np.max(np.linalg.norm(E_arr(r1, tt), axis=1)))
    amp_E_far = float(np.max(np.linalg.norm(E_arr(r1 + 2 * w1 * what, tt), axis=1)))
    rplus = abs(w1) * what
    S_plus_min = float(np.min(np.cross(E_arr(rplus, tt), H_arr(rplus, tt)) @ what))
    xw = abs(w1)
    plus_arg_ok = np.allclose(E((xw + 0.7) * what, t1 + 0.7 / v), E(xw * what, t1), atol=1e-9)    # depends on t - w/v
    minus_form = np.allclose(E((xw + 0.7) * what, t1 - 0.7 / v), E(xw * what, t1), atol=1e-9)    # would mean t + w/v
    lam_eps2 = (1 / math.sqrt(mu * 2 * eps)) / (p["f_MHz"] * 1e6)
    Jt = J0 * np.cos(om * tt)[:, None] * ahat[None, :]
    ampH = lambda er: float(np.max(np.linalg.norm(0.5 * np.cross(Jt, np.broadcast_to(what, Jt.shape)), axis=1)))
    ampE = lambda er: float(np.max(np.linalg.norm(math.sqrt(mu / (EPS0 * er)) / 2 * Jt, axis=1)))
    E0s = E_arr(1e-12 * what, tt)
    E_opp = bool(np.all(np.sum(E0s * Jt, axis=1) <= 1e-12))
    dly = re.search(r"\$([\d.]+)\\ \\mathrm\{ns\}\$", next(f["text"] for f in p["facts"] if f["id"] == "delay"))
    shown = float(dly.group(1))
    travel = abs(w1) / v * 1e9
    truth = {
        "E_cont": bc_E,
        "H_jump": bc_H,
        "E_opposes": E_opp,
        "S_double": n_max == 2,
        "short_lambda": lam_eps2 < c["lam"],
        "H_indep": abs(ampH(eps_r) - ampH(4 * eps_r)) < 1e-12 and abs(ampE(eps_r) - ampE(4 * eps_r)) > 1e-6,
        "E_flips": np.allclose(E1, -E2) and not np.allclose(E1, 0),
        "amp_full": abs(amp_E - eta * abs(J0)) < 1e-6 * amp_E,
        "spread": amp_E_far < amp_E * (1 - 1e-9),
        "S_toward": S_plus_min < -1e-12,
        "plus_arg": minus_form and not plus_arg_ok,
        "lambda_omega": abs(c["lam"] - v / om) < 1e-6 * c["lam"],
        "delay": abs(shown - travel) < 5e-3 * travel,
    }
    expect = {"E_cont": True, "H_jump": True, "E_opposes": True, "S_double": True, "short_lambda": True, "H_indep": True,
              "E_flips": False, "amp_full": False, "spread": False, "S_toward": False, "plus_arg": False, "lambda_omega": False}
    for k, val in expect.items():
        check(bool(truth[k]) == val, f"sheet: computed truth of {k} is {truth[k]}, expected {val} (self-test)")
    facts = p["facts"]
    check({f["id"] for f in facts} == set(truth) and len(facts) == len(truth), "sheet: statement ids")
    for f in facts:
        check((f["correct"] == "true") == bool(truth[f["id"]]), f"sheet: statement {f['id']} key")
        check(f["verdict"] == f["correct"], "sheet: verdict text")
