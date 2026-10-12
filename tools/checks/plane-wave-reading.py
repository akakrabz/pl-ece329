"""Independent check for waves/plane-wave-reading (Lecture 18).

The field is rebuilt from the *displayed* argument (the LaTeX numerator is parsed), then:
  * direction and speed: the peak is located numerically (coarse + fine grid argmax of |E|) at two times;
  * eps_r = (c/v)^2 from that speed; eta = sqrt(mu0/eps) with eps0 = 1/(mu0 c^2);
  * H is built as (1/eta) u x E with np.cross and validated by Faraday's and Ampere-Maxwell's laws
    (central finite differences in space and time), so a reversed H would fail;
  * the peak position at t1 from the fine argmax;
  * every checkbox statement is re-decided by a small computation (or, for two definitional ones, by this module's table).
"""
import math
import re

import numpy as np

C0 = 299792458.0
MU0 = 4e-7 * math.pi
EPS0 = 1.0 / (MU0 * C0 ** 2)
IDX = {"x": 0, "y": 1, "z": 2}
E3 = np.eye(3)


def g_of(shape):
    if shape == "gauss":
        return lambda s: np.exp(-np.asarray(s, float) ** 2)
    if shape == "tri":
        return lambda s: np.clip(1 - np.abs(np.asarray(s, float)), 0, None)
    return lambda s: np.where(np.asarray(s, float) >= 0, np.exp(-np.clip(np.asarray(s, float), 0, None)), 0.0)


def parse_lin(tex):
    """'-8\\,x - t + 20' -> {'x': -8, 't': -1, None: 20}."""
    out = {}
    for tok in re.findall(r"[+-]?[^+-]+", tex.replace(" ", "")):
        sgn = -1.0 if tok.startswith("-") else 1.0
        body = tok.lstrip("+-")
        if body[-1] in "txyz":
            coef = body[:-1].replace("\\,", "")
            out[body[-1]] = out.get(body[-1], 0.0) + sgn * (float(coef) if coef else 1.0)
        else:
            out[None] = out.get(None, 0.0) + sgn * float(body)
    return out


def parse_unit(tex):
    m = re.fullmatch(r"\$?([+-])\\hat\{([xyz])\}\$?", tex.strip())
    return (1 if m.group(1) == "+" else -1) * E3[IDX[m.group(2)]]


def curl_fd(F, r, h):
    J = np.zeros((3, 3))
    for j in range(3):
        dr = np.zeros(3); dr[j] = h
        J[:, j] = (F(r + dr) - F(r - dr)) / (2 * h)
    return np.array([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]])


def phys(d, check):
    p, c = d["params"], d["correct_answers"]
    u, e = p["u"], p["e"]
    lin = parse_lin(p["num_tex"])
    check(set(lin) <= {"t", u, None}, f"plane-wave: numerator has foreign symbols {lin}")
    ct, cu, c0 = lin.get("t", 0.0), lin.get(u, 0.0), lin.get(None, 0.0)
    check(abs(ct - p["ct"]) < 1e-12 and abs(cu - p["cu"]) < 1e-12 and abs(c0 - p["c0"]) < 1e-12,
          "plane-wave: displayed numerator differs from the stored coefficients")
    den = float(p["den_str"])
    check(p["s_tex"] == r"\dfrac{" + p["num_tex"] + "}{" + p["den_str"] + "}", "plane-wave: s_tex composition")
    am = re.fullmatch(r"(-?)(\d+)\\,\\hat\{([xyz])\}", p["E_amp_tex"])
    check(am is not None and am.group(3) == e, "plane-wave: amplitude tex")
    Eamp = (-1 if am.group(1) == "-" else 1) * float(am.group(2))
    g = g_of(p["shape"])

    def Es(uu, t):                     # scalar field (V/m) at position uu (m) and time t (ns)
        return Eamp * g((ct * t + cu * uu + c0) / den)

    def peak_at(t, amp=1.0):
        uu = np.arange(-400, 400, 0.01)
        i = int(np.argmax(np.abs(amp * Es(uu, t))))
        fine = np.arange(uu[i] - 0.02, uu[i] + 0.02, 1e-6)
        return float(fine[int(np.argmax(np.abs(amp * Es(fine, t))))])

    t1 = p["t1"]
    u1, u2 = peak_at(t1), peak_at(t1 + 100.0)
    vel = (u2 - u1) / 100.0                       # m/ns, signed along +u
    check(abs(abs(vel) * 10 - c["v8"]) < 2e-5 * c["v8"], f"plane-wave: speed {abs(vel)*10} vs {c['v8']}")
    v_ms = abs(vel) * 1e9
    eps_r = (C0 / v_ms) ** 2
    check(abs(eps_r - c["epsr"]) < 1e-4 * eps_r, f"plane-wave: eps_r {eps_r} vs {c['epsr']}")
    check(abs(u1 - c["upk"]) < 1e-4, f"plane-wave: peak position {u1} vs {c['upk']}")
    uhat = math.copysign(1.0, vel) * E3[IDX[u]]
    dmap = {x["id"]: x for x in p["dir_choices"]}
    good = "plus" if vel > 0 else "minus"
    check(dmap[good]["correct"] == "true" and sum(x["correct"] == "true" for x in p["dir_choices"]) == 1,
          "plane-wave: direction MC key")
    check(dmap[good]["text"] == f"toward ${'+' if vel > 0 else '-'}{u}$", "plane-wave: direction choice text")

    eps = EPS0 * eps_r
    eta = math.sqrt(MU0 / eps)
    check(abs(abs(Eamp) / eta * 1e3 - c["Hpk"]) < 1e-4 * c["Hpk"], f"plane-wave: |H| peak vs {c['Hpk']}")
    ehat = E3[IDX[e]]

    def Evec(r, t_s):                              # r in m, t in s
        return Es(float(np.dot(r, E3[IDX[u]])), t_s * 1e9) * ehat

    def Hvec(r, t_s):
        return np.cross(uhat, Evec(r, t_s)) / eta

    # Faraday and Ampere-Maxwell at a point where s = 0.3 (smooth for every shape)
    t_s = t1 * 1e-9
    u_s = (0.3 * den - c0 - ct * t1) / cu
    r = u_s * E3[IDX[u]] + np.array([0.37, -0.21, 0.55]) * (1 - E3[IDX[u]])
    hx = 1e-5 * den * (1 if p["form"] == "space" else 0.1)
    ht = 1e-6 * (1e-9 * den if p["form"] == "time" else den / v_ms)
    dHdt = (Hvec(r, t_s + ht) - Hvec(r, t_s - ht)) / (2 * ht)
    dEdt = (Evec(r, t_s + ht) - Evec(r, t_s - ht)) / (2 * ht)
    cE = curl_fd(lambda rr: Evec(rr, t_s), r, hx)
    cH = curl_fd(lambda rr: Hvec(rr, t_s), r, hx)
    check(np.linalg.norm(cE + MU0 * dHdt) < 1e-4 * np.linalg.norm(cE), "plane-wave: Faraday's law fails for H = u x E/eta")
    check(np.linalg.norm(cH - eps * dEdt) < 1e-4 * np.linalg.norm(cH), "plane-wave: Ampere-Maxwell fails")
    # direction of H at the peak
    rpk = u1 * E3[IDX[u]]
    Hpk = Hvec(rpk, t_s)
    hmap = {x["id"]: x for x in p["hdir_choices"]}
    check(sum(x["correct"] == "true" for x in p["hdir_choices"]) == 1 and hmap["ok"]["correct"] == "true", "plane-wave: H MC key")
    check(np.allclose(parse_unit(hmap["ok"]["text"]), Hpk / np.linalg.norm(Hpk)), "plane-wave: H direction choice disagrees with Faraday-checked H")
    check(np.allclose(parse_unit(hmap["rev"]["text"]), -Hpk / np.linalg.norm(Hpk)), "plane-wave: reversed-H distractor")

    # ---- statements
    zz = np.linspace(-3, 3, 7)
    a_pt, b_pt = [float(x) for x in re.findall(r"= (-?\d+)\\ \\mathrm\{m\}", next(f["text"] for f in p["facts"] if f["id"] == "order"))]
    t_a, t_b = t1 + (a_pt - u1) / vel, t1 + (b_pt - u1) / vel
    E_pts = [Evec(z * E3[IDX[u]] + 0.0, t_s) for z in (u1 - 0.1 * den, u1, u1 + 0.05)]
    H_pts = [np.cross(uhat, E) / eta for E in E_pts]
    perp = all(abs(np.dot(E, H)) < 1e-12 and abs(np.dot(E, uhat)) < 1e-12 and abs(np.dot(H, uhat)) < 1e-12 for E, H in zip(E_pts, H_pts))
    ratio_ok = all(np.linalg.norm(H) == 0 or abs(np.linalg.norm(E) / np.linalg.norm(H) - 376.730313 / math.sqrt(eps_r)) < 1e-3
                   for E, H in zip(E_pts, H_pts))
    # same_kind: h(z - v t) with a test h, its peak moves toward +z
    vt = 0.2
    hz = lambda z, t: np.exp(-(z - vt * t) ** 2)
    zg = np.linspace(-20, 40, 600001)
    same_kind = zg[np.argmax(hz(zg, 50.0))] > zg[np.argmax(hz(zg, 0.0))]
    # superpose: wave-equation residual of f(t - z/v) + g(t + z/v) by finite differences
    vv = 0.15
    F = lambda z, t: np.exp(-(t - z / vv) ** 2) + 0.5 * np.tanh(t + z / vv)
    hh = 1e-3
    z0, tt0 = 0.3, 0.7
    Fzz = (F(z0 + hh, tt0) - 2 * F(z0, tt0) + F(z0 - hh, tt0)) / hh ** 2
    Ftt = (F(z0, tt0 + hh) - 2 * F(z0, tt0) + F(z0, tt0 - hh)) / hh ** 2
    superpose = abs(Fzz - Ftt / vv ** 2) < 1e-3 * abs(Fzz)
    # shape: rigid translation
    shape_kept = all(abs(Es(x + vel * 37.0, t1 + 37.0) - Es(x, t1)) < 1e-4 * abs(Eamp) for x in (u1 - 0.3 * den, u1 + 0.2 * den))
    B_over_E = abs(MU0 * np.linalg.norm(H_pts[1]) - np.linalg.norm(E_pts[1]) / v_ms) < 1e-9 * np.linalg.norm(E_pts[1]) / v_ms
    vacuum_H_smaller = abs(Eamp) / math.sqrt(MU0 / EPS0) < abs(Eamp) / eta
    # quadrature: at a fixed point E and H peak at the same instant
    tg = t1 + np.linspace(-30, 30, 600001)
    Eu = Es(u1, tg)
    Hu = Eu / eta                                                          # |H| = |E|/eta, built as u x E / eta
    quadrature = abs(tg[np.argmax(np.abs(Hu))] - tg[np.argmax(np.abs(Eu))]) > 1e-6
    # longitudinal: div of z f(t - z/v) is not zero
    fz = lambda z: np.exp(-(1.0 - z / 0.2) ** 2)
    div_long = (fz(0.1 + 1e-6) - fz(0.1 - 1e-6)) / 2e-6          # off the peak
    longitudinal_ok = abs(div_long) < 1e-9
    eta0_claim = abs(eta - 376.73) < 0.5
    slowness_claim = abs(1 / v_ms - v_ms) < 1e-9 * v_ms
    eta_up = math.sqrt(MU0 / (EPS0 * 2 * eps_r)) > eta
    decay = abs(Es(peak_at(t1 + 300.0), t1 + 300.0)) < abs(Es(u1, t1)) * (1 - 1e-4)
    amp_speed = abs((peak_at(t1 + 100.0, 2.0) - peak_at(t1, 2.0)) / 100.0 - vel) > 1e-6
    # standing ratio: E_x/H_y for f(t - z/v) + g(t + z/v)
    vv, etaa = 0.2e9, 200.0
    fA = lambda t, z: math.exp(-(t - z / vv) ** 2 / 1e-18)
    fB = lambda t, z: 0.6 * math.exp(-(t + z / vv - 1e-9) ** 2 / 1e-18)
    tz = (0.4e-9, 0.03)
    Ex, Hy = fA(*tz) + fB(*tz), (fA(*tz) - fB(*tz)) / etaa
    standing = abs(Ex / Hy - etaa) < 1e-6 * etaa
    truth = {
        "perp": perp, "ratio": ratio_ok, "same_kind": bool(same_kind), "superpose": superpose, "shape": shape_kept,
        "B_over_E": B_over_E, "vacuum_H": vacuum_H_smaller, "quadrature": quadrature, "longitudinal": longitudinal_ok,
        "eta0": eta0_claim, "slowness": slowness_claim, "eta_up": eta_up, "decay": decay, "amp_speed": amp_speed,
        "standing_ratio": standing, "order": t_a < t_b,
    }
    expect = {"perp": True, "ratio": True, "same_kind": True, "superpose": True, "shape": True, "B_over_E": True,
              "vacuum_H": True, "quadrature": False, "longitudinal": False, "eta0": False, "slowness": False,
              "eta_up": False, "decay": False, "amp_speed": False, "standing_ratio": False}
    for k, v in expect.items():
        check(truth[k] == v, f"plane-wave: computed truth of {k} is {truth[k]}, expected {v} (check module self-test)")
    facts = p["facts"]
    check({f["id"] for f in facts} == set(truth) and len(facts) == len(truth), "plane-wave: statement ids")
    for f in facts:
        check((f["correct"] == "true") == bool(truth[f["id"]]), f"plane-wave: statement {f['id']} key")
        check(f["verdict"] == f["correct"], "plane-wave: verdict text")
