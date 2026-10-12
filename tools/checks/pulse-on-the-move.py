"""Independent check for midterm2/pulse-on-the-move.

* The snapshot is rebuilt from the corner list with np.interp; the field is the d'Alembert form for the STATED direction
  of travel, E(z,t) = e F(z - s v (t - t0)), and H = (1/eta) (s z-hat) x E is built with np.cross (eta = mu0 v).
* Faraday (curl E = -mu0 dH/dt) and Ampere-Maxwell (curl H = eps dE/dt, eps = eps_r eps0 with eps_r = (c/v)^2) are
  verified by centred finite differences of the full 3-D curl at points on both ramps: a wrong direction of travel, a
  wrong eta or a wrong sign of H fails here.
* Events are re-evaluated; for the third, the field at that place is checked to be zero for all earlier times and to
  become nonzero later (the pulse has not arrived, rather than already passed).
* MC (b): every displayed expression is parsed and evaluated at random (z, t); exactly the keyed one must agree.
"""
import math
import re

import numpy as np

MU0 = 4e-7 * math.pi
EPS0 = 8.854e-12
C = 1 / math.sqrt(MU0 * EPS0)
AX = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}


def phys(d, check):
    p, c = d["params"], d["correct_answers"]
    s, v, E0, t0 = p["s_dir"], p["v_val"], p["E0_val"], p["t0_val"]
    zA, zB, zC, zD = p["corners"]
    e_hat = AX[p["pol"]]
    k_hat = s * AX["z"]
    v_si = v * 1e9
    eps_r = (C / v_si) ** 2
    eta = MU0 * v_si
    xs = np.array([zA - 1.0, zA, zB, zC, zD, zD + 1.0])
    ys = np.array([0.0, 0.0, E0, E0, 0.0, 0.0])

    def Fz(z):
        z = np.asarray(z, float)
        return np.where((z <= zA) | (z >= zD), 0.0, np.interp(z, xs, ys))

    def E(r, t_ns):                                   # r = (x, y, z) in m, t in ns
        return float(Fz(r[2] - s * v * (t_ns - t0))) * e_hat

    def H(r, t_ns):
        return np.cross(k_hat, E(r, t_ns)) / eta

    # statement data and snapshot
    check(abs(float(p["v_si"]) - v_si) <= 1e-9 * v_si, "pulse: v in m/s")
    check(p["dir_word"] == ("+z" if s > 0 else "-z"), "pulse: direction word")
    check(len({round(zB - zA, 6), round(zD - zC, 6)}) == 2, "pulse: ramps must differ (asymmetric pulse)")
    for z in np.linspace(zA - 0.5, zD + 0.5, 41):
        check(abs(E(np.array([0, 0, z]), t0) @ e_hat - Fz(z)) < 1e-12, "pulse: field at t0 is not the snapshot")

    # Maxwell by finite differences (SI), at points on both ramps
    dz = 1e-5
    dt_ns = dz / v * 1e0                              # ns  (v in m/ns)
    dt = dt_ns * 1e-9
    for frac_pt in (zA + 0.5 * (zB - zA), zC + 0.5 * (zD - zC)):
        t_test = t0 + 3.0
        r = np.array([0.2, -0.1, frac_pt + s * v * (t_test - t0)])

        def curl(field, r, t_ns):
            cu = np.zeros(3)
            dF = {}
            for i in range(3):
                e = np.zeros(3); e[i] = dz
                dF[i] = (field(r + e, t_ns) - field(r - e, t_ns)) / (2 * dz)
            cu[0] = dF[1][2] - dF[2][1]
            cu[1] = dF[2][0] - dF[0][2]
            cu[2] = dF[0][1] - dF[1][0]
            return cu

        dHdt = (H(r, t_test + dt_ns) - H(r, t_test - dt_ns)) / (2 * dt)
        dEdt = (E(r, t_test + dt_ns) - E(r, t_test - dt_ns)) / (2 * dt)
        cE, cH = curl(E, r, t_test), curl(H, r, t_test)
        scale = abs(E0) / max(zB - zA, zD - zC) + 1e-30
        check(np.allclose(cE, -MU0 * dHdt, rtol=1e-5, atol=1e-6 * scale), f"pulse: Faraday fails {cE} vs {-MU0*dHdt}")
        check(np.allclose(cH, eps_r * EPS0 * dEdt, rtol=1e-5, atol=1e-6 * scale / eta),
              f"pulse: Ampere-Maxwell fails {cH} vs {eps_r*EPS0*dEdt}")
        S = np.cross(E(r, t_test), H(r, t_test))
        check(S @ k_hat > 0, "pulse: E x H does not point along the travel")

    # events
    ev = p["events"]
    vals = [E(np.array([0, 0, z]), t) @ e_hat for z, t in ev]
    for k, key in enumerate(("E1", "E2", "E3")):
        check(abs(vals[k] - c[key]) <= 1e-9 * max(1, abs(vals[k])), f"pulse: {key} {vals[k]} vs {c[key]}")
        zs, ts = float(p[f"z{k + 1}"]), float(p[f"t{k + 1}"])
        check(abs(zs - ev[k][0]) < 1e-9 and abs(ts - ev[k][1]) < 1e-9, f"pulse: displayed event {k + 1} differs from the data")
    check(abs(vals[0]) > 0.2 * abs(E0) and abs(vals[1]) > 0.2 * abs(E0) and vals[2] == 0, "pulse: event values")
    z3, t3 = ev[2]
    tt = np.linspace(t3 - 200, t3, 40001)
    check(np.all(Fz(z3 - s * v * (tt - t0)) == 0), "pulse: event 3 field nonzero earlier: already passed?")
    tl = np.linspace(t3, t3 + 200, 40001)
    check(np.any(Fz(z3 - s * v * (tl - t0)) != 0), "pulse: the pulse never reaches event 3")

    # eps_r and H at event 1
    check(abs(eps_r - c["epsr"]) <= 1e-9 * eps_r, f"pulse: eps_r {eps_r} vs {c['epsr']}")
    r1 = np.array([0, 0, ev[0][0]])
    H1 = H(r1, ev[0][1])
    check(abs(np.linalg.norm(H1) * 1e3 - c["H1"]) <= 1e-9 * c["H1"], "pulse: |H| at event 1")
    hdir = H1 / np.linalg.norm(H1)
    ch = p["hdir_choices"]
    check(sum(x["correct"] == "true" for x in ch) == 1, "pulse: H direction MC")
    for x in ch:
        m = re.match(r"\$([+-])\\hat\{([xyz])\}\$(.*)", x["text"])
        match = bool(m) and m.group(3) == "" and float(((1 if m.group(1) == "+" else -1) * AX[m.group(2)]) @ hdir) > 0.999
        check(match == (x["correct"] == "true"), f"pulse: H direction choice {x['text']}")

    # MC (b): parse and evaluate every expression
    rng = np.random.default_rng(12345)
    zz = rng.uniform(zA - 2, zD + 2, 2000)
    tq = rng.uniform(0, 3 * t0 + 10, 2000)
    truth = Fz(zz - s * v * (tq - t0))
    check(np.count_nonzero(truth) > 50, "pulse: too few sample points inside the pulse")
    hits = []
    for x in p["expr_choices"]:
        m = re.search(r"F\\big\((.*)\\big\)\$$", x["text"])
        check(m is not None, f"pulse: cannot parse {x['text']}")
        if not m:
            continue
        arg = m.group(1).replace("\\,", "*").replace(" ", "")
        check(re.fullmatch(r"[zt0-9.+\-*/()]+", arg) is not None, f"pulse: odd expression {arg}")
        vals_x = Fz(eval(arg, {"__builtins__": {}}, {"z": zz, "t": tq}))
        match = np.allclose(vals_x, truth, atol=1e-12)
        hits.append(match)
        check(match == (x["correct"] == "true"), f"pulse: expression choice {x['text']} (matches: {match})")
        check(("\\hat{" + p["pol"] + "}") in x["text"], "pulse: expression choice has the wrong polarization")
    check(sum(hits) == 1, "pulse: exactly one expression must reproduce the field")
