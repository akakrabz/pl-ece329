"""Independent check for questions/electrostatics/polarization-bound-charge.

Route: P(x) is rebuilt from the stated profile; rho_b by a central finite difference of P; face charges by
P . n_out with explicit normal vectors; the field by superposing the infinite charge sheets the slab is made of
(sheet field sigma/(2 eps0) pointing away from each sheet) -- this never uses D = 0 or E = -P/eps0.
The select-all pool is re-keyed from these numbers; the MC key from the integrated total charge.
"""
import math
import re

import numpy as np

EPS0 = 8.854e-12


def _profile(p):
    fam, P0, d, a = p["family"], p["P0"], p["d"], p["alpha"]
    if fam == "lin":
        return lambda x: P0 * (1 + a * np.asarray(x, float) / d)
    if fam == "quad":
        return lambda x: P0 * (np.asarray(x, float) / d) ** 2
    if fam == "cap":
        return lambda x: P0 * (1 - (np.asarray(x, float) / d) ** 2)
    raise ValueError(fam)


def phys(d, check):
    p, c = d["params"], d["correct_answers"]
    P = _profile(p)
    dd, x1 = float(p["d"]), float(p["x1"])
    check(0 < x1 < dd, "polarization: x1 not inside the slab")

    # (a) rho_b = -dP/dx by central difference; (uC/m^2)/mm = mC/m^3
    h = 1e-4 * dd
    rho_fd = -(P(x1 + h) - P(x1 - h)) / (2 * h)
    check(abs(rho_fd - c["rho_b"]) <= 1e-6 * max(1.0, abs(rho_fd)), f"polarization: rho_b {c['rho_b']} vs FD {rho_fd}")

    # (b) rho_sb = P . n_out
    Pvec = lambda x: np.array([float(P(x)), 0.0, 0.0])
    s0 = float(np.dot(Pvec(0.0), np.array([-1.0, 0.0, 0.0])))
    sd = float(np.dot(Pvec(dd), np.array([1.0, 0.0, 0.0])))
    check(abs(s0 - c["rho_s0"]) <= 1e-9 * max(1.0, abs(s0)), f"polarization: face x=0 {c['rho_s0']} vs {s0}")
    check(abs(sd - c["rho_sd"]) <= 1e-9 * max(1.0, abs(sd)), f"polarization: face x=d {c['rho_sd']} vs {sd}")

    # build the slab as sheets: N interior sheets (midpoints) + the two face sheets; charge per area in uC/m^2
    N = 200000
    edges = np.linspace(0.0, dd, N + 1)
    xm = 0.5 * (edges[1:] + edges[:-1])
    dx = dd / N
    rho_cells = -(P(edges[1:]) - P(edges[:-1])) / dx        # exact cell average of -dP/dx  (mC/m^3)
    pos = np.concatenate([xm, [0.0, dd]])
    sig = np.concatenate([rho_cells * dx, [s0, sd]])        # mC/m^3 * mm = uC/m^2
    total = float(np.sum(sig))
    scale = float(np.sum(np.abs(sig)))

    def Ex_at(x):                                            # kV/m
        return float(np.sum(sig * 1e-6 / (2 * EPS0) * np.sign(x - pos))) / 1e3

    # (c) total bound charge (it is the reason the field vanishes outside)
    check(abs(total) <= 1e-9 * max(1.0, scale), f"polarization: total bound charge {total} not zero")
    ch = p["tot_choices"]
    check(sum(x["correct"] == "true" for x in ch) == 1, "polarization: (c) needs exactly one correct")
    for x in ch:
        says_zero = x["text"].startswith("Zero")
        check((x["correct"] == "true") == says_zero, "polarization: (c) key is not the 'zero' choice")
        m = re.match(r"\$(-?[0-9.]+)", x["text"])
        if m:
            v = float(m.group(1))
            check(abs(v) > 1e-9 and abs(abs(v) - abs(sd + s0)) <= 1e-3 * abs(sd + s0),
                  f"polarization: (c) distractor value {v} is not +-(face charge) {sd + s0}")
        # the verbal distractors ("faces only" / "volume only") are false only because those totals are nonzero;
        # and no choice may print a charge value (it would give away (a) or (b))
        check(r"\mu\mathrm{C" not in x["text"], "polarization: (c) choice text shows a charge value")
    check(abs(sd + s0) > 1e-9 * max(1.0, scale), "polarization: face charges cancel, so a (c) distractor would be right")

    # (d) E at x1 by sheet superposition; also E = 0 outside and D = eps0 E + P = 0 inside
    E1 = Ex_at(x1)
    check(abs(E1 - c["Ex"]) <= 2e-6 * max(1.0, abs(E1)), f"polarization: Ex {c['Ex']} vs sheets {E1}")
    Eout = max(abs(Ex_at(-1.0)), abs(Ex_at(dd + 1.0)))
    check(Eout <= 1e-6 * max(1.0, abs(E1)), f"polarization: field outside not zero ({Eout})")
    for xi in (0.3 * dd, 0.7 * dd):                        # cell edges, so no sheet sits on the field point
        D = EPS0 * Ex_at(xi) * 1e3 + float(P(xi)) * 1e-6
        check(abs(D) <= 1e-6 * abs(float(P(xi))) * 1e-6 + 1e-18, f"polarization: D not zero inside ({D})")

    # (e) pool truths from the numbers above
    xs = np.linspace(0.05 * dd, 0.95 * dd, 7)
    rho_samples = -(P(xs + h) - P(xs - h)) / (2 * h)
    Ein = [Ex_at(x) for x in xs]
    jump = EPS0 * (Ex_at(dd + 1e-7) - Ex_at(dd - 1e-7)) * 1e3 / 1e-6   # uC/m^2
    truth = {
        "divD": True, "gaussE": True, "uniformP": True, "Ddef": True,
        "increasing": True,                          # rho_b = -dP/dx: sign rule, true by definition
        "pillbox": abs(jump - sd) <= 1e-3 * max(1.0, abs(sd)),
        "Ezero": max(abs(e) for e in Ein) == 0.0,
        "linear": False,                             # D = 0 (checked) with E != 0 rules out D = eps E, eps > 0
        "Pzero": max(abs(float(P(x))) for x in xs) == 0.0,
        "Econst": max(Ein) - min(Ein) <= 1e-9 * max(1.0, max(abs(e) for e in Ein)),
        "outside": Eout > 1e-6 * max(1.0, abs(E1)),
        "facescancel": abs(s0 + sd) <= 1e-9,
        "uniformrho": float(np.ptp(rho_samples)) <= 1e-6 * max(1.0, float(np.max(np.abs(rho_samples)))),
        "face0": abs(s0) <= 1e-12,
    }
    check(max(abs(e) for e in Ein) > 0, "polarization: E vanishes inside (statement 'linear' would be ambiguous)")
    ids = [x["id"] for x in p["pool"]]
    check(sorted(ids) == sorted(truth), f"polarization: pool ids {ids}")
    check(8 <= len(ids) <= 14, "polarization: pool size outside 8-14")
    for x in p["pool"]:
        if x["id"] in truth:
            check((x["correct"] == "true") == truth[x["id"]], f"polarization: pool statement {x['id']} keyed {x['correct']}")
    ntrue = sum(x["correct"] == "true" for x in p["pool"])
    check(ntrue >= 4 and len(ids) - ntrue >= 4, "polarization: pool needs >= 4 true and >= 4 false statements")
    check(math.isfinite(c["Ex"]), "polarization: Ex not finite")
