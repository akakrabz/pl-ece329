"""Independent check for conductor-shell-relaxation (Lecture 8).

* Surface charges and fields by brute force, without the shell theorem: the server's Q_inner and Q_outer are
  spread uniformly over r = a and r = b, and Coulomb's law is integrated over each sphere (Gauss-Legendre in
  cos(theta)) together with the point charge q.  The total field must vanish at several points inside the metal,
  the shell's two surface charges must add up to Q_s, and the field at r_out must equal the server's E_out.
* Relaxation: d rho/dt = -(sigma/eps) rho (continuity + Ohm + Gauss) is time-stepped with RK4; the times at
  which rho/rho0 crosses 1/e and the asked fraction are found by interpolation.
* Material constants compared with a handbook table written here; MC keys; checkbox truths.
"""
import math

import numpy as np

EPS0 = 8.854e-12
K = 1.0 / (4 * math.pi * EPS0)
_U, _W = np.polynomial.legendre.leggauss(3000)
HANDBOOK = {"sea water": (4.0, 81), "distilled water": (1e-4, 81),
            "intrinsic (undoped) silicon at room temperature": (4.4e-4, 11.7),
            "intrinsic (undoped) germanium at room temperature": (2.2, 16), "window glass": (1e-12, 6),
            r"lightly doped silicon (resistivity $10\ \Omega\,\mathrm{cm}$)": (1 / 0.1, 11.7)}
STATIC_TRUTH = {"Ezero": True, "equi": True, "normal": True, "surface": True, "inner_any": True, "tau_e": True,
                "fast": True, "neutral_inner": False, "field_q": False, "Vzero": False, "neutralized": False,
                "tangent": False, "wire": False}


def sphere_Ez(Q, R, z):
    """z-component of E at (0, 0, z) from charge Q spread uniformly over the sphere of radius R (direct Coulomb sum)."""
    u = _U                                                   # cos(theta) of the source element
    dist3 = (z * z + R * R - 2 * z * R * u) ** 1.5
    return K * (Q / 2) * float(np.sum(_W * (z - R * u) / dist3))


def total_Ez(z, q, Qin, Qout, a, b):
    return K * q / z ** 2 + sphere_Ez(Qin, a, z) + sphere_Ez(Qout, b, z)


def rk4_crossings(rate, levels, h):
    """Integrate x' = -rate x from x = 1; return the times at which x falls through each level."""
    t, x, out = 0.0, 1.0, {}
    f = lambda xx: -rate * xx
    while len(out) < len(levels):
        k1 = f(x); k2 = f(x + 0.5 * h * k1); k3 = f(x + 0.5 * h * k2); k4 = f(x + h * k3)
        xn = x + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        for lv in levels:
            if lv not in out and xn <= lv < x:
                out[lv] = t + h * (x - lv) / (x - xn)
        t, x = t + h, xn
    return out


def phys(d, check):
    p, c = d["params"], d["correct_answers"]
    a, b, q, Qs = p["a"] * 1e-2, p["b"] * 1e-2, p["q"] * 1e-9, p["Qs"] * 1e-9
    Qin, Qout = c["Q_in"] * 1e-9, c["Q_out"] * 1e-9
    check(abs(Qin + Qout - Qs) < 1e-18, "shell: Q_inner + Q_outer != Q_s (charge not conserved)")
    scale = K * abs(q) / a ** 2
    for frac in (0.2, 0.5, 0.8):
        z = a + frac * (b - a)
        Ez = total_Ez(z, q, Qin, Qout, a, b)
        check(abs(Ez) < 1e-7 * scale, f"shell: E = {Ez} V/m inside the metal at r = {z} (should vanish)")
    # the only inner-surface charge that kills the metal field is -q: a perturbed one must not
    Ez_bad = total_Ez(0.5 * (a + b), q, Qin + 0.1e-9, Qout - 0.1e-9, a, b)
    check(abs(Ez_bad) > 1e-3 * K * 0.1e-9 / b ** 2, "shell: brute-force field insensitive to the inner charge")
    z_out = p["r_out"] * 1e-2
    E_out = total_Ez(z_out, q, Qin, Qout, a, b) / 1e3
    check(abs(E_out - c["E_out"]) <= 1e-6 * max(abs(E_out), 1e-3), f"shell: E_out {E_out} kV/m vs {c['E_out']}")
    check(p["r_out"] > p["b"] and p["a"] < p["r_mid_val"] < p["b"], "shell: radii")

    # relaxation
    sig, er = p["sigma"], p["er"]
    ref = HANDBOOK.get(p["mat"])
    check(ref is not None and abs(ref[0] - sig) <= 1e-9 * sig and ref[1] == er, f"shell: material constants {p['mat']}")
    rate = sig / (er * EPS0)
    lv_e, lv_f = math.exp(-1.0), p["f"]
    cr = rk4_crossings(rate, [lv_e, lv_f], h=1.0 / rate / 400)
    tau_num, tf_num = cr[lv_e] / p["unit_s"], cr[lv_f] / p["unit_s"]
    check(abs(tau_num - c["tau"]) <= 2e-6 * c["tau"], f"shell: tau {tau_num} vs {c['tau']} {p['unit']}")
    check(abs(tf_num - c["t_f"]) <= 2e-6 * c["t_f"], f"shell: t_f {tf_num} vs {c['t_f']} {p['unit']}")
    check(0.01 <= c["tau"] < 1000 and 0.005 <= c["t_f"] < 1000, f"shell: tau {c['tau']} {p['unit']} out of range")

    # MC keys
    metal = {x["key"]: x["correct"] for x in p["metal"]}
    check(metal.get("zero") == "true" and sum(v == "true" for v in metal.values()) == 1 and len(metal) >= 4,
          f"shell: metal MC {metal}")
    check(("total" in metal) == (p["Qs"] != 0), "shell: 'total' distractor must appear only for a charged shell")
    move = {x["key"]: x["correct"] for x in p["move"]}
    check(move.get("inner_only") == "true" and sum(v == "true" for v in move.values()) == 1 and len(move) == 5,
          f"shell: move MC {move}")
    for lst in (p["metal"], p["move"]):
        check(len({x["text"] for x in lst}) == len(lst) and all(x["feedback"] for x in lst), "shell: MC texts/feedback")

    # checkbox
    for x in p["cb"]:
        tv = (Qout * q > 0) if x["key"] == "outer_sign" else STATIC_TRUTH.get(x["key"])
        check(tv is not None, f"shell: unknown statement {x['key']}")
        check((x["correct"] == "true") == bool(tv) and x["tf"] == ("True" if tv else "False"),
              f"shell: statement {x['key']} marked {x['correct']}")
    check(len(p["cb"]) == 14, "shell: pool size")
    check(not any(ch.isdigit() for x in p["cb"] if x["key"] == "outer_sign" for ch in x["feedback"]), "shell: numbers in feedback")

    # answer-panel numbers
    for key, s in (("tau", p["tau_str"]), ("t_f", p["tf_str"])):
        check(abs(float(s) - c[key]) <= 1e-3 * c[key], f"shell: panel {key} {s} vs {c[key]}")
    if c["E_out"] != 0:
        check(abs(float(p["E_out_str"]) - c["E_out"]) <= 1e-3 * abs(c["E_out"]), "shell: panel E_out")
