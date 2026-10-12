"""Independent check for questions/electrostatics/drude-lorentz-models.

Route: the Drude equation m dv/dt = qE - m v/tau is integrated in time (RK4) for one electron -- to steady state in a
constant field (sigma_DC, mobility), and through many periods of a cosine field, whose steady response is projected
onto cos/sin to get |sigma(w)|/sigma_DC and its phase. The Lorentz oscillator is written in state-space form
x' = A x + B E and its steady phasor response solved with numpy.linalg (no closed-form chi formula). The MC key and the
select-all truths are re-derived from these numbers.
"""
import math
import re

import numpy as np

E_CH = 1.602e-19
M_E = 9.109e-31
EPS0 = 8.854e-12


def _rk4_drude(q, m, tau, Efun, t_end, dt, v0=0.0, record_from=None):
    f = lambda t, v: q * Efun(t) / m - v / tau
    n = int(round(t_end / dt))
    t, v = 0.0, v0
    ts, vs = [], []
    for i in range(n):
        k1 = f(t, v)
        k2 = f(t + dt / 2, v + dt * k1 / 2)
        k3 = f(t + dt / 2, v + dt * k2 / 2)
        k4 = f(t + dt, v + dt * k3)
        v += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t = (i + 1) * dt
        if record_from is not None and t > record_from - 1e-3 * dt:
            ts.append(t)
            vs.append(v)
    return v, np.array(ts), np.array(vs)


def _lorentz_p(w, w0, alpha=0.0):
    """Steady phasor dipole moment per unit field amplitude for m r'' = -eE - m w0^2 r - 2 m alpha r'."""
    A = np.array([[0.0, 1.0], [-w0 ** 2, -2 * alpha]], dtype=complex)
    B = np.array([0.0, -E_CH / M_E], dtype=complex)
    X = np.linalg.solve(1j * w * np.eye(2) - A, B)      # [r, r'] phasors
    return -E_CH * X[0]                                 # p = -e r


def phys(d, check):
    p, c = d["params"], d["correct_answers"]
    N, tau = p["N28"] * 1e28, p["tau_fs"] * 1e-15
    q = -E_CH

    # (a) sigma_DC and mobility from the time-stepped steady state in E = 1 V/m
    v_ss, _, _ = _rk4_drude(q, M_E, tau, lambda t: 1.0, 40 * tau, tau / 200)
    sigma = N * q * v_ss / 1.0
    mob = abs(v_ss) / 1.0
    check(abs(sigma / 1e6 - c["sigma"]) <= 1e-6 * sigma / 1e6, f"drude: sigma {c['sigma']} vs ODE {sigma/1e6}")
    check(abs(mob * 1e4 - c["mob"]) <= 1e-6 * mob * 1e4, f"drude: mobility {c['mob']} vs ODE {mob*1e4}")
    check(abs(sigma / 1e6 - p["sig_meas"]) <= 0.03 * p["sig_meas"], f"drude: sigma {sigma/1e6} far from the handbook value {p['sig_meas']}")

    # (b) drift speed: E given -> mobility x E; J given -> field J/sigma, then mobility x field (route via sigma)
    if p["drive"] == "E":
        v = mob * p["Ef"]
    else:
        J = p["I_w"] / (p["A_w"] * 1e-6)
        v = mob * (J / sigma)
        check(1e6 <= J <= 1e7 + 1, "drude: current density outside 1-10 A/mm^2")
    check(abs(v * 1e3 - c["v"]) <= 1e-6 * v * 1e3, f"drude: drift {c['v']} vs {v*1e3} mm/s")

    # (c) AC: cosine field, integrate to steady state, project the last two periods onto cos/sin
    w = 2 * math.pi * p["f_THz"] * 1e12
    T = 2 * math.pi / w
    nper = math.ceil(max(40 * tau, 4 * T) / T)
    spp = math.ceil(T / (min(tau, T) / 100))
    dt = T / spp
    _, ts, vs = _rk4_drude(q, M_E, tau, lambda t: math.cos(w * t), nper * T, dt, record_from=(nper - 2) * T)
    ts, vs = ts[-2 * spp:], vs[-2 * spp:]
    a = 2 * np.mean(vs * np.cos(w * ts))
    b = 2 * np.mean(vs * np.sin(w * ts))
    sig_w = N * q * (a - 1j * b)                         # v(t) = Re{(a - jb) e^{jwt}} for unit field amplitude
    ratio, phase = abs(sig_w) / sigma, math.degrees(np.angle(sig_w))
    check(abs(ratio - c["ratio"]) <= 1e-5, f"drude: |sigma(w)|/sigma_DC {c['ratio']} vs ODE {ratio}")
    check(abs(phase - c["phase"]) <= 1e-3 * max(1.0, abs(phase)), f"drude: phase {c['phase']} vs ODE {phase}")
    check(phase < 0 and 0.2 <= w * tau <= 4.0, "drude: AC point should lag and lie at 0.2 <= w tau <= 4")

    # (d) Lorentz, lossless: chi(w)/chi(0) from the state-space steady response
    w0, r = p["w0"], p["r"]
    chi_w = p["chi0"] * (_lorentz_p(r * w0, w0) / _lorentz_p(0.0, w0)).real
    ch = p["chi_choices"]
    check(sum(x["correct"] == "true" for x in ch) == 1 and len(ch) == 4, "drude: (d) needs 4 choices, one correct")
    vals = []
    for x in ch:
        m = re.match(r"\$\\chi_e = (-?[0-9.]+)\$", x["text"])
        check(m is not None, f"drude: cannot parse chi choice {x['text'][:40]}")
        if m:
            vals.append((float(m.group(1)), x["correct"] == "true"))
    for val, is_key in vals:                       # values are shown to 3 significant figures
        close = abs(val - chi_w) <= 6e-3 * abs(chi_w)
        far = abs(val - chi_w) >= 0.05 * abs(chi_w)
        check(close == is_key and (close or far), f"drude: (d) choice {val} keyed {is_key}, physics chi = {chi_w}")
    check(len({round(v, 6) for v, _ in vals}) == len(vals), "drude: (d) duplicate values")
    check((chi_w > 0) == (r < 1), "drude: chi sign vs resonance")

    # (e) select-all truths from the models
    v2, _, _ = _rk4_drude(q, M_E, tau, lambda t: 2.0, 40 * tau, tau / 200)
    v_tau2, _, _ = _rk4_drude(q, M_E, 2 * tau, lambda t: 1.0, 80 * tau, tau / 200)
    v_pos, _, _ = _rk4_drude(-q, M_E, tau, lambda t: 1.0, 40 * tau, tau / 200)
    v_off, _, _ = _rk4_drude(q, M_E, tau, lambda t: 0.0, 5 * tau, tau / 200, v0=1.0)
    w1G = 2 * math.pi * 1e9
    sig_1G = N * q * (q / M_E) / (1j * w1G + 1 / tau)     # one-state phasor solve of the same ODE
    p_low = _lorentz_p(1e-3 * w0, w0)
    Jp_low = 1j * 1e-3 * w0 * p_low                       # dP/dt phasor per unit N and unit field
    truth = {
        "tau": True,
        "sigtau": abs(N * q * v_tau2 - 2 * sigma) <= 1e-6 * sigma,
        "oppdrift": v_ss < 0 < N * q * v_ss,
        "signs": abs(N * (-q) * v_pos - sigma) <= 1e-9 * sigma,
        "polcur": abs(Jp_low) > 0,
        "lead90": abs(math.degrees(np.angle(Jp_low)) - 90.0) <= 0.1,
        "rf": abs(sig_1G - sigma) / sigma < 1e-3,
        "stiff": abs(_lorentz_p(0.0, 0.5 * w0)) > abs(_lorentz_p(0.0, w0)),
        "sametau": abs(EPS0 / sigma - tau) <= 0.1 * tau,
        "Edep": abs(N * q * v2 / 2.0 - 2 * sigma) <= 1e-6 * sigma,
        "persist": abs(v_off) > 0.5,
        "pdir": (_lorentz_p(0.0, w0)).real < 0,
        "dcpol": False,                                   # d/dt of a constant P is zero
        "thermal": v >= 1e3,
    }
    ids = [x["id"] for x in p["pool"]]
    check(sorted(ids) == sorted(truth), f"drude: pool ids {ids}")
    for x in p["pool"]:
        if x["id"] in truth:
            check((x["correct"] == "true") == truth[x["id"]], f"drude: pool statement {x['id']} keyed {x['correct']}")
    ntrue = sum(x["correct"] == "true" for x in p["pool"])
    check(ntrue >= 4 and len(ids) - ntrue >= 4 and 8 <= len(ids) <= 14, "drude: pool balance/size")
