"""Independent check for midterm3/bounce-diagram-both-ends.

Route: exact time-domain simulation of the circuit (tools/checks/_tlsim.py: characteristics plus the circuit law at
each end, no coefficients). The coefficients of part (a) are read off the simulation as wave ratios; the wave
amplitudes, currents, end-voltage levels and all four plots are compared with simulated values; the final values are
also compared with the DC circuit.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tlsim import Cascade, step_index  # noqa: E402


def close(a, b, rel=1e-6, ab=1e-7):
    return abs(a - b) <= rel * max(abs(a), abs(b)) + ab


def phys(d, check):
    p, ca = d["params"], d["correct_answers"]
    q = "bounce-diagram-both-ends"
    Z0, Rg, RL, V0, T = float(p["Z0"]), float(p["Rg"]), float(p["RL"]), float(p["V0"]), p["T"]
    N = 20
    dt = T / N
    c = Cascade([Z0], [N], Rg, RL, np.full(80 * N, V0))
    f, g = c.f[0], c.g[0]
    V1p, V1m = f[0], g[N]
    V2p = f[2 * N] - f[2 * N - 1]
    check(close(V1p / V0, ca["tg"]) and close(V1m / V1p, ca["GL"]) and close(V2p / V1m, ca["Gg"]), f"{q}: coefficients")
    check(close(V1m, ca["V1m"]) and close(V2p, ca["V2p"]), f"{q}: wave voltages")
    mid = N // 2
    I_a = c.I(0, mid, step_index(0.75 * T, dt))             # only the first wave has passed
    I_b = c.I(0, mid, step_index(1.75 * T, dt))             # first wave and its reflection
    check(close(I_a * 1e3, ca["I1p"]) and close((I_b - I_a) * 1e3, ca["I1m"]), f"{q}: wave currents {I_a}, {I_b - I_a}")

    def V(k, t):
        return c.V(0, k, step_index(t, dt))

    check(close(V(0, 1.0 * T), ca["Vs1"]) and close(V(0, 3.0 * T), ca["Vs2"]), f"{q}: source-end levels")
    check(close(V(N, 2.0 * T), ca["VL1"]) and close(V(N, 4.0 * T), ca["VL2"]) and V(N, 0.5 * T) == 0.0, f"{q}: load-end levels")
    good = 0
    for ch in p["plots"]:
        def drawn(t, segs=ch["segs"]):
            return next(v for t0, t1, v in segs if t0 <= t < t1)
        grid = [(j + 0.5) * T / 4 for j in range(20)]                    # dense on 0..5T, never on a step
        ok = close(ch["segs"][-1][1], 5 * T) and all(close(V(N, t), drawn(t)) for t in grid)
        check((ch["correct"] == "true") == ok, f"{q}: plot {ch['key']} mis-marked")
        good += ok
    check(good == 1 and len({ch["text"] for ch in p["plots"]}) == 4, f"{q}: plot MC needs one true plot of four distinct")
    n_late = 78 * N
    check(close(c.V(0, 3, n_late), ca["Vinf"], 1e-6, 1e-6) and close(c.I(0, 3, n_late) * 1e3, ca["Iinf"], 1e-6, 1e-5), f"{q}: steady state (simulated)")
    check(close(V0 * RL / (Rg + RL), ca["Vinf"]) and close(1e3 * V0 / (Rg + RL), ca["Iinf"]), f"{q}: steady state (DC circuit)")
