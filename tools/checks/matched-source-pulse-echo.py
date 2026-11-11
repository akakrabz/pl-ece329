"""Independent check for midterm3/matched-source-pulse-echo.

Route: exact time-domain simulation (tools/checks/_tlsim.py; circuit laws at both ends, no coefficients). A long step
run gives the steady state of part (a) and, as wave ratios, the coefficients of (b). A pulse run gives V and I at the
three times of (c), is compared with all four plots of (d), and its energy book-keeping (source, R_g, far end, line)
decides the statements of (e).
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
    q = "matched-source-pulse-echo"
    Z0, V0, T, m = float(p["Z0"]), float(p["V0"]), p["T"], p["m"]
    RL = 0 if p["short"] else None
    N = 20
    dt = T / N
    st = Cascade([Z0], [N], Z0, RL, np.full(12 * N, V0))
    check(close(st.V(0, 0, 11 * N), ca["Vdc0"]) and close(st.I(0, N, 11 * N), ca["Idc"]), f"{q}: steady state of the step")
    f, g = st.f[0], st.g[0]
    check(close(f[0] / V0, ca["tg"]) and close(g[N] / f[0], ca["GL"]) and close(f[2 * N] - f[2 * N - 1], 0.0) and ca["Gg"] == 0.0, f"{q}: coefficients")
    nst = (m + 6) * N
    src = np.where(np.arange(nst) < m * N, V0, 0.0)
    pu = Cascade([Z0], [N], Z0, RL, src)
    for nm, t in zip("abc", p["times"]):
        n = step_index(t, dt)
        check(close(pu.V(0, 0, n), ca["V" + nm]) and close(pu.I(0, 0, n), ca["I" + nm]),
              f"{q}: t_{nm}={t}: sim V={pu.V(0, 0, n)}, I={pu.I(0, 0, n)} vs {ca['V' + nm]}, {ca['I' + nm]}")
    good = 0
    for ch in p["plots"]:
        def drawn(t, segs=ch["segs"]):
            return next(v for t0, t1, v in segs if t0 <= t < t1)
        grid = [(j + 0.5) * T / 4 for j in range(4 * (m + 3))]           # dense, never on a step
        ok = close(ch["segs"][-1][1], (m + 3) * T) and all(close(pu.V(0, 0, step_index(t, dt)), drawn(t)) for t in grid)
        check((ch["correct"] == "true") == ok, f"{q}: plot {ch['key']} mis-marked")
        good += ok
    check(good == 1 and len({ch["text"] for ch in p["plots"]}) == 4, f"{q}: plot MC")
    # energy ledger of the pulse run
    I0 = np.array([pu.I(0, 0, n) for n in range(nst)])
    VL = np.array([pu.V(0, N, n) for n in range(nst)])
    IL = np.array([pu.I(0, N, n) for n in range(nst)])
    Wsrc, Wrg, Wload = np.sum(src * I0) * dt, np.sum(Z0 * I0 ** 2) * dt, np.sum(VL * IL) * dt
    quiet = all(abs(pu.V(0, k, nst - 1)) + abs(pu.I(0, k, nst - 1)) < 1e-12 for k in range(N + 1))
    truth = {"rg": close(Wsrc, Wrg) and Wsrc > 0, "load": abs(Wload) > 1e-9 * Wsrc, "stored": not quiet, "none": abs(Wsrc) < 1e-9}
    for c in p["energy"]:
        check((c["correct"] == "true") == bool(truth[c["key"]]), f"{q}: energy statement {c['key']} mis-marked")
    check(sum(c["correct"] == "true" for c in p["energy"]) == 1, f"{q}: energy MC")
