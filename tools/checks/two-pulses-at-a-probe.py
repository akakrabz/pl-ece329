"""Independent check for midterm3/two-pulses-at-a-probe.

Route: the answers (v, R_g, R_L) are put into an exact time-domain simulation of the circuit (tools/checks/_tlsim.py;
no coefficients) with a one-step source pulse. The simulated probe record must be exactly what the statement
describes: one pulse at t1, one of rho times its height at t2, nothing else for a long time. Then A, the load peak
and, with the changed generator, the first and third pulses are read off further runs. The statements of the
multiple-choice part are judged from the simulated records.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tlsim import Cascade  # noqa: E402


def close(a, b, rel=1e-6, ab=1e-9):
    return abs(a - b) <= rel * max(abs(a), abs(b)) + ab


def record(Z0, Rg, RL, V0, N, k, nst):
    src = np.zeros(nst); src[0] = V0
    c = Cascade([Z0], [N], Rg, RL, src)
    return [(n, c.V(0, k, n)) for n in range(nst) if abs(c.V(0, k, n)) > 1e-12], c


def phys(d, check):
    p, ca = d["params"], d["correct_answers"]
    q = "two-pulses-at-a-probe"
    Z0, V0, ell = float(p["Z0"]), float(p["V0"]), float(p["ell"])
    N = 40
    k = N * p["frac"][0] // p["frac"][1]
    dt = (ell / ca["v"]) / N                     # microseconds per step, from the answered speed
    rec, c = record(Z0, ca["Rg"], ca["RL"], V0, N, k, 12 * N)
    check(len(rec) == 2, f"{q}: probe sees {len(rec)} pulses with the answered R_g, R_L")
    if len(rec) == 2:
        (n1, a1), (n2, a2) = rec
        check(close(n1 * dt, p["t1v"]) and close(n2 * dt, p["t2v"]), f"{q}: pulse times {n1 * dt}, {n2 * dt}")
        check(close(a1, ca["A"]) and close(a2 / a1, p["rho"]) and close(a2 / a1, ca["GL"]), f"{q}: pulse heights {a1}, {a2}")
    check(close((300.0 / ca["v"]) ** 2, ca["epsr"]), f"{q}: eps_r")
    check(close(c.V(0, N, N), ca["Vload"]), f"{q}: load peak {c.V(0, N, N)} vs {ca['Vload']}")
    rec2, _ = record(Z0, float(p["Rg2"]), ca["RL"], V0, N, k, 12 * N)
    check(len(rec2) >= 3 and close(rec2[0][1], ca["A2"]) and close(rec2[2][0] * dt, ca["t3"]) and close(rec2[2][1], ca["C"]),
          f"{q}: changed generator: {rec2[:3]} vs A'={ca['A2']}, t3={ca['t3']}, C={ca['C']}")
    # statements of (c): judged from simulations
    rec_load_matched, _ = record(Z0, ca["Rg"], Z0, V0, N, k, 12 * N)
    truth = {"matched": len(rec) == 2 and len(rec2) > 2,          # silence holds with R_g = Z0 and fails otherwise
             "loadabs": len(rec_load_matched) == 2,               # a matched load would still give two pulses? (no: one)
             "cancel": False, "lossy": False}
    for ch in p["why"]:
        check((ch["correct"] == "true") == truth[ch["key"]], f"{q}: statement {ch['key']} mis-marked")
    check(sum(ch["correct"] == "true" for ch in p["why"]) == 1, f"{q}: MC")
