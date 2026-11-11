"""Independent check for midterm3/three-line-sections.

Route: exact time-domain simulation of the three sections with a one-step source pulse (tools/checks/_tlsim.py: KVL
and KCL at the generator, both junctions and the load; no coefficients). Every answer is read off the simulated
waves. That the echo timed in (c) really comes from junction 2 is confirmed by a second run in which section 3 is
given the impedance of section 2 (and a matched load), which removes junction 2: the arrival must then vanish.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _tlsim import Cascade  # noqa: E402

DT = 0.05          # microseconds per step


def close(a, b, rel=1e-6, ab=1e-9):
    return abs(a - b) <= rel * max(abs(a), abs(b)) + ab


def phys(d, check):
    p, ca = d["params"], d["correct_answers"]
    q = "three-line-sections"
    Z = [float(z) for z in p["Z"]]
    N = [int(round(t / DT)) for t in p["T"]]
    check(all(close(n * DT, t) for n, t in zip(N, p["T"])), f"{q}: delays not on the time grid")
    V0, Rg, RL = float(p["V0"]), float(p["Rg"]), float(p["RL"])
    nst = 2 * sum(N) + 4 * N[0] + 40
    src = np.zeros(nst); src[0] = V0
    c = Cascade(Z, N, Rg, RL, src)
    f, g = c.f, c.g
    N1, N2, N3 = N
    V1 = f[0][0]
    check(close(V1, ca["V1"]) and close(V1 / V0, ca["tg"]), f"{q}: launch")
    check(close(g[0][N1] / V1, ca["G12"]) and close(f[1][N1] / V1, ca["t12"]), f"{q}: junction 1 coefficients")
    check(close(-g[1][N1 + N2] / f[1][N1], ca["GI23"]), f"{q}: current reflection at junction 2")
    n_e = 2 * (N1 + N2)
    arriving = g[0][n_e - N1]                              # backward wave on section 1 reaching z = 0 at step n_e
    check(close(n_e * DT, ca["techo"]) and close(arriving, ca["echo"]), f"{q}: junction-2 echo {arriving} vs {ca['echo']}")
    c2 = Cascade([Z[0], Z[1], Z[1]], N, Rg, Z[1], src)     # junction 2 removed
    check(abs(c2.g[0][n_e - N1]) < 1e-12, f"{q}: something other than the junction-2 echo arrives at t = {n_e * DT}")
    check(all(abs(c.g[0][n - N1] - c2.g[0][n - N1]) < 1e-12 for n in range(N1, n_e)), f"{q}: a junction-2 echo arrives before 2(T1+T2)")
    nL = N1 + N2 + N3
    load = [c.V(2, N3, n) for n in range(nL + 1)]
    check(all(abs(x) < 1e-12 for x in load[:-1]) and close(load[-1], ca["Vload"]) and close(nL * DT, ca["tload"]), f"{q}: load arrival {load[-1]} vs {ca['Vload']}")
    check(close(c.V(0, 0, 2 * N1), ca["Vgen"]) and all(abs(c.V(0, 0, n)) < 1e-12 for n in range(1, 2 * N1)), f"{q}: generator end at 2 T1")
    ratio = (f[1][N1] / Z[1]) / (V1 / Z[0])
    for ch in p["cur"]:
        check((ch["correct"] == "true") == close(ch["val"], ratio, 1e-6), f"{q}: current-transmission choice {ch['text']} mis-marked (true {ratio})")
    check(sum(ch["correct"] == "true" for ch in p["cur"]) == 1 and len({ch["text"] for ch in p["cur"]}) == 4, f"{q}: MC")
