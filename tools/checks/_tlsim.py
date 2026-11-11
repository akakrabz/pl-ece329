"""Exact time-domain simulation of lossless lines in cascade (helper for the Midterm 3 transmission-line checks).

Not a question check: test_questions.py only loads tools/checks/<question>.py, and the checks import this file.

Method of characteristics. Section i has impedance Z[i] and a one-way delay of N[i] time steps. f[i][n] is the forward
wave leaving its left end at step n, g[i][n] the backward wave leaving its right end at step n; a wave launched at
one end arrives at the other end N[i] steps later, unchanged. At every step the terminal voltages and currents are
solved from the circuit laws at that node, written with V = f + g and Z I = f - g on each side:
    generator:  V + Rg I = Vs[n]                    (KVL through the source resistance)
    junction:   same V on both sides, same I through (KVL, KCL)
    load:       V = RL I,  or I = 0 (open, RL = None),  or V = 0 (short, RL = 0)
No reflection, transmission or injection coefficient is used anywhere, so the checks that call this are independent
of the bounce-diagram formulas in the questions' server.py.
"""
import numpy as np


class Cascade:
    def __init__(self, Z, N, Rg, RL, Vs):
        self.Z, self.N, self.M = list(Z), list(N), len(Z)
        Vs = np.asarray(Vs, dtype=float)
        nst = len(Vs)
        self.f = [np.zeros(nst) for _ in Z]
        self.g = [np.zeros(nst) for _ in Z]
        f, g, M = self.f, self.g, self.M

        def past(arr, n, lag):
            return arr[n - lag] if n >= lag else 0.0

        Ag = np.linalg.inv(np.array([[1.0, Rg], [1.0, -Z[0]]]))
        if RL is None:
            Al = np.linalg.inv(np.array([[1.0, Z[-1]], [0.0, 1.0]]))
        elif RL == 0:
            Al = np.linalg.inv(np.array([[1.0, Z[-1]], [1.0, 0.0]]))
        else:
            Al = np.linalg.inv(np.array([[1.0, Z[-1]], [1.0, -RL]]))
        for n in range(nst):
            # generator end: V + Rg I = Vs,  V - Z I = 2 g_in
            V, I = Ag @ np.array([Vs[n], 2 * past(g[0], n, N[0])])
            f[0][n] = (V + Z[0] * I) / 2
            # junctions: unknowns (g_i, f_{i+1});  a + g_i = f_{i+1} + b,  (a - g_i)/Z_i = (f_{i+1} - b)/Z_{i+1}
            for i in range(M - 1):
                a, b = past(f[i], n, N[i]), past(g[i + 1], n, N[i + 1])
                A = np.array([[1.0, -1.0], [-1.0 / Z[i], -1.0 / Z[i + 1]]])
                rhs = np.array([b - a, -a / Z[i] - b / Z[i + 1]])
                g[i][n], f[i + 1][n] = np.linalg.solve(A, rhs)
            # load end: V + Z I = 2 f_in, plus the load law
            V, I = Al @ np.array([2 * past(f[-1], n, N[-1]), 0.0])
            g[-1][n] = (V - Z[-1] * I) / 2

    def F(self, i, k, n):
        m = n - k
        return self.f[i][m] if m >= 0 else 0.0

    def G(self, i, k, n):
        m = n - (self.N[i] - k)
        return self.g[i][m] if m >= 0 else 0.0

    def V(self, i, k, n):
        """Voltage on section i, k cells from its left end, at step n."""
        return self.F(i, k, n) + self.G(i, k, n)

    def I(self, i, k, n):
        return (self.F(i, k, n) - self.G(i, k, n)) / self.Z[i]


def step_index(t, dt):
    """Step n that holds time t: the stored values are valid on n dt <= t < (n + 1) dt."""
    return int(np.floor(t / dt + 1e-9))
