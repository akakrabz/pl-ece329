"""Independent check for midterm3/parallel-plate-line-fields.

Route: circuit quantities are rebuilt from the answered fields with vector algebra and compared with the statement.
  (a) Z0 = sqrt(L'/C'), v = 1/sqrt(L'C') with L' = mu0 d/W and C' = eps W/d (the server uses eta d/W and c/sqrt(eps_r));
  (b) with H = y_hat H_y and E = x_hat E_x taken from the answers: n x H on the named plate must give the stated
      current; E x H must point along the stated travel with |E|/|H| = sqrt(mu/eps); V by a numerical line integral;
      the plate charge from n . D, cross-checked two ways: charge per length = C' V, and current = (charge per
      length) x (velocity) for a single travelling wave;
  (c) the V/I choices against the computed ratio; (d) power by integrating Re(E x H*)/2 over the cross-section.
"""
import math

import numpy as np

MU0 = 4e-7 * math.pi
EPS0 = 1e-9 / (36 * math.pi)
X, Y, Z = np.eye(3)


def close(a, b, rel=1e-6):
    return abs(a - b) <= rel * max(abs(a), abs(b)) + 1e-12


def phys(d, check):
    p, ca = d["params"], d["correct_answers"]
    q = "parallel-plate-line-fields"
    W, dd, eps, s = p["W"], p["d"], p["epsr"] * EPS0, p["s"]
    Lp, Cp = MU0 * dd / W, eps * W / dd
    Z0, v = math.sqrt(Lp / Cp), 1 / math.sqrt(Lp * Cp)
    check(close(Z0, ca["Z0"], 1e-5) and close(v / 1e8, ca["v"], 1e-5), f"{q}: Z0 or v ({Z0}, {v})")
    H, E = Y * ca["Hy"], X * ca["Ex"]
    n = -X if p["plate"] == "top" else X
    Js = np.cross(n, H)
    check(close(Js @ Z * W, p["isg"] * p["I1"]) and abs(Js @ X) + abs(Js @ Y) < 1e-12, f"{q}: plate current does not match n x H")
    S = np.cross(E, H)
    check(S @ Z * s > 0 and close(abs(ca["Ex"] / ca["Hy"]), math.sqrt(MU0 / eps), 1e-5), f"{q}: E, H are not a wave toward {'+' if s > 0 else '-'}z")
    xs = np.linspace(0, dd, 1001)
    V = -np.trapezoid(np.full_like(xs, ca["Ex"]), xs)
    check(close(V, ca["V"], 1e-6), f"{q}: V {V} vs {ca['V']}")
    rho = (-X) @ (eps * E)
    check(close(rho * 1e9, ca["rho"], 1e-6), f"{q}: rho_s")
    Itop = np.cross(-X, H) @ Z * W
    check(close(rho * W, Cp * V, 1e-6), f"{q}: charge per length != C' V")
    check(close(Itop, s * v * rho * W, 1e-5), f"{q}: I != (charge per length) x velocity")
    for c in p["ratio"]:
        truth = c["key"] is not None and close(V / Itop, c["key"] * Z0, 1e-5)
        check((c["correct"] == "true") == truth, f"{q}: V/I choice {c['text']} mis-marked")
    check(sum(c["correct"] == "true" for c in p["ratio"]) == 1, f"{q}: V/I MC")
    E0 = p["E0"]
    H0 = E0 / math.sqrt(MU0 / eps)
    ys, xg = np.linspace(0, W, 201), np.linspace(0, dd, 201)
    P = np.trapezoid(np.trapezoid(np.full((201, 201), 0.5 * E0 * H0), xg, axis=1), ys)
    check(close(P, ca["P"], 1e-6) and close(E0 * dd, ca["V0"]), f"{q}: power {P} vs {ca['P']}")
    check(close(ca["V0"] ** 2 / (2 * Z0), ca["P"], 1e-5), f"{q}: P != V0^2 / 2 Z0")
