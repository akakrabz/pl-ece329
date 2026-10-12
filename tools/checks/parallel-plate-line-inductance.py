"""Independent check for midterm2/parallel-plate-line-inductance.

Routes that avoid the server's closed forms:
  * H: brute-force Biot-Savart. Each plate is a strip of width 10 000 d made of 4000 infinite filaments along z
    on a sinh-graded grid (field of each: dI/(2 pi) z x rho / rho^2, np.cross), summed at the middle of the gap.
  * B: not mu_r mu0 H, but mu0 times the vacuum field of free + bound currents, the bound sheets being
    J_sM = M x n_out on both faces of the filling with M = (mu_r - 1) H.
  * Psi: B . (vector area of the actual current loop source -> outgoing plate -> short -> return plate).
  * Energy: 1/2 B.H over the gap volume, and against 1/2 L I^2.
  * MC (b): direction of the numerical B; MC (f): ratios from re-running the same numerics on the rebuilt line.
"""
import math
import re
from fractions import Fraction

import numpy as np

MU0 = 4e-7 * math.pi
AX = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}


def _sheet_field(Jz, u_hat, q_hat, u_plate, r_f, d):
    """H at r_f from a strip of z-directed filaments (surface current Jz along +z) lying on u = u_plate.
    Filaments sit on a sinh-graded grid (dense under the field point), strip half-width 5000 d."""
    n = 4000
    h = 0.5 * d
    S = math.asinh(5000.0 * d / h)
    ds = 2 * S / n
    s = -S + (np.arange(n) + 0.5) * ds
    qs = (r_f @ q_hat) + h * np.sinh(s)
    dq = h * np.cosh(s) * ds
    pos = u_plate * u_hat[None, :] + qs[:, None] * q_hat[None, :]
    rho = r_f[None, :] - pos
    rho2 = np.sum(rho * rho, axis=1)
    zz = np.broadcast_to(AX["z"], rho.shape)
    dH = (Jz * dq)[:, None] / (2 * math.pi) * np.cross(zz, rho) / rho2[:, None]
    return dH.sum(axis=0)


def _line(u, out_plate, w, d, ell, I, mur):
    u_hat = AX[u]
    q_hat = AX["y" if u == "x" else "x"]
    u_out = d if out_plate == "d" else 0.0
    u_ret = 0.0 if out_plate == "d" else d
    Js = I / w
    r_f = 0.5 * d * u_hat
    J_at = {u_out: +Js, u_ret: -Js}                       # outgoing plate: toward the short, +z
    H = _sheet_field(J_at[d], u_hat, q_hat, d, r_f, d) + _sheet_field(J_at[0.0], u_hat, q_hat, 0.0, r_f, d)
    M = (mur - 1) * H
    JsM_top = np.cross(M, u_hat)                          # face u = d, outward normal +u
    JsM_bot = np.cross(M, -u_hat)                         # face u = 0, outward normal -u
    Hvac = (_sheet_field(J_at[d] + JsM_top[2], u_hat, q_hat, d, r_f, d)
            + _sheet_field(J_at[0.0] + JsM_bot[2], u_hat, q_hat, 0.0, r_f, d))
    B = MU0 * Hvac
    zz = AX["z"]
    P = [u_out * u_hat, u_out * u_hat + ell * zz, u_ret * u_hat + ell * zz, u_ret * u_hat]
    Avec = 0.5 * sum(np.cross(P[i], P[(i + 1) % 4]) for i in range(4))
    Psi = float(B @ Avec)
    W = 0.5 * float(B @ H) * w * d * ell
    return {"Jd": J_at[d], "J0": J_at[0.0], "H": H, "B": B, "Psi": Psi, "L": Psi / I, "W": W,
            "wm": 0.5 * float(B @ H), "JsM": JsM_top}


def phys(d, check):
    p, c = d["params"], d["correct_answers"]
    w = float(p["w_cm"]) / 100
    gap = float(p["d_mm"]) / 1000
    ell, I, mur = float(p["ell"]), float(p["I"]), float(p["mur"])
    u, out_plate = p["u"], p["out_plate"]
    check(gap <= w / 10 + 1e-15, "plate line: gap not << width")
    r = _line(u, out_plate, w, gap, ell, I, mur)

    def close(a, b, tol, msg):
        check(abs(a - b) <= tol * max(1e-9, abs(b)), f"plate line: {msg}: check {a} vs server {b}")

    close(r["Jd"], c["Js_d"], 1e-12, "Js on u = d")
    close(r["J0"], c["Js_0"], 1e-12, "Js on u = 0")
    Hn = float(np.linalg.norm(r["H"]))
    Bn = float(np.linalg.norm(r["B"]))
    close(Hn, c["H"], 2e-3, "|H| (numerical Biot-Savart)")
    close(Bn * 1e3, c["B"], 2e-3, "|B| (free + bound currents in vacuum)")
    close(r["Psi"] * 1e6, c["Psi"], 2e-3, "Psi")
    check(r["Psi"] > 0, "plate line: linked flux must be positive (L > 0)")
    close(r["L"] * 1e6, c["L"], 2e-3, "L")
    close(r["wm"], c["wm"], 4e-3, "energy density")
    close(r["W"] * 1e6, c["W"], 4e-3, "energy")
    close(0.5 * c["L"] * 1e-6 * I * I * 1e6, c["W"], 1e-9, "W = L I^2 / 2")
    close(r["JsM"][2] / 1e3, c["JsM"], 2e-3, "bound surface current on u = d")
    check(abs(r["JsM"][0]) + abs(r["JsM"][1]) <= 1e-6 * abs(r["JsM"][2]), "plate line: bound current not along z")
    check(np.sign(c["JsM"]) == np.sign(c["Js_d"]), "plate line: bound current must follow the adjacent free current")
    # field outside the line vanishes (numerically)
    u_hat = AX[u]
    q_hat = AX["y" if u == "x" else "x"]
    out_pt = 3.0 * gap * u_hat
    Hout = (_sheet_field(c["Js_d"], u_hat, q_hat, gap, out_pt, gap) + _sheet_field(c["Js_0"], u_hat, q_hat, 0.0, out_pt, gap))
    check(np.linalg.norm(Hout) <= 2e-3 * Hn, f"plate line: H outside the plates not ~0 ({np.linalg.norm(Hout)})")

    # MC (b): direction of B
    ch = p["dir_choices"]
    check(sum(x["correct"] == "true" for x in ch) == 1, "plate line: direction MC needs one correct")
    bdir = r["B"] / Bn
    for x in ch:
        m = re.fullmatch(r"\$([+-])\\hat\{([xyz])\}\$(.*)", x["text"])
        if m:
            vec = (1 if m.group(1) == "+" else -1) * AX[m.group(2)]
            match = float(vec @ bdir) > 0.999 and m.group(3) == ""
        else:
            match = False
        check(match == (x["correct"] == "true"), f"plate line: direction choice {x['text']} vs B along {bdir}")
    # MC (f): scaling, by re-running the numerics on the rebuilt line
    kd, kI = Fraction(p["kd"]), Fraction(p["kI"])
    r2 = _line(u, out_plate, w, gap * float(kd), ell, I * float(kI), mur)
    ratios = [float(np.linalg.norm(r2["B"])) / Bn, r2["Psi"] / r["Psi"], r2["L"] / r["L"]]
    sc = p["scale_choices"]
    check(sum(x["correct"] == "true" for x in sc) == 1, "plate line: scaling MC needs one correct")
    check(len({tuple(x["factors"]) for x in sc}) == len(sc), "plate line: scaling choices not distinct")
    for x in sc:
        f = [float(Fraction(v)) for v in x["factors"]]
        match = all(abs(a - b) <= 1e-3 * b for a, b in zip(ratios, f))
        check(match == (x["correct"] == "true"), f"plate line: scaling choice {x['factors']} vs numerical {ratios}")
        texf = re.findall(r"\\times (\\tfrac\{\d+\}\{\d+\}|\d+)", x["text"])
        texv = [float(Fraction(int(a), int(b))) if a else float(n)
                for t in texf for (a, b, n) in [re.match(r"\\tfrac\{(\d+)\}\{(\d+)\}|(\d+)", t).groups()]]
        check(len(texv) == 3 and all(abs(a - b) < 1e-12 for a, b in zip(texv, f)), f"plate line: choice text {x['text']}")
    # the figure's source arrow points into the outgoing plate
    check(("M60 107V93" in p["svg"]) == (out_plate == "d"), "plate line: figure source arrow")
    check(len(p["svg"]) <= 2000, "plate line: figure too large")
