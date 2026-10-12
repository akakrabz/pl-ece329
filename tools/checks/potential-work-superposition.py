"""Independent check for potential-work-superposition (Lecture 5).

* V_A, V_B: integrate -E.dl from infinity down a ray perpendicular to the plane of the charges, with E built
  as a Coulomb vector sum (Gauss-Legendre quadrature, substitution t = s u/(1-u) for the infinite range).
* W_ext: integrate the agent's force -q3 E along a path that leaves the plane (A -> A + h z -> B + h z -> B),
  i.e. not the straight segment, which also tests path independence.
* MC: the correct choice must say "increases" exactly when W_ext > 0, with |W_ext| in it; the field's work has
  the opposite sign.  Checkbox: static truths from a table written here, the A/B statement from the numbers.
"""
import math
import re

import numpy as np

EPS0 = 8.854e-12
K = 1.0 / (4 * math.pi * EPS0)
_X, _WX = np.polynomial.legendre.leggauss(400)
U01, W01 = 0.5 * (_X + 1.0), 0.5 * _WX          # Gauss-Legendre on [0, 1]

STATIC_TRUTH = {"perp": True, "path": True, "electron": True, "midpoint": True, "equip": True, "double": True,
                "V0E0": False, "proton": False, "loop": False, "vector": False, "fieldwork": False, "ev": False}


def efield(pts, charges):
    E = np.zeros_like(pts)
    for q, r0 in charges:
        R = pts - r0
        d = np.linalg.norm(R, axis=1)
        E += K * q * R / d[:, None] ** 3
    return E


def v_from_infinity(P, charges, s=0.05):
    """V(P) = -int_inf^P E.dl along the ray P + t z (t from inf to 0)  =  int_0^inf E_z(P + t z) dt."""
    t = s * U01 / (1.0 - U01)
    dt_du = s / (1.0 - U01) ** 2
    pts = P[None, :] + t[:, None] * np.array([0.0, 0.0, 1.0])
    return float(np.sum(efield(pts, charges)[:, 2] * dt_du * W01))


def line_integral(P, Q, charges):
    """int_P^Q E.dl along the straight segment P -> Q."""
    seg = Q - P
    pts = P[None, :] + U01[:, None] * seg[None, :]
    return float(np.sum(efield(pts, charges) @ seg * W01))


def phys(d, check):
    p, c = d["params"], d["correct_answers"]
    charges = [(p["q1"] * 1e-9, np.array([p["x1"], p["y1"], 0.0]) * 1e-2),
               (p["q2"] * 1e-9, np.array([p["x2"], p["y2"], 0.0]) * 1e-2)]
    A = np.array([p["xA"], p["yA"], 0.0]) * 1e-2
    B = np.array([p["xB"], p["yB"], 0.0]) * 1e-2
    VA, VB = v_from_infinity(A, charges), v_from_infinity(B, charges)
    check(abs(VA - c["VA"]) <= 1e-6 * max(1.0, abs(VA)), f"potential-work: V_A {VA} vs {c['VA']}")
    check(abs(VB - c["VB"]) <= 1e-6 * max(1.0, abs(VB)), f"potential-work: V_B {VB} vs {c['VB']}")

    h = np.array([0.0, 0.0, 0.05])
    int_E = (line_integral(A, A + h, charges) + line_integral(A + h, B + h, charges)
             + line_integral(B + h, B, charges))                     # int_A^B E.dl along a detour
    W = -p["q3"] * 1e-9 * int_E * 1e6                                  # agent's work [uJ]
    check(abs(W - c["W"]) <= 1e-6 * max(1e-3, abs(W)), f"potential-work: W_ext {W} vs {c['W']}")
    check(abs(W - p["q3"] * 1e-9 * (VB - VA) * 1e6) <= 1e-6 * abs(W), "potential-work: W != q3 (VB - VA)")

    # robustness of the numbers to hand rounding (the server promises this)
    check(abs(VB - VA) >= 50 and abs(VA) > 1 and abs(VB) > 1, "potential-work: near-cancellation slipped through")
    check(abs(VB - VA) >= 0.25 * (abs(VA) + abs(VB)) * (1 - 1e-9),
          "potential-work: V_B - V_A too small a difference for 3-s.f. V_A, V_B to reach W within 2 %")

    # (c) energy MC
    mc = p["mc"]
    check(sum(x["correct"] == "true" for x in mc) == 1 and len({x["text"] for x in mc}) == len(mc) == 5,
          "potential-work: MC needs 5 distinct choices, exactly one correct")
    corr = [x for x in mc if x["correct"] == "true"][0]
    txt = corr["text"]
    check(("increases" in txt) == (W > 0) and ("decreases" in txt) == (W < 0),
          f"potential-work: MC direction wrong for W = {W}: {txt}")
    check(("negative work" in txt) == (W > 0) and ("positive work" in txt) == (W < 0),
          f"potential-work: MC field-work sign wrong for W = {W}: {txt}")
    check(not re.search(r"\\mu|\d\.\d|\d\s*\\times", "".join(x["text"] for x in mc)),
          "potential-work: MC choices must not show numbers (leak of (b))")
    check(all(x["feedback"] for x in mc), "potential-work: every MC choice needs feedback")
    nums = [float(v) for v in re.findall(r"([+-]?\d+(?:\.\d+)?)\\ \\mu", p["c_words"])]
    check(len(nums) == 3 and abs(nums[0] - W) <= 6e-3 * abs(W) and abs(abs(nums[1]) - abs(W)) <= 6e-3 * abs(W)
          and abs(nums[2] + W) <= 6e-3 * abs(W), f"potential-work: panel (c) numbers {nums} vs W_ext {W}")
    check(all(not re.search(r"\d", x["feedback"]) for x in p["cb"] if x["key"] == "AB"), "potential-work: numbers in feedback")

    # (d) checkbox pool
    cb = p["cb"]
    check(8 <= len(cb) <= 14, "potential-work: checkbox pool size")
    for x in cb:
        want = (VA > VB) if x["key"] == "AB" else STATIC_TRUTH.get(x["key"])
        check(want is not None, f"potential-work: unknown statement key {x['key']}")
        check((x["correct"] == "true") == bool(want), f"potential-work: statement {x['key']} marked {x['correct']}")
        check(x["tf"] == ("True" if want else "False"), f"potential-work: tf label for {x['key']}")
    check(sum(x["correct"] == "false" for x in cb) >= 4 and sum(x["correct"] == "true" for x in cb) >= 4,
          "potential-work: pool needs >= 4 true and >= 4 false statements")

    # a numerical sanity check of two pool statements (midpoint between equal + charges; V = 0 midway in a +/- pair)
    pair = [(1e-9, np.array([-0.1, 0, 0])), (1e-9, np.array([0.1, 0, 0]))]
    mid = np.array([[0.0, 0.0, 0.0]])
    check(np.linalg.norm(efield(mid, pair)) < 1e-9 and v_from_infinity(mid[0], pair) > 100, "potential-work: midpoint statement")
    dip = [(1e-9, np.array([-0.1, 0, 0])), (-1e-9, np.array([0.1, 0, 0]))]
    check(abs(v_from_infinity(mid[0], dip)) < 1e-6 and np.linalg.norm(efield(mid, dip)) > 100, "potential-work: V=0 with E!=0")

    # answer-panel numbers agree with the keys
    for key, s in (("VA", p["VA_str"]), ("VB", p["VB_str"]), ("W", p["W_str"])):
        check(abs(float(s) - c[key]) <= 1e-3 * abs(c[key]), f"potential-work: panel {key} {s} vs {c[key]}")
