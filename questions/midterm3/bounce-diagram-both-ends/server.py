"""Midterm 3 exam-style: voltage and current bounce diagrams of one line, and the voltage at both ends.

Family of SP18 Exam 3 #3 and SP17 Exam 3 #5 (re-parameterized). Step source V0 u(t) behind Rg, line Z0 of one-way
delay T = l / v, resistive load RL.
    tau_g = Z0 / (Rg + Z0),  Gamma_L = (RL - Z0)/(RL + Z0),  Gamma_g = (Rg - Z0)/(Rg + Z0)
    voltage waves  V1+ = tau_g V0,  V1- = Gamma_L V1+,  V2+ = Gamma_g V1-,  V2- = Gamma_L V2+, ...
    current waves  I+ = V+/Z0,  I- = -V-/Z0
    source end: V1+ on (0, 2T), then + V1- + V2+ on (2T, 4T);  load end: 0 on (0, T), V1+ + V1- on (T, 3T), ...
    t -> infinity: the line is two wires,  V = V0 RL/(Rg + RL),  I = V0/(Rg + RL)
"""
import math
import random
from fractions import Fraction as Fr

RG = [Fr(1, 3), Fr(1, 2), Fr(2), Fr(3), Fr(1, 4), Fr(4), Fr(3, 5), Fr(3, 2), Fr(2, 3)]
RL = [Fr(1, 3), Fr(1, 2), Fr(2), Fr(3), Fr(5), Fr(1, 5), Fr(4), Fr(1, 4)]


def fnum(x, sig=4):
    if abs(x) < 1e-12:
        return "0"
    s = f"{x:.{sig}g}"
    if "e" in s:
        e = int(s.split("e")[1])
        s = f"{x:.{max(0, sig - 1 - e)}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def ftex(q):
    q = Fr(q)
    if q.denominator == 1:
        return str(q.numerator)
    return f"{'-' if q < 0 else ''}\\tfrac{{{abs(q.numerator)}}}{{{q.denominator}}}"


CIRCUIT = """<svg viewBox="0 0 420 92" width="420" style="max-width:100%" role="img" aria-label="Generator at z = 0, line of length l, load at z = l">
  <g fill="none" stroke="currentColor" stroke-width="1.5">
    <circle cx="30" cy="50" r="13"/><path d="M30 37V18h12l4-6 8 12 8-12 8 12 4-6h16M30 63v19h60M90 18h240M90 82h240M330 18v20M330 62v20"/>
    <rect x="322" y="38" width="16" height="24"/>
  </g>
  <g fill="currentColor" font-size="12">
    <text x="24" y="54">+</text><text x="44" y="10">Rg</text><text x="0" y="54">Vg</text>
    <text x="190" y="12">upper conductor, I along +z</text><text x="196" y="52">Z0, v</text><text x="345" y="54">RL</text>
    <text x="84" y="92">z = 0</text><text x="316" y="92">z = l</text>
  </g>
</svg>"""


def plot(segs, T, tu, lo, hi, ref):
    """Piecewise-constant record on 0..5T as an inline SVG. segs = [(t0, t1, value)] in units of T.
    No numbers on the vertical axis (they would give away other parts): one dashed reference line at ref = V1+."""
    W, H, L, R, top, bot = 370, 150, 50, 14, 20, 30
    span = (hi - lo) or 1.0

    def X(t):
        return L + (W - L - R) * t / 5.0

    def Y(v):
        return top + (H - top - bot) * (hi - v) / span

    out = [f'<svg viewBox="0 0 {W} {H}" width="{W}" style="max-width:100%" role="img" aria-label="voltage against time">',
           '<g fill="none" stroke="currentColor">']
    out.append(f'<path stroke-width="1" d="M{L} {top - 8}V{H - bot + 4}M{L} {Y(0):.1f}H{W - R + 6}"/>')
    for k in range(1, 6):
        out.append(f'<path stroke-width="0.6" stroke-dasharray="2 3" d="M{X(k):.1f} {top - 4}V{H - bot}"/>')
    out.append(f'<path stroke-width="1" stroke-dasharray="7 4" d="M{L - 4} {Y(ref):.1f}H{W - R}"/>')
    d = f"M{X(0):.1f} {Y(0):.1f}"
    for t0, t1, v in segs:
        d += f"V{Y(v):.1f}H{X(t1):.1f}"
    out.append(f'<path stroke-width="2.4" d="{d}"/></g><g fill="currentColor" font-size="11">')
    out.append(f'<text x="{L - 7}" y="{Y(ref) + 4:.1f}" text-anchor="end">V1+</text><text x="{L - 7}" y="{Y(0) + 4:.1f}" text-anchor="end">0</text>')
    for k in range(0, 6):
        out.append(f'<text x="{X(k):.1f}" y="{H - bot + 16}" text-anchor="middle">{fnum(k * T)}</text>')
    out.append(f'<text x="{W - R}" y="{H - 2}" text-anchor="end">t ({tu})</text><text x="{L + 6}" y="{top - 8}">V(l, t)</text></g></svg>')
    return "".join(out)


def generate(data):
    while True:
        Z0 = random.choice([50, 60, 75, 100, 150])
        rg, rl = random.choice(RG), random.choice(RL)
        Rg, RLv = Z0 * rg, Z0 * rl
        V1p = random.choice([6, 12, 24, 30, 60, 120])
        V0 = V1p * (1 + rg)
        Gg, GL = (rg - 1) / (rg + 1), (rl - 1) / (rl + 1)
        if Rg.denominator == 1 and RLv.denominator == 1 and V0.denominator == 1 and abs(Gg * GL) <= Fr(1, 2):
            break
    Rg, RLv, V0 = int(Rg), int(RLv), int(V0)
    tg = 1 / (1 + rg)
    ns = random.random() < 0.5
    T = random.choice([0.5, 1, 2, 2.5, 5] if ns else [0.5, 1, 2, 4])
    v8 = random.choice([1, 1.5, 2, 3])                           # 10^8 m/s
    ell = v8 * 1e8 * T * (1e-9 if ns else 1e-6)                  # metres
    tu = "ns" if ns else "\\mu s"
    ell_s, ell_u = (fnum(ell * 100), "cm") if ell < 1 else (fnum(ell), "m")

    V1m = GL * V1p
    V2p = Gg * V1m
    V2m = GL * V2p
    V3p = Gg * V2m
    Vs1, Vs2, Vs3 = V1p, V1p + V1m + V2p, V1p + V1m + V2p + V2m + V3p
    VL1, VL2 = V1p + V1m, V1p + V1m + V2p + V2m
    I1p, I1m = Fr(V1p) / Z0 * 1000, -V1m / Z0 * 1000             # mA
    Vinf, Iinf = Fr(V0 * RLv, Rg + RLv), Fr(V0 * 1000, Rg + RLv)

    # ---- (d) four candidate records for the load end
    w1m = -GL * V1p
    w2p = Gg * w1m
    w2m = -GL * w2p
    cands = [
        ("A", [(0, 1, 0), (1, 3, VL1), (3, 5, VL2)], "Correct: nothing until $T$; then every arriving wave and its reflection add at the load, with a new pair every $2T$."),
        ("B", [(0, 2, Vs1), (2, 4, Vs2), (4, 5, Vs3)], "This is the record at the source end: it starts at $t = 0$ and steps at $2T$, $4T$. The load sees nothing before $T$ and steps at $T$, $3T$, $5T$."),
        ("C", [(0, 1, 0), (1, 3, V1p), (3, 5, V1p + V1m + V2p)], "The timing is right, but each step counts only the arriving wave. At the load the reflected wave appears at the same instant: the first level is $V_1^+(1+\\Gamma_L)$."),
        ("D", [(0, 1, 0), (1, 3, V1p + w1m), (3, 5, V1p + w1m + w2p + w2m)], "This uses the opposite sign for $\\Gamma_L$. Check $\\Gamma_L = (R_L - Z_0)/(R_L + Z_0)$ against whether the load is larger or smaller than $Z_0$."),
    ]
    allv = [float(v) for _, sg, _ in cands for _, _, v in sg] + [0.0]
    lo, hi = min(allv), max(allv)
    pad = 0.12 * (hi - lo)
    plots = [{"key": k, "text": plot([(a, b, float(v)) for a, b, v in sg], T, "ns" if ns else "&#956;s", lo - (pad if lo < 0 else 0), hi + pad, float(V1p)),
              "correct": "true" if k == "A" else "false", "feedback": fb,
              "segs": [[a * T, b * T, float(v)] for a, b, v in sg]} for k, sg, fb in cands]
    random.shuffle(plots)

    f = float
    data["params"].update({
        "circuit": CIRCUIT, "V0": V0, "Rg": Rg, "RL": RLv, "Z0": Z0, "ell_s": ell_s, "ell_u": ell_u,
        "v_tex": f"{fnum(v8)}\\times10^{{8}}", "tu": tu, "plots": plots,
        "t2T": fnum(2 * T), "t4T": fnum(4 * T), "tT": fnum(T), "t3T": fnum(3 * T), "t5T": fnum(5 * T),
        # solution strings
        "s_tg": ftex(tg), "s_GL": ftex(GL), "s_Gg": ftex(Gg), "s_T": fnum(T),
        "s_V1p": fnum(f(V1p)), "s_V1m": fnum(f(V1m)), "s_V2p": fnum(f(V2p)), "s_V2m": fnum(f(V2m)),
        "s_I1p": fnum(f(I1p)), "s_I1m": fnum(f(I1m)), "s_I2p": fnum(f(V2p / Z0 * 1000)),
        "s_Vs1": fnum(f(Vs1)), "s_Vs2": fnum(f(Vs2)), "s_VL1": fnum(f(VL1)), "s_VL2": fnum(f(VL2)),
        "s_Vinf": fnum(f(Vinf)), "s_Iinf": fnum(f(Iinf)),
        # raw data for the independent check
        "T": T,
    })
    ca = data["correct_answers"]
    ca["tg"], ca["GL"], ca["Gg"] = f(tg), f(GL), f(Gg)
    ca["V1m"], ca["V2p"], ca["I1p"], ca["I1m"] = f(V1m), f(V2p), f(I1p), f(I1m)
    ca["Vs1"], ca["Vs2"], ca["VL1"], ca["VL2"] = f(Vs1), f(Vs2), f(VL1), f(VL2)
    ca["Vinf"], ca["Iinf"] = f(Vinf), f(Iinf)
