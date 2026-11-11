"""Midterm 3 exam-style: find Rg, RL, the pulse height and eps_r of a line from two pulses seen at a probe.

Family of FA23 Exam 3 #4 and SP17 Exam 3 #2(v)-(vi) (re-parameterized; time-domain reflectometry). A narrow pulse of
open-circuit height V0 is launched into a line of known Z0 and length l. A probe at z_p sees a pulse of height A at
t1, a pulse of height rho*A at t2, and nothing else.
    timing:   t1 = z_p / v,  t2 = (2 l - z_p) / v   ->  v, and eps_r = (c / v)^2 for a non-magnetic line
    ratio:    rho = Gamma_L   ->  RL = Z0 (1 + Gamma_L)/(1 - Gamma_L)
    silence:  the echo is not re-launched at the generator  ->  Gamma_g = 0,  Rg = Z0,  A = V0 / 2
    at the load the pulse and its echo add:  A (1 + Gamma_L)
    with a mismatched Rg':  A' = V0 Z0/(Rg' + Z0), and a third pulse A' Gamma_L Gamma_g' at (2 l + z_p)/v
"""
import random
from fractions import Fraction as Fr

GAMS = [Fr(1, 2), Fr(-1, 2), Fr(1, 3), Fr(-1, 3), Fr(1, 5), Fr(-1, 5), Fr(3, 5), Fr(-3, 5)]


def fnum(x, sig=4):
    x = float(x)
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


def rho_plain(q):
    """-1/2 -> '-A/2', 3/5 -> '3A/5' (for the SVG label)."""
    num = "" if abs(q.numerator) == 1 else str(abs(q.numerator))
    return f"{'&#8722;' if q < 0 else ''}{num}A/{q.denominator}"


def record_svg(t1, t2, rho):
    W, H, L, R, base = 400, 150, 44, 16, 86
    tmax = t2 * 1.3

    def X(t):
        return L + (W - L - R) * t / tmax

    h1, h2 = 54, 54 * float(rho)
    out = [f'<svg viewBox="0 0 {W} {H}" width="{W}" style="max-width:100%" role="img" aria-label="Voltage at the probe against time: two narrow pulses">',
           '<g fill="none" stroke="currentColor" stroke-width="1">',
           f'<path d="M{L} 14V{H - 18}M{L} {base}H{W - R + 6}"/>',
           f'<path stroke-width="2.2" d="M{L} {base}H{X(t1) - 3:.1f}V{base - h1}h6V{base}H{X(t2) - 3:.1f}V{base - h2:.1f}h6V{base}H{W - R}"/>',
           '</g><g fill="currentColor" font-size="12">',
           f'<text x="{X(t1):.1f}" y="{base - h1 - 6}" text-anchor="middle">A</text>',
           f'<text x="{X(t2):.1f}" y="{(base - h2 - 6) if rho > 0 else (base - h2 + 15):.1f}" text-anchor="middle">{rho_plain(rho)}</text>',
           f'<text x="{X(t1):.1f}" y="{base + 16}" text-anchor="middle">{fnum(t1)}</text>',
           f'<text x="{X(t2):.1f}" y="{base + 16 if rho > 0 else base - 8}" text-anchor="middle">{fnum(t2)}</text>',
           f'<text x="{L - 6}" y="{base + 4}" text-anchor="end">0</text>',
           f'<text x="{W - R}" y="{H - 4}" text-anchor="end">t (&#956;s)</text><text x="{L + 6}" y="18">V at the probe</text></g></svg>']
    return "".join(out)


def generate(data):
    while True:
        Z0 = random.choice([50, 60, 75, 100, 150])
        GL = random.choice(GAMS)
        RL = Z0 * (1 + GL) / (1 - GL)
        v = random.choice([100, 150, 200, 300])               # m / microsecond
        ell = random.choice([60, 90, 120, 150, 300, 600])
        frac = random.choice([Fr(1, 4), Fr(1, 2), Fr(3, 4)])
        t1, t2 = frac * ell / v, (2 - frac) * ell / v
        rgp = random.choice([Fr(3), Fr(1, 3), Fr(2), Fr(1, 2), Fr(4), Fr(1, 4)])
        Rg2 = Z0 * rgp
        if RL.denominator == 1 and (t1 * 1000).denominator == 1 and (t2 * 1000).denominator == 1 and Rg2.denominator == 1:
            break
    V0 = random.choice([10, 20, 30, 40, 60, 100])
    A = Fr(V0, 2)
    zp = frac * ell
    Vload = A * (1 + GL)
    A2 = V0 / (1 + rgp)
    Gg2 = (rgp - 1) / (rgp + 1)
    t3 = (2 + frac) * ell / v
    C = A2 * GL * Gg2
    where = {Fr(1, 4): "a quarter of the way along the line", Fr(1, 2): "at the midpoint of the line", Fr(3, 4): "three quarters of the way along the line"}[frac]

    why = [
        {"key": "matched", "correct": "true", "text": "The echo is completely absorbed when it returns to the generator: $\\Gamma_g = 0$.", "feedback": "Correct: a re-launched echo would pass the probe a third time."},
        {"key": "loadabs", "correct": "false", "text": "The load absorbs the pulse completely: $\\Gamma_L = 0$.", "feedback": "Then there would be no second pulse at all. The second pulse is the load's echo."},
        {"key": "cancel", "correct": "false", "text": "The second pulse cancels the first one, so nothing is left on the line.", "feedback": "The two pulses pass the probe at different times and travel in opposite directions; they do not cancel. The echo has to be absorbed somewhere."},
        {"key": "lossy", "correct": "false", "text": "The line is lossy, so later echoes are too small to see.", "feedback": "The line is lossless. A lossless line with a mismatched generator would keep producing pulses."},
    ]
    random.shuffle(why)

    f = float
    data["params"].update({
        "svg": record_svg(f(t1), f(t2), GL), "V0": V0, "Z0": Z0, "ell": ell, "zp": fnum(zp), "where": where,
        "t1": fnum(t1), "t2": fnum(t2), "rho_tex": ("" if GL > 0 else "-") + ("" if abs(GL) == 1 else ftex(abs(GL))) + "A",
        "Rg2": int(Rg2), "why": why,
        # solution strings
        "s_v": str(v), "s_epsr": fnum((300 / v) ** 2), "s_GL": ftex(GL), "s_RL": str(int(RL)), "s_A": fnum(A),
        "s_Vload": fnum(Vload), "s_A2": fnum(A2), "s_Gg2": ftex(Gg2), "s_t3": fnum(t3), "s_C": fnum(C),
        "s_back": fnum(2 * ell - zp), "s_third": fnum(2 * ell + zp),
        # raw data for the independent check
        "frac": [frac.numerator, frac.denominator], "rho": f(GL), "t1v": f(t1), "t2v": f(t2),
    })
    ca = data["correct_answers"]
    ca["v"], ca["epsr"] = f(v), (300 / v) ** 2
    ca["GL"], ca["RL"] = f(GL), f(RL)
    ca["Rg"], ca["A"] = f(Z0), f(A)
    ca["Vload"] = f(Vload)
    ca["A2"], ca["t3"], ca["C"] = f(A2), f(t3), f(C)
