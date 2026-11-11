"""Midterm 3 exam-style: a matched generator on a shorted or open line; steady state of a step, then a pulse.

Family of FA23 Exam 3 #5 (re-parameterized). Rg = Z0, so tau_g = 1/2 and Gamma_g = 0: there are only two waves, the
incident one and one echo, delayed by 2T at the source end. Gamma_L = -1 (short) or +1 (open).
    step V0 u(t), t -> infinity (DC):  short: V = 0, I = V0/Rg;   open: V = V0, I = 0
    pulse of height V0 and length tp = step minus delayed step. At z = 0, with h = V0/2:
        incident alone:  V = h,            I = h/Z0
        echo alone:      V = Gamma_L h,    I = -Gamma_L h/Z0
        both:            the sum
    energy: the load dissipates nothing and the line ends empty, so everything the source delivered is heat in Rg.
"""
import random


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


def merge(segs):
    out = []
    for t0, t1, v in segs:
        if out and abs(out[-1][2] - v) < 1e-12:
            out[-1] = (out[-1][0], t1, v)
        else:
            out.append((t0, t1, v))
    return out


def plot(segs, nT, T, lo, hi, h):
    """Piecewise-constant record on 0..nT*T (segs in units of T) as an inline SVG.
    The vertical axis carries symbolic ticks only (V0/2, V0, -V0/2), no numbers."""
    W, H, L, R, top, bot = 390, 150, 56, 14, 20, 30
    span = (hi - lo) or 1.0

    def X(t):
        return L + (W - L - R) * t / nT

    def Y(v):
        return top + (H - top - bot) * (hi - v) / span

    out = [f'<svg viewBox="0 0 {W} {H}" width="{W}" style="max-width:100%" role="img" aria-label="voltage against time">',
           '<g fill="none" stroke="currentColor">']
    out.append(f'<path stroke-width="1" d="M{L} {top - 8}V{H - bot + 4}M{L} {Y(0):.1f}H{W - R + 6}"/>')
    for k in range(1, nT + 1):
        out.append(f'<path stroke-width="0.6" stroke-dasharray="2 3" d="M{X(k):.1f} {top - 4}V{H - bot}"/>')
    ticks = [(lv, name) for lv, name in ((h, "V0/2"), (2 * h, "V0"), (-h, "&#8722;V0/2")) if lo - 1e-9 <= lv <= hi + 1e-9]
    for lv, _ in ticks:
        out.append(f'<path stroke-width="0.8" stroke-dasharray="7 4" d="M{L - 4} {Y(lv):.1f}H{W - R}"/>')
    d = f"M{X(0):.1f} {Y(0):.1f}"
    for t0, t1, v in segs:
        d += f"V{Y(v):.1f}H{X(t1):.1f}"
    out.append(f'<path stroke-width="2.4" d="{d}V{Y(0):.1f}"/></g><g fill="currentColor" font-size="11">')
    for lv, name in ticks + [(0.0, "0")]:
        out.append(f'<text x="{L - 7}" y="{Y(lv) + 4:.1f}" text-anchor="end">{name}</text>')
    for k in range(0, nT + 1):
        out.append(f'<text x="{X(k):.1f}" y="{H - bot + 16}" text-anchor="middle">{fnum(k * T)}</text>')
    out.append(f'<text x="{W - R}" y="{H - 2}" text-anchor="end">t (&#956;s)</text><text x="{L + 6}" y="{top - 8}">V(0, t)</text></g></svg>')
    return "".join(out)


def generate(data):
    Z0 = random.choice([50, 60, 75, 100])
    V0 = random.choice([30, 60, 120, 240])
    v = random.choice([100, 150, 200, 300])                 # m / microsecond
    T = random.choice([0.5, 1, 1.5, 2])                     # microseconds
    ell = v * T
    short = random.random() < 0.5
    G = -1 if short else 1
    m = random.choice([1, 3, 4, 6])
    tp = m * T
    h = V0 / 2

    Vdc0 = 0.0 if short else float(V0)
    Idc = V0 / Z0 if short else 0.0

    inc, echo = (h, h / Z0), (G * h, -G * h / Z0)
    if m > 2:
        times = [T, (2 * T + tp) / 2, tp + T]
        vals = [inc, (inc[0] + echo[0], inc[1] + echo[1]), echo]
        regime = ["only the incident wave is at $z = 0$", "the incident wave and the echo overlap", "the source is off and only the echo is left"]
        corr = [(0, 2, h), (2, m, h * (1 + G)), (m, m + 2, G * h), (m + 2, m + 3, 0)]
        flip = [(0, 2, h), (2, m, h * (1 - G)), (m, m + 2, -G * h), (m + 2, m + 3, 0)]
        early = [(0, 1, h), (1, m, h * (1 + G)), (m, m + 1, G * h), (m + 1, m + 3, 0)]
        none = [(0, 2, h), (2, m, h * (1 + G)), (m, m + 3, 0)]
    else:
        times = [0.5 * T, 1.5 * T, 2.5 * T]
        vals = [inc, (0.0, 0.0), echo]
        regime = ["only the incident wave is at $z = 0$", "the pulse has left and its echo has not yet returned", "only the echo is at $z = 0$"]
        corr = [(0, 1, h), (1, 2, 0), (2, 3, G * h), (3, 4, 0)]
        flip = [(0, 1, h), (1, 2, 0), (2, 3, -G * h), (3, 4, 0)]
        early = [(0, 1, h), (1, 2, G * h), (2, 4, 0)]
        none = [(0, 1, h), (1, 4, 0)]
    nT = m + 3
    end_name = "short" if short else "open end"
    cands = [
        ("A", corr, "Correct."),
        ("B", flip, f"The echo has the wrong sign: at {'a short' if short else 'an open end'} $\\Gamma_L = {G}$, so the echo is {'inverted' if short else 'upright'}."),
        ("C", early, "The echo returns too early here. It has to travel to the far end and back: it reaches $z = 0$ at $2T$, not at $T$."),
        ("D", none, "This record stops when the source switches off. The echo of the last part of the pulse is still on the line and keeps arriving for another $2T$."),
    ]
    lo, hi = -1.25 * h, 2.3 * h                       # same scale for every variant: room for -V0/2 ... V0
    plots = [{"key": k, "text": plot(merge(sg), nT, T, lo, hi, h), "correct": "true" if k == "A" else "false",
              "feedback": fb, "segs": [[a * T, b * T, float(val)] for a, b, val in merge(sg)]} for k, sg, fb in cands]
    random.shuffle(plots)

    energy = [
        {"key": "rg", "correct": "true", "text": "All the energy the source delivered has been turned into heat in $R_g$.", "feedback": "Correct."},
        {"key": "load", "correct": "false", "text": f"Half of the energy was dissipated at the {end_name}, the other half in $R_g$.",
         "feedback": f"At the {end_name} either $V$ or $I$ is zero at all times, so the power $VI$ delivered there is zero."},
        {"key": "stored", "correct": "false", "text": "The energy that was launched is still stored on the line, in its electric and magnetic fields.",
         "feedback": "The line carries no voltage or current once the echo has been absorbed, so it stores nothing."},
        {"key": "none", "correct": "false", "text": "The source delivered no net energy: the echo returned to it everything that was launched.",
         "feedback": "The echo is absorbed by $R_g$ (the generator is matched). The source itself supplied positive power while it was on."},
    ]
    random.shuffle(energy)

    names = ["a", "b", "c"]
    data["params"].update({
        "Z0": Z0, "V0": V0, "v": v, "ell": fnum(ell), "tp": fnum(tp), "T": T,
        "load_txt": "short-circuited" if short else "left open-circuited",
        "plots": plots, "energy": energy,
        "ta": fnum(times[0]), "tb": fnum(times[1]), "tc": fnum(times[2]),
        # solution strings
        "s_T": fnum(T), "s_2T": fnum(2 * T), "s_G": str(G), "s_h": fnum(h), "s_Ih": fnum(h / Z0),
        "s_dcV": fnum(Vdc0), "s_dcI": fnum(Idc),
        "s_dc_why": ("the short forces $V = 0$ everywhere, and the current is $V_0/R_g$" if short else "no current can flow into the open end, so nothing drops across $R_g$ and $V = V_0$ everywhere"),
        "s_echoV": fnum(G * h), "s_echoI": fnum(-G * h / Z0),
        "s_rows": [{"n": names[i], "t": fnum(times[i]), "why": regime[i], "V": fnum(vals[i][0]), "I": fnum(vals[i][1])} for i in range(3)],
        "s_end": fnum(tp + 2 * T),
        # raw data for the independent check
        "short": short, "m": m, "times": times,
    })
    ca = data["correct_answers"]
    ca["Vdc0"], ca["Idc"] = Vdc0, Idc
    ca["tg"], ca["GL"], ca["Gg"] = 0.5, float(G), 0.0
    for i, nm in enumerate(names):
        ca["V" + nm], ca["I" + nm] = float(vals[i][0]), float(vals[i][1])
