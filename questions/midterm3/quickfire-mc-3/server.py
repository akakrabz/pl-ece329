"""Midterm 3 quick-fire: six short multiple-choice items (Lectures 21-30), drawn from fourteen generators.

Each generator returns {gen, stem, choices[4], solution, data}; exactly one choice is correct, every distractor
carries feedback naming the slip, and every choice carries a machine-readable "val" (number) or "key" (label) so that
the independent check can decide each one without parsing text. Families: the concept and multiple-choice parts of
SP17 Exam 3 #1-#2, SP18 Exam 3 #1 and FA23 Exam 3 #1 (fresh items).
  phasor     phasor of A sin(wt -/+ beta z + phi)
  losstan    classify a medium from sigma, eps_r, f
  skin       skin depth after a change of frequency
  depth      fraction of the power left at a depth of m skin depths
  polname    name the polarization of E0 (a_hat + c b_hat) exp(-/+ j beta w)
  standing   spacing of E nulls / first H null in front of a mirror
  gamma      reflection coefficient at a dielectric interface, either way
  backward   current of a wave travelling toward -z (or +z) from its voltage
  inject     voltage launched at t = 0+
  gammai     current reflection coefficient of a load
  steady     load voltage long after a step
  echotime   when the first echo passes a probe
  junction   voltage transmitted across a junction of two lines
  ppline     how Z0 of a parallel-plate line scales
"""
import math
import random
from fractions import Fraction as F

AXES = ["x", "y", "z"]


def g3(x):
    if x == 0:
        return "0"
    s = f"{x:.3g}"
    if "e" in s:
        m, e = s.split("e")
        return f"{m}\\times10^{{{int(e)}}}"
    return s


def ftex(q):
    q = F(q)
    if q.denominator == 1:
        return str(q.numerator)
    return f"{'-' if q < 0 else ''}\\tfrac{{{abs(q.numerator)}}}{{{q.denominator}}}"


def num(x):
    s = f"{float(x):.4g}"
    return s


def ch(text, ok, fb, val=None, key=None):
    return {"text": text, "correct": "true" if ok else "false", "feedback": fb, "val": val, "key": key}


def add(choices, val, text, fb, sep=0.03):
    """Append a numeric distractor unless four choices exist or its value is too close to one already offered."""
    if len(choices) >= 4 or any(abs(val - c["val"]) <= sep * max(abs(val), abs(c["val"]), 1e-12) for c in choices):
        return
    choices.append(ch(text, False, fb, val=val))


def pad(choices, fmt, fb="Check the formula and the numbers once more."):
    """Fill up to four numeric choices with simple multiples of the correct value (used only when slips coincide)."""
    true = [c for c in choices if c["correct"] == "true"][0]["val"]
    for k in (1.5, 2.0 / 3, 3.0, 1.0 / 3, 4.0, 0.25, -1.0):
        add(choices, true * k, fmt(true * k), fb)


def finish(gen, stem, choices, solution, data):
    texts = [c["text"] for c in choices]
    assert len(choices) == 4 and len(set(texts)) == 4, (gen, texts)
    assert sum(c["correct"] == "true" for c in choices) == 1, gen
    random.shuffle(choices)
    return {"gen": gen, "stem": stem, "choices": choices, "solution": solution, "data": data}


def cross_sign(a, b):
    ia, ib = AXES.index(a), AXES.index(b)
    return AXES[3 - ia - ib], (1 if (ia + 1) % 3 == ib else -1)


# ------------------------------------------------------------------ generators
def g_phasor():
    A = random.choice([2, 3, 5, 8, 10])
    phi = random.choice([30, 45, 60, 120, 150])
    s = random.choice([1, -1])
    sg, sgw = ("-", "+") if s > 0 else ("+", "-")

    def tex(ang, sign):
        return f"${A}\\,e^{{{'' if ang >= 0 else '-'}j{abs(ang)}^\\circ}}\\,e^{{{sign}j\\beta z}}$"

    stem = (f"With the convention $E(z,t) = \\mathrm{{Re}}\\{{\\tilde E(z)\\,e^{{j\\omega t}}\\}}$, the phasor of "
            f"$E = {A}\\sin(\\omega t {sg} \\beta z + {phi}^\\circ)\\ \\mathrm{{V/m}}$ is")
    choices = [
        ch(tex(phi - 90, sg), True, "Right: $\\sin x = \\cos(x-90^\\circ)$.", val=[phi - 90, s]),
        ch(tex(phi + 90, sg), False, "A sine lags a cosine: subtract $90^\\circ$, do not add it.", val=[phi + 90, s]),
        ch(tex(phi, sg), False, "That is the phasor of a cosine with this phase. The sine needs an extra $-90^\\circ$.", val=[phi, s]),
        ch(tex(phi - 90, sgw), False, "The space factor has the wrong sign: this phasor travels the other way.", val=[phi - 90, -s]),
    ]
    sol = f"$\\sin(\\theta+{phi}^\\circ) = \\cos(\\theta+{phi}^\\circ-90^\\circ)$, so the phasor is ${A}\\,e^{{j({phi}^\\circ-90^\\circ)}}e^{{{sg}j\\beta z}}$."
    return finish("phasor", stem, choices, sol, {"A": A, "phi": phi, "s": s})


def g_losstan():
    while True:
        name, sigma, er = random.choice([("seawater", 4.0, 81), ("moist soil", 1e-2, 16), ("a ferrite", 1e-2, 9), ("distilled water", 1e-4, 81),
                                         ("dry sand", 1e-5, 4), ("silicon", 1e-1, 12)])
        f_txt, f = random.choice([("1\\ \\mathrm{kHz}", 1e3), ("100\\ \\mathrm{kHz}", 1e5), ("10\\ \\mathrm{MHz}", 1e7), ("1\\ \\mathrm{GHz}", 1e9), ("30\\ \\mathrm{GHz}", 3e10)])
        tanl = sigma * 18e9 / (f * er)                    # sigma / (2 pi f eps_r eps0) with eps0 = 1e-9/36 pi
        if not (0.03 <= tanl <= 0.3 or 3 <= tanl <= 30):
            break
    key = "good" if tanl > 10 else "diel" if tanl < 0.1 else "mid"
    stem = (f"At $f = {f_txt}$, {name} has $\\sigma = {g3(sigma)}\\ \\mathrm{{S/m}}$ and $\\epsilon_r = {er}$ (non-magnetic). "
            "Taking a good conductor to mean $\\sigma/\\omega\\epsilon \\gt 10$ and an imperfect dielectric $\\sigma/\\omega\\epsilon \\lt 0.1$, at this frequency it is")
    fb = f"Compute $\\sigma/\\omega\\epsilon = \\sigma/(2\\pi f\\epsilon_r\\epsilon_0) \\approx {g3(tanl)}$ and compare with 0.1 and 10."
    choices = [
        ch("a good conductor", key == "good", "Right." if key == "good" else fb, key="good"),
        ch("an imperfect (low-loss) dielectric", key == "diel", "Right." if key == "diel" else fb, key="diel"),
        ch("neither: in between", key == "mid", "Right." if key == "mid" else fb, key="mid"),
        ch("a perfect dielectric", False, "Its conductivity is not zero, so it is lossy at every frequency.", key="perfect"),
    ]
    sol = (f"$\\sigma/\\omega\\epsilon = \\sigma/(2\\pi f\\epsilon_r\\epsilon_0) \\approx {g3(tanl)}$, which is "
           + {"good": "above 10: a good conductor.", "diel": "below 0.1: an imperfect dielectric.", "mid": "between 0.1 and 10: neither."}[key])
    return finish("losstan", stem, choices, sol, {"sigma": sigma, "er": er, "f": f})


def g_skin():
    d1 = random.choice([0.2, 0.5, 1, 2, 3, 6])
    k = random.choice([2, 3, 4])
    up = random.choice([True, False])
    f1 = random.choice([1, 10, 50, 100])
    f2 = f1 * k * k if up else f1 / (k * k)
    true = d1 / k if up else d1 * k
    stem = (f"In a good conductor the skin depth is ${num(d1)}\\ \\mathrm{{mm}}$ at ${num(f1)}\\ \\mathrm{{kHz}}$. "
            f"At ${num(f2)}\\ \\mathrm{{kHz}}$ (still a good conductor, same material) the skin depth is")
    choices = [ch(f"${num(true)}\\ \\mathrm{{mm}}$", True, "Right: $\\delta \\propto 1/\\sqrt f$.", val=true)]
    add(choices, d1 / k ** 2 if up else d1 * k ** 2, f"${num(d1 / k ** 2 if up else d1 * k ** 2)}\\ \\mathrm{{mm}}$", "The dependence is a square root: $\\delta = 1/\\sqrt{\\pi f\\mu\\sigma}$.")
    add(choices, d1 * k if up else d1 / k, f"${num(d1 * k if up else d1 / k)}\\ \\mathrm{{mm}}$", "Wrong direction: a higher frequency means a thinner skin.")
    add(choices, d1, f"${num(d1)}\\ \\mathrm{{mm}}$", "The skin depth depends on frequency: $\\delta = 1/\\sqrt{\\pi f\\mu\\sigma}$.")
    sol = f"$\\delta = 1/\\sqrt{{\\pi f\\mu\\sigma}} \\propto 1/\\sqrt f$. The frequency changes by a factor of {k * k}, so $\\delta$ changes by a factor of {k}: ${num(true)}\\ \\mathrm{{mm}}$."
    pad(choices, lambda x: f"${num(x)}\\ \\mathrm{{mm}}$")
    return finish("skin", stem, choices, sol, {"d1": d1, "f1": f1, "f2": f2})


def g_depth():
    m = random.choice([F(1, 2), F(1), F(3, 2), F(2)])
    mt = "\\delta" if m == 1 else f"{ftex(m)}\\delta" if m.denominator > 1 else f"{m}\\delta"

    def pct(e):
        return f"$e^{{-{num(e)}}} \\approx {100 * math.exp(-e):.3g}\\,\\%$"

    stem = (f"A plane wave enters a good conductor whose skin depth is $\\delta$. Of the time-average power density just inside the surface, "
            f"the fraction that is left at a depth of ${mt}$ is")
    two = float(2 * m)
    choices = [ch(pct(two), True, "Right: the power decays as $e^{-2z/\\delta}$.", val=math.exp(-two))]
    add(choices, math.exp(-float(m)), pct(float(m)), "That is the decay of the field amplitude. The power goes as the amplitude squared: $e^{-2z/\\delta}$.")
    add(choices, 1 - math.exp(-two), f"$1-e^{{-{num(two)}}} \\approx {100 * (1 - math.exp(-two)):.3g}\\,\\%$", "That is the fraction that has been dissipated above this depth, not the fraction left.")
    add(choices, math.exp(-float(m) / 2), pct(float(m) / 2), "The exponent for power is $2z/\\delta$, not $z/2\\delta$.")
    add(choices, math.exp(-4 * float(m)), pct(4 * float(m)), "The exponent for power is $2z/\\delta$.")
    sol = f"$\\langle S\\rangle \\propto \\lvert E\\rvert^2 \\propto e^{{-2z/\\delta}}$. At $z = {mt}$ this is $e^{{-{num(two)}}} \\approx {100 * math.exp(-two):.3g}\\,\\%$."
    return finish("depth", stem, choices, sol, {"m": float(m)})


def g_polname():
    w = random.choice(AXES)
    a, b = random.sample([x for x in AXES if x != w], 2)
    s = random.choice([1, -1])
    kind = random.choice(["+j", "-j", "+j", "-j", "real", "2j"])
    coef = {"+j": "+ j", "-j": "- j", "real": random.choice(["+", "-"]), "2j": random.choice(["+ 2j", "- 2j"])}[kind]
    sab = cross_sign(a, b)[1]                      # a x b = sab w
    if kind in ("+j", "-j"):
        lead_b = kind == "+j"                      # +j: b leads a, so E turns from b toward a ... i.e. from a toward -b
        turn = -sab if lead_b else sab             # turning sense about +w
        key = "rc" if turn * s > 0 else "lc"
    else:
        key = "lin" if kind == "real" else "ell"
    stem = (f"The plane wave $\\tilde{{\\mathbf{{E}}}} = E_0\\,(\\hat{{{a}}} {coef}\\,\\hat{{{b}}})\\,e^{{{'-' if s > 0 else '+'}j\\beta {w}}}$ "
            "(with $E_0$ real, convention $e^{j\\omega t}$) is")
    fbs = {"rc": "Follow the tip of $\\mathbf{E}$ at a fixed point for a quarter period and compare the turning sense with the travel direction (right-hand rule).",
           "lc": "Follow the tip of $\\mathbf{E}$ at a fixed point for a quarter period and compare the turning sense with the travel direction (right-hand rule).",
           "lin": "Linear needs the two components in phase or exactly opposite (a real ratio).",
           "ell": "Elliptical needs unequal amplitudes or a phase difference that is not $0$, $90^\\circ$ or $180^\\circ$."}
    names = {"rc": "right-hand circularly polarized", "lc": "left-hand circularly polarized", "lin": "linearly polarized", "ell": "elliptically polarized"}
    choices = [ch(names[k], k == key, "Right." if k == key else fbs[k], key=k) for k in ("rc", "lc", "lin", "ell")]
    why = {"rc": "equal amplitudes and a quarter-cycle phase difference: circular, and the turning sense is right-handed about the travel direction",
           "lc": "equal amplitudes and a quarter-cycle phase difference: circular, and the turning sense is left-handed about the travel direction",
           "lin": "the two components are in phase (real ratio): linear, along a tilted line",
           "ell": "a quarter-cycle phase difference but unequal amplitudes: elliptical"}[key]
    sol = f"The components along $\\hat{{{a}}}$ and $\\hat{{{b}}}$ have " + why + f". The wave travels along ${'+' if s > 0 else '-'}\\hat{{{w}}}$."
    return finish("polname", stem, choices, sol, {"a": a, "b": b, "w": w, "s": s, "coef": coef.replace(" ", "")})


def g_standing():
    fG = random.choice([0.3, 0.6, 1, 1.5, 3, 6, 10])
    lam = 30.0 / fG                                # cm
    mode = random.choice(["Enull", "Hnull"])
    if mode == "Enull":
        ask, true = "the distance between neighbouring planes on which the total $\\mathbf{E}$ is zero at all times is", lam / 2
        why = "Nulls of the same field are half a wavelength apart."
    else:
        ask, true = "the plane nearest to the conductor on which the total $\\mathbf{H}$ is zero at all times lies at a distance of", lam / 4
        why = "$\\mathbf{H}$ is largest on the conductor (where $\\mathbf{E}$ is zero); its first null is a quarter wavelength out."
    stem = f"A ${num(fG)}\\ \\mathrm{{GHz}}$ plane wave in air is reflected at normal incidence by a perfect conductor. In front of the conductor, {ask}"

    def cm(x):
        return f"${num(x)}\\ \\mathrm{{cm}}$"

    choices = [ch(cm(true), True, "Right. " + why, val=true)]
    add(choices, lam / 4 if mode == "Enull" else lam / 2, cm(lam / 4 if mode == "Enull" else lam / 2),
        "That mixes up the two spacings: nulls of one field are $\\lambda/2$ apart, and the nulls of $\\mathbf{E}$ and $\\mathbf{H}$ are offset by $\\lambda/4$.")
    add(choices, lam, cm(lam), "That is a full wavelength. The standing-wave pattern repeats every half wavelength.")
    add(choices, 2 * lam, cm(2 * lam), "Check the wavelength: $\\lambda = c/f$.")
    sol = f"$\\lambda = c/f = {num(lam)}\\ \\mathrm{{cm}}$. {why} Answer: ${num(true)}\\ \\mathrm{{cm}}$."
    pad(choices, cm)
    return finish("standing", stem, choices, sol, {"fG": fG, "mode": mode})


def g_gamma():
    n = random.choice([F(3, 2), F(2), F(3), F(4), F(5)])
    into = random.choice([True, False])            # True: from air into the dielectric
    G = (1 - n) / (1 + n) if into else (n - 1) / (n + 1)
    er = n * n
    diel = "a lossless, non-magnetic dielectric with $\\epsilon_r = " + num(er) + "$"
    first, second = ("air", diel) if into else (diel, "air")
    stem = (f"A plane wave travelling in {first} meets, at normal incidence, the flat surface of {second}. "
            "The reflection coefficient for the electric field is")
    choices = [ch(f"${ftex(G)}$", True, "Right.", val=float(G))]
    add(choices, float(-G), f"${ftex(-G)}$", "The sign is reversed: $\\Gamma = (\\eta_2-\\eta_1)/(\\eta_2+\\eta_1)$, and the denser medium has the smaller $\\eta$.")
    Ge = (1 - er) / (1 + er) if into else (er - 1) / (er + 1)
    add(choices, float(Ge), f"${ftex(Ge)}$", "Use the impedances, which go as $1/\\sqrt{\\epsilon_r}$, not the permittivities.")
    add(choices, float(1 + G), f"${ftex(1 + G)}$", "That is the transmission coefficient $\\tau = 1+\\Gamma$.")
    add(choices, float(G * G), f"${ftex(G * G)}$", "That is the reflected fraction of the power, $\\Gamma^2$.")
    add(choices, float(-(1 + G)), f"${ftex(-(1 + G))}$", "Check both the formula and the sign.")
    sol = (f"$\\eta \\propto 1/\\sqrt{{\\epsilon_r}}$, so with $\\sqrt{{\\epsilon_r}} = {ftex(n)}$: "
           f"$\\Gamma = \\dfrac{{\\eta_2-\\eta_1}}{{\\eta_2+\\eta_1}} = {ftex(G)}$.")
    pad(choices, lambda x: f"${num(x)}$")
    return finish("gamma", stem, choices, sol, {"n": float(n), "into": into})


def g_backward():
    Z0 = random.choice([50, 75, 100, 150, 300])
    V = random.choice([3, 6, 12, 15, 30]) * random.choice([1, -1])
    s = random.choice([-1, -1, 1])
    I = s * V / Z0 * 1000                            # mA
    stem = (f"On a lossless line with $Z_0 = {Z0}\\ \\Omega$, a single pulse travels toward ${'+' if s > 0 else '-'}z$. At some point and instant its voltage is "
            f"${V}\\ \\mathrm{{V}}$. With $I$ counted positive along $+z$ in the upper conductor, its current there is")

    def mA(x):
        return f"${num(x)}\\ \\mathrm{{mA}}$"

    choices = [ch(mA(I), True, "Right.", val=I)]
    add(choices, -I, mA(-I), "The sign is that of a wave travelling the other way: $I = +V/Z_0$ toward $+z$, $I = -V/Z_0$ toward $-z$.")
    add(choices, 2 * I, mA(2 * I), "A single travelling wave has $\\lvert I\\rvert = \\lvert V\\rvert/Z_0$, with no factor of 2.")
    add(choices, -I / 2, mA(-I / 2), "A single travelling wave has $\\lvert I\\rvert = \\lvert V\\rvert/Z_0$, and the sign follows the travel direction.")
    sol = f"A wave toward ${'+' if s > 0 else '-'}z$ has $I = {'+' if s > 0 else '-'}V/Z_0 = {num(I)}\\ \\mathrm{{mA}}$."
    pad(choices, mA)
    return finish("backward", stem, choices, sol, {"Z0": Z0, "V": V, "s": s})


def _line():
    Z0 = random.choice([50, 75, 100])
    Rg = Z0 * random.choice([F(1, 3), F(1, 2), F(2), F(3), F(1)])
    RL = Z0 * random.choice([F(1, 3), F(1, 2), F(2), F(3), F(5)])
    V0 = random.choice([12, 24, 30, 60, 120])
    return Z0, Rg, RL, V0


def g_inject():
    Z0, Rg, RL, V0 = _line()
    Vp = V0 * Z0 / (Rg + Z0)
    stem = (f"A source ${V0}\\,u(t)\\ \\mathrm{{V}}$ with internal resistance ${num(Rg)}\\ \\Omega$ is connected at $t = 0$ to a lossless line with $Z_0 = {Z0}\\ \\Omega$ "
            f"that ends in ${num(RL)}\\ \\Omega$. The voltage of the wave launched onto the line at $t = 0^+$ is")

    def V(x):
        return f"${num(x)}\\ \\mathrm{{V}}$"

    choices = [ch(V(Vp), True, "Right: at $t = 0^+$ the generator sees only $Z_0$.", val=float(Vp))]
    add(choices, float(V0 * RL / (Rg + RL)), V(V0 * RL / (Rg + RL)), "That is the final (DC) voltage. At $t = 0^+$ the load is not yet felt: the generator sees $Z_0$.")
    add(choices, float(V0), V(V0), "Part of the source voltage drops across $R_g$.")
    add(choices, float(V0 * Rg / (Rg + Z0)), V(V0 * Rg / (Rg + Z0)), "That is the voltage across $R_g$. The line gets the share $Z_0/(R_g+Z_0)$.")
    add(choices, float(V0) / 2, V(V0 / 2), "Half only when $R_g = Z_0$.")
    add(choices, float(Vp) * 2, V(Vp * 2), "Check the divider $Z_0/(R_g+Z_0)$.")
    sol = f"$V^+ = V_0\\,\\dfrac{{Z_0}}{{R_g+Z_0}} = {num(Vp)}\\ \\mathrm{{V}}$, whatever the load is."
    pad(choices, V)
    return finish("inject", stem, choices, sol, {"Z0": Z0, "Rg": float(Rg), "RL": float(RL), "V0": V0})


def g_gammai():
    Z0 = random.choice([50, 75, 100])
    r = random.choice([F(1, 3), F(1, 2), F(2), F(3), F(5), F(1, 5)])
    RL = Z0 * r
    G = (r - 1) / (r + 1)
    stem = (f"A lossless line with $Z_0 = {Z0}\\ \\Omega$ ends in a resistor of ${num(RL)}\\ \\Omega$. For a wave arriving at the load, the reflection coefficient for the "
            "<em>current</em> (reflected current over incident current, both counted along $+z$) is")
    choices = [ch(f"${ftex(-G)}$", True, "Right: $-\\Gamma_L$.", val=float(-G))]
    add(choices, float(G), f"${ftex(G)}$", "That is the voltage reflection coefficient. The reflected wave travels backward, so its current has the opposite sign.")
    add(choices, float(1 - G), f"${ftex(1 - G)}$", "That is the total load current over the incident current, not the reflection coefficient.")
    add(choices, float(1 + G), f"${ftex(1 + G)}$", "That is the load voltage over the incident voltage.")
    add(choices, float(-(1 + G)), f"${ftex(-(1 + G))}$", "Check the definition: $\\Gamma_L = (R_L-Z_0)/(R_L+Z_0)$ for voltage, and $-\\Gamma_L$ for current.")
    sol = f"$\\Gamma_L = \\dfrac{{R_L-Z_0}}{{R_L+Z_0}} = {ftex(G)}$ for voltage. A backward wave carries $I = -V/Z_0$, so for current the coefficient is $-\\Gamma_L = {ftex(-G)}$."
    pad(choices, lambda x: f"${num(x)}$")
    return finish("gammai", stem, choices, sol, {"Z0": Z0, "RL": float(RL)})


def g_steady():
    while True:
        Z0, Rg, RL, V0 = _line()
        if Rg != Z0:
            break
    Vinf = V0 * RL / (Rg + RL)
    Vp = V0 * Z0 / (Rg + Z0)
    GL = (RL - Z0) / (RL + Z0)
    stem = (f"A source ${V0}\\,u(t)\\ \\mathrm{{V}}$ with internal resistance ${num(Rg)}\\ \\Omega$ drives a lossless line with $Z_0 = {Z0}\\ \\Omega$ that ends in "
            f"${num(RL)}\\ \\Omega$. A long time after $t = 0$, the voltage across the load is")

    def V(x):
        return f"${num(x)}\\ \\mathrm{{V}}$"

    choices = [ch(V(Vinf), True, "Right: in the end the line is a pair of wires.", val=float(Vinf))]
    add(choices, float(Vp), V(Vp), "That is the first wave launched. After all the bounces $Z_0$ no longer matters.")
    add(choices, float(Vp * (1 + GL)), V(Vp * (1 + GL)), "That is the load voltage right after the first arrival. More echoes follow because the generator is not matched.")
    add(choices, float(V0), V(V0), "The source voltage divides between $R_g$ and $R_L$.")
    add(choices, float(V0 * Rg / (Rg + RL)), V(V0 * Rg / (Rg + RL)), "That is the voltage across $R_g$.")
    sol = f"At $t\\to\\infty$ the line is two wires: $V = V_0\\,\\dfrac{{R_L}}{{R_g+R_L}} = {num(Vinf)}\\ \\mathrm{{V}}$."
    pad(choices, V)
    return finish("steady", stem, choices, sol, {"Z0": Z0, "Rg": float(Rg), "RL": float(RL), "V0": V0})


def g_echotime():
    v = random.choice([100, 150, 200, 300])
    ell = random.choice([150, 300, 600])
    frac = random.choice([F(1, 4), F(1, 3), F(1, 2), F(2, 3), F(3, 4)])
    zp = frac * ell
    t = (2 * ell - zp) / v

    def us(x):
        return f"${num(x)}\\ \\mu\\mathrm{{s}}$"

    stem = (f"A lossless line is ${ell}\\ \\mathrm{{m}}$ long with $v = {v}\\ \\mathrm{{m}}/\\mu\\mathrm{{s}}$, and its load is not matched. A narrow pulse is launched at $z = 0$ at $t = 0$. "
            f"A probe at $z = {num(zp)}\\ \\mathrm{{m}}$ sees the first <em>echo</em> from the load at")
    choices = [ch(us(t), True, "Right: out to the load and back to the probe.", val=float(t))]
    add(choices, float(zp / v), us(zp / v), "That is when the pulse first passes the probe, on its way out.")
    add(choices, float(2 * ell / v), us(2 * ell / v), "That is when the echo reaches the generator end, not the probe.")
    add(choices, float((ell + zp) / v), us((ell + zp) / v), "Measure the return leg from the load back to the probe: $\\ell - z_p$, not $z_p$.")
    add(choices, float(ell / v), us(ell / v), "That is when the pulse reaches the load.")
    sol = f"The echo has travelled $\\ell + (\\ell - z_p) = {num(2 * ell - zp)}\\ \\mathrm{{m}}$: $t = {num(t)}\\ \\mu\\mathrm{{s}}$."
    pad(choices, us)
    return finish("echotime", stem, choices, sol, {"v": v, "ell": ell, "frac": [frac.numerator, frac.denominator]})


def g_junction():
    Z1, Z2 = random.sample([25, 50, 75, 100, 150, 300], 2)
    Vi = random.choice([6, 12, 24, 30, 60])
    t = F(2 * Z2, Z1 + Z2)

    def V(x):
        return f"${num(x)}\\ \\mathrm{{V}}$"

    stem = (f"A line with $Z_1 = {Z1}\\ \\Omega$ is joined to a long line with $Z_2 = {Z2}\\ \\Omega$. A pulse of ${Vi}\\ \\mathrm{{V}}$ arrives at the junction from line 1. "
            "The voltage pulse transmitted onto line 2 is")
    choices = [ch(V(Vi * t), True, "Right: $\\tau_{12} = 2Z_2/(Z_1+Z_2)$.", val=float(Vi * t))]
    add(choices, float(Vi * F(2 * Z1, Z1 + Z2)), V(Vi * F(2 * Z1, Z1 + Z2)), "That uses $2Z_1/(Z_1+Z_2)$, the coefficient for current (or for a wave coming from line 2).")
    add(choices, float(Vi * F(Z2 - Z1, Z1 + Z2)), V(Vi * F(Z2 - Z1, Z1 + Z2)), "That is the reflected pulse, $\\Gamma_{12}V_i$. The transmitted one is $(1+\\Gamma_{12})V_i$.")
    add(choices, float(Vi), V(Vi), "The pulse is not passed on unchanged unless the two impedances are equal.")
    add(choices, float(Vi * F(Z2, Z1 + Z2)), V(Vi * F(Z2, Z1 + Z2)), "A factor of 2 is missing: the junction voltage is the incident plus the reflected pulse.")
    sol = f"$\\tau_{{12}} = \\dfrac{{2Z_2}}{{Z_1+Z_2}} = {ftex(t)}$, so the transmitted pulse is ${num(Vi * t)}\\ \\mathrm{{V}}$. It can exceed the incident voltage when $Z_2 \\gt Z_1$; the power does not."
    pad(choices, V)
    return finish("junction", stem, choices, sol, {"Z1": Z1, "Z2": Z2, "Vi": Vi})


def g_ppline():
    what, fac = random.choice([("the plate width is doubled", 0.5), ("the plate separation is doubled", 2.0), ("the plate separation is halved", 0.5),
                               ("the filling is replaced by one with four times the permittivity", 0.5), ("the plate width is halved", 2.0),
                               ("the line is made twice as long", 1.0)])
    stem = (f"A parallel-plate line (width much larger than the separation, non-magnetic filling) has characteristic impedance $Z_0$. If {what} "
            "and nothing else changes, the characteristic impedance becomes")
    opts = [(0.5, "$\\tfrac12 Z_0$"), (2.0, "$2Z_0$"), (1.0, "$Z_0$ (unchanged)"), (0.25, "$\\tfrac14 Z_0$"), (4.0, "$4Z_0$")]
    fb = "$Z_0 = \\eta\\,d/w$ with $\\eta = \\eta_0/\\sqrt{\\epsilon_r}$: proportional to the separation, inversely to the width and to $\\sqrt{\\epsilon_r}$, and independent of the length."
    good = [o for o in opts if o[0] == fac]
    bad = [o for o in opts if o[0] != fac]
    random.shuffle(bad)
    choices = [ch(o[1], o[0] == fac, "Right." if o[0] == fac else fb, val=o[0]) for o in good + bad[:3]]
    sol = fb + f" Here the factor is ${num(fac)}$."
    return finish("ppline", stem, choices, sol, {"what": what})


GENERATORS = [g_phasor, g_losstan, g_skin, g_depth, g_polname, g_standing, g_gamma, g_backward, g_inject, g_gammai, g_steady,
              g_echotime, g_junction, g_ppline]


def generate(data):
    gens = random.sample(GENERATORS, 6)
    items = []
    for i, gfun in enumerate(gens, start=1):
        it = gfun()
        it["name"] = f"q{i}"
        it["num"] = i
        items.append(it)
    data["params"]["items"] = items
