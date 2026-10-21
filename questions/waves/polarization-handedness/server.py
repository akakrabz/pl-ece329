"""Lecture 24 concept question: name polarizations by the notes' hand rule, the power of a circular wave,
phase and group velocity in a collisionless plasma, and a pool of polarization facts.

Phasor convention Re{E~ e^{j w t}}; a sine is -j.  A field with transverse components on the cyclic pair (a, b)
of the travel axis u (a_hat x b_hat = +u_hat) and travel sign s (e^{-j s beta u}) has E(0) = Re E~, E(T/4) = -Im E~
where the exponent is 1, and
    u_hat . [E(0) x E(T/4)] = -s Im(conj(Ea) Eb)     > 0 right-handed, < 0 left-handed, = 0 linear,
    circular  <=>  Ea^2 + Eb^2 = 0  (equal magnitudes, 90 deg apart).
(a) five fields, each written as a phasor or as real cosines/sines, travel along a random axis and sign.
(b) circular wave E0 [cos, sin] in a dielectric eps_r (eta0 = 120 pi): <S> = 2 E0^2/(2 eta) along the travel,
    |H| = E0/eta at every instant, S(t) constant.
(c) plasma: w_p^2 = N e^2/(m eps0); v_p = c/sqrt(1 - fp^2/f^2), v_g = c sqrt(1 - fp^2/f^2), delay L/v_g.
(d) checkbox pool; one statement's truth depends on the variant.
"""
import math
import random
from fractions import Fraction

AXES = ["x", "y", "z"]
CYC = {"x": ("y", "z"), "y": ("z", "x"), "z": ("x", "y")}
IDX = {"x": 0, "y": 1, "z": 2}
C = 3e8
ETA0 = 120 * math.pi
QE = 1.602e-19
ME = 9.109e-31
EPS0 = 8.854e-12
CAT_NAME = {"lin": "linear", "rc": "right-circular", "lc": "left-circular",
            "re": "right-handed elliptical", "le": "left-handed elliptical"}


def num(x, sig=4):
    if abs(x) < 1e-12:
        return "0"
    s = f"{x:.{sig}g}"
    if "e" in s:
        m, e = s.split("e")
        return f"{m}\\times10^{{{int(e)}}}"
    return s


def norm(th):
    """Phase in units of pi, reduced to (-1, 1]."""
    th = Fraction(th) % 2
    return th - 2 if th > 1 else th


def pi_tex(th):
    """|th| (units of pi, a Fraction) as LaTeX: 1/3 -> \\pi/3, 2/3 -> 2\\pi/3, 1 -> \\pi."""
    th = abs(th)
    n, d = th.numerator, th.denominator
    s = ("" if n == 1 else str(n)) + r"\pi"
    return s if d == 1 else s + "/" + str(d)


def cval(amp, th):
    return complex(amp * math.cos(math.pi * th), amp * math.sin(math.pi * th))


def hat(ax):
    return r"\hat{" + ax + "}"


def uhat(ax, sg):
    return ("+" if sg > 0 else "-") + hat(ax)


def vec_tex(v):
    """3-vector -> '3\\hat{x} - 1.5\\hat{y}'."""
    out = ""
    for i, ax in enumerate(AXES):
        x = v[i]
        if abs(x) < 1e-9:
            continue
        mag = num(abs(x))
        body = ("" if mag == "1" else mag) + hat(ax)
        if not out:
            out = ("-" if x < 0 else "") + body
        else:
            out += (" - " if x < 0 else " + ") + body
    return out or r"\mathbf{0}"


def join_terms(terms):
    out = terms[0]
    for t in terms[1:]:
        out += (" - " + t[1:]) if t.startswith("-") else (" + " + t)
    return out


def phasor_coef(amp, th):
    if th == 0:
        return f"{amp}"
    if th == 1:
        return f"-{amp}"
    if th == Fraction(1, 2):
        return f"{amp}j"
    if th == Fraction(-1, 2):
        return f"-{amp}j"
    return f"{amp}e^{{{'-' if th < 0 else ''}j{pi_tex(th)}}}"


def time_term(amp, th, arg):
    if th == 0:
        return f"{amp}\\cos({arg})"
    if th == 1:
        return f"-{amp}\\cos({arg})"
    if th == Fraction(-1, 2):
        return f"{amp}\\sin({arg})"
    if th == Fraction(1, 2):
        return f"-{amp}\\sin({arg})"
    return f"{amp}\\cos({arg} {'+' if th > 0 else '-'} {pi_tex(th)})"


def classify(Ea, Eb, s):
    cr = (Ea.conjugate() * Eb).imag
    if abs(cr) < 1e-9:
        return "lin"
    right = (-s * cr) > 0
    circ = abs(Ea * Ea + Eb * Eb) < 1e-9
    if circ:
        return "rc" if right else "lc"
    return "re" if right else "le"


def make_comps(cat, s):
    """Cyclic components [(amp, phase/pi), (amp, phase/pi)] of the requested category for travel sign s."""
    if cat == "lin":
        A = random.choice([2, 3, 4, 5, 6])
        B = random.choice([A, A, 2, 3, 4, 5, 6, 8])
        phi = random.choice([Fraction(0), Fraction(1)])
    elif cat in ("rc", "lc"):
        h = 1 if cat == "rc" else -1
        A = B = random.choice([2, 3, 4, 5, 6, 8])
        phi = Fraction(-h * s, 2)
    else:
        h = 1 if cat == "re" else -1
        kind = random.choice(["eq", "neq90", "neq"])
        if kind == "eq":
            A = B = random.choice([2, 3, 4, 5, 6])
            phi0 = random.choice([Fraction(1, 3), Fraction(1, 4), Fraction(2, 3), Fraction(3, 4)])
        else:
            A, B = random.sample([2, 3, 4, 5, 6, 8], 2)
            phi0 = Fraction(1, 2) if kind == "neq90" else random.choice([Fraction(1, 3), Fraction(2, 3), Fraction(1, 4)])
        phi = -h * s * phi0
    alpha = random.choice([Fraction(0), Fraction(1, 2), Fraction(1), Fraction(-1, 2)])
    return [(A, norm(alpha)), (B, norm(alpha + phi))]


def field_tex(u, s, comps, rep):
    """comps on the cyclic pair of u. Returns the LaTeX of the field (phasor or real time form)."""
    a, b = CYC[u]
    items = [(a, comps[0]), (b, comps[1])]
    random.shuffle(items)
    if rep == "phasor":
        simple = all(th in (0, 1, Fraction(1, 2), Fraction(-1, 2)) for _, (_, th) in items)
        pre = ""
        if simple and random.random() < 0.35:
            F = random.choice([Fraction(1, 2), Fraction(-1, 2), Fraction(1)])
            pre = {Fraction(1, 2): "j", Fraction(-1, 2): "-j", Fraction(1): "-"}[F]
            items = [(ax, (amp, norm(th - F))) for ax, (amp, th) in items]
        terms = [phasor_coef(amp, th) + r"\," + hat(ax) for ax, (amp, th) in items]
        expo = ("-" if s > 0 else "") + r"j\beta " + u
        return r"\tilde{\mathbf{E}} = " + pre + "(" + join_terms(terms) + r")\,e^{" + expo + "}"
    arg = r"\omega t " + ("-" if s > 0 else "+") + r" \beta " + u
    terms = [time_term(amp, th, arg) + r"\," + hat(ax) for ax, (amp, th) in items]
    return r"\mathbf{E} = " + join_terms(terms)


def snapshot(u, comps):
    a, b = CYC[u]
    Ea, Eb = cval(*comps[0]), cval(*comps[1])
    e0 = [0.0, 0.0, 0.0]
    e4 = [0.0, 0.0, 0.0]
    e0[IDX[a]], e0[IDX[b]] = round(Ea.real, 12), round(Eb.real, 12)
    e4[IDX[a]], e4[IDX[b]] = round(-Ea.imag, 12), round(-Eb.imag, 12)
    return e0, e4, Ea, Eb


def cross(p, q):
    return [p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0]]


def generate(data):
    # ------------------------------------------------------------------ (a) matching
    cats_all = ["lin", "rc", "lc", "re", "le"]
    while True:
        cats = random.sample(cats_all, 4) + [random.choice(cats_all)]
        if "rc" in cats and "lc" in cats:
            break
    random.shuffle(cats)
    reps = ["phasor", "phasor", "time", "time", random.choice(["phasor", "time"])]
    random.shuffle(reps)
    while True:
        signs = [random.choice([1, -1]) for _ in range(5)]
        if 1 <= sum(sg < 0 for sg in signs) <= 4:
            break
    items, texts = [], set()
    for cat, rep, s in zip(cats, reps, signs):
        while True:
            u = random.choice(AXES)
            comps = make_comps(cat, s)
            tex = field_tex(u, s, comps, rep)
            if tex not in texts:
                break
        texts.add(tex)
        e0, e4, Ea, Eb = snapshot(u, comps)
        got = classify(Ea, Eb, s)
        assert got == cat, (cat, got)
        uv = [0.0, 0.0, 0.0]
        uv[IDX[u]] = float(s)
        trip = sum(x * y for x, y in zip(uv, cross(e0, e4)))
        m0 = math.sqrt(sum(x * x for x in e0))
        m4 = math.sqrt(sum(x * x for x in e4))
        dot = sum(x * y for x, y in zip(e0, e4))
        if cat == "lin":
            shape = r"$\mathbf{E}(0)\times\mathbf{E}(T/4) = \mathbf{0}$: the vector stays on one line"
        elif cat in ("rc", "lc"):
            shape = (f"$\\lvert\\mathbf{{E}}(0)\\rvert = \\lvert\\mathbf{{E}}(T/4)\\rvert = {num(m0)}$ and "
                     r"$\mathbf{E}(0)\cdot\mathbf{E}(T/4) = 0$: circular")
        else:
            shape = (f"$\\lvert\\mathbf{{E}}(0)\\rvert = {num(m0)}$, $\\lvert\\mathbf{{E}}(T/4)\\rvert = {num(m4)}$, "
                     f"$\\mathbf{{E}}(0)\\cdot\\mathbf{{E}}(T/4) = {num(dot)}$: not circular, so elliptical")
        hand = "" if cat == "lin" else (", right-handed" if trip > 0 else ", left-handed")
        why = (f"Travel ${uhat(u, s)}$; $\\mathbf{{E}}(0) = {vec_tex(e0)}$, $\\mathbf{{E}}(T/4) = {vec_tex(e4)}$; "
               f"$\\hat{{u}}\\cdot[\\mathbf{{E}}(0)\\times\\mathbf{{E}}(T/4)] = {num(trip)}${hand}; {shape}.")
        items.append({"key": cat, "text": "$" + tex + "$", "verdict": CAT_NAME[cat], "why": why,
                      "u": u, "s": s, "rep": rep})

    # ------------------------------------------------------------------ (b) circular wave power
    ub = random.choice(AXES)
    sb = random.choice([1, -1])
    E0 = random.choice([2, 3, 4, 5, 6, 8, 10, 12])
    n = random.choice([1.0, 1.5, 2.0, 2.5, 3.0])
    eps_r = n * n
    hb = random.choice([1, -1])
    alpha = random.choice([Fraction(0), Fraction(1, 2), Fraction(1), Fraction(-1, 2)])
    comps_b = [(E0, norm(alpha)), (E0, norm(alpha + Fraction(-hb * sb, 2)))]
    wave_tex = field_tex(ub, sb, comps_b, "time")
    eta = ETA0 / n
    S_mW = sb * 2 * E0 ** 2 / (2 * eta) * 1e3
    H_mA = E0 / eta * 1e3
    medium_b = (r"vacuum ($\epsilon = \epsilon_0$, $\mu = \mu_0$)" if n == 1.0 else
                f"a lossless, non-magnetic dielectric with $\\epsilon_r = {num(eps_r)}$ ($\\mu_r = 1$)")
    s_inst = [
        {"id": "const", "correct": "true", "text": r"is constant in time and equal to $\langle\mathbf{S}\rangle$",
         "feedback": r"Right: $\lvert\mathbf{E}\rvert$ and $\lvert\mathbf{H}\rvert$ never change and $\mathbf{E}\perp\mathbf{H}$, so $\mathbf{E}\times\mathbf{H}$ is the same at every instant."},
        {"id": "pulse", "correct": "false", "text": r"pulses between $0$ and $2\langle\mathbf{S}\rangle$, twice per period",
         "feedback": r"That is a linearly polarized wave, whose $\lvert\mathbf{E}\rvert^2\propto\cos^2$. Here the two components take turns: $\cos^2+\sin^2 = 1$."},
        {"id": "rotate", "correct": "false", "text": r"rotates in the plane transverse to the travel, following $\mathbf{E}$",
         "feedback": r"$\mathbf{E}$ and $\mathbf{H}$ rotate together, but $\mathbf{E}\times\mathbf{H}$ stays along $\hat{u}$, the direction of travel."},
        {"id": "zeroavg", "correct": "false", "text": r"averages to zero over a period, because $\mathbf{E}$ and $\mathbf{H}$ keep rotating",
         "feedback": r"$\mathbf{S}$ always points along the travel with the same magnitude; its average is $\lvert\tilde{\mathbf{E}}\rvert^2/2\eta\neq 0$."},
    ]
    random.shuffle(s_inst)

    # ------------------------------------------------------------------ (c) plasma
    mant, ex = random.choice([(2, 11), (3, 11), (4, 11), (5, 11), (6, 11), (8, 11), (1, 12), (1.5, 12), (2, 12)])
    N = mant * 10.0 ** ex
    fp_MHz = math.sqrt(N * QE ** 2 / (ME * EPS0)) / (2 * math.pi) / 1e6
    while True:
        r = random.choice([1.2, 1.25, 1.4, 1.5, 1.75, 2.0, 2.5])
        f_MHz = math.ceil(fp_MHz * r * 2) / 2
        if f_MHz / fp_MHz >= 1.15:
            break
    L_km = random.choice([100, 150, 200, 250, 300])
    root = math.sqrt(1 - (fp_MHz / f_MHz) ** 2)
    vp8 = 3.0 / root
    vg8 = 3.0 * root
    delay_ms = L_km * 1e3 / (vg8 * 1e8) * 1e3
    crest_ms = L_km * 1e3 / (vp8 * 1e8) * 1e3
    vac_ms = L_km * 1e3 / C * 1e3
    N_tex = (f"{mant:g}" + r"\times10^{" + str(ex) + "}") if mant != 1 else (r"10^{" + str(ex) + "}")
    sig = [
        {"id": "vg", "correct": "true", "text": r"the group velocity $v_g$, which is less than $c$",
         "feedback": r"Right: the message is the envelope $m(t - z/v_g)$; $v_g = d\omega/d\beta \lt c$."},
        {"id": "vp", "correct": "false", "text": r"the phase velocity $v_p$, which is greater than $c$",
         "feedback": r"$v_p = \omega/\beta$ is the speed of the carrier crests. A pure cosine carries no information; the envelope moves at $v_g$."},
        {"id": "c", "correct": "false", "text": r"the vacuum speed $c$: the plasma moves only the crests, not the envelope",
         "feedback": r"The envelope is delayed: $v_g = c\sqrt{1 - f_p^2/f^2}\lt c$, so the message arrives later than in vacuum."},
        {"id": "mean", "correct": "false", "text": r"the average $(v_p + v_g)/2$ of the two speeds",
         "feedback": r"No such rule. The envelope moves at $v_g = d\omega/d\beta$ alone."},
        {"id": "none", "correct": "false", "text": r"no velocity: a collisionless plasma reflects signals at every frequency",
         "feedback": r"Only frequencies below $f_p$ fail to propagate; this carrier is above $f_p$, so $\beta$ is real."},
    ]
    random.shuffle(sig)

    # ------------------------------------------------------------------ (d) checkbox pool
    fac = random.choice([("j", "j"), ("-1", "-1"), ("-j", "-j"), (r"e^{j\pi/3}", "e^{j\\pi/3}")])
    us = random.choice(AXES)
    pq = list(CYC[us])
    sheet_pq = pq if random.random() < 0.5 else pq[::-1]
    uf = random.choice(AXES)
    fpq = list(CYC[uf])
    if random.random() < 0.5:
        fpq = fpq[::-1]
    fsg = random.choice([1, -1])
    # (p + fsg j q) e^{-j beta u}: right-circular iff -Im(conj(Ep) Eq) > 0 in cyclic order
    cyc_order = (fpq == list(CYC[uf]))
    Ep, Eq = 1 + 0j, complex(0, fsg)
    Ea, Eb = (Ep, Eq) if cyc_order else (Eq, Ep)
    rc_form_true = classify(Ea, Eb, 1) == "rc"
    rc_form_cls = CAT_NAME[classify(Ea, Eb, 1)]
    pm = "+" if fsg > 0 else "-"
    facts = [
        ("factor", True, f"Multiplying a wave's phasor $\\tilde{{\\mathbf{{E}}}}$ by ${fac[0]}$ changes neither its polarization nor its handedness.",
         r"An overall factor $e^{j\alpha}$ only shifts the whole pattern in time by $\alpha/\omega$; the rotation and its sense are unchanged."),
        ("H_same", True, r"In a circularly polarized plane wave, $\mathbf{H}$ rotates in the same sense as $\mathbf{E}$.",
         r"$\mathbf{H} = \hat{u}\times\mathbf{E}/\eta$ is $\mathbf{E}$ turned $90^\circ$ about $\hat{u}$ at every instant: same shape, same hand."),
        ("lin_sum", True, r"A linearly polarized wave is the sum of a right-circular and a left-circular wave of equal amplitude travelling in the same direction.",
         r"For travel along $+z$: $\hat{x} = \tfrac12(\hat{x} - j\hat{y}) + \tfrac12(\hat{x} + j\hat{y})$."),
        ("twice", True, r"In a lossless medium, a circularly polarized wave carries twice the time-average power density of either of its two linear components alone.",
         r"$\langle\mathbf{S}\rangle = (\lvert\tilde{E}_a\rvert^2 + \lvert\tilde{E}_b\rvert^2)/2\eta$, two equal terms."),
        ("clockwise", True, r"Seen by an observer looking along the direction of travel (from behind the wave), the $\mathbf{E}$ of a right-circular wave turns clockwise.",
         r"With the right thumb pointing away from you, the fingers curl clockwise as you see them."),
        ("same_fn", True, r"If two perpendicular components have the same time dependence, for example both $\propto\cos(\omega t-\beta z)$, the wave is linearly polarized whatever their amplitudes.",
         r"$\mathbf{E} = (A\hat{x} + B\hat{y})\cos(\omega t - \beta z)$ stays on the line of $A\hat{x} + B\hat{y}$."),
        ("quarter_any", False, r"Two perpendicular components that are $90^\circ$ out of phase always add up to a circularly polarized wave.",
         r"Equal amplitudes are needed too: $(3\hat{x} - 2j\hat{y})e^{-j\beta z}$ is right-handed elliptical."),
        ("reverse", False, r"Reversing the direction of travel while keeping the same rotation of $\mathbf{E}$ at a fixed point keeps the handedness.",
         r"The thumb flips, the fingers do not: $(\hat{x} - j\hat{y})e^{-j\beta z}$ is right-circular, $(\hat{x} - j\hat{y})e^{j\beta z}$ left-circular."),
        ("half_period", False, r"At a fixed point, the $\mathbf{E}$ of a circular wave makes one full turn every half period.",
         r"It makes one turn per period: the rotation frequency is the wave frequency."),
        ("sheet_same", False, f"The rotating sheet current $\\mathbf{{J}}_s = J_0[\\cos(\\omega t)\\,{hat(sheet_pq[0])} + \\sin(\\omega t)\\,{hat(sheet_pq[1])}]$ on the plane ${us} = 0$ radiates waves of the same handedness into both half-spaces.",
         r"$\mathbf{E} = -\tfrac{\eta}{2}\mathbf{J}_s(t\mp " + us + r"/v)$ rotates the same way on both sides, but the two waves travel in opposite directions: one is right-circular, the other left-circular."),
        ("unequal_lin", False, r"Adding a right-circular and a left-circular wave of unequal amplitudes, travelling in the same direction, gives a linearly polarized wave.",
         r"Only equal amplitudes give a line; otherwise the result is elliptical, with the hand of the larger part."),
        ("rc_form", rc_form_true, f"For travel along $+{hat(uf)}$, the phasor $({hat(fpq[0])} {pm} j{hat(fpq[1])})e^{{-j\\beta {uf}}}$ is right-circular.",
         f"At ${uf} = 0$: $\\mathbf{{E}}(0) = {hat(fpq[0])}$, $\\mathbf{{E}}(T/4) = {'-' if fsg > 0 else ''}{hat(fpq[1])}$, and "
         f"${hat(uf)}\\cdot[\\mathbf{{E}}(0)\\times\\mathbf{{E}}(T/4)] = {'+1' if rc_form_true else '-1'}$: it is {rc_form_cls}."),
    ]
    fact_list = [{"id": fid, "correct": "true" if t else "false", "text": txt,
                  "feedback": ("True: " if t else "False: ") + why, "verdict": "true" if t else "false", "why": why}
                 for fid, t, txt, why in facts]
    random.shuffle(fact_list)

    p = data["params"]
    p.update({
        "match_items": items,
        "wave_tex": wave_tex, "medium_b": medium_b, "ub": ub, "sb": sb, "E0": E0, "eps_r": eps_r, "n": n,
        "eta_str": num(eta), "n_str": num(n), "S_str": num(S_mW), "H_str": num(H_mA),
        "S_one_str": num(abs(S_mW) / 2), "trav_b": uhat(ub, sb), "sgn_b": "" if sb > 0 else "-",
        "s_inst": s_inst,
        "N_tex": N_tex, "N": N, "f_MHz": f_MHz, "f_str": f"{f_MHz:g}", "L_km": L_km,
        "fp_str": num(fp_MHz), "ratio_str": num(fp_MHz / f_MHz), "root_str": num(root),
        "vp_str": num(vp8), "vg_str": num(vg8), "delay_str": num(delay_ms), "crest_str": num(crest_ms),
        "vac_str": num(vac_ms), "wp_str": num(2 * math.pi * fp_MHz * 1e6),
        "sig": sig, "facts": fact_list,
    })
    ca = data["correct_answers"]
    ca["S_avg"] = float(S_mW)
    ca["H_mag"] = float(H_mA)
    ca["fp"] = float(fp_MHz)
    ca["vp8"] = float(vp8)
    ca["vg8"] = float(vg8)
    ca["delay"] = float(delay_ms)
