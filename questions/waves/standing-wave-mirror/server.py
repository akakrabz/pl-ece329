"""Lecture 26 concept question: a plane wave normally incident on a perfect mirror.

Air (vacuum, c = 3e8, eta0 = 120 pi) fills s1 w > 0; a perfect conductor fills s1 w < 0; mirror surface w = 0.
Incident wave (travels u_i = -s1 w_hat), h = 0 linear, h = +-1 circular:
    E_i = sig E0 [a cos(wt + s1 beta w) + h b sin(wt + s1 beta w)]      phasor sig E0 (a - j h b) e^{j s1 beta w}
Reflected (Gamma = -1, travels +s1 w_hat):  E_r phasor = -sig E0 (a - j h b) e^{-j s1 beta w}.
Distance from mirror D = s1 w:
    E = 2 sig E0 sin(beta D) [-a sin wt + h b cos wt],   |E| amplitude 2 E0 |sin beta D|
    H = -(2 s1 e sig E0/eta0) cos(beta w) [b cos wt - h a sin wt], e = eps(w, a, b);  amplitude 2 E0/eta0 |cos beta D|
E nulls D = m lambda/2, H nulls D = (2m+1) lambda/4.
J_s = n x H(0, t), n = s1 w_hat  ->  J_s = 2 E_i(0, t)/eta0.
Hand: u . (E(0) x E(T/4)) > 0 right; incident -s1 h e; reflected flips.
<P_rad> = 2 <S_i>/c, <S_i> = (1 + h^2) E0^2/(2 eta0); P(t) = mu0 |H(0,t)|^2/2.
"""
import math
import random
from fractions import Fraction

AXES = ["x", "y", "z"]
C0 = 3e8
ETA0 = 120 * math.pi

# (f in Hz, display, lambda in cm as a Fraction)
FREQS = [(3e8, r"300\ \mathrm{MHz}"), (5e8, r"500\ \mathrm{MHz}"), (7.5e8, r"750\ \mathrm{MHz}"),
         (1e9, r"1\ \mathrm{GHz}"), (1.2e9, r"1.2\ \mathrm{GHz}"), (2e9, r"2\ \mathrm{GHz}"),
         (2.5e9, r"2.5\ \mathrm{GHz}"), (3e9, r"3\ \mathrm{GHz}"), (5e9, r"5\ \mathrm{GHz}"),
         (6e9, r"6\ \mathrm{GHz}"), (7.5e9, r"7.5\ \mathrm{GHz}"), (1e10, r"10\ \mathrm{GHz}")]
LAM_CM = {3e8: Fraction(100), 5e8: Fraction(60), 7.5e8: Fraction(40), 1e9: Fraction(30), 1.2e9: Fraction(25),
          2e9: Fraction(15), 2.5e9: Fraction(12), 3e9: Fraction(10), 5e9: Fraction(6), 6e9: Fraction(5),
          7.5e9: Fraction(4), 1e10: Fraction(3)}
PROBE_R = [Fraction(1, 12), Fraction(1, 8), Fraction(1, 6), Fraction(1, 3), Fraction(3, 8), Fraction(5, 12),
           Fraction(7, 12), Fraction(5, 8), Fraction(2, 3), Fraction(5, 6), Fraction(7, 8), Fraction(11, 12),
           Fraction(9, 8), Fraction(7, 6)]
E0S = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 80, 100, 120, 150]
AREAS = [1, 2, 4, 5, 10]


def num(x, sig=4):
    if abs(x) < 1e-12:
        return "0"
    s = f"{x:.{sig}g}"
    if "e" in s:
        m, e = s.split("e")
        return f"{m}\\times10^{{{int(e)}}}"
    return s


def hat(ax):
    return r"\hat{" + ax + "}"


def sh(ax, sg):
    return ("+" if sg > 0 else "-") + hat(ax)


def lc(i, j, k):
    """Levi-Civita sign for axis letters."""
    return {("x", "y", "z"): 1, ("y", "z", "x"): 1, ("z", "x", "y"): 1,
            ("x", "z", "y"): -1, ("z", "y", "x"): -1, ("y", "x", "z"): -1}[(i, j, k)]


def generate(data):
    w = random.choice(AXES)
    a, b = random.sample([ax for ax in AXES if ax != w], 2)
    s1 = random.choice([1, -1])                  # air on the side s1 * w > 0
    sig = random.choice([1, -1])
    h = random.choice([0, 1, -1])                # 0 linear, +-1 circular
    circ = h != 0
    f, f_tex = random.choice(FREQS)
    lam = LAM_CM[f]
    E0 = random.choice(E0S)
    A = random.choice(AREAS)
    rs = [r for r in PROBE_R if (lam * r * 1000).denominator == 1]
    r = random.choice(rs)
    dp = lam * r                                  # cm, exact
    dp_f = float(dp)
    e = lc(w, a, b)
    beta = 2 * math.pi * f / C0

    # ------------------------------------------------------------------ answers
    dE = float(lam / 2)
    dH = float(lam / 4)
    Emax = 2 * E0 * abs(math.sin(2 * math.pi * float(r)))
    Hmax = 2 * E0 / ETA0 * abs(math.cos(2 * math.pi * float(r))) * 1e3      # mA/m
    Js = 2 * E0 / ETA0 * 1e3                                                   # mA/m
    Si = (1 + h * h) * E0 ** 2 / (2 * ETA0)                                    # W/m^2
    P = 2 * Si / C0 * 1e9                                                      # nPa
    F = P * A                                                                  # nN

    # ------------------------------------------------------------------ display of the incident field
    pm = "+" if s1 > 0 else "-"
    arg = r"\omega t " + pm + r" \beta " + w
    lead = "-" if sig < 0 else ""
    if circ:
        E_tex = (f"{lead}{E0}\\left[\\cos({arg})\\,{hat(a)} {'+' if h > 0 else '-'} "
                 f"\\sin({arg})\\,{hat(b)}\\right]")
    else:
        E_tex = f"{lead}{E0}\\cos({arg})\\,{hat(a)}"
    region = w + (r" \gt 0" if s1 > 0 else r" \lt 0")
    cond = w + (r" \lt 0" if s1 > 0 else r" \gt 0")
    ui = sh(w, -s1)
    ur = sh(w, s1)
    nhat = sh(w, s1)
    wp = s1 * dp_f

    # ------------------------------------------------------------------ hands (recipe of Lecture 24 section 4)
    inc_right = (-s1 * h * e) > 0
    hand_i = "right" if inc_right else "left"
    hand_r = "left" if inc_right else "right"
    e0_dir = sh(a, sig)                       # E_i(0, 0) direction
    eq_dir = sh(b, sig * h) if circ else ""   # E_i(0, T/4) direction
    cross_dir = sh(w, h * e) if circ else ""  # direction of a x (h b) = h e w_hat
    dot_sign = -s1 * h * e

    # ------------------------------------------------------------------ (c) reflected wave MC
    if circ:
        refl = []
        for hi in ("right", "left"):
            for hr in ("left", "right"):
                ok = hi == hand_i and hr == hand_r
                if ok:
                    fb = "Correct."
                elif hi == hand_i:
                    fb = (f"Your incident hand is correct, but the reflection flips it: at the mirror $\\mathbf{{E}}_r = -\\mathbf{{E}}_i$, "
                          f"so $\\mathbf{{E}}$ keeps turning the same way there, while the thumb must now point along ${ur}$.")
                elif hr != hi:
                    fb = (f"The flip is correct, but recheck the incident hand:thumb along $\\hat u_i = {ui}$; "
                          f"$\\mathbf{{E}}_i$ at the mirror points along ${e0_dir}$ at $t = 0$ and ${eq_dir}$ a quarter period later.")
                else:
                    fb = (f"Two slips. Thumb along $\\hat u_i = {ui}$ for the incident wave, and a perfect mirror reverses the hand "
                          r"(same rotation at the surface, opposite travel).")
                refl.append({"id": hi + "-" + hr, "correct": "true" if ok else "false",
                             "text": f"incident {hi}-circular; reflected {hr}-circular", "feedback": fb})
    else:
        refl = [
            {"id": "lin-inv", "correct": "true", "feedback": "Correct.",
             "text": f"linearly polarized along ${hat(a)}$, with $\\mathbf{{E}}_r = -\\mathbf{{E}}_i$ on the mirror at every instant"},
            {"id": "lin-same", "correct": "false",
             "feedback": r"That would double the tangential $\mathbf{E}$ on the conductor. A perfect conductor needs $\mathbf{E}_i + \mathbf{E}_r = 0$ there: $\Gamma = -1$.",
             "text": f"linearly polarized along ${hat(a)}$, with $\\mathbf{{E}}_r = +\\mathbf{{E}}_i$ on the mirror at every instant"},
            {"id": "lin-rot", "correct": "false",
             "feedback": f"Each tangential component reflects by itself with $\\Gamma = -1$, so $\\mathbf{{E}}_r$ stays along ${hat(a)}$. It is $\\mathbf{{H}}$ that lies along ${hat(b)}$.",
             "text": f"linearly polarized along ${hat(b)}$: the reflection turns the polarization by $90^\\circ$"},
            {"id": "circ", "correct": "false",
             "feedback": r"$\Gamma = -1$ is real: no $90^\circ$ phase is added, so a linear wave stays linear.",
             "text": r"circularly polarized: the mirror current adds a component $90^\circ$ out of phase"},
            {"id": "none", "correct": "false",
             "feedback": r"A perfect conductor has $\tau = 0$ and $\lvert\Gamma\rvert = 1$: the whole wave is reflected.",
             "text": r"there is no reflected wave: the conductor absorbs the incident wave"},
        ]

    # ------------------------------------------------------------------ (d) J_s direction at t1
    if circ:
        k = random.choice([0, 1, 2, 3])
    else:
        k = random.choice([0, 2])
    t1_tex = ["0", "T/4", "T/2", "3T/4"][k]
    # J_s(t) = (2 sig E0/eta0)[a cos wt + h b sin wt]; at wt = k pi/2
    jc = [(a, sig), (b, sig * h), (a, -sig), (b, -sig * h)][k]
    jax, jsg = jc
    oth = b if jax == a else a
    # H(0, t1) is along u_i x J_s direction (H_tot(0) = 2 H_i(0) = u_i x J_s/1 ... same orientation as u_i x E_i)
    hax = oth
    hsg = (-s1) * lc(w, jax, oth) * jsg       # u_i x (jsg jax_hat) = -s1 jsg (w x jax) = -s1 jsg lc(w, jax, oth) oth_hat
    js = []
    for ax, sg in ((a, 1), (a, -1), (b, 1), (b, -1)):
        if ax == jax and sg == jsg:
            fb = "Correct."
            cid = "J"
        elif ax == jax:
            fb = (f"That is $-\\hat n\\times\\mathbf{{H}}$. The normal points out of the conductor into the air, $\\hat n = {nhat}$; "
                  "equivalently $\\mathbf{J}_s = 2\\mathbf{E}_i(0,t)/\\eta_0$ runs along the incident $\\mathbf{E}$ at the surface.")
            cid = "-J"
        elif sg == hsg:
            fb = (f"${sh(hax, hsg)}$ is the direction of $\\mathbf{{H}}$ on the mirror at that instant. "
                  r"$\mathbf{J}_s = \hat n\times\mathbf{H}$ is perpendicular to $\mathbf{H}$.")
            cid = "H"
        else:
            if circ:
                fb = (r"$\mathbf{J}_s = 2\mathbf{E}_i(0,t)/\eta_0$ turns with the incident field; it points along "
                      f"${sh(ax, sg)}$ a quarter period away from $t = {t1_tex}$, not at that instant.")
            else:
                fb = (f"The incident $\\mathbf{{E}}$ lies along ${hat(a)}$, so $\\mathbf{{J}}_s = 2\\mathbf{{E}}_i(0,t)/\\eta_0$ does too; "
                      f"${hat(b)}$ is the axis of $\\mathbf{{H}}$, to which $\\mathbf{{J}}_s$ is perpendicular.")
            cid = "-H"
        js.append({"id": cid, "ax": ax, "sg": sg, "correct": "true" if cid == "J" else "false",
                   "text": f"${sh(ax, sg)}$", "feedback": fb})
    js.append({"id": "zero", "ax": "", "sg": 0, "correct": "false", "text": r"$\mathbf{J}_s = 0$ at that instant",
               "feedback": r"Tangential $\mathbf{H}$ does not vanish on a perfect conductor: the reflected $\mathbf{H}$ adds to the incident one, and $\mathbf{J}_s = \hat n\times\mathbf{H}$ carries the jump to zero inside."})

    # ------------------------------------------------------------------ (f) checkbox pool
    facts = [
        ("avgS0", True, r"At every point in front of the mirror the time-average Poynting vector $\langle\mathbf{S}\rangle$ is zero.",
         r"Incident and reflected waves carry equal average power in opposite directions; equivalently $\tilde{\mathbf{E}}$ and $\tilde{\mathbf{H}}$ are $90^\circ$ apart."),
        ("Enull_half", True, r"Adjacent points where $\mathbf{E}$ is zero at every instant are $\lambda/2$ apart, with $\lambda$ the incident wavelength.",
         r"$\lvert\tilde{\mathbf{E}}\rvert \propto \lvert\sin\beta D\rvert$ vanishes at $\beta D = m\pi$, i.e. every $\lambda/2$."),
        ("Etan0", True, r"The total tangential $\mathbf{E}$ on the mirror surface is zero at every instant.",
         r"That is the perfect-conductor boundary condition, and it is what fixes $\Gamma = -1$."),
        ("circ_flip", True, r"A circularly polarized wave of either hand, reflected at normal incidence by a perfect mirror, comes back with the opposite hand.",
         r"$\mathbf{E}_r = -\mathbf{E}_i$ on the mirror keeps the sense of rotation there, while the direction of travel reverses."),
        ("absorber_half", True, r"A perfectly absorbing (black) surface placed in the same incident wave would feel half the mirror's time-average radiation pressure.",
         r"An absorber stops the wave's momentum flux, $\langle S_i\rangle/c$; a mirror reverses it, $2\langle S_i\rangle/c$."),
        ("fixed_nodes", True, r"The pattern in front of the mirror does not travel: the points where $\mathbf{E}$ is always zero stay where they are.",
         r"Every component of the total field is a function of position times a function of time."),
        ("S_inst0", circ, r"In this problem, the instantaneous Poynting vector $\mathbf{E}\times\mathbf{H}$ is zero at every point and every instant.",
         (r"For this circular wave the total $\mathbf{E}$ and $\mathbf{H}$ are parallel at every point and instant, so $\mathbf{E}\times\mathbf{H} = 0$." if circ else
          r"For a linear standing wave only the average vanishes: $\mathbf{S}\propto\sin(2\beta D)\sin(2\omega t)$ sloshes energy back and forth.")),
        ("P_const", circ, r"In this problem, the radiation pressure on the mirror has the same value at every instant.",
         (r"$P(t) = \tfrac12\mu_0\lvert\mathbf{H}(0,t)\rvert^2$ and $\lvert\mathbf{H}(0,t)\rvert = 2E_0/\eta_0$ at all times for this circular wave." if circ else
          r"$P(t) = \tfrac12\mu_0\lvert\mathbf{H}(0,t)\rvert^2\propto\cos^2\omega t$: it pulses between zero and twice its mean.")),
        ("w0_Enode", not circ, r"In this problem, at a point where $\mathbf{E}$ is zero at every instant, the energy density still drops to zero at some instants.",
         (r"There $w = \tfrac12\mu_0H^2$ with $H\propto\cos\omega t$, which vanishes at $\omega t = \pi/2, 3\pi/2$." if not circ else
          r"For this circular wave $\lvert\mathbf{H}\rvert$ is constant in time at every point, so $w = \tfrac12\mu_0\lvert\mathbf{H}\rvert^2$ never vanishes at an $\mathbf{E}$ node.")),
        ("E0_Hmax", not circ, r"In this problem, at a point where neither field has a node, $\mathbf{E}$ is zero at the instants when $\lvert\mathbf{H}\rvert$ is largest.",
         (r"$\mathbf{E}\propto\sin\omega t$ and $\mathbf{H}\propto\cos\omega t$: a quarter period apart." if not circ else
          r"For this circular wave $\lvert\mathbf{E}\rvert = 2E_0\lvert\sin\beta D\rvert$ at every instant; it is never zero away from a node.")),
        ("Htan0", False, r"Like the tangential $\mathbf{E}$, the total tangential $\mathbf{H}$ on the mirror surface is zero at every instant.",
         r"The reflected $\mathbf{H}$ keeps the incident sign, so tangential $\mathbf{H}$ doubles to $2E_0/\eta_0$ at the surface; $\mathbf{J}_s$ carries the jump."),
        ("nodes_same", False, r"The points where $\mathbf{E}$ is always zero are also points where $\mathbf{H}$ is always zero.",
         r"$\mathbf{H}$ nodes lie a quarter wavelength from the $\mathbf{E}$ nodes; at an $\mathbf{E}$ node $\mathbf{H}$ is largest."),
        ("avgP0", False, r"Because $\langle\mathbf{S}\rangle = 0$ in front of the mirror, the time-average radiation pressure on it is zero.",
         r"The pressure comes from the incident flux, $2\langle S_i\rangle/c$; the zero net flux only says the reflected wave carries the power back."),
        ("Js_alongH", False, r"The surface current on the mirror flows parallel to the total $\mathbf{H}$ at the surface.",
         r"$\mathbf{J}_s = \hat n\times\mathbf{H}$ is perpendicular to $\mathbf{H}$; it runs along the incident $\mathbf{E}$ at the surface."),
        ("avgw0", False, r"The time-average energy density is zero at the points where $\mathbf{E}$ is always zero.",
         r"There the energy is all magnetic, and $\mathbf{H}$ is largest; $\langle w\rangle$ is the same at every point."),
        ("Js_eqH", False, r"The largest $\lvert\mathbf{J}_s\rvert$ on the mirror equals the largest $\lvert\mathbf{H}\rvert$ of the incident wave alone.",
         r"The total tangential $\mathbf{H}$ on the surface is twice the incident one, so $\lvert\mathbf{J}_s\rvert_{\max} = 2E_0/\eta_0$."),
        ("Emax_mirror", False, r"The amplitude of the total $\mathbf{E}$ is largest right at the mirror surface.",
         r"The mirror is an $\mathbf{E}$ node; the first $\mathbf{E}$ maximum is $\lambda/4$ in front of it."),
    ]
    facts_list = [{"id": fid, "correct": "true" if t else "false", "text": txt,
                   "feedback": ("True: " if t else "False: ") + why, "verdict": "true" if t else "false", "why": why}
                  for fid, t, txt, why in facts]
    random.shuffle(facts_list)

    # ------------------------------------------------------------------ answer-panel strings
    cE = 2 * sig * s1 * E0                    # E = cE sin(beta w)[...]
    cH = -2 * s1 * e * sig * E0 / ETA0 * 1e3  # H = cH cos(beta w)[...] (mA/m)
    if circ:
        Etot_tex = (f"{num(cE)}\\sin(\\beta {w})\\left[-\\sin(\\omega t)\\,{hat(a)} {'+' if h > 0 else '-'} "
                    f"\\cos(\\omega t)\\,{hat(b)}\\right]\\ \\mathrm{{V/m}}")
        Htot_tex = (f"{num(cH)}\\cos(\\beta {w})\\left[\\cos(\\omega t)\\,{hat(b)} {'-' if h > 0 else '+'} "
                    f"\\sin(\\omega t)\\,{hat(a)}\\right]\\ \\mathrm{{mA/m}}")
        Ji_tex = (f"{num(2 * sig * E0 / ETA0 * 1e3)}\\left[\\cos(\\omega t)\\,{hat(a)} {'+' if h > 0 else '-'} "
                  f"\\sin(\\omega t)\\,{hat(b)}\\right]\\ \\mathrm{{mA/m}}")
        Er_tex = (f"{'' if sig < 0 else '-'}{E0}\\left[\\cos(\\omega t {'-' if s1 > 0 else '+'} \\beta {w})\\,{hat(a)} "
                  f"{'+' if h > 0 else '-'} \\sin(\\omega t {'-' if s1 > 0 else '+'} \\beta {w})\\,{hat(b)}\\right]")
    else:
        Etot_tex = f"{num(-cE)}\\sin(\\beta {w})\\sin(\\omega t)\\,{hat(a)}\\ \\mathrm{{V/m}}"
        Htot_tex = f"{num(cH)}\\cos(\\beta {w})\\cos(\\omega t)\\,{hat(b)}\\ \\mathrm{{mA/m}}"
        Ji_tex = f"{num(2 * sig * E0 / ETA0 * 1e3)}\\cos(\\omega t)\\,{hat(a)}\\ \\mathrm{{mA/m}}"
        Er_tex = f"{'' if sig < 0 else '-'}{E0}\\cos(\\omega t {'-' if s1 > 0 else '+'} \\beta {w})\\,{hat(a)}"
    Er_tex += r"\ \ \mathrm{V/m}"

    p = data["params"]
    p.update({
        "w": w, "a": a, "b": b, "s1": s1, "sig": sig, "h": h, "circ": circ, "lin": not circ,
        "f_Hz": f, "f_tex": f_tex, "E0": E0, "A_m2": A, "E_tex": E_tex, "region": region, "cond": cond,
        "ui": ui, "ur": ur, "nhat": nhat, "lam_str": num(float(lam)), "beta_str": num(beta),
        "dp_str": num(dp_f, 6), "wp_str": num(wp, 6), "r_tex": f"{r.numerator}/{r.denominator}" if r.denominator != 1 else str(r.numerator),
        "phase_str": num(360 * float(r)), "sin_str": num(abs(math.sin(2 * math.pi * float(r)))),
        "cos_str": num(abs(math.cos(2 * math.pi * float(r)))),
        "dE_str": num(dE), "dH_str": num(dH), "Emax_str": num(Emax), "Hmax_str": num(Hmax),
        "Hi_str": num(E0 / ETA0 * 1e3), "Js_str": num(Js), "Si_str": num(Si), "P_str": num(P), "F_str": num(F),
        "hfac": r"2\times " if circ else "",
        "t1_tex": t1_tex, "J_dir": sh(jax, jsg), "H_dir": sh(hax, hsg),
        "hand_i": hand_i, "hand_r": hand_r, "e0_dir": e0_dir, "eq_dir": eq_dir, "cross_dir": cross_dir,
        "dot_str": "positive" if dot_sign > 0 else "negative",
        "Etot_tex": Etot_tex, "Htot_tex": Htot_tex, "Ji_tex": Ji_tex, "Er_tex": Er_tex,
        "c_prompt": ("Name the hand of the incident wave and of the reflected wave." if circ else
                     "Which statement describes the reflected wave?"),
        "refl_list": refl, "js_list": js, "facts_list": facts_list,
    })
    ca = data["correct_answers"]
    ca["dE"] = dE
    ca["dH"] = dH
    ca["Emax"] = float(Emax)
    ca["Hmax"] = float(Hmax)
    ca["Js"] = float(Js)
    ca["P"] = float(P)
    ca["F"] = float(F)
