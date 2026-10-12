"""Lecture 18 concept question: reading a plane-wave pulse E = sigma*E0 e_hat g(s), s = (numerator)/(den).

The numerator is written either in "time form", t -/+ p*u - t0 (t in ns, u in m, p = slowness in ns/m), or in
"space form", u -/+ v*t - u0 (v in m/ns), possibly multiplied by -1 as a whole ("flip").  Direction of travel
along +u when the t and u terms have opposite signs, -u when they have the same sign; speed = |coef_t / coef_u|.
    v = c/sqrt(eps_r) (mu_r = 1)  =>  eps_r = (c/v)^2,   eta = sqrt(mu0/(eps_r eps0)) = mu0 v,
    H = (1/eta) u_travel x E   (|H| at the peak = E0/eta),
    peak (g = 1) where the numerator vanishes: time form u_pk = d (t1 - t0)/p; space form u_pk = u0 + d v t1.
Shapes: Gaussian exp(-s^2), triangle 1-|s| (|s| < 1), one-sided exponential e^{-s} (s >= 0); all peak at s = 0.
"""
import math
import random

C0 = 299792458.0
MU0 = 4e-7 * math.pi
AXES = ["x", "y", "z"]
SHAPES = {
    "gauss": (r"g(s) = e^{-s^{2}}", "a Gaussian pulse"),
    "tri": (r"g(s) = 1-\lvert s\rvert\ \text{for}\ \lvert s\rvert\lt 1,\ \ g(s) = 0\ \text{otherwise}", "a triangular pulse"),
    "exp": (r"g(s) = e^{-s}\ \text{for}\ s\ge 0,\ \ g(s) = 0\ \text{for}\ s\lt 0", "a one-sided exponential pulse"),
}


def num(x, sig=4):
    s = f"{x:.{sig}g}"
    if "e" in s:
        m, e = s.split("e")
        return f"{m}\\times10^{{{int(e)}}}"
    return s


def coef_str(x):
    return f"{x:g}"


def lin_tex(terms):
    """terms: list of (symbol or None, coefficient) in display order -> 't - 5\\,y - 20'."""
    out = ""
    for sym, k in terms:
        if k == 0:
            continue
        mag = abs(k)
        if sym is None:
            body = coef_str(mag)
        elif mag == 1:
            body = sym
        else:
            body = coef_str(mag) + r"\," + sym
        if not out:
            out = ("-" if k < 0 else "") + body
        else:
            out += (" - " if k < 0 else " + ") + body
    return out


def generate(data):
    u = random.choice(AXES)
    e = random.choice([ax for ax in AXES if ax != u])
    sigE = random.choice([1, -1])
    E0 = random.choice([2, 3, 4, 5, 6, 8, 10, 12, 15])         # V/m
    d = random.choice([1, -1])                                  # travel toward d*u_hat
    form = random.choice(["time", "space"])
    flip = random.random() < 0.4
    shape = random.choice(list(SHAPES))

    if form == "time":
        p = random.choice([4, 5, 6, 8, 10])                     # ns/m
        v = 1.0 / p                                             # m/ns
        den = random.choice([2, 3, 4, 5])                       # ns
        while True:
            t0 = random.choice([-10, -5, 0, 5, 10, 15, 20])
            u_pk = random.choice([-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6])
            t1 = t0 + d * p * u_pk
            if 5 <= t1 <= 80:
                break
        # numerator = t - d p u - t0 ; coefficients (c_t [1/ns], c_u [ns/m per m -> ns/m], c_0 [ns])
        ct, cu, c0 = 1, -d * p, -t0
        den_unit = r"\mathrm{ns}"
    else:
        v = random.choice([0.10, 0.12, 0.15, 0.16, 0.20, 0.24, 0.25])   # m/ns
        p = None
        den = random.choice([0.4, 0.5, 0.6, 0.8, 1.0])          # m
        while True:
            u0 = random.choice([-3, -2, -1, 1, 2, 3])
            t1 = random.choice([10, 15, 20, 25, 30, 40, 50, 60])
            u_pk = u0 + d * v * t1
            if 0.5 <= abs(u_pk) <= 15:
                break
        ct, cu, c0 = -d * v, 1, -u0
        den_unit = r"\mathrm{m}"
    if flip:
        ct, cu, c0 = -ct, -cu, -c0
    # display order: lead with the term that the unflipped form leads with
    if form == "time":
        order = [("t", ct), (u, cu), (None, c0)] if not flip else [(u, cu), ("t", ct), (None, c0)]
    else:
        order = [(u, cu), ("t", ct), (None, c0)] if not flip else [("t", ct), (u, cu), (None, c0)]
    num_tex = lin_tex(order)
    den_str = coef_str(den)
    s_tex = r"\dfrac{" + num_tex + "}{" + den_str + "}"

    v_ms = v * 1e9
    eps_r = (C0 / v_ms) ** 2
    eta = MU0 * v_ms
    H_mA = E0 / eta * 1e3
    eta0 = MU0 * C0

    # ---------------------------------------------------------------- directions
    def unit(ax, sg):
        return ("+" if sg > 0 else "-") + ax

    def uhat(ax, sg):
        return ("+" if sg > 0 else "-") + "\\hat{" + ax + "}"
    idx = {"x": 0, "y": 1, "z": 2}

    def cross_axes(a, sa, b, sb):
        """(sa a_hat) x (sb b_hat) -> (axis, sign)."""
        ia, ib = idx[a], idx[b]
        ic = 3 - ia - ib
        sgn = 1 if (ib - ia) % 3 == 1 else -1
        return AXES[ic], sa * sb * sgn

    h_ax, h_sg = cross_axes(u, d, e, sigE)          # H at the peak along h_sg * h_ax
    trav = unit(u, d)

    dir_choices = [
        {"id": "plus", "text": f"toward $+{u}$", "correct": "true" if d > 0 else "false"},
        {"id": "minus", "text": f"toward $-{u}$", "correct": "false" if d > 0 else "true"},
        {"id": "alongE", "text": f"toward ${unit(e, sigE)}$, the direction of $\\mathbf{{E}}$ at the peak", "correct": "false"},
        {"id": "still", "text": "nowhere: the pulse stays where it is and only changes in time", "correct": "false"},
    ]
    sign_rule = ("the $t$ and ${u}$ terms have opposite signs" if d > 0 else "the $t$ and ${u}$ terms have the same sign").replace("{u}", u)
    for ch in dir_choices:
        if ch["correct"] == "true":
            ch["feedback"] = f"Right: {sign_rule}, so the pattern moves toward ${trav}$."
        elif ch["id"] in ("plus", "minus"):
            ch["feedback"] = (f"Check the relative sign: {sign_rule}, so a fixed value of $s$ moves toward ${trav}$. "
                              "The order of the terms and an overall minus sign do not matter.")
        elif ch["id"] == "alongE":
            ch["feedback"] = ("The direction of $\\mathbf{E}$ is the polarization, not the direction of travel; "
                              "a plane TEM wave travels perpendicular to $\\mathbf{E}$.")
        else:
            ch["feedback"] = ("$s$ mixes $t$ and " + f"${u}$" + ": a fixed value of $s$ (say the peak, $s = 0$) sits at a "
                              "position that changes with $t$, so the pattern travels.")
    random.shuffle(dir_choices)

    trav_hat, E_hat, h_hat = uhat(u, d), uhat(e, sigE), uhat(h_ax, h_sg)
    hdir_choices = [
        {"id": "ok", "text": f"${h_hat}$", "correct": "true"},
        {"id": "rev", "text": f"${uhat(h_ax, -h_sg)}$", "correct": "false"},
        {"id": "travel", "text": "along the direction of travel", "correct": "false"},
        {"id": "alongE", "text": f"${E_hat}$ (parallel to $\\mathbf{{E}}$)", "correct": "false"},
        {"id": "zero", "text": r"nowhere: $\mathbf{H} = \mathbf{0}$ at the peak, because $\mathbf{H}$ lags $\mathbf{E}$ by a quarter period", "correct": "false"},
    ]
    cross_txt = (f"$\\mathbf{{H}} = \\tfrac{{1}}{{\\eta}}\\,\\hat{{u}}\\times\\mathbf{{E}}$ with $\\hat{{u}} = {trav_hat}$ and $\\mathbf{{E}}$ along ${E_hat}$: "
                 f"$({trav_hat})\\times({E_hat}) = {h_hat}$")
    for ch in hdir_choices:
        if ch["id"] == "ok":
            ch["feedback"] = "Right: " + cross_txt + "."
        elif ch["id"] == "rev":
            ch["feedback"] = ("Reversed: " + cross_txt + ". Check that $\\mathbf{E}\\times\\mathbf{H}$ points along the travel; "
                              "this choice makes it point backwards.")
        elif ch["id"] == "travel":
            ch["feedback"] = "A plane TEM wave has no field component along its direction of travel."
        elif ch["id"] == "alongE":
            ch["feedback"] = "$\\mathbf{H}$ is perpendicular to $\\mathbf{E}$ in a plane wave."
        else:
            ch["feedback"] = ("In a single travelling wave in a lossless medium $\\mathbf{H} = \\hat{u}\\times\\mathbf{E}/\\eta$ at every point and instant: "
                              "$\\mathbf{E}$ and $\\mathbf{H}$ peak together.")
    random.shuffle(hdir_choices)

    # ---------------------------------------------------------------- checkbox pool
    a_pt, b_pt = random.sample([-4, -2, 0, 1, 3, 5], 2)
    order_true = (a_pt < b_pt) if d > 0 else (a_pt > b_pt)
    facts = [
        ("perp", True, r"At every point, $\mathbf{E}\perp\mathbf{H}$, and both are perpendicular to the direction of travel.",
         r"Uniform plane TEM wave: $\mathbf{H} = \hat{u}\times\mathbf{E}/\eta$ with $\mathbf{E}\perp\hat{u}$."),
        ("ratio", True, r"For this wave, $\lvert\mathbf{E}\rvert = (\eta_0/\sqrt{\epsilon_r})\,\lvert\mathbf{H}\rvert$ at every point and instant.",
         r"A single travelling wave has $\lvert\mathbf{E}\rvert = \eta\lvert\mathbf{H}\rvert$ everywhere, with $\eta = \sqrt{\mu_0/\epsilon} = \eta_0/\sqrt{\epsilon_r}$."),
        ("same_kind", True, r"For any waveforms $f$ and $h$, both $f(t-z/v)$ and $h(z-vt)$ describe waves travelling toward $+z$ at speed $v$.",
         r"$h(z-vt) = h\big(-v(t-z/v)\big)$ is also a function of $t-z/v$."),
        ("superpose", True, r"The sum of a pulse travelling toward $+z$ and a different pulse travelling toward $-z$ also satisfies the wave equation.",
         r"The wave equation is linear, and d'Alembert's general solution is exactly such a sum."),
        ("shape", True, r"The pulse keeps its shape as it travels, because every frequency component moves at the same speed in this medium.",
         r"With constant $\mu$ and $\epsilon$ (lossless, no dispersion) $v = 1/\sqrt{\mu\epsilon}$ is the same for all frequencies."),
        ("B_over_E", True, r"For this wave, $\lvert\mathbf{B}\rvert = \lvert\mathbf{E}\rvert/v$ at every point.",
         r"$\lvert\mathbf{B}\rvert = \mu_0\lvert\mathbf{H}\rvert = \mu_0\lvert\mathbf{E}\rvert/\eta$ and $\eta = \mu_0 v$."),
        ("vacuum_H", True, r"In vacuum, a wave with the same peak $\lvert\mathbf{E}\rvert$ would have a smaller peak $\lvert\mathbf{H}\rvert$ than this one.",
         r"$\lvert\mathbf{H}\rvert = \lvert\mathbf{E}\rvert/\eta$ and $\eta_0 \gt \eta_0/\sqrt{\epsilon_r}$ for $\epsilon_r \gt 1$."),
        ("quadrature", False, r"At a fixed point, $\mathbf{H}$ reaches its peak a quarter period after $\mathbf{E}$ does.",
         r"In a single travelling wave in a lossless medium $\mathbf{E}$ and $\mathbf{H}$ are in step: $\mathbf{H} = \hat{u}\times\mathbf{E}/\eta$ at every instant."),
        ("longitudinal", False, r"A field $\mathbf{E} = \hat{z}\,f(t-z/v)$ can be a plane wave travelling along $z$ in a charge-free region.",
         r"Its divergence $\partial E_z/\partial z = -f'/v$ is not zero, which violates Gauss's law with $\rho = 0$."),
        ("eta0", False, r"For a plane wave in any lossless medium, $\lvert\mathbf{E}\rvert/\lvert\mathbf{H}\rvert = 377\ \Omega$.",
         r"$377\ \Omega$ is $\eta_0$, the vacuum value; in a medium $\eta = \eta_0\sqrt{\mu_r/\epsilon_r}$."),
        ("slowness", False, r"In $f(t-z/v)$, the coefficient of $z$ is the speed of the wave.",
         r"The coefficient is $1/v$, the slowness in s/m; the speed is its reciprocal."),
        ("eta_up", False, r"With $\mu_r = 1$, a larger $\epsilon_r$ gives a larger intrinsic impedance $\eta$.",
         r"$\eta = \eta_0/\sqrt{\epsilon_r}$ decreases as $\epsilon_r$ grows."),
        ("decay", False, r"As it travels through this lossless medium, the pulse's peak value slowly decreases.",
         r"A uniform plane wave does not spread out; in a lossless medium it keeps its amplitude and shape."),
        ("amp_speed", False, r"Doubling the amplitude of $\mathbf{E}$ would make the pulse travel faster.",
         r"The speed $1/\sqrt{\mu\epsilon}$ depends only on the medium, not on the amplitude (the equations are linear)."),
        ("standing_ratio", False, r"For the sum of a wave travelling toward $+z$ and one travelling toward $-z$, $E_x/H_y = \eta$ at every point.",
         r"With $E_x = Af(t-z/v)+Bg(t+z/v)$, Faraday's law gives $H_y = [Af-Bg]/\eta$, so $E_x/H_y = \eta\,(Af+Bg)/(Af-Bg)$, which equals $\eta$ only if one of the two waves is absent."),
        ("order", order_true, f"The peak of this pulse reaches ${u} = {a_pt}\\ \\mathrm{{m}}$ before it reaches ${u} = {b_pt}\\ \\mathrm{{m}}$.",
         f"The pulse travels toward ${trav}$, so it reaches the point with the {'smaller' if d > 0 else 'larger'} ${u}$ first"
         + (": true." if order_true else ": here that is $" + u + f" = {b_pt}$ m, so the statement is false.")),
    ]
    fact_list = [{"id": fid, "correct": "true" if t else "false", "text": txt,
                  "feedback": ("True: " if t else "False: ") + why, "verdict": "true" if t else "false", "why": why}
                 for fid, t, txt, why in facts]
    random.shuffle(fact_list)

    # ---------------------------------------------------------------- answer-panel strings
    # unflipped numerator, for the explanation
    if form == "time":
        base = [("t", 1), (u, -d * p), (None, -t0)]
    else:
        base = [(u, 1), ("t", -d * v), (None, -u0)]
    base_tex = lin_tex(base)
    rel = "opposite signs" if d > 0 else "the same sign"
    if flip:
        read = (f"The numerator is $-({base_tex})$. An overall minus sign changes neither where $s = 0$ is nor how that point moves, "
                f"so read $({base_tex})$: its $t$ and ${u}$ terms have {rel}, so the wave travels toward ${trav}$. ")
    else:
        read = f"In the numerator ${base_tex}$ the $t$ and ${u}$ terms have {rel}, so the wave travels toward ${trav}$. "
    if form == "time":
        read += (f"The coefficient of ${u}$ is the slowness $1/v = {p}\\ \\mathrm{{ns/m}}$, not the speed: "
                 f"$v = 1/({p}\\ \\mathrm{{ns/m}}) = {num(v)}\\ \\mathrm{{m/ns}}$.")
        t0_part = "" if t0 == 0 else (f" - {t0}" if t0 > 0 else f" + {abs(t0)}")
        tpart = f"(t{t0_part})" if t0 else "t"
        t1part = f"({t1}{t0_part})" if t0 else f"{t1}"
        peak = (f"The peak ($s = 0$) is where ${base_tex} = 0$, i.e. ${u} = {'' if d > 0 else '-'}{tpart}/{p}$ with $t$ in ns. "
                f"At $t = {t1}\\ \\mathrm{{ns}}$: ${u} = {'' if d > 0 else '-'}{t1part}/{p} = {num(u_pk)}\\ \\mathrm{{m}}$.")
    else:
        read += f"The ratio of the coefficients of $t$ and ${u}$ is the speed: $v = {coef_str(v)}\\ \\mathrm{{m/ns}}$."
        peak = (f"The peak ($s = 0$) is where ${base_tex} = 0$, i.e. ${u} = {u0} {'+' if d > 0 else '-'} {coef_str(v)}\\,t$ with $t$ in ns. "
                f"At $t = {t1}\\ \\mathrm{{ns}}$: ${u} = {u0} {'+' if d > 0 else '-'} ({coef_str(v)})({t1}) = {num(u_pk)}\\ \\mathrm{{m}}$.")

    data["params"].update({
        "u": u, "e": e, "sigE": sigE, "E0": E0, "d": d, "form": form, "flip": flip, "shape": shape,
        "ct": ct, "cu": cu, "c0": c0, "den": den, "t1": t1,
        "E_amp_tex": ("" if sigE > 0 else "-") + f"{E0}\\,\\hat{{{e}}}", "s_tex": s_tex, "num_tex": num_tex,
        "den_str": den_str, "den_unit": den_unit, "g_tex": SHAPES[shape][0], "g_name": SHAPES[shape][1],
        "trav": trav, "trav_hat": trav_hat, "h_hat": h_hat, "E_hat": E_hat, "h_ax": h_ax, "h_sg": h_sg,
        "v_str": num(v), "v8_str": num(v * 10), "eps_str": num(eps_r), "eta_str": num(eta), "eta0_str": num(eta0),
        "H_str": num(H_mA), "upk_str": num(u_pk), "read_txt": read, "peak_txt": peak,
        "dir_choices": dir_choices, "hdir_choices": hdir_choices, "facts": fact_list,
    })
    data["correct_answers"]["v8"] = float(v * 10)            # units of 1e8 m/s
    data["correct_answers"]["epsr"] = float(eps_r)
    data["correct_answers"]["Hpk"] = float(H_mA)             # mA/m
    data["correct_answers"]["upk"] = float(u_pk)             # m
