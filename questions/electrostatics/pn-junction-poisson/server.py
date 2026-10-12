"""Lecture 7 (the junction example of Lectures 3 and 7): abrupt silicon pn junction, depletion approximation.

Metallurgical junction at x = 0; the p side is x < 0 or x > 0 (random). Depletion layers: p side, width Wp,
rho = -e Na; n side, width Wn, rho = +e Nd. Neutral regions: rho = 0 and E = 0. eps = 11.7 eps0.
    neutrality (E must return to zero):   Na Wp = Nd Wn
    dE_x/dx = rho/eps: E_x is linear in each layer, zero at the outer edges, largest in magnitude at x = 0:
        |E|max = e Na Wp / eps = e Nd Wn / eps
    E points from the n side (+ donor ions) to the p side (- acceptor ions)
    V'' = -rho/eps: V is a parabola in each layer;  V_bi = V(n edge) - V(p edge) = area under |E_x|
        = |E|max (Wp + Wn) / 2 = e (Na Wp^2 + Nd Wn^2) / (2 eps)       (Lecture 7's V_21)
The given width is the depletion-approximation width for the equilibrium built-in potential
(kT/e) ln(Na Nd / ni^2) at 300 K (ni = 1e10 cm^-3), rounded to 3 s.f., so every number is realistic; the student
works only from the given width and the dopings.
"""
import math
import random

EPS0 = 8.854e-12
E_CH = 1.602e-19
ER_SI = 11.7
EPS = ER_SI * EPS0
DOPINGS = [1e15, 2e15, 3e15, 5e15, 1e16, 2e16, 3e16, 5e16, 1e17]      # cm^-3


def fmt(x, sig=4):
    if abs(x) < 1e-12:
        return "0"
    s = f"{x:.{sig}g}"
    if "e" in s:
        m, e = s.split("e")
        return f"{m}\\times10^{{{int(e)}}}"
    return s


def dop_tex(n, shift=0):
    """2e16 -> 2\\times10^{16}; 1e16 -> 10^{16}; shift adds to the exponent (cm^-3 -> m^-3: +6)."""
    e = int(math.floor(math.log10(n) + 1e-9))
    m = round(n / 10 ** e)
    return (f"{m}\\times" if m != 1 else "") + f"10^{{{e + shift}}}"


def round3(x):
    return float(f"{x:.3g}")


def generate(data):
    while True:
        Na, Nd = random.choice(DOPINGS), random.choice(DOPINGS)
        if max(Na, Nd) / min(Na, Nd) >= 2:
            break
    p_left = random.random() < 0.5
    given = random.choice(["p", "n"])

    Na_m, Nd_m = Na * 1e6, Nd * 1e6                                    # m^-3
    vbi_real = 0.02585 * math.log(Na * Nd / 1e20)                      # V (only used to make W realistic)
    W_real = math.sqrt(2 * EPS * vbi_real * (Na_m + Nd_m) / (E_CH * Na_m * Nd_m))
    if given == "p":
        Wp_nm = round3(W_real * Nd_m / (Na_m + Nd_m) * 1e9)
        Wn_nm = Wp_nm * Na / Nd
        W_other = Wn_nm
    else:
        Wn_nm = round3(W_real * Na_m / (Na_m + Nd_m) * 1e9)
        Wp_nm = Wn_nm * Nd / Na
        W_other = Wp_nm
    Wp, Wn = Wp_nm * 1e-9, Wn_nm * 1e-9
    Emax = E_CH * Na_m * Wp / EPS                                       # V/m
    Vbi = 0.5 * Emax * (Wp + Wn)                                        # V
    n_wider = Wn > Wp

    # geometry in this orientation
    if p_left:
        xp, xn = "-W_p", "+W_n"                                         # symbolic: numbers would give away (a)
        e_sign = -1                                                     # from n (x > 0) to p (x < 0)
        p_region, n_region = r"$x \lt 0$", r"$x \gt 0$"
        p_layer, n_layer = f"$-{fmt(Wp_nm)} \\lt x \\lt 0$", f"$0 \\lt x \\lt {fmt(Wn_nm)}$"
    else:
        xp, xn = "+W_p", "-W_n"
        e_sign = +1
        p_region, n_region = r"$x \gt 0$", r"$x \lt 0$"
        p_layer, n_layer = f"$0 \\lt x \\lt {fmt(Wp_nm)}$", f"$-{fmt(Wn_nm)} \\lt x \\lt 0$"

    # ---------------------------------------------------------------- (b) where is |E| largest
    loc = [
        {"key": "junction", "correct": "true", "text": r"at the metallurgical junction, $x = 0$",
         "feedback": r"Right: $E_x$ grows linearly from zero at each outer edge, so its magnitude peaks where the two layers meet."},
        {"key": "p_edge", "correct": "false",
         "text": f"at the outer edge of the p-side depletion layer, $x = {xp}$",
         "feedback": r"That is where $E_x = 0$: entering from the neutral p region, the field starts from zero and grows toward the junction."},
        {"key": "n_edge", "correct": "false",
         "text": f"at the outer edge of the n-side depletion layer, $x = {xn}$",
         "feedback": r"That is where $E_x = 0$: the field of the two charged layers cancels outside them."},
        {"key": "uniform", "correct": "false", "text": r"nowhere in particular: $|E|$ is the same everywhere in the depletion region",
         "feedback": r"A uniform field needs $\rho = 0$. In the depletion layers $dE_x/dx = \rho/\epsilon \neq 0$, so $|E|$ changes linearly."},
        {"key": "mid_wide", "correct": "false", "text": r"in the middle of the wider depletion layer",
         "feedback": r"$|E_x|$ keeps growing all the way to $x = 0$; nothing special happens in the middle of a layer."},
    ]
    random.shuffle(loc)

    # ---------------------------------------------------------------- (d) direction of E
    wrong_sign = (r"That points from the p side to the n side. Field lines start on the positive donor ions (n side) "
                  r"and end on the negative acceptor ions (p side).")
    right = r"Right: $\mathbf{E}$ points from the positive donor ions on the n side to the negative acceptor ions on the p side."
    dirs = [
        {"key": "plus", "correct": "true" if e_sign > 0 else "false",
         "text": r"along $+\hat{x}$ everywhere in the depletion region", "feedback": right if e_sign > 0 else wrong_sign},
        {"key": "minus", "correct": "true" if e_sign < 0 else "false",
         "text": r"along $-\hat{x}$ everywhere in the depletion region", "feedback": right if e_sign < 0 else wrong_sign},
        {"key": "toward", "correct": "false", "text": r"toward the junction plane $x = 0$ from both sides",
         "feedback": r"$E_x$ does not change sign at $x = 0$: it is continuous there (no surface charge) and largest in magnitude there."},
        {"key": "away", "correct": "false", "text": r"away from the junction plane $x = 0$ on both sides",
         "feedback": r"$E_x$ does not change sign at $x = 0$: it is continuous there (no surface charge) and largest in magnitude there."},
        {"key": "zero", "correct": "false", "text": r"nowhere: in equilibrium there is no electric field in the depletion region",
         "feedback": r"The uncovered ion charges produce a field; it is this built-in field that stops further diffusion. $\mathbf{E} = 0$ only in the neutral regions."},
    ]
    random.shuffle(dirs)

    # ---------------------------------------------------------------- (e) Poisson / Laplace statements
    pool = [
        ("poisson", True, r"Inside each depletion layer, $V(x)$ obeys Poisson's equation $d^2V/dx^2 = -\rho/\epsilon$, not Laplace's equation.",
         r"The ionized dopants make $\rho \neq 0$ there."),
        ("parab", True, r"In each depletion layer $V(x)$ is a parabola (quadratic in $x$).",
         r"Integrating $V'' = -\rho/\epsilon$ twice with constant $\rho$ gives a quadratic."),
        ("Econt", True, r"$E_x$ is continuous at the metallurgical junction $x = 0$.",
         r"There is no surface charge at $x = 0$, only a change in volume charge density, so $\epsilon E_x$ has no jump."),
        ("neutralE", True, r"In the depletion approximation, $\mathbf{E} = 0$ throughout the neutral regions outside the depletion layers.",
         r"The two layers carry equal and opposite charge per unit area, so their fields cancel outside them."),
        ("concave_n", True, r"Where $\rho \gt 0$ (the n-side depletion layer), $V(x)$ is concave down.",
         r"$V'' = -\rho/\epsilon \lt 0$ there: a dome."),
        ("lighter", True, r"The more lightly doped side holds the wider depletion layer.",
         r"$N_aW_p = N_dW_n$: fewer ions per unit volume must be spread over a longer distance to supply the same charge."),
        ("area", True, r"The built-in potential equals the area under the graph of $|E_x|$ versus $x$ across the depletion region.",
         r"$V_{\text{bi}} = -\int\mathbf{E}\cdot d\mathbf{l}$ from the p edge to the n edge, and $\mathbf{E}$ points against that path."),
        ("mono", True, r"Going from the p side to the n side, $V(x)$ increases monotonically across the depletion region.",
         r"$\mathbf{E}$ points from n to p everywhere in the region, so $V$ rises steadily toward the n side."),
        ("linear", False, r"In each depletion layer $V(x)$ is linear in $x$, as Laplace's equation requires.",
         r"Laplace's equation holds only where $\rho = 0$; the depletion layers are charged."),
        ("Ejump", False, r"$E_x$ jumps at $x = 0$ because the charge density changes sign there.",
         r"A jump in $E$ needs a surface charge; a step in the volume density only changes the slope of $E_x$."),
        ("concave_p", False, r"Where $\rho \lt 0$ (the p-side depletion layer), $V(x)$ is concave down.",
         r"$V'' = -\rho/\epsilon \gt 0$ there: $V$ is concave up (a bowl)."),
        ("heavier", False, r"The more heavily doped side holds the wider depletion layer, because it has more charge to offer.",
         r"Neutrality $N_aW_p = N_dW_n$ makes the heavily doped layer the thinner one."),
        ("Vequal", False, r"Because the junction is in equilibrium, the electrostatic potential $V$ is the same in the neutral p and n regions.",
         r"The n region sits higher by $V_{\text{bi}}$; it is the field between them that holds the junction in equilibrium."),
    ]
    pool.append(("nwider", n_wider, r"In this junction the n-side depletion layer is wider than the p-side layer.",
                 f"$W_n = {fmt(Wn_nm)}$ nm and $W_p = {fmt(Wp_nm)}$ nm, so this is " + ("true." if n_wider else "false.")))
    fb = {"nwider": r"Neutrality, $N_aW_p = N_dW_n$, puts the wider layer on the more lightly doped side: compare $N_a$ and $N_d$."}
    cb = [{"key": k, "correct": "true" if t else "false", "text": txt, "tf": "True" if t else "False",
           "why": why, "feedback": ("True. " if t else "False. ") + fb.get(k, why)} for k, t, txt, why in pool]
    random.shuffle(cb)

    # ---------------------------------------------------------------- strings
    if given == "p":
        given_tex = f"W_p = {fmt(Wp_nm)}\\ \\mathrm{{nm}}"
        given_words = "into the p side"
        ask_label, ask_sym = "$W_n =$", "W_n"
        a_solve = (f"W_n = W_p\\,\\frac{{N_a}}{{N_d}} = {fmt(Wp_nm)}\\times\\frac{{ {dop_tex(Na)} }}{{ {dop_tex(Nd)} }}"
                   f" = {fmt(Wn_nm)}\\ \\mathrm{{nm}}")
    else:
        given_tex = f"W_n = {fmt(Wn_nm)}\\ \\mathrm{{nm}}"
        given_words = "into the n side"
        ask_label, ask_sym = "$W_p =$", "W_p"
        a_solve = (f"W_p = W_n\\,\\frac{{N_d}}{{N_a}} = {fmt(Wn_nm)}\\times\\frac{{ {dop_tex(Nd)} }}{{ {dop_tex(Na)} }}"
                   f" = {fmt(Wp_nm)}\\ \\mathrm{{nm}}")
    d_words = ("$-\\hat{x}$: from the n side ($x \\gt 0$) to the p side ($x \\lt 0$)" if p_left
               else "$+\\hat{x}$: from the n side ($x \\lt 0$) to the p side ($x \\gt 0$)")

    data["params"].update({
        "Na": Na, "Nd": Nd, "p_left": p_left, "given": given, "Wp_nm": Wp_nm, "Wn_nm": Wn_nm, "W_given_nm": Wp_nm if given == "p" else Wn_nm,
        "Na_tex": dop_tex(Na), "Nd_tex": dop_tex(Nd), "Na_m_tex": dop_tex(Na, 6), "Nd_m_tex": dop_tex(Nd, 6),
        "p_region": p_region, "n_region": n_region, "p_layer": p_layer, "n_layer": n_layer,
        "given_tex": given_tex, "given_words": given_words, "ask_label": ask_label, "ask_sym": ask_sym,
        "a_solve": a_solve, "Wp_str": fmt(Wp_nm), "Wn_str": fmt(Wn_nm), "Wsum_str": fmt(Wp_nm + Wn_nm, 5),
        "rho_p": fmt(E_CH * Na_m), "rho_n": fmt(E_CH * Nd_m),
        "Emax_str": fmt(Emax / 1e6), "Emax_Vm": fmt(Emax), "Vbi_str": fmt(Vbi), "d_words": d_words,
        "loc": loc, "dirs": dirs, "cb": cb,
    })
    data["correct_answers"]["W_other"] = float(W_other)        # nm
    data["correct_answers"]["Emax"] = float(Emax / 1e6)        # MV/m
    data["correct_answers"]["Vbi"] = float(Vbi)                # V
