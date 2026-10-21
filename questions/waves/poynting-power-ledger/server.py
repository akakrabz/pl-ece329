"""Lecture 20 concept question: Poynting's theorem as an energy ledger (real cosines only, no phasors).

Wire (parts b, c): radius a, length l, conductivity sigma, steady current I along sI*w_hat, uniform over the section.
    J = I/(pi a^2),  E = J/sigma (axial, uniform inside),  H_phi(a) = I/(2 pi a)  (Ampere),
    S just inside the surface = E x H = -rho_hat * E H_phi(a)   (radially inward),
    |S| = I^2/(2 pi^2 sigma a^3),   P_in = |S| * 2 pi a l = I^2 l/(sigma pi a^2) = I^2 R.
    At the surface point P = a*(sp*p_hat): H along (sI w_hat) x (sp p_hat), S along -sp*p_hat.
Plane wave (part d), non-magnetic dielectric eps_r = n^2, eta = 120 pi/n, c = 3e8 (statement says so):
    given E = sE*E0 cos(wt - d beta u) e_hat:  <S> = E0^2/(2 eta) along d*u_hat;
    given H = sH*H0 cos(wt - d beta u) h_hat:  <S> = eta H0^2/2  along d*u_hat;
    window of area A whose normal is at theta to the travel: P = <S> A cos(theta).
Checkbox (part a): a pool of 15 statements on the theorem; one ("inner") depends on the variant:
    power crossing an inner coaxial cylinder of radius a/k is I^2 R/k^2 (shown as 1/k^2: true, or 1/k: false).
"""
import math
import random

AXES = ["x", "y", "z"]
IDX = {"x": 0, "y": 1, "z": 2}
ETA0 = 120 * math.pi

# handbook resistivities at 20 C -> sigma = 1/rho (S/m)
MATERIALS = [
    ("copper", 5.96e7), ("aluminium", 3.77e7), ("tungsten", 1.79e7), ("iron", 1.03e7),
    ("constantan (Cu-Ni alloy)", 2.04e6), ("type 304 stainless steel", 1.39e6),
    ("Kanthal (Fe-Cr-Al heater alloy)", 6.9e5),
]


def num(x, sig=4):
    s = f"{x:.{sig}g}"
    if "e" in s:
        m, e = s.split("e")
        return f"{m}\\times10^{{{int(e)}}}"
    return s


def uhat(ax, sg):
    return ("+" if sg > 0 else "-") + "\\hat{" + ax + "}"


def cross_axes(a, sa, b, sb):
    """(sa a_hat) x (sb b_hat) -> (axis, sign); a != b."""
    ia, ib = IDX[a], IDX[b]
    ic = 3 - ia - ib
    sgn = 1 if (ib - ia) % 3 == 1 else -1
    return AXES[ic], sa * sb * sgn


def kpi(k):
    """k*pi as TeX (k a multiple of 0.5)."""
    return r"\pi" if k == 1 else f"{k:g}\\pi"


def generate(data):
    # ================================================================ wire
    w = random.choice(AXES)
    sI = random.choice([1, -1])
    p_ax = random.choice([ax for ax in AXES if ax != w])
    sp = random.choice([1, -1])
    mat, sigma = random.choice(MATERIALS)
    while True:
        a_mm = random.choice([0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5])
        I = random.choice([0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 15, 20])
        ell = random.choice([0.5, 0.8, 1, 1.2, 1.5, 2, 2.5, 3, 4, 5])
        a = a_mm * 1e-3
        J = I / (math.pi * a * a)
        Ez = J / sigma
        Hs = I / (2 * math.pi * a)
        S = Ez * Hs
        R = ell / (sigma * math.pi * a * a)
        P = S * 2 * math.pi * a * ell
        if J <= 2.5e7 and 0.05 <= S / 1e3 <= 500 and 0.05 <= P <= 300:
            break
    h_ax, h_sg = cross_axes(w, sI, p_ax, sp)          # H direction at P
    s_ax, s_sg = cross_axes(w, sI, h_ax, h_sg)         # S direction at P (E along the current)
    assert s_ax == p_ax and s_sg == -sp
    coords = {ax: "0" for ax in AXES}
    coords[p_ax] = ("" if sp > 0 else "-") + "a"
    P_tex = "(" + ", ".join(coords[ax] for ax in AXES) + ")"

    sdir = [
        {"id": "in", "text": f"${uhat(p_ax, -sp)}$", "correct": "true",
         "feedback": (f"Right: $\\mathbf{{E}}$ is along the current, ${uhat(w, sI)}$, and $\\mathbf{{H}}$ circles it, along ${uhat(h_ax, h_sg)}$ at $P$; "
                      f"$({uhat(w, sI)})\\times({uhat(h_ax, h_sg)}) = {uhat(p_ax, -sp)}$, radially into the wire.")},
        {"id": "out", "text": f"${uhat(p_ax, sp)}$", "correct": "false",
         "feedback": (f"That is radially outward. The wire is a load ($\\mathbf{{J}}\\cdot\\mathbf{{E}} = \\sigma E^2 \\gt 0$) in steady state, so energy must converge on it: "
                      f"$({uhat(w, sI)})\\times({uhat(h_ax, h_sg)}) = {uhat(p_ax, -sp)}$.")},
        {"id": "along", "text": f"${uhat(w, sI)}$", "correct": "false",
         "feedback": r"That is the picture of energy carried down the wire by the current. But inside the wire $\mathbf{E}$ is parallel to the current, and $\mathbf{S} = \mathbf{E}\times\mathbf{H}$ is perpendicular to $\mathbf{E}$: no component along the wire."},
        {"id": "against", "text": f"${uhat(w, -sI)}$", "correct": "false",
         "feedback": r"$\mathbf{S} = \mathbf{E}\times\mathbf{H}$ is perpendicular to $\mathbf{E}$, and inside the wire $\mathbf{E}$ is along the wire axis, so $\mathbf{S}$ has no axial component in either sense."},
        {"id": "alongH", "text": f"${uhat(h_ax, h_sg)}$", "correct": "false",
         "feedback": (f"That is the direction of $\\mathbf{{H}}$ at $P$ (right-hand rule around the current). $\\mathbf{{S}} = \\mathbf{{E}}\\times\\mathbf{{H}}$ is perpendicular to $\\mathbf{{H}}$.")},
    ]
    random.shuffle(sdir)

    # ================================================================ plane wave
    u = random.choice(AXES)
    d = random.choice([1, -1])
    combos = [(n, f) for n in (1.5, 2.0, 2.5, 3.0) for f in (50, 75, 100, 150, 200, 300)
              if abs((f * n / 150) * 2 - round((f * n / 150) * 2)) < 1e-9]
    n, f_MHz = random.choice(combos)
    eps_r = n * n
    eta = ETA0 / n
    kw = 2 * f_MHz / 100                  # omega = kw*pi*1e8
    kb = f_MHz * n / 150                  # beta = kb*pi rad/m
    given = random.choice(["E", "H"])
    if given == "E":
        e_ax = random.choice([ax for ax in AXES if ax != u])
        sE = random.choice([1, -1])
        E0 = random.choice([10, 12, 15, 18, 20, 24, 30, 36, 40, 45, 50, 60, 75, 90, 120])
        H0 = E0 / eta
        Savg = E0 ** 2 / (2 * eta)
        hp_ax, hp_sg = cross_axes(u, d, e_ax, sE)           # partner H direction (u x E)
        f_ax, f_sg, f_amp_tex, f_unit, f_sym = e_ax, sE, f"{E0:g}", r"\mathrm{V/m}", r"\mathbf{E}"
        partner_sym, partner_amp, partner_unit = r"\mathbf{H}", num(H0 * 1e3), r"\mathrm{mA/m}"
        partner_rule = r"\mathbf{H} = \tfrac{1}{\eta}\,\hat{u}\times\mathbf{E}"
        S_formula = f"\\frac{{E_0^2}}{{2\\eta}} = \\frac{{({E0:g})^2}}{{2({num(eta)})}}"
    else:
        h_given = random.choice([ax for ax in AXES if ax != u])
        sH = random.choice([1, -1])
        H0_mA = random.choice([40, 50, 60, 80, 100, 120, 150, 200, 250, 300])
        H0 = H0_mA * 1e-3
        Savg = eta * H0 ** 2 / 2
        hp_ax, hp_sg = cross_axes(h_given, sH, u, d)        # partner E direction (H x u)
        f_ax, f_sg, f_amp_tex, f_unit, f_sym = h_given, sH, f"{H0_mA:g}", r"\mathrm{mA/m}", r"\mathbf{H}"
        partner_sym, partner_amp, partner_unit = r"\mathbf{E}", num(eta * H0), r"\mathrm{V/m}"
        partner_rule = r"\mathbf{E} = \eta\,\mathbf{H}\times\hat{u}"
        S_formula = f"\\frac{{\\eta H_0^2}}{{2}} = \\frac{{({num(eta)})({num(H0)})^2}}{{2}}"
    Su = d * Savg
    A_cm2 = random.choice([20, 25, 40, 50, 80, 100, 150, 200])
    theta = random.choice([0, 30, 45, 60])
    cth = math.cos(math.radians(theta))
    P_mW = Savg * A_cm2 * 1e-4 * cth * 1e3
    arg_tex = f"{kpi(kw)}\\times10^{{8}}\\,t {'-' if d > 0 else '+'} {kpi(kb)}\\,{u}"
    field_tex = f"{f_sym} = {'' if f_sg > 0 else '-'}{f_amp_tex}\\cos\\big({arg_tex}\\big)\\,\\hat{{{f_ax}}}\\ \\ {f_unit}"
    if theta == 0:
        window_txt = "facing the wave squarely (its normal is along the direction of travel)"
    else:
        window_txt = f"tilted so that its normal makes an angle of ${theta}^\\circ$ with the direction of travel"

    # ================================================================ checkbox pool
    k_in = random.choice([2, 3, 4])
    inner_true = random.random() < 0.5
    frac_shown = f"\\tfrac{{1}}{{{k_in * k_in}}}" if inner_true else f"\\tfrac{{1}}{{{k_in}}}"
    facts = [
        ("units", True,
         r"$\mathbf{S} = \mathbf{E}\times\mathbf{H}$ has the unit $(\mathrm{V/m})(\mathrm{A/m}) = \mathrm{W/m^2}$: a power per unit area of a surface perpendicular to $\mathbf{S}$.",
         r"$\mathrm{V\cdot A = W}$: $\mathbf{S}$ is an energy flux density, as $\mathbf{J}$ is a charge flux density, and its divergence (W/m³) matches the other two terms."),
        ("source", True,
         r"Where $\mathbf{J}\cdot\mathbf{E} \lt 0$, energy is passing from the charges to the field: that region acts as a source, like a battery or a radiating current sheet.",
         r"$\mathbf{J}\cdot\mathbf{E}$ is the power per unit volume the field gives to the charges, so a negative value is a transfer the other way. On a radiating sheet $\mathbf{E} = -\tfrac{\eta}{2}\mathbf{J}_s$, so $\mathbf{J}_s\cdot\mathbf{E} \lt 0$, and $-\mathbf{J}_s\cdot\mathbf{E}$ is exactly the power the two waves carry away."),
        ("div_out", True,
         r"At an instant when $\nabla\cdot\mathbf{S} \gt 0$ at a point where $\mathbf{J} = \mathbf{0}$, the energy density $w$ at that point is decreasing.",
         r"With $\mathbf{J} = \mathbf{0}$ the theorem reads $\partial w/\partial t = -\nabla\cdot\mathbf{S}$: a net outflow from a point drains the energy stored there."),
        ("instant", True,
         r"The point form $\partial w/\partial t + \nabla\cdot\mathbf{S} + \mathbf{J}\cdot\mathbf{E} = 0$ holds at every instant, not only on average over a period.",
         r"It is Faraday's law dotted with $\mathbf{H}$ minus the Ampère–Maxwell law dotted with $\mathbf{E}$, at each point and instant; averaging over a period is an extra step for sinusoidal steady state."),
        ("lti", True,
         r"Turning $\mathbf{E}\cdot\partial\mathbf{D}/\partial t$ into $\partial(\tfrac12\epsilon E^2)/\partial t$ in the derivation requires a linear medium ($\mathbf{D} = \epsilon\mathbf{E}$) whose $\epsilon$ does not change with time.",
         r"$\partial(\tfrac12\epsilon E^2)/\partial t = \epsilon\,\mathbf{E}\cdot\partial\mathbf{E}/\partial t + \tfrac12E^2\,\partial\epsilon/\partial t$: the last term must vanish, and $\mathbf{D}$ must be $\epsilon\mathbf{E}$ to begin with. (An $\epsilon$ that varies with position is harmless.)"),
        ("standing", True,
         r"For two plane waves of the same frequency, polarization and amplitude travelling in opposite directions through a lossless medium, $\langle\mathbf{S}\rangle = \mathbf{0}$ at every point, yet the time-averaged energy density $\langle w\rangle$ is not zero.",
         r"Each wave carries $E_0^2/(2\eta)$, in opposite directions, so the averages cancel; but each stores $\tfrac12\epsilon E_0^2$ on average and the energies add, $\langle w\rangle = \epsilon E_0^2$."),
        ("closed_avg", True,
         r"In a lossless region with no currents, in sinusoidal steady state, the time-averaged power flowing out through any closed surface drawn inside that region is zero.",
         r"Averaging the integral form over a period removes $dW/dt$ (the stored energy is periodic), leaving $\oint\langle\mathbf{S}\rangle\cdot d\mathbf{A} = -\int\langle\mathbf{J}\cdot\mathbf{E}\rangle\,dV = 0$."),
        ("inner", inner_true,
         f"In a straight wire of radius $a$ carrying a steady current spread uniformly over its cross-section, the power flowing inward across a coaxial cylinder of radius $a/{k_in}$ is ${frac_shown}$ of the $I^2R$ dissipated in the whole wire.",
         (f"Inside, the axial $E = I/(\\sigma\\pi a^2)$ is uniform and $H_\\phi = I\\rho/(2\\pi a^2)$, so the inflow across radius $\\rho$ is $EH_\\phi\\cdot2\\pi\\rho\\ell = I^2R\\,(\\rho/a)^2$, "
          f"the heat produced inside that cylinder: $\\tfrac{{1}}{{{k_in * k_in}}}$ of $I^2R$ at $\\rho = a/{k_in}$" + ("." if inner_true else f", not $\\tfrac{{1}}{{{k_in}}}$."))),
        ("S_alongE", False,
         r"$\mathbf{S}$ points along $\mathbf{E}$, the direction in which the field pushes positive charges.",
         r"$\mathbf{S} = \mathbf{E}\times\mathbf{H}$ is perpendicular to $\mathbf{E}$ (and to $\mathbf{H}$)."),
        ("ends", False,
         r"In a resistive wire carrying a steady current, the energy that becomes heat enters through the wire's end faces, carried along the wire with the current.",
         r"Inside the wire $\mathbf{E}$ is axial and $\mathbf{H}$ azimuthal, so $\mathbf{S}$ is radial: no axial component, and the energy enters through the side surface (part (b))."),
        ("magnetic_work", False,
         r"Poynting's theorem misses the work that the magnetic force does on moving charges, because $\mathbf{J}\cdot\mathbf{E}$ contains only the electric field.",
         r"The magnetic force $q\mathbf{u}\times\mathbf{B}$ is perpendicular to $\mathbf{u}$ and does no work: $\rho\,\mathbf{u}\cdot(\mathbf{E} + \mathbf{u}\times\mathbf{B}) = \mathbf{J}\cdot\mathbf{E}$ is the whole power delivered to the charges."),
        ("S_over_w", False,
         r"For a single plane wave in a lossless dielectric, $\langle S\rangle = \langle w\rangle/v$, with $v$ the wave speed.",
         r"The energy moves with the wave: $\langle S\rangle = v\,\langle w\rangle$, like $\mathbf{J} = \rho\mathbf{u}$. Units: $(\mathrm{J/m^3})(\mathrm{m/s}) = \mathrm{W/m^2}$."),
        ("dwdt_zero", False,
         r"In sinusoidal steady state, $\partial w/\partial t = 0$ at every point and at every instant.",
         r"Only its time average is zero: for a plane wave $w\propto\cos^2(\omega t-\beta u)$, which swings at $2\omega$."),
        ("S_reverses", False,
         r"The instantaneous Poynting vector of a single plane wave in a lossless medium reverses its direction twice in every period.",
         r"$\mathbf{E}$ and $\mathbf{H}$ change sign together, so $\mathbf{S}\propto\cos^2(\omega t-\beta u)\,\hat{u}$ never points against the travel: it pulses between $0$ and $2\langle S\rangle$."),
    ]
    fact_list = [{"id": fid, "correct": "true" if t else "false", "text": txt,
                  "feedback": ("True: " if t else "False: ") + why, "verdict": "true" if t else "false", "why": why}
                 for fid, t, txt, why in facts]
    random.shuffle(fact_list)

    # ================================================================ params
    trav_hat = uhat(u, d)
    data["params"].update({
        # wire
        "w": w, "sI": sI, "p_ax": p_ax, "sp": sp, "mat": mat, "sigma": sigma, "sigma_str": num(sigma, 3),
        "a_mm": a_mm, "I": I, "ell": ell, "I_str": f"{I:g}", "a_str": f"{a_mm:g}", "ell_str": f"{ell:g}",
        "Idir": uhat(w, sI), "P_tex": P_tex, "J_str": num(J), "Ez_str": num(Ez), "Hs_str": num(Hs),
        "S_str": num(S), "SkW_str": num(S / 1e3), "R_str": num(R), "Pw_str": num(P), "area_str": num(2 * math.pi * a * ell),
        "hP": uhat(h_ax, h_sg), "sP": uhat(p_ax, -sp), "rhat": uhat(p_ax, sp),
        "sdir_list": sdir,
        # wave
        "u": u, "d": d, "n": n, "eps_r": eps_r, "eps_str": f"{eps_r:g}", "f_MHz": f_MHz, "kw": kw, "kb": kb,
        "given": given, "f_ax": f_ax, "f_sg": f_sg, "field_tex": field_tex, "arg_tex": arg_tex,
        "eta_str": num(eta), "n_str": f"{n:g}", "trav_hat": trav_hat, "trav_word": ("+" if d > 0 else "-") + u,
        "partner_sym": partner_sym, "partner_amp": partner_amp, "partner_unit": partner_unit,
        "partner_hat": uhat(hp_ax, hp_sg), "partner_rule": partner_rule, "S_formula": S_formula,
        "Savg_str": num(Savg), "Su_str": num(Su), "A_cm2": A_cm2, "theta": theta, "cth_str": num(cth),
        "window_txt": window_txt, "P_mW_str": num(P_mW), "P_W_str": num(P_mW / 1e3), "Speak_str": num(2 * Savg),
        "f_amp": f_amp_tex, "d_pos": d > 0, "f_given_hat": uhat(f_ax, f_sg),
        # checkbox
        "k_in": k_in, "facts_list": fact_list,
    })
    data["correct_answers"]["S_surf"] = float(S / 1e3)       # kW/m^2
    data["correct_answers"]["P_in"] = float(P)               # W
    data["correct_answers"]["S_avg"] = float(Su)             # W/m^2, signed along +u
    data["correct_answers"]["P_win"] = float(P_mW)           # mW
