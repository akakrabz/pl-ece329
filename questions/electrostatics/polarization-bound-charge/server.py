"""Frozen-in polarization in an electret slab: what div P tells us (Lectures 8-9).

Slab 0 < x < d, unbounded in y and z; P = P(x) x-hat inside, P = 0 outside. No free charge anywhere and no applied
field (E -> 0 far from the slab on both sides).
    rho_b  = -dP/dx                            bound volume charge
    rho_sb = P . n_out                         face x = d: +P(d);  face x = 0: -P(0)
    total bound charge per unit area = int_0^d rho_b dx + P(d) - P(0) = 0
    div D = rho_free = 0 with D = D_x(x) x-hat  ->  D_x is one constant; it is 0 far away  ->  D = 0 everywhere,
    so inside eps0 E = D - P = -P  ->  E = -P/eps0 (and E = 0 outside).
Profiles: "lin"  P0 (1 + a x/d)      rho_b = -P0 a/d (uniform)
          "quad" P0 (x/d)^2          rho_b = -2 P0 x/d^2      (face x = 0 uncharged)
          "cap"  P0 (1 - (x/d)^2)    rho_b = +2 P0 x/d^2      (face x = d uncharged)
Units: P in uC/m^2 and lengths in mm, so dP/dx comes out in (uC/m^2)/mm = mC/m^3; E in kV/m.
"""
import random

EPS0 = 8.854e-12


def num(v, sig=4):
    """Number for LaTeX/plain text: up to `sig` significant figures, no exponent notation in the used ranges."""
    if v == 0:
        return "0"
    s = f"{v:.{sig}g}"
    if "e" in s:
        m, ex = s.split("e")
        return m + r"\times10^{" + str(int(ex)) + "}"
    return s


def generate(data):
    family = random.choice(["lin", "quad", "cap"])
    P0 = random.choice([0.5, 0.8, 1.2, 1.5, 2, 2.5, 3]) * random.choice([1, -1])    # uC/m^2
    d = random.choice([1, 2, 2.5, 4, 5])                                             # mm
    frac = random.choice([0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8])
    x1 = round(frac * d, 6)                                                          # mm
    a = random.choice([-0.5, 0.5, 1, 1.5, 2]) if family == "lin" else 0

    if family == "lin":
        def P(x):
            return P0 * (1 + a * x / d)
        dPdx = P0 * a / d
        P_tex = r"P_0\left(1 + \alpha\,\dfrac{x}{d}\right)\hat{x}"
        P_consts = (r"with $P_0 = " + num(P0) + r"\ \mu\mathrm{C/m^2}$ and $\alpha = " + num(a) + "$")
        rho_expr = r"-\dfrac{P_0\alpha}{d} = -\dfrac{(" + num(P0) + r")(" + num(a) + r")}{" + num(d) + "}"
        rho_note = "the same at every point of the slab, because $P_x$ is linear in $x$"
    elif family == "quad":
        def P(x):
            return P0 * (x / d) ** 2
        dPdx = 2 * P0 * x1 / d ** 2
        P_tex = r"P_0\left(\dfrac{x}{d}\right)^{2}\hat{x}"
        P_consts = r"with $P_0 = " + num(P0) + r"\ \mu\mathrm{C/m^2}$"
        rho_expr = r"-\dfrac{2P_0x_1}{d^2} = -\dfrac{2(" + num(P0) + r")(" + num(x1) + r")}{(" + num(d) + r")^2}"
        rho_note = "it grows in magnitude linearly with $x$"
    else:
        def P(x):
            return P0 * (1 - (x / d) ** 2)
        dPdx = -2 * P0 * x1 / d ** 2
        P_tex = r"P_0\left[1 - \left(\dfrac{x}{d}\right)^{2}\right]\hat{x}"
        P_consts = r"with $P_0 = " + num(P0) + r"\ \mu\mathrm{C/m^2}$"
        rho_expr = r"+\dfrac{2P_0x_1}{d^2} = \dfrac{2(" + num(P0) + r")(" + num(x1) + r")}{(" + num(d) + r")^2}"
        rho_note = "it grows in magnitude linearly with $x$"

    Pz, Pd, P1 = P(0.0), P(float(d)), P(x1)
    rho_b = -dPdx                       # mC/m^3
    rho_s0 = -Pz                        # uC/m^2, n = -x on the face x = 0
    rho_sd = Pd                         # uC/m^2, n = +x on the face x = d
    S = Pd - Pz                         # net face charge per unit area (nonzero for every variant)
    Ex = -P1 * 1e-6 / EPS0 / 1e3        # kV/m

    # ---------------------------------------------------------------- (c) total bound charge, MC
    tot_choices = [
        {"text": r"Zero, whatever the profile $P(x)$: the bound volume charge exactly cancels the net bound charge "
                 r"on the two faces.",
         "correct": "true",
         "feedback": r"Right: $\int_0^d\rho_b\,dx = -[P(d) - P(0)]$ cancels the face charges $-P(0) + P(d)$. "
                     r"Polarizing only shifts charge that was already there, so the slab stays neutral."},
        {"text": r"Equal to the net bound charge on the two faces, $\rho_{sb}(0) + \rho_{sb}(d)$: bound charge sits only "
                 r"on the faces of a polarized body.",
         "correct": "false",
         "feedback": r"That is the face charge alone, $" + num(S) + r"\ \mu\mathrm{C/m^2}$ here. A nonuniform $\mathbf{P}$ also leaves the volume charge "
                     r"$\rho_b = -dP_x/dx$, whose integral is $-[P(d) - P(0)]$; only a uniform $\mathbf{P}$ has $\rho_b = 0$."},
        {"text": r"Equal to the bound volume charge $\int_0^d\rho_b\,dx$ alone: the charges on the two faces are always "
                 r"equal and opposite.",
         "correct": "false",
         "feedback": r"The faces carry $-P(0)$ and $+P(d)$, which are equal and opposite only when $P(0) = P(d)$. "
                     r"Here they add to $" + num(S) + r"\ \mu\mathrm{C/m^2}$, exactly what the volume charge cancels."},
        {"text": r"It cannot be found without the permittivity $\epsilon$ of the electret.",
         "correct": "false",
         "feedback": r"Bound charge follows from $\mathbf{P}$ alone: $\rho_b = -\nabla\cdot\mathbf{P}$, $\rho_{sb} = "
                     r"\mathbf{P}\cdot\hat{n}$. An electret has no $\epsilon$: its $\mathbf{P}$ is frozen in, not proportional to $\mathbf{E}$."},
    ]
    random.shuffle(tot_choices)

    # ---------------------------------------------------------------- (e) select-all pool (id = what the check re-derives)
    def st(sid, text, truth, why):
        return {"id": sid, "text": text, "correct": "true" if truth else "false",
                "feedback": ("True: " if truth else "False: ") + why}

    pool = [
        st("divD", r"With no free charge, $dD_x/dx = 0$, so $D_x$ has one and the same value inside and outside the slab.",
           True, r"$\nabla\cdot\mathbf{D} = \rho_f = 0$ and $\mathbf{D}$ depends on $x$ only; that constant is $0$ far away, hence everywhere."),
        st("gaussE", r"$\nabla\cdot(\epsilon_0\mathbf{E}) = \rho_f + \rho_b$: bound charge is a source of $\mathbf{E}$ just as free charge is.",
           True, r"$\mathbf{E}$ counts all charge; $\mathbf{D}$ is the field that counts free charge only."),
        st("uniformP", r"A uniform $\mathbf{P}$ has $\rho_b = 0$: all of its bound charge sits on the surfaces of the body.",
           True, r"$-\nabla\cdot\mathbf{P} = 0$ for constant $\mathbf{P}$; only $\mathbf{P}\cdot\hat{n}$ on the surfaces remains."),
        st("increasing", r"For any polarization of the form $\mathbf{P} = P_x(x)\hat{x}$, the bound volume charge density is negative "
                         r"wherever $P_x$ increases with $x$.",
           True, r"$\rho_b = -dP_x/dx \lt 0$ where $dP_x/dx \gt 0$."),
        st("Ddef", r"$\mathbf{D} = \epsilon_0\mathbf{E} + \mathbf{P}$ holds whether or not $\mathbf{P}$ is proportional to $\mathbf{E}$.",
           True, r"it is the definition of $\mathbf{D}$; $\mathbf{D} = \epsilon\mathbf{E}$ is the extra, linear-medium assumption."),
        st("pillbox", r"A pillbox across the face $x = d$ gives $\epsilon_0\,[E_x(d^+) - E_x(d^-)] = \rho_{sb}(d)$.",
           True, r"Gauss's law for $\epsilon_0\mathbf{E}$ counts the bound surface charge: $0 - (-P(d)) = P(d)$."),
        st("Ezero", r"Because there is no free charge anywhere, $\mathbf{E} = 0$ inside the slab.",
           False, r"bound charge makes a real field: inside, $\mathbf{E} = -\mathbf{P}/\epsilon_0 \neq 0$."),
        st("linear", r"Inside the electret $\mathbf{D} = \epsilon\mathbf{E}$ for some constant permittivity $\epsilon \gt 0$.",
           False, r"here $\mathbf{D} = 0$ while $\mathbf{E} \neq 0$; an electret is not a linear medium."),
        st("Pzero", r"A material cannot be polarized where $\mathbf{D} = 0$: wherever $\mathbf{D} = 0$, $\mathbf{P} = 0$ as well.",
           False, r"$\mathbf{D} = 0$ only means $\epsilon_0\mathbf{E} = -\mathbf{P}$, not $\mathbf{P} = 0$; in this slab $\mathbf{D} = 0$ "
                  r"everywhere while $\mathbf{P} \neq 0$ inside."),
        st("Econst", r"With no free charge, Gauss's law forces $E_x$, not just $D_x$, to have the same value everywhere.",
           False, r"$\epsilon_0\,dE_x/dx = \rho_b \neq 0$ inside, and $E_x$ jumps at the charged faces."),
        st("outside", r"Outside the slab $\mathbf{E} \neq 0$, because the faces carry bound charge.",
           False, r"$\mathbf{D} = 0$ and $\mathbf{P} = 0$ outside give $\mathbf{E} = 0$: the neutral slab's charge layers cancel there."),
        st("facescancel", r"The bound charges on the two faces of this slab add up to zero.",
           False, r"$-P(0) + P(d) = " + num(S) + r"\ \mu\mathrm{C/m^2} \neq 0$; the volume charge supplies the opposite amount."),
        st("uniformrho", r"The bound volume charge density has the same value at every point inside this slab.",
           family == "lin",
           r"$\rho_b = -dP_x/dx$ is constant only when $P_x$ is linear in $x$; " +
           ("here it is." if family == "lin" else "here it varies with $x$.")),
        st("face0", r"The face $x = 0$ of this slab carries no bound surface charge.",
           family == "quad",
           r"$\rho_{sb}(0) = -P_x(0) = " + num(rho_s0) + r"\ \mu\mathrm{C/m^2}$."),
    ]

    # ---------------------------------------------------------------- answer-panel strings
    if family == "lin":
        tot_words = (r"$\int_0^d\rho_b\,dx = \rho_b d = " + num(rho_b * d) + r"\ \mu\mathrm{C/m^2}$")
    else:
        tot_words = (r"$\int_0^d\rho_b\,dx = -[P(d) - P(0)] = " + num(-S) + r"\ \mu\mathrm{C/m^2}$")

    data["params"].update({
        "family": family, "P0": P0, "alpha": a, "d": d, "x1": x1, "frac": frac,
        "P_tex": P_tex, "P_consts": P_consts,
        "x1_str": num(x1), "d_str": num(d),
        "P0_at": num(Pz), "Pd_at": num(Pd), "P1_at": num(P1),
        "rho_expr": rho_expr, "rho_note": rho_note, "rho_b_str": num(rho_b),
        "rho_s0_str": num(rho_s0), "rho_sd_str": num(rho_sd), "S_str": num(S), "mS_str": num(-S),
        "tot_words": tot_words, "Ex_str": num(Ex),
        "tot_choices": tot_choices, "pool": pool,
    })
    data["correct_answers"]["rho_b"] = float(rho_b) + 0.0      # mC/m^3   (+ 0.0 turns a -0.0 into 0.0)
    data["correct_answers"]["rho_s0"] = float(rho_s0) + 0.0    # uC/m^2
    data["correct_answers"]["rho_sd"] = float(rho_sd) + 0.0    # uC/m^2
    data["correct_answers"]["Ex"] = float(Ex) + 0.0            # kV/m
