"""Displacement current in a charging capacitor; fields at a perfect conductor (Lecture 16). Real cosines only.

Capacitor: circular plates of radius a at z = 0 (bottom) and z = d (top), gap filled with eps_r; leads on the axis;
no fringing (E uniform along z in the gap, zero outside). i = lead current counted along +z.
  mode "const": a constant current I flows along s_I * z in both leads -> the bottom plate gains charge at s_I * I:
        dD_z/dt = s_I I / (pi a^2)                         (D_z = rho_s of the bottom plate, whatever eps_r)
  mode "ac":    V_bottom - V_top = s_v V0 cos(w t)  ->  E_z = s_v V0 cos(w t) / d,  D_z = eps E_z,
        dD_z/dt = -s_v eps0 eps_r V0 w sin(w t) / d
Ampere-Maxwell on a circle of radius r in the mid-plane (counterclockwise from +z, flat disk, no conduction current):
        H_phi(r < a) = r J_d / 2,   H_phi(r > a) = a^2 J_d / (2 r) = (pi a^2 J_d) / (2 pi r)
Perfect conductor filling a half-space; n = unit normal out of the conductor (medium 2 -> medium 1):
        J_s = n x H_1   (H_1 tangential just outside);  E_1 normal; rho_s = n . D_1.
"""
import math
import random

EPS0 = 8.854e-12
MATERIALS = [("air", 1.0), ("PTFE (Teflon)", 2.1), ("polyethylene", 2.25), ("polystyrene", 2.6),
             ("fused quartz", 3.8)]
UNIT = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}


def tex_num(x, sig=4):
    """Number for LaTeX: 4 significant figures, with \\times10^{n} instead of Python's e-notation."""
    s = f"{x:.{sig}g}"
    if "e" in s:
        mant, ex = s.split("e")
        return mant + r"\times10^{" + str(int(ex)) + "}"
    return s


def cross(u, v):
    return (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])


def dot(u, v):
    return sum(x * y for x, y in zip(u, v))


def vec_tex(v):
    """Integer-component vector as LaTeX, e.g. 3\\,\\hat{x} - 5\\,\\hat{z}."""
    out = ""
    for comp, ax in zip(v, "xyz"):
        if comp == 0:
            continue
        mag = f"{abs(comp):g}"
        if not out:
            out = ("-" if comp < 0 else "") + mag + r"\,\hat{" + ax + "}"
        else:
            out += (" - " if comp < 0 else " + ") + mag + r"\,\hat{" + ax + "}"
    return out or "0"


def generate(data):
    p = data["params"]
    mode = random.choice(["const", "ac"])
    mat, er = random.choice(MATERIALS)
    while True:
        a_cm = random.choice([2, 2.5, 3, 4, 5, 6, 8])
        d_mm = random.choice([0.5, 1, 1.5, 2])
        a, d = a_cm * 1e-2, d_mm * 1e-3
        if mode == "const":
            I_mA = random.choice([10, 20, 25, 40, 50, 80, 100, 150, 200])
            s_I = random.choice([1, -1])
            Jd = s_I * I_mA * 1e-3 / (math.pi * a * a)
            i_lead = s_I * I_mA * 1e-3
        else:
            V0 = random.choice([5, 10, 20, 50, 100])
            f_kHz = random.choice([50, 100, 200, 250, 500, 1000])
            j = random.choice([k for k in range(1, 20) if k != 10])
            s_v = random.choice([1, -1])
            w = 2 * math.pi * f_kHz * 1e3
            T = 1 / (f_kHz * 1e3)
            t1 = j * T / 20
            Jd = -s_v * EPS0 * er * V0 * w * math.sin(w * t1) / d
            i_lead = math.pi * a * a * Jd
        frac = random.choice([0.25, 0.4, 0.5, 0.6, 0.75])
        mult = random.choice([1.25, 1.5, 2, 2.5, 3])
        if abs(frac * mult - 1) < 0.15:
            continue                  # r1 r2 = a^2 would make H(r1) = H(r2) and hide a swapped formula
        r1, r2 = frac * a, mult * a
        H1 = r1 * Jd / 2
        H2 = a * a * Jd / (2 * r2)
        if 0.3 <= abs(Jd) <= 300 and 1e-3 <= min(abs(H1), abs(H2)) and max(abs(H1), abs(H2)) <= 3:
            break
    r1_cm, r2_cm = round(r1 * 100, 4), round(r2 * 100, 4)

    if mode == "const":
        drive = (r"The capacitor is being charged by a constant current $I = " + f"{I_mA:g}" + r"\ \mathrm{mA}$ that "
                 r"flows along $" + ("+" if s_I > 0 else "-") + r"\hat{z}$ in both leads (" +
                 ("into the bottom plate from below and out of the top plate upward" if s_I > 0 else
                  "into the top plate from above and out of the bottom plate downward") + ").")
        when = r"(at any instant while it charges)"
        a_words = (
            r"Gauss's law at the bottom plate: just above it $D_z = \rho_s = Q_b/(\pi a^2)$, with $\hat n = +\hat z$ "
            r"out of that plate and $Q_b$ its charge. The lead current changes $Q_b$ at the rate $dQ_b/dt = i = " +
            ("+" if s_I > 0 else "-") + r"I$ (counted along $+\hat z$), so"
            r"$$\frac{\partial D_z}{\partial t} = \frac{i}{\pi a^2} = \frac{" + ("" if s_I > 0 else "-") +
            f"{I_mA * 1e-3:g}" + r"\ \mathrm{A}}{\pi(" + f"{a:g}" + r"\ \mathrm{m})^2} = " + f"{Jd:.4g}" +
            r"\ \mathrm{A/m^2}.$$"
            r"The filling ($\epsilon_r = " + f"{er:g}" + r"$) does not enter: $D$ is set by the free charge on the "
            r"plates." + ("" if er == 1 else
                          r" Of this displacement current, the vacuum part $\epsilon_0\,\partial E_z/\partial t$ "
                          r"carries the fraction $1/\epsilon_r = " + f"{1 / er:.3g}" + r"$ and the polarization "
                          r"current $\partial P_z/\partial t$ of the dielectric the remaining $1 - 1/\epsilon_r = " +
                          f"{1 - 1 / er:.3g}" + r"$.")
        )
    else:
        who = "bottom plate relative to the top plate" if s_v > 0 else "top plate relative to the bottom plate"
        drive = (r"A generator holds the potential of the " + who + r" at $v(t) = V_0\cos\omega t$ with $V_0 = " +
                 f"{V0:g}" + r"\ \mathrm{V}$ and $f = \omega/2\pi = " + (r"1\ \mathrm{MHz}" if f_kHz == 1000 else
                 f"{f_kHz:g}" + r"\ \mathrm{kHz}") + r"$ (the "
                 r"capacitor is tiny compared with the wavelength, so the uniform-field picture holds). Evaluate "
                 r"the fields at $t_1 = " + f"{t1 * 1e6:.4g}" + r"\ \mu\mathrm{s}$.")
        when = r"at $t = t_1$"
        a_words = (
            r"$E_z = (V_{\text{bottom}} - V_{\text{top}})/d = " + ("" if s_v > 0 else "-") + r"V_0\cos(\omega t)/d$, "
            r"so $D_z = \epsilon_0\epsilon_rE_z$ and"
            r"$$\frac{\partial D_z}{\partial t} = " + ("-" if s_v > 0 else "") +
            r"\frac{\epsilon_0\epsilon_rV_0\,\omega}{d}\sin\omega t.$$"
            r"With $\omega = 2\pi f = " + tex_num(w) + r"\ \mathrm{rad/s}$ and $\omega t_1 = " + f"{w * t1:.4g}" +
            r"\ \mathrm{rad}$ ($t_1 = " + f"{j}" + r"T/20$): $\partial D_z/\partial t = " + ("-" if s_v > 0 else "") +
            r"(8.854\times10^{-12})(" + f"{er:g}" + r")(" + f"{V0:g}" + r")(" + tex_num(w) + r")\sin(" +
            f"{w * t1:.4g}" + r")/(" + f"{d:g}" + r") = " + f"{Jd:.4g}" + r"\ \mathrm{A/m^2}$. Here $\epsilon_r$ "
            r"matters: with the voltage prescribed, $D = \epsilon E$ and its rate of change scale with $\epsilon$."
        )
    b_words = (
        r"Take a circle of radius $r$ in the mid-plane, centred on the axis and traversed counterclockwise seen from "
        r"$+z$, and the flat disk it bounds (normal $+\hat z$). No conduction current crosses that disk; the "
        r"displacement current through it is $\partial D_z/\partial t$ times the part of its area inside the "
        r"plates, $\pi\min(r,a)^2$. By symmetry $H_\phi$ is constant on the circle, so"
        r"$$H_\phi(r_1) = \frac{r_1}{2}\frac{\partial D_z}{\partial t} = \frac{" + f"{r1:g}" + r"}{2}(" +
        f"{Jd:.4g}" + r") = " + f"{H1 * 1e3:.4g}" + r"\ \mathrm{mA/m},\qquad H_\phi(r_2) = \frac{a^2}{2r_2}"
        r"\frac{\partial D_z}{\partial t} = \frac{(" + f"{a:g}" + r")^2}{2(" + f"{r2:g}" + r")}(" + f"{Jd:.4g}" +
        r") = " + f"{H2 * 1e3:.4g}" + r"\ \mathrm{mA/m}.$$"
        r"Inside, $H_\phi$ grows linearly from zero on the axis to its largest value at $r = a$; outside, it falls as "
        r"$1/r$, exactly as around the lead itself: $H_\phi(r_2) = i/(2\pi r_2)$ with $i = \pi a^2\,\partial D_z/"
        r"\partial t = " + f"{i_lead * 1e3:.4g}" + r"\ \mathrm{mA}$."
    )

    # ---------------- (c) two surfaces on one circle around the lower lead
    Rc_cm = round(random.choice([1.5, 2]) * a_cm, 4)
    zc_mm = random.choice([5, 10, 20])
    paradox = [
        {"key": "both", "correct": "true",
         "text": (r"$S_1$ carries the conduction current $i$ of the lead and no displacement current; no charge "
                  r"crosses $S_2$, but the displacement current through it, $\int_{S_2}\partial\mathbf{D}/\partial t"
                  r"\cdot d\mathbf{S}$, equals $i$. Both give $\oint_C\mathbf{H}\cdot d\mathbf{l} = i$."),
         "feedback": (r"Right: together $S_1$ and $S_2$ enclose the bottom plate, and by Gauss's law the flux of "
                      r"$\mathbf{D}$ through $S_2$ is the plate's charge, whose rate of change is $i$.")},
        {"key": "static", "correct": "false",
         "text": (r"$S_1$ gives $\oint_C\mathbf{H}\cdot d\mathbf{l} = i$ and $S_2$ gives 0: the result depends on the "
                  r"surface, so Ampère's law cannot be used for a charging capacitor."),
         "feedback": (r"That is the paradox of the static law $\oint\mathbf{H}\cdot d\mathbf{l} = I_{\text{enc}}$. With "
                      r"Maxwell's term the right side is the flux of $\mathbf{J} + \partial\mathbf{D}/\partial t$, which "
                      r"is the same for every surface on $C$.")},
        {"key": "zero", "correct": "false",
         "text": (r"Both give 0: no current gets through a capacitor, so there is no magnetic field around its "
                  r"leads."),
         "feedback": (r"The lead carries a real conduction current $i$ through $S_1$, so the circulation around $C$ "
                      r"cannot be zero.")},
        {"key": "leak", "correct": "false",
         "text": (r"Both give $i$, because charge leaks through the dielectric and crosses $S_2$ as a conduction "
                  r"current."),
         "feedback": (r"The dielectric is perfect: no charge crosses the gap. What passes through $S_2$ is "
                      r"displacement current, the rate of change of the electric flux.")},
    ]
    if er > 1:
        paradox.append(
            {"key": "vacuum", "correct": "false",
             "text": (r"$S_1$ gives $i$ but $S_2$ gives only $\epsilon_0\pi a^2\,\partial E_z/\partial t = "
                      r"i/\epsilon_r$, so the two surfaces disagree unless the gap is empty."),
             "feedback": (r"The displacement current density is $\partial\mathbf{D}/\partial t$, not "
                          r"$\epsilon_0\partial\mathbf{E}/\partial t$; the polarization current $\partial\mathbf{P}/"
                          r"\partial t$ of the dielectric carries the rest, and the total through $S_2$ is $i$.")})
    else:
        paradox.append(
            {"key": "half", "correct": "false",
             "text": (r"Both give $i/2$: the circulation splits the difference between the conduction current through "
                      r"$S_1$ and the zero conduction current through $S_2$."),
             "feedback": (r"There is no averaging: each surface gives the full $i$, $S_1$ as conduction current and "
                          r"$S_2$ as displacement current.")})
    c_words = (
        r"$S_1$ is pierced by the lead, so it carries $i$; with $\mathbf{E} = 0$ outside the gap there is no "
        r"displacement current through it. $S_2$ is pierced by nothing, but its bottom lies in the gap, where "
        r"$\partial D_z/\partial t \ne 0$: the displacement current through it is $\pi a^2\,\partial D_z/\partial t$. "
        r"Since $S_1$ and $S_2$ together enclose the bottom plate, Gauss's law makes the $\mathbf{D}$-flux through "
        r"$S_2$ equal to the plate's charge $Q_b$, and $dQ_b/dt = i$: both surfaces give $\oint_C\mathbf{H}\cdot "
        r"d\mathbf{l} = i$. The static law would give $i$ and $0$, a contradiction; only $\partial\mathbf{D}/"
        r"\partial t$ (all of it, including the polarization current in a dielectric) removes it."
    )

    # ---------------- (d) perfect conductor: J_s = n x H1
    axis = random.choice(["x", "y", "z"])
    side = random.choice([1, -1])                      # conductor occupies axis*side > 0
    e = UNIT[axis]
    n = tuple(-side * x for x in e)                    # out of the conductor
    tang = [ax for ax in "xyz" if ax != axis]
    comps = [random.choice([-8, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 8]) for _ in tang]
    Hpec = [0, 0, 0]
    for ax, cmp in zip(tang, comps):
        Hpec["xyz".index(ax)] = cmp
    Hpec = tuple(Hpec)
    Js = cross(n, Hpec)
    region = axis + (r" \gt 0" if side > 0 else r" \lt 0")
    free = axis + (r" \lt 0" if side > 0 else r" \gt 0")
    n_tex = ("+" if n["xyz".index(axis)] > 0 else "-") + r"\hat{" + axis + "}"
    d_words = (
        r"The conductor is medium 2 and free space is medium 1, so $\hat n$ points out of the conductor: "
        r"$\hat n = " + n_tex + r"$. Inside the perfect conductor $\mathbf{H} = 0$, and the boundary condition "
        r"$\hat n\times(\mathbf{H}_1 - \mathbf{H}_2) = \mathbf{J}_s$ gives"
        r"$$\mathbf{J}_s = \hat n\times\mathbf{H}_1 = (" + n_tex + r")\times(" + vec_tex(Hpec) + r") = " +
        vec_tex(Js) + r"\ \mathrm{A/m}.$$"
        r"It lies in the surface (its $" + axis + r"$ component is zero) and is perpendicular to $\mathbf{H}_1$; "
        r"check: $\mathbf{J}_s\times\hat n = " + vec_tex(cross(Js, n)) + r" = \mathbf{H}_1$."
    )

    # ---------------- (e) checkbox
    def rand_tangential():
        v = [0, 0, 0]
        for ax in tang:
            v["xyz".index(ax)] = random.choice([-4, -3, -2, 2, 3, 4])
        return tuple(v)

    kE = random.choice([-5, -3, -2, 2, 3, 5])
    typeE = random.choice(["normal", "tangential", "oblique"])
    if typeE == "normal":
        vE = tuple(kE * x for x in n)
    elif typeE == "tangential":
        vE = rand_tangential()
    else:
        t_ = rand_tangential()
        vE = tuple(kE * x + y for x, y in zip(n, t_))
    typeH = random.choice(["tangential", "normal", "oblique"])
    kH = random.choice([-4, -3, 2, 3, 5])
    if typeH == "tangential":
        vH = rand_tangential()
    elif typeH == "normal":
        vH = tuple(kH * x for x in n)
    else:
        t_ = rand_tangential()
        vH = tuple(kH * x + y for x, y in zip(n, t_))
    kR = random.choice([-6, -4, -3, -2, 2, 3, 4, 6])
    vR = tuple(kR * x for x in n)
    pec = [
        ("vecE", typeE == "normal",
         r"At the point of part (d), $\mathbf{E}_1 = " + vec_tex(vE) + r"\ \mathrm{kV/m}$ is a possible electric field "
         r"just outside the conductor.",
         (r"It is along $\hat n$: $\mathbf{E}$ just outside a perfect conductor must be normal ($\hat n\times"
          r"\mathbf{E}_1 = 0$)." if typeE == "normal" else
          r"$\hat n\times\mathbf{E}_1 = " + vec_tex(cross(n, vE)) + r"\ \mathrm{kV/m} \ne 0$, so it has a tangential "
          r"part; but tangential $\mathbf{E}$ is continuous and $\mathbf{E} = 0$ inside, so $\mathbf{E}_1$ must be "
          r"normal."), {"vec": vE, "type": typeE}),
        ("vecH", typeH == "tangential",
         r"At the point of part (d), $\mathbf{H} = " + vec_tex(vH) + r"\ \mathrm{A/m}$ is a possible magnetic field "
         r"just outside the conductor at some instant.",
         (r"$\hat n\cdot\mathbf{H} = 0$: it is tangential, as $\mathbf{H}$ just outside a perfect conductor must be "
          r"(normal $\mathbf{B}$ is continuous and $\mathbf{B} = 0$ inside)." if typeH == "tangential" else
          r"$\hat n\cdot\mathbf{H} = " + f"{dot(n, vH):g}" + r"\ \mathrm{A/m} \ne 0$, but normal $\mathbf{B}$ is "
          r"continuous and there is no time-varying $\mathbf{B}$ inside, so $\mathbf{H}_1$ must be tangential."),
         {"vec": vH, "type": typeH}),
        ("rho_sign", kR > 0,
         r"If the electric field just outside the conductor at the point of part (d) is $\mathbf{E}_1 = " + vec_tex(vR) +
         r"\ \mathrm{kV/m}$, the surface charge density there is positive.",
         r"$\rho_s = \hat n\cdot\mathbf{D}_1 = \epsilon_0\,\hat n\cdot\mathbf{E}_1 = " + f"{kR:g}" +
         r"\,\mathrm{kV/m}\times\epsilon_0 \approx " + f"{kR * 1e3 * EPS0 * 1e9:.4g}" + r"\ \mathrm{nC/m^2}$: " +
         (r"positive, since $\mathbf{E}_1$ points away from the conductor." if kR > 0 else
          r"negative, since $\mathbf{E}_1$ points into the conductor."), {"vec": vR}),
        ("js_perp", True,
         r"The surface current of part (d) lies in the conductor's surface and is perpendicular to $\mathbf{H}_1$.",
         r"$\hat n\times\mathbf{H}_1$ is perpendicular to both $\hat n$ and $\mathbf{H}_1$.", None),
        ("no_dBdt", True,
         r"Inside a perfect conductor, where $\mathbf{E} = 0$, Faraday's law rules out any time-varying $\mathbf{B}$.",
         r"$\partial\mathbf{B}/\partial t = -\nabla\times\mathbf{E} = 0$ there.", None),
        ("Et_dynamic", False,
         r"For time-varying fields, $\mathbf{E}$ just outside a perfect conductor may have a tangential component, "
         r"because the static argument based on $\oint\mathbf{E}\cdot d\mathbf{l} = 0$ no longer applies.",
         r"Faraday's law on a thin loop straddling the surface encloses a flux that vanishes with the loop's width, "
         r"so tangential $\mathbf{E}$ is still continuous, and it is zero inside.", None),
        ("Bn", True,
         r"For time-varying fields, the normal component of $\mathbf{B}$ just outside a perfect conductor is zero.",
         r"$B_n$ is continuous (pillbox and $\oint\mathbf{B}\cdot d\mathbf{S} = 0$), and the time-varying "
         r"$\mathbf{B}$ is zero inside.", None),
    ]
    er_true = mode == "ac"
    cap = [
        ("continuity", True,
         r"Taking the divergence of $\nabla\times\mathbf{H} = \mathbf{J} + \partial\mathbf{D}/\partial t$ and using "
         r"$\nabla\cdot\mathbf{D} = \rho$ gives the continuity equation $\nabla\cdot\mathbf{J} = -\partial\rho/"
         r"\partial t$.",
         r"$0 = \nabla\cdot(\nabla\times\mathbf{H}) = \nabla\cdot\mathbf{J} + \partial(\nabla\cdot\mathbf{D})/"
         r"\partial t = \nabla\cdot\mathbf{J} + \partial\rho/\partial t$.", None),
        ("charges_cross", False,
         r"The displacement current in the capacitor's gap is carried by charges that cross from one plate to the "
         r"other.",
         r"No charge crosses a perfect dielectric; the displacement current is the rate of change of the electric "
         r"flux (plus the bound-charge polarization current inside the dielectric).", None),
        ("divB_faraday", False,
         r"Faraday's law by itself, with no other assumption, implies $\nabla\cdot\mathbf{B} = 0$.",
         r"Its divergence only gives $\partial(\nabla\cdot\mathbf{B})/\partial t = 0$; that $\nabla\cdot\mathbf{B}$ "
         r"is zero rather than some other constant takes the absence of magnetic charge.", None),
        ("H_axis", False,
         r"In the capacitor's gap, $|H_\phi|$ is largest on the axis, $r = 0$.",
         r"$H_\phi = (r/2)\,\partial D_z/\partial t$ is zero on the axis and largest at the plate edge $r = a$.", None),
        ("er_effect", er_true,
         (r"Filling the gap with a dielectric of larger $\epsilon_r$, with the same charging current $I$, would "
          r"increase the displacement current density in the gap." if mode == "const" else
          r"Filling the gap with a dielectric of larger $\epsilon_r$, with the same applied voltage $v(t)$, would "
          r"increase the displacement current density in the gap."),
         (r"With the current prescribed, $\partial D_z/\partial t = I/(\pi a^2)$ whatever fills the gap; a larger "
          r"$\epsilon_r$ only lowers $E$." if mode == "const" else
          r"With the voltage prescribed, $E_z = v/d$ is fixed and $\partial D_z/\partial t = \epsilon_0\epsilon_r\,"
          r"\partial E_z/\partial t$ grows with $\epsilon_r$ (and the lead current grows with it)."), None),
        ("lead_equal", True,
         r"In the model of parts (a) and (b), $H_\phi$ at $r_2$ in the mid-plane equals $H_\phi$ at the same "
         r"distance $r_2$ from the lower lead, below the bottom plate.",
         r"Both circles enclose the same total current $i$: as displacement current in the mid-plane, as conduction "
         r"current below the plate. So both give $H_\phi = i/(2\pi r_2)$.", None),
    ]
    while True:
        sel = random.sample(pec, 2) + random.sample(cap, 3)
        nt = sum(1 for x in sel if x[1])
        if 1 <= nt <= 4:
            break
    random.shuffle(sel)
    p["facts"] = [{"key": k, "correct": "true" if t else "false", "verdict": "True" if t else "False",
                   "text": txt, "why": why, "extra": ex} for k, t, txt, why, ex in sel]

    p.update({
        "mode": mode, "material": mat, "er": f"{er:g}", "a_cm": f"{a_cm:g}", "d_mm": f"{d_mm:g}",
        "r1_cm": f"{r1_cm:g}", "r2_cm": f"{r2_cm:g}", "drive": drive, "when": when,
        "a_words": a_words, "b_words": b_words, "c_words": c_words, "d_words": d_words,
        "Rc_cm": f"{Rc_cm:g}", "zc_mm": zc_mm, "paradox": paradox,
        "pec_region": region, "pec_free": free, "H1_tex": vec_tex(Hpec), "n": list(n), "Hpec": list(Hpec),
        "pec_axis": axis, "pec_side": side,
    })
    if mode == "const":
        p.update({"I_mA": I_mA, "s_I": s_I})
    else:
        p.update({"V0": V0, "f_kHz": f_kHz, "j": j, "s_v": s_v, "t1_us": t1 * 1e6})
    data["correct_answers"]["Jd"] = Jd             # A/m^2, along +z
    data["correct_answers"]["H1"] = H1 * 1e3       # mA/m, along +phi
    data["correct_answers"]["H2"] = H2 * 1e3       # mA/m, along +phi
    data["correct_answers"]["Jsx"] = float(Js[0])  # A/m
    data["correct_answers"]["Jsy"] = float(Js[1])
    data["correct_answers"]["Jsz"] = float(Js[2])
