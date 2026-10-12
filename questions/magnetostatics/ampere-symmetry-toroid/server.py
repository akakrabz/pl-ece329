"""Ampere's law by symmetry, a toroid, and the vector potential (Lecture 13).

(a) matching: 5 of 14 symmetric current distributions -> how |H| depends on distance
    (options 1/r, r, uniform, 1/r^2 (never right), 1/r^3, zero).
(b) ideal toroid, N rectangular turns, core a < r < b, |z| < h/2, current along s*z on the inner wall (s = +-1):
        circle about z traversed along +phi (counterclockwise seen from +z), right-hand rule -> I_enc = s N I
        H_phi = s N I / (2 pi r)                                   [A/m]
        Psi (one turn, normal +phi) = s mu0 N I h ln(b/a) / (2 pi)  [uWb]
(c) MC: H = 0 at a point outside the core; the correct *reason* depends on the region
    (central hole / above the core: no wire pierces the disk; beyond the outer wall: 2N wires, opposite currents).
(d) MC: direction of A (Coulomb gauge) near a straight wire = direction of the current.
(e) checkbox: 5 of 13 statements on A and on the toroid, 1-4 of them true.
"""
import math
import random

MU0 = 4e-7 * math.pi

OPTIONS = [
    ("lin_r", r"$|\mathbf{H}|$ grows in proportion to the distance ($\propto r$)"),
    ("uniform", r"$|\mathbf{H}|$ is nonzero and independent of the distance"),
    ("inv_r", r"$|\mathbf{H}|\propto 1/r$"),
    ("inv_r2", r"$|\mathbf{H}|\propto 1/r^2$"),
    ("inv_r3", r"$|\mathbf{H}|\propto 1/r^3$"),
    ("zero", r"$\mathbf{H} = 0$ throughout the region named"),
]

MATCH_POOL = [
    ("wire_out", "inv_r",
     r"Outside a long straight wire carrying a steady current $I$, as a function of the distance $r$ from the wire",
     r"A circle of radius $r$ around the wire encloses all of $I$: $H_\phi = I/(2\pi r)$."),
    ("wire_in", "lin_r",
     r"Inside a long straight round wire of radius $R$ whose current is spread uniformly over its cross-section, "
     r"as a function of the distance $r \lt R$ from its axis",
     r"A circle of radius $r$ encloses the fraction $r^2/R^2$ of the current: $H_\phi = Ir/(2\pi R^2)$."),
    ("sheet", "uniform",
     r"On either side of an infinite plane sheet of uniform surface current $\mathbf{J}_s$, as a function of the "
     r"distance from the sheet",
     r"$\mathbf{H} = \tfrac12\mathbf{J}_s\times\hat n$: magnitude $J_s/2$ at every distance (the direction reverses "
     r"across the sheet)."),
    ("sol_in", "uniform",
     r"Inside an ideal solenoid, as a function of the distance from its axis",
     r"A rectangle with one long side inside and one outside gives $H = nI$, whatever the radius of the inside side."),
    ("sol_out", "zero",
     r"Outside an ideal solenoid",
     r"A rectangle with both long sides outside links no net current, so the exterior field is uniform, and it must "
     r"vanish far away: $\mathbf{H} = 0$."),
    ("tor_in", "inv_r",
     r"Inside the core of an ideal toroid, as a function of the distance $r$ from the toroid's symmetry axis",
     r"A circle inside the core encloses the $N$ inner-wall wires: $H_\phi = NI/(2\pi r)$."),
    ("tor_out", "zero",
     r"Outside the core of an ideal toroid (in its central hole, or beyond its outer wall)",
     r"Such a circle encloses either no wire at all or $N$ wires each way: $I_{\text{enc}} = 0$, and by symmetry "
     r"$H_\phi = 0$."),
    ("coax_out", "zero",
     r"Outside a long coaxial cable whose inner conductor carries $I$ and whose outer conductor carries the return "
     r"current $-I$",
     r"$I_{\text{enc}} = I - I = 0$."),
    ("coax_mid", "inv_r",
     r"In the gap between the inner conductor (current $I$) and the outer conductor (current $-I$) of a long coaxial "
     r"cable, as a function of the distance $r$ from its axis",
     r"Only the inner conductor is enclosed: $H_\phi = I/(2\pi r)$, as for a single wire."),
    ("loop_far", "inv_r3",
     r"On the axis of a small circular loop of steady current, far from the loop ($|z|$ much larger than the loop's "
     r"radius), as a function of $|z|$",
     r"$H_z = Ia^2/[2(a^2+z^2)^{3/2}] \to Ia^2/(2|z|^3)$: the loop is a magnetic dipole."),
    ("slab_in", "lin_r",
     r"Inside an infinite slab (between two parallel planes) carrying a uniform current density $\mathbf{J} = "
     r"J_0\hat z$ parallel to its faces, as a function of the distance from the slab's mid-plane",
     r"A rectangle centred on the mid-plane encloses current in proportion to its half-width: $|\mathbf{H}| = "
     r"J_0|x|$, a ramp."),
    ("slab_out", "uniform",
     r"Outside an infinite slab of thickness $W$ carrying a uniform current density $J_0$ parallel to its faces, as a "
     r"function of the distance from the slab",
     r"The rectangle encloses the slab's whole current per unit length, $J_0W$: $|\mathbf{H}| = J_0W/2$ at any "
     r"distance, as for a sheet with $J_s = J_0W$."),
    ("two_in", "uniform",
     r"Between two infinite parallel sheets carrying equal and opposite surface currents, $\mathbf{J}_s$ and "
     r"$-\mathbf{J}_s$, as a function of the distance from either sheet",
     r"The two half-fields $J_s/2$ point the same way between the sheets and add: $|\mathbf{H}| = J_s$ everywhere "
     r"there."),
    ("two_out", "zero",
     r"Outside two infinite parallel sheets carrying equal and opposite surface currents",
     r"The two half-fields point opposite ways outside and cancel: $\mathbf{H} = 0$."),
]


def tex_num(x, sig=4):
    """Number for LaTeX: 4 significant figures, with \\times10^{n} instead of Python's e-notation."""
    s = f"{x:.{sig}g}"
    if "e" in s:
        mant, ex = s.split("e")
        return mant + r"\times10^{" + str(int(ex)) + "}"
    return s


def unit_tex(axis, sgn):
    return ("+" if sgn > 0 else "-") + r"\hat{" + axis + "}"


def generate(data):
    p = data["params"]

    # ---------------- (a) matching: 5 statements, at least 4 different answers
    while True:
        sel = random.sample(MATCH_POOL, 5)
        keys = [m[1] for m in sel]
        if len(set(keys)) >= 4:
            break
    opt_text = dict(OPTIONS)
    p["options"] = [{"name": k, "text": t} for k, t in OPTIONS]
    p["match"] = [{"id": i, "key": k, "text": t, "answer": opt_text[k], "why": w} for i, k, t, w in sel]

    # ---------------- (b) toroid
    while True:
        N = random.choice([200, 250, 300, 400, 500, 600, 800, 1000])
        I = random.choice([0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8, 1.0, 1.2, 1.5])
        a_cm = random.choice([2, 2.5, 3, 4, 5, 6])
        b_cm = a_cm + random.choice([1, 1.5, 2, 3, 4])
        h_cm = random.choice([1, 1.5, 2, 2.5, 3])
        # "closely wound": the turn spacing at the outer wall, 2 pi b / N, is at most 1/2.5 of both h/2 and
        # (b - a)/4, so the discrete winding acts as the ideal toroid wherever the question probes it
        if N * h_cm >= 10 * math.pi * b_cm and N * (b_cm - a_cm) >= 20 * math.pi * b_cm:
            break
    s = random.choice([1, -1])
    frac = random.choice([0.25, 0.5, 0.75])
    r0_cm = round(a_cm + frac * (b_cm - a_cm), 4)
    a, b, h, r0 = a_cm * 1e-2, b_cm * 1e-2, h_cm * 1e-2, r0_cm * 1e-2

    H = s * N * I / (2 * math.pi * r0)
    Psi = s * MU0 * N * I * h * math.log(b / a) / (2 * math.pi)

    inner_dir = r"$" + unit_tex("z", s) + r"$"
    outer_dir = r"$" + unit_tex("z", -s) + r"$"
    if s > 0:
        faces = r"radially outward across the top face ($z = h/2$) and inward across the bottom face"
    else:
        faces = r"radially outward across the bottom face ($z = -h/2$) and inward across the top face"
    sg = "" if s > 0 else "-"
    p.update({
        "N": N, "I": f"{I:g}", "a_cm": f"{a_cm:g}", "b_cm": f"{b_cm:g}", "h_cm": f"{h_cm:g}", "r0_cm": f"{r0_cm:g}",
        "s": s, "inner_dir": inner_dir, "outer_dir": outer_dir, "faces": faces,
        "H_str": tex_num(H), "Psi_str": f"{Psi * 1e6:.4g}",
    })
    b_words = (
        r"A circle of radius $r$ about the $z$ axis inside the core, traversed along $+\hat\phi$, bounds a flat disk "
        r"pierced by the $N$ inner-wall wires, each carrying $I$ along " + inner_dir + r"; by the right-hand rule "
        r"$I_{\text{enc}} = " + sg + r"NI$. Symmetry makes $H_\phi$ the same all around the circle, so "
        r"$2\pi r\,H_\phi = I_{\text{enc}}$ and"
        r"$$H_\phi = \frac{" + sg + r"NI}{2\pi r} = \frac{" + sg + f"({N})({I:g})" + r"}{2\pi\times" + f"{r0:g}" +
        r"\ \mathrm{m}} = " + tex_num(H) + r"\ \mathrm{A/m}.$$"
        r"The flux through one turn adds up $B_\phi = \mu_0H_\phi$ over the rectangle ($dS = dr\,dz$, normal "
        r"$+\hat\phi$):"
        r"$$\Psi = \int_{-h/2}^{h/2}\!\int_a^b\frac{" + sg + r"\mu_0NI}{2\pi r}\,dr\,dz = " + sg +
        r"\frac{\mu_0NIh}{2\pi}\ln\frac{b}{a} = " + sg + r"(2\times10^{-7})" + f"({N})({I:g})({h:g})" +
        r"\ln\frac{" + f"{b_cm:g}" + "}{" + f"{a_cm:g}" + r"} = " + f"{Psi * 1e6:.4g}" + r"\ \mu\mathrm{Wb}.$$"
        r"The field is not uniform over the cross-section: it is $b/a = " + f"{b / a:.4g}" + r"$ times stronger at "
        r"the inner wall than at the outer wall, which is why the flux needs the integral rather than "
        r"$B\times$ area."
    )
    p["b_words"] = b_words

    # ---------------- (c) H at a point outside the core
    region = random.choice(["hole", "beyond", "above"])
    if region == "hole":
        rP_cm = round(a_cm * random.choice([0.4, 0.5, 0.6]), 4)
        zP_cm = 0.0
        P_desc = (r"in the toroid's central hole, at $r = " + f"{rP_cm:g}" + r"\ \mathrm{cm}$, $z = 0$")
    elif region == "beyond":
        rP_cm = b_cm + random.choice([1, 2, 3])
        zP_cm = 0.0
        P_desc = (r"beyond the outer wall, at $r = " + f"{rP_cm:g}" + r"\ \mathrm{cm}$, $z = 0$")
    else:
        rP_cm = round(a_cm + 0.5 * (b_cm - a_cm), 4)
        zP_cm = h_cm / 2 + max(1.0, h_cm / 2)
        P_desc = (r"above the core, at $r = " + f"{rP_cm:g}" + r"\ \mathrm{cm}$ (between $a$ and $b$), $z = " +
                  f"{zP_cm:g}" + r"\ \mathrm{cm}$")
    reason_none = r"the flat disk it bounds is pierced by no wire of the winding, so $I_{\text{enc}} = 0$"
    reason_cancel = (r"the flat disk it bounds is pierced by the $N$ inner-wall wires and by the $N$ outer-wall "
                     r"wires, which carry opposite currents, so $I_{\text{enc}} = NI - NI = 0$")
    right_reason, wrong_reason = ((reason_cancel, reason_none) if region == "beyond" else (reason_none, reason_cancel))
    right_key, wrong_key = (("cancel", "none") if region == "beyond" else ("none", "cancel"))
    H_full = N * I / (2 * math.pi * rP_cm * 1e-2)
    if region == "hole":
        wrong_fb = (r"The value is right but the counting is not: a circle with $r \lt a$ does not reach the inner "
                    r"wall, so no wire pierces its disk at all.")
        full_fb = (r"Count the currents through the flat disk bounded by the circle: for $r \lt a$ no wire pierces "
                   r"it. Only a circle inside the core encloses a net current $NI$.")
    elif region == "beyond":
        wrong_fb = (r"The value is right but the counting is not: a circle with $r \gt b$ encloses both walls, so "
                    r"$2N$ wires pierce its disk; it is their opposite currents that cancel.")
        full_fb = (r"For $r \gt b$ the disk is pierced by the outer-wall wires too, whose currents are opposite to "
                   r"the inner-wall ones: the net enclosed current is $NI - NI = 0$, not $NI$.")
    else:
        wrong_fb = (r"The value is right but the counting is not: the flat disk at that height lies above the core, "
                    r"and no wire of the winding passes through it.")
        full_fb = (r"At that height the circle's flat disk lies above the core and no wire pierces it. The formula "
                   r"$NI/(2\pi r)$ holds only inside the core, $|z| \lt h/2$.")
    out_choices = [
        {"key": "zero_right", "reason": right_key, "correct": "true",
         "text": (r"$\mathbf{H} = 0$. Take the circle through P centred on the $z$ axis: " + right_reason +
                  r"; by symmetry $H_\phi$ has the same value all around that circle, so $2\pi r\,H_\phi = 0$."),
         "feedback": r"Right: zero enclosed current plus the rotational symmetry of the winding force $H_\phi = 0$."},
        {"key": "zero_wrong", "reason": wrong_key, "correct": "false",
         "text": (r"$\mathbf{H} = 0$. Take the circle through P centred on the $z$ axis: " + wrong_reason +
                  r"; by symmetry $H_\phi$ has the same value all around that circle, so $2\pi r\,H_\phi = 0$."),
         "feedback": wrong_fb},
        {"key": "full", "reason": None, "correct": "false",
         "text": (r"$|\mathbf{H}| \approx " + tex_num(H_full) + r"\ \mathrm{A/m}$, the value the inside-the-core "
                  r"result gives at this radius: the circle through P still goes around the toroid's winding, just "
                  r"as a circle inside the core does."),
         "feedback": full_fb},
        {"key": "nonuniform", "reason": None, "correct": "false",
         "text": (r"$\mathbf{H} \ne 0$: $\oint\mathbf{H}\cdot d\mathbf{l} = 0$ only means that $H_\phi$ is positive "
                  r"on part of the circle and negative on the rest."),
         "feedback": (r"The winding looks the same after any rotation about the $z$ axis, so $H_\phi$ cannot differ "
                      r"from one point of a circle centred on that axis to another: $2\pi r H_\phi = 0$ forces "
                      r"$H_\phi = 0$.")},
        {"key": "dipole", "reason": None, "correct": "false",
         "text": (r"$\mathbf{H}$ is small but not zero: seen from P, the toroid acts like a magnetic dipole whose "
                  r"field falls off as $1/r^3$."),
         "feedback": (r"Each turn's dipole moment points along $\pm\hat\phi$, and around the ring these add to zero. "
                      r"For an ideal toroid Ampère's law with the symmetry gives exactly zero, not a weak dipole "
                      r"field.")},
    ]
    if region == "beyond":
        c_words = (r"The circle through P has radius $r = " + f"{rP_cm:g}" + r"\ \mathrm{cm} \gt b$; its flat disk "
                   r"is pierced by the $N$ inner-wall wires (current along " + inner_dir + r") and by the $N$ "
                   r"outer-wall wires (current along " + outer_dir + r"), so $I_{\text{enc}} = 0$.")
    elif region == "hole":
        c_words = (r"The circle through P has radius $r = " + f"{rP_cm:g}" + r"\ \mathrm{cm} \lt a$; its flat disk "
                   r"is pierced by no wire at all, so $I_{\text{enc}} = 0$.")
    else:
        c_words = (r"The circle through P lies at $z = " + f"{zP_cm:g}" + r"\ \mathrm{cm} \gt h/2$; its flat disk "
                   r"is above the core and is pierced by no wire, so $I_{\text{enc}} = 0$.")
    c_words += (r" The ideal toroid is unchanged by any rotation about the $z$ axis and its currents have no "
                r"$\hat\phi$ component, so $\mathbf{H} = H_\phi\hat\phi$ with $H_\phi$ constant on the circle: "
                r"$2\pi r H_\phi = 0$ gives $\mathbf{H} = 0$. The formula $NI/(2\pi r)$ (which would give $" +
                tex_num(H_full) + r"\ \mathrm{A/m}$ there) applies only inside the core; a field that is merely "
                r"weak (dipole-like) or that averages to zero around the circle is ruled out by the symmetry.")
    p.update({"P_desc": P_desc, "region": region, "rP_cm": rP_cm, "zP_cm": zP_cm,
              "out_choices": out_choices, "c_words": c_words})

    # ---------------- (d) direction of A near a straight wire
    axis = random.choice(["x", "y", "z"])
    sgn = random.choice([1, -1])
    e = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[axis]
    cur = [sgn * c for c in e]
    anti = [-c for c in cur]
    A_choices = [
        {"key": "parallel", "vec": cur, "correct": "true",
         "text": r"along $" + unit_tex(axis, sgn) + r"$, parallel to the current",
         "feedback": r"Right: every current element contributes $\mu_0I\,d\mathbf{l}/(4\pi R)$ along its own direction."},
        {"key": "anti", "vec": anti, "correct": "false",
         "text": (r"along $" + unit_tex(axis, -sgn) + r"$, opposite to the current: that is the sense that makes "
                  r"$\nabla\times\mathbf{A}$ circle the wire by the right-hand rule"),
         "feedback": (r"The Coulomb integral adds $\mu_0I\,d\mathbf{l}/(4\pi R)$ with a positive coefficient, so "
                      r"$\mathbf{A}$ is parallel to the current. Because $|\mathbf{A}|$ decreases away from the wire, "
                      r"its curl circles the wire in the right-hand sense.")},
        {"key": "azimuthal", "vec": None, "correct": "false",
         "text": r"around the wire, in the same sense as $\mathbf{B}$ (right-hand rule about the current)",
         "feedback": (r"That is the direction of $\mathbf{B} = \nabla\times\mathbf{A}$, not of $\mathbf{A}$. The "
                      r"Coulomb integral only adds vectors parallel to the current elements.")},
        {"key": "radial", "vec": None, "correct": "false",
         "text": r"radially away from the wire, perpendicular to both the current and $\mathbf{B}$",
         "feedback": (r"No current flows radially, and the Coulomb integral only adds vectors parallel to the "
                      r"current elements.")},
        {"key": "zero", "vec": None, "correct": "false",
         "text": r"nowhere: $\mathbf{A} = 0$ near the wire, because the Coulomb gauge requires $\nabla\cdot\mathbf{A} = 0$",
         "feedback": (r"A field along the wire whose size depends only on the distance from the wire has zero "
                      r"divergence; the Coulomb gauge does not force $\mathbf{A}$ to vanish.")},
    ]
    d_words = (
        r"In $\mathbf{A} = \int\mu_0\mathbf{J}\,dV'/(4\pi|\mathbf{r}-\mathbf{r}'|)$ every current element contributes a "
        r"vector parallel to itself with a positive weight $1/|\mathbf{r}-\mathbf{r}'|$, and near the wire its own "
        r"elements dominate, so $\mathbf{A}$ points along $" + unit_tex(axis, sgn) + r"$, the direction of the current "
        r"(for a long wire $|\mathbf{A}| \approx \frac{\mu_0I}{2\pi}\ln\frac{D}{\rho}$, with $\rho$ the distance from "
        r"the wire and $D$ the distance to the return path). Its curl, $\mathbf{B}$, circles the wire; "
        r"$\mathbf{A}$ itself does not, and it is not radial. The Coulomb gauge is satisfied because $\mathbf{A}$ "
        r"does not change along its own direction."
    )
    p.update({"wire_axis": axis, "wire_sgn": sgn, "wire_dir": r"$" + unit_tex(axis, sgn) + r"$",
              "A_choices": A_choices, "d_words": d_words})

    # ---------------- (e) checkbox: 5 statements, 1-4 true
    ratio = b / a
    ratio_true = random.random() < 0.5
    stated = ratio if ratio_true else ratio ** 2
    pool = [
        ("div_curl", True,
         r"Any field of the form $\mathbf{B} = \nabla\times\mathbf{A}$ satisfies $\nabla\cdot\mathbf{B} = 0$, "
         r"whatever $\mathbf{A}$ is.",
         r"$\nabla\cdot(\nabla\times\mathbf{A}) = 0$ identically: the mixed partial derivatives cancel in pairs."),
        ("gauge", True,
         r"Replacing $\mathbf{A}$ by $\mathbf{A} + \nabla\lambda$, for any scalar function $\lambda$, leaves "
         r"$\mathbf{B}$ unchanged.",
         r"$\nabla\times\nabla\lambda = 0$; this is the gauge freedom of $\mathbf{A}$."),
        ("poisson", True,
         r"In the Coulomb gauge ($\nabla\cdot\mathbf{A} = 0$), each Cartesian component of $\mathbf{A}$ obeys "
         r"Poisson's equation: $\nabla^2A_x = -\mu_0J_x$, and likewise for $y$ and $z$.",
         r"$\nabla\times\nabla\times\mathbf{A} = \nabla(\nabla\cdot\mathbf{A}) - \nabla^2\mathbf{A} = \mu_0\mathbf{J}$, "
         r"and the first term vanishes in this gauge."),
        ("units", True,
         r"The SI unit of $\mathbf{A}$ is the weber per metre (equivalently, tesla-metre).",
         r"$\mathbf{B} = \nabla\times\mathbf{A}$ has the units of $\mathbf{A}$ divided by metres, and "
         r"$1\ \mathrm{T} = 1\ \mathrm{Wb/m^2}$."),
        ("stokes", True,
         r"The circulation $\oint_C\mathbf{A}\cdot d\mathbf{l}$ around a closed path $C$ equals the magnetic flux "
         r"through any surface bounded by $C$ (oriented by the right-hand rule).",
         r"Stokes' theorem: $\oint_C\mathbf{A}\cdot d\mathbf{l} = \int_S(\nabla\times\mathbf{A})\cdot d\mathbf{S} = "
         r"\int_S\mathbf{B}\cdot d\mathbf{S}$."),
        ("toroid_h", True,
         r"Doubling the height $h$ of this toroid's core (same $N$, $I$, $a$, $b$) leaves $H_\phi$ inside the core "
         r"unchanged and doubles the flux through each turn.",
         r"$H_\phi = NI/(2\pi r)$ contains no $h$, while $\Psi = \mu_0NIh\ln(b/a)/(2\pi)$ is proportional to $h$."),
        ("ratio", ratio_true,
         r"In this toroid, $|\mathbf{H}|$ just inside the inner wall ($r = a$) is $" + f"{stated:.4g}" +
         r"$ times $|\mathbf{H}|$ just inside the outer wall ($r = b$).",
         r"$|\mathbf{H}| \propto 1/r$, so the ratio is $b/a = " + f"{ratio:.4g}" + r"$" +
         ("." if ratio_true else r", not $(b/a)^2$.")),
        ("unique", False,
         r"Since $\mathbf{B} = \nabla\times\mathbf{A}$, a given magnetic field determines $\mathbf{A}$ uniquely.",
         r"$\mathbf{A}$ and $\mathbf{A} + \nabla\lambda$ give the same $\mathbf{B}$; a gauge condition such as "
         r"$\nabla\cdot\mathbf{A} = 0$ is what narrows the choice."),
        ("gauge_law", False,
         r"The Coulomb-gauge condition $\nabla\cdot\mathbf{A} = 0$ is required by Maxwell's equations: a vector "
         r"potential with $\nabla\cdot\mathbf{A} \ne 0$ would give a field with $\nabla\cdot\mathbf{B} \ne 0$.",
         r"$\nabla\cdot(\nabla\times\mathbf{A}) = 0$ for every $\mathbf{A}$; $\nabla\cdot\mathbf{A} = 0$ is a "
         r"convenient choice, not a law."),
        ("parallel", False,
         r"The vector potential always points in the same direction as $\mathbf{B}$.",
         r"Near a straight wire $\mathbf{A}$ is along the current while $\mathbf{B}$ circles the wire."),
        ("toroid_uniform", False,
         r"Inside this toroid's core, $|\mathbf{H}|$ is the same at every point of the cross-section.",
         r"$H_\phi = NI/(2\pi r)$ is largest at the inner wall and smallest at the outer wall."),
        ("ampere_alone", False,
         r"Ampère's law, applied to one well-chosen closed path, gives $\mathbf{H}$ for any steady current "
         r"distribution, even one with no symmetry.",
         r"One circulation is one number. Only when symmetry fixes the direction of $\mathbf{H}$ and makes "
         r"$|\mathbf{H}|$ constant along the path can that number be solved for $\mathbf{H}$; otherwise use "
         r"Biot–Savart."),
        ("a_zero", False,
         r"Wherever $\mathbf{B} = 0$ throughout a region, $\mathbf{A} = 0$ in that region too.",
         r"Any constant vector, or any gradient $\nabla\lambda$, has zero curl; outside an ideal solenoid "
         r"$\mathbf{B} = 0$, yet $\oint\mathbf{A}\cdot d\mathbf{l}$ around it equals the flux inside, so "
         r"$\mathbf{A} \ne 0$ there."),
    ]
    while True:
        sel = random.sample(pool, 5)
        nt = sum(1 for x in sel if x[1])
        if 1 <= nt <= 4:
            break
    p["facts"] = [{"key": k, "truth": t, "correct": "true" if t else "false", "verdict": "True" if t else "False",
                   "text": txt, "why": why, "ratio_stated": stated if k == "ratio" else None}
                  for k, t, txt, why in sel]

    data["correct_answers"]["H"] = H                 # A/m, signed along +phi
    data["correct_answers"]["Psi"] = Psi * 1e6       # uWb, normal +phi
