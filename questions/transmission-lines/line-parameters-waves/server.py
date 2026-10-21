"""Lecture 27 concept question: line parameters of a coax or a parallel-plate line, and V/I waves on it.

eps0 = 1e-9/(36 pi), mu0 = 4 pi 1e-7 (so c = 3e8, eta0 = 120 pi); filling mu = mu0, eps = eps_r eps0.
Geometrical factor GF: plates W/d, coax 2 pi/ln(b/a).  C = eps GF, L = mu/GF, v = 1/sqrt(LC), Z0 = sqrt(L/C).
Wave V(z,t) = V0 exp(-u^2) with u = (t -+ z/v - t0)/tau or (z -+ v t - z0)/ell; direction s = +-1.
I = s V/Z0 (current along +z on the reference conductor), p = V I = s V^2/Z0 (power toward +z, W); I in mA.
Scaling part: factors (f_v, f_Z0) for a stated change of the line.
"""
import math
import random

MU0 = 4e-7 * math.pi
EPS0 = 1e-9 / (36 * math.pi)
C0 = 3e8
ETA0 = 120 * math.pi

FILLS = [("air", 1.0), ("polyethylene", 2.25), ("polystyrene", 2.56), ("an epoxy resin", 4.0),
         ("a glass", 6.25), ("alumina", 9.0)]
COAX_A = [0.3, 0.4, 0.45, 0.6, 0.8, 1.0, 1.2, 1.5]          # mm
COAX_RATIO = [2, 2.5, 3, 4, 5, 6, 8]
PLATE_W = [8, 10, 12, 15, 16, 20, 24, 25, 30]                # mm
PLATE_D = [0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]                # mm
V0S = [3, 4, 5, 6, 8, 10, 12, 15]
US = [0.0, 0.5, -0.5, 1.0, -1.0]


def num(x, sig=4):
    if abs(x) < 1e-12:
        return "0"
    s = f"{x:.{sig}g}"
    if "e" in s:
        m, e = s.split("e")
        return f"{m}\\times10^{{{int(e)}}}"
    return s


def sci(x, sig=4):
    m, e = f"{x:.{sig - 1}e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def gnum(x):
    """Exact short decimal for given data."""
    s = f"{x:.6g}"
    return s


def line_params(geom, dims, er):
    if geom == "coax":
        a, b = dims
        GF = 2 * math.pi / math.log(b / a)
    else:
        W, dd = dims
        GF = W / dd
    eps = er * EPS0
    Cp = eps * GF
    Lp = MU0 / GF
    return Lp, Cp, 1 / math.sqrt(Lp * Cp), math.sqrt(Lp / Cp)


def fac_text(x):
    if abs(x - 1) < 1e-9:
        return "unchanged"
    if abs(x - 2) < 1e-9:
        return "doubled"
    if abs(x - 0.5) < 1e-9:
        return "halved"
    return "multiplied by " + num(round(x, 9), 3)   # same text for equal factors


def generate(data):
    geom = random.choice(["coax", "plates"])
    fill, er = random.choice(FILLS)
    if geom == "coax":
        a = random.choice(COAX_A)
        ratio = random.choice(COAX_RATIO)
        b = round(a * ratio, 6)
        dims = (a * 1e-3, b * 1e-3)
        dims_mm = (a, b)
    else:
        while True:
            W = random.choice(PLATE_W)
            dd = random.choice(PLATE_D)
            if W / dd >= 8:
                break
        dims = (W * 1e-3, dd * 1e-3)
        dims_mm = (W, dd)
    Lp, Cp, v, Z0 = line_params(geom, dims, er)
    eta = ETA0 / math.sqrt(er)
    GF = (2 * math.pi / math.log(dims[1] / dims[0])) if geom == "coax" else dims[0] / dims[1]

    # ------------------------------------------------------------------ the wave
    s = random.choice([1, -1])
    form = random.choice(["t", "z"])
    V0 = random.choice(V0S) * random.choice([1, 1, -1])
    u = random.choice(US)
    vns = v * 1e-9                                       # m per ns
    if form == "t":
        tau = random.choice([4, 5, 8, 10, 20])           # ns
        if s > 0:
            t0 = random.choice([20, 30, 40, 50])
            k = random.choice([10, 20, 25, 40, 50, 60, 75, 80, 100])
            t1 = k + t0 + u * tau
        else:
            t0 = random.choice([150, 200, 250, 300])
            k = random.choice([kk for kk in [10, 20, 25, 40, 50, 60, 75, 80, 100] if kk <= t0 - 30])
            t1 = t0 + u * tau - k
        z1 = vns * k
        inner = (r"\frac{t " + ("-" if s > 0 else "+") + r" z/v - " + f"{t0}" + r"\ \mathrm{ns}}{" + f"{tau}" + r"\ \mathrm{ns}}")
        width_txt = f"{tau}\\ \\mathrm{{ns}}"
    else:
        ell = random.choice([0.5, 1.0, 1.5, 2.0, 3.0])   # m
        t1 = random.choice([10, 20, 25, 40, 50])
        if s > 0:
            while True:
                z0 = random.choice([0, 1, 2, 5])
                z1 = z0 + vns * t1 + u * ell
                if z1 > 0.2:
                    break
            inner = (r"\frac{z - vt" + (f" - {z0}\\ \\mathrm{{m}}" if z0 else "") + r"}{" + gnum(ell) + r"\ \mathrm{m}}")
        else:
            z0 = random.choice([20, 30, 40, 50])
            z1 = z0 - vns * t1 + u * ell
            inner = r"\frac{z + vt - " + f"{z0}" + r"\ \mathrm{m}}{" + gnum(ell) + r"\ \mathrm{m}}"
        width_txt = gnum(ell) + r"\ \mathrm{m}"
    z1 = round(z1, 6)
    t1 = round(t1, 6)
    V_tex = f"{V0}\\,\\exp\\!\\left[-\\left({inner}\\right)^{{2}}\\right]\\ \\mathrm{{V}}"
    V1 = V0 * math.exp(-u * u)
    I1 = s * V1 / Z0
    p1 = s * V1 * V1 / Z0

    # ------------------------------------------------------------------ direction MC
    up = "+z" if s > 0 else "-z"
    dn = "-z" if s > 0 else "+z"
    if form == "t":
        why = (r"Holding $t " + ("-" if s > 0 else "+") + r" z/v$ fixed while $t$ grows needs $z$ to " +
               ("grow" if s > 0 else "shrink") + f": the shape moves toward ${up}$.")
    else:
        why = (r"Holding $z " + ("-" if s > 0 else "+") + r" vt$ fixed while $t$ grows needs $z$ to " +
               ("grow" if s > 0 else "shrink") + f": the shape moves toward ${up}$.")
    dirs = [
        {"id": "+z", "correct": "true" if s > 0 else "false", "text": r"toward $+z$",
         "feedback": "Correct." if s > 0 else why},
        {"id": "-z", "correct": "true" if s < 0 else "false", "text": r"toward $-z$",
         "feedback": "Correct." if s < 0 else why},
        {"id": "stand", "correct": "false", "text": r"neither: it oscillates in place, like a standing wave",
         "feedback": r"$V$ depends on $z$ and $t$ only through one combination, so its shape moves rigidly. A standing pattern is a function of $z$ times a function of $t$."},
        {"id": "sign", "correct": "false", "text": r"it cannot be told without knowing the sign of the current",
         "feedback": r"A single d'Alembert wave's argument fixes its direction, whatever the sign of its amplitude; the current then follows from $V$."},
    ]

    # ------------------------------------------------------------------ scaling MC
    if geom == "coax":
        mods = ["scale", "fill", "outer"]
    else:
        mods = ["scale", "fill", "width", "gap"]
    mod = random.choice(mods)
    if mod == "scale":
        kf = random.choice([2, 3])
        if geom == "coax":
            mod_txt = f"both radii are multiplied by {kf}, with the same filling"
        else:
            mod_txt = f"the width and the gap are both multiplied by {kf}, with the same filling"
        dims2 = (dims[0] * kf, dims[1] * kf)
        er2 = er
    elif mod == "fill":
        fill2, er2 = random.choice([fx for fx in FILLS if fx[1] != er])
        mod_txt = (f"the filling is replaced by {fill2} ($\\epsilon_r = {gnum(er2)}$, $\\mu_r = 1$), "
                   "with the same conductors")
        dims2 = dims
    elif mod == "outer":
        b2 = dims_mm[1] ** 2 / dims_mm[0]
        mod_txt = (f"the outer radius is increased to $b' = b^2/a = {num(b2, 5)}\\ \\mathrm{{mm}}$, "
                   "with the same inner conductor and filling")
        dims2 = (dims[0], dims[1] ** 2 / dims[0])
        er2 = er
    elif mod == "width":
        mod_txt = "the plate width is doubled, with the same gap and filling"
        dims2 = (2 * dims[0], dims[1])
        er2 = er
    else:
        mod_txt = "the gap is doubled, with the same plate width and filling"
        dims2 = (dims[0], 2 * dims[1])
        er2 = er
    _, _, v2, Z02 = line_params(geom, dims2, er2)
    fv, fz = v2 / v, Z02 / Z0
    if mod == "fill":
        rr = fv
        cands = [((1, 1), r"Both change: $v = 1/\sqrt{\mu\epsilon}$ and $Z_0 = \eta/\mathrm{GF}$ with $\eta = \sqrt{\mu/\epsilon}$."),
                 ((rr, 1), r"$Z_0 = \eta/\mathrm{GF}$ also depends on the filling, through $\eta = \sqrt{\mu/\epsilon}$."),
                 ((1, rr), r"The speed depends on the filling: $v = 1/\sqrt{\mathcal{L}\mathcal{C}} = 1/\sqrt{\mu\epsilon}$."),
                 ((rr, 1 / rr), r"$\mathcal{C}\propto\epsilon$ while $\mathcal{L}$ is unchanged, so $Z_0 = \sqrt{\mathcal{L}/\mathcal{C}}\propto 1/\sqrt{\epsilon}$, the same factor as $v$.")]
    elif mod == "scale":
        cands = [((1, kf), r"$\mathcal{L}$ and $\mathcal{C}$ depend on the cross-section only through the ratio " + ("$b/a$" if geom == "coax" else "$W/d$") + ", which is unchanged."),
                 ((1, 1 / kf), r"$\mathcal{L}$ and $\mathcal{C}$ depend on the cross-section only through the ratio " + ("$b/a$" if geom == "coax" else "$W/d$") + ", which is unchanged."),
                 ((1 / kf, 1), r"$v = 1/\sqrt{\mu\epsilon}$ depends on the filling only, not on the size of the line."),
                 ((kf, kf), r"Neither changes: the geometrical factor depends on a ratio of dimensions, and $v$ on the filling only.")]
    else:
        c2 = math.sqrt(1 / fz)
        cands = [((1, 1 / fz), r"Check which way the geometrical factor moves: " + (r"$Z_0 = (\eta/2\pi)\ln(b/a)$." if geom == "coax" else r"$Z_0 = \eta\,d/W$.")),
                 ((1, 1), r"$Z_0 = \eta/\mathrm{GF}$ follows the geometrical factor, which changes here."),
                 ((c2, fz), r"$\mathcal{L}$ and $\mathcal{C}$ change by inverse factors, so $\mathcal{L}\mathcal{C} = \mu\epsilon$ and $v$ stay the same."),
                 ((1, math.sqrt(fz)), r"$Z_0$ is proportional to " + (r"$\ln(b/a)$" if geom == "coax" else r"$d/W$") + ", not to its square root.")]
    scl = [{"fv": fv, "fz": fz, "correct": "true", "feedback": "Correct.",
            "text": f"$v$ {fac_text(fv)}; $Z_0$ {fac_text(fz)}"}]
    for (a1, a2), fb in cands:
        txt = f"$v$ {fac_text(a1)}; $Z_0$ {fac_text(a2)}"
        if all(abs(a1 - c["fv"]) > 1e-9 or abs(a2 - c["fz"]) > 1e-9 for c in scl) and txt not in [c["text"] for c in scl]:
            scl.append({"fv": a1, "fz": a2, "correct": "false", "feedback": fb, "text": txt})
    random.shuffle(scl)

    # ------------------------------------------------------------------ checkbox pool
    facts = [
        ("LC_mue", True, r"For any two-conductor line filled with one homogeneous medium, $\mathcal{L}\mathcal{C} = \mu\epsilon$.",
         r"$\mathcal{C} = \epsilon\,\mathrm{GF}$ and $\mathcal{L} = \mu/\mathrm{GF}$ share one geometrical factor, which cancels in the product."),
        ("VI_speed", True, r"The voltage wave and the current wave on a lossless line travel at the same speed.",
         r"Both are d'Alembert waves of the same argument, $t\mp z/v$, with $v = 1/\sqrt{\mathcal{L}\mathcal{C}}$."),
        ("I_sign", True, r"For a single wave travelling toward $+z$, $I = +V/Z_0$; for a single wave travelling toward $-z$, $I = -V/Z_0$.",
         r"Putting $V = f(t\mp z/v)$ into $-\partial I/\partial z = \mathcal{C}\,\partial V/\partial t$ gives $I = \pm f/Z_0$."),
        ("geom_med", True, r"$Z_0$ depends on both the cross-section and the filling, while $v$ depends only on the filling.",
         r"$Z_0 = \eta/\mathrm{GF}$ and $v = 1/\sqrt{\mu\epsilon}$."),
        ("ladder", True, r"A short section $\Delta z\ll\lambda$ of line acts like a series inductor $\mathcal{L}\Delta z$ followed by a shunt capacitor $\mathcal{C}\Delta z$.",
         r"That is the telegrapher's equations written over $\Delta z$: an inductive drop along the wire and a capacitor current across the gap."),
        ("equipart", True, r"For a single travelling wave, the electric and magnetic energies stored per metre of line are equal.",
         r"$\tfrac12\mathcal{L}I^2 = \tfrac12\mathcal{L}V^2/Z_0^2 = \tfrac12\mathcal{C}V^2$."),
        ("sum_ratio", True, r"Where a forward and a backward wave are both present, $V/I$ at a point is in general neither $+Z_0$ nor $-Z_0$.",
         r"$V/I = Z_0(f+g)/(f-g)$, which takes any value."),
        ("Z0_length", False, r"Of two lines with the same cross-section and filling, the longer one has the larger $Z_0$.",
         r"$Z_0 = \sqrt{\mathcal{L}/\mathcal{C}}$ is built from per-metre quantities; the length does not enter."),
        ("Z0_resist", False, r"$Z_0$ is a resistance: a wave on a lossless line loses power at the rate $I^2Z_0$ per metre.",
         r"A lossless line dissipates nothing; $Z_0$ is only the ratio $V/I$ of one travelling wave."),
        ("double_W", False, r"Doubling the plate width $W$ of a parallel-plate line, with the gap fixed, doubles its $Z_0$.",
         r"$Z_0 = \eta\,d/W$: doubling $W$ halves it ($\mathcal{C}$ doubles, $\mathcal{L}$ halves)."),
        ("coax_speed", False, r"Signals on a coax travel faster when the ratio $b/a$ of its radii is larger.",
         r"$v = 1/\sqrt{\mathcal{L}\mathcal{C}} = 1/\sqrt{\mu\epsilon}$ whatever $b/a$."),
        ("back_Ipos", False, r"A single wave travelling toward $-z$ with $V \gt 0$ has $I \gt 0$, with $V$ and $I$ defined by the reference directions above.",
         r"A backward wave has $I = -V/Z_0$, so $V \gt 0$ means $I \lt 0$."),
        ("shrink_C", False, r"Halving both radii of a coax, with the same filling, halves its capacitance per metre.",
         r"$\mathcal{C} = 2\pi\epsilon/\ln(b/a)$ depends only on the ratio $b/a$."),
        ("Z0_eta", False, r"For every line, $Z_0$ equals the intrinsic impedance $\eta$ of the filling.",
         r"$Z_0 = \eta/\mathrm{GF}$; for plates $\eta\,d/W$, for a coax $(\eta/2\pi)\ln(b/a)$."),
        ("power_cross", False, r"Where a forward wave $f$ and a backward wave $g$ are both present, the power $VI$ contains a cross term $2fg/Z_0$.",
         r"$VI = (f+g)(f-g)/Z_0 = (f^2-g^2)/Z_0$: the cross terms cancel, so each wave carries its own power."),
    ]
    facts_list = [{"id": fid, "correct": "true" if t else "false", "text": txt,
                   "feedback": ("True: " if t else "False: ") + why2, "verdict": "true" if t else "false", "why": why2}
                  for fid, t, txt, why2 in facts]
    random.shuffle(facts_list)

    # ------------------------------------------------------------------ text
    if geom == "coax":
        geom_txt = (f"a coaxial cable with inner-conductor radius $a = {gnum(dims_mm[0])}\\ \\mathrm{{mm}}$ and outer-conductor "
                    f"inner radius $b = {gnum(dims_mm[1])}\\ \\mathrm{{mm}}$")
        conv_txt = (r"$V(z,t)$ is the potential of the inner conductor relative to the outer one, and $I(z,t)$ is the current "
                    r"along $+z$ on the inner conductor (the outer conductor carries the return current).")
        L_formula = r"\mathcal{L} = \frac{\mu_0}{2\pi}\ln\frac{b}{a}"
        C_formula = r"\mathcal{C} = \frac{2\pi\epsilon}{\ln(b/a)}"
        GF_txt = r"\mathrm{GF} = 2\pi/\ln(b/a) = " + num(GF)
        Z_formula = r"Z_0 = \frac{\eta}{2\pi}\ln\frac{b}{a}"
    else:
        geom_txt = (f"two perfectly conducting plates of width $W = {gnum(dims_mm[0])}\\ \\mathrm{{mm}}$, at $x = 0$ and "
                    f"$x = d = {gnum(dims_mm[1])}\\ \\mathrm{{mm}}$ (neglect fringing)")
        conv_txt = (r"$V(z,t) = \int_0^d E_x\,dx$ is the potential of the plate at $x = 0$ relative to the plate at $x = d$, and "
                    r"$I(z,t)$ is the current along $+z$ on the plate at $x = 0$ (the plate at $x = d$ carries the return current).")
        L_formula = r"\mathcal{L} = \mu_0\frac{d}{W}"
        C_formula = r"\mathcal{C} = \epsilon\frac{W}{d}"
        GF_txt = r"\mathrm{GF} = W/d = " + num(GF)
        Z_formula = r"Z_0 = \eta\frac{d}{W}"
    fill_txt = "air" if er == 1 else f"{fill}"

    p = data["params"]
    p.update({
        "geom": geom, "is_coax": geom == "coax", "is_plates": geom == "plates",
        "dim1_mm": dims_mm[0], "dim2_mm": dims_mm[1], "eps_r": er, "fill": fill_txt, "er_str": gnum(er),
        "geom_txt": geom_txt, "conv_txt": conv_txt,
        "V_tex": V_tex, "form": form, "z1_str": gnum(z1), "t1_str": gnum(t1), "width_txt": width_txt,
        "L_formula": L_formula, "C_formula": C_formula, "Z_formula": Z_formula, "GF_txt": GF_txt,
        "L_str": num(Lp * 1e9), "C_str": num(Cp * 1e12), "v_str": num(v / 1e8), "Z0_str": num(Z0),
        "eta_str": num(eta), "rter": num(math.sqrt(er)), "LC_str": sci(Lp * Cp), "mue_str": sci(MU0 * er * EPS0),
        "V0": V0, "u_str": num(u + 0.0), "u2_str": num(u * u), "V1_str": num(V1), "I1_str": num(I1 * 1e3), "p1_str": num(p1),
        "up": up, "s_sign": "+" if s > 0 else "-", "dir_why": why,
        "mod_txt": mod_txt, "fv_str": fac_text(fv), "fz_str": fac_text(fz), "mod": mod,
        "v2_str": num(v2 / 1e8), "Z02_str": num(Z02),
        "dirs_list": dirs, "scl_list": scl, "facts_list": facts_list,
    })
    ca = data["correct_answers"]
    ca["Lp"] = Lp * 1e9
    ca["Cp"] = Cp * 1e12
    ca["v"] = v / 1e8
    ca["Z0"] = Z0
    ca["V1"] = float(V1)
    ca["I1"] = float(I1 * 1e3)
    ca["p1"] = float(p1)
