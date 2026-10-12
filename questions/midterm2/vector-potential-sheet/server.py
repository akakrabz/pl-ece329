"""Vector potential A = a_hat k f(u):  f = |u| (a current sheet on u = 0)  or  f = u^2 (a uniform volume current).
Lectures 13 and 15. Family: Summer 2019 HE2 #2b (A = x_hat |z|: show div A = 0, find B, find the sheet current).
Changed: other orientations (the exam's two are excluded), a coefficient with units and a sign, the u^2 variant,
B asked component by component at two points, the direction MC, and a time-varying A with Phi = 0 (Lecture 15).

With s_c = (u_hat x a_hat) . c_hat (c = the third axis):
  B = curl(A_a(u) a_hat) = A_a'(u) (u_hat x a_hat) = s_c A_a'(u) c_hat
    |u|: B = s_c k sgn(u) c_hat (uniform on each side)      u^2: B = 2 s_c k u c_hat
  |u|: J_s = u_hat x (H(0+) - H(0-)) = (2 s_c k/mu0) (u_hat x c_hat) = -(2k/mu0) a_hat      (u_hat x c_hat = -s_c a_hat)
  u^2: J = curl H = H_c'(u) (u_hat x c_hat) = -(2k/mu0) a_hat (uniform)
  -> J antiparallel to A in both cases (laplacian A = -mu0 J).   div A = dA_a/da = 0.
  (e) A(t) = a_hat (k + beta t) f(u), Phi = 0:  E = -dA/dt = -beta f(u) a_hat.
Units: k in mT (|u|) or mT/m (u^2), coordinates in m -> A in mWb/m, B in mT; H[kA/m] = B[mT] * 1e-6/mu0;
beta in mT/s or mT/(m s) -> E in mV/m.
"""
import math
import random

MU0 = 4e-7 * math.pi
AXES = "xyz"
COMBOS = [("x", "y"), ("y", "x"), ("z", "x"), ("z", "y")]      # (a = direction of A, u = the coordinate); exam (x,z), (y,z) excluded


def hat(a):
    return r"\hat{" + a + "}"


def num(x, p=4):
    x = float(f"{float(x):.{p}g}") + 0.0
    return f"{x:g}"


def dir_tex(s, a):
    return ("+" if s > 0 else "-") + hat(a)


def cyc(k):
    """(i, j) with (i, j, k) a cyclic permutation of (x, y, z)."""
    n = AXES.index(k)
    return AXES[(n + 1) % 3], AXES[(n + 2) % 3]


def curl_tex(F, k):
    i, j = cyc(k)
    return f"(\\nabla\\times\\mathbf{{{F}}})_{k} = \\partial {F}_{j}/\\partial {i} - \\partial {F}_{i}/\\partial {j}"


def generate(data):
    a, u = random.choice(COMBOS)
    cax = next(x for x in AXES if x not in (a, u))
    s_c = 1 if (AXES.index(a) - AXES.index(u)) % 3 == 1 else -1      # u_hat x a_hat = s_c c_hat
    mode = random.choice(["abs", "sq"])
    ah, uh, ch = hat(a), hat(u), hat(cax)

    if mode == "abs":
        k = random.choice([1, 1.5, 2, 2.5, 3, 4, 5, 6]) * random.choice([1, -1])           # mT
        beta = random.choice([2, 4, 5, 10, 20]) * random.choice([1, -1])                    # mT/s
        f_tex, f = f"\\lvert {u}\\rvert", (lambda x: abs(x))
        k_unit, beta_unit = r"\mathrm{mT}", r"\mathrm{mT/s}"
    else:
        k = random.choice([2, 3, 4, 5, 6, 8, 10]) * random.choice([1, -1])                  # mT/m
        beta = random.choice([5, 10, 20, 40, 50]) * random.choice([1, -1])                  # mT/(m s)
        f_tex, f = f"{u}^2", (lambda x: x * x)
        k_unit, beta_unit = r"\mathrm{mT/m}", r"\mathrm{mT/(m\,s)}"

    # points: P1 on the +u side, P2 on the -u side (metres)
    others = [-1.2, -0.8, -0.5, -0.3, 0, 0.2, 0.4, 0.6, 1.0, 1.5]
    u1 = random.choice([0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.2, 1.5])
    u2 = -random.choice([0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.2, 1.5])
    P1 = {x: random.choice(others) for x in AXES}
    P2 = {x: random.choice(others) for x in AXES}
    P1[u], P2[u] = u1, u2

    def B_at(uu):                      # mT, dict of components
        Bc = s_c * (k * (1 if uu > 0 else -1) if mode == "abs" else 2 * k * uu)
        return {x: (Bc if x == cax else 0.0) for x in AXES}

    B1, B2 = B_at(u1), B_at(u2)
    Jmag = 2 * abs(k) * 1e-6 / MU0                     # kA/m (sheet) or kA/m^2 (volume)
    sJ = -1 if k > 0 else 1                            # J along sJ * a_hat: antiparallel to A
    E1 = -beta * f(u1)                                 # mV/m, component along +a

    # ------------------------------------------------------------------ (a) div A
    dAdu = f"{num(k)}\\,\\mathrm{{sgn}}({u})" if mode == "abs" else f"{num(2 * k)}\\,{u}"
    lap = f"{num(2 * k)}\\,\\delta({u})" if mode == "abs" else f"{num(2 * k)}"
    div_choices = [
        {"text": f"$\\nabla\\cdot\\mathbf{{A}} = 0$ everywhere" + (f", including the plane ${u} = 0$" if mode == "abs" else "")
                 + ": this $\\mathbf{A}$ is in the Coulomb gauge.",
         "correct": "true", "claim": "zero",
         "feedback": f"Right: $\\nabla\\cdot\\mathbf{{A}} = \\partial A_{a}/\\partial {a}$, and $A_{a}$ depends only on ${u}$."},
        {"text": f"$\\nabla\\cdot\\mathbf{{A}} = {dAdu}$ mT" + ("" if mode == "abs" else f" (with ${u}$ in m)"),
         "correct": "false", "claim": "dAdu",
         "feedback": f"That is $\\partial A_{a}/\\partial {u}$, which enters the curl (it is $\\pm B$), not the divergence."},
        {"text": f"$\\nabla\\cdot\\mathbf{{A}} = {lap}$" + (" mT/m" if mode == "sq" else ""),
         "correct": "false", "claim": "laplacian",
         "feedback": f"That is $\\nabla^2 A_{a} = \\partial^2 A_{a}/\\partial {u}^2$, the quantity that gives the current through $\\nabla^2\\mathbf{{A}} = -\\mu_0\\mathbf{{J}}$."},
        {"text": "It cannot be zero, because $\\lvert\\mathbf{A}\\rvert$ grows without bound with distance from the plane $" + u + " = 0$.",
         "correct": "false", "claim": "growth",
         "feedback": f"Growth of $\\lvert\\mathbf{{A}}\\rvert$ along ${uh}$ says nothing about the divergence: $\\mathbf{{A}}$ points along ${ah}$ and does not change along ${ah}$."},
    ]
    if mode == "abs":
        div_choices.append(
            {"text": f"It is zero for ${u} \\ne 0$ but undefined on ${u} = 0$, where $\\lvert {u}\\rvert$ has a kink.",
             "correct": "false", "claim": "kink",
             "feedback": f"The kink is in the ${u}$ direction; the divergence only differentiates $A_{a}$ with respect to ${a}$, and $A_{a}$ does not depend on ${a}$ at all."})
    random.shuffle(div_choices)

    # ------------------------------------------------------------------ (d) direction of the current
    dir_choices = [
        {"text": f"${dir_tex(sJ, a)}$: antiparallel to $\\mathbf{{A}}$", "correct": "true", "vec": [sJ if x == a else 0 for x in AXES],
         "feedback": "Right: $\\nabla^2\\mathbf{A} = -\\mu_0\\mathbf{J}$, and here $\\nabla^2 A_" + a + "$ has the sign of $k$."},
        {"text": f"${dir_tex(-sJ, a)}$: parallel to $\\mathbf{{A}}$", "correct": "false", "vec": [-sJ if x == a else 0 for x in AXES],
         "feedback": "Mind the minus sign in $\\nabla^2\\mathbf{A} = -\\mu_0\\mathbf{J}$; the jump in $\\mathbf{H}$ (or $\\nabla\\times\\mathbf{H}$) gives the same answer."},
        {"text": f"$\\pm{ch}$: along the field lines of $\\mathbf{{B}}$", "correct": "false",     # no sign: B's sign is part (b)'s answer
         "vec": "alongB",
         "feedback": "The current is perpendicular to $\\mathbf{B}$ here: $\\mathbf{J} = \\nabla\\times\\mathbf{H}$, and $\\mathbf{H}$ has only a " + f"${cax}$" + "-component that varies with " + f"${u}$" + "."},
        {"text": f"$\\pm{uh}$: straight across the planes ${u} = $ const", "correct": "false", "vec": "normal",
         "feedback": f"$\\mathbf{{H}}$ depends only on ${u}$, so $\\nabla\\times\\mathbf{{H}}$ has no ${u}$-component; a current crossing the planes would also violate $\\nabla\\cdot\\mathbf{{J}} = 0$ here."},
    ]
    if mode == "sq":
        sr = random.choice([1, -1]) * sJ               # random sign on the u > 0 side, so this distractor does not point at the key
        dir_choices.append(
            {"text": f"${dir_tex(sr, a)}$ for ${u} \\gt 0$ and ${dir_tex(-sr, a)}$ for ${u} \\lt 0$: it reverses where $\\mathbf{{B}}$ does",
             "correct": "false", "vec": "reverses",
             "feedback": f"$\\mathbf{{J}} = \\nabla\\times\\mathbf{{H}}$ depends on the slope $dH_{cax}/d{u}$, which is the same on both sides, not on the sign of $H_{cax}$."})
    else:
        dir_choices.append(
            {"text": "There is no current anywhere: $\\mathbf{B}$ is uniform on each side, so $\\nabla\\times\\mathbf{H} = 0$",
             "correct": "false", "vec": "none",
             "feedback": "True away from the plane " + f"${u} = 0$" + ", but $\\mathbf{H}$ jumps across that plane, and a jump in tangential $\\mathbf{H}$ is a surface current."})
    random.shuffle(dir_choices)

    # ------------------------------------------------------------------ display strings
    def pt_tex(P):
        return "(" + ",\\ ".join(num(P[x]) for x in AXES) + ")"

    A_tex = f"{num(k)}\\,{f_tex}\\,{ah}"
    Bc_tex = f"B_{cax}"
    i, j = cyc(cax)
    s_curl = 1 if (i, j) == (u, a) else -1             # (curl A)_c = s_curl * dA_a/du  (= s_c)
    assert s_curl == s_c
    curlB_line = (f"{curl_tex('A', cax)} = {'' if s_c > 0 else '-'}dA_{a}/d{u}")
    i2, j2 = cyc(a)
    s_curlH = 1 if (i2, j2) == (u, cax) else -1         # (curl H)_a = s_curlH * dH_c/du  (= -s_c)
    assert s_curlH == -s_c
    curlH_line = (f"{curl_tex('H', a)} = {'' if s_curlH > 0 else '-'}dH_{cax}/d{u}")
    if mode == "abs":
        B_formula = f"{Bc_tex} = {'' if s_c > 0 else '-'}\\dfrac{{d}}{{d{u}}}\\big({num(k)}\\lvert {u}\\rvert\\big) = {num(s_c * k)}\\,\\mathrm{{sgn}}({u})\\ \\mathrm{{mT}}"
    else:
        B_formula = f"{Bc_tex} = {'' if s_c > 0 else '-'}\\dfrac{{d}}{{d{u}}}\\big({num(k)}\\,{u}^2\\big) = {num(2 * s_c * k)}\\,{u}\\ \\mathrm{{mT}}"
    H1c = B1[cax] * 1e-6 / MU0                            # kA/m
    H2c = B2[cax] * 1e-6 / MU0
    if mode == "abs":
        c_words = (f"Just above the plane, $H_{cax} = {Bc_tex}/\\mu_0 = {num(s_c * k)}\\times10^{{-3}}/(4\\pi\\times10^{{-7}}) = {num(s_c * k * 1e-6 / MU0)}$ kA/m; "
                   f"just below, ${num(-s_c * k * 1e-6 / MU0)}$ kA/m. With $\\hat{{n}} = {uh}$ (from the side ${u} \\lt 0$ into the side ${u} \\gt 0$), "
                   f"$\\mathbf{{J}}_s = \\hat{{n}}\\times(\\mathbf{{H}}_1 - \\mathbf{{H}}_2) = ({num(2 * s_c * k * 1e-6 / MU0)})\\,({uh}\\times{ch}) "
                   f"= ({num(2 * s_c * k * 1e-6 / MU0)})({dir_tex(-s_c, a)}) = {num(sJ * Jmag)}\\,{ah}$ kA/m, "
                   f"so $\\lvert\\mathbf{{J}}_s\\rvert = 2\\lvert k\\rvert/\\mu_0 = {num(Jmag)}$ kA/m. "
                   f"Away from the plane $\\mathbf{{H}}$ is uniform and $\\nabla\\times\\mathbf{{H}} = 0$: all the current is in the sheet. "
                   f"Equivalently, $d^2A_{a}/d{u}^2 = 2({num(k)})\\,\\delta({u})$ mT/m and $\\nabla^2\\mathbf{{A}} = -\\mu_0\\mathbf{{J}}$ give the same sheet.")
    else:
        c_words = (f"$H_{cax} = {Bc_tex}/\\mu_0 = \\dfrac{{{num(2 * s_c * k)}\\times10^{{-3}}}}{{4\\pi\\times10^{{-7}}}}\\,{u}$ A/m, so "
                   f"$dH_{cax}/d{u} = {num(2 * s_c * k * 1e-6 / MU0)}\\ \\mathrm{{kA/m^2}}$ everywhere. "
                   f"Then ${curlH_line}$ gives $J_{a} = {num(sJ * Jmag)}\\ \\mathrm{{kA/m^2}}$ at every point (the other components vanish), "
                   f"so $\\lvert\\mathbf{{J}}\\rvert = 2\\lvert k\\rvert/\\mu_0 = {num(Jmag)}$ kA/m². "
                   f"Equivalently, $\\nabla^2 A_{a} = 2({num(k)})$ mT/m and $\\nabla^2\\mathbf{{A}} = -\\mu_0\\mathbf{{J}}$.")
    e_words = (f"$\\mathbf{{E}} = -\\nabla\\Phi - \\partial\\mathbf{{A}}/\\partial t = -({num(beta)})\\,{f_tex}\\,{ah}$ mV/m; at $P_1$ "
               f"(${u} = {num(u1)}$ m), $E_{a} = -({num(beta)})({num(f(u1))}) = {num(E1)}$ mV/m, constant in time. "
               f"(The changing $\\mathbf{{B}}$ and this $\\mathbf{{E}}$ satisfy Faraday's law automatically: $\\nabla\\times\\mathbf{{E}} = -\\partial(\\nabla\\times\\mathbf{{A}})/\\partial t = -\\partial\\mathbf{{B}}/\\partial t$.)")

    data["params"].update({
        "a": a, "u": u, "cax": cax, "s_c": s_c, "mode": mode, "k": k, "beta": beta,
        "ahat": ah, "uhat": uh, "chat": ch, "A_tex": A_tex, "f_tex": f_tex, "k_unit": k_unit, "beta_unit": beta_unit,
        "k_str": num(k), "beta_str": num(beta),
        "P1": P1, "P2": P2, "P1_tex": pt_tex(P1), "P2_tex": pt_tex(P2), "u1": u1, "u2": u2,
        "div_choices": div_choices, "dir_choices": dir_choices,
        "is_abs": mode == "abs", "is_sq": mode == "sq",
        "J_label": r"$\lvert\mathbf{J}_s\rvert =$" if mode == "abs" else r"$\lvert\mathbf{J}\rvert =$",
        "J_suffix": r"$\mathrm{kA/m}$" if mode == "abs" else r"$\mathrm{kA/m^2}$",
        "curlB_line": curlB_line, "B_formula": B_formula,
        "B1c": num(B1[cax]), "B2c": num(B2[cax]), "Bc_tex": Bc_tex,
        "zero_comps": ", ".join(f"$B_{x}$" for x in AXES if x != cax),
        "c_words": c_words, "e_words": e_words, "Jmag_str": num(Jmag), "E1_str": num(E1),
        "H1c": num(H1c), "H2c": num(H2c),
        "A_t_tex": f"({num(k)} {'+' if beta > 0 else '-'} {num(abs(beta))}\\,t)\\,{f_tex}\\,{ah}",
        "dir_correct_text": f"${dir_tex(sJ, a)}$, antiparallel to $\\mathbf{{A}}$ (which points along ${dir_tex(-sJ, a)}$ wherever it is nonzero)",
        "curv_words": ("curves upward (a V-shaped valley at the plane)" if (mode == "abs" and k > 0) else
                       "curves downward (an inverted V at the plane)" if mode == "abs" else
                       "curves upward (a parabola opening upward)" if k > 0 else "curves downward (a parabola opening downward)"),
    })
    for P, B in (("1", B1), ("2", B2)):
        for x in AXES:
            data["correct_answers"][f"B{P}{x}"] = float(B[x])
    data["correct_answers"]["Jmag"] = float(Jmag)
    data["correct_answers"]["E1"] = float(E1)
