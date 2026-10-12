"""A square loop moving at constant velocity through a static field with a linear gradient (Lecture 14).
Family: Summer 2018 HE2 #2 and Summer 2020 HE2 #3 (1 m loop, v = 4 x_hat m/s, B = y_hat or z_hat 25e-6 (1 + x/L) T:
orientations for zero / maximum current, Psi, emf and its time dependence, the current).
Changed: other axes (the two exam orientations are excluded) and signs, the curl-free completion of the field
(so it can exist in empty space), a start position off the origin, a select-all part on orientations and motions
(including tilted loops), and a stated circulation that may be clockwise.

Field (b = main field axis, g = gradient and motion axis, t = third axis, sig = +-1):
    B = sig B0 [ (1 + g/L) b_hat + (b/L) g_hat ]      div B = 0, curl B = 0;  on the plane b = 0: sig B0 (1 + g/L) b_hat
For ANY flat loop, because B is linear in position:  Psi = A n_hat . B(centroid).  The centroid starts on the g axis,
where B = sig B0 (1 + g/L) b_hat, so a current flows only if n_hat . b_hat != 0 AND the centroid moves along g.
Loop in the plane b = 0, centre g_c(t) = g0 + v t, circulation C ccw (s_C = +1) or cw (s_C = -1) seen from +b:
    Psi(t) = s_C sig B0 (1 + g_c/L) s^2,   emf = -dPsi/dt = -s_C sig B0 v s^2 / L  (constant),   I = emf / R
Units: B0 in uT, s in m -> Psi in uWb; emf in uV; I = emf/R in uA.
"""
import math
import random

AXES = "xyz"
COMBOS = [("x", "y"), ("x", "z"), ("y", "z"), ("z", "y")]      # (b, g); the exams' (y, x) and (z, x) are excluded


def hat(a):
    return r"\hat{" + a + "}"


def num(x, p=4):
    x = float(f"{float(x):.{p}g}") + 0.0
    return f"{x:g}"


def unit(a):
    return [1.0 if x == a else 0.0 for x in AXES]


def comb(*terms):
    """sum of (coef, axis) unit vectors, normalized."""
    v = [0.0, 0.0, 0.0]
    for cf, a in terms:
        v[AXES.index(a)] += cf
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def generate(data):
    b, g = random.choice(COMBOS)
    t_ax = next(x for x in AXES if x not in (b, g))
    sig = random.choice([1, -1])
    B0 = random.choice([20, 25, 30, 40, 50, 60, 80])          # uT
    L = random.choice([2, 2.5, 4, 5, 8])                       # m
    s = random.choice([0.5, 0.6, 0.8, 1.0, 1.2])               # m
    R = random.choice([0.1, 0.2, 0.25, 0.4, 0.5])              # ohm
    while True:
        v = random.choice([2, 3, 4, 5, 6]) * random.choice([1, -1])     # m/s along +g
        g0 = random.choice([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])            # m
        t1 = random.choice([0.2, 0.25, 0.5, 0.75, 1.0])                 # s
        gc = g0 + v * t1
        if abs(g0) + s / 2 <= 0.6 * L and abs(gc) + s / 2 <= 0.6 * L:
            break
    s_C = random.choice([1, -1])                               # C ccw (+1) / cw (-1) seen from +b
    Ccw = "counterclockwise" if s_C > 0 else "clockwise"

    A = s * s
    Psi = s_C * sig * B0 * (1 + gc / L) * A                    # uWb
    emf = -s_C * sig * B0 * v * A / L                          # uV
    I = emf / R                                                # uA
    emf_ccw = -sig * B0 * v * A / L                            # uV, ccw seen from +b
    ind_ccw = emf_ccw > 0

    bh, gh, th = hat(b), hat(g), hat(t_ax)
    vtex = f"{num(v)}\\,{gh}"

    # ------------------------------------------------------------------ (a) pool: (text, normal, velocity dir, no-current?)
    sq2 = "/\\sqrt{2}"
    pool = [
        (f"Normal along ${gh}$ (the loop's plane is perpendicular to $\\mathbf{{v}}$), moving with the given $\\mathbf{{v}}$.",
         unit(g), unit(g), True,
         f"Only $B_{g}$ threads this loop, and $B_{g} \\propto {b}$ is positive on one half of the loop and negative on the other: $\\Psi = 0$ at all times."),
        (f"Normal along ${th}$ (both $\\mathbf{{B}}$ and $\\mathbf{{v}}$ lie in the loop's plane), moving with the given $\\mathbf{{v}}$.",
         unit(t_ax), unit(g), True,
         f"$\\mathbf{{B}}$ has no ${t_ax}$-component, so no flux threads this loop at any time."),
        (f"Normal along $({gh}+{th}){sq2}$, moving with the given $\\mathbf{{v}}$.",
         comb((1, g), (1, t_ax)), unit(g), True,
         f"The normal has no ${b}$-component, and $\\hat{{n}}\\cdot\\mathbf{{B}}$ at the centre is $B_{g}/\\sqrt2 = 0$ on the ${g}$ axis: $\\Psi = 0$ throughout."),
        (f"Normal along ${bh}$, but moving along ${th}$ instead (same speed).",
         unit(b), unit(t_ax), True,
         f"$B_{b}$ does not change along ${t_ax}$, so the flux stays constant."),
        (f"Normal along ${bh}$, but moving along ${bh}$ instead (same speed), so the loop rises off the plane ${b} = 0$.",
         unit(b), unit(b), True,
         f"$B_{b}$ does not change along ${b}$, and $B_{g}$ lies in the loop's plane: the flux stays constant."),
        (f"Normal along ${bh}$, moving with the given $\\mathbf{{v}}$.",
         unit(b), unit(g), False,
         f"This is the orientation of parts (b) to (e): $\\Psi = A\\,B_{b}(\\text{{centre}})$ changes at the rate $\\pm B_0 v A/L$ (the largest possible)."),
        (f"Normal along $({bh}+{gh}){sq2}$ (tilted 45° from ${bh}$ toward ${gh}$), moving with the given $\\mathbf{{v}}$.",
         comb((1, b), (1, g)), unit(g), False,
         f"$\\Psi = A\\,\\hat{{n}}\\cdot\\mathbf{{B}}(\\text{{centre}}) = A\\,B_{b}(\\text{{centre}})/\\sqrt2$ still changes as the centre moves along ${g}$: a smaller, nonzero emf."),
        (f"Normal along $({bh}+{th}){sq2}$, moving with the given $\\mathbf{{v}}$.",
         comb((1, b), (1, t_ax)), unit(g), False,
         f"$\\Psi = A\\,B_{b}(\\text{{centre}})/\\sqrt2$ changes as the centre moves along ${g}$: a smaller, nonzero emf."),
        (f"Normal along ${bh}$, moving along $({gh}+{th}){sq2}$ (same speed).",
         unit(b), comb((1, g), (1, t_ax)), False,
         f"The centre still moves along ${g}$ (at $\\lvert v\\rvert/\\sqrt2$), so $B_{b}$ at the centre, and the flux, change."),
    ]
    trues = [x for x in pool if x[3]]
    falses = [x for x in pool if not x[3]]
    k = random.choice([2, 3])
    picked = random.sample(trues, k) + random.sample(falses, 5 - k)
    random.shuffle(picked)
    def vel(vd, tx):          # the actual velocity vector of the statement (m/s)
        return [v * x for x in vd] if "given" in tx else [abs(v) * x for x in vd]
    orient = [{"text": tx, "correct": "true" if z else "false", "n": n, "vel": vel(vd, tx), "feedback": fb,
               "verdict": ("no current" if z else "current flows")} for (tx, n, vd, z, fb) in picked]

    # ------------------------------------------------------------------ (c) sense MC, (d) time MC
    sense_choices = [
        {"text": f"counterclockwise as seen from $+{b}$", "correct": "true" if ind_ccw else "false", "claim": "ccw",
         "feedback": "Right." if ind_ccw else "Check the sign of the emf against the stated circulation, or use Lenz: which way must the induced field point?"},
        {"text": f"clockwise as seen from $+{b}$", "correct": "false" if ind_ccw else "true", "claim": "cw",
         "feedback": "Right." if not ind_ccw else "Check the sign of the emf against the stated circulation, or use Lenz: which way must the induced field point?"},
        {"text": "it does not circulate: a static magnetic field cannot induce a current", "correct": "false", "claim": "none",
         "feedback": "A static but nonuniform field still gives a changing flux through a moving loop (motional emf)."},
        {"text": f"it reverses direction when the loop's centre passes ${g} = 0$", "correct": "false", "claim": "reverses",
         "feedback": "Nothing special happens at " + f"${g} = 0$" + ": the flux keeps changing at the same rate, and the field does not change sign there."},
    ]
    random.shuffle(sense_choices)
    time_choices = [
        {"text": f"It is constant: $\\Psi$ changes linearly in time, because $\\mathbf{{B}}$ is linear in ${g}$ and $\\mathbf{{v}}$ is constant.",
         "correct": "true", "claim": "constant", "feedback": "Right: the emf depends on the gradient B0/L, which is the same everywhere."},
        {"text": f"It changes in proportion to the field at the loop's centre, $B_0(1 + {g}_c/L)$, as the loop moves.",
         "correct": "false", "claim": "follows_B",
         "feedback": "The flux follows the field at the centre, but the emf is its rate of change, which depends only on the gradient."},
        {"text": "It is zero at all times, because $\\mathbf{B}$ does not change in time.",
         "correct": "false", "claim": "zero",
         "feedback": "The loop moves through a nonuniform field, so the flux it links changes even though B at each fixed point does not."},
        {"text": f"It reverses sign when the loop's centre passes ${g} = 0$.",
         "correct": "false", "claim": "reverses",
         "feedback": "The field does not change sign at " + f"${g} = 0$" + "; only its size changes, at a steady rate."},
    ]
    random.shuffle(time_choices)

    # ------------------------------------------------------------------ figure: view from +b, right = +g, up = b x g
    ib, ig = AXES.index(b), AXES.index(g)
    up_is_plus_t = (ig - ib) % 3 == 1            # b x g = +t iff (b, g, t) cyclic
    marks = []
    for col, x in enumerate((40, 95, 150, 205, 260)):
        for y in (28, 172):
            r = 3.0 + 1.1 * col
            mk = {"x": x, "y": y, "r": f"{r:.1f}", "dot": sig > 0, "cross": ""}
            if sig < 0:
                q = r * 0.7071
                mk["cross"] = f"M{x - q:.1f},{y - q:.1f} L{x + q:.1f},{y + q:.1f} M{x - q:.1f},{y + q:.1f} L{x + q:.1f},{y - q:.1f}"
            marks.append(mk)
    c_arrow = "144,60 156,55 156,65" if s_C > 0 else "156,60 144,55 144,65"      # top edge: ccw moves left
    if v > 0:
        v_line, v_head, v_lx = ("200", "232"), "242,80 232,75 232,85", "220"
    else:
        v_line, v_head, v_lx = ("100", "68"), "58,80 68,75 68,85", "74"
    t_y1, t_y2, t_head, t_ly = (("150", "122", "20,112 16,122 24,122", "117") if up_is_plus_t else
                                ("112", "140", "20,150 16,140 24,140", "152"))
    view_caption = (f"view from +{b}: B points {'out of' if sig > 0 else 'into'} the page, stronger toward +{g}")

    # ------------------------------------------------------------------ display strings
    sig_tex = "" if sig > 0 else "-"
    B_tex = f"{sig_tex}B_0\\left[\\left(1 + \\frac{{{g}}}{{L}}\\right){bh} + \\frac{{{b}}}{{L}}\\,{gh}\\right]"
    Bplane_tex = f"{sig_tex}B_0\\left(1 + \\frac{{{g}}}{{L}}\\right){bh}"
    dS_tex = ("+" if s_C > 0 else "-") + bh
    pm = "" if sig * s_C > 0 else "-"                       # sign of B . dS on the loop
    gc_frac = (f"1 + \\frac{{{num(gc)}}}{{{num(L)}}}" if gc >= 0 else f"1 - \\frac{{{num(-gc)}}}{{{num(L)}}}")
    flux_sign_words = (f"On the loop's plane $\\mathbf{{B}}$ points along ${'+' if sig > 0 else '-'}{bh}$ and $d\\mathbf{{S}} = {dS_tex}\\,dS$, "
                       f"so $\\mathbf{{B}}\\cdot d\\mathbf{{S}} = {pm}B_0(1 + {g}/L)\\,dS$. Because this is linear in ${g}$, its integral over the "
                       f"square is its value at the centre times the area:")
    flux_line = (f"\\Psi(t_1) = {pm}B_0\\Big(1 + \\frac{{{g}_c(t_1)}}{{L}}\\Big)s^2 = "
                 f"{pm}({B0})\\Big({gc_frac}\\Big)({num(s)})^2 = {num(Psi)}\\ \\mu\\mathrm{{Wb}}")
    emf_line = (f"\\mathcal{{E}} = -\\frac{{d\\Psi}}{{dt}} = {'-' if sig * s_C > 0 else ''}\\frac{{B_0\\,v_{g}\\,s^2}}{{L}} = "
                f"{'-' if sig * s_C > 0 else ''}\\frac{{({B0})({num(v)})({num(s)})^2}}{{{num(L)}}} = {num(emf)}\\ \\mu\\mathrm{{V}}")
    lenz_words = (f"Lenz check: the field points along ${'+' if sig > 0 else '-'}{bh}$ and the loop moves toward "
                  f"{'larger' if v > 0 else 'smaller'} ${g}$, into {'stronger' if v > 0 else 'weaker'} field, so the flux along "
                  f"${'+' if sig > 0 else '-'}{bh}$ {'grows' if v > 0 else 'shrinks'}. The induced current makes its own field along "
                  f"${'-' if sig * v > 0 else '+'}{bh}$ inside the loop to oppose that change: {'counterclockwise' if ind_ccw else 'clockwise'} as seen from $+{b}$.")
    edge_words = (f"Motional check: on the plane ${b} = 0$, $\\mathbf{{v}}\\times\\mathbf{{B}} = ({num(v)}\\,{gh})\\times({sig_tex}B_0(1+{g}/L)\\,{bh})$ "
                  f"is parallel to the ${t_ax}$ axis, so only the two edges parallel to ${th}$ contribute; they sit ${num(s)}$ m apart in ${g}$, where $B_{b}$ differs by "
                  f"$B_0 s/L = {num(B0 * s / L)}\\ \\mu\\mathrm{{T}}$, and the difference of their two 'batteries' is $\\lvert v\\rvert\\,s\\,(B_0s/L) = {num(abs(emf))}\\ \\mu\\mathrm{{V}}$, the size of the emf found above.")

    data["params"].update({
        "b": b, "g": g, "t": t_ax, "sig": sig, "B0": B0, "L": L, "s": s, "R": R, "v": v, "g0": g0, "t1": t1,
        "gc": gc, "s_C": s_C, "Ccw": Ccw, "bhat": bh, "ghat": gh, "that": th, "vtex": vtex,
        "B_tex": B_tex, "Bplane_tex": Bplane_tex, "dS_tex": dS_tex, "flux_line": flux_line, "emf_line": emf_line,
        "flux_sign_words": flux_sign_words, "k_true": k,
        "orient": orient, "sense_choices": sense_choices, "time_choices": time_choices,
        "I_str": num(I), "emf_str": num(emf), "Psi_str": num(Psi), "gc_str": num(gc), "A_str": num(A),
        "lenz_words": lenz_words, "edge_words": edge_words,
        "marks": marks, "c_arrow": c_arrow, "v_x1": v_line[0], "v_x2": v_line[1], "v_head": v_head, "v_lx": v_lx,
        "t_y1": t_y1, "t_y2": t_y2, "t_head": t_head, "t_ly": t_ly, "view_caption": view_caption,
        "fig_up_is_plus_t": up_is_plus_t, "fig_ccw_left": True,
    })
    data["correct_answers"]["Psi"] = float(Psi)       # uWb
    data["correct_answers"]["emf"] = float(emf)       # uV
    data["correct_answers"]["I"] = float(I)           # uA
