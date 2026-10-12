"""Two coaxial ideal solenoids: superposed H, the cancelling inner current, Lenz, and the emf of a coaxial loop.
Lectures 13-14. Family: Summer 2017 HE2 #1b and Summer 2020 HE2 #2b (outer coil n1, a1, I1; inner coil n2, a2;
find H0, the I2 that cancels H inside the inner coil, and how I2 responds when I1 changes).
Changed: a random axis and sense, unequal turn densities, part (a) with a given I2, part (c) in the annulus,
Lenz with I1 increasing or decreasing, and a new part (e): the emf of a single coaxial loop in a random region
while both currents ramp.

Sign conventions. Currents are counted positive when they circulate in I1's sense; I1 circulates
counterclockwise (s1 = +1) or clockwise (s1 = -1) as seen from the +axis side, i.e. right-handed about s1*axis_hat.
  ideal solenoid: H = n I (right-hand rule) inside, 0 outside
  r < a2:       H_axis = s1 (n1 I1 + n2 I2)
  a2 < r < a1:  H_axis = s1 n1 I1          (the inner coil's field stays inside it)
  cancel:       I2 = -n1 I1 / n2
  loop of radius r0 traversed in I1's sense (dS along s1 axis_hat):
     Psi = mu0 pi [ n1 I1 min(r0, a1)^2 + n2 I2 min(r0, a2)^2 ],  emf = -dPsi/dt
"""
import math
import random

MU0 = 4e-7 * math.pi


def num(x, p=4):
    x = float(f"{float(x):.{p}g}") + 0.0
    return f"{x:g}"


def generate(data):
    axis = random.choice("xyz")
    s1 = random.choice([1, -1])
    sense1 = "counterclockwise" if s1 > 0 else "clockwise"
    n1 = random.choice([1000, 1200, 1500, 2000, 2500, 3000])
    n2 = random.choice([v for v in (400, 500, 600, 800, 1000, 1200, 1500, 2000, 2500) if v != n1])
    a1 = random.choice([3, 4, 5, 6])                                     # cm
    a2 = random.choice([v for v in (1, 1.5, 2, 2.5) if v <= 0.7 * a1])  # cm
    I1 = random.choice([0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8, 1.0, 1.2, 1.5])          # A
    while True:
        I2 = random.choice([0.2, 0.25, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0]) * random.choice([1, -1])
        if abs(n1 * I1 + n2 * I2) >= 100:
            break

    Ha = s1 * (n1 * I1 + n2 * I2)            # A/m along +axis, r < a2
    I2c = -n1 * I1 / n2                      # A, in I1's sense
    Hc = s1 * n1 * I1                        # A/m along +axis, a2 < r < a1

    change = random.choice(["increases", "decreases"])

    region = random.choice(["core", "annulus", "outside"])
    if region == "core":
        r0 = round(a2 * random.choice([0.4, 0.5, 0.6]), 2)
    elif region == "annulus":
        r0 = round(0.5 * (a1 + a2), 2)
    else:
        r0 = a1 * random.choice([1.5, 2, 3])
    while True:
        al1 = random.choice([5, 10, 20, 25]) * random.choice([1, -1])        # A/s
        al2 = random.choice([5, 10, 20, 40]) * random.choice([1, -1])        # A/s
        t1 = MU0 * math.pi * n1 * al1 * (min(r0, a1) * 1e-2) ** 2
        t2 = MU0 * math.pi * n2 * al2 * (min(r0, a2) * 1e-2) ** 2
        if abs(t1 + t2) > 0.2 * max(abs(t1), abs(t2)):
            break
    emf = -(t1 + t2) * 1e6                    # uV, positive along I1's sense

    # ------------------------------------------------------------------ (d) Lenz MC
    inc = change == "increases"
    lenz_choices = [
        {"text": "It circulates opposite to $I_1$, so inside the inner coil its own field opposes the outer coil's field.",
         "correct": "true" if inc else "false", "claim": "opposite",
         "feedback": ("Right: the flux through each inner turn is growing, and the induced current's field pushes back against the growth." if inc else
                      "That would be the response to a growing flux. Here the flux is shrinking, and the induced current tries to keep it up.")},
        {"text": "It circulates in the same sense as $I_1$, so inside the inner coil its own field adds to the outer coil's field.",
         "correct": "false" if inc else "true", "claim": "same",
         "feedback": ("That would prop up a shrinking flux. Here the flux through each inner turn is growing, and Lenz's rule says the induced current fights the growth." if inc else
                      "Right: the flux through each inner turn is shrinking, and the induced current's field tries to replace what is being lost.")},
        {"text": "No current is induced: the outer coil's field is uniform over the inner coil's cross-section, and a uniform field induces nothing.",
         "correct": "false", "claim": "zero_uniform",
         "feedback": "Uniform in space is not constant in time. The flux through every inner turn changes, so there is an emf around each turn."},
        {"text": "No current is induced: the inner coil is not connected to the outer coil, and an ideal solenoid has no field outside its own winding.",
         "correct": "false", "claim": "zero_contact",
         "feedback": "The inner coil sits inside the outer coil, where the outer coil's field is strong; no wire connection is needed for induction."},
    ]
    random.shuffle(lenz_choices)

    # ------------------------------------------------------------------ display
    ax_hat = r"\hat{" + axis + "}"
    s1_tex = ("+" if s1 > 0 else "-") + ax_hat
    reg_words = {"core": f"inside the inner coil ($r_0 \\lt a_2$)",
                 "annulus": f"between the two windings ($a_2 \\lt r_0 \\lt a_1$)",
                 "outside": f"outside both coils ($r_0 \\gt a_1$), where $\\mathbf{{B}} = 0$ on the loop itself"}[region]
    m1 = min(r0, a1)
    m2 = min(r0, a2)
    psi_words = {
        "core": (f"The loop lies inside both windings, so both fields thread its whole area: "
                 f"$\\Psi = \\mu_0\\pi r_0^2\\,(n_1I_1 + n_2I_2)$."),
        "annulus": (f"The loop encloses the outer coil's field only over its own area $\\pi r_0^2$, but it encloses the whole inner coil, "
                    f"so it links all of the inner coil's flux: $\\Psi = \\mu_0\\pi\\,(n_1I_1 r_0^2 + n_2I_2 a_2^2)$."),
        "outside": (f"The loop encloses both coils entirely, so it links all of both fluxes even though $\\mathbf{{B}} = 0$ on the wire: "
                    f"$\\Psi = \\mu_0\\pi\\,(n_1I_1 a_1^2 + n_2I_2 a_2^2)$."),
    }[region]
    emf_line = (f"\\mathcal{{E}} = -\\dfrac{{d\\Psi}}{{dt}} = -\\mu_0\\pi\\Big[({n1})({num(al1)})({num(m1 * 1e-2)})^2 + ({n2})({num(al2)})({num(m2 * 1e-2)})^2\\Big]"
                f" = -\\big[{num(t1 * 1e6)} {'+' if t2 >= 0 else '-'} {num(abs(t2) * 1e6)}\\big]\\ \\mu\\mathrm{{V}} = {num(emf)}\\ \\mu\\mathrm{{V}}")

    # SVG: end view seen from the +axis side (axis toward the viewer); radii to scale
    R = max(a1, r0)
    sc = 88.0 / R
    r_px1, r_px2, r_px0 = a1 * sc, a2 * sc, r0 * sc
    cx = cy = 110.0
    arrow = (f"{cx - 6:.1f},{cy - r_px1:.1f} {cx + 4:.1f},{cy - r_px1 - 5:.1f} {cx + 4:.1f},{cy - r_px1 + 5:.1f}" if s1 > 0 else
             f"{cx + 6:.1f},{cy - r_px1:.1f} {cx - 4:.1f},{cy - r_px1 - 5:.1f} {cx - 4:.1f},{cy - r_px1 + 5:.1f}")

    def lab(rpx, ang_deg, dr):
        t = math.radians(ang_deg)
        return f"{cx + (rpx + dr) * math.cos(t):.1f}", f"{cy - (rpx + dr) * math.sin(t):.1f}"

    la1, la2, lr0 = lab(r_px1, -35, 10), lab(r_px2, -70, 8), lab(r_px0, 215, 10 if region != "core" else 6)

    data["params"].update({
        "axis": axis, "s1": s1, "sense1": sense1, "ax_hat": ax_hat, "s1_tex": s1_tex,
        "n1": n1, "n2": n2, "a1": a1, "a2": a2, "I1": I1, "I2": I2, "I2_str": num(I2),
        "change": change, "region": region, "r0": r0, "al1": al1, "al2": al2,
        "lenz_choices": lenz_choices, "reg_words": reg_words, "psi_words": psi_words, "emf_line": emf_line,
        "Ha_str": num(Ha / 1e3), "Hc_str": num(Hc / 1e3), "I2c_str": num(I2c), "emf_str": num(emf),
        "n1I1": num(n1 * I1), "n2I2": num(n2 * I2), "sum_nI": num(n1 * I1 + n2 * I2),
        "s1_sign": "+" if s1 > 0 else "-",
        "Ha_dir_words": ("along $+" + ax_hat + "$" if Ha > 0 else "along $-" + ax_hat + "$"),
        "I2c_words": ("opposite to $I_1$'s sense" if I2c < 0 else "in $I_1$'s sense"),
        "lenz_words": ("increasing" if inc else "decreasing"),
        "lenz_resp": ("opposite to $I_1$" if inc else "in the same sense as $I_1$"),
        "flux_dir": s1_tex,
        "r_px1": f"{r_px1:.1f}", "r_px2": f"{r_px2:.1f}", "r_px0": f"{r_px0:.1f}", "arrow": arrow,
        "la1x": la1[0], "la1y": la1[1], "la2x": la2[0], "la2y": la2[1], "lr0x": lr0[0], "lr0y": lr0[1],
    })
    data["correct_answers"]["Ha"] = float(Ha / 1e3)       # kA/m
    data["correct_answers"]["I2c"] = float(I2c)           # A
    data["correct_answers"]["Hc"] = float(Hc / 1e3)       # kA/m
    data["correct_answers"]["emf"] = float(emf)           # uV
