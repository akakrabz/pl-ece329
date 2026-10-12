"""Lecture 6: magnetic boundary conditions at current sheets, free space (B = mu0 H) on both sides.

n points from medium 2 into medium 1.
    n x (H1 - H2) = Js   ->   H1t - H2t = Js x n   (cross with n; n x (n x v) = -v for tangential v)
    n . (B1 - B2) = 0    ->   H1n = H2n            (mu0 on both sides)
The tangential component of H along Js does not jump; the one perpendicular to Js jumps by |Js|.
(a) plane z = 0 with known Js, n = +z (medium 1 above); H given on one side (random), the other side asked.
(b) plane x = 0 or y = 0, H given on both sides, Js asked: Js = n x (H1 - H2) with n pointing into the side
    called 1 (the answer does not depend on which side that is).
(c) checkbox pool on the four conditions; (d) matching: integral law + shape -> condition.
"""
import random

AX = "xyz"
UNIT = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def add(a, b, s=1):
    return tuple(x + s * y for x, y in zip(a, b))


def vec_tex(v):
    """3\\,\\hat{x} - \\hat{y} + 5\\,\\hat{z}; zero components dropped."""
    out = ""
    for comp, ax in zip(v, AX):
        if comp == 0:
            continue
        mag = "" if abs(comp) == 1 else f"{abs(comp)}\\,"
        if not out:
            out = ("-" if comp < 0 else "") + mag + f"\\hat{{{ax}}}"
        else:
            out += (" - " if comp < 0 else " + ") + mag + f"\\hat{{{ax}}}"
    return out or "0"


def nz(lo=-6, hi=6):
    return random.choice([k for k in range(lo, hi + 1) if k != 0])


def jump_pair():
    """Two tangential surface-current components, at least one nonzero (both nonzero 70 % of the time)."""
    if random.random() < 0.7:
        return nz(), nz()
    j = [nz(), 0]
    random.shuffle(j)
    return tuple(j)


def generate(data):
    # ------------------------------------------------------------------ (a) z = 0, known Js
    Jx, Jy = jump_pair()
    Js_a = (Jx, Jy, 0)
    jump_a = cross(Js_a, UNIT["z"])                  # H1t - H2t = Js x z = (Jy, -Jx, 0)
    while True:
        Hg = (random.randint(-8, 8), random.randint(-8, 8), nz(-8, 8))
        given_above = random.random() < 0.5
        Hask = add(Hg, jump_a, -1) if given_above else add(Hg, jump_a, +1)
        if max(abs(h) for h in Hask) <= 12 and Hask[:2] != Hg[:2]:
            break
    H1 = Hg if given_above else Hask
    H2 = Hask if given_above else Hg
    assert cross(UNIT["z"], add(H1, H2, -1)) == Js_a and H1[2] == H2[2]

    # ------------------------------------------------------------------ (b) x = 0 or y = 0, Js asked
    u = random.choice(["x", "y"])
    t1, t2 = [a for a in AX if a != u]                # tangential axes, in x, y, z order
    j1, j2 = jump_pair()
    Js_b = [0, 0, 0]
    Js_b[AX.index(t1)], Js_b[AX.index(t2)] = j1, j2
    Js_b = tuple(Js_b)
    Hneg = [random.randint(-8, 8) for _ in range(3)]
    Hneg[AX.index(u)] = nz(-8, 8)
    Hneg = tuple(Hneg)
    Hpos = add(Hneg, cross(Js_b, UNIT[u]))           # medium 1 = positive side, n = +u
    dH = add(Hpos, Hneg, -1)
    assert cross(UNIT[u], dH) == Js_b and dH[AX.index(u)] == 0
    pos_first = random.random() < 0.5

    # ------------------------------------------------------------------ (c) checkbox pool
    pool = [
        ("Et", True, r"Tangential $\mathbf{E}$ is continuous across every interface, charged or not.",
         r"A thin loop straddling the interface has zero circulation, and only its two long edges contribute."),
        ("Dn", True, r"Normal $\mathbf{D}$ jumps by the free surface charge density: $\hat n\cdot(\mathbf{D}_1-\mathbf{D}_2) = \rho_s$.",
         r"Gauss's law on a flat pillbox: only the two caps carry flux, and the enclosed charge is $\rho_s$ times the cap area."),
        ("Bn", True, r"Normal $\mathbf{B}$ is continuous across every interface, including one that carries a surface current.",
         r"$\oint\mathbf{B}\cdot d\mathbf{S} = 0$ on a pillbox; a surface current changes nothing here."),
        ("Ht", True, r"Tangential $\mathbf{H}$ jumps by the surface current: $\hat n\times(\mathbf{H}_1-\mathbf{H}_2) = \mathbf{J}_s$.",
         r"Ampère's law on a thin loop: the long edges give the jump in $H_t$, and the current threading the loop is $J_s$ times its length."),
        ("par", True, r"At a current sheet, the tangential component of $\mathbf{H}$ parallel to $\mathbf{J}_s$ has the same value on both sides.",
         r"$\mathbf{H}_{1t}-\mathbf{H}_{2t} = \mathbf{J}_s\times\hat n$ is perpendicular to $\mathbf{J}_s$, so the component along $\mathbf{J}_s$ does not jump."),
        ("swap", True, r"If you call the other side medium 1 but keep the same $\hat n$, the expression $\hat n\cdot(\mathbf{D}_1-\mathbf{D}_2)$ comes out as $-\rho_s$.",
         r"Swapping the labels reverses $\mathbf{D}_1-\mathbf{D}_2$; only swapping the labels and $\hat n$ together leaves the result unchanged."),
        ("En", False, r"Normal $\mathbf{E}$ is continuous across a sheet of free surface charge in vacuum.",
         r"With $\mathbf{D} = \epsilon_0\mathbf{E}$ on both sides the pillbox gives a jump of $\rho_s/\epsilon_0$: a lone charged sheet's field points away from it on both sides."),
        ("Dt", False, r"Tangential $\mathbf{D}$ is continuous across every interface.",
         r"Tangential $\mathbf{E}$ is; $\mathbf{D}_t = \epsilon\mathbf{E}_t$ jumps wherever $\epsilon$ does."),
        ("Bt", False, r"Tangential $\mathbf{B}$ is continuous across a current sheet in free space.",
         r"In free space $\mathbf{B}_t = \mu_0\mathbf{H}_t$, and $\mathbf{H}_t$ jumps by the surface current."),
        ("Hn", False, r"Normal $\mathbf{H}$ jumps by $|\mathbf{J}_s|$ at a current sheet in free space.",
         r"Normal $\mathbf{B}$ is continuous and $\mathbf{B} = \mu_0\mathbf{H}$ on both sides, so normal $\mathbf{H}$ is continuous."),
        ("order", False, r"The jump in tangential $\mathbf{H}$ is $\mathbf{H}_{1t}-\mathbf{H}_{2t} = \hat n\times\mathbf{J}_s$.",
         r"Crossing $\hat n\times(\mathbf{H}_1-\mathbf{H}_2) = \mathbf{J}_s$ with $\hat n$ gives $\mathbf{J}_s\times\hat n$; the reversed order flips the jump."),
        ("units", False, r"A surface current density $\mathbf{J}_s$ is measured in $\mathrm{A/m^2}$, like a volume current density.",
         r"$\mathbf{J}_s$ is current per unit width, in $\mathrm{A/m}$, the same units as $\mathbf{H}$."),
    ]
    cb = [{"key": k, "correct": "true" if t else "false", "text": txt, "tf": "True" if t else "False",
           "why": why, "feedback": ("True. " if t else "False. ") + why} for k, t, txt, why in pool]
    random.shuffle(cb)

    # ------------------------------------------------------------------ (d) matching: law + shape -> condition
    statements = [
        {"key": "gaussD", "match": "Dn",
         "text": r"Gauss's law $\oint\mathbf{D}\cdot d\mathbf{S} = Q_{\text{enc}}$, applied to a flat pillbox straddling the interface"},
        {"key": "gaussB", "match": "Bn",
         "text": r"$\oint\mathbf{B}\cdot d\mathbf{S} = 0$, applied to a flat pillbox straddling the interface"},
        {"key": "circE", "match": "Et",
         "text": r"$\oint\mathbf{E}\cdot d\mathbf{l} = 0$, applied to a thin rectangular loop straddling the interface"},
        {"key": "ampere", "match": "Ht",
         "text": r"Ampère's law $\oint\mathbf{H}\cdot d\mathbf{l} = I_{\text{enc}}$, applied to a thin rectangular loop straddling the interface"},
    ]
    options = [
        {"name": "Dn", "text": r"$\hat n\cdot(\mathbf{D}_1-\mathbf{D}_2) = \rho_s$ (normal $\mathbf{D}$ jumps by the free surface charge)"},
        {"name": "Bn", "text": r"$\hat n\cdot(\mathbf{B}_1-\mathbf{B}_2) = 0$ (normal $\mathbf{B}$ is continuous)"},
        {"name": "Et", "text": r"$\hat n\times(\mathbf{E}_1-\mathbf{E}_2) = 0$ (tangential $\mathbf{E}$ is continuous)"},
        {"name": "Ht", "text": r"$\hat n\times(\mathbf{H}_1-\mathbf{H}_2) = \mathbf{J}_s$ (tangential $\mathbf{H}$ jumps by the surface current)"},
        {"name": "En", "text": r"$\hat n\cdot(\mathbf{E}_1-\mathbf{E}_2) = 0$ (normal $\mathbf{E}$ is continuous)"},
        {"name": "Dt", "text": r"$\hat n\times(\mathbf{D}_1-\mathbf{D}_2) = 0$ (tangential $\mathbf{D}$ is continuous)"},
        {"name": "Hn", "text": r"$\hat n\cdot(\mathbf{H}_1-\mathbf{H}_2) = |\mathbf{J}_s|$ (normal $\mathbf{H}$ jumps by the surface current)"},
    ]
    random.shuffle(statements)
    random.shuffle(options)

    # ------------------------------------------------------------------ strings
    asked = "2" if given_above else "1"
    if given_above:
        a_setup = (r"Just above the plane ($z = 0^+$, medium 1) the field is $\mathbf{H}_1 = " + vec_tex(H1)
                   + r"\ \mathrm{A/m}$. Find $\mathbf{H}_2$ just below the plane ($z = 0^-$).")
        a_solve = (r"$\mathbf{H}_{2t} = \mathbf{H}_{1t} - (\mathbf{J}_s\times\hat z) = (" + vec_tex((H1[0], H1[1], 0)) + r") - ("
                   + vec_tex(jump_a) + r") = " + vec_tex((H2[0], H2[1], 0)) + r"$ A/m")
    else:
        a_setup = (r"Just below the plane ($z = 0^-$, medium 2) the field is $\mathbf{H}_2 = " + vec_tex(H2)
                   + r"\ \mathrm{A/m}$. Find $\mathbf{H}_1$ just above the plane ($z = 0^+$).")
        a_solve = (r"$\mathbf{H}_{1t} = \mathbf{H}_{2t} + (\mathbf{J}_s\times\hat z) = (" + vec_tex((H2[0], H2[1], 0)) + r") + ("
                   + vec_tex(jump_a) + r") = " + vec_tex((H1[0], H1[1], 0)) + r"$ A/m")
    Hpos_tex, Hneg_tex = vec_tex(Hpos), vec_tex(Hneg)
    side_pos = f"\\mathbf{{H}}({u} = 0^+) = {Hpos_tex}"
    side_neg = f"\\mathbf{{H}}({u} = 0^-) = {Hneg_tex}"
    b_fields = (side_pos + r",\qquad " + side_neg) if pos_first else (side_neg + r",\qquad " + side_pos)
    jx_tex = vec_tex(Js_a)
    data["params"].update({
        "Jx": Jx, "Jy": Jy, "Js_a_tex": jx_tex, "given_above": given_above, "asked": asked,
        "H1": list(H1), "H2": list(H2), "H1_tex": vec_tex(H1), "H2_tex": vec_tex(H2),
        "jump_a_tex": vec_tex(jump_a), "a_setup": a_setup, "a_solve": a_solve,
        "Hz_common": H1[2],
        "u": u, "t1": t1, "t2": t2, "Hpos": list(Hpos), "Hneg": list(Hneg), "b_fields": b_fields,
        "dH_tex": vec_tex(dH), "Js_b_tex": vec_tex(Js_b), "un": u, "Hn_b": Hneg[AX.index(u)],
        "cb": cb, "statements": statements, "options": options,
    })
    Hask_v = H2 if given_above else H1
    data["correct_answers"]["Hax"] = float(Hask_v[0])
    data["correct_answers"]["Hay"] = float(Hask_v[1])
    data["correct_answers"]["Haz"] = float(Hask_v[2])
    data["correct_answers"]["Jb1"] = float(j1)
    data["correct_answers"]["Jb2"] = float(j2)
