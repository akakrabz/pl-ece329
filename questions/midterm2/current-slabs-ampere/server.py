"""Two current slabs plus a current sheet on the mid-plane of the gap: H by Ampere's law (Lectures 12-13).

Family: Summer 2019 HE2 #2a (slabs carrying z-directed current, bounded in x, plus a sheet on x = 0).
Changed: the orientation (current along c, slabs bounded in u, field along the third axis h; the two
exam orientations are excluded), millimetre slabs with kA/m^2 currents, the "shape across a slab" MC with
the slope, and a part where the sheet is reversed or removed.

Sign bookkeeping. s_h = (c_hat x u_hat) . h_hat = +1 if (c, u, h) is a cyclic permutation of (x, y, z).
  one sheet K c_hat on u = 0:   H = (1/2) K c_hat x n_hat,  n_hat = sgn(u) u_hat   ->  H_h = s_h (K/2) sgn(u)
  one slab J c_hat, a < u < b:  stack of sheets dK = J du'  ->  H_h = s_h J (clip(u, a, b) - m),  m = (a + b)/2
                                (inside: ramp of slope s_h J through zero at m; outside: +-s_h J w/2)
  curl check: (curl H)_c = s_h dH_h/du = J_c.
Units: J in kA/m^2 times positions in mm gives A/m directly.
"""
import random

AXES = "xyz"
COMBOS = [("x", "z"), ("y", "x"), ("y", "z"), ("z", "y")]   # (current axis c, thickness axis u); exam's (z,x), (x,y) excluded


def hat(a):
    return r"\hat{" + a + "}"


def num(x):
    x = round(float(x), 9) + 0.0
    return f"{x:g}"


def sgn(x):
    return (x > 0) - (x < 0)


def signed_term(coef, unit=""):
    """'+ 3' / '- 3' style piece for sums written out in the solution."""
    return ("+ " if coef >= 0 else "- ") + num(abs(coef)) + unit


def shifted(var, m):
    """(z - 3.5), (z + 3.5) or z."""
    if m == 0:
        return var
    return f"({var} {'-' if m > 0 else '+'} {num(abs(m))})"


def generate(data):
    c, u = random.choice(COMBOS)
    h = next(a for a in AXES if a not in (c, u))
    s_h = 1 if (AXES.index(u) - AXES.index(c)) % 3 == 1 else -1      # c_hat x u_hat = s_h h_hat

    g = random.choice([1, 2])                                          # gap -g < u < g (mm), sheet on u = 0
    w1, w2 = random.choice([2, 3, 4]), random.choice([2, 3, 4])
    a1, b1, a2, b2 = -g - w1, -g, g, g + w2
    m1, m2 = (a1 + b1) / 2, (a2 + b2) / 2
    J1 = random.choice([-6, -5, -4, -3, -2, 2, 3, 4, 5, 6])          # kA/m^2
    J2 = random.choice([-6, -5, -4, -3, -2, 2, 3, 4, 5, 6])
    K = random.choice([-8, -6, -4, -2, 2, 4, 6, 8])                    # A/m

    pts = [a1 - random.choice([1, 2]),
           a1 + random.choice([0.5 * i for i in range(1, 2 * w1)]),
           random.choice([-1, 1]) * g / 2,
           a2 + random.choice([0.5 * i for i in range(1, 2 * w2)]),
           b2 + random.choice([1, 2])]

    def parts(x, Ksheet):
        p1 = s_h * J1 * (min(max(x, a1), b1) - m1)
        p2 = s_h * J2 * (min(max(x, a2), b2) - m2)
        ps = s_h * 0.5 * Ksheet * sgn(x)
        return p1, p2, ps

    Hs = [sum(parts(x, K)) for x in pts]

    # (d) sheet reversed or removed, H at one of the points
    dmode = random.choice(["reversed", "removed"])
    kd = random.randrange(5)
    Knew = -K if dmode == "reversed" else 0
    Hd = sum(parts(pts[kd], Knew))
    sheet_old = parts(pts[kd], K)[2]
    sheet_new = parts(pts[kd], Knew)[2]

    # (c) shape across one slab
    ks = random.choice([1, 2])
    Jk, wk, ak, bk, mk = (J1, w1, a1, b1, m1) if ks == 1 else (J2, w2, a2, b2, m2)
    slope = s_h * Jk                                                   # A/m per mm = kA/m^2

    Hh, Hc, Hu = f"H_{h}", f"H_{c}", f"H_{u}"
    hh, ch, uh = hat(h), hat(c), hat(u)
    plus_h = ("+" if s_h > 0 else "-") + hh                             # direction of c_hat x u_hat

    # ---------------------------------------------------------------- (b) which components vanish
    comp_choices = [
        {"text": f"${Hc} = {Hu} = 0$ everywhere, so only ${Hh}$ survives: every current filament's field circles that "
                 f"filament (nothing along ${ch}$), and two filaments placed symmetrically in ${h}$ about the field point "
                 f"cancel each other's ${u}$-components.",
         "correct": "true", "zero": [c, u],
         "feedback": "Right: this is the sheet symmetry argument, and it holds for every thin layer of the slabs."},
        {"text": f"${Hc} = 0$, but ${Hu} \\ne 0$: on each side the field points away from the slabs, the way $\\mathbf{{D}}$ "
                 f"points away from a charged slab.",
         "correct": "false", "zero": [c],
         "feedback": "That is the electric picture. A current sheet's field is tangential to it: $\\tfrac12\\mathbf{J}_s\\times\\hat{n}$ is perpendicular to $\\hat{n}$."},
        {"text": f"${Hu} = {Hh} = 0$: inside the slabs the field runs along the current, ${Hc} \\ne 0$.",
         "correct": "false", "zero": [u, h],
         "feedback": "Biot-Savart: $d\\mathbf{B}\\propto I\\,d\\mathbf{l}\\times\\hat{R}$ is perpendicular to every current element, so a current along " + f"${ch}$" + " gives no field along it."},
        {"text": f"Only ${Hu} = 0$; ${Hc}$ and ${Hh}$ are both nonzero inside the slabs.",
         "correct": "false", "zero": [u],
         "feedback": "No current element produces a field component along its own direction, so " + f"${Hc} = 0$" + " everywhere."},
    ]
    random.shuffle(comp_choices)

    # ---------------------------------------------------------------- (c) shape across slab ks
    shape_choices = [
        {"text": f"It varies linearly in ${u}$ with slope $d{Hh}/d{u} = {num(slope)}$ A/m per mm, and it is continuous at both faces.",
         "correct": "true", "kind": "linear", "slope": float(slope),
         "feedback": "Right: the slope is the current density (curl H = J) and nothing jumps, because the faces carry no surface current."},
        {"text": f"It varies linearly in ${u}$ with slope $d{Hh}/d{u} = {num(-slope)}$ A/m per mm, and it is continuous at both faces.",
         "correct": "false", "kind": "linear", "slope": float(-slope),
         "feedback": "Linear and continuous, yes, but the sign of the slope is off: fix it with the cross product (or with the curl component along the current)."},
        {"text": f"It is constant inside the slab and jumps by ${num(abs(Jk) * wk)}$ A/m at each face.",
         "correct": "false", "kind": "const_jump",
         "feedback": "That treats the faces as current sheets. The current is spread through the volume, so the change happens gradually across the slab and H is continuous at the faces."},
        {"text": f"It varies linearly inside the slab and jumps by ${num(abs(Jk) * wk / 2)}$ A/m at each face.",
         "correct": "false", "kind": "linear_jump",
         "feedback": "Tangential H jumps only where there is a surface current; the faces of a slab carry none."},
        {"text": "It falls off as one over the distance from the slab's centre plane.",
         "correct": "false", "kind": "inverse",
         "feedback": "A 1/r fall-off belongs to a line current. A planar current has no length scale to fall off with."},
    ]
    random.shuffle(shape_choices)

    # ---------------------------------------------------------------- display strings
    def contrib_str(x):
        return [num(v) for v in parts(x, K)]

    rows = []
    for i, (x, H) in enumerate(zip(pts, Hs)):
        p1, p2, ps = contrib_str(x)
        rows.append({"P": f"$P_{i + 1}$", "u": num(x), "s1": p1, "s2": p2, "sh": ps, "tot": num(H)})

    def slab_line(idx, J, w, a, b, m):
        inside = f"{num(s_h * J)}\\,{shifted(u, m)}"
        return (f"Slab {idx} (${num(a)} \\lt {u} \\lt {num(b)}$ mm, centre plane ${u} = {num(m)}$ mm, "
                f"$J_{idx}w_{idx} = {num(J * w)}$ A/m): ${Hh} = {inside}$ inside; "
                f"${Hh} = {num(-s_h * J * w / 2)}$ A/m for ${u} \\lt {num(a)}$ and ${num(s_h * J * w / 2)}$ A/m for ${u} \\gt {num(b)}$.")

    cross_tex = f"{ch}\\times{uh} = {plus_h}"
    if s_h > 0:
        curl_tex = f"(\\nabla\\times\\mathbf{{H}})_{c} = \\partial {Hh}/\\partial {u} - \\partial {Hu}/\\partial {h} = d{Hh}/d{u}"
    else:
        curl_tex = f"(\\nabla\\times\\mathbf{{H}})_{c} = \\partial {Hu}/\\partial {h} - \\partial {Hh}/\\partial {u} = -d{Hh}/d{u}"

    px = lambda x: 20 + (x - (pts[0] - 1)) * 320 / ((pts[4] + 1) - (pts[0] - 1))   # SVG x-coordinate
    svg = {"x_a1": f"{px(a1):.1f}", "x_b1": f"{px(b1):.1f}", "x_a2": f"{px(a2):.1f}", "x_b2": f"{px(b2):.1f}",
           "x_0": f"{px(0):.1f}", "w_s1": f"{px(b1) - px(a1):.1f}", "w_s2": f"{px(b2) - px(a2):.1f}",
           "x_m1": f"{px(m1):.1f}", "x_m2": f"{px(m2):.1f}",
           "pts_svg": [{"x": f"{px(x):.1f}", "lab": str(i + 1)} for i, x in enumerate(pts)]}

    point_list = ", ".join(f"$P_{i + 1}$: ${u} = {num(x)}$ mm" for i, x in enumerate(pts))
    d_what = "reversed, $\\mathbf{J}_s \\to " + num(-K) + "\\," + ch + "$ A/m" if dmode == "reversed" else "removed"
    d_words = (f"Only the sheet's term changes. At $P_{kd + 1}$ (${u} = {num(pts[kd])}$ mm, on the "
               f"{'+' if pts[kd] > 0 else '-'}${u}$ side of the sheet) it was ${num(sheet_old)}$ A/m; "
               + (f"reversing the sheet flips it to ${num(sheet_new)}$ A/m" if dmode == "reversed"
                  else "removing the sheet deletes it")
               + f", so ${Hh} = {num(Hs[kd])} {signed_term(sheet_new - sheet_old)} = {num(Hd)}$ A/m. "
               + (f"The change, ${num(sheet_new - sheet_old)}$ A/m, is a full $|J_s| = {num(abs(K))}$ A/m: the sheet's half-field "
                  f"$\\tfrac12|J_s|$ changes sign."
                  if dmode == "reversed" else
                  f"The change, ${num(sheet_new - sheet_old)}$ A/m, is half the sheet current, $\\tfrac12|J_s|$, in size. "
                  f"Everything else in the table is untouched."))

    data["params"].update({
        "c": c, "u": u, "h": h, "s_h": s_h, "chat": ch, "uhat": uh, "hhat": hh, "plus_h": plus_h,
        "Hh_tex": Hh, "Hc_tex": Hc, "Hu_tex": Hu,
        "a1": a1, "b1": b1, "a2": a2, "b2": b2, "m1": m1, "m2": m2, "w1": w1, "w2": w2, "g": g,
        "J1": J1, "J2": J2, "K": K, "Kneg": -K, "pts": pts, "point_list": point_list,
        "lab1": f"${Hh}(P_1) =$", "lab2": f"${Hh}(P_2) =$", "lab3": f"${Hh}(P_3) =$",
        "lab4": f"${Hh}(P_4) =$", "lab5": f"${Hh}(P_5) =$", "Hh_tex_d": f"${Hh}(P_{kd + 1}) =$",
        "Kabs": num(abs(K)),
        "comp_choices": comp_choices, "shape_choices": shape_choices,
        "ks": ks, "ks_a": num(ak), "ks_b": num(bk), "dmode": dmode, "kd": kd, "kd1": kd + 1,
        "Pd_u": num(pts[kd]), "d_what": d_what, "d_words": d_words,
        "cross_tex": cross_tex, "curl_tex": curl_tex, "slope": num(slope), "Jk": num(Jk),
        "sheet_line": (f"Sheet: $\\mathbf{{H}} = \\tfrac12\\mathbf{{J}}_s\\times\\hat{{n}}$ with $\\hat{{n}} = \\mathrm{{sgn}}({u})\\,{uh}$, so "
                       f"$\\mathbf{{H}} = \\tfrac12({num(K)})\\,\\mathrm{{sgn}}({u})\\,({ch}\\times{uh}) = \\tfrac12({num(K)})\\,\\mathrm{{sgn}}({u})\\,({plus_h})$: "
                       f"${Hh} = {num(s_h * K / 2)}$ A/m for ${u} \\gt 0$ and ${num(-s_h * K / 2)}$ A/m for ${u} \\lt 0$."),
        "slab1_line": slab_line(1, J1, w1, a1, b1, m1), "slab2_line": slab_line(2, J2, w2, a2, b2, m2),
        "rows": rows, "Hd_str": num(Hd),
        "minus_h": ("-" if s_h > 0 else "+") + hh,
    })
    data["params"].update(svg)
    for i, H in enumerate(Hs):
        data["correct_answers"][f"H{i + 1}"] = float(H)
    data["correct_answers"]["Hd"] = float(Hd)
