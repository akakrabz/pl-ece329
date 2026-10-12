"""Exam-style: a shorted parallel-plate line filled with a magnetic material (Lectures 13, 15, 17; family SP18 Exam 2 #3).

Plates on u = 0 and u = d (u = x or y at random), width w along the third axis q, running along z from the source
(z = 0) to the short (z = l); d << w, l, fringing neglected. The gap is filled with a linear magnetic material
mu = mu_r mu0 (the changed detail). The source drives I along the "outgoing" plate (u = d or u = 0 at random) toward the
short; it returns along the other plate. With s_top = +1 if the outgoing plate is u = d and -1 otherwise:
    J_s(u = d) = s_top (I/w) z-hat,     J_s(u = 0) = -s_top (I/w) z-hat
    H in the gap = 1/2 J_s(d) x (-u-hat) + 1/2 J_s(0) x (+u-hat) = s_top (I/w) (u-hat x z-hat);  zero outside the plates
      (u = y: u-hat x z-hat = +x-hat;   u = x: u-hat x z-hat = -y-hat).  Free currents set H; mu_r does not change it.
    B = mu_r mu0 H;   Psi = B d l (rectangle bounded by the current path);   L = Psi / I = mu_r mu0 d l / w
    w_m = 1/2 mu H^2,  W = w_m (w d l) = 1/2 L I^2
    M = (mu_r - 1) H;  bound current on the filling's face u = d (outward normal +u-hat):
        J_sM = M x u-hat = (mu_r - 1) s_top (I/w) z-hat   (same sense as the free current on the adjacent plate)
    Scaling d -> k_d d, I -> k_I I (same filling): B x k_I,  Psi x k_d k_I,  L x k_d.
"""
import math
import random
from fractions import Fraction

MU0 = 4e-7 * math.pi
# (description, mu_r, largest B allowed so the material stays well inside its linear range, T).
# Typical initial permeabilities: carbonyl/powdered-iron mixes 10 and 75; NiZn ferrites 125 and 800; MnZn ferrite 2000.
MATERIALS = [
    ("a powdered-iron composite", 10, 0.3),
    ("a powdered-iron composite", 75, 0.3),
    ("a nickel&ndash;zinc ferrite", 125, 0.1),
    ("a nickel&ndash;zinc ferrite", 800, 0.1),
    ("a manganese&ndash;zinc ferrite", 2000, 0.1),
]
VEC = {"x": r"\hat{x}", "y": r"\hat{y}", "z": r"\hat{z}"}


def fnum(x, sig=4):
    if abs(x) < 1e-12:
        return "0"
    s = f"{x:.{sig}g}"
    if "e" in s:
        e = int(s.split("e")[1])
        s = f"{x:.{max(0, sig - 1 - e)}f}"
    return s


def sci(x, sig=4):
    """LaTeX scientific notation, e.g. 2.000\\times10^{-6}."""
    m, e = f"{x:.{sig - 1}e}".split("e")
    return m + r"\times10^{" + str(int(e)) + "}"


def signed_vec(sign, axis):
    return ("+" if sign > 0 else "-") + VEC[axis]


def ftex(fr):
    fr = Fraction(fr)
    if fr.denominator == 1:
        return str(fr.numerator)
    return r"\tfrac{" + str(fr.numerator) + "}{" + str(fr.denominator) + "}"


def build_svg(u, q_axis, q_into, out_top):
    """Side view in the (z, u) plane: z to the right, u up; plates, filling, source, short, current arrows."""
    yt, yb, x0, x1 = 70, 130, 90, 320
    p = ['<svg viewBox="0 0 420 210" width="420" style="max-width:100%;height:auto" role="img" '
         'aria-label="Side view of the shorted parallel-plate line">',
         f'<rect x="{x0}" y="{yt}" width="{x1 - x0}" height="{yb - yt}" fill="currentColor" fill-opacity="0.12" stroke="none"/>',
         '<g fill="none" stroke="currentColor">',
         f'<path d="M{x0} {yt}H{x1}M{x0} {yb}H{x1}" stroke-width="4"/>',
         f'<path d="M{x1} {yt}V{yb}" stroke-width="4"/>',
         f'<path d="M{x0} {yt}H60V88M60 112V{yb}H{x0}" stroke-width="1.5"/>',
         '<circle cx="60" cy="100" r="12" stroke-width="1.5"/>',
         ]
    # source arrow: pushes current toward the outgoing plate
    if out_top:
        p.append('<path d="M60 107V93M55 98L60 92L65 98" stroke-width="1.5"/>')
    else:
        p.append('<path d="M60 93V107M55 102L60 108L65 102" stroke-width="1.5"/>')
    # current arrows next to the plates: toward the short on the outgoing plate, back on the other
    ya, yr = (yt - 14, yb + 14) if out_top else (yb + 14, yt - 14)
    p.append(f'<path d="M170 {ya}H250M242 {ya - 5}L250 {ya}L242 {ya + 5}" stroke-width="1.5"/>')
    p.append(f'<path d="M250 {yr}H170M178 {yr - 5}L170 {yr}L178 {yr + 5}" stroke-width="1.5"/>')
    # axes
    p.append('<path d="M360 120H400M394 115L400 120L394 125M360 120V80M355 86L360 80L365 86" stroke-width="1.2"/>')
    p.append('</g><g fill="currentColor" font-family="sans-serif" font-size="13" text-anchor="middle">')
    p.append(f'<text x="210" y="{ya - 5 if ya < yt else ya + 16}">I</text>')
    p.append(f'<text x="210" y="{yr - 5 if yr < yt else yr + 16}">I</text>')
    p.append(f'<text x="205" y="{(yt + yb) // 2 + 5}">magnetic filling</text>')
    p.append(f'<text x="{x0 + 22}" y="{yt - 6}" font-size="12">{u} = d</text>')
    p.append(f'<text x="{x0 + 22}" y="{yb + 18}" font-size="12">{u} = 0</text>')
    p.append(f'<text x="60" y="{yb + 34}" font-size="12">source, z = 0</text>')
    p.append(f'<text x="{x1}" y="{yb + 34}" font-size="12">short, z = &#8467;</text>')
    p.append('<text x="410" y="124" font-size="12">z</text>')
    p.append(f'<text x="360" y="74" font-size="12">{u}</text>')
    p.append(f'<text x="210" y="24" font-size="12">width w along {q_axis}, {"into" if q_into else "out of"} the page</text>')
    p.append('</g></svg>')
    return "".join(p)


def generate(data):
    while True:
        mat, mur, Bmax = random.choice(MATERIALS)
        w_cm = random.choice([2, 2.5, 3, 4, 5, 6, 8, 10])
        d_mm = random.choice([0.2, 0.25, 0.4, 0.5, 0.8, 1.0, 1.5, 2.0])
        ell = random.choice([0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
        I = random.choice([0.2, 0.25, 0.4, 0.5, 0.8, 1, 1.5, 2, 2.5, 3, 4, 5, 8, 10])
        kd = random.choice([Fraction(2), Fraction(3), Fraction(1, 2), Fraction(1, 3)])
        kI = random.choice([Fraction(2), Fraction(3), Fraction(1, 2), Fraction(1, 3), Fraction(4), Fraction(1, 4)])
        w, d = w_cm / 100, d_mm / 1000
        H = I / w
        B = mur * MU0 * H
        L = mur * MU0 * d * ell / w
        Psi = L * I
        wm = 0.5 * mur * MU0 * H * H
        W = 0.5 * L * I * I
        JsM = (mur - 1) * H
        vals = [H, B * 1e3, Psi * 1e6, L * 1e6, wm, W * 1e6, JsM / 1e3]
        if (B <= Bmax and all(0.01 <= v <= 1000 for v in vals)
                and float(kI) * B <= Bmax and float(kd) * d <= w / 10):
            break

    u = random.choice(["x", "y"])
    q_axis = "y" if u == "x" else "x"
    # screen: z to the right, u up, so the out-of-page axis is z x u:  z x y = -x (x into the page); z x x = +y (y out)
    q_into = (u == "y")
    out_top = random.choice([True, False])           # outgoing current on the plate at u = d ?
    s_top = 1 if out_top else -1
    # H direction = s_top (u-hat x z-hat): u = y -> +x ; u = x -> -y
    h_axis, h_sign = (("x", s_top) if u == "y" else ("y", -s_top))
    Js_top = s_top * H
    Js_bot = -s_top * H
    JsM_z = s_top * JsM

    dir_choices = [
        {"text": f"${signed_vec(h_sign, h_axis)}$", "correct": "true",
         "feedback": "Right: each plate gives $\\tfrac12\\mathbf{J}_s\\times\\hat{n}$ along this direction, and the two add."},
        {"text": f"${signed_vec(-h_sign, h_axis)}$", "correct": "false",
         "feedback": "Opposite. Evaluate $\\tfrac12\\mathbf{J}_s\\times\\hat{n}$ for each plate, current first, with $\\hat{n}$ pointing from that plate toward the field point."},
        {"text": f"${signed_vec(1, u)}$, normal to the plates", "correct": "false",
         "feedback": "A current sheet's magnetic field is tangential to the sheet. A field normal to the plates is what a charged capacitor has."},
        {"text": f"${signed_vec(s_top, 'z')}$, along the current in the plate at ${u} = d$", "correct": "false",
         "feedback": "The field of a current sheet is perpendicular to the current: $\\tfrac12\\mathbf{J}_s\\times\\hat{n}$."},
        {"text": "nowhere: $\\mathbf{B} = 0$ between the plates, because the equal and opposite plate currents cancel",
         "correct": "false",
         "feedback": "The two plates' fields cancel outside the line, but between the plates they add."},
    ]
    random.shuffle(dir_choices)

    def triple(b, p, l):
        return (Fraction(b), Fraction(p), Fraction(l))

    cands = [
        (triple(kI, kd * kI, kd), "true",
         "Right: $B = \\mu I/w$ ignores the gap, $\\Psi = Bd\\ell$, and $L = \\mu d\\ell/w$ ignores the current."),
        (triple(kI / kd, kI, 1), "false",
         "That treats $B$ like the field of a capacitor, $E = V/d$. Here $B = \\mu I/w$ is set by the surface current alone, whatever the gap."),
        (triple(kI, kd * kI, kd * kI), "false",
         "$L = \\Psi/I$ depends only on the geometry and $\\mu$: the current cancels."),
        (triple(kI, kI / kd, 1 / kd), "false",
         "That is how a capacitance $C = \\epsilon w\\ell/d$ scales. A wider gap encloses more flux per ampere, so $L = \\mu d\\ell/w$ grows with $d$."),
    ]
    scale_choices = []
    for (b, p_, l), corr, fb in cands:
        scale_choices.append({"text": f"$B\\times {ftex(b)}$, $\\ \\Psi\\times {ftex(p_)}$, $\\ L\\times {ftex(l)}$",
                              "correct": corr, "feedback": fb, "factors": [str(b), str(p_), str(l)]})
    random.shuffle(scale_choices)

    vol = w * d * ell
    plate_d_word = "outgoing" if out_top else "return"
    plate_0_word = "return" if out_top else "outgoing"
    data["params"].update({
        "mat": mat, "mur": mur, "w_cm": fnum(w_cm), "d_mm": fnum(d_mm), "ell": fnum(ell), "I": fnum(I),
        "u": u, "q": q_axis, "out_plate": "d" if out_top else "0", "ret_plate": "0" if out_top else "d",
        "svg": build_svg(u, q_axis, q_into, out_top),
        "kd_tex": ftex(kd), "kI_tex": ftex(kI),
        "dir_choices": dir_choices, "scale_choices": scale_choices,
        # answer panel
        "w_m": fnum(w), "d_m": fnum(d), "H_str": fnum(H), "Jstop_str": fnum(Js_top), "Jsbot_str": fnum(Js_bot),
        "plate_d_word": plate_d_word, "plate_0_word": plate_0_word,
        "Js_top_vec": signed_vec(s_top, "z"), "Js_bot_vec": signed_vec(-s_top, "z"),
        "nd_vec": signed_vec(-1, u), "n0_vec": signed_vec(1, u),
        "H_vec": signed_vec(h_sign, h_axis), "H_axis": h_axis,
        "B_str": fnum(B * 1e3), "Psi_str": fnum(Psi * 1e6), "L_str": fnum(L * 1e6),
        "wm_str": fnum(wm), "vol_str": sci(vol), "W_str": fnum(W * 1e6), "kIkd_tex": ftex(kI * kd),
        "M_str": fnum((mur - 1) * H), "chi": mur - 1, "JsM_str": fnum(JsM_z / 1e3), "Jtot_str": fnum(mur * H),
        "u_vec": signed_vec(1, u), "M_vec": signed_vec(h_sign, h_axis),
        # raw data for the independent check
        "h_sign": h_sign, "s_top": s_top, "kd": str(kd), "kI": str(kI),
    })
    data["correct_answers"]["Js_d"] = Js_top             # A/m, z-component on the plate at u = d
    data["correct_answers"]["Js_0"] = Js_bot             # A/m, z-component on the plate at u = 0
    data["correct_answers"]["H"] = H                     # A/m
    data["correct_answers"]["B"] = B * 1e3               # mT
    data["correct_answers"]["Psi"] = Psi * 1e6           # uWb
    data["correct_answers"]["L"] = L * 1e6               # uH
    data["correct_answers"]["wm"] = wm                   # J/m^3
    data["correct_answers"]["W"] = W * 1e6               # uJ
    data["correct_answers"]["JsM"] = JsM_z / 1e3         # kA/m, z-component on the filling's face at u = d
