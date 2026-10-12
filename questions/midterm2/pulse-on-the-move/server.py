"""Exam-style: a pulse moving rigidly through a dielectric (Lecture 18; family SP18 Exam 2 #4).

Snapshot at t = t0:  E(z, t0) = e-hat F(z), e-hat = x-hat or y-hat, F piecewise linear (asymmetric triangle or
trapezoid): 0 for z < zA, rising to E0 at zB, flat to zC (zC = zB for a triangle), falling to 0 at zD, 0 beyond.
The pulse travels toward s z-hat (s = +1 or -1) at speed v without change of shape, so
    E(z, t) = e-hat F(z - s v (t - t0))                      (d'Alembert: a function of t - s z / v)
Events (z_i, t_i): one on the rising ramp, one on the falling ramp, one ahead of the front (E = 0) at a time t3 < t0
chosen so that the wrong direction of shift would land inside the pulse.
    eps_r = (c/v)^2 with c = 1/sqrt(mu0 eps0) (mu_r = 1)
    eta = mu0 v;  H = (1/eta) (s z-hat) x E;   |H| = |E|/eta
"""
import math
import random

MU0 = 4e-7 * math.pi
EPS0 = 8.854e-12
C = 1 / math.sqrt(MU0 * EPS0)          # m/s
VEC = {"x": r"\hat{x}", "y": r"\hat{y}", "z": r"\hat{z}"}


def fnum(x, sig=4):
    if abs(x) < 1e-12:
        return "0"
    s = f"{x:.{sig}g}"
    if "e" in s:
        e = int(s.split("e")[1])
        s = f"{x:.{max(0, sig - 1 - e)}f}"
    return s


def zfmt(z):
    """Positions/times: up to 3 decimals, trailing zeros removed."""
    s = f"{round(z, 3):.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def F(z, zA, zB, zC, zD, E0):
    if z <= zA or z >= zD:
        return 0.0
    if z < zB:
        return E0 * (z - zA) / (zB - zA)
    if z <= zC:
        return E0
    return E0 * (zD - z) / (zD - zC)


def signed_vec(sign, axis):
    return ("+" if sign > 0 else "-") + VEC[axis]


def build_svg(zA, zB, zC, zD, E0, pol, t0, s):
    L = zD - zA
    zmin, zmax = zA - 0.3 * L, zD + 0.3 * L
    X = lambda z: round(40 + (z - zmin) / (zmax - zmin) * 340, 1)
    base, peak = (118, 48) if E0 > 0 else (58, 128)
    lab_y = base + 17 if E0 > 0 else base - 8
    p = ['<svg viewBox="0 0 400 176" width="400" style="max-width:100%;height:auto" role="img" '
         'aria-label="Snapshot of the pulse at t0">',
         '<g fill="none" stroke="currentColor">',
         f'<path d="M30 {base}H392M386 {base - 5}L392 {base}L386 {base + 5}" stroke-width="1.2"/>',
         f'<path d="M{X(zmin) - 5} {base}L{X(zA)} {base}L{X(zB)} {peak}L{X(zC)} {peak}L{X(zD)} {base}" stroke-width="2.5"/>',
         f'<path d="M{X(zB)} {peak}V{base}" stroke-dasharray="3 3" stroke-width="1"/>']
    if zC > zB:
        p.append(f'<path d="M{X(zC)} {peak}V{base}" stroke-dasharray="3 3" stroke-width="1"/>')
    ay = 22 if E0 > 0 else 160
    if s > 0:
        p.append(f'<path d="M300 {ay}H350M344 {ay - 5}L350 {ay}L344 {ay + 5}" stroke-width="1.5"/>')
    else:
        p.append(f'<path d="M350 {ay}H300M306 {ay - 5}L300 {ay}L306 {ay + 5}" stroke-width="1.5"/>')
    p.append('</g><g fill="currentColor" font-family="sans-serif" font-size="12" text-anchor="middle">')
    ticks = [zA, zB] + ([zC] if zC > zB else []) + [zD]
    for z in ticks:
        p.append(f'<text x="{X(z)}" y="{lab_y}">{zfmt(z)}</text>')
    p.append(f'<text x="392" y="{lab_y}" text-anchor="end">z (m)</text>')
    p.append(f'<text x="{(X(zB) + X(zC)) / 2}" y="{peak - 7 if E0 > 0 else peak + 16}">{zfmt(E0).replace("-", "&#8722;")} V/m</text>')
    p.append(f'<text x="325" y="{ay - 8 if E0 > 0 else ay + 14}">travel</text>')
    p.append(f'<text x="110" y="{ay if E0 > 0 else ay}">E{pol}(z, {zfmt(t0)} ns)</text>')
    p.append('</g></svg>')
    return "".join(p)


def generate(data):
    found = False
    while not found:
        s = random.choice([1, -1])
        pol = random.choice(["x", "y"])
        v = random.choice([0.10, 0.12, 0.15, 0.18, 0.25])          # m/ns
        E0 = random.choice([2, 3, 4, 5, 6, 8, 10]) * random.choice([1, -1])
        t0 = random.choice([2, 3, 4, 5, 6, 8])
        shape = random.choice(["triangle", "trapezoid"])
        a, b = random.sample([0.4, 0.6, 0.8, 1.2], 2)               # rise and fall lengths (m), unequal
        flat = 0.0 if shape == "triangle" else random.choice([0.4, 0.6])
        zA = random.choice([-0.6, -0.4, 0.0, 0.2, 0.4, 0.6, 1.0])
        zB = round(zA + a, 3)
        zC = round(zB + flat, 3)
        zD = round(zC + b, 3)
        prof = (lambda zA, zB, zC, zD, E0: (lambda z: F(z, zA, zB, zC, zD, E0)))(zA, zB, zC, zD, E0)

        def event(zeta, t):
            return round(zeta + s * v * (t - t0), 4)

        def wrong(z, t):                                         # value with the shift in the wrong direction
            return prof(z + s * v * (t - t0))

        for _ in range(300):
            f1, f2 = random.sample([0.25, 0.5, 0.75], 2)            # different fractions: E1 != E2
            zeta1 = round(zA + f1 * a, 4)                      # rising ramp: F = f1 E0
            zeta2 = round(zD - f2 * b, 4)                      # falling ramp: F = f2 E0
            t1 = t0 + random.choice([1, 2, 3, 4, 5, 6, 8, 10])
            t2 = t0 + random.choice([-2, -1, 1, 2, 3, 4, 6, 8])
            t3 = random.randrange(0, t0)
            delta = random.choice([0.2, 0.3, 0.4, 0.6])
            zeta3 = round(zD + delta if s > 0 else zA - delta, 4)   # ahead of the front
            z1, z2, z3 = event(zeta1, t1), event(zeta2, t2), event(zeta3, t3)
            if (abs(wrong(z1, t1) - prof(zeta1)) >= 0.2 * abs(E0) and abs(wrong(z2, t2) - prof(zeta2)) >= 0.2 * abs(E0)
                    and abs(wrong(z3, t3)) >= 0.2 * abs(E0) and len({z1, z2, z3}) == 3):
                found = True
                break
    E1, E2, E3 = prof(zeta1), prof(zeta2), prof(zeta3)

    v_si = v * 1e9
    eps_r = (C / v_si) ** 2
    eta = MU0 * v_si
    H1 = abs(E1) / eta
    # direction of H at event 1: (s z-hat) x (sign(E1) e-hat);  z x x = +y,  z x y = -x
    sgnE1 = 1 if E1 > 0 else -1
    if pol == "x":
        h_axis, h_sign = "y", s * sgnE1
    else:
        h_axis, h_sign = "x", -s * sgnE1

    # (b) general expression, F the snapshot function
    vs, t0s = zfmt(v), zfmt(t0)
    op = "-" if s > 0 else "+"
    nop = "+" if s > 0 else "-"
    e_tex = VEC[pol]
    expr_choices = [
        {"text": f"$\\mathbf{{E}}(z,t) = {e_tex}\\,F\\big(z {op} {vs}\\,(t - {t0s})\\big)$", "correct": "true",
         "feedback": "Right: the field at $(z,t)$ is the snapshot value at the point the pattern has moved away from."},
        {"text": f"$\\mathbf{{E}}(z,t) = {e_tex}\\,F\\big(z {nop} {vs}\\,(t - {t0s})\\big)$", "correct": "false",
         "feedback": "This pulse moves the wrong way. Check one feature: at a later time it must sit further along the direction of travel."},
        {"text": f"$\\mathbf{{E}}(z,t) = {e_tex}\\,F\\big(z {op} {vs}\\,t\\big)$", "correct": "false",
         "feedback": "At $t = t_0$ this does not reproduce the given snapshot: the shift must vanish at $t_0$, not at $t = 0$."},
        {"text": f"$\\mathbf{{E}}(z,t) = {e_tex}\\,F\\big(z {op} {vs}\\,(t + {t0s})\\big)$", "correct": "false",
         "feedback": "The shift is the distance travelled since $t_0$, which is $v\\,(t - t_0)$."},
        {"text": f"$\\mathbf{{E}}(z,t) = {e_tex}\\,F\\big(z {op} (t - {t0s})/{vs}\\big)$", "correct": "false",
         "feedback": "The distance travelled is $v$ times the elapsed time; dividing by $v$ gives units of ns$^2$/m, not m."},
    ]
    random.shuffle(expr_choices)

    hdir_choices = [
        {"text": f"${signed_vec(h_sign, h_axis)}$", "correct": "true",
         "feedback": "Right: $\\mathbf{H} = \\hat{k}\\times\\mathbf{E}/\\eta$, and $\\mathbf{E}\\times\\mathbf{H}$ points along the travel."},
        {"text": f"${signed_vec(-h_sign, h_axis)}$", "correct": "false",
         "feedback": "Then $\\mathbf{E}\\times\\mathbf{H}$ would point against the direction of travel. Use $\\mathbf{H} = \\hat{k}\\times\\mathbf{E}/\\eta$, with $\\hat{k}$ first."},
        {"text": f"${signed_vec(sgnE1, pol)}$, parallel to $\\mathbf{{E}}$", "correct": "false",
         "feedback": "In a plane TEM wave $\\mathbf{H}$ is perpendicular to $\\mathbf{E}$."},
        {"text": f"${signed_vec(s, 'z')}$, along the direction of travel", "correct": "false",
         "feedback": "A plane wave is transverse: neither $\\mathbf{E}$ nor $\\mathbf{H}$ has a component along the direction of travel."},
    ]
    random.shuffle(hdir_choices)

    # answer-panel helpers
    def zeta_line(i, z, t, zeta):
        return (f"\\zeta_{i} = z_{i} {op} v(t_{i} - t_0) = {zfmt(z)} {op} {vs}\\,({zfmt(t)} - {t0s}) = {zfmt(zeta)}\\ \\mathrm{{m}}")

    front_t0 = zD if s > 0 else zA
    front_t3 = front_t0 + s * v * (t3 - t0)
    data["params"].update({
        "dir_word": "+z" if s > 0 else "-z", "pol": pol, "e_tex": e_tex,
        "v": vs, "v_si": fnum(v_si), "v_tex": fnum(v_si / 1e8) + r"\times10^{8}", "E0": zfmt(E0), "t0": t0s,
        "zA": zfmt(zA), "zB": zfmt(zB), "zC": zfmt(zC), "zD": zfmt(zD), "a": zfmt(a), "b": zfmt(b),
        "flat_desc": (f"keeps that value up to $z = {zfmt(zC)}\\ \\mathrm{{m}}$, " if flat > 0 else ""),
        "svg": build_svg(zA, zB, zC, zD, E0, pol, t0, s),
        "z1": zfmt(z1), "t1": zfmt(t1), "z2": zfmt(z2), "t2": zfmt(t2), "z3": zfmt(z3), "t3": zfmt(t3),
        "expr_choices": expr_choices, "hdir_choices": hdir_choices,
        # answer panel
        "op": op, "nop": nop, "zeta1": zfmt(zeta1), "zeta2": zfmt(zeta2), "zeta1_line": zeta_line(1, z1, t1, zeta1), "zeta2_line": zeta_line(2, z2, t2, zeta2),
        "zeta3_line": zeta_line(3, z3, t3, zeta3),
        "E1_str": fnum(E1), "E2_str": fnum(E2), "f1": fnum(f1), "f2": fnum(f2),
        "front_t0": zfmt(front_t0), "front_t3": zfmt(front_t3),
        "front_word": ("the end at $z = " + zfmt(zD) + "$ m, the high-$z$ end, since the pulse moves toward $+z$" if s > 0
                       else "the end at $z = " + zfmt(zA) + "$ m, the low-$z$ end, since the pulse moves toward $-z$"),
        "eps_str": fnum(eps_r), "eta_str": fnum(eta), "H1_str": fnum(H1 * 1e3), "absE1": fnum(abs(E1)),
        "H_vec": signed_vec(h_sign, h_axis), "k_vec": signed_vec(s, "z"), "E1_vec": signed_vec(sgnE1, pol),
        # raw data for the independent check
        "s_dir": s, "v_val": v, "E0_val": E0, "t0_val": t0,
        "corners": [zA, zB, zC, zD], "events": [[z1, t1], [z2, t2], [z3, t3]], "h_sign": h_sign, "h_axis": h_axis,
    })
    data["correct_answers"]["E1"] = E1        # V/m, signed component along e-hat
    data["correct_answers"]["E2"] = E2
    data["correct_answers"]["E3"] = E3        # 0: not yet arrived
    data["correct_answers"]["epsr"] = eps_r
    data["correct_answers"]["H1"] = H1 * 1e3  # mA/m, magnitude
