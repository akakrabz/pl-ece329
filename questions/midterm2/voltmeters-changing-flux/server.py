"""Exam-style: what voltmeters read in a changing flux (Lecture 14; family Summer 2017 HE2 #3 / Summer 2018 HE2 #3).

Square circuit ABCD of side s in the plane z = 0: A = (0,0), B = (s,0), C = (s,s), D = (0,s), x to the right, y up,
z out of the page. Side k joins corner k and corner k+1 (AB, BC, CD, DA); two or three sides carry resistors, the rest
are ideal wires. A uniform B = (B0 + beta t) z-hat fills the plane. Reference circulation A->B->C->D (counterclockwise
seen from +z), so dS = +z-hat:
    Psi(t1) = (B0 + beta t1) s^2,   emf = -dPsi/dt = -beta s^2,   I = emf / R_total   (I > 0: current flows A->B->C->D)
Ideal voltmeters V1, V2, V3 all have the + lead on corner X and the - lead on the opposite corner Y.
    V1: leads hug sides X->W1->Y, W1 = the corner after X going counterclockwise (the reference direction)
    V2: leads hug sides X->W2->Y, W2 = the corner after X going clockwise
    V3: leads run straight along the diagonal XY
A meter reads the line integral of E from its + lead to its - lead along its own leads. E.dl = 0 along ideal wires and
leads; across a resistor the integral is (current) x R in the direction of the current. Faraday for the loop
"meter leads + a circuit branch from X to Y":
    V1 (zero enclosed area): V1 = drops along X->W1->Y (reference direction) = +I (R[X] + R[W1])
    V2 (zero enclosed area): V2 = drops along X->W2->Y (against it)          = -I (R[W2] + R[Y])
        V1 - V2 = I R_total = emf      (the two sets of leads enclose the whole square)
    V3: triangle X->W1->Y->X is counterclockwise with area s^2/2:  V1 - V3 = emf/2   ->   V3 = V1 - emf/2 = (V1 + V2)/2
Part (f): the same dPsi/dt confined to a thin solenoid inside triangle X W1 Y (V1's half) -> the loop formed by V3's and
V2's leads encloses no changing flux -> V3 = V2.  Inside triangle X W2 Y (V2's half) -> V3 = V1.
"""
import random

CORNER = "ABCD"
SIDE = ["AB", "BC", "CD", "DA"]
PX = [(120, 220), (280, 220), (280, 60), (120, 60)]        # figure coordinates of A, B, C, D (y down)
CENTRE = (200, 140)


def fnum(x, sig=4):
    """Plain decimal with `sig` significant figures (never exponent notation)."""
    if abs(x) < 1e-12:
        return "0"
    s = f"{x:.{sig}g}"
    if "e" in s:
        e = int(s.split("e")[1])
        s = f"{x:.{max(0, sig - 1 - e)}f}"
    return s


def unit(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = abs(dx) + abs(dy)                  # axis-aligned sides only
    return dx // n, dy // n


def outward(k):
    """Outward unit normal (figure coordinates) of side k."""
    (x0, y0), (x1, y1) = PX[k], PX[(k + 1) % 4]
    mx, my = (x0 + x1) // 2, (y0 + y1) // 2
    dx, dy = mx - CENTRE[0], my - CENTRE[1]
    n = abs(dx) + abs(dy)
    return dx // n, dy // n


def side_path(k, has_r):
    a, b = PX[k], PX[(k + 1) % 4]
    if not has_r:
        return f"M{a[0]} {a[1]}L{b[0]} {b[1]}"
    ux, uy = unit(a, b)
    nx, ny = -uy, ux
    mx, my = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2
    pts = []
    for t, o in zip([0, 4, 12, 20, 28, 36, 44, 48], [0, 7, -7, 7, -7, 7, -7, 0]):
        pts.append((mx + (t - 24) * ux + o * nx, my + (t - 24) * uy + o * ny))
    return f"M{a[0]} {a[1]}" + "".join(f"L{x} {y}" for x, y in pts) + f"L{b[0]} {b[1]}"


def lead_path(X, W, Y, sX, sY):
    """Dashed lead route X -> (meter outside corner W) -> Y hugging sides sX (between X and W) and sY (between W and
    Y); the dashes stop at the meter circle (radius 12) centred on the outside corner."""
    g, r = 16, 12
    n1, n2 = outward(sX), outward(sY)
    p0 = PX[X]
    p1 = (PX[X][0] + g * n1[0], PX[X][1] + g * n1[1])
    pw = (PX[W][0] + g * (n1[0] + n2[0]), PX[W][1] + g * (n1[1] + n2[1]))
    p3 = (PX[Y][0] + g * n2[0], PX[Y][1] + g * n2[1])
    p4 = PX[Y]
    u1, u2 = unit(p1, pw), unit(pw, p3)
    qa = (pw[0] - r * u1[0], pw[1] - r * u1[1])
    qb = (pw[0] + r * u2[0], pw[1] + r * u2[1])
    d = (f"M{p0[0]} {p0[1]}L{p1[0]} {p1[1]}L{qa[0]} {qa[1]}"
         f"M{qb[0]} {qb[1]}L{p3[0]} {p3[1]}L{p4[0]} {p4[1]}")
    return d, pw


def diag_out(k):
    dx, dy = PX[k][0] - CENTRE[0], PX[k][1] - CENTRE[1]
    return (1 if dx > 0 else -1), (1 if dy > 0 else -1)


def build_svg(R, rname, X, Y, W1, W2):
    parts = ['<svg viewBox="0 0 400 296" width="400" style="max-width:100%;height:auto" role="img" '
             'aria-label="Square circuit ABCD with its resistors and the routes of the three voltmeter leads">',
             '<g fill="none" stroke="currentColor" stroke-width="2">']
    for k in range(4):
        parts.append(f'<path d="{side_path(k, R[k] > 0)}"/>')
    parts.append('</g><g fill="none" stroke="currentColor" stroke-width="1.3" stroke-dasharray="5 3">')
    d1, m1 = lead_path(X, W1, Y, X, W1)
    d2, m2 = lead_path(X, W2, Y, W2, Y)
    parts.append(f'<path d="{d1}"/><path d="{d2}"/>')
    ox, oy = diag_out(Y)                    # unit-ish step from X toward Y along the diagonal
    gx, gy = CENTRE[0] - 9 * ox, CENTRE[1] - 9 * oy
    hx, hy = CENTRE[0] + 9 * ox, CENTRE[1] + 9 * oy
    parts.append(f'<path d="M{PX[X][0]} {PX[X][1]}L{gx} {gy}M{hx} {hy}L{PX[Y][0]} {PX[Y][1]}"/></g>')
    parts.append('<g fill="none" stroke="currentColor" stroke-width="1.3">')
    for (cx, cy) in (m1, m2, CENTRE):
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="12"/>')
    parts.append('</g><g fill="currentColor" font-size="12" text-anchor="middle" font-family="sans-serif">')
    for (cx, cy), lab in ((m1, "V1"), (m2, "V2"), (CENTRE, "V3")):
        parts.append(f'<text x="{cx}" y="{cy + 4}">{lab}</text>')
    for k in (W1, W2):                                         # labels inside, near corners without a diagonal
        ox, oy = diag_out(k)
        parts.append(f'<text x="{PX[k][0] - 14 * ox}" y="{PX[k][1] - 14 * oy + 4}">{CORNER[k]}</text>')
    for k, sgn in ((X, "+"), (Y, "−")):                    # labels outside, on the diagonal
        ox, oy = diag_out(k)
        parts.append(f'<text x="{PX[k][0] + 14 * ox}" y="{PX[k][1] + 14 * oy + 4}">{CORNER[k]}</text>')
        parts.append(f'<text x="{PX[k][0] + 27 * ox}" y="{PX[k][1] + 27 * oy + 4}" font-size="15">{sgn}</text>')
    for k in range(4):
        if R[k] > 0:
            nx, ny = outward(k)
            mx, my = (PX[k][0] + PX[(k + 1) % 4][0]) // 2, (PX[k][1] + PX[(k + 1) % 4][1]) // 2
            parts.append(f'<text x="{mx - 22 * nx}" y="{my - 22 * ny + 4}">{rname[k]}</text>')
    legend = ",  ".join(f"{rname[k]} = {R[k]} Ω" for k in range(4) if R[k] > 0)
    parts.append(f'<text x="200" y="290">{legend}</text></g></svg>')
    return "".join(parts)


def generate(data):
    while True:
        s_cm = random.choice([10, 15, 20, 25, 30, 40])
        layout = random.choice(["p0", "p1", "p2", "p3", "o02", "o13"])        # plain-wire side(s)
        plain = {int(layout[1])} if layout[0] == "p" else {int(layout[1]), int(layout[2])}
        vals = iter(random.sample([2, 3, 4, 5, 6, 8, 10, 12, 15, 20], 4 - len(plain)))
        R = [0 if k in plain else next(vals) for k in range(4)]
        X = random.randrange(4)
        Y, W1, W2 = (X + 2) % 4, (X + 1) % 4, (X + 3) % 4
        Rp1 = R[X] + R[W1]                    # sides X-W1 and W1-Y
        Rp2 = R[W2] + R[Y]                    # sides W2-X and Y-W2
        beta = random.choice([-5, -4, -3, -2.5, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5])   # dBz/dt, T/s
        B0 = random.choice([-1.2, -0.8, -0.6, -0.4, 0.4, 0.6, 0.8, 1.2])                          # T
        t1 = random.choice([0.1, 0.2, 0.25, 0.4, 0.5])                                             # s
        Bt1 = round(B0 + beta * t1, 6)
        if Rp1 > 0 and Rp2 > 0 and Rp1 != Rp2 and abs(Bt1) >= 0.05:
            break
    sol_half = random.choice([1, 2])          # part (f): the solenoid sits in V1's half (1) or V2's half (2)

    s = s_cm / 100
    area = s * s
    Rtot = sum(R)
    Psi = Bt1 * area                          # Wb
    emf = -beta * area                        # V, along A->B->C->D
    I = emf / Rtot                            # A, positive along A->B->C->D
    V1 = I * Rp1
    V2 = -I * Rp2
    V3 = V1 - emf / 2
    V3f = V2 if sol_half == 1 else V1         # part (f)

    # resistor names R1, R2(, R3) in side order AB, BC, CD, DA
    rname, n = {}, 0
    for k in range(4):
        if R[k] > 0:
            n += 1
            rname[k] = f"R{n}"
    rtex = {k: f"R_{rname[k][1]}" for k in rname}
    res_list = [f"${rtex[k]} = {R[k]}\\ \\Omega$ on side {SIDE[k]}" for k in range(4) if R[k] > 0]
    res_sentence = (", ".join(res_list[:-1]) + " and " + res_list[-1]) if len(res_list) > 1 else res_list[0]
    plain_sides = [SIDE[k] for k in range(4) if R[k] == 0]
    if len(plain_sides) == 1:
        plain_sentence = f"Side {plain_sides[0]} is a plain wire."
    else:
        plain_sentence = f"Sides {plain_sides[0]} and {plain_sides[1]} are plain wires."
    res_sentence = "The circuit contains " + res_sentence + ". " + plain_sentence

    cX, cY, cW1, cW2 = CORNER[X], CORNER[Y], CORNER[W1], CORNER[W2]
    beta_signed = (f"+ {fnum(beta)}" if beta > 0 else f"- {fnum(abs(beta))}")

    # term strings for the answer panel
    def drop_terms(sides, sign):
        sym = []
        for k in sides:
            if R[k] > 0:
                sym.append(("" if sign > 0 else "-") + f"I{rtex[k]}")
            else:
                sym.append("0")
        return " + ".join(sym).replace("+ -", "- ")

    v1_sym = drop_terms([X, W1], +1)
    v2_sym = drop_terms([W2, Y], -1)
    side_names_1 = f"{cX}{cW1} and {cW1}{cY}"
    side_names_2 = f"{cX}{cW2} and {cW2}{cY}"

    I_mA, emf_mV = I * 1e3, emf * 1e3
    V1_mV, V2_mV, V3_mV, V3f_mV = V1 * 1e3, V2 * 1e3, V3 * 1e3, V3f * 1e3

    if emf > 0:
        sense_words = (r"positive: the emf, and the current, circulate in the reference direction "
                       r"A$\to$B$\to$C$\to$D (counterclockwise seen from $+z$). Lenz: $dB_z/dt \lt 0$, so the "
                       r"$+\hat{z}$ flux is falling and the current makes its own $+\hat{z}$ field inside the loop to oppose the change.")
    else:
        sense_words = (r"negative: the emf, and the current, circulate against the reference direction, "
                       r"i.e. A$\to$D$\to$C$\to$B (clockwise seen from $+z$). Lenz: $dB_z/dt \gt 0$, so the "
                       r"$+\hat{z}$ flux is growing and the current makes a $-\hat{z}$ field inside the loop to oppose the change.")

    half_tri = {1: f"{cX}{cW1}{cY}", 2: f"{cX}{cW2}{cY}"}
    sol_tri = half_tri[sol_half]
    other_tri = half_tri[3 - sol_half]
    f_choices = [
        {"text": f"$V_3 = V_2 = {fnum(V2_mV)}\\ \\mathrm{{mV}}$",
         "correct": "true" if sol_half == 1 else "false",
         "feedback": ("Right: the loop formed by the leads of $V_3$ and $V_2$ encloses no changing flux." if sol_half == 1 else
                      "$V_3$ and $V_2$ now differ by the emf of the loop their leads form, which encloses the whole solenoid.")},
        {"text": f"$V_3 = V_1 = {fnum(V1_mV)}\\ \\mathrm{{mV}}$",
         "correct": "true" if sol_half == 2 else "false",
         "feedback": ("Right: the loop formed by the leads of $V_3$ and $V_1$ encloses no changing flux." if sol_half == 2 else
                      "$V_1$ and $V_3$ now differ by the full emf: the loop formed by their leads is the one that encloses the solenoid.")},
        {"text": f"unchanged, $V_3 = \\tfrac12(V_1 + V_2) = {fnum(V3_mV)}\\ \\mathrm{{mV}}$, because the total $d\\Psi/dt$ through the square is the same",
         "correct": "false",
         "feedback": "The emf and current are unchanged, but a reading depends on where the changing flux is relative to the meter's own leads."},
        {"text": "$V_3 = 0$: no changing field reaches the leads, so the meter reads nothing",
         "correct": "false",
         "feedback": "The induced $\\mathbf{E}$ reaches far outside the solenoid, and $V_3$ still sees the $IR$ drops of the circuit."},
    ]
    random.shuffle(f_choices)
    f_words = (f"The meter readings differ by the emf of the loop formed by their leads. With all the changing flux inside "
               f"triangle {sol_tri}, the loop formed by the diagonal and the leads of "
               + ("$V_2$" if sol_half == 1 else "$V_1$") +
               f" (around triangle {other_tri}) encloses none of it, so those two readings are equal: "
               + (f"$V_3 = V_2 = {fnum(V2_mV)}$ mV" if sol_half == 1 else f"$V_3 = V_1 = {fnum(V1_mV)}$ mV")
               + ". The loop formed by the diagonal and the other meter's leads now encloses the whole solenoid, so that "
               "pair differs by the full emf. Nothing else changes: the circuit links the same $d\\Psi/dt$, so "
               "$\\mathcal{E}$, $I$, $V_1$ and $V_2$ keep their values.")

    data["params"].update({
        "s_cm": s_cm, "B0": fnum(B0), "beta_signed": beta_signed, "t1": fnum(t1),
        "res_sentence": res_sentence,
        "X": cX, "Y": cY, "W1": cW1, "W2": cW2,
        "svg": build_svg(R, {k: rname[k] for k in rname}, X, Y, W1, W2),
        "side_names_1": side_names_1, "side_names_2": side_names_2,
        "sol_tri": sol_tri, "f_choices": f_choices, "f_words": f_words,
        # answer panel
        "area_str": fnum(area), "Bt1_str": fnum(Bt1), "Psi_str": fnum(Psi * 1e3),
        "beta_str": fnum(beta), "emf_str": fnum(emf_mV), "sense_words": sense_words,
        "Rtot_sym": " + ".join(rtex[k] for k in range(4) if R[k] > 0),
        "Rtot": Rtot, "I_str": fnum(I_mA),
        "v1_sym": v1_sym, "v2_sym": v2_sym, "Rp1": Rp1, "Rp2": Rp2,
        "V1_str": fnum(V1_mV), "V2_str": fnum(V2_mV), "V3_str": fnum(V3_mV), "half_emf_str": fnum(emf_mV / 2),
        "diff_str": fnum(V1_mV - V2_mV),
        # V3 = (V1 + V2)/2 is a small difference of the other readings: let its tolerance reflect their precision
        "V3_atol": fnum(0.01 * max(abs(V1_mV), abs(V2_mV)), 2),
        # data for the independent check (geometry and layout)
        "R_sides": R, "Xi": X, "sol_half": sol_half, "beta": beta, "s": s, "t1_val": t1, "B0_val": B0,
    })
    data["correct_answers"]["Psi"] = Psi * 1e3       # mWb
    data["correct_answers"]["emf"] = emf_mV          # mV
    data["correct_answers"]["I"] = I_mA              # mA
    data["correct_answers"]["V1"] = V1_mV            # mV
    data["correct_answers"]["V2"] = V2_mV            # mV
    data["correct_answers"]["V3"] = V3_mV            # mV
