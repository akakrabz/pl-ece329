"""Exam-style: a current sheet launches two plane waves (Lecture 19; family SP18 Exam 2 #5).

Sheet on the plane n = 0 (n = x, y or z) carrying J_s(t) = sigma J0 g(t) a-hat (a-hat tangential), in a lossless
non-magnetic dielectric eps_r (c = 3e8 m/s and eta0 = 120 pi as stated, so v = 0.3/sqrt(eps_r) m/ns, eta = 120 pi/sqrt(eps_r)).
g(t): 0 for t < 0, linear rise to 1 at t_r, (flat for t_h,) linear fall to 0 at T = t_r + t_h + t_f, 0 afterwards.
Course formulas (Lecture 19), n-hat_t = unit normal from the sheet toward the field point, d = |n|:
    E = -(eta/2) J_s(t - d/v)                (same on both sides, opposite to the current)
    H = 1/2 J_s(t - d/v) x n-hat_t          (reverses across the sheet);   E x H points away from the sheet
Asked: directions (MC); signed E_a, H_h at one point on each side at time t1 (h = the third axis);
the interval during which a probe at distance dq sees a field: dq/v < t < dq/v + T; and where the pulse is at t1 (MC):
    v (t1 - T) < |n| < v t1, strongest at |n| = v (t1 - t_r)  (triangle)  or on v(t1 - t_r - t_h) < |n| < v(t1 - t_r).
Excluded (plane, current) pairs: (y, x) [the exam's], (x, z) [the site's worked problem].
"""
import math
import random

AXES = "xyz"
UNIT = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}
VEC = {"x": r"\hat{x}", "y": r"\hat{y}", "z": r"\hat{z}"}
MEDIA = [(1.0, "air (take $\\epsilon_r = 1$)"), (2.25, "polyethylene, $\\epsilon_r = 2.25$"),
         (6.25, "a glass with $\\epsilon_r = 6.25$"), (9.0, "an alumina ceramic, $\\epsilon_r = 9$")]
C_MNS = 0.3                                   # m/ns  (c = 3e8 m/s, as stated)
ETA0 = 120 * math.pi


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def fnum(x, sig=4):
    if abs(x) < 1e-12:
        return "0"
    s = f"{x:.{sig}g}"
    if "e" in s:
        e = int(s.split("e")[1])
        s = f"{x:.{max(0, sig - 1 - e)}f}"
    return s


def dfmt(x):
    s = f"{round(x, 4):.4f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def sv(sign, axis):
    return ("+" if sign > 0 else "-") + VEC[axis]


def g_of(t, tr, th, tf):
    T = tr + th + tf
    if t <= 0 or t >= T:
        return 0.0
    if t < tr:
        return t / tr
    if t <= tr + th:
        return 1.0
    return (T - t) / tf


def build_svg(tr, th, tf):
    T = tr + th + tf
    tmax = T + 1.5
    X = lambda t: round(50 + t / tmax * 320, 1)
    base, top = 110, 40
    p = ['<svg viewBox="0 0 400 140" width="400" style="max-width:100%;height:auto" role="img" '
         'aria-label="Waveform g(t) of the sheet current">',
         '<g fill="none" stroke="currentColor">',
         f'<path d="M40 {base}H392M386 {base - 5}L392 {base}L386 {base + 5}M50 {base + 4}V18M45 24L50 18L55 24" stroke-width="1.2"/>',
         f'<path d="M50 {base}L{X(tr)} {top}L{X(tr + th)} {top}L{X(T)} {base}L{X(tmax)} {base}" stroke-width="2.5"/>',
         f'<path d="M50 {top}H{X(tr)}" stroke-dasharray="3 3" stroke-width="1"/>',
         f'<path d="M{X(tr)} {top}V{base}" stroke-dasharray="3 3" stroke-width="1"/>']
    if th > 0:
        p.append(f'<path d="M{X(tr + th)} {top}V{base}" stroke-dasharray="3 3" stroke-width="1"/>')
    p.append('</g><g fill="currentColor" font-family="sans-serif" font-size="12" text-anchor="middle">')
    for t in [0, tr] + ([tr + th] if th > 0 else []) + [T]:
        p.append(f'<text x="{X(t)}" y="{base + 17}">{dfmt(t)}</text>')
    p.append(f'<text x="392" y="{base + 17}" text-anchor="end">t (ns)</text>')
    p.append(f'<text x="40" y="{top + 4}" text-anchor="end">1</text>')
    p.append('<text x="64" y="16" text-anchor="start">g(t)</text>')
    p.append('</g></svg>')
    return "".join(p)


def generate(data):
    n_ax, a_ax = random.choice([("x", "y"), ("y", "z"), ("z", "x"), ("z", "y")])
    h_ax = [c for c in AXES if c not in (n_ax, a_ax)][0]
    sigma = random.choice([1, -1])
    eps_r, med = random.choice(MEDIA)
    v = C_MNS / math.sqrt(eps_r)               # m/ns
    eta = ETA0 / math.sqrt(eps_r)
    shape = random.choice(["triangle", "trapezoid"])
    tr, tf = random.sample([1, 2, 3, 4], 2)    # unequal rise and fall
    th = 0 if shape == "triangle" else random.choice([1, 2, 3])
    T = tr + th + tf
    J0 = random.choice([40, 60, 80, 100, 120, 160, 200, 240])     # mA/m

    n_hat, a_hat = UNIT[n_ax], UNIT[a_ax]
    axh = cross(a_hat, n_hat)                  # a x n, along +/- h
    s_an = axh[AXES.index(h_ax)]               # +1 or -1

    def fields(side, d, t):
        """Signed E_a (V/m) and H_h (mA/m) at distance d on side +-1 at time t (ns)."""
        gval = g_of(t - d / v, tr, th, tf)
        Js = sigma * J0 * 1e-3 * gval                      # A/m along a-hat
        E_a = -eta / 2 * Js
        H_h = 0.5 * Js * s_an * side * 1e3                 # 1/2 Js a x (side n) . h-hat, mA/m
        return E_a, H_h, gval

    # events at one instant t1 > T: one retarded time on the rise, one on the fall (random order of sides)
    t1 = T + random.choice([1, 2, 3, 4])
    gv1, gv2 = random.sample([0.25, 0.5, 0.75], 2)
    tau_rise = gv1 * tr
    tau_fall = tr + th + (1 - gv2) * tf
    if random.random() < 0.5:
        tau1, tau2 = tau_rise, tau_fall
    else:
        tau1, tau2 = tau_fall, tau_rise
    side1 = random.choice([1, -1])
    side2 = -side1
    d1, d2 = round(v * (t1 - tau1), 6), round(v * (t1 - tau2), 6)
    E1, H1, g1 = fields(side1, d1, t1)
    E2, H2, g2 = fields(side2, d2, t1)
    n1, n2 = side1 * d1, side2 * d2

    # probe
    dq = random.choice([0.6, 0.9, 1.2, 1.5, 1.8, 2.4])
    sideq = random.choice([1, -1])
    t_on, t_off = dq / v, dq / v + T

    # (a) directions while g > 0
    e_sign = -sigma                                         # E along -sigma a-hat on both sides
    h_plus = sigma * s_an                                   # H along h_plus h-hat on the + side, opposite on the - side

    def dir_text(hp, ep, hm, em):
        return (f"${n_ax} \\gt 0$: $\\mathbf{{H}}$ along ${sv(hp, h_ax)}$, $\\mathbf{{E}}$ along ${sv(ep, a_ax)}$; &nbsp; "
                f"${n_ax} \\lt 0$: $\\mathbf{{H}}$ along ${sv(hm, h_ax)}$, $\\mathbf{{E}}$ along ${sv(em, a_ax)}$")

    dir_choices = [
        {"text": dir_text(h_plus, e_sign, -h_plus, e_sign), "correct": "true", "dirs": [h_plus, e_sign, -h_plus, e_sign],
         "feedback": "Right: $\\mathbf{E} = -\\tfrac{\\eta}{2}\\mathbf{J}_s$ on both sides, $\\mathbf{H} = \\tfrac12\\mathbf{J}_s\\times\\hat{n}$ with $\\hat{n}$ flipping, and $\\mathbf{E}\\times\\mathbf{H}$ points away from the sheet on both sides."},
        {"text": dir_text(h_plus, e_sign, -h_plus, -e_sign), "correct": "false", "dirs": [h_plus, e_sign, -h_plus, -e_sign],
         "feedback": "Tangential $\\mathbf{E}$ cannot jump at the sheet: $\\mathbf{E} = -\\tfrac{\\eta}{2}\\mathbf{J}_s$ has the same direction on both sides. Only $\\mathbf{H}$ reverses."},
        {"text": dir_text(h_plus, e_sign, h_plus, e_sign), "correct": "false", "dirs": [h_plus, e_sign, h_plus, e_sign],
         "feedback": "$\\hat{n}$ in $\\tfrac12\\mathbf{J}_s\\times\\hat{n}$ points toward the field point, so it flips across the sheet and so does $\\mathbf{H}$; tangential $\\mathbf{H}$ jumps by $J_s$."},
        {"text": dir_text(h_plus, -e_sign, -h_plus, -e_sign), "correct": "false", "dirs": [h_plus, -e_sign, -h_plus, -e_sign],
         "feedback": "$\\mathbf{E}$ opposes the current, $\\mathbf{E} = -\\tfrac{\\eta}{2}\\mathbf{J}_s$: the sheet does work on the field. With this choice $\\mathbf{E}\\times\\mathbf{H}$ would point toward the sheet."},
        {"text": dir_text(-h_plus, e_sign, h_plus, e_sign), "correct": "false", "dirs": [-h_plus, e_sign, h_plus, e_sign],
         "feedback": "That is $\\hat{n}\\times\\mathbf{J}_s$, the reverse order. The rule is $\\tfrac12\\mathbf{J}_s\\times\\hat{n}$, current first."},
    ]
    random.shuffle(dir_choices)

    # (e) where the pulse is at t1
    nm = f"|{n_ax}|"

    def region(vel, both=True, mirrored=False):
        lo, hi = vel * (t1 - T), vel * t1
        if th == 0:
            pk = vel * (t1 - (tf if mirrored else tr))
            strong = f"strongest at ${nm} = {dfmt(pk)}\\ \\mathrm{{m}}$"
            pk_lo = pk_hi = pk
        else:
            first = tf if mirrored else tr
            pk_lo, pk_hi = vel * (t1 - first - th), vel * (t1 - first)
            strong = f"strongest (and constant) for ${dfmt(pk_lo)}\\ \\mathrm{{m}} \\lt {nm} \\lt {dfmt(pk_hi)}\\ \\mathrm{{m}}$"
        where = "on both sides of the sheet" if both else f"only on the side ${n_ax} \\gt 0$ (zero for ${n_ax} \\lt 0$)"
        text = f"nonzero only for ${dfmt(lo)}\\ \\mathrm{{m}} \\lt {nm} \\lt {dfmt(hi)}\\ \\mathrm{{m}}$, {where}; {strong}"
        return text, {"lo": lo, "hi": hi, "pk_lo": pk_lo, "pk_hi": pk_hi, "both": both, "zero": False}

    t_ok, d_ok = region(v)
    t_mir, d_mir = region(v, mirrored=True)
    t_one, d_one = region(v, both=False)
    sketch_choices = [
        {"text": t_ok, "correct": "true", "geom": d_ok,
         "feedback": "Right: each feature of $g$ emitted at time $t'$ is now a distance $v(t_1 - t')$ from the sheet, on both sides."},
        {"text": t_mir, "correct": "false", "geom": d_mir,
         "feedback": "This copies the time waveform onto the distance axis. What left the sheet first has travelled farthest, so the snapshot is the waveform mirrored: the start of the pulse is at the outer edge."},
        {"text": t_one, "correct": "false", "geom": d_one,
         "feedback": "A sheet radiates symmetrically: one wave leaves each face, with the same $|\\mathbf{E}|$ and $|\\mathbf{H}|$ at equal distances."},
        {"text": "zero everywhere: the current stopped at $t = " + dfmt(T) + "$ ns, so the sheet no longer produces any field",
         "correct": "false", "geom": {"zero": True},
         "feedback": "The fields are delayed by $d/v$: the waves launched while the current flowed keep travelling outward after the current stops. Only the quasi-static picture would make them vanish at once."},
    ]
    if eps_r != 1.0:
        t_c, d_c = region(C_MNS)
        sketch_choices.append({"text": t_c, "correct": "false", "geom": d_c,
                               "feedback": "These distances use the vacuum speed $c$. In the dielectric the waves travel at $v = c/\\sqrt{\\epsilon_r}$."})
    random.shuffle(sketch_choices)

    J_sign = "" if sigma > 0 else "-"
    flat_desc = (f"stays at $1$ until $t = {dfmt(tr + th)}\\ \\mathrm{{ns}}$, " if th > 0 else "")
    side_word = {1: f"{n_ax} \\gt 0", -1: f"{n_ax} \\lt 0"}
    nhat_word = {1: sv(1, n_ax), -1: sv(-1, n_ax)}

    def panel_point(k, side, d, t, tau, gval, E, H):
        return (f"Point {k} is on the side ${side_word[side]}$ at distance $d = {dfmt(d)}\\ \\mathrm{{m}}$, so it sees the current "
                f"emitted at $t - d/v = {dfmt(t)} - {dfmt(d)}/{dfmt(v)} = {dfmt(tau)}\\ \\mathrm{{ns}}$, when $g = {fnum(gval)}$ and "
                f"$J_{{s,{a_ax}}} = {J_sign}{J0}\\times{fnum(gval)} = {fnum(sigma * J0 * gval)}\\ \\mathrm{{mA/m}}$. "
                f"Then $E_{{{a_ax}}} = -\\tfrac{{\\eta}}{{2}}J_{{s,{a_ax}}} = -({fnum(eta / 2)}\\ \\Omega)({fnum(sigma * J0 * gval)}\\ \\mathrm{{mA/m}}) = {fnum(E)}\\ \\mathrm{{V/m}}$, "
                f"and with $\\hat{{n}} = {nhat_word[side]}$, $\\tfrac12\\mathbf{{J}}_s\\times\\hat{{n}}$ gives "
                f"$H_{{{h_ax}}} = {fnum(H)}\\ \\mathrm{{mA/m}}$.")

    data["params"].update({
        "n": n_ax, "a": a_ax, "h": h_ax, "J_sign": J_sign, "J0": J0, "a_vec": VEC[a_ax],
        "medium": med, "eps_r": fnum(eps_r), "v": dfmt(v), "eta": fnum(eta), "eta_half": fnum(eta / 2),
        "eta_tex": ("120\\pi" if eps_r == 1.0 else f"120\\pi/\\sqrt{{ {fnum(eps_r)} }}"),
        "tr": dfmt(tr), "T": dfmt(T), "flat_desc": flat_desc, "svg": build_svg(tr, th, tf),
        "t1": dfmt(t1), "n1": dfmt(n1), "n2": dfmt(n2), "nq": dfmt(sideq * dq),
        "dir_choices": dir_choices, "sketch_choices": sketch_choices,
        # answer panel
        "panel1": panel_point(1, side1, d1, t1, tau1, g1, E1, H1),
        "panel2": panel_point(2, side2, d2, t1, tau2, g2, E2, H2),
        "dq": dfmt(dq), "t_on": fnum(t_on), "t_off": fnum(t_off),
        "Ea_dir": sv(e_sign, a_ax), "Hp_dir": sv(h_plus, h_ax), "Hm_dir": sv(-h_plus, h_ax),
        "J_dir": sv(sigma, a_ax), "np_vec": sv(1, n_ax), "nm_vec": sv(-1, n_ax),
        "S_plus": sv(1, n_ax), "S_minus": sv(-1, n_ax),
        "front": dfmt(v * t1), "back": dfmt(v * (t1 - T)),
        "sketch_ok": t_ok,
        # raw data for the independent check
        "sigma": sigma, "eps_val": eps_r, "wave": [tr, th, tf], "J0_val": J0,
        "pts": [[side1 * d1, t1], [side2 * d2, t1]], "probe": sideq * dq,
    })
    data["correct_answers"]["E1"] = E1          # V/m, signed a-component at point 1
    data["correct_answers"]["H1"] = H1          # mA/m, signed h-component at point 1
    data["correct_answers"]["E2"] = E2
    data["correct_answers"]["H2"] = H2
    data["correct_answers"]["t_on"] = t_on      # ns
    data["correct_answers"]["t_off"] = t_off    # ns
