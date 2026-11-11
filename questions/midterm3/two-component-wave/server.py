"""Midterm 3 exam-style: a two-component plane wave (phasors, the partner field, average power, handedness).

Family of SP17 Exam 3 #4 and FA23 Exam 3 #2 (re-parameterized). In a lossless non-magnetic dielectric,
    E = a_hat A cos(theta) + b_hat B sin(theta + phi),      theta = omega t -/+ beta w   (travel along u = +/- w_hat)
  phasor     E~ = (a_hat A - j b_hat B e^{j phi}) e^{-/+ j beta w}          (sin x = cos(x - 90 deg))
  partner    H = u x E / eta,  eta = 120 pi / sqrt(eps_r)
  power      <S> = u (A^2 + B^2) / (2 eta)       (the two linear components carry power independently)
  handedness with E = a_hat A cos(theta) + b_hat B cos(theta + psi): E turns from a_hat toward b_hat when b lags
             (sin psi < 0); the wave is right-handed when that turning sense, by the right-hand rule, is along u.
The matching part gives three waves through H = x_hat A cos(theta) + y_hat B cos(theta + phi); H turns the same way as E.
"""
import math
import random

AX = ["x", "y", "z"]
ETA0 = 120 * math.pi


def fnum(x, sig=4):
    if abs(x) < 1e-12:
        return "0"
    s = f"{x:.{sig}g}"
    if "e" in s:
        e = int(s.split("e")[1])
        s = f"{x:.{max(0, sig - 1 - e)}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def cross(a, sa, b, sb):
    ia, ib = AX.index(a), AX.index(b)
    return AX[3 - ia - ib], sa * sb * (1 if (ia + 1) % 3 == ib else -1)


def hat(ax, sg=1):
    return ("+" if sg > 0 else "-") + "\\hat{" + ax + "}"


def pi_tex(k):
    return f"{k}\\pi"


def deg(ph):
    return f"{'+' if ph >= 0 else '-'}{abs(ph)}^\\circ"


def kind(Aa, Bb, psi_deg, sab_s):
    """E = a A cos(theta) + b B cos(theta + psi); sab_s = (a x b . w) * (travel sign). -> lin / rc / lc / re / le."""
    sp = math.sin(math.radians(psi_deg))
    if abs(sp) < 1e-9:
        return "lin"
    right = (-sp) * sab_s > 0
    if abs(Aa - Bb) < 1e-9 and abs(abs(sp) - 1) < 1e-9:
        return "rc" if right else "lc"
    return "re" if right else "le"


def generate(data):
    w = random.choice(AX)
    a, b = random.sample([x for x in AX if x != w], 2)
    s = random.choice([1, -1])
    sab = cross(a, 1, b, 1)[1]                      # a x b = sab * w
    n = random.choice([1, 1.5, 2, 3])
    epsr = n * n
    eta = ETA0 / n
    kA, kB = random.sample([6, 12, 18, 24, 30, 60], 2)       # amplitudes kA*pi, kB*pi V/m
    A, B = kA * math.pi, kB * math.pi
    phi = random.choice([30, 45, 60, 120, 135, 150, -30, -45, -60, -120, -135, -150])
    sg = "-" if s > 0 else "+"
    theta = f"\\omega t {sg} \\beta {w}"
    field_tex = (f"\\mathbf{{E}} = \\hat{{{a}}}\\,{pi_tex(kA)}\\cos({theta}) + \\hat{{{b}}}\\,{pi_tex(kB)}\\sin({theta} {deg(phi)})")
    medium = "free space" if n == 1 else f"a lossless, non-magnetic dielectric with $\\epsilon_r = {fnum(epsr)}$"

    # ---- (a) the phasor
    ex = f"e^{{{sg}j\\beta {w}}}"
    exw = f"e^{{{'+' if s > 0 else '-'}j\\beta {w}}}"
    ph = f"e^{{{'' if phi > 0 else '-'}j{abs(phi)}^\\circ}}"
    phc = f"e^{{{'-' if phi > 0 else ''}j{abs(phi)}^\\circ}}"
    r = math.radians(phi)

    def tex(mid, e):
        return f"$\\big(\\hat{{{a}}}\\,{pi_tex(kA)} {mid}\\big)\\,{e}$"

    def cplx(z):
        return [z.real, z.imag]

    e_p, e_m = complex(math.cos(r), math.sin(r)), complex(math.cos(r), -math.sin(r))
    cands = [
        (tex(f"- j\\,\\hat{{{b}}}\\,{pi_tex(kB)}\\,{ph}", ex), True, -1j * B * e_p, s, "Correct: $\\sin x = \\cos(x-90^\\circ)$ gives the factor $-j$, and the phase adds as $e^{+j\\phi}$."),
        (tex(f"+ j\\,\\hat{{{b}}}\\,{pi_tex(kB)}\\,{ph}", ex), False, 1j * B * e_p, s, "A sine lags a cosine by $90^\\circ$: its phasor carries $-j$, not $+j$."),
        (tex(f"+ \\hat{{{b}}}\\,{pi_tex(kB)}\\,{ph}", ex), False, B * e_p, s, "That would be the phasor of a cosine in the second component. A sine needs the factor $-j$."),
        (tex(f"- j\\,\\hat{{{b}}}\\,{pi_tex(kB)}\\,{phc}", ex), False, -1j * B * e_m, s, "The phase is added to $\\omega t$, so it appears as $e^{+j\\phi}$ with its own sign."),
        (tex(f"- j\\,\\hat{{{b}}}\\,{pi_tex(kB)}\\,{ph}", exw), False, -1j * B * e_p, -s, "The space factor has the wrong sign: this phasor describes a wave travelling the other way."),
    ]
    wrong = cands[1:]
    random.shuffle(wrong)
    ph_choices = [{"text": t, "correct": "true" if ok else "false", "feedback": fb, "cb": cplx(cb), "ksign": ks}
                  for t, ok, cb, ks, fb in [cands[0]] + wrong[:3]]
    random.shuffle(ph_choices)

    # ---- (b) H at the origin at t = 0, (c) power
    Ea0, Eb0 = A, B * math.sin(r)
    h1, s1 = cross(w, s, a, 1)            # u x a_hat -> along b
    h2, s2 = cross(w, s, b, 1)            # u x b_hat -> along a
    Hb, Ha = s1 * Ea0 / eta, s2 * Eb0 / eta
    Sw = s * (A * A + B * B) / (2 * eta)

    # ---- (d) which change makes it circular with a given hand
    target = random.choice(["rc", "lc"])
    tname = {"rc": "right-circular", "lc": "left-circular"}[target]
    opts = [(1, 0), (1, 180), (1, random.choice([90, -90])), (2, 0)]
    circ_choices = []
    for ratio, p2 in opts:
        kd = kind(1.0, float(ratio), p2 - 90, sab * s)
        txt = f"$B = {'' if ratio == 1 else ratio}A,\\ \\phi = {p2}^\\circ$"
        if kd == target:
            fb = "Correct."
        elif kd in ("rc", "lc"):
            fb = "This wave is circular, but with the other handedness. Follow $\\mathbf{E}$ at a fixed point from $\\omega t = 0$ to a quarter period later and compare with the travel direction."
        elif kd == "lin":
            fb = "$\\sin(\\theta \\pm 90^\\circ) = \\pm\\cos\\theta$: the two components are then in phase (or opposite), which is a linear wave."
        else:
            fb = "Unequal amplitudes cannot give a circle. This wave is elliptical."
        circ_choices.append({"text": txt, "correct": "true" if kd == target else "false", "feedback": fb, "ratio": ratio, "phi2": p2})
    random.shuffle(circ_choices)

    # ---- (e) three waves given through H (FA23 #2 format)
    s3 = random.choice([1, -1])
    amps = [2, 3, 4, 5]
    pool = []
    q = random.choice(amps)
    pool.append((q, q, 90)); q = random.choice(amps)
    pool.append((q, q, -90))
    p1, p2_ = random.sample(amps, 2)
    pool.append((p1, p2_, random.choice([0, 180])))
    q = random.choice(amps)
    pool.append(random.choice([(q, q, random.choice([45, 60, -45, 120])), (p2_, p1, random.choice([90, -90]))]))
    q = random.choice(amps)
    pool.append((q, q, random.choice([0, 180])))
    rows = random.sample(pool, 3)
    names = {"lin": "lin", "rc": "rc", "lc": "lc", "re": "ell", "le": "ell"}
    match_items = []
    for Ah, Bh, ph3 in rows:
        ptx = f"{ph3}^\\circ" if ph3 in (0, 180) else deg(ph3)
        match_items.append({"text": f"$A = {Ah},\\ B = {Bh},\\ \\phi = {ptx}$",
                            "key": names[kind(float(Ah), float(Bh), ph3, s3)], "A": Ah, "B": Bh, "phi": ph3})
    th3 = f"\\omega t {'-' if s3 > 0 else '+'} kz"
    hfield = f"\\mathbf{{H}} = \\hat{{x}}\\,A\\cos({th3}) + \\hat{{y}}\\,B\\cos({th3} + \\phi)\\ \\ \\mathrm{{mA/m}}"
    long_name = {"lin": "linear", "rc": "right-circular", "lc": "left-circular", "ell": "elliptical"}
    sol_rows = "; ".join(f"{it['text']}: <b>{long_name[it['key']]}</b>" for it in match_items)

    correct_circ = [c for c in circ_choices if c["correct"] == "true"][0]
    data["params"].update({
        "field_tex": field_tex, "medium": medium, "a": a, "b": b, "w": w, "tname": tname,
        "ph_choices": ph_choices, "circ_choices": circ_choices, "match_items": match_items, "hfield": hfield,
        "s3dir": hat("z", s3),
        # solution strings
        "s_udir": hat(w, s), "s_eta": fnum(eta), "s_etatex": "120\\pi" if n == 1 else f"120\\pi/{fnum(n)}",
        "s_correct_ph": [c for c in ph_choices if c["correct"] == "true"][0]["text"].strip("$"),
        "s_Ea0": fnum(Ea0), "s_Eb0": fnum(Eb0), "s_Ha": fnum(Ha), "s_Hb": fnum(Hb),
        "s_x1": f"({hat(w, s)})\\times\\hat{{{a}}} = {hat(h1, s1)}", "s_x2": f"({hat(w, s)})\\times\\hat{{{b}}} = {hat(h2, s2)}",
        "s_Sw": fnum(Sw), "s_Smag": fnum(abs(Sw)), "s_kA": pi_tex(kA), "s_kB": pi_tex(kB),
        "s_circ": correct_circ["text"].strip("$"),
        "s_turn": f"\\hat{{{a}}} \\to {hat(b, 1 if correct_circ['phi2'] == 0 else -1)}",
        "s_rows": sol_rows,
        # raw data for the independent check
        "s": s, "n": n, "kA": kA, "kB": kB, "phi": phi, "target": target, "s3": s3,
    })
    ca = data["correct_answers"]
    ca["Ha"], ca["Hb"], ca["Sw"] = Ha, Hb, Sw
