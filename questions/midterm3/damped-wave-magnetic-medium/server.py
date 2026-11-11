"""Midterm 3 exam-style: decoding a damped plane wave in a conducting, possibly magnetic medium.

Family of SP19 Exam 3 #1 and SP17 Exam 3 #3 (re-parameterized): the field is given with "nice" alpha and beta,
    E = a_hat E0 exp(-/+ alpha w) cos(omega t -/+ beta w + phi)      (travel along +/- w),
and everything else follows from gamma = alpha + j beta and the two relations printed on those papers,
    gamma * eta = j omega mu,        gamma / eta = sigma + j omega eps.
  |gamma|, angle(gamma)  ->  |eta| = omega mu / |gamma|,  tau = angle(eta) = 90 deg - angle(gamma)
  H phasor = (u_hat x a_hat) (E0/|eta|) exp(j(phi - tau)) exp(-/+ gamma w)        (H lags E by tau)
  <S> = u_hat |E0|^2 exp(-2 alpha d) cos(tau) / (2|eta|)   a distance d along the travel
  loss tangent sigma/(omega eps) = 2 alpha beta / (beta^2 - alpha^2); it scales as 1/omega for fixed sigma, eps.
(alpha, beta) come from 30-60-90, 3-4-5 and 5-12-13 triangles so that every step can be done by hand.
"""
import math
import random

MU0 = 4e-7 * math.pi
EPS0 = 1e-9 / (36 * math.pi)
AX = ["x", "y", "z"]
FAMS = [
    {"a": 1.0, "b": math.sqrt(3.0), "mag": 2.0, "sq3": True, "given": ""},
    {"a": 3.0, "b": 4.0, "mag": 5.0, "sq3": False, "given": "\\tan^{-1}(4/3) = 53.13^\\circ"},
    {"a": 5.0, "b": 12.0, "mag": 13.0, "sq3": False, "given": "\\tan^{-1}(12/5) = 67.38^\\circ"},
]


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


def sci(x, sig=3):
    """LaTeX number: plain between 0.01 and 10000, otherwise m x 10^e."""
    if x == 0:
        return "0"
    if 0.01 <= abs(x) < 1e4:
        return fnum(x, sig)
    e = math.floor(math.log10(abs(x)))
    return f"{fnum(x / 10 ** e, sig)}\\times10^{{{e}}}"


def cross(a, sa, b, sb):
    """(sa a_hat) x (sb b_hat) for two different axes -> (axis, sign)."""
    ia, ib = AX.index(a), AX.index(b)
    c = AX[3 - ia - ib]
    return c, sa * sb * (1 if (ia + 1) % 3 == ib else -1)


def hat(ax, sg=1):
    return ("+" if sg > 0 else "-") + "\\hat{" + ax + "}"


def deg_tex(ph):
    return "" if ph == 0 else f"{'+' if ph > 0 else '-'}{abs(ph)}^\\circ"


def generate(data):
    while True:
        fam = random.choice(FAMS)
        k = random.choice([0.1, 0.2, 0.5, 1, 2])
        alpha, beta, gmag = fam["a"] * k, fam["b"] * k, fam["mag"] * k
        m, n = random.choice([2, 4, 8]), random.choice([6, 7, 8])
        omega = m * math.pi * 10 ** n
        mur = random.choice([1, 1, 2, 4])
        mu = mur * MU0
        eps = (beta ** 2 - alpha ** 2) / (omega ** 2 * mu)
        if 1.0 <= eps / EPS0 <= 90:
            break
    sigma = 2 * alpha * beta / (omega * mu)
    gang = math.degrees(math.atan2(beta, alpha))
    eta_mag, tau = omega * mu / gmag, 90.0 - gang
    E0 = random.choice([2, 3, 5, 10, 20, 50])
    phiE = random.choice([0, 30, 45, 60, 90, 120, -30, -45])
    w, a = random.sample(AX, 2)
    s = random.choice([1, -1])
    hax, hsg = cross(w, s, a, 1)
    time_form = random.random() < 0.5

    # ---- how the field is printed
    a_tex = fnum(alpha)
    b_tex = (("" if k == 1 else fnum(k)) + "\\sqrt{3}") if fam["sq3"] else fnum(beta)
    om_tex = f"{m}\\pi\\times10^{{{n}}}"
    if time_form:
        field_tex = (f"\\mathbf{{E}} = {E0}\\,e^{{{'-' if s > 0 else ''}{a_tex}{w}}}"
                     f"\\cos\\!\\big({om_tex}\\,t {'-' if s > 0 else '+'} {b_tex}\\,{w}{deg_tex(phiE)}\\big)\\,\\hat{{{a}}}")
        intro = "The electric field of a uniform plane wave is"
    else:
        ph = "" if phiE == 0 else f"\\,e^{{{'' if phiE > 0 else '-'}j{abs(phiE)}^\\circ}}"
        field_tex = (f"\\tilde{{\\mathbf{{E}}}} = {E0}\\,e^{{{'-' if s > 0 else ''}({a_tex}+j{b_tex}){w}}}{ph}\\,\\hat{{{a}}}")
        intro = "The electric-field phasor of a uniform plane wave is"
    mu_tex = "\\mu = \\mu_0" if mur == 1 else f"\\mu = {mur}\\mu_0"

    # ---- (a) direction of travel
    dir_choices = []
    for ax in AX:
        for sg in (1, -1):
            if ax == w and sg == s:
                fb = "Correct: the amplitude decays and the phase lags further in this direction."
            elif ax == w:
                fb = "The amplitude decays in the direction of travel, and the phase term has the opposite sign of the time term along it. Here both point the other way."
            elif ax == a:
                fb = "That is the axis of E. A plane wave travels at right angles to its fields, along the coordinate that appears in the exponent."
            else:
                fb = "The field does not depend on this coordinate. The wave travels along the coordinate that appears in the exponent."
            dir_choices.append({"text": f"${hat(ax, sg)}$", "correct": "true" if (ax == w and sg == s) else "false",
                                "feedback": fb, "key": ("+" if sg > 0 else "-") + ax})

    # ---- (d) the H phasor
    H0 = E0 / eta_mag                      # A/m
    phiH = phiE - tau
    h_choices = []
    for ax, sg in ((hax, hsg), (hax, -hsg), (a, 1), (a, -1)):
        ok = (ax == hax and sg == hsg)
        fb = ("Correct: with this direction E x H points along the travel." if ok else
              "With this direction E x H points against the travel. Use H = u x E / eta with u the travel direction." if ax == hax else
              "H is perpendicular to E (and to the travel direction).")
        h_choices.append({"text": f"${hat(ax, sg)}$", "correct": "true" if ok else "false", "feedback": fb,
                          "key": ("+" if sg > 0 else "-") + ax})
    random.shuffle(h_choices)

    # ---- (e) average power density a distance d further along the travel
    d = random.choice([x for x in (0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 4) if 0.2 <= alpha * x <= 1.6])
    w1 = s * d
    S0 = E0 ** 2 * math.cos(math.radians(tau)) / (2 * eta_mag)        # W/m^2 at w = 0
    S1 = S0 * math.exp(-2 * alpha * d)

    # ---- (f) loss tangent, and the same material at another frequency
    tanl = 2 * alpha * beta / (beta ** 2 - alpha ** 2)
    want = random.choice(["good", "diel", "mid"])
    opts = [(1000, "raised by a factor of 1000"), (100, "raised by a factor of 100"), (10, "raised by a factor of 10"),
            (4, "raised by a factor of 4"), (0.25, "lowered by a factor of 4"), (0.1, "lowered by a factor of 10"),
            (0.01, "lowered by a factor of 100"), (0.001, "lowered by a factor of 1000")]
    random.shuffle(opts)

    def klass(t):
        return "good" if t > 10 else "diel" if t < 0.1 else "mid"

    def clear(t):                      # stay away from the two thresholds
        return not (0.05 <= t <= 0.2 or 5 <= t <= 20)

    pick = [o for o in opts if clear(tanl / o[0]) and klass(tanl / o[0]) == want] or [o for o in opts if clear(tanl / o[0])]
    K, K_txt = pick[0]
    tanl2 = tanl / K
    cls = klass(tanl2)
    cls_choices = [
        {"key": "good", "text": "a good conductor", "feedback": "A good conductor needs $\\sigma/\\omega\\epsilon \\gt 10$: conduction current dominates."},
        {"key": "diel", "text": "an imperfect (low-loss) dielectric", "feedback": "An imperfect dielectric needs $\\sigma/\\omega\\epsilon \\lt 0.1$: displacement current dominates."},
        {"key": "mid", "text": "neither: in between the two", "feedback": "Compare the new loss tangent with 0.1 and with 10."},
        {"key": "perfect", "text": "a perfect dielectric (lossless)", "feedback": "The conductivity is not zero, so the medium is not lossless at any frequency."},
    ]
    for c in cls_choices:
        c["correct"] = "true" if c["key"] == cls else "false"
        if c["key"] == cls:
            c["feedback"] = "Correct."
    random.shuffle(cls_choices)

    up = "raised" in K_txt
    data["params"].update({
        "intro": intro, "field_tex": field_tex, "mu_tex": mu_tex, "om_tex": om_tex, "w": w, "a": a,
        "time_form": time_form, "given": fam["given"], "has_given": bool(fam["given"]),
        "dir_choices": dir_choices, "h_choices": h_choices, "cls_choices": cls_choices,
        "exp_tex": f"e^{{{'-' if s > 0 else '+'}\\gamma {w}}}",
        "d_s": fnum(d), "w1_s": fnum(w1), "K_txt": K_txt,
        # ---- solution strings
        "s_alpha": a_tex, "s_beta": b_tex, "s_beta_num": fnum(beta), "s_gmag": fnum(gmag), "s_gang": fnum(gang, 4),
        "s_lam": fnum(2 * math.pi / beta), "s_vp": sci(omega / beta), "s_vp8": fnum(omega / beta / 1e8), "s_delta": fnum(1 / alpha),
        "s_omu": fnum(omega * mu), "s_eta": fnum(eta_mag), "s_tau": fnum(tau, 4),
        "s_H0": fnum(H0 * 1e3), "s_phiH": fnum(phiH, 4), "s_phiE": str(phiE),
        "s_hdir": hat(hax, hsg), "s_udir": hat(w, s), "s_cross": f"({hat(w, s)})\\times\\hat{{{a}}} = {hat(hax, hsg)}",
        "s_S0": fnum(S0 * 1e3), "s_S1": fnum(S1 * 1e3), "s_2ad": fnum(2 * alpha * d), "s_sgn": "" if s > 0 else "-",
        "s_tanl": fnum(tanl), "s_tanl2": sci(tanl2), "s_Kop": ("\\div" if up else "\\times") + fnum(K if up else 1 / K),
        "s_sigma": sci(sigma), "s_epsr": fnum(eps / EPS0, 3),
        "s_cls": {"good": "a good conductor", "diel": "an imperfect dielectric", "mid": "in between"}[cls],
        # ---- raw data for the independent check
        "alpha": alpha, "beta": beta, "omega": omega, "mur": mur, "E0": E0, "phiE": phiE, "s": s,
        "h_key": ("+" if hsg > 0 else "-") + hax, "w1": w1, "K": K, "cls": cls,
    })
    ca = data["correct_answers"]
    ca["alpha"], ca["beta"] = alpha, beta
    ca["lam"], ca["vp"], ca["delta"] = 2 * math.pi / beta, omega / beta / 1e8, 1 / alpha
    ca["eta_mag"], ca["tau"] = eta_mag, tau
    ca["H0"], ca["phiH"] = H0 * 1e3, phiH
    ca["Sw"] = s * S1 * 1e3
    ca["tanl"], ca["tanl2"] = tanl, tanl2
