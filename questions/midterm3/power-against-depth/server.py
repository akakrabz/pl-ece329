"""Midterm 3 exam-style: power against depth for a wave going down into a good conductor.

Family of FA23 Exam 3 #6 (with the regime items of SP17 #1a, SP18 #1(i), SP19 #1d-e): the phasor is given as
    E = a_hat E0 exp(-a z) exp(-j a z),     alpha = beta = a   (the signature of a good conductor),
in a medium of known sigma (non-magnetic). Then
    delta = 1/a,  lambda = 2 pi / a,  a = sqrt(pi f mu0 sigma)  ->  f = a^2 / (pi mu0 sigma),
    eta = (1 + j) a / sigma  ->  |eta| = sqrt(2) a / sigma, angle 45 deg,
    <S>(z) = z_hat |E0|^2 cos(45 deg) / (2 |eta|) exp(-2 a z) = z_hat sigma |E0|^2 / (4 a) exp(-2 a z),
    power attenuation 20 log10(e) a = 8.686 a dB/m,
    heat per unit area in 0 < z < d:  <S>(0) - <S>(d)   (Poynting's theorem; equals the integral of sigma |E|^2 / 2).
"""
import math
import random

MU0 = 4e-7 * math.pi
EPS0 = 8.854e-12
MEDIA = [
    {"name": "seawater", "desc": "the sea", "sigma": 4.0, "epsr": 81, "a": [0.1, 0.2, 0.4, 0.5, 1, 2], "E0": [1, 2, 5]},
    {"name": "moist ground", "desc": "moist ground", "sigma": 0.01, "epsr": 15, "a": [0.01, 0.02, 0.04, 0.05], "E0": [5, 10, 20]},
    {"name": "brine", "desc": "a deep tank of brine", "sigma": 1.0, "epsr": 80, "a": [0.5, 1, 2], "E0": [1, 2, 5]},
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
    if x == 0:
        return "0"
    if 0.01 <= abs(x) < 1e4:
        return fnum(x, sig)
    e = math.floor(math.log10(abs(x)))
    m = fnum(x / 10 ** e, sig)
    return f"10^{{{e}}}" if m == "1" else f"{m}\\times10^{{{e}}}"


def generate(data):
    med = random.choice(MEDIA)
    sigma, a = med["sigma"], random.choice(med["a"])
    E0 = random.choice(med["E0"])
    ax = random.choice(["x", "y"])
    f = a ** 2 / (math.pi * MU0 * sigma)
    eta_mag = math.sqrt(2) * a / sigma
    S0 = sigma * E0 ** 2 / (4 * a)                       # W/m^2
    m_d = random.choice([1, 2, 3, 4])                    # 2 a d = m_d
    d = m_d / (2 * a)
    Sd = S0 * math.exp(-m_d)
    XdB = random.choice([10, 20, 30, 40])
    dBm = 20 * math.log10(math.e) * a
    zX = XdB / dBm
    Qd = S0 - Sd
    if S0 < 0.5:
        Sfac, Sunit = 1e3, "\\mathrm{mW/m^2}"
    else:
        Sfac, Sunit = 1.0, "\\mathrm{W/m^2}"

    regime = [
        {"key": "good", "correct": "true", "text": "The medium is a good conductor at this frequency: conduction current dominates ($\\sigma/\\omega\\epsilon \\gg 1$), and $\\angle\\eta = 45^\\circ$.",
         "feedback": "Correct."},
        {"key": "lossless", "correct": "false", "text": "The medium is a perfect dielectric: $\\alpha = \\beta$ holds in every lossless medium.",
         "feedback": "A lossless medium has $\\alpha = 0$: the wave does not decay at all."},
        {"key": "lowloss", "correct": "false", "text": "The medium is an imperfect dielectric with a small loss tangent.",
         "feedback": "A low-loss dielectric has $\\alpha \\ll \\beta$: many wavelengths fit in one penetration depth."},
        {"key": "always", "correct": "false", "text": "Nothing: $\\alpha = \\beta$ in every conducting medium, whatever the frequency.",
         "feedback": "In general $\\alpha \\lt \\beta$. They become equal only in the limit $\\sigma/\\omega\\epsilon \\gg 1$."},
    ]
    random.shuffle(regime)

    # (f) one parameter changes; the medium stays a good conductor
    while True:
        mfac = random.choice([2, 3])
        what = random.choice(["f_up", "f_down", "sigma_up"])
        f2 = f * mfac ** 2 if what == "f_up" else f / mfac ** 2 if what == "f_down" else f
        s2 = sigma * mfac ** 2 if what == "sigma_up" else sigma
        if s2 / (2 * math.pi * f2 * med["epsr"] * EPS0) > 100:        # still clearly a good conductor
            break
    change = {"f_up": f"the frequency is raised by a factor of {mfac ** 2}",
              "f_down": f"the frequency is lowered by a factor of {mfac ** 2}",
              "sigma_up": f"the wave enters a different material whose conductivity is {mfac ** 2} times larger (same frequency)"}[what]
    true_ratio = mfac if what == "f_down" else 1.0 / mfac          # delta_new / delta_old
    frac = {2: "\\tfrac12", 3: "\\tfrac13", 4: "\\tfrac14", 9: "\\tfrac19"}
    cands = [(1.0 / mfac, f"${frac[mfac]}\\delta$"), (float(mfac), f"${mfac}\\delta$"),
             (1.0 / mfac ** 2, f"${frac[mfac ** 2]}\\delta$"), (float(mfac ** 2), f"${mfac ** 2}\\delta$"), (1.0, "$\\delta$ (unchanged)")]
    scal = []
    for r, txt in cands:
        ok = abs(r - true_ratio) < 1e-9
        if ok:
            fb = "Correct: $\\delta = 1/\\sqrt{\\pi f\\mu\\sigma}$."
        elif r == 1.0:
            fb = "The skin depth depends on both the frequency and the conductivity: $\\delta = 1/\\sqrt{\\pi f\\mu\\sigma}$."
        elif (r < 1) == (true_ratio < 1):
            fb = "Right direction, but the dependence is a square root: $\\delta \\propto 1/\\sqrt{f\\sigma}$."
        else:
            fb = "Wrong direction: a higher frequency or a higher conductivity makes the skin depth smaller."
        scal.append({"key": r, "text": txt, "correct": "true" if ok else "false", "feedback": fb})
    wrong = [c for c in scal if c["correct"] == "false"]
    random.shuffle(wrong)
    scal = [c for c in scal if c["correct"] == "true"] + wrong[:3]
    random.shuffle(scal)

    a_s = fnum(a)
    data["params"].update({
        "desc": med["desc"], "name": med["name"], "sig_tex": sci(sigma), "epsr": med["epsr"],
        "field_tex": f"\\tilde{{\\mathbf{{E}}}} = \\hat{{{ax}}}\\,{fnum(E0)}\\,e^{{-{a_s}z}}\\,e^{{-j{a_s}z}}",
        "regime": regime, "scal": scal, "change": change,
        "d_s": fnum(d), "XdB": XdB, "S_unit": Sunit,
        "s_a": a_s, "s_delta": fnum(1 / a), "s_lam": fnum(2 * math.pi / a), "s_f": fnum(f / 1e3), "s_eta": sci(eta_mag),
        "s_S0": fnum(S0 * Sfac), "s_Sd": fnum(Sd * Sfac), "s_md": str(m_d), "s_dBm": fnum(dBm), "s_zX": fnum(zX),
        "s_Qd": fnum(Qd * Sfac), "s_E0": fnum(E0),
        "s_ratio": {0.5: "\\tfrac12", 1 / 3: "\\tfrac13", 2.0: "2", 3.0: "3"}[true_ratio],
        "s_why": {"f_up": f"$f$ is {mfac ** 2} times larger", "f_down": f"$f$ is {mfac ** 2} times smaller",
                  "sigma_up": f"$\\sigma$ is {mfac ** 2} times larger"}[what],
        # raw data for the independent check
        "a": a, "sigma": sigma, "E0": E0, "ax": ax, "d": d, "Sfac": Sfac, "what": what, "mfac": mfac,
    })
    ca = data["correct_answers"]
    ca["delta"], ca["lam"], ca["f"], ca["eta_mag"] = 1 / a, 2 * math.pi / a, f / 1e3, eta_mag
    ca["S0"], ca["Sd"] = S0 * Sfac, Sd * Sfac
    ca["dBm"], ca["zX"] = dBm, zX
    ca["Qd"] = Qd * Sfac
