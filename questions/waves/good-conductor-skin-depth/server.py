"""Lecture 23: loss tangent, good-conductor skin depth, attenuation in dB, and the 45-degree lag.

p = sigma/(w eps) with the medium's own eps: good conductor p > 10, imperfect dielectric p < 0.1 (pool keeps a margin of 2).
Good conductor: delta = 1/sqrt(pi f mu0 sigma), |eta| = sqrt(w mu0/sigma), angle 45 deg, lambda = 2 pi delta.
Power density falls as e^{-2 l/delta}: dB = 10 log10(e^{2 l/delta}) = 8.686 l/delta.
Constants: mu0 = 4 pi 1e-7 H/m, eps0 = 8.854e-12 F/m.
"""
import math
import random

MU0 = 4e-7 * math.pi
EPS0 = 8.854e-12
C0 = 1 / math.sqrt(MU0 * EPS0)

# name, sigma [S/m], sigma tex, eps_r, kind (handbook values as on the Lecture 23 page; stainless: 304, about 1.4e6 S/m)
MEDIA = [
    ("copper", 5.8e7, r"5.8\times10^{7}", 1, "metal"),
    ("aluminium", 3.5e7, r"3.5\times10^{7}", 1, "metal"),
    ("seawater", 4.0, "4", 81, "water"),
    ("moist loam soil", 1e-2, "10^{-2}", 10, "other"),
    ("fresh water", 1e-3, "10^{-3}", 80, "water"),
    ("dry earth", 1e-5, "10^{-5}", 5, "other"),
    ("Bakelite", 1e-9, "10^{-9}", 4.9, "other"),
    ("mica", 1e-11, "10^{-11}", 6, "other"),
]
FREQS = [50, 60, 400, 1e3, 5e3, 2e4, 5e4, 1e5, 2.5e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8, 3e8, 5e8, 1e9,
         2.45e9, 5e9, 1e10]
CLS = {"good": "good conductor", "imperfect": "imperfect dielectric", "neither": "neither (in between)"}


def num(x, sig=4):
    s = f"{x:.{sig}g}"
    if "e" in s:
        m, e = s.split("e")
        return f"{m}\\times10^{{{int(e)}}}"
    return s


def f_tex(f):
    for scale, unit in ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz"), (1, "Hz")):
        if f >= scale:
            return f"{f / scale:g}\\ \\mathrm{{{unit}}}"
    return f"{f:g}\\ \\mathrm{{Hz}}"


def pick_unit(x, units):
    """units: list of (scale, tex) from large to small; first with x/scale >= 1 (else the last)."""
    for scale, tex in units:
        if x >= scale:
            return scale, tex
    return units[-1]


LEN_U = [(1.0, r"\mathrm{m}"), (1e-3, r"\mathrm{mm}"), (1e-6, r"\mu\mathrm{m}")]
OHM_U = [(1.0, r"\Omega"), (1e-3, r"\mathrm{m}\Omega"), (1e-6, r"\mu\Omega")]
TIME_U = [(1e-3, r"\mathrm{ms}"), (1e-6, r"\mu\mathrm{s}"), (1e-9, r"\mathrm{ns}"), (1e-12, r"\mathrm{ps}")]


def classify(p):
    if p >= 20:
        return "good"
    if p <= 0.05:
        return "imperfect"
    if 0.2 <= p <= 5:
        return "neither"
    return None


def generate(data):
    # ------------------------------------------------------------------ (a) classification pool
    cands = []
    for mi, (name, sig, stex, er, kind) in enumerate(MEDIA):
        for f in FREQS:
            if (kind == "water" or "soil" in name) and f > 5e8:      # keep wet media below water's relaxation range
                continue
            p = sig / (2 * math.pi * f * er * EPS0)
            k = classify(p)
            if k:
                cands.append((mi, f, p, k))
    while True:
        pool = random.sample(cands, 8)
        mis = [x[0] for x in pool]
        kinds = [x[3] for x in pool]
        n_metal = sum(MEDIA[m][4] == "metal" for m in mis)
        if (all(kinds.count(k) >= 2 for k in CLS) and n_metal <= 1 and max(mis.count(m) for m in mis) <= 2
                and len({(m, f) for m, f, _, _ in pool}) == 8):
            break
    cls_items = []
    for mi, f, p, k in pool:
        name, sig, stex, er, kind = MEDIA[mi]
        text = (f"{name} ($\\sigma = {stex}\\ \\mathrm{{S/m}}$, $\\epsilon_r = {er:g}$) at $f = {f_tex(f)}$")
        p0 = p * er
        trap = ""
        k0 = "good" if p0 > 10 else ("imperfect" if p0 < 0.1 else "neither")
        if k0 != k:
            trap = f" (with $\\epsilon_0$ in place of the medium's $\\epsilon$ one would get ${num(p0, 3)}$ and the wrong verdict)"
        cls_items.append({"key": k, "text": text, "p_str": num(p, 3), "cls_name": CLS[k], "trap": trap})

    # ------------------------------------------------------------------ (b)-(e) one good conductor
    kind = random.choice(["metal", "metal", "water"])
    if kind == "metal":
        mat, sig, stex, er = random.choice([("copper", 5.8e7, r"5.8\times10^{7}", 1), ("aluminium", 3.5e7, r"3.5\times10^{7}", 1),
                                            ("type-304 stainless steel", 1.4e6, r"1.4\times10^{6}", 1)])
        f = random.choice([50, 400, 2e3, 2e4, 2.5e5, 5e6, 5e7, 3e8, 2.45e9, 5.8e9])
    else:
        mat, sig, stex, er = "seawater", 4.0, "4", 81
        f = random.choice([40, 82, 200, 3e3, 1.5e4, 4e4, 2e5, 5e5])
    om = 2 * math.pi * f
    p = sig / (om * er * EPS0)
    delta = 1 / math.sqrt(math.pi * f * MU0 * sig)
    eta = math.sqrt(om * MU0 / sig)
    lam = 2 * math.pi * delta
    vp = om * delta
    d_sc, d_u = pick_unit(delta, LEN_U)
    e_sc, e_u = pick_unit(eta, OHM_U)
    while True:
        m = random.choice([0.8, 1, 1.2, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6])
        l_sc0, l_u0 = pick_unit(m * delta, LEN_U)
        ell_disp = float(f"{m * delta / l_sc0:.2g}")
        ell = ell_disp * l_sc0
        if 0.7 <= ell / delta <= 6.5:
            break
    nd = ell / delta
    dB = 10 * math.log10(math.exp(2 * nd))
    pct = 100 * math.exp(-2 * nd)
    # pct is exponentially sensitive: 3-s.f. delta shifts it by ~2l/delta * 0.5 %
    pct_rtol = max(0.02, math.ceil(2 * nd * 0.0075 * 1000) / 1000)
    T = 1 / f
    t_sc, t_u = pick_unit(T / 8, TIME_U)
    T8 = num(T / 8 / t_sc, 3) + r"\ " + t_u
    T4 = num(T / 4 / t_sc, 3) + r"\ " + t_u
    l_sc, l_u = pick_unit(lam, LEN_U)
    lam_s = num(lam / l_sc, 3) + r"\ " + l_u
    dl_s = num(delta / d_sc, 3) + r"\ " + d_u
    if kind == "water":
        lam_lossless = C0 / (f * math.sqrt(er))
        ll_sc, ll_u = pick_unit(lam_lossless, [(1e3, r"\mathrm{km}")] + LEN_U)
        free_txt = (f"$\\lambda = c/(f\\sqrt{{\\epsilon_r}}) = {num(lam_lossless / ll_sc, 3)}\\ {ll_u}$, "
                    r"as in the same water without its conductivity")
        free_fb = (r"That is the lossless value $2\pi/\omega\sqrt{\mu\epsilon}$. Here conduction dominates: $\beta = 1/\delta$ "
                   r"is far larger than $\omega\sqrt{\mu\epsilon}$, and the wavelength is much shorter.")
    else:
        lam_lossless = C0 / f
        ll_sc, ll_u = pick_unit(lam_lossless, [(1e3, r"\mathrm{km}")] + LEN_U)
        free_txt = (f"$\\lambda = c/f = {num(lam_lossless / ll_sc, 3)}\\ {ll_u}$, the free-space wavelength, "
                    r"because the frequency does not change")
        free_fb = (r"The frequency is the same, but the phase constant is not: inside, $\beta = 1/\delta$, so "
                   r"$\lambda = 2\pi\delta$, far shorter than in free space.")
    lag_choices = [
        {"id": "ok", "correct": "true",
         "text": f"lags $\\mathbf{{E}}$ by $45^\\circ$: the magnitude of $\\mathbf{{H}}$ peaks $T/8 = {T8}$ after that of $\\mathbf{{E}}$",
         "feedback": r"Right: $\eta\approx\sqrt{\omega\mu/\sigma}\,e^{j45^\circ}$, and dividing by $\eta$ subtracts $45^\circ$, an eighth of a period."},
        {"id": "lead", "correct": "false",
         "text": f"leads $\\mathbf{{E}}$ by $45^\\circ$: the magnitude of $\\mathbf{{H}}$ peaks $T/8 = {T8}$ before that of $\\mathbf{{E}}$",
         "feedback": r"The angle is right but the sign is not: $\tilde{\mathbf{H}} = \hat{u}\times\tilde{\mathbf{E}}/\eta$ and $\angle\eta = +45^\circ$, so $\mathbf{H}$ lags."},
        {"id": "inphase", "correct": "false",
         "text": r"is in phase with $\mathbf{E}$, as in a lossless medium",
         "feedback": r"Only a real $\eta$ keeps them in phase. A good conductor has $\angle\eta\approx45^\circ$."},
        {"id": "quarter", "correct": "false",
         "text": f"lags $\\mathbf{{E}}$ by $90^\\circ$: the magnitude of $\\mathbf{{H}}$ peaks $T/4 = {T4}$ after that of $\\mathbf{{E}}$",
         "feedback": (r"$\angle\eta = \tfrac12\tan^{-1}(\sigma/\omega\epsilon)$ approaches $45^\circ$, never $90^\circ$; "
                      r"a $90^\circ$ lag would mean no time-average power flow at all.")},
    ]
    random.shuffle(lag_choices)
    lam_choices = [
        {"id": "ok", "correct": "true",
         "text": f"$\\lambda = 2\\pi\\delta = {lam_s}$, and over one wavelength the field falls to $e^{{-2\\pi}}\\approx0.19\\ \\%$ of its value",
         "feedback": r"Right: $\alpha = \beta = 1/\delta$, so $\lambda = 2\pi/\beta = 2\pi\delta$ and $e^{-\alpha\lambda} = e^{-2\pi}$."},
        {"id": "free", "correct": "false", "text": free_txt, "feedback": free_fb},
        {"id": "delta", "correct": "false",
         "text": f"$\\lambda = \\delta = {dl_s}$, because $\\alpha = \\beta$",
         "feedback": r"$\alpha = \beta$ means $\beta = 1/\delta$, so $\lambda = 2\pi/\beta = 2\pi\delta$, not $\delta$."},
        {"id": "weak", "correct": "false",
         "text": f"$\\lambda = 2\\pi\\delta = {lam_s}$, and over one wavelength the field falls only by a factor $e$, to $37\\ \\%$",
         "feedback": r"The field falls by $e$ over one skin depth, which is $\lambda/2\pi$; over a whole wavelength it falls by $e^{2\pi}\approx535$."},
    ]
    random.shuffle(lam_choices)

    where = "into the " + mat if kind == "metal" else "down into the seawater"
    data["params"].update({
        "cls_items": cls_items, "mat": mat, "kind": kind, "sig_tex": stex, "er": f"{er:g}", "f_tex": f_tex(f),
        "f_hz": f, "ell_str": f"{ell_disp:g}", "ell_u": l_u0, "d_u": d_u, "e_u": e_u, "where": where,
        "lag_choices": lag_choices, "lam_choices": lam_choices,
        "om_str": num(om), "p_str": num(p, 3), "delta_str": num(delta / d_sc), "eta_str": num(eta / e_sc),
        "eta_re_str": num(1 / (sig * delta) / e_sc), "nd_str": num(nd), "dB_str": num(dB), "pct_str": num(pct), "pct_rtol": f"{pct_rtol:.3f}",
        "dBpm_str": num(8.686 / delta), "lam_str": lam_s, "T8_str": T8, "vp_str": num(vp),
        "pis_str": num(math.pi * f * MU0 * sig), "omeps_str": num(om * er * EPS0),
    })
    data["correct_answers"].update({
        "lt": float(p), "delta": float(delta / d_sc), "eta": float(eta / e_sc), "dB": float(dB), "pct": float(pct),
    })
