"""Lecture 21 concept question: phasors (cosine reference, peak values, a(t) = Re{A e^{j w t}}), c = 3e8, eta0 = 120 pi.

(a) matching: five fields s*A0 f(w t -/+ beta u + phi) e_hat (f = cos or sin; some written as f(beta u - w t + phi))
    -> phasors A0 e^{j theta} e^{-/+ j beta u} e_hat:  cos -> 1, sin -> -j, minus sign -> e^{j pi},
    phase phi -> e^{j phi}; cos(beta u - w t + phi) = cos(w t - beta u - phi), sin(beta u - w t + phi) = -sin(w t - beta u - phi);
    travel toward +u <-> e^{-j beta u}.  Three extra options are the usual slips (sine -> +j, wrong exponent sign, dropped
    phase or sign, unflipped phase of a reversed argument), never equal to a correct phasor.
(b) E~ = (a e1 + j b e2) e^{-j d beta u} in a non-magnetic dielectric eps_r = n^2: u_hat = d u_hat,
    H~ = (1/eta) u_hat x E~, eta = 120 pi/n;  <S> = |E~0|^2/(2 eta) u_hat = (a^2 + b^2)/(2 eta) along d u_hat.
(c) template E~ = E0 e3 e^{j(a p + b q)} in vacuum: (a, b) = -beta u_hat, beta = sqrt(a^2 + b^2), lambda = 2 pi/beta, f = c/lambda.
(d) collisionless plasma: beta = (w/c) sqrt(1 - fp^2/f^2), v_p = c/sqrt(1 - fp^2/f^2) for f > fp; no travelling wave below fp.
"""
import cmath
import math
import random
from fractions import Fraction

AXES = ["x", "y", "z"]
IDX = {"x": 0, "y": 1, "z": 2}
ETA0 = 120 * math.pi
C = 3e8


def num(x, sig=4):
    s = f"{x:.{sig}g}"
    if "e" in s:
        m, e = s.split("e")
        return f"{m}\\times10^{{{int(e)}}}"
    return s


def uhat(ax, sg):
    return ("+" if sg > 0 else "-") + "\\hat{" + ax + "}"


def cross_axes(a, sa, b, sb):
    ia, ib = IDX[a], IDX[b]
    ic = 3 - ia - ib
    sgn = 1 if (ib - ia) % 3 == 1 else -1
    return AXES[ic], sa * sb * sgn


def pi_tex(fr):
    """|fr|*pi as TeX, fr a positive Fraction: 1 -> \\pi, 1/3 -> \\pi/3, 2/3 -> 2\\pi/3."""
    nu, de = fr.numerator, fr.denominator
    s = ("" if nu == 1 else str(nu)) + r"\pi"
    return s if de == 1 else s + "/" + str(de)


def phase_term(fr):
    """' + \\pi/3' / ' - \\pi/4' / '' for a phase fr*pi."""
    if fr == 0:
        return ""
    return (" + " if fr > 0 else " - ") + pi_tex(abs(fr))


def norm_frac(fr):
    """reduce a phase (in units of pi) to (-1, 1]."""
    fr = Fraction(fr) % 2
    return fr - 2 if fr > 1 else fr


def coef_tex(A0, th):
    """A0 e^{j th pi} in the course's natural form."""
    if th == 0:
        return f"{A0}"
    if th == 1:
        return f"-{A0}"
    if th == Fraction(1, 2):
        return f"{A0}j"
    if th == Fraction(-1, 2):
        return f"-{A0}j"
    return f"{A0}\\,e^{{{'-' if th < 0 else ''}j{pi_tex(abs(th))}}}"


def kpi(k):
    return r"\pi" if k == 1 else f"{k:g}\\pi"


PHASES = [Fraction(n, d) for n, d in ((1, 6), (1, 4), (1, 3), (1, 2), (2, 3), (3, 4), (5, 6))]


def generate(data):
    # ================================================================ (a) matching
    u = random.choice(AXES)
    e = random.choice([ax for ax in AXES if ax != u])
    A0 = random.choice([2, 3, 4, 5, 6, 8, 10, 12])

    def item(sg, fn, rev, s, ph):
        """theta (units of pi) and travel sign of the phasor, plus the TeX of the time function."""
        if not rev:
            th = ph + (Fraction(-1, 2) if fn == "sin" else 0) + (1 if sg < 0 else 0)
            arg = f"\\omega t {'-' if s > 0 else '+'} \\beta {u}{phase_term(ph)}"
        else:                                    # f(beta u - w t + ph), s = +1
            th = -ph + (Fraction(1, 2) if fn == "sin" else 0) + (1 if sg < 0 else 0)
            arg = f"\\beta {u} - \\omega t{phase_term(ph)}"
        tex = f"{'-' if sg < 0 else ''}{A0}\\{fn}({arg})\\,\\hat{{{e}}}"
        return norm_frac(th), s, tex

    def slips(sg, fn, rev, s, ph, th):
        out = [(th, -s)]                                                    # exponent sign
        if fn == "sin":
            out.append((norm_frac(th + 1), s))                              # sine -> +j
            out.append((norm_frac(th + Fraction(1, 2)), s))                 # sine treated as cosine
        if sg < 0:
            out.append((norm_frac(th + 1), s))                              # dropped minus sign
        if ph != 0:
            out.append((norm_frac(th - (ph if not rev else -ph)), s))      # dropped phase
        if rev:
            alt = ph + (Fraction(-1, 2) if fn == "sin" else 0) + (1 if sg < 0 else 0)
            out.append((norm_frac(alt), s))                                 # read as f(w t - beta u + ph)
        return out

    while True:
        plan = ["sin", "neg", "back", "rev", "plain"]
        specs = []
        for kind in plan:
            ph = random.choice([Fraction(0)] + [sgn * x for x in PHASES for sgn in (1, -1)])
            if kind == "sin":
                spec = (1, "sin", False, random.choice([1, -1]), ph)
            elif kind == "neg":
                ph = ph or Fraction(1, 3)
                spec = (-1, random.choice(["cos", "sin"]), False, random.choice([1, -1]), ph)
            elif kind == "back":
                spec = (1, random.choice(["cos", "sin"]), False, -1, ph)
            elif kind == "rev":
                ph = ph or Fraction(-1, 4)
                spec = (1, random.choice(["cos", "sin"]), True, 1, ph)
            else:
                ph = ph or Fraction(2, 3)
                spec = (1, "cos", False, random.choice([1, -1]), ph)
            specs.append(spec)
        items = [item(*sp) for sp in specs]
        keys = [(th, s) for th, s, _ in items]
        if len(set(keys)) == 5 and len({t for _, _, t in items}) == 5:
            break
    cand = []
    for sp, (th, s, _) in zip(specs, items):
        for k in slips(*sp, th):
            if k not in keys and k not in cand:
                cand.append(k)
    random.shuffle(cand)
    extra = cand[:3]
    while len(extra) < 3:
        k = (norm_frac(Fraction(random.randint(-5, 6), 6)), random.choice([1, -1]))
        if k not in keys and k not in extra:
            extra.append(k)

    def opt_tex(th, s):
        return f"{coef_tex(A0, th)}\\,e^{{{'-' if s > 0 else '+'}j\\beta {u}}}\\,\\hat{{{e}}}"
    opts = [{"k": k, "text": "$" + opt_tex(*k) + "$"} for k in keys + extra]
    random.shuffle(opts)
    for i, o in enumerate(opts):
        o["name"] = f"p{i + 1}"
    name_of = {o["k"]: o["name"] for o in opts}
    conv_items = []
    for sp, (th, s, tex) in zip(specs, items):
        sg, fn, rev, ss, ph = sp
        steps = []
        if rev:
            if fn == "cos":
                steps.append(f"cosine is even, $\\cos(\\beta {u} - \\omega t{phase_term(ph)}) = \\cos(\\omega t - \\beta {u}{phase_term(-ph)})$")
            else:
                steps.append(f"sine is odd, $\\sin(\\beta {u} - \\omega t{phase_term(ph)}) = -\\sin(\\omega t - \\beta {u}{phase_term(-ph)})$")
        terms = []
        ph_eff = -ph if rev else ph
        if ph_eff != 0:
            terms.append(f"{'-' if ph_eff < 0 else ''}{pi_tex(abs(ph_eff))}\\ (\\text{{phase}})")
        if fn == "sin":
            terms.append(r"-\pi/2\ (\text{sine} \to -j)")
        if (sg < 0) != (rev and fn == "sin"):
            terms.append(r"\pi\ (\text{minus sign})")
        elif sg < 0 and rev and fn == "sin":
            terms.append(r"\pi + \pi\ (\text{two minus signs})")
        if terms:
            joined = " ".join(("+ " + t) if (i and not t.startswith("-")) else (("- " + t[1:]) if i else t) for i, t in enumerate(terms))
            th_tex = "0" if th == 0 else (("-" if th < 0 else "") + pi_tex(abs(th)))
            steps.append(f"angle ${joined} = {th_tex}$ (mod $2\\pi$)")
        else:
            steps.append("angle $0$")
        steps.append(f"travel toward ${'+' if ss > 0 else '-'}{u}$ gives $e^{{{'-' if ss > 0 else '+'}j\\beta {u}}}$")
        conv_items.append({"text": "$" + tex + "$", "key": name_of[(th, s)], "ans": "$" + opt_tex(th, s) + "$",
                           "why": "; ".join(steps) + "."})
    random.shuffle(conv_items)
    conv_opts = [{"name": o["name"], "text": o["text"]} for o in opts]
    extra_tex = ", ".join("$" + opt_tex(*k) + "$" for k in extra)

    # ================================================================ (b) partner field and power
    ub = random.choice(AXES)
    db = random.choice([1, -1])
    trans = [ax for ax in AXES if ax != ub]
    random.shuffle(trans)
    e1, e2 = trans                                      # e1 real part, e2 imaginary part
    a_b = random.choice([3, 4, 5, 6, 8, 10, 12]) * random.choice([1, -1])
    b_b = random.choice([3, 4, 5, 6, 8, 10, 12]) * random.choice([1, -1])
    while abs(b_b) == abs(a_b):
        b_b = random.choice([3, 4, 5, 6, 8, 10, 12]) * random.choice([1, -1])
    nb = random.choice([1.5, 2.0, 2.5, 3.0, 4.0])
    kb = random.choice([1, 2, 3, 4, 5, 6])
    fb = 150 * kb / nb                                  # MHz
    eta_b = ETA0 / nb
    # H~0 = (1/eta) u x (a e1 + j b e2)
    h1_ax, h1_sg = cross_axes(ub, db, e1, 1)            # u x e1
    h2_ax, h2_sg = cross_axes(ub, db, e2, 1)            # u x e2  (h1_ax = e2, h2_ax = e1)
    Hc = {ax: 0j for ax in AXES}
    Hc[h1_ax] += h1_sg * a_b / eta_b
    Hc[h2_ax] += h2_sg * 1j * b_b / eta_b
    Su_b = db * (a_b ** 2 + b_b ** 2) / (2 * eta_b)     # W/m^2 along +ub

    def term(cf, jj, ax, first):
        sgn = "-" if cf < 0 else ("" if first else "+")
        return f"{sgn}{'' if first else ' '}{abs(cf)}{'j' if jj else ''}\\,\\hat{{{ax}}}"
    tms = [(a_b, False, e1), (b_b, True, e2)]
    random.shuffle(tms)
    Eb_tex = ("\\tilde{\\mathbf{E}} = \\big(" + term(*tms[0], True) + " " + term(*tms[1], False)
              + f"\\big)\\,e^{{{'-' if db > 0 else '+'}j\\,{kpi(kb)} {ub}}}\\ \\ \\mathrm{{V/m}}")
    ans_axes = sorted(trans, key=lambda ax: IDX[ax])
    Hpart = {}
    for ax in ans_axes:
        Hpart[ax] = (Hc[ax].real * 1e3, Hc[ax].imag * 1e3)      # mA/m

    def ctex(z):
        re_, im_ = z.real * 1e3, z.imag * 1e3
        if abs(im_) < 1e-12:
            return num(re_)
        return ("-" if im_ < 0 else "") + num(abs(im_)) + "j"
    H_tex = ("\\tilde{\\mathbf{H}} = \\big(" + " + ".join(f"({ctex(Hc[ax])})\\,\\hat{{{ax}}}" for ax in ans_axes)
             + f"\\big)\\,e^{{{'-' if db > 0 else '+'}j\\,{kpi(kb)} {ub}}}\\ \\ \\mathrm{{mA/m}}")

    # ================================================================ (c) template direction
    plane = random.choice([("x", "y"), ("y", "z"), ("x", "z")])
    pc, qc = plane
    e3 = [ax for ax in AXES if ax not in plane][0]
    m1, m2, m3 = random.choice([(3, 4, 5), (4, 3, 5), (5, 12, 13), (12, 5, 13), (8, 15, 17), (15, 8, 17)])
    sc = random.choice([0.5, 1, 2])
    s1, s2 = random.choice([1, -1]), random.choice([1, -1])
    a_c, b_c = s1 * m1 * sc, s2 * m2 * sc                # in units of pi rad/m
    beta_c = m3 * sc                                      # in units of pi
    E0c = random.choice([2, 4, 5, 6, 8, 10])

    def lin(cp, cq):
        t1 = f"{'-' if cp < 0 else ''}{kpi(abs(cp))}\\,{pc}"
        t2 = f" {'-' if cq < 0 else '+'} {kpi(abs(cq))}\\,{qc}"
        return t1 + t2
    tmpl_tex = f"\\tilde{{\\mathbf{{E}}}} = {E0c}\\,\\hat{{{e3}}}\\,e^{{j({lin(a_c, b_c)})}}\\ \\ \\mathrm{{V/m}}"

    def vec_tex(cp, cq):
        """(cp p + cq q)/m3 with integer cp, cq (already scaled to the triple)."""
        t1 = f"{'-' if cp < 0 else ''}{abs(cp)}\\,\\hat{{{pc}}}"
        t2 = f" {'-' if cq < 0 else '+'} {abs(cq)}\\,\\hat{{{qc}}}"
        return f"$({t1}{t2})/{m3}$"
    ip, iq = s1 * m1, s2 * m2
    dir_choices = [
        {"id": "ok", "text": vec_tex(-ip, -iq), "correct": "true",
         "feedback": f"Right: with the time factor, the phase is $\\omega t + ({lin(a_c, b_c)})$; it stays constant when the position moves along $-(a, b)$, so $\\hat{{u}} = -(a\\,\\hat{{{pc}}} + b\\,\\hat{{{qc}}})/\\beta$."},
        {"id": "same", "text": vec_tex(ip, iq), "correct": "false",
         "feedback": "That is the direction of $(a, b)$ itself. The template has a plus sign in the exponent, $e^{+j(ax+by)}$, and a wave travels toward where the phase lags, so $(a, b) = -\\beta\\hat{u}$: reverse it."},
        {"id": "half1", "text": vec_tex(ip, -iq), "correct": "false",
         "feedback": "Both components of $(a, b)$ flip sign, not just one: $\\hat{u} = -(a, b)/\\beta$."},
        {"id": "swap", "text": vec_tex(-iq, -ip), "correct": "false",
         "feedback": f"The components are swapped: the coefficient of ${pc}$ belongs to $\\hat{{{pc}}}$ and that of ${qc}$ to $\\hat{{{qc}}}$."},
        {"id": "pol", "text": f"$+\\hat{{{e3}}}$, the direction of $\\tilde{{\\mathbf{{E}}}}$", "correct": "false",
         "feedback": f"$\\hat{{{e3}}}$ is the polarization. A plane wave travels perpendicular to its field; the direction is in the exponent, which has no ${e3}$ in it."},
    ]
    random.shuffle(dir_choices)
    lam_c = 2 / beta_c                                    # m
    f_c = C / lam_c / 1e9                                 # GHz

    # ================================================================ (d) plasma
    while True:
        fp = random.choice([2.5, 3, 4, 4.5, 5, 6, 7.5, 8, 9, 10, 12])        # MHz
        r_num, r_den = random.choice([(5, 3), (5, 4), (13, 5), (13, 12), (17, 8), (17, 15), (2, 1), (3, 1)])
        f1 = fp * r_num / r_den                           # MHz
        if abs(f1 * 1000 - round(f1 * 1000)) < 1e-6 and (fp, f1) != (6, 10):
            break
    root = math.sqrt(1 - (r_den / r_num) ** 2)
    beta_p = 2 * math.pi * f1 * 1e6 / C * root            # rad/m
    vp8 = 3.0 / root                                      # 1e8 m/s
    below = random.random() < 0.7
    r2 = random.choice([0.5, 0.6, 0.75, 0.8, 0.9]) if below else random.choice([1.5, 2.5, 4])
    f2 = fp * r2
    pchoices = [
        {"id": "fast", "text": r"It travels through the plasma, with a phase velocity greater than $c$.", "correct": "false" if below else "true"},
        {"id": "slow", "text": r"It travels through the plasma, with a phase velocity smaller than $c$.", "correct": "false"},
        {"id": "none", "text": r"It cannot travel through the plasma: $\beta^2 \lt 0$, so there is no travelling wave, and the plasma turns the wave back.", "correct": "true" if below else "false"},
        {"id": "atc", "text": r"It travels through the plasma at exactly $c$, because a collisionless plasma has no losses.", "correct": "false"},
        {"id": "absorb", "text": r"It travels into the plasma but is absorbed, its energy heating the electrons.", "correct": "false"},
    ]
    fb_txt = {
        "fast": (f"${f2:g}\\ \\mathrm{{MHz}}$ is below $f_p$: $\\beta^2 = (\\omega^2-\\omega_p^2)/c^2 \\lt 0$, so there is no real $\\beta$ and no travelling wave at all." if below
                 else f"Right: ${f2:g}\\ \\mathrm{{MHz}} \\gt f_p$, so $\\beta = \\tfrac{{\\omega}}{{c}}\\sqrt{{1-f_p^2/f^2}}$ is real and smaller than $\\omega/c$: $v_p = \\omega/\\beta \\gt c$."),
        "slow": r"In this plasma $\beta = \tfrac{\omega}{c}\sqrt{1-f_p^2/f^2} \lt \omega/c$ whenever it is real, so $v_p = \omega/\beta$ is never below $c$; " + ("and below $f_p$ there is no travelling wave at all, not a slower one." if below else "it is the group velocity that is below $c$."),
        "none": (r"Right: below $f_p$, $\beta^2 = (\omega^2-\omega_p^2)/c^2 \lt 0$; the electrons respond in time to cancel the field, and the wave is turned back (shortwave hops off the ionosphere this way)." if below
                 else f"${f2:g}\\ \\mathrm{{MHz}}$ is above $f_p = {fp:g}\\ \\mathrm{{MHz}}$, so $\\beta^2 = (\\omega^2-\\omega_p^2)/c^2 \\gt 0$: it does travel."),
        "atc": r"Lossless does not mean non-dispersive: the electrons' current changes $\beta$, so $\beta \neq \omega/c$ at any finite frequency.",
        "absorb": r"With no collisions the electron current is $90^\circ$ out of phase with $\mathbf{E}$ (imaginary $\sigma$), so $\langle\mathbf{J}\cdot\mathbf{E}\rangle = 0$: nothing is absorbed.",
    }
    for ch in pchoices:
        ch["feedback"] = fb_txt[ch["id"]]
    random.shuffle(pchoices)

    exp_b = f"{'-' if db > 0 else '+'}j\\,{kpi(kb)} {ub}"
    b_read = (f"The exponent $e^{{{exp_b}}}$ has a {'minus' if db > 0 else 'plus'} sign, so the wave travels toward "
              f"$\\hat{{u}} = {uhat(ub, db)}$ ($\\beta = {kpi(kb)}\\ \\mathrm{{rad/m}} = 2\\pi f\\sqrt{{\\epsilon_r}}/c$, consistent with the medium).")
    if below:
        d_read = (f"At ${f2:g}\\ \\mathrm{{MHz}} \\lt f_p$, $\\beta^2 = (\\omega^2-\\omega_p^2)/c^2 \\lt 0$: there is no real $\\beta$ and no travelling wave. "
                  "The field only decays into the plasma and the wave is turned back. Not a slower wave, not a wave at $c$, and not absorption: "
                  "with no collisions the electron current is $90^\\circ$ out of phase with $\\mathbf{E}$, so $\\langle\\mathbf{J}\\cdot\\mathbf{E}\\rangle = 0$.")
    else:
        root2 = math.sqrt(1 - 1 / r2 ** 2)
        d_read = (f"At ${f2:g}\\ \\mathrm{{MHz}} \\gt f_p$ the wave travels, again with $v_p = c/\\sqrt{{1-f_p^2/f^2}} = {num(3 / root2)}\\times10^{{8}}\\ \\mathrm{{m/s}} \\gt c$. "
                  "It is never slower than $c$ in phase, never exactly $c$, and nothing is absorbed in a collisionless plasma.")

    # ================================================================ params
    data["params"].update({
        "u": u, "e": e, "A0": A0, "conv_items": conv_items, "conv_opts": conv_opts, "extra_tex": extra_tex,
        # (b)
        "ub": ub, "db": db, "e1": e1, "e2": e2, "a_b": a_b, "b_b": b_b, "nb": nb, "epsb_str": f"{nb * nb:g}",
        "nb_str": f"{nb:g}", "kb": kb, "fb_str": f"{fb:g}", "Eb_tex": Eb_tex, "eta_b_str": num(eta_b),
        "ax1": ans_axes[0], "ax2": ans_axes[1], "ub_hat": uhat(ub, db), "H_tex": H_tex,
        "ue1": uhat(h1_ax, h1_sg), "ue2": uhat(h2_ax, h2_sg), "Esq": a_b ** 2 + b_b ** 2,
        "Su_b_str": num(Su_b * 1e3), "Sb_mag_str": num(abs(Su_b) * 1e3),
        # (c)
        "pc": pc, "qc": qc, "e3": e3, "tmpl_tex": tmpl_tex, "lin_tex": lin(a_c, b_c), "a_c_str": kpi(abs(a_c)) if a_c > 0 else "-" + kpi(abs(a_c)),
        "b_c_str": kpi(abs(b_c)) if b_c > 0 else "-" + kpi(abs(b_c)), "beta_c_str": kpi(beta_c),
        "m1": m1, "m2": m2, "m3": m3, "uc_tex": vec_tex(-ip, -iq).strip("$"), "lam_c_str": num(lam_c), "f_c_str": num(f_c),
        "dir_list": dir_choices,
        # (d)
        "fp_str": f"{fp:g}", "f1_str": f"{f1:g}", "f2_str": f"{f2:g}", "root_str": num(root), "rn": r_num, "rd": r_den,
        "beta_p_str": num(beta_p), "vp8_str": num(vp8), "w_c_str": num(2 * math.pi * f1 * 1e6 / C), "below": below,
        "vg8_str": num(3.0 * root), "plasma_list": pchoices, "exp_b": exp_b, "b_read": b_read, "d_read": d_read,
    })
    ca = data["correct_answers"]
    ca["H1re"] = float(Hpart[ans_axes[0]][0])
    ca["H1im"] = float(Hpart[ans_axes[0]][1])
    ca["H2re"] = float(Hpart[ans_axes[1]][0])
    ca["H2im"] = float(Hpart[ans_axes[1]][1])
    ca["Sb"] = float(Su_b * 1e3)                          # mW/m^2 along +ub
    ca["lam_c"] = float(lam_c)                            # m
    ca["f_c"] = float(f_c)                                # GHz
    ca["beta_p"] = float(beta_p)                          # rad/m
    ca["vp8"] = float(vp8)                                # 1e8 m/s
