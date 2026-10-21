"""Lecture 22: decode a damped plane wave (given E or H, time-domain or phasor form).

gamma = alpha + j beta with alpha:beta = 1:2 (gamma^2 on a 3-4-5 triangle), 3:4 (gamma on 3-4-5) or 1:sqrt3 (30-60-90).
    eta = j w mu / gamma:  |eta| = w mu/|gamma|,  tau = angle(eta) = atan(alpha/beta)
    gamma/eta = gamma^2/(j w mu) = sigma + j w eps:  sigma = 2 alpha beta/(w mu),  eps = (beta^2 - alpha^2)/(w^2 mu)
    other field: H~ = u x E~/eta (E given) or E~ = eta H~ x u (H given); amplitude decays as e^{-alpha d}.
Constants as the statement gives them: mu0 = 4 pi 1e-7, eps0 = 1e-9/(36 pi) (c = 3e8).
"""
import cmath
import math
import random

MU0 = 4e-7 * math.pi
EPS0 = 1e-9 / (36 * math.pi)
AXES = ["x", "y", "z"]
IDX = {"x": 0, "y": 1, "z": 2}
SQ3 = math.sqrt(3.0)


def num(x, sig=4):
    s = f"{x:.{sig}g}"
    if "e" in s:
        m, e = s.split("e")
        return f"{m}\\times10^{{{int(e)}}}"
    return s


def g(x):
    return f"{x:g}"


def cross_axes(a, sa, b, sb):
    """(sa a_hat) x (sb b_hat) -> (axis, sign)."""
    ia, ib = IDX[a], IDX[b]
    ic = 3 - ia - ib
    sgn = 1 if (ib - ia) % 3 == 1 else -1
    return AXES[ic], sa * sb * sgn


def uhat(ax, sg):
    return ("+" if sg > 0 else "-") + "\\hat{" + ax + "}"


def wrap_deg(x):
    while x <= -180:
        x += 360
    while x > 180:
        x -= 360
    return x


def generate(data):
    # ------------------------------------------------------------------ the wave and the medium
    while True:
        ratio = random.choice(["12", "34", "r3"])
        if ratio == "12":
            k = random.choice([0.5, 1, 1.5, 2, 2.5, 3, 4, 5])
            alpha, beta, bk = k, 2 * k, None
        elif ratio == "34":
            s = random.choice([0.25, 0.5, 1, 1.5, 2])
            alpha, beta, bk = 3 * s, 4 * s, None
        else:
            k = random.choice([0.5, 1, 1.5, 2, 3, 4])
            alpha, beta, bk = k, SQ3 * k, k
        mur = random.choice([1, 1, 1, 2, 3, 4])
        wm = random.choice([1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8])
        wn = random.choice([7, 8, 9])
        om = wm * 10 ** wn
        mu = mur * MU0
        eps = (beta ** 2 - alpha ** 2) / (om ** 2 * mu)
        eps_r = eps / EPS0
        sigma = 2 * alpha * beta / (om * mu)
        if 2.0 <= eps_r <= 60 and 1e-3 <= sigma <= 2.0:
            break
    gam = complex(alpha, beta)
    eta = 1j * om * mu / gam
    eta_mag, tau = abs(eta), math.atan2(alpha, beta)
    tau_deg = math.degrees(tau)
    p_loss = sigma / (om * eps)

    u = random.choice(AXES)
    d = random.choice([1, -1])                       # travel toward d u_hat
    e = random.choice([ax for ax in AXES if ax != u])
    sF = random.choice([1, -1])
    field = "E" if random.random() < 0.65 else "H"
    form = random.choice(["time", "phasor"])
    phi0 = random.choice([0, 30, 45, 60, -30, -45, -60, 90])
    if field == "E":
        A = random.choice([2, 3, 4, 5, 6, 8, 10, 12, 15, 20])            # V/m
        unit_tex, other, other_unit = r"\mathrm{V/m}", "H", r"\mathrm{mA/m}"
    else:
        A = random.choice([10, 20, 25, 40, 50, 60, 80, 100])             # mA/m
        unit_tex, other, other_unit = r"\mathrm{mA/m}", "E", r"\mathrm{V/m}"

    # ------------------------------------------------------------------ display strings
    om_tex = (g(wm) + r"\times10^{" + str(wn) + "}")
    a_tex = "" if alpha == 1 else g(alpha)
    if bk is None:
        b_coef = "" if beta == 1 else g(beta)
        b_tex_time = b_coef + u
        b_tex_j = g(beta) if beta != 1 else ""
        beta_disp = g(beta)
    else:
        b_coef = ("" if bk == 1 else g(bk)) + r"\sqrt{3}"
        b_tex_time = b_coef + r"\," + u
        b_tex_j = b_coef
        beta_disp = b_coef
    amp_tex = ("-" if sF < 0 else "") + r"\hat{" + e + r"}\," + g(A)
    if form == "time":
        expo = ("-" if d > 0 else "") + a_tex + u
        ph = "" if phi0 == 0 else (f" + {phi0}^\\circ" if phi0 > 0 else f" - {-phi0}^\\circ")
        field_tex = (r"\mathbf{" + field + "}(" + u + ",t) = " + amp_tex + r"\,e^{" + expo + r"}\cos(" + om_tex
                     + r"\,t " + ("-" if d > 0 else "+") + " " + b_tex_time + ph + ")")
    else:
        ph = "" if phi0 == 0 else (r"\,e^{j" + str(phi0) + r"^\circ}" if phi0 > 0 else r"\,e^{-j" + str(-phi0) + r"^\circ}")
        field_tex = (r"\tilde{\mathbf{" + field + "}}(" + u + ") = " + amp_tex + ph + r"\,e^{" + ("-" if d > 0 else "")
                     + "(" + g(alpha) + " + j" + b_tex_j + ")" + u + "}")
    mu_tex = r"\mu = \mu_0" if mur == 1 else (r"\mu = " + str(mur) + r"\mu_0")

    # ------------------------------------------------------------------ the other field
    trav_ax, trav_sg = u, d
    ref_sg = random.choice([1, -1])                  # reference direction of the given field at the instant asked
    if field == "E":
        o_ax, o_sg = cross_axes(u, d, e, ref_sg)     # u x e_ref
        o_amp0 = A / eta_mag * 1e3                    # mA/m at u = 0
        o_phase = wrap_deg(phi0 - tau_deg)
    else:
        o_ax, o_sg = cross_axes(e, ref_sg, u, d)     # h_ref x u
        o_amp0 = A * 1e-3 * eta_mag                   # V/m
        o_phase = wrap_deg(phi0 + tau_deg)
    while True:
        D = random.choice([0.1, 0.2, 0.25, 0.4, 0.5, 0.75, 1, 1.5, 2, 2.5, 3])
        if 0.25 <= alpha * D <= 2.5:
            break
    u1 = d * D
    o_amp1 = o_amp0 * math.exp(-alpha * D)

    ref_hat = uhat(e, ref_sg)
    o_hat = uhat(o_ax, o_sg)
    trav_hat = uhat(trav_ax, trav_sg)
    F, O = r"\mathbf{" + field + "}", r"\mathbf{" + other + "}"
    if field == "E":
        rule = (r"$\tilde{\mathbf{H}} = \hat{u}\times\tilde{\mathbf{E}}/\eta$: $(" + trav_hat + r")\times(" + ref_hat + ") = " + o_hat + "$")
    else:
        rule = (r"$\tilde{\mathbf{E}} = \eta\,\tilde{\mathbf{H}}\times\hat{u}$: $(" + ref_hat + r")\times(" + trav_hat + ") = " + o_hat + "$")
    odir = [
        {"id": "ok", "text": f"${o_hat}$", "correct": "true", "feedback": "Right: " + rule + "."},
        {"id": "rev", "text": f"${uhat(o_ax, -o_sg)}$", "correct": "false",
         "feedback": "Reversed: " + rule + r". Check that $\mathbf{E}\times\mathbf{H}$ points along the travel; this choice makes it point backwards."},
        {"id": "travel", "text": f"${trav_hat}$, the direction of travel", "correct": "false",
         "feedback": r"A uniform plane wave, damped or not, has no field component along its direction of travel."},
        {"id": "par", "text": f"${ref_hat}$, parallel to ${F}$", "correct": "false",
         "feedback": r"$\mathbf{E}$ and $\mathbf{H}$ of a plane wave are perpendicular to each other."},
        {"id": "zero", "text": f"nowhere: ${O} = \\mathbf{{0}}$ at that instant, because the two fields are out of phase", "correct": "false",
         "feedback": (f"They are out of phase by only $\\tau = {num(tau_deg)}^\\circ$ (always below $45^\\circ$), so at that instant ${O}$ is "
                      f"$\\cos\\tau = {num(math.cos(tau), 3)}$ of its own peak, not zero.")},
    ]
    random.shuffle(odir)

    # ------------------------------------------------------------------ direction of travel
    if form == "time":
        rel = "opposite signs" if d > 0 else "the same sign"
        read = (f"In the cosine the $t$ and ${u}$ terms have {rel}, so the wave travels toward ${trav_hat}$, "
                f"and its amplitude $e^{{{'-' if d > 0 else ''}{a_tex}{u}}}$ indeed decays in that direction.")
    else:
        read = (f"With $\\mathrm{{Re}}\\{{\\tilde{{F}}e^{{j\\omega t}}\\}}$, the factor $e^{{{'-' if d > 0 else '+'}j{b_tex_j}{u}}}$ "
                f"inside $e^{{{'-' if d > 0 else ''}\\gamma {u}}}$ is a phase that moves toward ${trav_hat}$, and "
                f"$e^{{{'-' if d > 0 else ''}{a_tex}{u}}}$ decays in that direction.")
    dir_choices = [
        {"id": "plus", "text": f"toward $+{u}$", "correct": "true" if d > 0 else "false"},
        {"id": "minus", "text": f"toward $-{u}$", "correct": "false" if d > 0 else "true"},
        {"id": "alongF", "text": f"toward ${uhat(e, sF)}$, along ${F}$", "correct": "false",
         "feedback": r"That is the polarization of the field. A plane wave travels perpendicular to $\mathbf{E}$ and $\mathbf{H}$."},
        {"id": "stand", "text": "nowhere: a decaying field is a standing wave", "correct": "false",
         "feedback": (r"Decay does not stop the phase from moving: the zero crossings move at $v_p = \omega/\beta$. "
                      r"A standing wave would need two counter-propagating waves.")},
    ]
    wrong_dir = ("The relative sign of the $t$ and position terms decides the direction (opposite: toward $+$, same: toward $-$); "
                 "the amplitude must then decay along the travel, as it does here.") if form == "time" else (
                 r"$e^{-\gamma u}$ travels toward $+u$ and $e^{+\gamma u}$ toward $-u$ (with the $e^{j\omega t}$ convention); "
                 "the amplitude must decay along the travel, as it does here.")
    for ch in dir_choices:
        if ch["id"] in ("plus", "minus"):
            ch["feedback"] = ("Right: " + read) if ch["correct"] == "true" else ("Check the sign. " + wrong_dir)
    random.shuffle(dir_choices)

    # ------------------------------------------------------------------ checkbox pool
    lam = 2 * math.pi / beta
    delta = 1 / alpha
    lag_ns = tau / om * 1e9
    per_lam = math.exp(-alpha * lam)
    lag_true = random.random() < 0.5
    pl_true = random.random() < 0.5
    lt_true = random.random() < 0.5
    lag_tex = num(lag_ns, 3)
    pl_val = per_lam * 100 if pl_true else math.exp(-2 * math.pi) * 100
    lt_val = p_loss if lt_true else alpha / beta
    facts = [
        ("power", True, r"The time-average power density of this wave falls as $e^{-2\alpha d}$ with the distance $d$ travelled; one penetration depth further on it is about $13.5\ \%$ of its earlier value.",
         r"$\langle\mathbf{S}\rangle\propto\lvert E\rvert^2\propto e^{-2\alpha d}$, and $e^{-2} = 0.135$."),
        ("power_delta", False, r"One penetration depth $\delta = 1/\alpha$ further on, the time-average power density has fallen to $1/e$ (about $37\ \%$) of its earlier value.",
         r"The field falls to $1/e$ over $\delta$; the power density, $\propto e^{-2\alpha d}$, falls to $e^{-2} = 13.5\ \%$."),
        ("lag", lag_true, (f"At a fixed point, the magnitude of $\\mathbf{{H}}$ peaks ${lag_tex}\\ \\mathrm{{ns}}$ "
                           + ("after" if lag_true else "before") + r" the magnitude of $\mathbf{E}$ does."),
         (f"$\\mathbf{{H}}$ lags $\\mathbf{{E}}$ by $\\tau = {num(tau_deg)}^\\circ$, i.e. by $\\tau/\\omega = {lag_tex}\\ \\mathrm{{ns}}$"
          + ("." if lag_true else r"; it never leads when $\sigma$, $\epsilon$, $\mu$ are real and positive."))),
        ("per_lambda", pl_true, f"Over one wavelength along its travel, the amplitude of this wave falls to ${num(pl_val, 3)}\\ \\%$ of its value.",
         (f"$e^{{-\\alpha\\lambda}} = e^{{-2\\pi\\alpha/\\beta}} = {num(per_lam * 100, 3)}\\ \\%$"
          + ("." if pl_true else r"; $e^{-2\pi} = 0.187\ \%$ belongs to the good-conductor limit $\alpha = \beta$, not to this medium."))),
        ("loss_tan", lt_true, f"In this medium the amplitude of the conduction current density $\\sigma\\mathbf{{E}}$ is ${num(lt_val, 3)}$ times that of the displacement current density.",
         (f"$\\sigma/\\omega\\epsilon = \\tan 2\\tau = 2\\alpha\\beta/(\\beta^2-\\alpha^2) = {num(p_loss, 3)}$"
          + ("." if lt_true else f"; ${num(alpha / beta, 3)} = \\alpha/\\beta = \\tan\\tau$, the tangent of the angle of $\\eta$, not of the loss angle $2\\tau$."))),
        ("mu_cancel", True, r"A wave with the same $\omega$, $\alpha$ and $\beta$ in a medium with twice this $\mu$ would imply the same ratio $\sigma/\omega\epsilon$ as here.",
         r"$\sigma = 2\alpha\beta/\omega\mu$ and $\epsilon = (\beta^2-\alpha^2)/\omega^2\mu$ both scale as $1/\mu$; their ratio does not depend on $\mu$."),
        ("vp_slower", True, r"The phase velocity $\omega/\beta$ of this wave is smaller than $1/\sqrt{\mu\epsilon}$ evaluated with this medium's $\mu$ and $\epsilon$.",
         r"$\beta^2 = \omega^2\mu\epsilon + \alpha^2 \gt \omega^2\mu\epsilon$, so $\omega/\beta \lt 1/\sqrt{\mu\epsilon}$."),
        ("double_f", False, r"If the frequency were doubled (same $\sigma$, $\epsilon$ and $\mu$), $\beta$ would exactly double.",
         r"In a conducting medium $\beta$ is not proportional to $\omega$ (the ratio $\sigma/\omega\epsilon$ changes too): the medium is dispersive."),
        ("alpha_gt_beta", False, r"A medium with real, positive $\sigma$, $\epsilon$ and $\mu$ can carry a plane wave whose $\alpha$ exceeds its $\beta$.",
         r"$\alpha/\beta = \tan\tau$ with $\tau = \tfrac12\tan^{-1}(\sigma/\omega\epsilon) \lt 45^\circ$, so $\alpha \lt \beta$ always."),
        ("steady", True, r"At a fixed point the field oscillates with an amplitude that does not change in time; the decay is from point to point.",
         r"The source keeps feeding the wave: at each point the field is a steady sinusoid with amplitude $\propto e^{-\alpha d}$."),
        ("eta_real", False, r"For this wave $\lvert\mathbf{E}\rvert/\lvert\mathbf{H}\rvert$ (ratio of the amplitudes) equals $\sqrt{\mu/\epsilon}$ for this medium's $\mu$ and $\epsilon$.",
         r"With $\sigma \gt 0$, $\lvert\eta\rvert = \sqrt{\mu/\epsilon}\,/\,[1 + (\sigma/\omega\epsilon)^2]^{1/4}$ is smaller than $\sqrt{\mu/\epsilon}$."),
        ("grows", False, r"Followed along its direction of travel, the amplitude of this wave increases.",
         r"A wave in a passive medium decays in the direction it travels; the field is largest on the source side."),
        ("Sdir", True, r"The time-average Poynting vector $\tfrac12\mathrm{Re}\{\tilde{\mathbf{E}}\times\tilde{\mathbf{H}}^*\}$ of this wave points along its direction of travel.",
         r"$\langle\mathbf{S}\rangle = \hat{u}\,(\lvert E_0\rvert^2/2\lvert\eta\rvert)\cos\tau\,e^{-2\alpha d}$ with $\cos\tau \gt 0$."),
    ]
    fact_list = [{"id": fid, "correct": "true" if t else "false", "text": txt,
                  "feedback": ("True: " if t else "False: ") + why, "verdict": "true" if t else "false", "why": why}
                 for fid, t, txt, why in facts]
    random.shuffle(fact_list)

    # ------------------------------------------------------------------ answer-panel strings
    om_mu = om * mu
    gmag = abs(gam)
    gang = math.degrees(cmath.phase(gam))
    g2 = gam * gam
    if ratio == "12":
        tri = r"$\gamma^2 \propto (1 + j2)^2 = -3 + j4$, a 3-4-5 triangle"
    elif ratio == "34":
        tri = r"$\gamma \propto 3 + j4$, a 3-4-5 triangle: $\lvert\gamma\rvert = 5\alpha/3$"
    else:
        tri = r"$\gamma \propto 1 + j\sqrt{3} = 2e^{j60^\circ}$, a 30-60-90 triangle"
    o_phase = round(o_phase, 6) + 0.0
    if field == "E":
        f_ax, f_sg = cross_axes(u, d, e, sF)
    else:
        f_ax, f_sg = cross_axes(e, sF, u, d)
    expo_o = ("-" if d > 0 else "") + a_tex + u
    ph_o = "" if o_phase == 0 else (f" + {num(o_phase)}^\\circ" if o_phase > 0 else f" - {num(-o_phase)}^\\circ")
    o_expr = (r"\mathbf{" + other + "}(" + u + ",t) = " + ("-" if f_sg < 0 else "") + r"\hat{" + f_ax + r"}\," + num(o_amp0)
              + r"\,e^{" + expo_o + r"}\cos(" + om_tex + r"\,t " + ("-" if d > 0 else "+") + " " + b_tex_time + ph_o + ")")
    if field == "E":
        other_txt = (f"$\\tilde{{\\mathbf{{H}}}} = \\hat{{u}}\\times\\tilde{{\\mathbf{{E}}}}/\\eta$ with $\\hat{{u}} = {trav_hat}$: divide the amplitude by "
                     f"$\\lvert\\eta\\rvert$ and subtract $\\tau$ from the phase. At $u = 0$ the amplitude is ${g(A)}/{num(eta_mag)} = {num(o_amp0)}\\ \\mathrm{{mA/m}}$ and the phase "
                     f"${phi0}^\\circ - {num(tau_deg)}^\\circ = {num(o_phase)}^\\circ$. So ${o_expr}\\ \\mathrm{{mA/m}}$.")
    else:
        other_txt = (f"$\\tilde{{\\mathbf{{E}}}} = \\eta\\,\\tilde{{\\mathbf{{H}}}}\\times\\hat{{u}}$ with $\\hat{{u}} = {trav_hat}$: multiply the amplitude by "
                     f"$\\lvert\\eta\\rvert$ and add $\\tau$ to the phase. At $u = 0$ the amplitude is $({g(A)}\\times10^{{-3}})({num(eta_mag)}) = {num(o_amp0)}\\ \\mathrm{{V/m}}$ and the phase "
                     f"${phi0}^\\circ + {num(tau_deg)}^\\circ = {num(o_phase)}^\\circ$. So ${o_expr}\\ \\mathrm{{V/m}}$.")
    data["params"].update({
        "field": field, "other": other, "form": form, "u": u, "e": e, "sF": sF, "A": A, "phi0": phi0, "d": d,
        "ratio": ratio, "mur": mur, "wm": wm, "wn": wn,
        "field_tex": field_tex, "unit_tex": unit_tex, "other_unit": other_unit, "om_tex": om_tex, "mu_tex": mu_tex,
        "is_phasor": form == "phasor", "F_tex": F, "O_tex": O, "ref_hat": ref_hat, "u1_str": g(u1), "D_str": g(D),
        "dir_choices": dir_choices, "odir_choices": odir, "facts": fact_list,
        "read_txt": read, "alpha_str": g(alpha), "beta_str": num(beta),
        "beta_eq": beta_disp if bk is None else beta_disp + " = " + num(beta),
        "lam_str": num(lam), "delta_str": num(delta),
        "ommu_str": num(om_mu), "gmag_str": num(gmag), "gang_str": num(gang), "eta_str": num(eta_mag), "tau_str": num(tau_deg),
        "eta_re": num(eta.real), "eta_im": num(eta.imag), "tri_txt": tri, "cos_tau": num(math.cos(tau), 3),
        "other_txt": other_txt, "o_expr": o_expr, "o_hat": o_hat, "trav_hat": trav_hat, "oamp0_str": num(o_amp0), "oamp1_str": num(o_amp1),
        "aD_str": num(alpha * D), "g2re": num(g2.real), "g2im": num(g2.imag),
        "sigma_str": num(sigma * 1e3), "eps_str": num(eps), "epsr_str": num(eps_r), "p_str": num(p_loss, 3),
        "omeps_str": num(om * eps), "lag_str": lag_tex,
    })
    data["correct_answers"].update({
        "alpha": float(alpha), "beta": float(beta), "lam": float(lam), "delta": float(delta),
        "eta_mag": float(eta_mag), "tau_deg": float(tau_deg), "Oamp": float(o_amp1),
        "sigma": float(sigma * 1e3), "epsr": float(eps_r),
    })
