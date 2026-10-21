"""Lecture 25 concept question: normal incidence on a planar interface between two lossless media,
and the perfect-conductor limit.

Interface w = 0; region 1 (incident side) is w > 0 if s1 = +1, w < 0 if s1 = -1; the incident wave travels
u_i = -s1 w_hat:  E_i = sig E0 cos(omega t + s1 beta1 w) a_hat  (phasor sig E0 e^{j s1 beta1 w} a_hat).
    eta = eta0 sqrt(mu_r/eps_r) (eta0 = 120 pi),  Gamma = (eta2 - eta1)/(eta2 + eta1),  tau = 1 + Gamma.
At w = 0, t = 0:  E_r = Gamma sig E0 a_hat,  E_t = tau sig E0 a_hat,
    H_r = (u_r x a_hat) Gamma sig E0/eta1 with u_r = s1 w_hat  (H_r/H_i = -Gamma).
Power: 1 - Gamma^2 is transmitted;  <S_t> = tau^2 E0^2/(2 eta2) along u_i.
Mirror (region 2 a perfect conductor): Gamma = -1, H(0) = 2 H_i(0), J_s = n_hat x H with n_hat = s1 w_hat
(out of the conductor), which gives J_s = (2 sig E0/eta1) cos(omega t) a_hat.
A matched pair (eta1 = eta2) is drawn about one time in ten.
"""
import math
import random

AXES = ["x", "y", "z"]
IDX = {"x": 0, "y": 1, "z": 2}
ETA0 = 120 * math.pi
# (eps_r, mu_r): lossless media with realistic constants (polyethylene 2.25; glasses 4-6.25; alumina-like 9;
# high-permittivity ceramic 16; magnetodielectric composites with mu_r up to 9)
MEDIA = [(1, 1), (2.25, 1), (4, 1), (6.25, 1), (9, 1), (16, 1), (1, 4), (4, 9), (9, 4)]
MATCHED = [((1, 1), (4, 4)), ((1, 1), (2.25, 2.25)), ((2.25, 1), (9, 4))]


def num(x, sig=4):
    if abs(x) < 1e-12:
        return "0"
    s = f"{x:.{sig}g}"
    if "e" in s:
        m, e = s.split("e")
        return f"{m}\\times10^{{{int(e)}}}"
    return s


def hat(ax):
    return r"\hat{" + ax + "}"


def uhat(ax, sg):
    return ("+" if sg > 0 else "-") + hat(ax)


def eps_cyc(i, j, k):
    """Levi-Civita sign of axis letters."""
    return {("x", "y", "z"): 1, ("y", "z", "x"): 1, ("z", "x", "y"): 1,
            ("x", "z", "y"): -1, ("z", "y", "x"): -1, ("y", "x", "z"): -1}[(i, j, k)]


def medium_tex(m):
    er, mr = m
    if er == 1 and mr == 1:
        return r"air ($\epsilon_r = 1$, $\mu_r = 1$)"
    return f"a lossless medium with $\\epsilon_r = {num(er)}$, $\\mu_r = {num(mr)}$"


def eta_of(m):
    return ETA0 * math.sqrt(m[1] / m[0])


def generate(data):
    w = random.choice(AXES)
    a = random.choice([ax for ax in AXES if ax != w])
    b = [ax for ax in AXES if ax not in (w, a)][0]
    s1 = random.choice([1, -1])
    sig = random.choice([1, -1])
    E0 = random.choice([2, 3, 4, 5, 6, 8, 10, 12, 15, 20])
    matched = random.random() < 0.1
    if matched:
        m1, m2 = random.choice(MATCHED)
        if random.random() < 0.5:
            m1, m2 = m2, m1
    else:
        while True:
            m1, m2 = random.sample(MEDIA, 2)
            if abs(eta_of(m1) - eta_of(m2)) > 1e-9:
                break
    eta1, eta2 = eta_of(m1), eta_of(m2)
    Gam = (eta2 - eta1) / (eta2 + eta1)
    if matched:
        Gam = 0.0
    tau = 1 + Gam
    sw = s1                                   # reflected wave travels toward s1 * w_hat
    ui_sign = -s1
    e_wab = eps_cyc(w, a, b)                  # w_hat x a_hat = e_wab * b_hat
    Er = Gam * sig * E0
    Et = tau * sig * E0
    Hi_b = ui_sign * e_wab * sig * E0 / eta1 * 1e3          # mA/m
    Hr_b = sw * e_wab * Gam * sig * E0 / eta1 * 1e3          # mA/m
    Ht_b = ui_sign * e_wab * tau * sig * E0 / eta2 * 1e3     # mA/m
    frac_t = (1 - Gam ** 2) * 100
    Si = E0 ** 2 / (2 * eta1) * 1e3
    Sr = Gam ** 2 * Si
    St = tau ** 2 * E0 ** 2 / (2 * eta2) * 1e3
    St_w = ui_sign * St
    Js = 2 * E0 / eta1 * 1e3
    Htot_pec_b = 2 * Hi_b

    reg1_side = w + (r" \gt 0" if s1 > 0 else r" \lt 0")
    reg2_side = w + (r" \lt 0" if s1 > 0 else r" \gt 0")
    arg = r"\omega t " + ("+" if s1 > 0 else "-") + r" \beta_1 " + w
    E_tex = ("-" if sig < 0 else "") + f"{E0}\\cos({arg})\\,{hat(a)}"
    Ehat = uhat(a, sig)

    # ---------------------------------------------------------------- (d) MC: reflected E
    ok = "none" if matched else ("same" if Gam > 0 else "opp")
    why_sign = (f"$\\eta_2 = {num(eta2)}\\ \\Omega$ {'equals' if matched else ('exceeds' if Gam > 0 else 'is less than')} "
                f"$\\eta_1 = {num(eta1)}\\ \\Omega$")
    refl = [
        {"id": "same", "text": r"points the same way as the incident $\mathbf{E}$ at every instant ($0^\circ$ phase difference)"},
        {"id": "opp", "text": r"points opposite to the incident $\mathbf{E}$ at every instant ($180^\circ$ phase difference)"},
        {"id": "quad", "text": r"is $90^\circ$ out of phase with the incident $\mathbf{E}$"},
        {"id": "none", "text": r"is zero at every instant: there is no reflected wave"},
    ]
    fb = {
        "same": (f"Right: {why_sign}, so $\\Gamma \\gt 0$ and $\\tilde{{E}}_r(0) = \\Gamma\\tilde{{E}}_i(0)$ points along the incident field."
                 if ok == "same" else
                 f"$\\Gamma = (\\eta_2-\\eta_1)/(\\eta_2+\\eta_1)$ and {why_sign}, so $\\Gamma$ is {'zero' if matched else 'negative'}."),
        "opp": (f"Right: {why_sign}, so $\\Gamma \\lt 0$ and the reflected $\\mathbf{{E}}$ is inverted at the interface."
                if ok == "opp" else
                (f"{why_sign}, so $\\Gamma$ is {'zero' if matched else 'positive'}. Reversing the travel flips the reflected "
                 r"$\mathbf{H}$ (ratio $-\Gamma$), not the reflected $\mathbf{E}$ (ratio $\Gamma$).")),
        "quad": r"Both media are lossless, so both impedances are real and so is $\Gamma$: the reflected $\mathbf{E}$ is in phase ($\Gamma \gt 0$) or inverted ($\Gamma \lt 0$).",
        "none": ("Right: the impedances are equal, so $\\Gamma = 0$, although the two media differ." if ok == "none" else
                 f"No reflection needs $\\eta_2 = \\eta_1$; here {why_sign}."),
    }
    for ch in refl:
        ch["correct"] = "true" if ch["id"] == ok else "false"
        ch["feedback"] = fb[ch["id"]]
    random.shuffle(refl)

    # ---------------------------------------------------------------- (e) MC: J_s direction on a mirror
    nhat = uhat(w, s1)
    js = [
        {"id": "alongE", "correct": "true", "text": f"${Ehat}$",
         "feedback": f"Right: $\\mathbf{{J}}_s = \\hat{{n}}\\times\\mathbf{{H}}$ with $\\hat{{n}} = {nhat}$ (out of the conductor) and the doubled $\\mathbf{{H}}$; it runs along the incident $\\mathbf{{E}}$ at the surface."},
        {"id": "antiE", "correct": "false", "text": f"${uhat(a, -sig)}$",
         "feedback": f"That is $-\\hat{{n}}\\times\\mathbf{{H}}$. The normal must point out of the conductor into region 1: $\\hat{{n}} = {nhat}$."},
        {"id": "bplus", "correct": "false", "text": f"${uhat(b, 1)}$",
         "feedback": f"The surface $\\mathbf{{H}}$ lies along $\\pm{hat(b)}$; $\\mathbf{{J}}_s = \\hat{{n}}\\times\\mathbf{{H}}$ is perpendicular to it and to $\\hat{{n}}$, so it lies along $\\pm{hat(a)}$."},
        {"id": "bminus", "correct": "false", "text": f"${uhat(b, -1)}$",
         "feedback": f"The surface $\\mathbf{{H}}$ lies along $\\pm{hat(b)}$; $\\mathbf{{J}}_s = \\hat{{n}}\\times\\mathbf{{H}}$ is perpendicular to it and to $\\hat{{n}}$, so it lies along $\\pm{hat(a)}$."},
        {"id": "normal", "correct": "false", "text": f"${nhat}$, normal to the surface",
         "feedback": r"A surface current flows in the surface: $\hat{n}\times\mathbf{H}$ has no normal component."},
    ]
    random.shuffle(js)

    # ---------------------------------------------------------------- (f) checkbox pool
    facts = [
        ("E_cont", True, r"At every instant, the total tangential $\mathbf{E}$ just on the region-1 side of the interface equals the tangential $\mathbf{E}$ just on the region-2 side.",
         r"That is the condition $1 + \Gamma = \tau$: equal phasors give equal real fields at all $t$."),
        ("H_cont", True, r"With no surface current on the interface, the total tangential $\mathbf{H}$ is continuous across it at every instant.",
         r"$\hat{n}\times(\mathbf{H}_1-\mathbf{H}_2) = \mathbf{J}_s = 0$; this is the condition $(1-\Gamma)/\eta_1 = \tau/\eta_2$."),
        ("match", True, r"A lossless material with $\mu_r = \epsilon_r = 3$ in contact with vacuum reflects nothing at normal incidence, although waves travel at $c/3$ in it.",
         r"$\eta = \eta_0\sqrt{3/3} = \eta_0$, so $\Gamma = 0$. Reflection depends on $\mu/\epsilon$, speed on $\mu\epsilon$."),
        ("tau_gt1", True, r"A transmission coefficient $\tau \gt 1$ does not violate energy conservation.",
         r"The transmitted power fraction is $\tau^2\eta_1/\eta_2 = 1 - \Gamma^2 \le 1$; e.g. $\eta_2 = 2\eta_1$ gives $\tau = 4/3$ but $8/9$ of the power."),
        ("swap", True, r"Sending a wave the other way across the interface of parts (a) to (d) (from region 2 into region 1) replaces $\Gamma$ by $-\Gamma$, so the reflected fraction of the power, $\Gamma^2$, is the same.",
         f"Swapping $\\eta_1$ and $\\eta_2$ changes the sign of $\\eta_2-\\eta_1$: $\\Gamma = {num(Gam)}$ becomes ${num(-Gam)}$."),
        ("H_ratio", True, r"At the interface, the reflected $\mathbf{H}$ equals $-\Gamma$ times the incident $\mathbf{H}$.",
         r"$\tilde{\mathbf{H}} = \hat{u}\times\tilde{\mathbf{E}}/\eta$, and $\hat{u}$ reverses for the reflected wave while $\tilde{\mathbf{E}}$ scales by $\Gamma$."),
        ("tau_1mg", False, r"For every pair of media, $\tau = 1 - \Gamma$.",
         r"$\tau = 1 + \Gamma$; it is the magnetic-field ratio that is $1-\Gamma$ (and $\tau\eta_1/\eta_2 = 1 - \Gamma$)."),
        ("inv_hi", False, r"If $\eta_2 \gt \eta_1$, the reflected $\mathbf{E}$ at the interface points opposite to the incident $\mathbf{E}$.",
         r"$\eta_2 \gt \eta_1$ gives $\Gamma \gt 0$: the reflected $\mathbf{E}$ is in phase; it is the reflected $\mathbf{H}$ that is inverted."),
        ("speed", False, r"Two lossless media in which waves travel at the same speed never reflect a normally incident wave.",
         r"A medium with $\epsilon_r = 1$, $\mu_r = 4$ against one with $\epsilon_r = 4$, $\mu_r = 1$: same speed $c/2$, but $\eta = 2\eta_0$ against $\eta_0/2$, so $\Gamma = \pm 0.6$."),
        ("pec_both", False, r"At the surface of a perfect conductor, the total tangential $\mathbf{E}$ and the total tangential $\mathbf{H}$ both vanish.",
         r"Tangential $\mathbf{E}$ vanishes ($\Gamma = -1$), but tangential $\mathbf{H}$ doubles to $2E_0/\eta_1$; the jump is carried by $\mathbf{J}_s$."),
        ("tau_le1", False, r"Because energy is conserved, $\lvert\tau\rvert$ can never exceed $1$.",
         r"$\tau = 2\eta_2/(\eta_2+\eta_1)$ exceeds $1$ whenever $\eta_2 \gt \eta_1$; the power, not the field ratio, is bounded."),
        ("frac_tau2", False, r"The fraction of the incident power that is transmitted always equals $\tau^2$.",
         r"It is $\tau^2\eta_1/\eta_2 = 1-\Gamma^2$: the transmitted wave lives in a medium with a different $\eta$."),
        ("amp", tau > 1 + 1e-12, r"For the two media of parts (a) to (d), the transmitted $\mathbf{E}$ at the interface is larger in amplitude than the incident $\mathbf{E}$.",
         f"$\\tau = {num(tau)}" + (r" \gt 1$, because $\eta_2 \gt \eta_1$." if tau > 1 + 1e-12 else
                                   (r"$: matched media, no change." if matched else r" \lt 1$, because $\eta_2 \lt \eta_1$."))),
    ]
    fact_list = [{"id": fid, "correct": "true" if t else "false", "text": txt,
                  "feedback": ("True: " if t else "False: ") + why, "verdict": "true" if t else "false", "why": why}
                 for fid, t, txt, why in facts]
    random.shuffle(fact_list)

    p = data["params"]
    p.update({
        "w": w, "a": a, "b": b, "s1": s1, "sig": sig, "E0": E0, "matched": matched,
        "er1": m1[0], "mr1": m1[1], "er2": m2[0], "mr2": m2[1],
        "medium1": medium_tex(m1), "medium2": medium_tex(m2),
        "reg1_side": reg1_side, "reg2_side": reg2_side, "E_tex": E_tex, "Ehat": Ehat,
        "ui": uhat(w, ui_sign), "ur": uhat(w, sw), "nhat": nhat, "n_x_b": uhat(a, s1 * eps_cyc(w, b, a)),
        "ur_x_a": uhat(b, sw * e_wab), "ui_x_a": uhat(b, ui_sign * e_wab),
        "eta1_str": num(eta1), "eta2_str": num(eta2), "rt1": num(math.sqrt(m1[1] / m1[0])), "rt2": num(math.sqrt(m2[1] / m2[0])),
        "Gam_str": num(Gam), "tau_str": num(tau), "sigE0": num(sig * E0),
        "Er_str": num(Er + 0.0), "Et_str": num(Et), "Hi_str": num(Hi_b), "Hr_str": num(Hr_b), "Ht_str": num(Ht_b),
        "HiHr_str": num(Hi_b + Hr_b), "mG_str": num(-Gam), "G2_str": num(Gam ** 2), "G2pct": num(Gam ** 2 * 100),
        "frac_str": num(frac_t), "Si_str": num(Si), "Sr_str": num(Sr), "St_str": num(St), "Stw_str": num(St_w),
        "Js_str": num(Js), "Hpec_str": num(Htot_pec_b), "Jsig_str": num(sig * Js),
        "refl": refl, "js": js, "facts": fact_list,
    })
    ca = data["correct_answers"]
    ca["Gam"] = float(Gam) + 0.0
    ca["tau"] = float(tau)
    ca["Er"] = float(Er) + 0.0
    ca["Et"] = float(Et)
    ca["Hr"] = float(Hr_b) + 0.0
    ca["frac"] = float(frac_t)
    ca["St"] = float(St_w)
    ca["Js"] = float(Js)
