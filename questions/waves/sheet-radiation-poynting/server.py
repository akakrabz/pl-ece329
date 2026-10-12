"""Lecture 19 concept question: a sinusoidal current sheet radiating into a lossless dielectric (real cosines only).

Sheet on the plane w = 0, J_s(t) = sJ*J0*cos(w t) a_hat, medium eps_r = n^2 (mu_r = 1); the statement says to use
c = 3e8 m/s and eta0 = 120 pi ohm, so every number is exact.
    v_p = c/n,  lambda = v_p/f,  beta = 2 pi/lambda = omega/v_p,  eta = eta0/n.
At distance |w| on the side with outward normal n_hat = sign(w) w_hat (course formulas):
    E = -(eta/2) J_s(t - |w|/v_p),   H = (1/2) J_s(t - |w|/v_p) x n_hat,   S = E x H = (eta/4) J0^2 cos^2(phase) n_hat,
    phase = omega t - beta |w|.
The point is chosen with |w1| = k lambda/8 and t1 = m T/8, (m - k) mod 8 in {0, 1, 3, 4, 5, 7}, so |cos(phase)| is 1 or 1/sqrt 2.
E is even across the sheet (same on both sides), H is odd, S points away from the sheet on both sides.
"""
import math
import random

AXES = ["x", "y", "z"]
C = 3e8
ETA0 = 120 * math.pi


def num(x, sig=4):
    s = f"{x:.{sig}g}"
    if "e" in s:
        m, e = s.split("e")
        return f"{m}\\times10^{{{int(e)}}}"
    return s


def exact(x):
    """Terminating decimals printed in full (the point and the time are chosen so that they terminate)."""
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s if s not in ("-0", "") else "0"


def uhat(ax, sg):
    return ("+" if sg > 0 else "-") + "\\hat{" + ax + "}"


def generate(data):
    w = random.choice(AXES)
    a = random.choice([ax for ax in AXES if ax != w])
    b = [ax for ax in AXES if ax not in (w, a)][0]
    sJ = random.choice([1, -1])
    J0 = random.choice([0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0])          # A/m
    f_MHz = random.choice([25, 50, 100, 125, 200, 250])
    n = random.choice([1.0, 1.5, 2.0, 2.5, 3.0])
    eps_r = n * n
    T_ns = 1000.0 / f_MHz
    lam = 300.0 / (f_MHz * n)                                            # m
    vp8 = 3.0 / n                                                        # 1e8 m/s
    beta = 2 * math.pi / lam
    eta = ETA0 / n
    sw = random.choice([1, -1])
    while True:
        k = random.randint(1, 15)
        m = random.randint(0, 15)
        if (m - k) % 8 in (0, 1, 3, 4, 5, 7):
            break
    w1 = sw * k * lam / 8
    t1 = m * T_ns / 8
    w1_s, t1_s = exact(w1), exact(t1)
    # answers from the displayed values (they are exact)
    w1v, t1v = float(w1_s), float(t1_s)
    phase = 2 * math.pi * f_MHz * 1e6 * t1v * 1e-9 - beta * abs(w1v)
    cph = math.cos(phase)
    Ea = -(eta / 2) * sJ * J0 * cph                                       # V/m, along +a
    idx = {"x": 0, "y": 1, "z": 2}
    axw = 1 if (idx[w] - idx[a]) % 3 == 1 else -1                       # a_hat x w_hat = axw * b_hat
    Hb = 0.5 * sJ * J0 * cph * sw * axw                                  # A/m, along +b
    Sw = (eta / 4) * J0 ** 2 * cph ** 2 * sw                              # W/m^2, along +w

    medium = "vacuum" if n == 1.0 else f"a lossless, non-magnetic dielectric with $\\epsilon_r = {num(eps_r)}$"
    J_tex = ("" if sJ > 0 else "-") + f"{num(J0)}\\cos(\\omega t)\\,\\hat{{{a}}}"
    side = "+" if sw > 0 else "-"
    away_here, away_there = uhat(w, sw), uhat(w, -sw)

    # ---------------------------------------------------------------- MC: S on the other side
    other_rel = r"\lt" if sw > 0 else r"\gt"
    s_choices = [
        {"id": "away", "text": f"along ${away_there}$", "correct": "true",
         "feedback": f"Right: on the side ${w} {other_rel} 0$ the wave travels toward ${away_there}$, away from the sheet, and $\\mathbf{{S}}$ points with it."},
        {"id": "same", "text": f"along ${away_here}$", "correct": "false",
         "feedback": (f"That would be toward the sheet. $\\mathbf{{H}} = \\tfrac12\\mathbf{{J}}_s\\times\\hat{{n}}$ uses the normal pointing toward the field point, "
                      f"$\\hat{{n}} = {away_there}$ on that side; $\\mathbf{{E}}$ is the same as on this side, $\\mathbf{{H}}$ is reversed, so $\\mathbf{{S}}$ is reversed too.")},
        {"id": "zero", "text": r"nowhere: $\mathbf{S} = \mathbf{0}$ there, because the waves from the two faces cancel", "correct": "false",
         "feedback": f"Each face launches its own wave into its own half-space; nothing cancels. At ${w} = {exact(-w1v)}$ m the delay is the same as at ${w} = {w1_s}$ m, so $\\lvert\\mathbf{{S}}\\rvert$ is the same as in part (c)."},
        {"id": "alongJ", "text": f"along ${uhat(a, sJ)}$, parallel to the current", "correct": "false",
         "feedback": r"$\mathbf{E}$ is parallel to $\mathbf{J}_s$ and $\mathbf{S} = \mathbf{E}\times\mathbf{H}$ is perpendicular to $\mathbf{E}$, so $\mathbf{S}$ has no component along the current."},
    ]
    random.shuffle(s_choices)
    parity = [
        {"id": "Eeven", "correct": "true",
         "text": r"$\mathbf{E}$ is the same on both sides; $\mathbf{H}$ has the same magnitude but the opposite direction.",
         "feedback": r"Right: tangential $\mathbf{E}$ is continuous at the sheet, and tangential $\mathbf{H}$ jumps by $J_s$, half on each side."},
        {"id": "Hevenl", "correct": "false",
         "text": r"$\mathbf{E}$ has the opposite direction on the two sides; $\mathbf{H}$ is the same on both sides.",
         "feedback": r"Reversed roles: a jump in tangential $\mathbf{E}$ is forbidden (it would need a magnetic current), while $\mathbf{H} = \tfrac12\mathbf{J}_s\times\hat{n}$ flips with $\hat{n}$."},
        {"id": "both_odd", "correct": "false",
         "text": r"Both $\mathbf{E}$ and $\mathbf{H}$ have opposite directions on the two sides.",
         "feedback": r"$\mathbf{E} = -\tfrac{\eta}{2}\mathbf{J}_s(t-\lvert " + w + r"\rvert/v_p)$ has no $\hat{n}$ in it: it is the same on both sides and continuous."},
        {"id": "both_even", "correct": "false",
         "text": r"Both $\mathbf{E}$ and $\mathbf{H}$ are the same on both sides.",
         "feedback": r"Then $\mathbf{H}$ would be continuous and could not account for the sheet current: $\hat{n}\times(\mathbf{H}_1-\mathbf{H}_2) = \mathbf{J}_s$."},
    ]
    random.shuffle(parity)

    # ---------------------------------------------------------------- checkbox pool
    dt_true = abs(w1v) / (vp8 * 1e8) * 1e9                               # ns
    if n == 1.0:
        delay_true, dt_shown = True, dt_true
    else:
        delay_true = random.random() < 0.5
        dt_shown = dt_true if delay_true else abs(w1v) / C * 1e9
    facts = [
        ("E_cont", True, r"At every instant, $\mathbf{E}$ is the same just on either side of the sheet.",
         r"Tangential $\mathbf{E}$ is continuous; $\mathbf{E} = -\tfrac{\eta}{2}\mathbf{J}_s$ at the sheet on both faces."),
        ("H_jump", True, r"The tangential $\mathbf{H}$ just on the two sides of the sheet differ by exactly the sheet current: $\hat{n}\times(\mathbf{H}_1-\mathbf{H}_2) = \mathbf{J}_s$, with $\hat{n}$ pointing from side 2 into side 1.",
         r"Each side carries $\tfrac12\mathbf{J}_s\times\hat{n}$ with its own $\hat{n}$, so the jump is $J_s$."),
        ("E_opposes", True, r"At the sheet, $\mathbf{E}$ points opposite to $\mathbf{J}_s$ whenever $\mathbf{J}_s\neq\mathbf{0}$, so $\mathbf{J}_s\cdot\mathbf{E}\le 0$: the current gives energy to the fields.",
         r"$\mathbf{J}_s\cdot\mathbf{E} = -\tfrac{\eta}{2}\lvert\mathbf{J}_s\rvert^2\le 0$; the sheet is the source of the energy the two waves carry away."),
        ("S_double", True, r"At a fixed point, the instantaneous $\lvert\mathbf{S}\rvert$ goes through two maxima in every period of the source current.",
         r"$\lvert\mathbf{S}\rvert\propto\cos^2(\omega t-\beta\lvert " + w + r"\rvert)$, which repeats twice per period."),
        ("short_lambda", True, r"At the same frequency, a medium with a larger $\epsilon_r$ has a shorter wavelength.",
         r"$\lambda = v_p/f = c/(f\sqrt{\epsilon_r})$."),
        ("H_indep", True, r"For a given sheet current, the amplitude of $\mathbf{H}$ does not depend on $\epsilon_r$, but the amplitude of $\mathbf{E}$ does.",
         r"$\lvert\mathbf{H}\rvert = \tfrac12J_0$ is fixed by the jump condition; $\lvert\mathbf{E}\rvert = \tfrac{\eta}{2}J_0$ with $\eta = \eta_0/\sqrt{\epsilon_r}$."),
        ("E_flips", False, r"$\mathbf{E}$ reverses direction across the sheet, just as $\mathbf{H}$ does.",
         r"$\mathbf{E}$ is the same on both sides; only $\mathbf{H}$ flips."),
        ("amp_full", False, r"The amplitude of $\mathbf{E}$ on each side is $\eta J_0$.",
         r"Each side gets half: $\tfrac{\eta}{2}J_0$ (and $\tfrac12J_0$ for $\mathbf{H}$)."),
        ("spread", False, r"The amplitude of the waves decreases with distance from the sheet, because their energy spreads out.",
         r"An infinite sheet launches uniform plane waves, which do not spread; in a lossless medium the amplitude is the same at every distance."),
        ("S_toward", False, f"At some instants, $\\mathbf{{S}}$ on the side ${w} \\gt 0$ points toward the sheet.",
         r"$\mathbf{S} = \tfrac{\eta}{4}J_0^2\cos^2(\omega t-\beta\lvert " + w + r"\rvert)\,\hat{n}$ never points against $\hat{n}$: $\mathbf{E}$ and $\mathbf{H}$ change sign together."),
        ("plus_arg", False, f"On the side ${w} \\gt 0$, the fields depend on $t$ and ${w}$ through $t + {w}/v_p$.",
         f"Outgoing waves only: on ${w} \\gt 0$ the argument is $t - {w}/v_p$ (travel toward $+{w}$)."),
        ("lambda_omega", False, r"The wavelength is $\lambda = v_p/\omega$.",
         r"$\lambda = 2\pi/\beta = 2\pi v_p/\omega = v_p/f$."),
        ("delay", delay_true, f"The wave takes ${num(dt_shown)}\\ \\mathrm{{ns}}$ to travel from the sheet to the point of part (b).",
         f"The travel time is $\\lvert {w}_1\\rvert/v_p = ({exact(abs(w1v))}\\ \\mathrm{{m}})/({num(vp8)}\\times10^{{8}}\\ \\mathrm{{m/s}}) = {num(dt_true)}\\ \\mathrm{{ns}}$"
         + ("." if delay_true else r"; the value quoted uses $c$ instead of $v_p$.")),
    ]
    fact_list = [{"id": fid, "correct": "true" if t else "false", "text": txt,
                  "feedback": ("True: " if t else "False: ") + why, "verdict": "true" if t else "false", "why": why}
                 for fid, t, txt, why in facts]
    random.shuffle(fact_list)

    # ---------------------------------------------------------------- answer-panel strings
    ph_frac = ((m - k) % 8)
    ph_tex = {0: "0", 1: r"\pi/4", 3: r"3\pi/4", 4: r"\pi", 5: r"5\pi/4", 7: r"7\pi/4"}[ph_frac]
    data["params"].update({
        "art_a": "an" if a == "x" else "a", "art_b": "an" if b == "x" else "a",
        "w": w, "a": a, "b": b, "sJ": sJ, "J0": J0, "J0_str": num(J0), "f_MHz": f_MHz, "n": n, "eps_r": eps_r,
        "medium": medium, "J_tex": J_tex, "T_str": num(T_ns), "lam_str": exact(lam), "vp8_str": num(vp8),
        "beta_str": num(beta), "eta_str": num(eta), "n_str": num(n), "sw": sw, "side": side, "k": k, "m": m,
        "w1_str": w1_s, "t1_str": t1_s, "mw1_str": exact(-w1v), "absw1_str": exact(abs(w1v)),
        "wt_str": num(2 * math.pi * f_MHz * 1e6 * t1v * 1e-9), "bw_str": num(beta * abs(w1v)),
        "ph_tex": ph_tex, "cph_str": num(cph), "Ea_str": num(Ea), "Hb_str": num(Hb), "Sw_str": num(Sw),
        "Js_now_str": num(sJ * J0 * cph), "nhat": away_here, "nhat_other": away_there,
        "side_tex": w + (r" \gt 0" if sw > 0 else r" \lt 0"),
        "aw_cross": uhat(b, axw), "Jdir": uhat(a, sJ),
        "s_choices": s_choices, "parity": parity, "facts": fact_list,
    })
    data["correct_answers"]["vp8"] = float(vp8)
    data["correct_answers"]["beta"] = float(beta)
    data["correct_answers"]["lam"] = float(lam)
    data["correct_answers"]["Ea"] = float(Ea)
    data["correct_answers"]["Hb"] = float(Hb)
    data["correct_answers"]["Sw"] = float(Sw)
