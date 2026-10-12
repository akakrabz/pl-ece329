"""Mutual inductance of a pick-up coil inside a long solenoid, and an RL switch-on (Lecture 15).

Ideal solenoid: n1 turns/m, radius a1, length l (l >= 10 a1); field mu0 n1 I1 uniform inside.
Pick-up coil: N2 turns, radius a2 < a1, coaxial, far from the ends.
    M = N2 Psi21 / I1 = mu0 n1 N2 pi a2^2                       [uH]
    |emf2| = M |dI1/dt|                                          [mV]
    reciprocity: flux linked by the solenoid's turns per ampere in the coil = M
RL: the solenoid (L1 = mu0 n1^2 pi a1^2 l) in series with R, switched onto V0 at t = 0, I1(0) = 0:
    I1(t) = (V0/R)(1 - exp(-t/tau)),  tau = L1/R,  W_inf = L1 (V0/R)^2 / 2
    t = 0+: I1 = 0, L1 dI1/dt = V0, |emf2| = M V0/L1 (largest);  t >> tau: I1 = V0/R, no voltage on L1, emf2 = 0.
"""
import math
import random

MU0 = 4e-7 * math.pi


def generate(data):
    p = data["params"]
    while True:
        n1 = random.choice([500, 600, 800, 1000, 1200, 1500, 2000])
        a1_cm = random.choice([1.5, 2, 2.5, 3, 4])
        ell_cm = random.choice([30, 40, 50, 60, 80])
        N2 = random.choice([10, 20, 25, 40, 50, 80, 100, 150, 200])
        a2_cm = random.choice([0.5, 0.6, 0.8, 1.0, 1.2, 1.5])
        rate = random.choice([10, 20, 50, 100, 200, 500])
        V0 = random.choice([1.5, 3, 4.5, 6, 9, 12])
        R = random.choice([0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12])
        if ell_cm < 10 * a1_cm or a2_cm > 0.75 * a1_cm:
            continue
        a1, a2, ell = a1_cm * 1e-2, a2_cm * 1e-2, ell_cm * 1e-2
        N1 = round(n1 * ell)
        M = MU0 * n1 * N2 * math.pi * a2 ** 2
        L1 = MU0 * n1 ** 2 * math.pi * a1 ** 2 * ell
        Iinf = V0 / R
        tau = L1 / R
        emf2 = M * rate
        W = 0.5 * L1 * Iinf ** 2
        if not (0.2 <= Iinf <= 5 and 5e-5 <= tau <= 0.05 and 0.05e-3 <= emf2 <= 0.2 and 1e-5 <= W <= 0.2):
            continue
        r_area, r_turns = (a1 / a2) ** 2, N1 / N2
        if min(abs(r_area - 1), abs(r_turns - 1), abs(r_area / r_turns - 1)) < 0.05:
            continue                       # keep the reciprocity distractors distinct from M and from each other
        break
    increasing = random.choice([True, False])
    k = random.choice([0.5, 0.75, 1, 1.5, 2, 2.5, 3])
    t1_ms = float(f"{k * tau * 1e3:.2g}")
    t1 = t1_ms * 1e-3
    I_t1 = Iinf * (1 - math.exp(-t1 / tau))
    emf0 = M * V0 / L1                      # |emf2| at t = 0+

    def f4(x):
        return f"{x:.4g}"

    # ---------------- (b) reciprocity MC
    rec = [
        {"key": "same", "correct": "true", "val": M,
         "text": r"$M$, the same value as in (a)",
         "feedback": (r"Right: mutual inductance is reciprocal, $M_{12} = M_{21}$, for any two circuits, so the "
                      r"hard field calculation is unnecessary.")},
        {"key": "area", "correct": "false", "val": M * r_area,
         "text": (r"$M(a_1/a_2)^2$: each solenoid turn encloses the larger area $\pi a_1^2$, so it catches more of "
                  r"the coil's field"),
         "feedback": (r"The coil's flux that goes up inside its own turns comes back down outside them, partly inside "
                      r"radius $a_1$; a larger loop does not catch more net flux. Reciprocity gives exactly $M$.")},
        {"key": "turns", "correct": "false", "val": M * r_turns,
         "text": (r"$MN_1/N_2$, with $N_1 = n_1\ell = " + f"{N1}" +
                  r"$: the coil's flux now links the solenoid's $N_1$ turns instead of the coil's $N_2$ turns"),
         "feedback": (r"Counting the solenoid's turns is right, but each of them links only part of the short coil's "
                      r"spreading flux, less and less the farther the turn is from the coil; summed over all of "
                      r"them, the linkage per ampere is exactly $M$.")},
        {"key": "zero", "correct": "false", "val": 0.0,
         "text": (r"zero: the short coil's field stays inside the coil, so none of it threads the solenoid's turns"),
         "feedback": (r"A short coil is not an ideal solenoid: its field spreads out beyond its ends and threads the "
                      r"surrounding solenoid turns. By reciprocity the total linkage per ampere is $M$.")},
        {"key": "unknown", "correct": "false", "val": None,
         "text": (r"it cannot be found without working out the short coil's complicated field, including the flux "
                  r"that returns outside it"),
         "feedback": (r"Reciprocity, $M_{12} = M_{21}$, holds for any pair of circuits, so that calculation is not "
                      r"needed: the answer is $M$.")},
    ]

    # ---------------- (d) t = 0+ or t >> tau
    moment = random.choice(["start", "late"])
    start_right = moment == "start"
    st = [
        {"key": "start_state", "correct": "true" if start_right else "false", "val": emf0,
         "text": (r"$I_1 = 0$; the whole battery voltage appears across the solenoid's inductance, "
                  r"$L_1\,dI_1/dt = V_0$; and the pick-up coil's emf has its largest magnitude, $MV_0/L_1$"),
         "feedback": (r"Right: an inductor's current cannot jump, so $I_1(0^+) = 0$, all of $V_0$ appears across "
                      r"$L_1$, and $dI_1/dt = V_0/L_1$ is at its largest, and so is the pick-up emf "
                      r"$M\,dI_1/dt$." if start_right else
                      r"That describes $t = 0^+$. Long after, $dI_1/dt \to 0$: the inductance has no voltage across "
                      r"it, the current is $V_0/R$, and the pick-up emf $-M\,dI_1/dt$ has died away.")},
        {"key": "late_state", "correct": "false" if start_right else "true", "val": Iinf,
         "text": (r"$I_1 = V_0/R = " + f4(Iinf) + r"\ \mathrm{A}$; there is no voltage across the solenoid's "
                  r"inductance; and the pick-up coil's emf is zero even though the flux through it is at its largest"),
         "feedback": (r"That is the final state. Just after closing the switch the current has had no time to grow: "
                      r"an inductor's current cannot jump." if start_right else
                      r"Right: the current has settled at $V_0/R$, so $L_1\,dI_1/dt = 0$, and the emf "
                      r"$-M\,dI_1/dt$ is zero although the linked flux is at its largest.")},
        {"key": "jump", "correct": "false",
         "text": (r"$I_1 = V_0/R = " + f4(Iinf) + r"\ \mathrm{A}$ and the voltage across the solenoid's inductance "
                  r"is $V_0$"),
         "feedback": (r"These two cannot hold together: KVL, $V_0 = RI_1 + L_1\,dI_1/dt$, leaves no voltage for the "
                      r"inductance when $I_1 = V_0/R$. And the current through an inductance cannot jump from 0 to "
                      r"$V_0/R$; it rises with time constant $\tau$.")},
        {"key": "nochange", "correct": "false",
         "text": (r"$I_1 = 0$ and $dI_1/dt = 0$, so the pick-up coil's emf is zero: no flux links it yet"),
         "feedback": (r"At $t = 0^+$ the current is zero but its rate of change is $V_0/L_1$, not zero, and the "
                      r"pick-up emf depends on $dI_1/dt$, not on $I_1$. Long after, $I_1 = V_0/R \ne 0$.")},
        {"key": "fluxmax", "correct": "false",
         "text": (r"$I_1 = V_0/R = " + f4(Iinf) + r"\ \mathrm{A}$, and the pick-up coil's emf is at its largest, "
                  r"because the flux through the coil is at its largest"),
         "feedback": (r"The induced emf is $-M\,dI_1/dt$: it is set by how fast the flux changes, not by how large "
                      r"it is. When the flux is largest it is no longer changing, and the emf is zero.")},
    ]

    # ---------------- (e) checkbox
    pool = [
        ("m_N2", True, r"$M$ is proportional to the number of turns $N_2$ of the pick-up coil.",
         r"Each of the $N_2$ turns links the same flux $\mu_0n_1I_1\pi a_2^2$, so the linkage $N_2\Psi_{21}$ is "
         r"proportional to $N_2$."),
        ("m_shift", True,
         r"Shifting the pick-up coil sideways inside the solenoid (still parallel to the axis and entirely inside it) "
         r"would not change $M$.",
         r"The ideal solenoid's field is uniform over its whole cross-section, so each turn of the coil links the same "
         r"flux wherever the coil sits inside."),
        ("m_I", True, r"$M$ does not depend on the current $I_1$.",
         r"The field, and so the linked flux, is proportional to $I_1$; their ratio depends on the geometry only."),
        ("scale_n1", True,
         r"Doubling $n_1$ (same $a_1$ and $\ell$) multiplies $L_1$ by 4 but $M$ only by 2.",
         r"$L_1 = \mu_0n_1^2\pi a_1^2\ell$ (the field and the number of linking turns both double), while "
         r"$M = \mu_0n_1N_2\pi a_2^2$ (only the field doubles)."),
        ("steady_zero", True,
         r"With the pick-up coil open-circuited, its emf is zero whenever $I_1$ is steady, however large $I_1$ is.",
         r"$\mathcal{E}_2 = -M\,dI_1/dt$."),
        ("tau63", True,
         r"At $t = \tau$ the solenoid current has reached $1 - 1/e \approx 63\%$ of its final value.",
         r"$I_1(\tau) = (V_0/R)(1 - e^{-1}) = 0.632\,V_0/R$."),
        ("energy_field", True,
         r"The final stored energy $\tfrac12L_1I_\infty^2$ equals $\tfrac12\mu_0H^2$ times the volume inside the "
         r"solenoid.",
         r"$\tfrac12\mu_0(n_1I)^2\,\pi a_1^2\ell = \tfrac12(\mu_0n_1^2\pi a_1^2\ell)I^2$: the energy lives in the "
         r"field."),
        ("m_area1", False, r"$M$ is proportional to the solenoid's cross-sectional area $\pi a_1^2$.",
         r"Only the flux through the coil's own turns counts, $M = \mu_0n_1N_2\pi a_2^2$; the solenoid's field "
         r"outside radius $a_2$ never passes through the coil."),
        ("flux_emf", False,
         r"During the switch-on of part (c), the emf in the pick-up coil is largest when the flux through it is "
         r"largest.",
         r"The emf is $-M\,dI_1/dt$: largest at $t = 0^+$, when the flux is still zero, and zero once the flux has "
         r"settled at its largest value."),
        ("tau_V0", False, r"Doubling the battery voltage $V_0$ would double the time constant.",
         r"$\tau = L_1/R$ does not involve $V_0$; a larger $V_0$ only scales the current."),
        ("more_R_slower", False,
         r"A larger series resistance $R$ would make the current approach its final value more slowly.",
         r"$\tau = L_1/R$: a larger $R$ gives a shorter time constant (and a smaller final current)."),
        ("energy_half", False,
         r"At $t = \tau$ the stored energy is already more than half of its final value.",
         r"$W \propto I_1^2$, so $W(\tau)/W_\infty = (1 - 1/e)^2 \approx 0.40$."),
        ("energy_zero", False,
         r"Once the current has stopped changing, the energy stored in the solenoid's magnetic field is zero.",
         r"The energy is $\tfrac12L_1I_1^2 = W_\infty \gt 0$; it is the induced emf that vanishes when the current "
         r"is steady."),
    ]
    while True:
        sel = random.sample(pool, 5)
        nt = sum(1 for x in sel if x[1])
        if 1 <= nt <= 4:
            break
    p["facts"] = [{"key": kk, "correct": "true" if t else "false", "verdict": "True" if t else "False",
                   "text": txt, "why": why} for kk, t, txt, why in sel]

    # ---------------- answer-panel text
    a_words = (
        r"Inside the ideal solenoid $\mathbf{H} = n_1I_1$ along the axis, uniform, so each turn of the coil links "
        r"$\Psi_{21} = \mu_0n_1I_1\,\pi a_2^2$ (the coil's own area, not the solenoid's). Then"
        r"$$M = \frac{N_2\Psi_{21}}{I_1} = \mu_0n_1N_2\pi a_2^2 = (4\pi\times10^{-7})(" + f"{n1}" + r")(" +
        f"{N2}" + r")\,\pi\,(" + f"{a2:g}" + r")^2 = " + f4(M * 1e6) + r"\ \mu\mathrm{H}.$$"
        r"The open coil's emf is $\mathcal{E}_2 = -N_2\,d\Psi_{21}/dt = -M\,dI_1/dt$, so $|\mathcal{E}_2| = (" +
        f4(M * 1e6) + r"\ \mu\mathrm{H})(" + f"{rate}" + r"\ \mathrm{A/s}) = " + f4(emf2 * 1e3) +
        r"\ \mathrm{mV}$, constant while the rate is constant."
    )
    b_words = (
        r"Mutual inductance is reciprocal: the flux linked by circuit 1 per ampere in circuit 2 equals the flux "
        r"linked by circuit 2 per ampere in circuit 1 ($M_{12} = M_{21}$; the double-integral formula for $M$ is "
        r"symmetric in the two circuits). So the answer is $M = " + f4(M * 1e6) + r"\ \mu\mathrm{H}$. A direct "
        r"calculation would be hard: each solenoid turn links a different part of the short coil's spreading field "
        r"(more near the coil, less far from it), and only the sum over all $N_1 = " + f"{N1}" + r"$ turns comes "
        r"out to $M$. Neither the area ratio $(a_1/a_2)^2$ nor the turn ratio $N_1/N_2$ enters, and the answer is "
        r"certainly not zero."
    )
    c_words = (
        r"KVL around the circuit: $V_0 = RI_1 + L_1\,dI_1/dt$ with $I_1(0) = 0$, so"
        r"$$I_1(t) = \frac{V_0}{R}\left(1 - e^{-t/\tau}\right),\qquad \tau = \frac{L_1}{R} = \frac{" + f4(L1 * 1e3) +
        r"\ \mathrm{mH}}{" + f"{R:g}" + r"\ \Omega} = " + f4(tau * 1e3) + r"\ \mathrm{ms}.$$"
        r"At $t_1 = " + f"{t1_ms:g}" + r"\ \mathrm{ms}$: $I_1 = " + f4(Iinf) + r"\,(1 - e^{-" + f4(t1 / tau) +
        r"}) = " + f4(I_t1) + r"\ \mathrm{A}$. Once the current has settled at $I_\infty = V_0/R = " + f4(Iinf) +
        r"\ \mathrm{A}$, the stored energy is $W = \tfrac12L_1I_\infty^2 = \tfrac12(" + f4(L1 * 1e3) +
        r"\ \mathrm{mH})(" + f4(Iinf) + r"\ \mathrm{A})^2 = " + f4(W * 1e3) + r"\ \mathrm{mJ}$."
    )
    if start_right:
        d_words = (r"Just after closing the switch the current is still 0 (an inductor's current cannot jump), so KVL "
                   r"puts all of $V_0$ across the inductance: $dI_1/dt = V_0/L_1$, the largest it will ever be. The "
                   r"pick-up emf $M\,dI_1/dt = MV_0/L_1 = " + f4(emf0 * 1e3) + r"\ \mathrm{mV}$ is therefore largest "
                   r"at the very moment the flux through the coil is zero. Choices with $I_1 = V_0/R$ describe the "
                   r"final state, and $dI_1/dt = 0$ is wrong at $t = 0^+$.")
    else:
        d_words = (r"Long after closing the switch, $dI_1/dt \to 0$: the inductance has no voltage across it (it acts "
                   r"like a plain wire), $I_1 = V_0/R = " + f4(Iinf) + r"\ \mathrm{A}$, and the pick-up emf "
                   r"$-M\,dI_1/dt$ is zero even though the linked flux is now at its largest. The emf measures the "
                   r"rate of change of flux, not the flux.")

    p.update({
        "n1": n1, "a1_cm": f"{a1_cm:g}", "ell_cm": ell_cm, "N2": N2, "a2_cm": f"{a2_cm:g}", "rate": rate,
        "rate_word": "increasing" if increasing else "decreasing", "V0": f"{V0:g}", "R": f"{R:g}",
        "L1_mH": f4(L1 * 1e3), "N1": N1, "t1_ms": f"{t1_ms:g}", "moment": moment,
        "moment_text": (r"just after the switch closes ($t = 0^+$)" if start_right else
                        r"long after the switch closes ($t \gg \tau$)"),
        "rec_choices": rec, "state_choices": st,
        "a_words": a_words, "b_words": b_words, "c_words": c_words, "d_words": d_words,
    })
    data["correct_answers"]["M"] = M * 1e6          # uH
    data["correct_answers"]["emf"] = emf2 * 1e3     # mV
    data["correct_answers"]["tau"] = tau * 1e3      # ms
    data["correct_answers"]["I1"] = I_t1            # A
    data["correct_answers"]["W"] = W * 1e3          # mJ
