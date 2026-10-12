"""Pulling the plates of an air capacitor apart: charge fixed (isolated) or voltage fixed (battery) -- Lecture 10.

Plates of area A, gap d1 -> d2 = k d1 (k > 1), charged to V0. C(x) = eps0 A / x.
    fixed Q (battery disconnected):  Q = C1 V0;  W(x) = Q^2 x / (2 eps0 A) grows;  E = Q/(eps0 A) unchanged;
        W_agent = W2 - W1 = W1 (k - 1);  attractive force F = Q^2/(2 eps0 A) = W1/d1, the same at d1 and d2.
    fixed V (battery connected):     W(x) = eps0 A V0^2 / (2x) falls;  Q2 = C2 V0 < Q1, so -dQ flows back into the battery;
        battery work V0 dQ = V0^2 (C2 - C1) < 0 (it absorbs V0^2 (C1 - C2));
        W_agent = dW_stored - V0 dQ = 1/2 V0^2 (C1 - C2) = -1/2 V0^2 dC > 0;
        F(x) = eps0 A V0^2 / (2 x^2):  F(d1) = W1/d1,  F(d2) = F(d1)/k^2.
Units: pF, uJ, mN.
"""
import random

EPS0 = 8.854e-12


def num(v, sig=4):
    if v == 0:
        return "0"
    s = f"{v:.{sig}g}"
    if "e" in s:
        m, ex = s.split("e")
        return m + r"\times10^{" + str(int(ex)) + "}"
    return s


def generate(data):
    mode = random.choice(["Q", "V"])
    A_cm2 = random.choice([100, 120, 150, 200, 250, 300])
    d1 = random.choice([0.5, 0.8, 1.0, 1.2, 1.5])            # mm
    k = random.choice([1.5, 2, 2.5, 3, 4])
    d2 = round(k * d1, 6)                                    # mm
    V0 = random.choice([100, 150, 200, 250, 300, 400, 500])  # V  (E0 = V0/d1 <= 1 MV/m, well below air breakdown)

    A = A_cm2 * 1e-4
    C1 = EPS0 * A / (d1 * 1e-3)
    C2 = EPS0 * A / (d2 * 1e-3)
    Q1 = C1 * V0
    W1 = 0.5 * C1 * V0 ** 2
    F1 = Q1 ** 2 / (2 * EPS0 * A)
    if mode == "Q":
        Q2, V2 = Q1, Q1 / C2
        W2 = Q1 ** 2 / (2 * C2)
        W_batt = 0.0
        F2 = Q2 ** 2 / (2 * EPS0 * A)
    else:
        Q2, V2 = C2 * V0, V0
        W2 = 0.5 * C2 * V0 ** 2
        W_batt = V0 * (Q2 - Q1)
        F2 = Q2 ** 2 / (2 * EPS0 * A)
    W_agent = (W2 - W1) - W_batt                             # energy conservation: W_agent + W_batt = dW

    # ---------------------------------------------------------------- (e) energy bookkeeping, MC (mode-specific)
    if mode == "Q":
        energy_choices = [
            {"text": r"All of your work is stored: the gap grows while $E$, and so the energy density $\tfrac12\epsilon_0E^2$, "
                     r"stays the same, so the stored energy rises by exactly the work you do.",
             "correct": "true", "feedback": r"Right: with $Q$ fixed, $W = Q^2x/(2\epsilon_0A)$ grows linearly with the gap $x$."},
            {"text": r"The stored energy falls, because $C$ falls and $W = \tfrac12CV^2$; your work and the released field energy both end up as heat.",
             "correct": "false",
             "feedback": r"$\tfrac12CV^2$ with $V$ held at $V_0$ is the battery-connected case. Isolated plates keep $Q$, so $V = Q/C$ rises and $W = Q^2/(2C)$ grows."},
            {"text": r"Half of your work is stored in the field and the other half is dissipated, as when a battery charges a capacitor.",
             "correct": "false",
             "feedback": r"The half-and-half split belongs to charging from a fixed-voltage source through a resistance. No charge moves anywhere here, so nothing is dissipated."},
            {"text": r"None of it is stored: $Q$ does not change, so the stored energy does not change either.",
             "correct": "false",
             "feedback": r"$W = Q^2/(2C)$ depends on $C$ as well as $Q$; with $Q$ fixed and $C$ falling, $W$ grows by your work."},
        ]
    else:
        energy_choices = [
            {"text": r"The stored energy falls while charge flows back into the battery, which absorbs twice that drop: "
                     r"one half comes from the field, the other half is your work.",
             "correct": "true",
             "feedback": r"Right: $V_0\,\Delta Q = V_0^2\Delta C = 2\,\Delta W$, so $W_{\text{you}} = \Delta W - V_0\Delta Q = -\Delta W \gt 0$."},
            {"text": r"Your work is negative: the stored energy falls, so the plates push apart by themselves and you only hold them back.",
             "correct": "false",
             "feedback": r"The plates carry opposite charges and always attract. The field does release energy, but the battery absorbs twice as much, and you supply the difference."},
            {"text": r"The battery supplies energy to hold the voltage at $V_0$; that energy, plus your work, ends up stored in the field.",
             "correct": "false",
             "feedback": r"$Q = CV_0$ decreases as $C$ falls, so charge flows back into the battery: it is being charged, not discharged."},
            {"text": r"The battery does no work because its voltage stays constant, so your work is just the change in stored energy, $W_2 - W_1$.",
             "correct": "false",
             "feedback": r"A source at fixed $V_0$ does work $V_0\,\Delta Q$ whenever charge passes through it, and here $\Delta Q = V_0\,\Delta C \neq 0$. "
                         r"(Without it, your work would be $W_2 - W_1 \lt 0$, although the plates attract.)"},
        ]
    random.shuffle(energy_choices)

    # ---------------------------------------------------------------- (f) select-all pool, truths from the computed states
    def st(sid, text, truth, why):
        return {"id": sid, "text": text, "correct": "true" if truth else "false",
                "feedback": ("True: " if truth else "False: ") + why}

    same_E = mode == "Q"
    pool = [
        st("sameE", r"The field $E$ in the gap is the same at $d_2$ as at $d_1$.", same_E,
           r"with $Q$ fixed, $E = Q/(\epsilon_0A)$ does not depend on the gap." if same_E else
           r"with $V_0$ fixed, $E = V_0/x$ falls as the gap $x$ grows."),
        st("sameQ", r"The charge on each plate is the same at $d_2$ as at $d_1$.", mode == "Q",
           r"the isolated plates cannot gain or lose charge." if mode == "Q" else
           r"$Q = C V_0$ falls with $C$; the difference returns to the battery."),
        st("sameV", r"The voltage between the plates is the same at $d_2$ as at $d_1$.", mode == "V",
           r"the battery holds it at $V_0$." if mode == "V" else
           r"$V = Q/C$ grows in proportion to the gap."),
        st("Wup", r"The stored energy is larger at $d_2$ than at $d_1$.", W2 > W1,
           r"$W = Q^2x/(2\epsilon_0A)$ grows with the gap at fixed $Q$." if mode == "Q" else
           r"$W = \epsilon_0AV_0^2/(2x)$ falls as the gap grows at fixed $V_0$."),
        st("density", r"The energy density in the gap is the same at $d_2$ as at $d_1$.", same_E,
           r"$\tfrac12\epsilon_0E^2$ with $E$ unchanged." if same_E else
           r"$\tfrac12\epsilon_0E^2$ falls with $E = V_0/x$."),
        st("backflow", r"Charge flows from the plates back into the battery during the separation.", mode == "V",
           r"$Q = CV_0$ falls, and the battery takes the difference." if mode == "V" else
           r"the battery was disconnected before the plates moved; no charge can leave them."),
        st("positive", r"You must do positive work to pull the plates apart.", W_agent > 0,
           r"the plates attract, so the pull is along the displacement."),
        st("Cdown", r"Pulling the plates apart lowers the capacitance.", C2 < C1, r"$C = \epsilon_0A/x$."),
        st("sameforce", r"At the initial gap $d_1$, the force between the plates is the same whether or not the battery is connected.", True,
           r"at $d_1$ the charges and the field are the same in both cases, and the force depends only on them."),
        st("fullfield", r"The force on one plate equals its charge times the full field in the gap, $QE$.", False,
           r"a plate is pushed only by the other plate's field, $E/2$, so $F = \tfrac12QE$."),
        st("growforce", r"The attractive force grows as the plates move apart.", F2 > F1,
           r"it stays constant at fixed $Q$ and falls as $1/x^2$ at fixed $V_0$; it never grows."),
        st("doubleV", r"Charging to $2V_0$ instead (same plates) would double the force at $d_1$.", False,
           r"$F = \epsilon_0AV_0^2/(2d_1^2)$ grows as $V_0^2$: four times, not twice."),
        st("workisdW", r"The work you do equals the change in stored energy, $W_2 - W_1$.", abs(W_agent - (W2 - W1)) <= 1e-9 * W1,
           r"with no battery attached, energy conservation has only these two terms." if mode == "Q" else
           r"$W_2 - W_1 \lt 0$ while your work is positive: the battery's work $V_0\,\Delta Q$ also enters, and "
           r"$W_{\text{you}} = -(W_2 - W_1)$ here."),
    ]

    # ---------------------------------------------------------------- statement and answer-panel strings
    if mode == "Q":
        scenario = (r"The battery is then <b>disconnected</b>, leaving the plates isolated, and only after that does "
                    r"an external agent (you) slowly pull the plates apart, from $d_1$ to $d_2$.")
        b_expl = (r"With $Q = C_1V_0 = " + num(Q1 * 1e9) + r"\ \mathrm{nC}$ fixed: $W_1 = \tfrac12C_1V_0^2 = " + num(W1 * 1e6) +
                  r"\ \mu\mathrm{J}$ and $W_2 = \dfrac{Q^2}{2C_2} = " + num(W2 * 1e6) + r"\ \mu\mathrm{J}$. "
                  r"The voltage rose to $V_2 = Q/C_2 = " + num(V2) + r"\ \mathrm{V}$, and the energy grew by the factor $d_2/d_1 = " + num(k) + "$.")
        c_expl = (r"No battery is attached, so energy conservation is just $W_{\text{you}} = \Delta W = W_2 - W_1 = " + num(W_agent * 1e6) +
                  r"\ \mu\mathrm{J}$. Check: the force is constant, $F = " + num(F1 * 1e3) + r"\ \mathrm{mN}$, over a distance $d_2 - d_1 = " +
                  num(d2 - d1) + r"\ \mathrm{mm}$, and $F(d_2 - d_1) = " + num(F1 * (d2 - d1) * 1e-3 * 1e6) + r"\ \mu\mathrm{J}$.")
        d_expl = (r"$F = \dfrac{Q^2}{2\epsilon_0A} = " + num(F1 * 1e3) + r"\ \mathrm{mN}$ at <em>both</em> gaps: $Q$, hence $E$, does not change, "
                  r"and each plate sits in the field $E/2$ of the other. From energy: $F = -\partial W/\partial x|_Q = -Q^2/(2\epsilon_0A)$ (attractive).")
        e_expl = (r"Everything you put in stays in the field: the volume of the gap grows at constant energy density $\tfrac12\epsilon_0E^2$. "
                  r"Nothing is dissipated, because no charge moves through any resistance.")
    else:
        scenario = (r"The battery stays <b>connected</b> throughout, holding the voltage between the plates at $V_0$, "
                    r"while an external agent (you) slowly pulls the plates apart, from $d_1$ to $d_2$.")
        b_expl = (r"With $V = V_0$ fixed: $W_1 = \tfrac12C_1V_0^2 = " + num(W1 * 1e6) + r"\ \mu\mathrm{J}$ and $W_2 = \tfrac12C_2V_0^2 = " +
                  num(W2 * 1e6) + r"\ \mu\mathrm{J}$: the stored energy <em>falls</em>, by the factor $d_1/d_2$.")
        c_expl = (r"The plate charge drops from $Q_1 = C_1V_0 = " + num(Q1 * 1e9) + r"\ \mathrm{nC}$ to $Q_2 = C_2V_0 = " + num(Q2 * 1e9) +
                  r"\ \mathrm{nC}$; the difference flows back into the battery, which does work $V_0\,\Delta Q = " + num(W_batt * 1e6) +
                  r"\ \mu\mathrm{J}$ (it absorbs $" + num(-W_batt * 1e6) + r"\ \mu\mathrm{J}$). Energy conservation, "
                  r"$W_{\text{you}} + V_0\Delta Q = \Delta W$, gives $W_{\text{you}} = \Delta W - V_0\Delta Q = -\tfrac12V_0^2\Delta C = " +
                  num(W_agent * 1e6) + r"\ \mu\mathrm{J} \gt 0$. Check: $\displaystyle\int_{d_1}^{d_2}\frac{\epsilon_0AV_0^2}{2x^2}\,dx = "
                  r"\tfrac12V_0^2(C_1 - C_2)$, the same number.")
        d_expl = (r"$F(x) = \dfrac{\epsilon_0AV_0^2}{2x^2}$: $F(d_1) = " + num(F1 * 1e3) + r"\ \mathrm{mN}$ and $F(d_2) = F(d_1)\,(d_1/d_2)^2 = " +
                  num(F2 * 1e3) + r"\ \mathrm{mN}$. At $d_1$ the state is the same as for isolated plates, so the force is too; "
                  r"it weakens as charge drains back into the battery. (From energy at fixed $V$: $F = +\partial W/\partial x|_V$, with the battery's share included.)")
        e_expl = (r"The battery absorbs $V_0^2|\Delta C| = " + num(-W_batt * 1e6) + r"\ \mu\mathrm{J}$: the drop in stored energy, $" +
                  num((W1 - W2) * 1e6) + r"\ \mu\mathrm{J}$, plus your work, $" + num(W_agent * 1e6) + r"\ \mu\mathrm{J}$ — equal shares, since $V_0\Delta Q = 2\Delta W$.")

    data["params"].update({
        "mode": mode, "A_cm2": A_cm2, "d1": d1, "d2": d2, "k": k, "V0": V0,
        "d1_str": num(d1), "d2_str": num(d2), "scenario": scenario,
        "C1_str": num(C1 * 1e12), "C2_str": num(C2 * 1e12), "W1_str": num(W1 * 1e6), "W2_str": num(W2 * 1e6),
        "b_expl": b_expl, "c_expl": c_expl, "d_expl": d_expl, "e_expl": e_expl,
        "energy_choices": energy_choices, "pool": pool,
    })
    data["correct_answers"]["C1"] = float(C1 * 1e12)          # pF
    data["correct_answers"]["C2"] = float(C2 * 1e12)          # pF
    data["correct_answers"]["W1"] = float(W1 * 1e6)           # uJ
    data["correct_answers"]["W2"] = float(W2 * 1e6)           # uJ
    data["correct_answers"]["Wagent"] = float(W_agent * 1e6)  # uJ
    data["correct_answers"]["F1"] = float(F1 * 1e3)           # mN
    data["correct_answers"]["F2"] = float(F2 * 1e3)           # mN
