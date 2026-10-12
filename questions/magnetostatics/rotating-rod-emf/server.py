"""Motional emf of a rod rotating about one end (Lecture 14).

Rod of length L in the plane perpendicular to axis e (random x/y/z), pivot at the origin,
angular velocity w = sw * omega * e, uniform static field B = sb * B0 * e.
    v = w x r,  v x B = (w x r) x B = r (w . B) - w (r . B) = (w . B) r       (r perpendicular to e, B along e)
    emf pivot -> tip = int_0^L (w . B) s ds = (1/2) (w . B) L^2 = sw * sb * (1/2) B0 omega L^2
    open circuit: E = -(v x B) inside, so V_tip - V_pivot = emf(pivot -> tip): the tip is + iff sw * sb > 0.
With R between pivot and rail (rod, rail, leads ideal; self-inductance neglected):
    I = |emf| / R,  P_mech = |emf| I = I^2 R,  braking torque (1/2) I B0 L^2 = P / omega.
"""
import math
import random

PLANE = {"x": "yz", "y": "zx", "z": "xy"}


def generate(data):
    p = data["params"]
    while True:
        L_cm = random.choice([10, 12, 15, 20, 25, 30, 40])
        B_mT = random.choice([20, 40, 50, 60, 80, 100, 150, 200, 250])
        omega = random.choice([20, 25, 30, 40, 50, 60, 75, 80, 100, 120])
        R = random.choice([0.5, 1, 2, 2.5, 4, 5, 10])
        L, B0 = L_cm * 1e-2, B_mT * 1e-3
        emf = 0.5 * B0 * omega * L ** 2                 # V, magnitude
        I = emf / R
        if 0.01 <= emf <= 0.5 and 0.002 <= I <= 1.0:
            break
    axis = random.choice(["x", "y", "z"])
    sw, sb = random.choice([1, -1]), random.choice([1, -1])
    tip_plus = sw * sb > 0
    P = emf * I                                         # W
    torque = P / omega                                  # N m

    sense = "counterclockwise" if sw > 0 else "clockwise"
    w_tex = ("" if sw > 0 else "-") + f"{omega:g}" + r"\,\hat{" + axis + "}"
    B_tex = ("" if sb > 0 else "-") + f"{B_mT:g}" + r"\,\hat{" + axis + "}"
    wdotB = sw * sb * omega * B0
    hi, lo = ("tip", "pivot") if tip_plus else ("pivot", "tip")
    outward = "outward (from the pivot toward the tip)" if tip_plus else "inward (from the tip toward the pivot)"

    # ---------------- (b) MC: which end is at the higher potential
    vxb_words = (r"Here $\boldsymbol\omega\cdot\mathbf{B} = (" + f"{sw * omega:g}" + r")(" + f"{sb * B0:g}" +
                 r") = " + f"{wdotB:g}" + r"\ \mathrm{T/s}$, so $\mathbf{v}\times\mathbf{B}$ points " + outward +
                 r" and pushes positive charge to the " + hi + ".")
    choices = [
        {"key": "tip", "correct": "true" if tip_plus else "false", "text": "the tip (the free end)",
         "feedback": (r"Right. " + vxb_words) if tip_plus else vxb_words},
        {"key": "pivot", "correct": "false" if tip_plus else "true", "text": "the pivot",
         "feedback": vxb_words if tip_plus else (r"Right. " + vxb_words)},
        {"key": "none", "correct": "false",
         "text": "neither: the field is uniform and static, so no emf is induced",
         "feedback": (r"A uniform static field induces nothing in a loop that translates rigidly, but this rod "
                      r"rotates: its charges move across $\mathbf{B}$ and feel $q\mathbf{v}\times\mathbf{B}$ along "
                      r"the rod, a motional emf.")},
        {"key": "alternate", "correct": "false",
         "text": "the tip during half of each revolution and the pivot during the other half",
         "feedback": (r"$\mathbf{v}\times\mathbf{B} = (\boldsymbol\omega\cdot\mathbf{B})\,\mathbf{r}$ points along "
                      r"the rod with the same sense at every angle, so the polarity never changes: a DC "
                      r"(homopolar) generator.")},
        {"key": "carriers", "correct": "false",
         "text": "it depends on whether the mobile charges in copper are negative or positive",
         "feedback": (r"Every charge moving with the rod feels the same force per unit charge, "
                      r"$\mathbf{v}\times\mathbf{B}$. The carriers settle where the electrostatic field cancels it, "
                      r"$\mathbf{E} = -\mathbf{v}\times\mathbf{B}$, whatever their sign, so the polarity does not "
                      r"depend on them.")},
    ]

    # ---------------- (d) checkbox
    if tip_plus:
        rod_why = (r"Inside the rod (the source) the current flows toward the $+$ terminal, the tip, driven by "
                   r"$\mathbf{v}\times\mathbf{B}$; it returns from the rail through $R$ to the pivot.")
    else:
        rod_why = (r"Here the pivot is the $+$ terminal, and inside the rod (the source) the current flows toward "
                   r"it, from the tip to the pivot, driven by $\mathbf{v}\times\mathbf{B}$; it returns from the pivot "
                   r"through $R$ to the rail.")
    pool = [
        ("motional", True,
         r"All of the emf is motional: $\mathbf{B}$ does not change in time, so $\oint\mathbf{E}\cdot d\mathbf{l} = 0$ "
         r"around every closed path in the lab, and the emf comes entirely from the $\mathbf{v}\times\mathbf{B}$ force "
         r"on the moving charges.",
         r"$\nabla\times\mathbf{E} = -\partial\mathbf{B}/\partial t = 0$: the lab $\mathbf{E}$ is the curl-free field "
         r"of the separated charges, and the energy source is the $\mathbf{v}\times\mathbf{B}$ term."),
        ("quadruple", True,
         r"Doubling the rod's length (same $\omega$ and $\mathbf{B}$) multiplies the emf by 4.",
         r"$|\mathcal{E}| = \tfrac12B\omega L^2 \propto L^2$: both the length and the average speed double."),
        ("tip_speed", True,
         r"Treating every point of the rod as moving at the tip speed $\omega L$ would overestimate the emf by a "
         r"factor of 2.",
         r"$(B\,\omega L)\,L = B\omega L^2$ is twice $\tfrac12B\omega L^2$: the average speed along the rod is "
         r"$\omega L/2$."),
        ("reverse", True,
         r"Reversing the sense of rotation (same $\mathbf{B}$) makes the other end of the rod the higher-potential end.",
         r"$V_{\text{tip}} - V_{\text{pivot}} = \tfrac12(\boldsymbol\omega\cdot\mathbf{B})L^2$ changes sign with "
         r"$\boldsymbol\omega$."),
        ("swept", True,
         r"The magnitude of the emf equals $|\mathbf{B}|$ times the area the rod sweeps out per unit time.",
         r"In a time $dt$ the rod sweeps a sector of area $\tfrac12L^2\omega\,dt$, and $B\,dA/dt = \tfrac12B\omega L^2$: "
         r"the flux-rule view of the same result."),
        ("axial_only", True,
         r"Adding a uniform field component lying in the plane of rotation would not change the emf between the "
         r"pivot and the tip.",
         r"For such a component $\mathbf{v}\times\mathbf{B}$ points along the rotation axis, perpendicular to the rod, "
         r"so it adds nothing to $\int(\mathbf{v}\times\mathbf{B})\cdot d\mathbf{l}$ along the rod."),
        ("open_no_torque", True,
         r"With nothing connected (open circuit, once the charges have settled), no torque is needed to keep the rod "
         r"turning at constant $\omega$, apart from overcoming friction.",
         r"No current flows, so there is no magnetic force on a current in the rod; the forces on the separated "
         r"charges point along the rod and exert no torque about the pivot."),
        ("rod_dir", tip_plus,
         r"With the resistor connected, the current inside the rod flows from the pivot toward the tip.",
         rod_why),
        ("zero_uniform", False,
         r"Because $\mathbf{B}$ is uniform and does not change in time, no emf can be induced in the rod.",
         r"That is true for a loop translating rigidly through a uniform field. This rod rotates: its charges move "
         r"across $\mathbf{B}$ at speeds that grow along the rod and feel a net $\mathbf{v}\times\mathbf{B}$ along it."),
        ("lenz_faster", False,
         r"With the resistor connected, the magnetic force on the current in the rod pushes the rod forward, so less "
         r"external torque is needed than with the circuit open.",
         r"The force $I\,d\mathbf{l}\times\mathbf{B}$ on the rod's current has a torque opposite to "
         r"$\boldsymbol\omega$ (Lenz): it brakes the rotation, and the agent must supply the power "
         r"$\mathcal{E}I$."),
        ("ac", False,
         r"The current through the resistor reverses direction twice per revolution.",
         r"$\mathbf{v}\times\mathbf{B} = (\boldsymbol\omega\cdot\mathbf{B})\,\mathbf{r}$ points the same way along the "
         r"rod at every angle: the current is steady (DC)."),
        ("more_R", False,
         r"Replacing the resistor by one with a larger resistance would increase the mechanical power needed to keep "
         r"$\omega$ constant.",
         r"$P = \mathcal{E}^2/R$: with the same emf, a larger $R$ draws less current and needs less power."),
        ("angle", False,
         r"The emf between the pivot and the tip depends on the rod's angular position.",
         r"Every position of the rod is equivalent with respect to the axial field: $\tfrac12B\omega L^2$ is constant "
         r"in time."),
    ]
    while True:
        sel = random.sample(pool, 5)
        nt = sum(1 for x in sel if x[1])
        if 1 <= nt <= 4:
            break
    p["facts"] = [{"key": k, "correct": "true" if t else "false", "verdict": "True" if t else "False",
                   "text": txt, "why": why} for k, t, txt, why in sel]

    # ---------------- answer-panel text
    a_words = (
        r"A point of the rod at distance $s$ from the pivot moves with $\mathbf{v} = \boldsymbol\omega\times"
        r"\mathbf{r}$, of size $\omega s$, perpendicular to the rod. With $\mathbf{B}$ along the rotation axis, "
        r"$\mathbf{v}\times\mathbf{B} = (\boldsymbol\omega\times\mathbf{r})\times\mathbf{B} = "
        r"(\boldsymbol\omega\cdot\mathbf{B})\,\mathbf{r}$: it points along the rod and grows linearly with $s$. "
        r"Integrating from the pivot to the tip,"
        r"$$\mathcal{E} = \int_0^L(\boldsymbol\omega\cdot\mathbf{B})\,s\,ds = \tfrac12(\boldsymbol\omega\cdot"
        r"\mathbf{B})L^2,\qquad |\mathcal{E}| = \tfrac12B\omega L^2 = \tfrac12(" + f"{B0:g}" + r")(" + f"{omega:g}" +
        r")(" + f"{L:g}" + r")^2 = " + f"{emf * 1e3:.4g}" + r"\ \mathrm{mV}.$$"
        r"(Using the tip speed $\omega L$ for the whole rod would give twice this.)"
    )
    b_words = (
        vxb_words + r" (A uniform, static field does induce an emf here, because the rod moves through it.)"
        r" In the open rod the charges settle until the electrostatic field cancels "
        r"$\mathbf{v}\times\mathbf{B}$ inside, $\mathbf{E} = -\mathbf{v}\times\mathbf{B}$. Because $\mathbf{B}$ is "
        r"static, this $\mathbf{E}$ is curl-free and the potential is well defined: $V_{\text{tip}} - "
        r"V_{\text{pivot}} = -\int_{\text{pivot}}^{\text{tip}}\mathbf{E}\cdot d\mathbf{l} = \tfrac12"
        r"(\boldsymbol\omega\cdot\mathbf{B})L^2 = " + f"{(1 if tip_plus else -1) * emf * 1e3:.4g}" +
        r"\ \mathrm{mV}$, so the <b>" + hi + r"</b> is at the higher potential. The polarity is the same at every "
        r"angle (no alternation) and does not depend on the sign of the carriers; reversing either "
        r"$\boldsymbol\omega$ or $\mathbf{B}$ would reverse it."
    )
    c_words = (
        r"The rod is a battery of emf $" + f"{emf * 1e3:.4g}" + r"\ \mathrm{mV}$ with no internal resistance, so "
        r"$I = |\mathcal{E}|/R = " + f"{emf * 1e3:.4g}" + r"\ \mathrm{mV}/" + f"{R:g}" + r"\ \Omega = " +
        f"{I * 1e3:.4g}" + r"\ \mathrm{mA}$. Inside the rod it flows from the " + lo + " to the " + hi +
        r" (toward the $+$ end, as inside a battery) and returns through the resistor. The force "
        r"$I\,d\mathbf{l}\times\mathbf{B}$ on the rod's current has a torque about the pivot of size "
        r"$\int_0^L sIB\,ds = \tfrac12IBL^2 = " + f"{torque * 1e3:.4g}" + r"\ \mathrm{mN\,m}$, opposite to "
        r"$\boldsymbol\omega$ (Lenz). Holding $\omega$ constant takes an equal applied torque, i.e. the power"
        r"$$P = \tfrac12IBL^2\,\omega = |\mathcal{E}|\,I = I^2R = " + f"{P * 1e3:.4g}" + r"\ \mathrm{mW},$$"
        r"all of which ends up as heat in $R$."
    )
    p.update({
        "axis": axis, "plane": PLANE[axis], "sw": sw, "sb": sb, "L_cm": L_cm, "B_mT": B_mT, "omega": omega,
        "R": f"{R:g}", "w_tex": w_tex, "B_tex": B_tex, "sense": sense, "tip_plus": tip_plus,
        "choices": choices, "a_words": a_words, "b_words": b_words, "c_words": c_words,
    })
    data["correct_answers"]["emf"] = emf * 1e3      # mV, magnitude
    data["correct_answers"]["I"] = I * 1e3          # mA
    data["correct_answers"]["P"] = P * 1e3          # mW
