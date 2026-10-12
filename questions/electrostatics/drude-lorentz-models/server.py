"""Drude and Lorentz models (Lecture 11), in the forms used on the course site.

Drude:   m dv/dt = qE - m v/tau  ->  steady v = (q tau/m) E;  mobility = |q| tau/m;  J = N q v = sigma E,
         sigma = N q^2 tau/m  (q = -e, m = m_e for the conduction electrons of a metal).
AC:      phasors E(t) = Re{E e^{j w t}}:  sigma(w) = sigma_DC/(1 + j w/nu), nu = 1/tau
         -> |sigma|/sigma_DC = 1/sqrt(1 + (w tau)^2), angle(sigma) = -atan(w tau)  (J lags E).
Lorentz: m r'' = -eE - m w0^2 r - 2 m alpha r'; lossless (alpha = 0) steady state
         chi_e(w) = (N_d e^2/(m eps0))/(w0^2 - w^2) = chi_e(0)/(1 - (w/w0)^2):
         > 0 below resonance (P in phase with E), < 0 above (P opposite to E).
Polarization current J_p = dP/dt.

Metals: free-electron densities N (Ashcroft & Mermin Table 1.1 / HyperPhysics) and tau = m_e/(N e^2 rho) from the
CRC resistivity at 293 K (Cu 16.78, Ag 15.87, Au 22.14, Al 26.50, Zn 59.0, Mg 43.9 nOhm m), rounded to 2 s.f.
"""
import math
import random

E_CH = 1.602e-19
M_E = 9.109e-31

METALS = [  # name, N (1e28 m^-3), tau (fs), measured sigma at 20 C (MS/m)
    ("copper", 8.47, 25, 59.6), ("silver", 5.86, 38, 63.0), ("gold", 5.90, 27, 45.2),
    ("aluminum", 18.1, 7.4, 37.7), ("zinc", 13.2, 4.6, 16.9), ("magnesium", 8.61, 9.4, 22.8),
]


def num(v, sig=4):
    if v == 0:
        return "0"
    s = f"{v:.{sig}g}"
    if "e" in s:
        m, ex = s.split("e")
        return m + r"\times10^{" + str(int(ex)) + "}"
    return s


def sci(v, sig=3):
    """v as  m\\times10^{e}  with 1 <= m < 10 (LaTeX)."""
    ex = int(math.floor(math.log10(abs(v))))
    m = v / 10 ** ex
    return f"{m:.{sig}g}" + r"\times10^{" + str(ex) + "}"


def generate(data):
    name, N28, tau_fs, sig_meas = random.choice(METALS)
    N, tau = N28 * 1e28, tau_fs * 1e-15

    sigma = N * E_CH ** 2 * tau / M_E                   # S/m
    mob = E_CH * tau / M_E                              # m^2/(V s)

    drive = random.choice(["E", "J"])
    if drive == "E":
        Ef = random.choice([0.02, 0.05, 0.1, 0.15, 0.2])        # V/m
        v = mob * Ef
        drive_text = (r"a uniform field $\mathbf{E} = " + num(Ef) + r"\,\hat{x}\ \mathrm{V/m}$ is applied along the wire")
        I_w = A_w = J = 0.0
    else:
        pairs = [(I, Aw) for I in (2, 3, 5, 10, 15, 20) for Aw in (1, 1.5, 2.5, 4) if 1 <= I / Aw <= 10]
        I_w, A_w = random.choice(pairs)                          # A, mm^2
        J = I_w / (A_w * 1e-6)                                   # A/m^2
        v = J / (N * E_CH)
        drive_text = (r"the wire, of cross-section $" + num(A_w) + r"\ \mathrm{mm^2}$, carries a steady current $I = " +
                      num(I_w) + r"\ \mathrm{A}$ along $+\hat{x}$")
        Ef = 0.0

    f_opts = [f for f in (0.5, 1, 2, 3, 4, 5, 8, 10, 15, 20, 30, 40, 50, 60) if 0.2 <= 2 * math.pi * f * 1e12 * tau <= 4]
    f_THz = random.choice(f_opts)
    wt = 2 * math.pi * f_THz * 1e12 * tau
    ratio = 1 / math.sqrt(1 + wt ** 2)
    phase = -math.degrees(math.atan(wt))

    # ---------------------------------------------------------------- Lorentz part
    w0 = random.choice([1.0, 1.2, 1.6, 2.0]) * 1e16              # rad/s
    r = random.choice([0.25, 0.4, 0.5, 0.75, 1.25, 1.5, 2, 2.5, 3])     # not near 1, and not so low that chi ~ chi(0)
    w = r * w0
    chi0 = random.choice([0.5, 0.8, 1.0, 1.25, 1.5])
    chi_w = chi0 / (1 - r ** 2)
    below = r < 1
    inv = chi0 * (1 - r ** 2)
    if below:
        chi_choices = [
            {"text": r"$\chi_e = " + num(chi_w, 3) + r"$: positive and larger than the static value, so $\mathbf{P}$ oscillates in phase with $\mathbf{E}$.",
             "correct": "true", "feedback": r"Right: below resonance the spring dominates, and the response is the static one, enhanced by $1/(1-\omega^2/\omega_0^2)$."},
            {"text": r"$\chi_e = " + num(-chi_w, 3) + r"$: negative, so $\mathbf{P}$ points opposite to $\mathbf{E}$.",
             "correct": "false", "feedback": r"The sign flips only above resonance. Here $\omega \lt \omega_0$, so $\omega_0^2 - \omega^2 \gt 0$ and $\chi_e \gt 0$."},
            {"text": r"$\chi_e = " + num(chi0, 3) + r"$, the static value: a susceptibility is a material constant, the same at every frequency.",
             "correct": "false", "feedback": r"The static value holds only for $\omega \ll \omega_0$. In general $\chi_e(\omega) = \chi_e(0)/(1-\omega^2/\omega_0^2)$ depends on $\omega$ (dispersion)."},
            {"text": r"$\chi_e = " + num(inv, 3) + r"$: still positive, but smaller than the static value, since the electrons cannot fully follow the field.",
             "correct": "false", "feedback": r"That multiplies $\chi_e(0)$ by $1 - \omega^2/\omega_0^2$ instead of dividing by it. Below resonance the response grows as $\omega$ approaches $\omega_0$."},
        ]
    else:
        chi_choices = [
            {"text": r"$\chi_e = " + num(chi_w, 3) + r"$: negative, so $\mathbf{P}$ points opposite to $\mathbf{E}$ at every instant.",
             "correct": "true", "feedback": r"Right: above resonance inertia dominates, the electron moves opposite to the force $-e\mathbf{E}$, so $\mathbf{p} = -e\mathbf{r}$ is antiparallel to $\mathbf{E}$."},
            {"text": r"$\chi_e = " + num(-chi_w, 3) + r"$: positive, so $\mathbf{P}$ oscillates in phase with $\mathbf{E}$.",
             "correct": "false", "feedback": r"Here $\omega \gt \omega_0$, so $\omega_0^2 - \omega^2 \lt 0$: the lossless $\chi_e(\omega)$ is negative."},
            {"text": r"$\chi_e = " + num(chi0, 3) + r"$, the static value: a susceptibility is a material constant, the same at every frequency.",
             "correct": "false", "feedback": r"The static value holds only for $\omega \ll \omega_0$. In general $\chi_e(\omega) = \chi_e(0)/(1-\omega^2/\omega_0^2)$, which changes sign at $\omega_0$."},
            {"text": r"$\chi_e = " + num(inv, 3) + r"$: negative, because the electrons lag the field by more than a quarter period.",
             "correct": "false", "feedback": r"The sign is right but the size is not: that multiplies $\chi_e(0)$ by $1 - \omega^2/\omega_0^2$ instead of dividing by it."},
        ]
    random.shuffle(chi_choices)

    # ---------------------------------------------------------------- select-all pool
    def st(sid, text, truth, why):
        return {"id": sid, "text": text, "correct": "true" if truth else "false",
                "feedback": ("True: " if truth else "False: ") + why}

    x1G = 2 * math.pi * 1e9 * tau
    dev1G = x1G / math.sqrt(1 + x1G ** 2)            # |sigma(w) - sigma_DC| / sigma_DC at 1 GHz
    pool = [
        st("tau", r"In the Drude model $\tau$ is the mean time between collisions of a carrier with the lattice; the collision rate is $\nu = 1/\tau$.",
           True, r"the friction term $-m\mathbf{v}/\tau$ stands for collisions at the average rate $\nu = 1/\tau$."),
        st("sigtau", r"$\sigma\propto\tau$ because a carrier that collides less often accelerates for longer, reaching a larger drift speed in the same field.",
           True, r"steady drift $\mathbf{v} = (q\tau/m)\mathbf{E}$ grows with $\tau$, and $\mathbf{J} = Nq\mathbf{v}$."),
        st("oppdrift", r"Conduction electrons drift opposite to $\mathbf{E}$, yet their current density $\mathbf{J} = Nq\mathbf{v}$ points along $\mathbf{E}$.",
           True, r"$q = -e$ and $\mathbf{v} = -(e\tau/m)\mathbf{E}$, so $Nq\mathbf{v} = (Ne^2\tau/m)\mathbf{E}$."),
        st("signs", r"Carriers of either sign, with the same $N$, $|q|$, $m$ and $\tau$, give the same conductivity.",
           True, r"$q$ enters $\sigma = Nq^2\tau/m$ squared."),
        st("polcur", r"A perfect dielectric ($\sigma = 0$, $\chi_e \gt 0$) in a sinusoidal field carries a nonzero current density $\partial\mathbf{P}/\partial t$.",
           True, r"the bound electrons move as $\mathbf{E}$ changes; that motion is the polarization current."),
        st("lead90", r"For $\omega \ll \omega_0$, the polarization current density in a sinusoidal field leads $\mathbf{E}$ by $90^\circ$.",
           True, r"$\mathbf{P} = \epsilon_0\chi_e\mathbf{E}$ is in phase with $\mathbf{E}$, and the time derivative of a cosine leads it by a quarter period."),
        st("rf", r"For this metal at $f = 1\ \mathrm{GHz}$, $|\sigma(\omega) - \sigma_{\text{DC}}|$ is less than $0.1\%$ of $\sigma_{\text{DC}}$.",
           dev1G < 1e-3, r"$\omega\tau = " + num(x1G, 2) + r"$ at 1 GHz, so $|1/(1+j\omega\tau) - 1| \approx \omega\tau$ is about $" + num(100 * dev1G, 2) + r"\%$."),
        st("stiff", r"In the Lorentz model, for the same density $N_d$, bound electrons with a lower natural frequency $\omega_0$ give a larger static $\chi_e$.",
           True, r"$\chi_e(0) = N_de^2/(m\epsilon_0\omega_0^2)$: a softer spring stretches more."),
        st("sametau", r"The Drude collision time $\tau$ and the charge-relaxation time $\epsilon/\sigma$ of Lecture 8 are the same quantity.",
           False, r"they differ by orders of magnitude (copper: $\sim10^{-14}$ s versus $\sim10^{-19}$ s); one is how fast carriers respond, the other how fast excess charge disperses."),
        st("Edep", r"Doubling the applied field doubles the conductivity.",
           False, r"doubling $\mathbf{E}$ doubles $\mathbf{v}$ and $\mathbf{J}$, but $\sigma = J/E$ stays the same."),
        st("persist", r"If $\mathbf{E}$ is switched off, the current in a uniform conductor persists, because no force acts on the carriers any more.",
           False, r"the collision term $-m\mathbf{v}/\tau$ still acts: $\mathbf{v}(t) = \mathbf{v}(0)e^{-t/\tau}$."),
        st("pdir", r"In the Lorentz model the induced dipole moment $\mathbf{p}$ points opposite to a static $\mathbf{E}$, because the electron is pulled against $\mathbf{E}$.",
           False, r"the electron is displaced against $\mathbf{E}$, $\mathbf{r} = -e\mathbf{E}/(m\omega_0^2)$, so $\mathbf{p} = -e\mathbf{r}$ points along $\mathbf{E}$, not against it."),
        st("dcpol", r"A perfect dielectric can carry a steady (DC) polarization current.",
           False, r"in a constant field $\mathbf{P}$ is constant, so $\partial\mathbf{P}/\partial t = 0$: the springs stop the electrons."),
        st("thermal", r"The drift speed found in part (b) is at least $1\%$ of the electrons' typical thermal speed, about $10^5\ \mathrm{m/s}$.",
           v >= 0.01 * 1e5, r"the drift speed is about $" + num(v * 1e3, 2) + r"\ \mathrm{mm/s}$, roughly $10^{" +
           str(round(math.log10(1e5 / v))) + r"}$ times smaller than $10^5\ \mathrm{m/s}$."),
    ]

    # ---------------------------------------------------------------- answer-panel strings
    if drive == "E":
        b2 = (r"Drift speed: $v = (\text{mobility})\times E = (" + num(mob, 4) + r"\ \mathrm{m^2/(V\,s)})(" + num(Ef) +
              r"\ \mathrm{V/m}) = " + num(v * 1e3) + r"\ \mathrm{mm/s}$, directed along $-\hat{x}$ (against $\mathbf{E}$, since $q = -e$); "
              r"the current density $\sigma\mathbf{E} = " + num(sigma * Ef / 1e6, 3) + r"\,\hat{x}\ \mathrm{MA/m^2}$ points along $+\hat{x}$.")
    else:
        b2 = (r"Drift speed: $J = I/A = " + num(I_w) + r"/(" + num(A_w) + r"\times10^{-6}) = " + num(J / 1e6, 3) +
              r"\times10^{6}\ \mathrm{A/m^2}$ and $J = Ne\,v$, so $v = \dfrac{J}{Ne} = " + num(v * 1e3) +
              r"\ \mathrm{mm/s}$, directed along $-\hat{x}$ (electrons move against the current). This route does not need $\tau$; "
              r"the field driving it is $E = J/\sigma = " + num(J / sigma, 3) + r"\ \mathrm{V/m}$, and mobility $\times E$ gives the same speed.")
    if below:
        d_expl = (r"$\omega/\omega_0 = " + num(r) + r" \lt 1$: $\chi_e(\omega) = \dfrac{\chi_e(0)}{1 - (\omega/\omega_0)^2} = \dfrac{" + num(chi0) +
                  r"}{1 - " + num(r * r) + r"} = " + num(chi_w, 3) + r"$, positive and larger than $\chi_e(0)$. Below resonance the restoring force wins "
                  r"over inertia, so each electron is displaced against $\mathbf{E}$ just as in a static field and $\mathbf{P}$ follows $\mathbf{E}$ in phase.")
    else:
        d_expl = (r"$\omega/\omega_0 = " + num(r) + r" \gt 1$: $\chi_e(\omega) = \dfrac{\chi_e(0)}{1 - (\omega/\omega_0)^2} = \dfrac{" + num(chi0) +
                  r"}{1 - " + num(r * r) + r"} = " + num(chi_w, 3) + r"$, negative. Above resonance inertia wins: the electron moves "
                  r"opposite to the force $-e\mathbf{E}$, i.e. along $\mathbf{E}$, so $\mathbf{p} = -e\mathbf{r}$ and $\mathbf{P}$ point against $\mathbf{E}$.")

    data["params"].update({
        "metal": name, "N28": N28, "tau_fs": tau_fs, "sig_meas": sig_meas, "drive": drive, "Ef": Ef,
        "I_w": I_w, "A_w": A_w, "f_THz": f_THz, "w0": w0, "r": r, "chi0": chi0,
        "N_str": num(N28), "tau_str": num(tau_fs), "drive_text": drive_text, "f_str": num(f_THz),
        "w0_tex": sci(w0), "w_tex": sci(w), "chi0_str": num(chi0),
        "sigma_str": num(sigma / 1e6), "mob_str": num(mob * 1e4), "mob_si": num(mob, 4), "b2": b2,
        "wt_str": num(wt), "ratio_str": num(ratio), "phase_str": num(phase), "sig_meas_str": num(sig_meas),
        "d_expl": d_expl, "chi_choices": chi_choices, "pool": pool,
    })
    data["correct_answers"]["sigma"] = float(sigma / 1e6)       # MS/m
    data["correct_answers"]["mob"] = float(mob * 1e4)           # cm^2/(V s)
    data["correct_answers"]["v"] = float(v * 1e3)               # mm/s
    data["correct_answers"]["ratio"] = float(ratio)
    data["correct_answers"]["phase"] = float(phase)             # degrees
