"""Lecture 8: induced charge on a conducting shell around a point charge; charge relaxation tau = eps/sigma.

Isolated conducting spherical shell, inner radius a, outer radius b, net charge Q_s (zero for a neutral shell),
with a point charge q at the centre of the cavity; free space elsewhere; electrostatic equilibrium.
    Gauss on a sphere inside the metal (E = 0):  Q_inner = -q        (whatever the position of q in the cavity)
    charge conservation on the shell:            Q_outer = Q_s + q   (uniform: the metal screens the cavity)
    r > b:  E_r = (q + Q_s) / (4 pi eps0 r^2)    (positive = outward);  a < r < b:  E = 0
Relaxation (continuity + Ohm + Gauss in a homogeneous conductor): rho(t) = rho0 exp(-t/tau), tau = eps/sigma;
time to fall to a fraction f: t_f = tau ln(1/f).  Materials: handbook (sigma, eps_r) at room temperature.
"""
import math
import random

EPS0 = 8.854e-12
K = 1.0 / (4 * math.pi * EPS0)

# name, sigma [S/m], sigma (LaTeX), eps_r, display unit for tau, unit (LaTeX), seconds per unit
MATERIALS = [
    ("sea water", 4.0, "4", 81, "ns", r"\mathrm{ns}", 1e-9),
    ("distilled water", 1e-4, "10^{-4}", 81, "us", r"\mu\mathrm{s}", 1e-6),
    ("intrinsic (undoped) silicon at room temperature", 4.4e-4, r"4.4\times10^{-4}", 11.7, "us", r"\mu\mathrm{s}", 1e-6),
    ("intrinsic (undoped) germanium at room temperature", 2.2, "2.2", 16, "ps", r"\mathrm{ps}", 1e-12),
    ("window glass", 1e-12, "10^{-12}", 6, "s", r"\mathrm{s}", 1.0),
    (r"lightly doped silicon (resistivity $10\ \Omega\,\mathrm{cm}$)", 10.0, "10", 11.7, "ps", r"\mathrm{ps}", 1e-12),
]
FRACTIONS = [(0.5, "50%"), (0.1, "10%"), (0.05, "5%"), (0.01, "1%"), (0.001, "0.1%")]


def fmt(x, sig=4):
    if abs(x) < 1e-12:
        return "0"
    s = f"{x:.{sig}g}"
    if "e" in s:
        m, e = s.split("e")
        return f"{m}\\times10^{{{int(e)}}}"
    return s


def signed(v):
    return f"+{v}" if v > 0 else f"{v}"


def generate(data):
    a = random.choice([2, 3, 4, 5])                           # cm
    b = a + random.choice([1, 2, 3])
    q = random.choice([k for k in range(-6, 7) if k != 0])    # nC
    Qs = 0 if random.random() < 0.3 else random.choice([k for k in range(-9, 10) if k != 0])
    r_out = b + random.choice([1, 2, 3, 4, 5])
    r_mid = 0.5 * (a + b)

    Q_in, Q_out = -q, Qs + q
    E_out = K * Q_out * 1e-9 / (r_out * 1e-2) ** 2            # V/m, radial component

    name, sigma, sigma_tex, er, unit, unit_tex, unit_s = random.choice(MATERIALS)
    f, f_txt = random.choice(FRACTIONS)
    tau = er * EPS0 / sigma                                     # s
    t_f = tau * math.log(1 / f)

    # ---------------------------------------------------------------- (b) field inside the metal
    metal = [
        {"key": "zero", "correct": "true", "text": r"$\mathbf{E} = 0$",
         "feedback": r"Right: in electrostatic equilibrium the field inside the metal is zero; the inner-surface charge $-q$ cancels the field of $q$ there."},
        {"key": "q_alone", "correct": "false",
         "text": r"$\mathbf{E} = \dfrac{q}{4\pi\epsilon_0 r^2}\,\hat r$, the field of $q$ alone, since electric field lines pass through metal",
         "feedback": r"Any field in the metal would drive a current. The free charges move until the induced charge $-q$ on the inner surface cancels the field of $q$ there."},
        {"key": "hold", "correct": "false",
         "text": r"a nonzero field pointing toward the cavity, which holds the inner-surface charge in place",
         "feedback": r"No field is needed to keep charge on a surface in equilibrium; a field inside the metal would make charge flow."},
        {"key": "unknown", "correct": "false",
         "text": r"it cannot be found without knowing how the charge is distributed over the two surfaces",
         "feedback": r"It does not depend on the details: $\mathbf{E} = 0$ inside a conductor in electrostatic equilibrium, whatever the surface distributions are."},
    ]
    if Qs != 0:
        metal.append({"key": "total", "correct": "false",
                      "text": r"$\mathbf{E} = \dfrac{q + Q_s}{4\pi\epsilon_0 r^2}\,\hat r$, from Gauss's law with all the charge of the system enclosed",
                      "feedback": r"A Gaussian sphere inside the metal does not enclose the outer-surface charge; it encloses $q$ and the inner-surface charge $-q$, a total of zero."})
    random.shuffle(metal)

    # ---------------------------------------------------------------- (c) q moved off centre
    move = [
        {"key": "inner_only", "correct": "true",
         "text": r"Only the way the inner-surface charge is distributed (and with it the field in the cavity); the surface totals, the uniform outer-surface charge and the field outside the shell stay the same.",
         "feedback": r"Right: the metal screens the cavity. $q$ and the inner-surface charge together produce zero field in the metal and beyond, so the outside cannot tell where $q$ is."},
        {"key": "outer_shift", "correct": "false",
         "text": r"The outer-surface charge crowds toward the side nearest $q$, so the field outside is no longer that of a charge at the centre.",
         "feedback": r"The outer surface sees no field from $q$ and the inner-surface charge (they cancel in the metal and beyond), so it spreads uniformly, as on an isolated sphere."},
        {"key": "inner_total", "correct": "false",
         "text": r"The total charge on the inner surface changes, because part of that surface is now closer to $q$.",
         "feedback": r"A Gaussian sphere inside the metal still encloses zero net charge, so the inner surface still carries exactly $-q$."},
        {"key": "nothing", "correct": "false",
         "text": r"Nothing changes at all: the inner-surface charge stays uniformly spread.",
         "feedback": r"A uniform inner layer could cancel only the field of a centred charge; with $q$ off centre the inner charge must crowd toward $q$ to keep $\mathbf{E} = 0$ in the metal."},
        {"key": "metal_field", "correct": "false",
         "text": r"A field appears inside the metal on the side nearest $q$.",
         "feedback": r"In equilibrium $\mathbf{E} = 0$ everywhere in the metal, wherever $q$ is; the inner-surface charge rearranges until that is so."},
    ]
    random.shuffle(move)

    # ---------------------------------------------------------------- (e) statements about conductors
    outer_same = Q_out * q > 0
    pool = [
        ("Ezero", True, r"In electrostatic equilibrium, $\mathbf{E} = 0$ everywhere inside the metal of the shell.",
         r"Any field there would keep driving the free charges; equilibrium means it has been cancelled."),
        ("equi", True, r"The whole shell, inner and outer surfaces included, is at one potential.",
         r"With $\mathbf{E} = 0$ in the metal, $V(P_2) - V(P_1) = -\int_{P_1}^{P_2}\mathbf{E}\cdot d\mathbf{l} = 0$ for any two points $P_1$, $P_2$ of the metal."),
        ("normal", True, r"Just outside the outer surface, $\mathbf{E} = (\rho_s/\epsilon_0)\,\hat n$, where $\hat n$ is the outward normal and $\rho_s$ the local surface charge density.",
         r"The boundary conditions with zero field inside: $\hat n\cdot\mathbf{D} = \rho_s$ and $\hat n\times\mathbf{E} = 0$."),
        ("surface", True, r"The shell's charge sits on its two surfaces; the charge density inside the metal is zero.",
         r"$\rho = \nabla\cdot\mathbf{D} = 0$ wherever $\mathbf{E} = 0$."),
        ("inner_any", True, r"The total charge on the inner surface is $-q$ wherever $q$ sits in the cavity.",
         r"A Gaussian sphere inside the metal encloses zero net charge for any position of $q$."),
        ("tau_e", True, r"After one relaxation time $\tau = \epsilon/\sigma$, an interior charge density has fallen to $1/e$ (about $37\%$) of its initial value.",
         r"$\rho(t) = \rho_0 e^{-t/\tau}$; it never reaches zero in a finite time."),
        ("fast", True, r"A material whose relaxation time is much shorter than the time over which the applied fields change behaves as a conductor.",
         r"Its free charge keeps up with the field and reaches equilibrium ($\mathbf{E} = 0$ inside) at every instant."),
        ("neutral_inner", False, r"If the shell is electrically neutral, its inner surface carries no charge.",
         r"The inner surface carries $-q$ regardless; a neutral shell then carries $+q$ on its outer surface."),
        ("field_q", False, r"Inside the metal, $\mathbf{E}$ is the field of $q$ alone, because a conductor does not block electric fields.",
         r"The induced charge $-q$ on the inner surface cancels the field of $q$ throughout the metal."),
        ("Vzero", False, r"With $V(\infty) = 0$, every conductor in electrostatic equilibrium is at zero potential.",
         r"$\mathbf{E} = 0$ inside makes $V$ constant, not zero: an isolated sphere of radius $R$ carrying charge $Q$ is at $Q/(4\pi\epsilon_0R)$."),
        ("neutralized", False, r"Excess charge placed inside a conductor decays because it is neutralized where it sits; none of it reaches the surface.",
         r"The charge is conserved: it migrates to the surface, which is where it ends up."),
        ("tangent", False, r"Just outside a charged conductor, $\mathbf{E}$ can have a component tangential to the surface where the surface is curved.",
         r"Tangential $\mathbf{E}$ is continuous and zero inside, so it is zero just outside on any surface, curved or flat."),
        ("wire", False, r"A copper wire carrying a steady current has $\mathbf{E} = 0$ inside it.",
         r"A steady current needs $\mathbf{E} = \mathbf{J}/\sigma \neq 0$; only $\rho = 0$ holds inside."),
    ]
    pool.append(("outer_sign", outer_same, r"The outer surface of this shell carries charge of the same sign as $q$.",
                 f"$Q_{{\\text{{outer}}}} = Q_s + q = {Q_out}\\ \\mathrm{{nC}}$ and $q = {signed(q)}\\ \\mathrm{{nC}}$, so this is "
                 + ("true." if outer_same else "false.")))
    fb = {"outer_sign": r"The outer surface carries $Q_s + q$ (part (a)); compare its sign with that of $q$."}
    cb = [{"key": k, "correct": "true" if t else "false", "text": txt, "tf": "True" if t else "False",
           "why": why, "feedback": ("True. " if t else "False. ") + fb.get(k, why)} for k, t, txt, why in pool]
    random.shuffle(cb)

    # ---------------------------------------------------------------- strings
    if Q_out > 0:
        e_dir = "The field points radially outward."
    elif Q_out < 0:
        e_dir = "The negative sign means the field points radially inward."
    else:
        e_dir = "Here $q + Q_s = 0$: the enclosed charges cancel and there is no field outside the shell at all."
    shell_words = "is electrically neutral" if Qs == 0 else f"carries a net charge $Q_s = {signed(Qs)}\\ \\mathrm{{nC}}$"

    data["params"].update({
        "a": a, "b": b, "q": q, "Qs": Qs, "r_out": r_out, "r_mid": fmt(r_mid), "r_mid_val": r_mid,
        "q_s": signed(q), "Qs_s": signed(Qs), "shell_words": shell_words,
        "Q_in": Q_in, "Q_out": Q_out,
        "E_out_str": fmt(E_out / 1e3), "E_out_Vm": fmt(E_out), "e_dir": e_dir,
        "mat": name, "sigma": sigma, "sigma_tex": sigma_tex, "er": er, "unit": unit, "unit_tex": unit_tex,
        "unit_s": unit_s, "f": f, "f_txt": f_txt, "ln_f": fmt(math.log(1 / f)), "inv_f": fmt(1 / f),
        "tau_str": fmt(tau / unit_s), "tau_s": fmt(tau), "tf_str": fmt(t_f / unit_s),
        "metal": metal, "move": move, "cb": cb,
    })
    data["correct_answers"]["Q_in"] = float(Q_in)            # nC
    data["correct_answers"]["Q_out"] = float(Q_out)          # nC
    data["correct_answers"]["E_out"] = float(E_out / 1e3)    # kV/m, radial (positive outward)
    data["correct_answers"]["tau"] = float(tau / unit_s)     # display unit
    data["correct_answers"]["t_f"] = float(t_f / unit_s)     # display unit
