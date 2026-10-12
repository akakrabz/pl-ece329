"""Lecture 5: potential by scalar superposition, work by an external agent, potential-energy bookkeeping.

Two fixed point charges q1, q2 [nC] at grid points [cm] in the plane z = 0; points A and B in the same plane.
    V(P)    = k [ q1/|P - P1| + q2/|P - P2| ],   k = 1/(4 pi eps0) = 8.988e9 m/F,   V(inf) = 0
    W_ext   = q3 [V(B) - V(A)] = Delta U     (slow transport: the kinetic energy of q3 does not change)
    W_field = -W_ext = q3 [V(A) - V(B)]
Positions are re-drawn until V_A, V_B and V_B - V_A are not small differences of large terms, so that a hand
calculation carried to 3-4 significant figures (terms, then V_A and V_B) lands inside the 2 % tolerance.
"""
import math
import random

EPS0 = 8.854e-12
K = 1.0 / (4 * math.pi * EPS0)          # 8.988e9 m/F
CHARGES = [q for q in range(-9, 10) if q != 0]
Q3S = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]


def fmt(x, sig=4):
    """Number for LaTeX: `sig` significant figures, no e-notation, never '-0'."""
    if abs(x) < 1e-12:
        return "0"
    s = f"{x:.{sig}g}"
    if "e" in s:
        m, e = s.split("e")
        return f"{m}\\times10^{{{int(e)}}}"
    return s


def fmt0(x, sig=4):
    """Like fmt, but keeps trailing zeros (10.30, 0.1030): for intermediate values shown in the solution."""
    s = f"{x:#.{sig}g}"
    if "e" in s:
        return fmt(x, sig)
    return s[:-1] if s.endswith(".") else s


def paren(x, sig=4):
    s = fmt(x, sig)
    return f"({s})" if x < 0 else s


def signed(q):
    return f"+{q}" if q > 0 else f"{q}"


def pt(p):
    return f"({p[0]},\\ {p[1]})"


def dist(p, r):
    return math.hypot(p[0] - r[0], p[1] - r[1])


def dist_tex(p, r):
    """sqrt((dx)^2 + (dy)^2) = value, with the squares written out (cm)."""
    dx, dy = p[0] - r[0], p[1] - r[1]
    n = dx * dx + dy * dy
    root = math.isqrt(n)
    val = str(root) if root * root == n else fmt0(math.sqrt(n), 4)
    return f"\\sqrt{{ {dx*dx} + {dy*dy} }} = {val}"


def r_m(p, r):
    """Distance in metres for the solution: exact for integer cm distances, else 4 s.f. with trailing zeros."""
    n = (p[0] - r[0]) ** 2 + (p[1] - r[1]) ** 2
    root = math.isqrt(n)
    return fmt(root / 100) if root * root == n else fmt0(math.sqrt(n) / 100)


def term(q_nC, r_cm):
    return K * q_nC * 1e-9 / (r_cm * 1e-2)


def draw():
    while True:
        q1, q2 = random.choice(CHARGES), random.choice(CHARGES)
        P1 = (random.randint(-5, 5), random.randint(-5, 5))
        P2 = (random.randint(-5, 5), random.randint(-5, 5))
        if dist(P1, P2) < 4:
            continue
        A = (random.randint(-7, 7), random.randint(-7, 7))
        B = (random.randint(-7, 7), random.randint(-7, 7))
        if min(dist(A, P1), dist(A, P2), dist(B, P1), dist(B, P2)) < 2 or dist(A, B) < 3:
            continue
        tA = [term(q1, dist(A, P1)), term(q2, dist(A, P2))]
        tB = [term(q1, dist(B, P1)), term(q2, dist(B, P2))]
        VA, VB = sum(tA), sum(tB)
        if abs(VA) < 0.3 * max(map(abs, tA)) or abs(VB) < 0.3 * max(map(abs, tB)):
            continue                       # V is a near-cancellation: hand rounding would wreck it
        if (abs(VB - VA) < 0.3 * max(map(abs, tA + tB)) or abs(VB - VA) < 50
                or abs(VB - VA) < 0.25 * (abs(VA) + abs(VB))):
            continue                       # the work would be a near-cancellation (of V_A, V_B rounded to 3 s.f.)
        return q1, q2, P1, P2, A, B, tA, tB, VA, VB


def generate(data):
    q1, q2, P1, P2, A, B, tA, tB, VA, VB = draw()
    q3 = random.choice(Q3S)
    W = q3 * 1e-9 * (VB - VA)              # J, external agent
    W_uJ = W * 1e6
    up = W > 0                             # potential energy of q3 increases

    # ---------------------------------------------------------------- (c) energy bookkeeping MC
    ws = fmt(abs(W_uJ), 3)

    def combo(u_up, field_pos):            # no numbers here: they would give away the magnitude asked in (b)
        return (f"$U$ {'increases' if u_up else 'decreases'}, and the electric field does "
                f"{'positive' if field_pos else 'negative'} work on $q_3$.")

    mc = [
        {"key": "correct", "text": combo(up, not up), "correct": "true",
         "feedback": r"Right: for a slow move $\Delta U = W_{\text{ext}}$, and the field does $W_{\text{field}} = -\Delta U$."},
        {"key": "same_sign", "text": combo(up, up), "correct": "false",
         "feedback": r"The field works against the agent: $q_3$ starts and ends at rest, so $W_{\text{ext}} + W_{\text{field}} = \Delta KE = 0$ and the two works have opposite signs."},
        {"key": "flipped", "text": combo(not up, up), "correct": "false",
         "feedback": r"Sign reversed: $\Delta U = q_3\,[V(B) - V(A)]$, final minus initial, with the sign of $q_3$ included."},
        {"key": "rest", "text": r"$U$ does not change: $q_3$ starts and ends at rest, so no net work is done on it.",
         "correct": "false",
         "feedback": r"Starting and ending at rest fixes the kinetic energy, not $U$. The agent and the field do equal and opposite work, and $U$ changes by $W_{\text{ext}}$."},
        {"key": "path", "text": r"The change in $U$ depends on the path from $A$ to $B$, so it cannot be found from $V_A$ and $V_B$ alone.",
         "correct": "false",
         "feedback": r"A static $\mathbf{E}$ is curl-free, so the work between two points is path-independent: $\Delta U = q_3\,[V(B) - V(A)]$."},
    ]
    random.shuffle(mc)

    # ---------------------------------------------------------------- (d) statements about potential
    pool = [
        ("perp", True, r"Wherever $\mathbf{E}\neq0$, it is perpendicular to the equipotential surface through that point and points toward lower potential.",
         r"$dV = -\mathbf{E}\cdot d\mathbf{l}$ vanishes for every step taken within an equipotential, so $\mathbf{E}$ has no component along it; $\mathbf{E} = -\nabla V$ points downhill."),
        ("path", True, r"The work an external agent does to carry a charge slowly between two given points is the same for every path joining them.",
         r"A static $\mathbf{E}$ is curl-free, so $\int\mathbf{E}\cdot d\mathbf{l}$ between two points does not depend on the path, and $W_{\text{ext}} = q\,\Delta V$."),
        ("electron", True, r"An electron released from rest at a point where $\mathbf{E}\neq0$ starts to move toward higher potential.",
         r"Its force $-e\mathbf{E}$ points against $\mathbf{E}$, i.e. up the potential hill; its potential energy $U = -eV$ still decreases."),
        ("midpoint", True, r"Midway between two equal positive point charges, $\mathbf{E} = 0$ but $V \neq 0$ (with $V(\infty) = 0$).",
         r"The two fields cancel, but the two potentials $q/(4\pi\epsilon_0 r)$ are both positive and add."),
        ("equip", True, r"Carrying a charge slowly along an equipotential surface takes no work.",
         r"$W_{\text{ext}} = q\,\Delta V$, and $\Delta V = 0$ along the surface."),
        ("double", True, r"If every source charge is doubled, $V$ doubles at every point (with $V(\infty) = 0$).",
         r"$V = \sum_i q_i/(4\pi\epsilon_0 R_i)$ is linear in the charges."),
        ("V0E0", False, r"If $V = 0$ at a point, then $\mathbf{E} = 0$ at that point.",
         r"$\mathbf{E}$ depends on how $V$ changes, not on its value: midway between $+q$ and $-q$, $V = 0$ while $\mathbf{E}$ points from $+q$ toward $-q$."),
        ("proton", False, r"A proton released from rest at a point where $\mathbf{E}\neq0$ starts to move toward higher potential.",
         r"Its force $+e\mathbf{E}$ points along $\mathbf{E}$, i.e. toward lower $V$."),
        ("loop", False, r"The electrostatic field does nonzero net work on a charge carried once around a closed loop that encircles a positive point charge.",
         r"$\oint\mathbf{E}\cdot d\mathbf{l} = 0$ around every closed loop in electrostatics, whatever the loop encircles."),
        ("vector", False, r"The potentials of several point charges must be added as vectors, component by component.",
         r"Potential is a scalar: add the numbers $q_i/(4\pi\epsilon_0 R_i)$ with their signs."),
        ("fieldwork", False, r"The work done by the electric field on a charge $q$ that moves from $A$ to $B$ is $q\,[V(B) - V(A)]$.",
         r"That is the external agent's work, the change in $U$. The field does the opposite amount, $q\,[V(A) - V(B)]$."),
        ("ev", False, r"An electron that moves from a point at $0\ \mathrm{V}$ to a point at $+10\ \mathrm{V}$ gains $10\ \mathrm{eV}$ of potential energy.",
         r"$\Delta U = q\,\Delta V = (-e)(+10\ \mathrm{V}) = -10\ \mathrm{eV}$: its potential energy falls by $10\ \mathrm{eV}$."),
    ]
    ab_true = VA > VB
    pool.append(("AB", ab_true, r"Point $A$ is at a higher potential than point $B$.",
                 f"$V_A = {fmt(VA)}\\ \\mathrm{{V}}$ and $V_B = {fmt(VB)}\\ \\mathrm{{V}}$, so this is "
                 + ("true." if ab_true else "false.")))
    # feedback (shown after grading, while the student may still retry) carries no numbers from this variant
    fb = {"AB": r"Compare your values of $V_A$ and $V_B$ from part (a)."}
    cb = [{"key": k, "correct": "true" if t else "false", "text": txt, "tf": "True" if t else "False",
           "why": why, "feedback": ("True. " if t else "False. ") + fb.get(k, why)} for k, t, txt, why in pool]
    random.shuffle(cb)

    # ---------------------------------------------------------------- answer-panel strings
    if up:
        c_words = (f"$W_{{\\text{{ext}}}} = {fmt(W_uJ)}\\ \\mu\\mathrm{{J}} \\gt 0$: the agent pushes $q_3$ against the field, "
                   f"so $U$ increases by ${ws}\\ \\mu\\mathrm{{J}}$ and the field does $-{ws}\\ \\mu\\mathrm{{J}}$ of work.")
    else:
        c_words = (f"$W_{{\\text{{ext}}}} = {fmt(W_uJ)}\\ \\mu\\mathrm{{J}} \\lt 0$: the field pulls $q_3$ along and the agent only holds it back, "
                   f"so $U$ decreases by ${ws}\\ \\mu\\mathrm{{J}}$ and the field does $+{ws}\\ \\mu\\mathrm{{J}}$ of work.")

    data["params"].update({
        "q1": q1, "q2": q2, "q3": q3, "x1": P1[0], "y1": P1[1], "x2": P2[0], "y2": P2[1],
        "xA": A[0], "yA": A[1], "xB": B[0], "yB": B[1],
        "q1_s": signed(q1), "q2_s": signed(q2), "q3_s": signed(q3),
        "P1_tex": pt(P1), "P2_tex": pt(P2), "A_tex": pt(A), "B_tex": pt(B),
        "rA1_tex": dist_tex(A, P1), "rA2_tex": dist_tex(A, P2), "rB1_tex": dist_tex(B, P1), "rB2_tex": dist_tex(B, P2),
        "rA1_m": r_m(A, P1), "rA2_m": r_m(A, P2), "rB1_m": r_m(B, P1), "rB2_m": r_m(B, P2),
        "tA1": fmt(tA[0]), "tA2": paren(tA[1]), "tB1": fmt(tB[0]), "tB2": paren(tB[1]),
        "VA_str": fmt(VA), "VB_str": fmt(VB), "dV_str": fmt(VB - VA), "dV_paren": paren(VB - VA),
        "W_str": fmt(W_uJ), "W_J": fmt(W), "c_words": c_words,
        "mc": mc, "cb": cb,
    })
    data["correct_answers"]["VA"] = float(VA)
    data["correct_answers"]["VB"] = float(VB)
    data["correct_answers"]["W"] = float(W_uJ)
