"""Lorentz force, the work it does, and the velocity selector (Lecture 1; the selector is Lecture 2, section 1).

Part 1. Charge q (uC) with velocity v (km/s) in uniform static E (kV/m) and B (T), integer components.
    F = q (E + v x B).   Units: (km/s)(T) = kV/m and (uC)(kV/m) = mN, so F [mN] = q[uC] (E + v x B)[kV/m].
    P = F . v = q E . v + q (v x B) . v = q E . v       ((v x B) . v = 0 identically).
    Units: (uC)(kV/m)(km/s) = W.   P > 0 <=> the speed is increasing at this instant.
Part 2. Velocity selector: E = E0 e_E (kV/m), B = B0 e_B (mT), e_E perpendicular to e_B; particles injected along
    one of the six axis directions.  F = 0 needs v x B = -E.  Along e_v = e_E x e_B:  (e_E x e_B) x e_B = -e_E,
    so v x B = -v B0 e_E = -E  <=>  v = E0/B0.  Along -e_v the magnetic force adds to qE; along +-e_E it is
    perpendicular to E; along +-e_B it vanishes: no speed works for those five.
    E0/B0 with E0 in kV/m and B0 in mT is in units of 1e6 m/s, so v [km/s] = 1000 E0/B0.
"""
import random

AXES = ("x", "y", "z")
UNIT = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def vec_tex(c):
    """Integer components -> '2\\,\\hat{x} - \\hat{y} + 3\\,\\hat{z}'."""
    terms = []
    for comp, ax in zip(c, AXES):
        if comp == 0:
            continue
        mag = "" if abs(comp) == 1 else f"{abs(comp)}\\,"
        terms.append(("-" if comp < 0 else "+", f"{mag}\\hat{{{ax}}}"))
    if not terms:
        return "0"
    s = ("-" if terms[0][0] == "-" else "") + terms[0][1]
    for sign, t in terms[1:]:
        s += f" {sign} {t}"
    return s


def dir_tex(u):
    """Signed unit axis vector -> '+\\hat{z}' / '-\\hat{x}'."""
    i = [abs(c) for c in u].index(1)
    return ("+" if u[i] > 0 else "-") + "\\hat{" + AXES[i] + "}"


def pr(a, b):
    """A product written out in the solution: '(3)(-2)'."""
    return f"({a})({b})"


# ------------------------------------------------------------------ statement pool for part (c)
# (id, group, truth, text, reason). At most one statement per group is shown.
STATIC_POOL = [
    ("perp_v", "perp", True,
     r"The magnetic part of the force, $q\,\mathbf{v}\times\mathbf{B}$, is always perpendicular to the particle's velocity.",
     r"a cross product is perpendicular to both of its factors, so $(\mathbf{v}\times\mathbf{B})\cdot\mathbf{v} = 0$."),
    ("perp_B", "perp", True,
     r"The magnetic part of the force is always perpendicular to $\mathbf{B}$.",
     r"$\mathbf{v}\times\mathbf{B}$ is perpendicular to $\mathbf{B}$ as well as to $\mathbf{v}$."),
    ("no_work", "work", True,
     r"The magnetic part of the force does no work on the particle.",
     r"the power it delivers is $q(\mathbf{v}\times\mathbf{B})\cdot\mathbf{v} = 0$ at every instant."),
    ("dir_not_speed", "work", True,
     r"Acting alone, a static magnetic field can change the direction of a charged particle's velocity but not its speed.",
     r"the magnetic force is perpendicular to $\mathbf{v}$, so it bends the path (circles, helices) while $|\mathbf{v}|$ stays constant."),
    ("rest", "rest", True,
     r"A charged particle at rest feels no magnetic force, however strong $\mathbf{B}$ is.",
     r"with $\mathbf{v} = 0$ the term $q\,\mathbf{v}\times\mathbf{B}$ vanishes; only $q\mathbf{E}$ can act on a charge at rest."),
    ("parallel", "rest", True,
     r"A charged particle moving parallel or antiparallel to $\mathbf{B}$ feels no magnetic force.",
     r"$\mathbf{v}\times\mathbf{B} = 0$ when $\mathbf{v}$ and $\mathbf{B}$ are parallel or antiparallel."),
    ("E_def_rest", "Edef", True,
     r"$\mathbf{E}$ at a point is the force per unit charge on a small test charge held at rest at that point.",
     r"that is the operational definition (set $\mathbf{v} = 0$ in $\mathbf{F} = q(\mathbf{E}+\mathbf{v}\times\mathbf{B})$, leaving $\mathbf{F}/q = \mathbf{E}$)."),
    ("along_B", "perp", False,
     r"The magnetic force on a moving positive charge points along $\mathbf{B}$.",
     r"$q\,\mathbf{v}\times\mathbf{B}$ is perpendicular to $\mathbf{B}$. 'Force along the field' is true of $\mathbf{E}$, not of $\mathbf{B}$."),
    ("speeds_up", "work", False,
     r"A strong enough static magnetic field, acting alone, can speed up a charged particle.",
     r"the magnetic force is always perpendicular to $\mathbf{v}$ and does no work, so the kinetic energy cannot change, however strong $\mathbf{B}$ is."),
    ("no_change_v", "work2", False,
     r"Because the magnetic force does no work, it cannot change the particle's velocity vector.",
     r"no work means no change of speed only; the direction of $\mathbf{v}$, and so the velocity vector, does change, which is why charges circle in a uniform $\mathbf{B}$."),
    ("E_def_moving", "Edef", False,
     r"$\mathbf{E}$ at a point is the total force per unit charge on a test charge moving through that point, whatever its velocity.",
     r"a moving test charge also feels $q\,\mathbf{v}\times\mathbf{B}$, so $\mathbf{F}/q = \mathbf{E} + \mathbf{v}\times\mathbf{B}$. $\mathbf{E}$ is defined with the test charge at rest."),
    ("max_parallel", "rest", False,
     r"The magnetic force on a moving charge is largest when $\mathbf{v}$ is parallel to $\mathbf{B}$.",
     r"$|\mathbf{v}\times\mathbf{B}| = vB\sin\theta$ is zero for parallel vectors and largest when $\mathbf{v}\perp\mathbf{B}$."),
    ("neg_along", "neg", False,
     r"For a negative charge moving in a direction not parallel to $\mathbf{B}$, the magnetic force points along $\mathbf{v}\times\mathbf{B}$.",
     r"with $q \lt 0$ the force $q\,\mathbf{v}\times\mathbf{B}$ points along $-\mathbf{v}\times\mathbf{B}$."),
    ("mag_power", "power", False,
     r"Part of the power found in (b) is delivered by the magnetic part of the force.",
     r"$(\mathbf{v}\times\mathbf{B})\cdot\mathbf{v} = 0$, so all of the power in (b) comes from $q\mathbf{E}\cdot\mathbf{v}$."),
]


def pick_statements(pool, k=5):
    """Pick k statements, at most one per group, with 1..4 of them true (weighted toward 2-3)."""
    while True:
        nt = random.choice([1, 2, 2, 3, 3, 3, 4])
        trues = [s for s in pool if s["truth"]]
        falses = [s for s in pool if not s["truth"]]
        random.shuffle(trues)
        random.shuffle(falses)
        chosen, groups = [], set()
        for s in trues:
            if sum(c["truth"] for c in chosen) == nt:
                break
            if s["group"] not in groups:
                chosen.append(s)
                groups.add(s["group"])
        for s in falses:
            if len(chosen) == k:
                break
            if s["group"] not in groups:
                chosen.append(s)
                groups.add(s["group"])
        if len(chosen) == k and 1 <= sum(c["truth"] for c in chosen) <= 4:
            random.shuffle(chosen)
            return chosen


def generate(data):
    # ------------------------------------------------------------------ part 1: force and power
    while True:
        q = random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])            # uC
        E = tuple(random.randint(-6, 6) for _ in range(3))                 # kV/m
        v = tuple(random.randint(-5, 5) for _ in range(3))                 # km/s
        B = tuple(random.randint(-3, 3) for _ in range(3))                 # T
        if sum(c != 0 for c in E) < 2 or sum(c != 0 for c in v) < 2 or sum(c != 0 for c in B) < 2:
            continue
        vxB = cross(v, B)                                                  # kV/m
        if vxB == (0, 0, 0):
            continue
        tot = tuple(e + m for e, m in zip(E, vxB))                         # kV/m
        if tot == (0, 0, 0):
            continue
        F = tuple(q * t for t in tot)                                      # mN
        if max(abs(f) for f in F) > 150:
            continue
        break
    Edotv = dot(E, v)
    P = q * Edotv                                                          # W
    assert dot(vxB, v) == 0

    if P > 0:
        p_words = (r"$P \gt 0$: the fields are feeding kinetic energy to the particle, so at this instant it is "
                   r"speeding up (the angle between $\mathbf{F}$ and $\mathbf{v}$ is less than $90^\circ$).")
        p_state = r" $\gt 0$: the particle is speeding up."
    elif P < 0:
        p_words = (r"$P \lt 0$: the electric force has a component against $\mathbf{v}$, so at this instant the "
                   r"particle is slowing down.")
        p_state = r" $\lt 0$: the particle is slowing down."
    else:
        p_words = (r"$P = 0$: here $\mathbf{E}\perp\mathbf{v}$ as well, so at this instant the speed is not changing "
                   r"even though the force is not zero.")
        p_state = r": the speed is momentarily constant."
    p_reason = r"$d(\tfrac12 mv^2)/dt = \mathbf{F}\cdot\mathbf{v} = P$, and here $P = " + f"{P}" + r"\ \mathrm{W}$" + p_state

    # ------------------------------------------------------------------ part (c): statements
    pool = [dict(id=i, group=g, truth=t, text=tx, reason=r) for i, g, t, tx, r in STATIC_POOL]
    if random.random() < 0.5:
        pool.append(dict(id="speeding_up", group="instant", truth=P > 0, reason=p_reason,
                         text=r"At the instant described above, the particle's speed is increasing."))
    else:
        pool.append(dict(id="slowing_down", group="instant", truth=P < 0, reason=p_reason,
                         text=r"At the instant described above, the particle's speed is decreasing."))
    facts = []
    for s in pick_statements(pool):
        verdict = "True" if s["truth"] else "False"
        facts.append({"id": s["id"], "text": s["text"], "correct": "true" if s["truth"] else "false",
                      "verdict": verdict, "why": s["reason"], "feedback": f"{verdict}: {s['reason']}"})

    # ------------------------------------------------------------------ part 2: velocity selector
    axE, axB = random.sample(AXES, 2)
    sE, sB = random.choice([1, -1]), random.choice([1, -1])
    eE = tuple(sE * c for c in UNIT[axE])
    eB = tuple(sB * c for c in UNIT[axB])
    ev = cross(eE, eB)                                                     # the selected direction (unit vector)
    while True:
        E0 = random.choice([10, 12, 15, 16, 20, 24, 25, 30, 36, 40, 45, 48, 50, 60, 72, 75, 80, 90, 100, 120])  # kV/m
        B0 = random.choice([20, 25, 30, 40, 50, 60, 75, 80, 100, 120, 125, 150, 200, 250])                      # mT
        if (1000 * E0) % B0 == 0 and 100 <= 1000 * E0 // B0 <= 2000:
            break
    speed = 1000 * E0 // B0                                                # km/s
    particle = random.choice(["protons", "electrons", "alpha particles (charge $+2e$)",
                              "chloride ions Cl$^-$", "lithium ions Li$^+$"])

    neg_ev = tuple(-c for c in ev)
    third = AXES[[abs(c) for c in ev].index(1)]
    vxB_sel = cross(ev, eB)
    assert vxB_sel == tuple(-c for c in eE)
    dir_choices = [
        {"text": f"${dir_tex(ev)}$", "correct": "true",
         "feedback": (r"Right: $(" + dir_tex(ev) + r")\times(" + dir_tex(eB) + r") = " + dir_tex(vxB_sel) +
                      r"$ is opposite to $\mathbf{E}$, so at the right speed the magnetic force cancels $q\mathbf{E}$, for either sign of $q$.")},
        {"text": f"${dir_tex(neg_ev)}$", "correct": "false",
         "feedback": (r"Along $" + dir_tex(neg_ev) + r"$ (the direction of $\mathbf{B}\times\mathbf{E}$), $\mathbf{v}\times\mathbf{B}$ "
                      r"points along $+\mathbf{E}$, so the magnetic force adds to $q\mathbf{E}$ instead of cancelling it.")},
        {"text": f"${dir_tex(eE)}$", "correct": "false",
         "feedback": (r"A velocity along $\mathbf{E}$ makes $\mathbf{v}\times\mathbf{B}$ perpendicular to $\mathbf{E}$, "
                      r"so nothing can cancel the electric force $q\mathbf{E}$.")},
        {"text": f"${dir_tex(eB)}$", "correct": "false",
         "feedback": r"Moving along $\mathbf{B}$, the particle feels no magnetic force at all, so $q\mathbf{E}$ is unbalanced."},
        {"text": (f"either $+\\hat{{{third}}}$ or $-\\hat{{{third}}}$: once the speed is right, the sense of motion "
                  f"along that axis does not matter"), "correct": "false",
         "feedback": (r"Reversing $\mathbf{v}$ reverses the magnetic force but not the electric one: one sense gives zero "
                      r"net force, the other gives $2q\mathbf{E}$.")},
    ]
    random.shuffle(dir_choices)

    data["params"].update({
        "q": q,
        "E_tex": vec_tex(E), "v_tex": vec_tex(v), "B_tex": vec_tex(B),
        "E_list": list(E), "v_list": list(v), "B_list": list(B),
        "cross_work": (r"\hat{x}\big[" + f"{pr(v[1], B[2])} - {pr(v[2], B[1])}" + r"\big]"
                       r" - \hat{y}\big[" + f"{pr(v[0], B[2])} - {pr(v[2], B[0])}" + r"\big]"
                       r" + \hat{z}\big[" + f"{pr(v[0], B[1])} - {pr(v[1], B[0])}" + r"\big]"),
        "vxB_tex": vec_tex(vxB), "tot_tex": vec_tex(tot), "F_tex": vec_tex(F),
        "Fx": F[0], "Fy": F[1], "Fz": F[2],
        "Fv_dot": f"{pr(F[0], v[0])} + {pr(F[1], v[1])} + {pr(F[2], v[2])}",
        "Ev_dot": f"{pr(E[0], v[0])} + {pr(E[1], v[1])} + {pr(E[2], v[2])}",
        "vxBv_dot": f"{pr(vxB[0], v[0])} + {pr(vxB[1], v[1])} + {pr(vxB[2], v[2])}",
        "Edotv": Edotv, "P": P, "p_words": p_words,
        "facts": facts,
        # velocity selector
        "E0": E0, "B0": B0, "speed": speed, "particle": particle,
        "sE": sE, "sB": sB, "axE": axE, "axB": axB,
        "Esel_tex": f"{sE * E0}\\,\\hat{{{axE}}}", "Bsel_tex": f"{sB * B0}\\,\\hat{{{axB}}}",
        "ev_dir": dir_tex(ev), "negev_dir": dir_tex(neg_ev), "eE_dir": dir_tex(eE), "eB_dir": dir_tex(eB),
        "minus_eE_dir": dir_tex(tuple(-c for c in eE)), "third": third,
        "dir_choices": dir_choices,
    })
    data["correct_answers"]["Fx"] = float(F[0])
    data["correct_answers"]["Fy"] = float(F[1])
    data["correct_answers"]["Fz"] = float(F[2])
    data["correct_answers"]["P"] = float(P)
    data["correct_answers"]["vsel"] = float(speed)
