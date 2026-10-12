"""Gauss's law and symmetry (Lecture 3).

(a) matching: 5 distributions + field points drawn from a pool of 17 -> how |E| depends on the stated distance s
    (1/s^2, 1/s, nonzero constant, proportional to s, zero).
(b) multiple choice: the one distribution (from a non-symmetric pool) for which a single Gaussian surface does not
    give E; distractors are symmetric distributions (including non-uniform rho(r), rho(|x|)) and the
    "Gauss always works" misconception.
(c) uniform ball (radius a, charge Qb) inside a concentric thin shell (radius b > a, charge Qs), free space:
        r < a:      E_r = Qb r / (4 pi eps0 a^3)          (Q_enc = Qb r^3/a^3)
        a < r < b:  E_r = Qb / (4 pi eps0 r^2)
        r > b:      E_r = (Qb + Qs) / (4 pi eps0 r^2)
    asked in kV/m, signed (positive = away from the centre).
(d) checkbox on that configuration; some truths depend on the signs and sizes of Qb, Qs.
"""
import math
import random

EPS0 = 8.854e-12
K = 1.0 / (4.0 * math.pi * EPS0)


def fmt(x, n=4):
    s = f"{x:.{n}g}"
    return "0" if s in ("-0", "0") else s


def signed(x):
    return ("+" if x > 0 else "") + fmt(x)


OPTION_TEXT = {
    "inv2": r"$|\mathbf{E}|\propto 1/s^2$",
    "inv1": r"$|\mathbf{E}|\propto 1/s$",
    "const": r"$|\mathbf{E}|$ is nonzero and independent of $s$",
    "lin": r"$|\mathbf{E}|\propto s$",
    "zero": r"$\mathbf{E} = 0$",
}

MATCH_POOL = [
    ("point", "inv2", "A point charge; field point at distance $s$ from it.",
     r"a concentric sphere of radius $s$ encloses all of $Q$: $E\,4\pi s^2 = Q/\epsilon_0$."),
    ("ball_out", "inv2", "A solid ball of radius $a$ with charge spread uniformly through its volume; field point outside it, at distance $s \\gt a$ from its centre.",
     r"a concentric sphere of radius $s \gt a$ encloses the whole charge, so the ball acts like a point charge at its centre."),
    ("shell_out", "inv2", "A thin spherical shell of radius $a$ with charge spread uniformly over its surface; field point outside it, at distance $s \\gt a$ from its centre.",
     r"a concentric sphere of radius $s \gt a$ encloses all of the shell's charge: the field of a point charge at the centre."),
    ("line", "inv1", "An infinitely long, uniformly charged thin straight line; field point at distance $s$ from the line.",
     r"a coaxial cylinder of radius $s$ and length $L$: $E\,2\pi sL = \rho_l L/\epsilon_0$ (the end caps carry no flux)."),
    ("cyl_out", "inv1", "An infinitely long solid cylinder of radius $a$ with charge spread uniformly through its volume; field point outside it, at distance $s \\gt a$ from its axis.",
     r"a coaxial cylinder of radius $s \gt a$ encloses $\rho\pi a^2L$ whatever $s$ is, so outside it the field is that of a line charge."),
    ("coax_mid", "inv1", "An infinitely long coaxial cable whose inner conductor carries $+\\rho_l$ per unit length and whose outer conductor carries $-\\rho_l$; field point between the conductors, at distance $s$ from the axis.",
     r"a coaxial cylinder between the conductors encloses only $+\rho_l L$: a line-charge field."),
    ("sheet", "const", "An infinite plane sheet with uniform surface charge density $\\rho_s$; field point at distance $s$ from the sheet.",
     r"a pillbox straddling the sheet: $2EA = \rho_sA/\epsilon_0$, so $E = \rho_s/(2\epsilon_0)$ at every distance."),
    ("slab_out", "const", "An infinite slab of thickness $W$ with uniform volume charge density $\\rho$; field point outside it, at distance $s \\gt W/2$ from its mid-plane.",
     r"a pillbox with both caps outside the slab encloses $\rho WA$ however far out the caps are: $E = \rho W/(2\epsilon_0)$."),
    ("sheets_pm_in", "const", "Two parallel infinite sheets with uniform surface charge densities $+\\rho_s$ and $-\\rho_s$; field point between them, at distance $s$ from the positive sheet.",
     r"each sheet gives $\rho_s/(2\epsilon_0)$; between the sheets both fields point from $+$ to $-$ and add to $\rho_s/\epsilon_0$, the same everywhere between."),
    ("ball_in", "lin", "A solid ball of radius $a$ with charge spread uniformly through its volume; field point inside it, at distance $s \\lt a$ from its centre.",
     r"a concentric sphere of radius $s$ encloses $Q s^3/a^3$, so $E\,4\pi s^2 = Qs^3/(\epsilon_0 a^3)$ and $E = Qs/(4\pi\epsilon_0 a^3)$."),
    ("cyl_in", "lin", "An infinitely long solid cylinder of radius $a$ with charge spread uniformly through its volume; field point inside it, at distance $s \\lt a$ from its axis.",
     r"a coaxial cylinder of radius $s$ encloses $\rho\pi s^2L$: $E\,2\pi sL = \rho\pi s^2L/\epsilon_0$, so $E = \rho s/(2\epsilon_0)$."),
    ("slab_in", "lin", "An infinite slab of thickness $W$ with uniform volume charge density $\\rho$; field point inside it, at distance $s \\lt W/2$ from its mid-plane.",
     r"a pillbox with caps at $\pm s$ encloses $2\rho sA$: $2EA = 2\rho sA/\epsilon_0$, so $E = \rho s/\epsilon_0$."),
    ("shell_in", "zero", "A thin spherical shell of radius $a$ with charge spread uniformly over its surface; field point inside it, at distance $s \\lt a$ from its centre.",
     r"a concentric sphere inside the shell encloses no charge, and symmetry makes $E_r$ the same all over it, so $E = 0$."),
    ("coax_out", "zero", "An infinitely long coaxial cable whose inner conductor carries $+\\rho_l$ per unit length and whose outer conductor carries $-\\rho_l$; field point outside the cable, at distance $s$ from its axis.",
     r"a coaxial cylinder outside the cable encloses $\rho_lL - \rho_lL = 0$."),
    ("sheets_pm_out", "zero", "Two parallel infinite sheets with uniform surface charge densities $+\\rho_s$ and $-\\rho_s$; field point outside the pair, at distance $s$ from the nearer sheet.",
     r"outside the pair the two sheet fields, $\rho_s/(2\epsilon_0)$ each, point in opposite directions and cancel."),
    ("sheets_pp_in", "zero", "Two parallel infinite sheets, both with uniform surface charge density $+\\rho_s$; field point between them, at distance $s$ from one of the sheets.",
     r"between two like sheets the two fields, $\rho_s/(2\epsilon_0)$ each, point in opposite directions and cancel."),
    ("shells_pm_out", "zero", "Two concentric thin spherical shells carrying charges $+Q$ (inner) and $-Q$ (outer), each spread uniformly; field point outside both, at distance $s$ from the centre.",
     r"a concentric sphere outside both shells encloses zero net charge."),
]

NONSYM = [
    ("rod", r"a thin rod of finite length $L$ with uniform line charge, the field point beside it at a distance comparable to $L$",
     r"Right: no closed surface has $|\mathbf{E}|$ constant and normal over it, because near the ends the field tilts and weakens. Gauss's law still holds but cannot be solved for $E$; use the Coulomb integral (the 5-step recipe)."),
    ("cube", r"a solid cube with charge spread uniformly through its volume",
     r"Right: a cube has no surface on which $|\mathbf{E}|$ is constant and normal (the field is stronger opposite the face centres than toward the corners). Gauss's law gives only the total flux."),
    ("disk", r"a thin circular disk with uniform surface charge, the field point on its axis at a distance comparable to the disk's radius",
     r"Right: a finite disk is not an infinite sheet, so its field is neither uniform nor normal on any pillbox. Use the Coulomb integral over rings."),
    ("ring", r"a thin circular ring with uniform line charge, the field point on its axis",
     r"Right: no closed surface around a ring has $|\mathbf{E}|$ constant over it. On the axis the Coulomb integral is easy; Gauss's law alone is not enough."),
    ("hemi", r"a solid hemisphere with charge spread uniformly through its volume",
     r"Right: half a ball has no spherical symmetry, so no Gaussian surface has $|\mathbf{E}|$ constant and normal over it."),
]

SYM = [
    ("inf_line", r"an infinitely long, thin straight line with uniform line charge",
     r"Gauss works here: cylindrical symmetry makes $\mathbf{E}$ radial with $|\mathbf{E}|$ fixed by the distance from the line, so a coaxial cylinder gives $E\,2\pi rL = \rho_lL/\epsilon_0$."),
    ("inf_sheet", r"an infinite plane sheet with uniform surface charge",
     r"Gauss works here: planar symmetry makes $\mathbf{E}$ normal to the sheet and equal at equal distances on both sides, so a pillbox gives $2EA = \rho_sA/\epsilon_0$."),
    ("ball_rho_r", r"a solid ball whose charge density depends only on the distance from its centre, $\rho(r)$",
     r"Gauss works here: spherical symmetry does not need a uniform density. $\mathbf{E} = E(r)\hat{r}$, and a concentric sphere gives $E\,4\pi r^2 = Q_{\text{enc}}(r)/\epsilon_0$, with $Q_{\text{enc}}$ an integral of $\rho(r)$."),
    ("shell", r"a thin spherical shell with uniform surface charge",
     r"Gauss works here: spherical symmetry, with concentric spheres inside (no enclosed charge, so $E = 0$) and outside (point-charge field)."),
    ("cyl", r"an infinitely long solid cylinder with uniform volume charge",
     r"Gauss works here: cylindrical symmetry, with coaxial cylinders inside ($E\propto r$) and outside ($E\propto 1/r$)."),
    ("slab_rho_x", r"an infinite slab whose charge density depends only on the distance from its mid-plane",
     r"Gauss works here: planar symmetry about the mid-plane, so a pillbox straddling it gives $2EA = Q_{\text{enc}}/\epsilon_0$, with $Q_{\text{enc}}$ an integral of $\rho$."),
    ("coax", r"an infinitely long coaxial cable whose conductors carry charges $\pm\rho_l$ per unit length",
     r"Gauss works here: cylindrical symmetry, with a coaxial cylinder in each region."),
]


def generate(data):
    # ------------------------------------------------------------------ (a) matching
    while True:
        chosen = random.sample(MATCH_POOL, 5)
        keys = [c[1] for c in chosen]
        if len(set(keys)) >= 3 and max(keys.count(k) for k in set(keys)) <= 2:
            break
    match_st = [{"id": i, "key": k, "text": t, "why": w, "answer": OPTION_TEXT[k]} for i, k, t, w in chosen]

    # ------------------------------------------------------------------ (b) which one defeats Gauss
    ns = random.choice(NONSYM)
    sy = random.sample(SYM, 3)
    gch = [{"id": ns[0], "text": ns[1], "correct": "true", "feedback": ns[2]}]
    gch += [{"id": s[0], "text": s[1], "correct": "false", "feedback": s[2]} for s in sy]
    random.shuffle(gch)
    gch.append({"id": "none", "correct": "false",
                "text": r"none of them: Gauss's law holds for every closed surface, so it gives $\mathbf{E}$ for every one of these distributions",
                "feedback": (r"Gauss's law is always true, but it fixes only the total flux. It yields $\mathbf{E}$ only when symmetry "
                             r"makes $|\mathbf{E}|$ constant and normal on a surface you can choose.")})
    gauss_answer = ns[1]
    gauss_why = ns[2][len("Right: "):]
    gauss_why = gauss_why[0].upper() + gauss_why[1:]

    # ------------------------------------------------------------------ (c) ball + shell
    a = random.choice([2, 3, 4, 5])                                    # cm
    b = a + random.choice([2, 3, 4, 5])                                # cm
    r1 = round(a * random.choice([0.25, 0.4, 0.5, 0.6, 0.75]), 2)      # cm, inside the ball
    r2 = random.choice([x / 2 for x in range(2 * a + 1, 2 * b)])       # cm, a < r2 < b
    r3 = b + random.choice([1, 2, 3, 4, 5, 6])                         # cm
    Qb = random.choice([-8, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 8])     # nC
    if random.random() < 0.1:
        Qs = -Qb
    else:
        Qs = random.choice([x for x in (-12, -10, -8, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 8, 10, 12)])
    Qout = Qb + Qs
    Qenc1 = Qb * (r1 / a) ** 3
    E1 = K * Qenc1 * 1e-9 / (r1 * 1e-2) ** 2 / 1e3                     # kV/m
    E2 = K * Qb * 1e-9 / (r2 * 1e-2) ** 2 / 1e3
    E3 = K * Qout * 1e-9 / (r3 * 1e-2) ** 2 / 1e3
    Ea = K * Qb * 1e-9 / (a * 1e-2) ** 2 / 1e3                         # at the ball's surface
    Eb_in = K * Qb * 1e-9 / (b * 1e-2) ** 2 / 1e3                      # just inside the shell
    Eb_out = K * Qout * 1e-9 / (b * 1e-2) ** 2 / 1e3                   # just outside the shell
    jump = Eb_out - Eb_in

    # ------------------------------------------------------------------ (d) statements about this configuration
    out_zero = Qout == 0
    flip = Qb * Qout < 0
    louder = abs(Qout) > abs(Qb)
    pool = [
        ("qs_r1", "qs", False, r"Changing the shell's charge $Q_s$ (keeping it uniform) would change $E_r$ at $r_1$.",
         r"a uniform shell produces no field inside itself, and a sphere of radius $r_1$ does not enclose any of $Q_s$."),
        ("qs_r2", "qs", False, r"Changing the shell's charge $Q_s$ (keeping it uniform) would change $E_r$ at $r_2$.",
         r"a uniform shell produces no field inside itself, and a sphere of radius $r_2 \lt b$ does not enclose any of $Q_s$."),
        ("qs_r3", "qs", True, r"Changing the shell's charge $Q_s$ (keeping it uniform) would change $E_r$ at $r_3$.",
         r"a sphere of radius $r_3 \gt b$ encloses $Q_b+Q_s$, so $E_r(r_3)$ depends on $Q_s$."),
        ("cont_a", "cont", True, r"$E_r$ is continuous at $r = a$, the surface of the ball.",
         rf"there is no surface charge at $r = a$: just inside, $Q_b a/(4\pi\epsilon_0 a^3)$, and just outside, $Q_b/(4\pi\epsilon_0 a^2)$, are both ${fmt(Ea)}\ \mathrm{{kV/m}}$."),
        ("cont_b", "cont", False, r"$E_r$ is continuous at $r = b$, across the shell.",
         rf"the shell's surface charge makes $E_r$ jump by $Q_s/(4\pi\epsilon_0 b^2) = {signed(jump)}\ \mathrm{{kV/m}}$ (from ${fmt(Eb_in)}$ to ${fmt(Eb_out)}$ kV/m)."),
        ("flux_out", "flux", True, r"The outward flux of $\mathbf{D}$ through any closed surface that encloses both the ball and the shell is $Q_b+Q_s$.",
         rf"Gauss's law: the flux equals the enclosed charge, here $Q_b+Q_s = {fmt(Qout)}\ \mathrm{{nC}}$, whatever the shape of the surface."),
        ("flux_r1", "flux", False, r"The outward flux of $\mathbf{D}$ through a concentric sphere of radius $r_1$ is $Q_b$.",
         rf"that sphere encloses only the fraction $(r_1/a)^3$ of the ball's charge: $Q_b(r_1/a)^3 = {fmt(Qenc1)}\ \mathrm{{nC}}$."),
        ("point_r2", "point", True, r"At $r_2$, $E_r$ is the same as if all of the ball's charge were concentrated at its centre.",
         r"a concentric sphere of radius $r_2$ encloses all of $Q_b$ and none of $Q_s$, so $E_r = Q_b/(4\pi\epsilon_0 r_2^2)$."),
        ("point_r1", "point", False, r"At $r_1$, $E_r$ is the same as if all of the ball's charge were concentrated at its centre.",
         rf"inside the ball only the charge within radius $r_1$ counts, so $E_r$ is smaller by the factor $(r_1/a)^3 = {fmt((r1 / a) ** 3)}$."),
        ("louder", "jumpsize", louder, r"Just outside the shell, $|\mathbf{E}|$ is larger than just inside it.",
         rf"just inside, $|E_r| = |Q_b|/(4\pi\epsilon_0 b^2) = {fmt(abs(Eb_in))}\ \mathrm{{kV/m}}$; just outside, $|Q_b+Q_s|/(4\pi\epsilon_0 b^2) = {fmt(abs(Eb_out))}\ \mathrm{{kV/m}}$."),
        ("flip", "sign", flip, r"$E_r$ has opposite signs at $r_2$ and at $r_3$.",
         rf"$E_r(r_2)$ has the sign of $Q_b = {fmt(Qb)}$ nC and $E_r(r_3)$ the sign of $Q_b+Q_s = {fmt(Qout)}$ nC" +
         (r"; opposite signs." if flip else (r"; $E_r(r_3) = 0$, which has no sign." if out_zero else r"; the same sign."))),
        ("out_zero", "zero", out_zero, r"The electric field outside the shell ($r \gt b$) is zero.",
         rf"a sphere of radius $r \gt b$ encloses $Q_b+Q_s = {fmt(Qout)}\ \mathrm{{nC}}$" +
         (r", so $E_r = 0$ there." if out_zero else r", which is not zero.")),
    ]
    while True:
        nt = random.choice([1, 2, 2, 3, 3, 3, 4])
        cand = pool[:]
        random.shuffle(cand)
        picked, groups = [], set()
        for s in cand:
            if s[1] in groups:
                continue
            if s[2] and sum(c[2] for c in picked) >= nt:
                continue
            if not s[2] and sum(not c[2] for c in picked) >= 5 - nt:
                continue
            picked.append(s)
            groups.add(s[1])
            if len(picked) == 5:
                break
        if len(picked) == 5 and 1 <= sum(c[2] for c in picked) <= 4:
            break
    facts = [{"id": i, "text": t, "correct": "true" if tr else "false", "verdict": "True" if tr else "False",
              "why": w, "feedback": ("True: " if tr else "False: ") + w} for i, g, tr, t, w in picked]

    data["params"].update({
        "match_st": match_st, "gauss_choices": gch, "gauss_answer": gauss_answer, "gauss_why": gauss_why,
        "a": a, "b": b, "r1": fmt(r1), "r2": fmt(r2), "r3": r3, "Qb": Qb, "Qs": Qs,
        "Qb_tex": signed(Qb), "Qs_tex": signed(Qs), "Qout_str": fmt(Qout), "Qenc1_str": fmt(Qenc1),
        "frac1_str": fmt((r1 / a) ** 3),
        "E1_str": fmt(E1), "E2_str": fmt(E2), "E3_str": fmt(E3),
        "facts": facts,
    })
    data["correct_answers"]["E1"] = float(E1)
    data["correct_answers"]["E2"] = float(E2)
    data["correct_answers"]["E3"] = float(E3)
