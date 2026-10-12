"""Null point of two point charges on the x axis, field at a probe point by superposition (Lecture 2).

q1 = s1 m^2 u at x = 0, q2 = s2 n^2 u at x = d (m != n, so never equal and opposite; sqrt of the ratio is n/m).
On the axis E_x(x) = k q1 sgn(x)/x^2 + k q2 sgn(x-d)/(x-d)^2.
  * like signs: the fields oppose only between the charges ->  x0 = d m/(m+n)   (closer to the smaller charge)
  * unlike signs: they oppose only outside; the null is beyond the SMALLER charge (weak but close beats strong
    but far):  x0 = d m/(m-n)   (negative if |q1| < |q2|, larger than d if |q1| > |q2|).
  The other root of the squared balance |q1|/x^2 = |q2|/(x-d)^2 is the other formula: there the magnitudes match
  but the directions agree.
Probe P = (xp, yp) off the axis: E = sum k q_i (P - r_i)/|P - r_i|^3, asked as E_x, E_y in kV/m.
Scaling both charges by the same factor (any sign) multiplies E everywhere by that factor: the null point stays.
"""
import math
import random

EPS0 = 8.854e-12
K = 1.0 / (4.0 * math.pi * EPS0)


def fmt(x, n=4):
    s = f"{x:.{n}g}"
    return "0" if s in ("-0", "0", "-0.0") else s


def qtex(q):
    return ("+" if q > 0 else "-") + fmt(abs(q))


def ratio_tex(a, b):
    """LaTeX for a/b, simplified when b is 1."""
    return str(a) if b == 1 else f"\\dfrac{{ {a} }}{{ {b} }}"


def md_over(mm, den):
    """LaTeX for mm*d/den, simplified when mm or den is 1."""
    top = "d" if mm == 1 else f"{mm}\\,d"
    return top if den == 1 else f"\\dfrac{{ {top} }}{{ {den} }}"


def generate(data):
    # ------------------------------------------------------------------ charges and geometry
    while True:
        m, n = random.choice([(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2), (1, 4), (4, 1), (3, 4), (4, 3)])
        u = random.choice([0.5, 1, 2])
        if max(m, n) ** 2 * u > 18:
            continue
        break
    s1, s2 = random.choice([1, -1]), random.choice([1, -1])
    q1, q2 = s1 * m * m * u, s2 * n * n * u              # nC
    d = random.choice([6, 8, 10, 12, 15, 20])            # cm
    like = s1 == s2
    if like:
        x0 = d * m / (m + n)
        x_other = d * m / (m - n)
        region = "between"
    else:
        x0 = d * m / (m - n)
        x_other = d * m / (m + n)
        region = "left" if m < n else "right"
    small_is_1 = m < n

    # ------------------------------------------------------------------ probe point (avoid near-cancelling components)
    while True:
        xp = random.randint(-d // 2, d + d // 2)
        yp = random.choice([-8, -6, -5, -4, -3, 3, 4, 5, 6, 8])
        r1 = math.hypot(xp, yp)
        r2 = math.hypot(xp - d, yp)
        e1 = K * q1 * 1e-9 / (r1 * 1e-2) ** 2 / 1e3        # kV/m, signed magnitude along R1-hat
        e2 = K * q2 * 1e-9 / (r2 * 1e-2) ** 2 / 1e3
        E1 = (e1 * xp / r1, e1 * yp / r1)
        E2 = (e2 * (xp - d) / r2, e2 * yp / r2)
        Ex, Ey = E1[0] + E2[0], E1[1] + E2[1]
        Emag = math.hypot(Ex, Ey)
        if abs(Ex) >= 0.15 * Emag and abs(Ey) >= 0.15 * Emag and Emag > 0.05:
            break

    # ------------------------------------------------------------------ (a) region multiple choice
    if like:
        same_way = "both along $-\\hat{x}$" if s1 > 0 else "both along $+\\hat{x}$"
        same_way_r = "both along $+\\hat{x}$" if s1 > 0 else "both along $-\\hat{x}$"
        fb = {
            "between": "Right: between like charges the two fields point in opposite directions, and each dominates near its own charge, so they cancel in between.",
            "left": f"To the left of both charges the two fields point the same way ({same_way}), so they add.",
            "right": f"To the right of both charges the two fields point the same way ({same_way_r}), so they add.",
            "two": (f"Balancing magnitudes alone gives a second root, $x = {fmt(x_other)}$ cm, outside the pair; there "
                    f"$|\\mathbf{{E}}_1| = |\\mathbf{{E}}_2|$ but both fields point the same way, so they add."),
            "none": "That happens only for equal and opposite charges. Between like charges the fields always oppose, so they cancel somewhere in between.",
        }
    else:
        pos_first = s1 > 0
        between_dir = "$+\\hat{x}$" if pos_first else "$-\\hat{x}$"
        big = "q_2" if small_is_1 else "q_1"
        fb = {
            "between": f"Between unlike charges both fields point the same way, from the positive charge toward the negative one (here {between_dir}), so they add.",
            "left": (f"Right: outside the pair the fields oppose, and to the left of $q_1$ the smaller charge is also the closer one, so the two can balance."
                     if small_is_1 else
                     f"To the left of $q_1$ the fields do oppose, but ${big}$ is both the larger and the closer charge there, so its field always wins."),
            "right": (f"Right: outside the pair the fields oppose, and to the right of $q_2$ the smaller charge is also the closer one, so the two can balance."
                      if not small_is_1 else
                      f"To the right of $q_2$ the fields do oppose, but ${big}$ is both the larger and the closer charge there, so its field always wins."),
            "two": (f"Balancing magnitudes alone gives a second root, $x = {fmt(x_other)}$ cm, between the charges; there "
                    f"$|\\mathbf{{E}}_1| = |\\mathbf{{E}}_2|$ but both fields point the same way, so they add."),
            "none": "A null point exists whenever the two charges differ in magnitude; for unlike charges it lies outside the pair, beyond the smaller one.",
        }
    region_choices = []
    for rid, text in (("between", "between the charges ($0 \\lt x \\lt d$)"),
                      ("left", "to the left of $q_1$ ($x \\lt 0$)"),
                      ("right", "to the right of $q_2$ ($x \\gt d$)"),
                      ("two", "at two points, one between the charges and one outside them"),
                      ("none", "nowhere at a finite distance")):
        region_choices.append({"id": rid, "text": text, "correct": "true" if rid == region else "false",
                               "feedback": fb[rid]})

    # ------------------------------------------------------------------ (d) scaling multiple choice
    kfac = random.choice([2, 3, 4, -1, -2, -3])
    if kfac == -1:
        op_text = r"both charges are replaced by charges of the same magnitudes and opposite signs ($q_1\to-q_1$, $q_2\to-q_2$)"
    elif kfac > 0:
        op_text = f"both charges are multiplied by {kfac} ($q_1\\to {kfac}q_1$, $q_2\\to {kfac}q_2$)"
    else:
        op_text = (f"both charges are multiplied by ${kfac}$ ($q_1\\to {kfac}q_1$, $q_2\\to {kfac}q_2$): "
                   f"both change sign and grow {abs(kfac)}-fold")
    sc = [{"id": "same", "text": "It stays exactly where it was.", "correct": "true",
           "feedback": ("Right: every contribution to $\\mathbf{E}$ is multiplied by the same factor, so the total field becomes "
                        f"${kfac}$ times the old one at every point: zero stays zero, and no new zero appears.")},
          {"id": "gone", "text": "There is no longer a null point at a finite distance.", "correct": "false",
           "feedback": r"If $\mathbf{E}_1+\mathbf{E}_2 = 0$ at a point, then $k\mathbf{E}_1+k\mathbf{E}_2 = 0$ there too."},
          {"id": "larger", "text": "It moves toward the charge of larger magnitude.", "correct": "false",
           "feedback": ("Both magnitudes change by the same factor, so neither charge becomes relatively stronger; the ratio "
                        r"$|q_2|/|q_1|$, which fixes the null point, is unchanged.")}]
    if abs(kfac) != 1:
        sc.append({"id": "sqrt", "text": f"Its distance from $q_1$ is multiplied by $\\sqrt{{ {abs(kfac)} }}$.", "correct": "false",
                   "feedback": (r"$E\propto q/r^2$ suggests this for one charge, but both fields grow by the same factor, so "
                                r"their ratio, and the balance point, are unchanged.")})
    if kfac > 0:
        sc.append({"id": "farther", "text": "It moves farther from both charges, because the fields are now stronger.",
                   "correct": "false",
                   "feedback": "Stronger fields from both charges still cancel where the old ones did: the balance depends only on the ratio of the charges."})
    else:
        sc.append({"id": "mirror", "text": "It moves to the mirror-image point, reflected through the midpoint $x = d/2$.",
                   "correct": "false",
                   "feedback": "Reversing both signs reverses both fields everywhere; the cancellation is unaffected, and nothing singles out the midpoint."})
    if kfac == -1:
        if region == "between":
            rtext = "It moves out of the segment between the charges."
        else:
            rtext = "It moves into the segment between the charges."
        sc.append({"id": "region", "text": rtext, "correct": "false",
                   "feedback": "Reversing both signs keeps like charges like and unlike charges unlike, so the null point stays in the same region, in fact at the same point."})
    random.shuffle(sc)

    # ------------------------------------------------------------------ (e) statements
    both_pos = s1 > 0 and s2 > 0
    pool = [
        ("vector_sum", "sum", True,
         r"The total field at any point is the vector sum of the fields that $q_1$ and $q_2$ would each produce there alone.",
         "that is superposition: Coulomb's law is linear in the charges."),
        ("mag_sum", "sum", False,
         r"The magnitude of the total field at any point equals $|\mathbf{E}_1| + |\mathbf{E}_2|$, the sum of the magnitudes of the two charges' fields.",
         (r"magnitudes add only where the two fields point the same way. At the probe point of (c), $|\mathbf{E}| = "
          + fmt(Emag) + r"\ \mathrm{kV/m}$, while $|\mathbf{E}_1| + |\mathbf{E}_2| = " + fmt(abs(e1) + abs(e2)) + r"\ \mathrm{kV/m}$.")),
        ("begin_end", "lines1", True,
         "Field lines begin on positive charges (or come in from infinity) and end on negative charges (or run off to infinity).",
         "that is how field lines are drawn: they leave positive charge and terminate on negative charge."),
        ("lines_both", "lines1", both_pos,
         "Field lines leave both charges (both act as sources of field lines).",
         ("both charges are positive here, so lines leave both." if both_pos else
          ("both charges are negative here, so lines end on both." if s1 < 0 and s2 < 0 else
           "the charges have opposite signs here: lines leave the positive one and end on the negative one."))),
        ("density", "lines2", True,
         r"Where the field lines crowd closer together, $|\mathbf{E}|$ is larger.",
         "line density is drawn proportional to the field strength."),
        ("path", "lines2", False,
         "A positive test charge released from rest always moves along the field line through its starting point, even where the lines curve.",
         "the force (hence the acceleration) is along the field line, but once the charge has velocity its inertia carries it off a curved line."),
        ("cross_null", "lines3", True,
         r"Away from the charges themselves, two field lines can cross only at a point where $\mathbf{E} = 0$.",
         r"$\mathbf{E}$ has a single direction wherever it is nonzero, so lines can meet only where it has none, as at the null point."),
        ("test_sign", "lines3", False,
         "The position of the null point depends on the sign of the test charge used to probe the field.",
         r"$\mathbf{E} = \mathbf{F}/q$ does not depend on the test charge: a negative test charge feels the opposite force, but $\mathbf{F}/q$ is the same."),
        ("null_equal", "null", True,
         r"At the null point, the fields of $q_1$ and $q_2$ have equal magnitudes and opposite directions.",
         "they are not individually zero; they cancel."),
        ("both_zero", "null", False,
         "At the null point, the field of each charge on its own is zero.",
         r"a single point charge produces a nonzero field at every finite distance; at the null point the two fields cancel."),
        ("closer_smaller", "null2", True,
         "The null point is closer to the charge of smaller magnitude than to the other charge.",
         r"at the null point $|q_1|/r_1^2 = |q_2|/r_2^2$, so the smaller charge must be the closer one."),
        ("far", "far", True,
         r"Far from both charges (at distances much larger than $d$), the field is approximately that of a single point charge $q_1+q_2$.",
         rf"here $q_1+q_2 = {fmt(q1 + q2)}\ \mathrm{{nC}} \ne 0$, and from far away the two charges look like one."),
        ("far_dipole", "far", False,
         r"Far from both charges, the field falls off as $1/r^3$, like that of a dipole.",
         rf"a $1/r^3$ fall-off needs zero net charge; here $q_1+q_2 = {fmt(q1 + q2)}\ \mathrm{{nC}}$, so the field falls off as $1/r^2$."),
        ("Ex_between", "between", not like,
         r"On the $x$ axis between the two charges, $E_x$ has the same sign everywhere.",
         (r"the charges have opposite signs, so between them both fields point from the positive one toward the negative one."
          if not like else r"like charges: between them the two fields oppose, so $E_x$ changes sign at the null point.")),
    ]
    while True:
        nt = random.choice([1, 2, 2, 3, 3, 3, 4])
        cand = pool[:]
        random.shuffle(cand)
        chosen, groups = [], set()
        for s in cand:
            if s[1] in groups:
                continue
            if s[2] and sum(c[2] for c in chosen) >= nt:
                continue
            if not s[2] and sum(not c[2] for c in chosen) >= 5 - nt:
                continue
            chosen.append(s)
            groups.add(s[1])
            if len(chosen) == 5:
                break
        if len(chosen) == 5 and 1 <= sum(c[2] for c in chosen) <= 4:
            break
    facts = [{"id": i, "text": t, "correct": "true" if tr else "false", "verdict": "True" if tr else "False",
              "why": w, "feedback": ("True: " if tr else "False: ") + w} for i, g, tr, t, w in chosen]

    # ------------------------------------------------------------------ worked-solution strings
    if like:
        a_words = (r"The charges have the same sign. Between them the two fields point in opposite directions "
                   r"(each " + ("away from" if s1 > 0 else "toward") + r" its own charge), and each dominates close to its own charge, "
                   r"so they must cancel somewhere in between. Outside the pair both fields point the same way and add. "
                   r"The null point is between the charges.")
        b_words = (r"For $0 \lt x_0 \lt d$: $\dfrac{|q_1|}{x_0^2} = \dfrac{|q_2|}{(d-x_0)^2}$, so "
                   rf"$\dfrac{{d-x_0}}{{x_0}} = \sqrt{{\dfrac{{|q_2|}}{{|q_1|}}}} = {ratio_tex(n, m)}$ and "
                   rf"$x_0 = {md_over(m, m + n)} = {fmt(x0)}\ \mathrm{{cm}}$, closer to the smaller charge.")
    else:
        side = r"to the left of $q_1$" if small_is_1 else r"to the right of $q_2$"
        smaller = "q_1" if small_is_1 else "q_2"
        a_words = (r"The charges have opposite signs. Between them both fields point from the positive charge toward the negative "
                   r"one, so they add. Outside the pair they oppose, but they can balance only where the weaker charge is the "
                   rf"closer one: beyond the smaller charge ${smaller}$, i.e. {side}.")
        if small_is_1:
            b_words = (r"For $x_0 \lt 0$ the distances are $|x_0|$ and $d+|x_0|$: $\dfrac{|q_1|}{x_0^2} = \dfrac{|q_2|}{(d+|x_0|)^2}$, so "
                       rf"$\dfrac{{d+|x_0|}}{{|x_0|}} = \sqrt{{\dfrac{{|q_2|}}{{|q_1|}}}} = {ratio_tex(n, m)}$, "
                       rf"$|x_0| = {md_over(m, n - m)}$ and $x_0 = {fmt(x0)}\ \mathrm{{cm}}$.")
        else:
            b_words = (r"For $x_0 \gt d$ the distances are $x_0$ and $x_0-d$: $\dfrac{|q_1|}{x_0^2} = \dfrac{|q_2|}{(x_0-d)^2}$, so "
                       rf"$\dfrac{{x_0}}{{x_0-d}} = \sqrt{{\dfrac{{|q_1|}}{{|q_2|}}}} = {ratio_tex(m, n)}$ and "
                       rf"$x_0 = {md_over(m, m - n)} = {fmt(x0)}\ \mathrm{{cm}}$.")

    data["params"].update({
        "q1": q1, "q2": q2, "q1_tex": qtex(q1), "q2_tex": qtex(q2), "d": d, "m": m, "n": n, "u": u,
        "x0_str": fmt(x0), "x_other_str": fmt(x_other), "region": region,
        "xp": xp, "yp": yp, "kfac": kfac, "op_text": op_text,
        "region_choices": region_choices, "scale_choices": sc, "facts": facts,
        "a_words": a_words, "b_words": b_words,
        # (c) worked solution
        "R2x": xp - d, "r1_str": fmt(r1), "r2_str": fmt(r2),
        "e1_str": fmt(e1), "e2_str": fmt(e2),
        "E1x_str": fmt(E1[0]), "E1y_str": fmt(E1[1]), "E2x_str": fmt(E2[0]), "E2y_str": fmt(E2[1]),
        "Ex_str": fmt(Ex), "Ey_str": fmt(Ey), "Emag_str": fmt(Emag),
        "sum_mags_str": fmt(abs(e1) + abs(e2)),
        "K_str": "8.988",
        "r1null_str": fmt(abs(x0)), "r2null_str": fmt(abs(x0 - d)),
        "enull_str": fmt(K * abs(q1) * 1e-9 / (abs(x0) * 1e-2) ** 2 / 1e3),
        "enull2_str": fmt(K * abs(q2) * 1e-9 / (abs(x0 - d) * 1e-2) ** 2 / 1e3),
    })
    data["correct_answers"]["x0"] = float(x0)
    data["correct_answers"]["Ex"] = float(Ex)
    data["correct_answers"]["Ey"] = float(Ey)
