"""Divergence, curl and what they allow, for a linear field F = M r (Lecture 4).

F_i = sum_j M_ij x_j with an integer matrix M (V/m^2), so dF_i/dx_j = M_ij everywhere:
    div F  = M_11 + M_22 + M_33                       (the trace)
    curl F = (M_32 - M_23, M_13 - M_31, M_21 - M_12)  (zero exactly when M is symmetric)
Four cases, drawn with equal probability: symmetric & traceless / symmetric, trace != 0 /
traceless, not symmetric / neither.
(c) Stokes: circulation around a circle of radius R in the plane x_k = 0, counterclockwise seen from +k
    (normal +k by the right-hand rule) = pi R^2 (curl F)_k, since the curl is uniform.
(d) curl F = 0  <=> could be electrostatic <=> F = -grad V exists (V = -r.M.r/2) <=> zero circulation / path
    independence;  div F = 0 <=> could be a B field <=> zero net flux through every closed surface <=> no charge
    if F were E;  curl F != 0 <=> could be E with dB/dt = -curl F != 0.
(e) continuity: if F were J (A/m^2), d(rho)/dt = -div J = -trace(M)  C/(m^3 s).
"""
import math
import random
from fractions import Fraction

AX = ("x", "y", "z")


def lin_tex(row):
    """[2, -1, 0] -> '2x - y' (coefficients of x, y, z)."""
    out = ""
    for c, v in zip(row, AX):
        if c == 0:
            continue
        mag = "" if abs(c) == 1 else str(abs(c))
        if not out:
            out = ("-" if c < 0 else "") + mag + v
        else:
            out += (" - " if c < 0 else " + ") + mag + v
    return out or "0"


def field_tex(M):
    parts = []
    for row, ax in zip(M, AX):
        if all(c == 0 for c in row):
            continue
        nz = [c for c in row if c != 0]
        if len(nz) == 1:
            body = lin_tex(row)
            sign = "-" if body.startswith("-") else "+"
            body = body.lstrip("-")
            parts.append((sign, body + r"\,\hat{" + ax + "}"))
        else:
            parts.append(("+", r"\left(" + lin_tex(row) + r"\right)\hat{" + ax + "}"))
    s = ("-" if parts[0][0] == "-" else "") + parts[0][1]
    for sign, body in parts[1:]:
        s += f" {sign} {body}"
    return s


def matrix_tex(M):
    rows = [" &amp; ".join(str(c) for c in row) for row in M]
    return r"\begin{pmatrix} " + r" \\ ".join(rows) + r" \end{pmatrix}"


def poly_tex(terms):
    """terms: list of (Fraction coefficient, monomial tex). Returns a signed sum, '0' if empty."""
    out = ""
    for c, mono in terms:
        if c == 0:
            continue
        a = abs(c)
        if a.denominator == 1:
            mag = "" if a == 1 else str(a.numerator)
        else:
            mag = r"\tfrac{" + str(a.numerator) + "}{" + str(a.denominator) + "}"
        if not out:
            out = ("-" if c < 0 else "") + mag + mono
        else:
            out += (" - " if c < 0 else " + ") + mag + mono
    return out or "0"


def gen_M(case):
    sym = case in ("sym_traceless", "sym_trace")
    traceless = case in ("sym_traceless", "curl_traceless")
    vals = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    while True:
        M = [[0] * 3 for _ in range(3)]
        for i in range(3):
            for j in range(i + 1, 3):
                M[i][j] = random.choice(vals)
                M[j][i] = M[i][j] if sym else random.choice(vals)
        d1, d2 = random.randint(-5, 5), random.randint(-5, 5)
        if traceless:
            d3 = -(d1 + d2)
            if abs(d3) > 6:
                continue
        else:
            d3 = random.randint(-5, 5)
            if d1 + d2 + d3 == 0:
                continue
        M[0][0], M[1][1], M[2][2] = d1, d2, d3
        curl = (M[2][1] - M[1][2], M[0][2] - M[2][0], M[1][0] - M[0][1])
        if not sym and curl == (0, 0, 0):
            continue
        if any(all(c == 0 for c in row) for row in M):
            continue
        if sum(c != 0 for row in M for c in row) < 5:
            continue
        if sum(M[i][j] != 0 for i in range(3) for j in range(3) if i != j) < 2:
            continue
        return M


def generate(data):
    case = random.choice(["sym_traceless", "sym_trace", "curl_traceless", "neither"])
    M = gen_M(case)
    div = M[0][0] + M[1][1] + M[2][2]
    curl = (M[2][1] - M[1][2], M[0][2] - M[2][0], M[1][0] - M[0][1])
    curl0 = curl == (0, 0, 0)
    div0 = div == 0

    # ------------------------------------------------------------------ (c) Stokes circulation, multiples of pi volts
    nonzero = [n for n in range(3) if curl[n] != 0]
    if nonzero and random.random() < 0.8:
        k = random.choice(nonzero)                        # usually a loop that the curl threads
    else:
        k = random.randrange(3)
    i, j = (k + 1) % 3, (k + 2) % 3                       # (i, j, k) cyclic
    R = random.choice([1, 2, 3])
    ck = curl[k]                                          # (curl F)_k = M[j][i] - M[i][j]
    assert ck == M[j][i] - M[i][j]
    c0 = R * R * ck
    pi_tex = lambda n: "$0$" if n == 0 else ("$" + ("-" if n < 0 else "") + ("" if abs(n) == 1 else str(abs(n))) +
                                            r"\pi\ \mathrm{V} \approx " + f"{n * math.pi:.1f}" + r"\ \mathrm{V}$")
    dx = (rf"\partial F_{AX[j]}/\partial {AX[i]}", rf"\partial F_{AX[i]}/\partial {AX[j]}")
    cand = [
        (-c0, r"Right size, wrong sign: counterclockwise seen from $+" + AX[k] + r"$ goes with the normal $+\hat{" + AX[k] +
              r"}$ by the right-hand rule, so $\Gamma = +\pi R^2(\nabla\times\mathbf{F})_" + AX[k] + "$."),
        (0, r"$\oint\mathbf{F}\cdot d\mathbf{l}$ vanishes for every loop only when $\nabla\times\mathbf{F} = 0$; "
            r"here the curl component normal to this loop is $(\nabla\times\mathbf{F})_" + AX[k] + f" = {ck}" + r"\ \mathrm{V/m^2}$."),
        (R * R * (M[j][i] + M[i][j]), r"That adds the two cross derivatives; the curl component is their difference, $" + dx[0] + " - " + dx[1] + "$."),
        (R * R * div, r"That is $\pi R^2$ times the divergence. Circulation is governed by the curl (Stokes); net outward flux by the divergence."),
        (R * R * (M[i][i] + M[j][j]), r"That is the outward flux of $\mathbf{F}$ across the circle within its plane (it uses $\partial F_" + AX[i] +
              r"/\partial " + AX[i] + r" + \partial F_" + AX[j] + r"/\partial " + AX[j] + r"$), not the circulation."),
        (2 * R * ck, r"Stokes multiplies the uniform curl by the enclosed area $\pi R^2$, not by the circumference $2\pi R$."),
        (2 * R * R * ck, r"Stokes gives $\Gamma = \pi R^2(\nabla\times\mathbf{F})_" + AX[k] + r"$, with no factor of 2."),
    ]
    for extra in (1, -1, 2, -2, 3, -3, 4, -4):
        cand.append((c0 + extra * R * R, r"Stokes: $\Gamma = \int_S(\nabla\times\mathbf{F})\cdot d\mathbf{S} = \pi R^2(\nabla\times\mathbf{F})_" + AX[k] + "$ for a uniform curl."))
    distract, seen = [], {c0}
    for val, fb in cand:
        if val not in seen:
            distract.append((val, fb))
            seen.add(val)
        if len(distract) == 4:
            break
    right_fb = (r"Right: the curl is uniform, so Stokes gives $\Gamma = \pi R^2(\nabla\times\mathbf{F})_" + AX[k] +
                f" = \\pi({R * R})({ck})" + r"\ \mathrm{V}$" +
                ("; only the curl component normal to the loop matters." if ck == 0 and not curl0 else "."))
    circ_choices = [{"value": c0, "text": pi_tex(c0), "correct": "true", "feedback": right_fb}]
    circ_choices += [{"value": v, "text": pi_tex(v), "correct": "false", "feedback": fb} for v, fb in distract]
    random.shuffle(circ_choices)

    # ------------------------------------------------------------------ (d) statements
    curl_tex = f"({curl[0]},\\ {curl[1]},\\ {curl[2]})\\ \\mathrm{{V/m^2}}"
    negcurl_tex = f"({-curl[0]},\\ {-curl[1]},\\ {-curl[2]})\\ \\mathrm{{V/m^2}}"
    if not curl0:
        Vc = []
        V_tex = ""
    else:
        h = Fraction(1, 2)
        Vterms = [(-h * M[0][0], "x^2"), (-h * M[1][1], "y^2"), (-h * M[2][2], "z^2"),
                  (Fraction(-M[0][1]), "xy"), (Fraction(-M[0][2]), "xz"), (Fraction(-M[1][2]), "yz")]
        V_tex = poly_tex(Vterms)
        Vc = [float(c) for c, _ in Vterms]
    curl_fams = {
        "curlA": [("static_E", curl0, r"$\mathbf{F}$ could be an electrostatic field, that is, an electric field in a situation where nothing changes with time.",
                   (r"an electrostatic field only has to be curl-free, and $\nabla\times\mathbf{F} = 0$ here; its divergence just sets the charge density."
                    if curl0 else rf"every electrostatic field is curl-free, but here $\nabla\times\mathbf{{F}} = {curl_tex} \ne 0$.")),
                  ("potential", curl0, r"There is a scalar function $V$ with $\mathbf{F} = -\nabla V$ everywhere.",
                   (rf"a curl-free field on all of space is a gradient: $V = {V_tex}$ (in volts, plus any constant) gives $-\nabla V = \mathbf{{F}}$."
                    if curl0 else r"the curl of a gradient is always zero, while here $\nabla\times\mathbf{F} \ne 0$, so no such $V$ exists."))],
        "curlB": [("circ_zero", curl0, r"$\oint_C\mathbf{F}\cdot d\mathbf{l} = 0$ around every closed path $C$.",
                   (r"by Stokes' theorem every circulation equals a flux of $\nabla\times\mathbf{F} = 0$."
                    if curl0 else r"by Stokes' theorem, a small loop of area $\Delta S$ whose normal is along $\nabla\times\mathbf{F}$ has circulation $|\nabla\times\mathbf{F}|\,\Delta S \ne 0$.")),
                  ("path_indep", curl0, r"The line integral $\int\mathbf{F}\cdot d\mathbf{l}$ from the origin to the point $(1, 1, 1)$ m is the same for every path.",
                   (r"curl-free means conservative: two paths between the same points form a closed loop with zero circulation."
                    if curl0 else r"two paths that together enclose a loop with nonzero circulation give different results, since $\nabla\times\mathbf{F} \ne 0$."))],
        "curlC": [("changing_B", not curl0, r"$\mathbf{F}$ could be the electric field in a region where $\mathbf{B}$ is changing with time.",
                   (rf"Faraday: $\partial\mathbf{{B}}/\partial t = -\nabla\times\mathbf{{E}}$, which would be ${negcurl_tex}$ here."
                    if not curl0 else r"Faraday: $\partial\mathbf{B}/\partial t = -\nabla\times\mathbf{E} = 0$ for this field, so $\mathbf{B}$ could not be changing there."))],
    }
    div_fams = {
        "divA": [("could_be_B", div0, r"If its units were tesla instead of V/m, $\mathbf{F}$ could be a magnetic flux density $\mathbf{B}$.",
                  (r"the only local requirement on $\mathbf{B}$ is $\nabla\cdot\mathbf{B} = 0$, and $\nabla\cdot\mathbf{F} = 0$ here."
                   if div0 else rf"every magnetic field has $\nabla\cdot\mathbf{{B}} = 0$, but here $\nabla\cdot\mathbf{{F}} = {div}\ \mathrm{{V/m^2}}$."))],
        "divB": [("flux_zero", div0, r"The net outward flux $\oint_S\mathbf{F}\cdot d\mathbf{S}$ is zero for every closed surface $S$.",
                  (r"by the divergence theorem it equals $\int_V\nabla\cdot\mathbf{F}\,dV = 0$."
                   if div0 else rf"by the divergence theorem it equals ${div}\ \mathrm{{V/m^2}}$ times the enclosed volume, which is not zero."))],
        "divC": [("rho_uniform", not div0, r"If $\mathbf{F}$ were an electric field, the charge density in the region would be uniform and nonzero.",
                  (rf"$\rho = \epsilon_0\nabla\cdot\mathbf{{E}} = {div}\,\epsilon_0\ \mathrm{{C/m^3}}$, the same at every point."
                   if not div0 else r"$\rho = \epsilon_0\nabla\cdot\mathbf{E} = 0$: the region would be free of charge.")),
                 ("charge_free", div0, r"If $\mathbf{F}$ were an electric field, the region would contain no charge.",
                  (r"$\rho = \epsilon_0\nabla\cdot\mathbf{E} = 0$ everywhere."
                   if div0 else rf"$\rho = \epsilon_0\nabla\cdot\mathbf{{E}} = {div}\,\epsilon_0\ \mathrm{{C/m^3}} \ne 0$."))],
    }
    jn = next(j for j in range(3) if any(M[r][j] != 0 for r in range(3)))   # a unit point where F != 0
    unit_pt = ("(1,0,0)", "(0,1,0)", "(0,0,1)")[jn]
    static = [
        ("uniform_field", False, r"Because every partial derivative $\partial F_i/\partial x_j$ is a constant, $\mathbf{F}$ has the same value at every point.",
         rf"constant derivatives mean $\mathbf{{F}}$ changes linearly with position: $\mathbf{{F}}(0,0,0) = 0$ but $\mathbf{{F}}{unit_pt} = ({M[0][jn]},\ {M[1][jn]},\ {M[2][jn]})\ \mathrm{{V/m}}$."),
        ("derivs_const", True, r"Neither $\nabla\cdot\mathbf{F}$ nor $\nabla\times\mathbf{F}$ depends on position.",
         r"both are built from first derivatives of a linear field, and those are constants."),
        ("curlfree_divfree", False, r"Any field with zero curl must also have zero divergence.",
         r"counterexample: $x\,\hat{x}$ has zero curl and divergence 1. A static $\mathbf{E}$ is curl-free, yet $\nabla\cdot\mathbf{E} = \rho/\epsilon_0$."),
        ("divfree_curlfree", False, r"Any field with zero divergence must also have zero curl.",
         r"counterexample: $y\,\hat{x}$ has zero divergence and curl $-\hat{z}$."),
        ("add_const", False, r"Adding a uniform field (a constant vector) to $\mathbf{F}$ would change its divergence or its curl.",
         r"the derivatives of a constant vector vanish, so the divergence and curl are unchanged."),
    ]
    while True:
        cg = random.sample(list(curl_fams), 2)
        dg = random.sample(list(div_fams), 2)
        picked = [random.choice(curl_fams[g]) for g in cg] + [random.choice(div_fams[g]) for g in dg]
        picked.append(random.choice(static))
        nt = sum(s[1] for s in picked)
        if 1 <= nt <= 4:
            break
    random.shuffle(picked)
    facts = [{"id": s[0], "text": s[2], "correct": "true" if s[1] else "false", "verdict": "True" if s[1] else "False",
              "why": s[3], "feedback": ("True: " if s[1] else "False: ") + s[3]} for s in picked]

    # ------------------------------------------------------------------ words for the solution
    kind_words = {
        (True, True): r"curl-free <em>and</em> divergence-free: it could be an electrostatic field in a charge-free region, and (in tesla) also a magnetic field in a current-free region",
        (True, False): r"curl-free but not divergence-free: it could be an electrostatic field produced by a uniform charge density, but it cannot be a magnetic field",
        (False, True): r"divergence-free but not curl-free: it could be a magnetic field (produced by a uniform current density), but it cannot be an electrostatic field",
        (False, False): r"neither curl-free nor divergence-free: it can be neither an electrostatic field nor a magnetic field",
    }[(curl0, div0)]

    data["params"].update({
        "case": case, "M": M, "F_tex": field_tex(M), "M_tex": matrix_tex(M),
        "M11": M[0][0], "M22": M[1][1], "M33": M[2][2],
        "M32": M[2][1], "M23": M[1][2], "M13": M[0][2], "M31": M[2][0], "M21": M[1][0], "M12": M[0][1],
        "div": div, "curlx": curl[0], "curly": curl[1], "curlz": curl[2], "negdiv": -div,
        "k_ax": AX[k], "kidx": k, "R": R, "R2": R * R, "ck": ck, "circ_coef": c0, "circ_tex": pi_tex(c0).strip("$"),
        "circ_choices": circ_choices,
        "facts": facts, "Vc": Vc, "kind_words": kind_words,
    })
    data["correct_answers"]["div"] = float(div)
    data["correct_answers"]["curlx"] = float(curl[0])
    data["correct_answers"]["curly"] = float(curl[1])
    data["correct_answers"]["curlz"] = float(curl[2])
    data["correct_answers"]["drho"] = float(-div)
