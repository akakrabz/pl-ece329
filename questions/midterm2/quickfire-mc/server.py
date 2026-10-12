"""Midterm 2 quick-fire: six short multiple-choice items (Lectures 11-19), drawn from twelve generators.

Each generator returns {gen, stem, choices[4], solution, data}; exactly one choice is correct and every distractor
carries feedback naming the slip.  The template loops over params.items; answers-names are q1..q6.
Generators (family of SP18 Exam 2 #1, fresh items):
  wires     force between parallel wires: attract/repel after changes, F/l = mu0 I1 I2/(2 pi d)
  loop      induced current from a uniform dB/dt with an in-plane part: I = |dB_n/dt| A / R
  magnet    bar magnet through a ring: current sense on approach and departure (Lenz)
  dcond     which D is compatible with curl H = dD/dt when J = 0: d(div D)/dt = 0
  potentials E = -grad(Phi) - dA/dt and B = curl A
  sheetH    H on the far side of a current sheet: n x (H1 - H2) = J_s, normal H continuous (same mu)
  phase     wavelength or frequency from a phase change over a time or a distance in a dielectric
  drude     sigma = N q^2 tau / m ratios
  magcur    J_sM = M x n_hat on a face of a uniformly magnetized cube
  disp      displacement-current amplitude omega eps E0 in a dielectric
  etav      eta = eta0 sqrt(mur/epsr), v = c/sqrt(mur epsr)
  poynting  E x H along the travel; H = u x E / eta0
"""
import math
import random
from fractions import Fraction as F

MU0 = 4e-7 * math.pi
EPS0 = 8.8541878128e-12
C0 = 1 / math.sqrt(MU0 * EPS0)
ETA0 = math.sqrt(MU0 / EPS0)
AXES = ["x", "y", "z"]
IDX = {"x": 0, "y": 1, "z": 2}


# ------------------------------------------------------------------ formatting helpers
def g3(x):
    """3 significant figures, LaTeX scientific notation outside [0.01, 1e4)."""
    if x == 0:
        return "0"
    s = f"{x:.3g}"
    if "e" in s:
        m, e = s.split("e")
        return f"{m}\\times10^{{{int(e)}}}"
    return s


def ftex(q):
    """Fraction -> LaTeX."""
    q = F(q)
    if q.denominator == 1:
        return str(q.numerator)
    sgn = "-" if q < 0 else ""
    return f"{sgn}\\tfrac{{{abs(q.numerator)}}}{{{q.denominator}}}"


def coef_tex(q, sym):
    """k*sym with k a Fraction: 1 -> sym, 6 -> 6sym, 3/2 -> \\tfrac{3}{2}sym."""
    return sym if q == 1 else ftex(q) + sym


def uhat(ax, sg=1):
    return ("+" if sg > 0 else "-") + "\\hat{" + ax + "}"


def cross_ax(a, b):
    """a_hat x b_hat for distinct axes -> (axis, sign)."""
    ia, ib = IDX[a], IDX[b]
    c = AXES[3 - ia - ib]
    return c, (1 if (ib - ia) % 3 == 1 else -1)


def numstr(x):
    return f"{x:g}"


def vec_tex(v, unit=""):
    """v: list of 3 numbers -> '3\\,\\hat{x} - 1.5\\,\\hat{z}' (or '\\mathbf{0}')."""
    out = ""
    for ax, val in zip(AXES, v):
        if val == 0:
            continue
        mag = numstr(abs(val))
        body = f"{mag}\\,\\hat{{{ax}}}"
        if not out:
            out = ("-" if val < 0 else "") + body
        else:
            out += (" - " if val < 0 else " + ") + body
    if not out:
        return "\\mathbf{0}"
    return out + (f"\\ \\mathrm{{{unit}}}" if unit else "")


def mono_tex(powers):
    """{'t': 1, 'y': 2} -> 't\\,y^{2}' (order t, x, y, z)."""
    parts = []
    for var in ["t", "x", "y", "z"]:
        pw = powers.get(var, 0)
        if pw == 1:
            parts.append(var)
        elif pw > 1:
            parts.append(f"{var}^{{{pw}}}")
    return "\\,".join(parts)


def terms_tex(terms):
    """terms: list of (coef, axis, powers) -> '2\\,t\\,y\\,\\hat{x} - x^{2}\\,\\hat{z}'."""
    out = ""
    for k, ax, pw in terms:
        if k == 0:
            continue
        m = mono_tex(pw)
        mag = abs(k)
        body = ("" if mag == 1 else numstr(mag) + "\\,") + (m + "\\," if m else ("1\\," if mag == 1 else "")) + f"\\hat{{{ax}}}"
        if not out:
            out = ("-" if k < 0 else "") + body
        else:
            out += (" - " if k < 0 else " + ") + body
    return out or "\\mathbf{0}"


def poly_tex(poly):
    """poly: dict powers-tuple(t,x,y,z) -> coef -> LaTeX scalar polynomial."""
    out = ""
    for key in sorted(poly, reverse=True):
        k = poly[key]
        if k == 0:
            continue
        m = mono_tex(dict(zip(["t", "x", "y", "z"], key)))
        mag = abs(k)
        body = (numstr(mag) if (mag != 1 or not m) else "") + ("\\," if (mag != 1 and m) else "") + m
        if not out:
            out = ("-" if k < 0 else "") + body
        else:
            out += (" - " if k < 0 else " + ") + body
    return out or "0"


def div_poly(terms):
    """Divergence of sum coef * monomial * axis_hat, as a dict powers-tuple(t,x,y,z) -> coef."""
    res = {}
    for k, ax, pw in terms:
        n = pw.get(ax, 0)
        if n == 0:
            continue
        new = dict(pw)
        new[ax] = n - 1
        key = tuple(new.get(v, 0) for v in ["t", "x", "y", "z"])
        res[key] = res.get(key, 0) + k * n
    return {key: v for key, v in res.items() if v != 0}


def finish(gen, stem, choices, solution, data):
    texts = [c["text"] for c in choices]
    assert len(choices) == 4 and len(set(texts)) == 4, (gen, texts)
    assert sum(c["correct"] == "true" for c in choices) == 1, gen
    random.shuffle(choices)
    return {"gen": gen, "stem": stem, "choices": choices, "solution": solution, "data": data}


def ch(text, ok, fb):
    return {"text": text, "correct": "true" if ok else "false", "feedback": fb}


def add_numeric(choices, vals, val, text, fb, sep=0.12):
    """Append a distractor only if its value is at least `sep` (relative) away from every value already offered."""
    if len(choices) >= 4 or any(abs(val - u) <= sep * max(abs(val), abs(u)) for u in vals):
        return
    vals.append(val)
    choices.append(ch(text, False, fb))


# ------------------------------------------------------------------ generators
def g_wires():
    ax = random.choice(AXES)
    a = random.choice([F(2), F(3), F(1, 2)])
    b = random.choice([F(1), F(2), F(3), F(1, 2)])
    c = random.choice([F(2), F(3), F(1, 2), F(1, 3), F(4)])
    d_cm = random.choice([dd for dd in range(2, 11) if (c * dd).denominator == 1])
    same0 = random.choice([True, False])
    rev2 = random.choice([True, False])
    k, k2 = a * b / c, a * b / (c * c)
    same1 = same0 != rev2
    kind, other = ("attractive", "repulsive") if same1 else ("repulsive", "attractive")
    words = {F(2): "doubled", F(3): "tripled", F(1, 2): "halved"}
    if b == 1:
        i2 = "keeps its magnitude but is reversed in direction" if rev2 else "is unchanged"
    else:
        i2 = "is " + words[b] + (" and reversed in direction" if rev2 else "")
    d_new = c * d_cm
    stem = (f"Two long, straight, parallel wires along the ${ax}$ direction, ${d_cm}\\ \\mathrm{{cm}}$ apart, carry currents $I_1$ and $I_2$ "
            f"in {'the same direction' if same0 else 'opposite directions'}; the magnetic force per unit length on wire 2 has magnitude $F_0$. "
            f"Now $I_1$ is {words[a]}, $I_2$ {i2}, and the separation is changed to ${numstr(float(d_new))}\\ \\mathrm{{cm}}$. "
            "The force per unit length on wire 2 is now")
    kt, k2t = coef_tex(k, "F_0"), coef_tex(k2, "F_0")
    dirs = "the same direction" if same1 else "opposite directions"
    fb_type = (f"Currents in the same direction attract and opposite currents repel; after the change they flow in {dirs}.")
    fb_sq = r"$F/\ell = \mu_0I_1I_2/(2\pi d)$ falls off as $1/d$, not $1/d^2$ (that is the point-charge law)."
    choices = [
        ch(f"${kt}$, {kind}", True, f"Right: the factor is $({ftex(a)})({ftex(b)})/({ftex(c)}) = {ftex(k)}$, and the currents now flow in {dirs}."),
        ch(f"${kt}$, {other}", False, fb_type),
        ch(f"${k2t}$, {kind}", False, fb_sq),
        ch(f"${k2t}$, {other}", False, fb_sq + " " + fb_type),
    ]
    sol = (f"$F/\\ell = \\mu_0I_1I_2/(2\\pi d)$ is proportional to $I_1I_2/d$, so the new force is $({ftex(a)})({ftex(b)})/({ftex(c)})\\,F_0 = {kt}$. "
           f"After the change the currents flow in {dirs}, so the force is {kind} (parallel currents attract, antiparallel currents repel).")
    data = {"ax": ax, "d_cm": d_cm, "same0": same0, "rev2": rev2, "a": str(a), "b": str(b), "c": str(c)}
    return finish("wires", stem, choices, sol, data)


def g_loop():
    nax = random.choice(AXES)
    e1 = random.choice([x for x in AXES if x != nax])
    plane = "".join(sorted([x for x in AXES if x != nax]))
    if random.random() < 0.5:
        size = random.choice([5, 8, 10, 12, 15, 20])
        shape, A, P = "square", (size / 100) ** 2, 4 * size / 100
        desc = f"a square loop of side ${size}\\ \\mathrm{{cm}}$"
    else:
        size = random.choice([4, 5, 6, 8, 10])
        shape, A, P = "circle", math.pi * (size / 100) ** 2, 2 * math.pi * size / 100
        desc = f"a circular loop of radius ${size}\\ \\mathrm{{cm}}$"
    vals = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.2, 1.5, 2.0]
    while True:
        pin = random.choice(vals) * random.choice([1, -1])
        qn = random.choice(vals) * random.choice([1, -1])
        ratio = abs(pin) / abs(qn)
        if 0.6 <= ratio <= 2.5 and abs(ratio - 1) >= 0.15:
            break
    R = random.choice([0.5, 1, 2, 4, 5])
    dB = [0.0, 0.0, 0.0]
    dB[IDX[e1]], dB[IDX[nax]] = pin, qn
    I = abs(qn) * A / R * 1e3                                           # mA
    cands = [
        (math.hypot(pin, qn) * A / R * 1e3, r"Only the component of $d\mathbf{B}/dt$ along the loop's normal changes the flux; the in-plane part threads none of it."),
        (abs(pin) * A / R * 1e3, r"That component lies in the plane of the loop and contributes no flux; use the normal component."),
        (abs(qn) * P / R * 1e3, r"The emf is the rate of change of the flux $\int\mathbf{B}\cdot d\mathbf{S}$ over the loop's area, not something along its perimeter."),
        (abs(qn) * A * R * 1e3, r"Ohm's law gives $I = \mathcal{E}/R$, not $\mathcal{E}R$."),
        ((abs(pin) + abs(qn)) * A / R * 1e3, r"Components do not add as numbers, and only the normal one threads the loop anyway."),
    ]
    choices = [ch(f"${g3(I)}\\ \\mathrm{{mA}}$", True, "Right: $I = \\lvert dB_n/dt\\rvert A/R$.")]
    used = [I]
    for val, fb in cands:
        add_numeric(choices, used, val, f"${g3(val)}\\ \\mathrm{{mA}}$", fb)
    stem = (f"{desc[0].upper() + desc[1:]} lies in the ${plane}$ plane; its resistance is $R = {numstr(R)}\\ \\Omega$. "
            f"A spatially uniform magnetic field changes at the constant rate $d\\mathbf{{B}}/dt = ({vec_tex(dB)})\\ \\mathrm{{T/s}}$. "
            "The magnitude of the induced current is")
    sol = (f"Only $B_{nax}$ threads the loop: $\\lvert\\mathcal{{E}}\\rvert = \\lvert dB_{nax}/dt\\rvert A = ({numstr(abs(qn))}\\ \\mathrm{{T/s}})({g3(A)}\\ \\mathrm{{m^2}})$ "
           f"$= {g3(abs(qn) * A * 1e3)}\\ \\mathrm{{mV}}$, so $I = \\lvert\\mathcal{{E}}\\rvert/R = {g3(I)}\\ \\mathrm{{mA}}$. The ${e1}$ component is parallel to the loop and adds no flux.")
    data = {"shape": shape, "size_cm": size, "nax": nax, "dBdt": dB, "R": R}
    return finish("loop", stem, choices, sol, data)


def g_magnet():
    motion = random.choice(["down", "up"])
    lead = random.choice(["N", "S"])
    view = random.choice(["above", "below"])
    upper_is_N = (motion == "down" and lead == "S") or (motion == "up" and lead == "N")
    s = 1 if upper_is_N else -1                       # sign of the magnet's moment along +z (S -> N inside)
    ccw_above = {"approach": -s, "leave": s}          # sign of the counterclockwise-from-above current
    flip = 1 if view == "above" else -1
    word = lambda sg: "counterclockwise" if sg * flip > 0 else "clockwise"
    w1, w2 = word(ccw_above["approach"]), word(ccw_above["leave"])
    other = {"counterclockwise": "clockwise", "clockwise": "counterclockwise"}
    pole_word = {"N": "north", "S": "south"}[lead]
    start = "above" if motion == "down" else "below"
    how = "dropped, so that it falls" if motion == "down" else "pushed upward, so that it rises"
    stem = (f"A bar magnet is held vertically {start} a horizontal conducting ring, on the ring's axis, with its {pole_word} pole pointing "
            f"{'down' if motion == 'down' else 'up'}, toward the ring. It is {how} straight through the ring. Seen from {view}, the current induced in the ring "
            "while the magnet approaches, and then while it moves away after passing through, circulates")
    mdir = "up" if s > 0 else "down"
    fb_rev = (f"Reversed. The magnet's flux through the ring points {mdir} (from its south to its north pole) the whole time; "
              f"it grows on approach, so the induced field must point {'down' if s > 0 else 'up'} inside the ring, which is {w1} seen from {view}.")
    fb_same = ("The flux keeps its direction throughout, but it grows while the magnet approaches and shrinks while it leaves. "
               "Lenz's law opposes the change, not the field, so the current reverses.")
    choices = [
        ch(f"{w1}, then {w2}", True, "Right: the induced current opposes the growth of the flux, then its decrease."),
        ch(f"{other[w1]}, then {other[w2]}", False, fb_rev),
        ch(f"{w1} both times", False, fb_same),
        ch(f"{other[w1]} both times", False, fb_same),
    ]
    up_or_down = lambda wd: ("an upward" if wd == "up" else "a downward")
    sol = (f"The magnet's moment points from its south pole to its north pole, here {mdir}, and the flux it sends through the ring points {mdir} "
           f"the whole time. Approaching: the {mdir}ward flux grows, so the ring makes {up_or_down('down' if s > 0 else 'up')} field inside itself: "
           f"{w1} seen from {view}. Leaving: the {mdir}ward flux shrinks, so the ring makes {up_or_down(mdir)} field: {w2} seen from {view}.")
    data = {"motion": motion, "lead": lead, "view": view}
    return finish("magnet", stem, choices, sol, data)


def g_dcond():
    def pick(n):
        return random.sample(AXES, n)
    allowed = []
    p, q = pick(2)
    allowed.append([(random.choice([1, 2, 3]), p, {q: 1, "t": 1})])
    p, q = pick(2)
    r = random.choice([x for x in AXES if x != q])
    allowed.append([(random.choice([1, 2]), p, {p: 1}), (random.choice([1, 3]), r, {q: 1, "t": 1})])
    p, q = pick(2)
    allowed.append([(1, p, {q: 1, "t": 2}), (random.choice([1, 2]), q, {p: 2})])
    forb = []
    p = random.choice(AXES)
    forb.append([(random.choice([1, 2, 3]), p, {p: 1, "t": 1})])
    p, q = pick(2)
    r = random.choice(AXES)
    forb.append([(random.choice([1, 2]), p, {q: 1}), (1, r, {r: 1, "t": 2})])
    p = random.choice(AXES)
    forb.append([(1, p, {p: 2, "t": 1})])
    p, q = pick(2)
    forb.append([(1, p, {p: 1, "t": 1}), (random.choice([1, 2]), q, {q: 1, "t": 1})])
    p, q = pick(2)
    r = random.choice([x for x in AXES if x != p])
    forb.append([(random.choice([1, 2]), p, {q: 1, "t": 1}), (1, r, {r: 1, "t": 1})])
    good = random.choice(allowed)
    bad = random.sample(forb, 3)

    def ftext(terms):
        return f"$\\mathbf{{D}} = D_0\\left({terms_tex(terms)}\\right)$"

    def divt(terms):
        dp = div_poly(terms)
        if not dp:
            return "0"
        if dp == {(0, 0, 0, 0): 1}:
            return "D_0"
        return "D_0\\,(" + poly_tex(dp) + ")"
    choices = [ch(ftext(good), True, f"Right: $\\nabla\\cdot\\mathbf{{D}} = {divt(good)}$ does not change with time.")]
    for b in bad:
        choices.append(ch(ftext(b), False,
                          f"$\\nabla\\cdot\\mathbf{{D}} = {divt(b)}$ changes with time, but the divergence of $\\nabla\\times\\mathbf{{H}}$ is zero, "
                          "so with $\\mathbf{J} = \\mathbf{0}$ the charge density $\\nabla\\cdot\\mathbf{D}$ cannot change."))
    if len({c["text"] for c in choices}) < 4:
        return g_dcond()
    stem = (r"In a region where the current density is $\mathbf{J} = \mathbf{0}$ at all times, the Ampère–Maxwell law reads "
            r"$\partial\mathbf{D}/\partial t = \nabla\times\mathbf{H}$. Which of these could be $\mathbf{D}$ there, for a suitable $\mathbf{H}$? "
            r"($D_0$ is a constant; $x, y, z$ in m, $t$ in s.)")
    sol = (r"Take the divergence: $\partial(\nabla\cdot\mathbf{D})/\partial t = \nabla\cdot(\nabla\times\mathbf{H}) = 0$, so $\rho = \nabla\cdot\mathbf{D}$ may be "
           r"nonzero but must not change in time (charge cannot appear without a current to bring it; it is the continuity equation). "
           f"Here $\\nabla\\cdot\\mathbf{{D}} = {divt(good)}$ for the correct field; each of the others has a time-dependent divergence.")
    return finish("dcond", stem, choices, sol, {})


def g_potentials():
    x1 = random.choice(AXES)
    p = random.choice([x for x in AXES if x != x1])
    q = random.choice([x for x in AXES if x != p])
    alpha = random.choice([2, 3, 4, 5]) * random.choice([1, -1])
    beta = random.choice([2, 3, 4, 6]) * random.choice([1, -1])
    bax, bsg = cross_ax(q, p)                           # grad(q) x p_hat
    E_ok = [(-2 * alpha, x1, {x1: 1}), (-beta, p, {q: 1})]
    E_plus = [(-2 * alpha, x1, {x1: 1}), (beta, p, {q: 1})]
    E_stat = [(-2 * alpha, x1, {x1: 1})]
    B_ok = [(beta * bsg, bax, {"t": 1})]
    B_bad = [(-beta * bsg, bax, {"t": 1})]

    def txt(E, B):
        return f"$\\mathbf{{E}} = {terms_tex(E)}\\ \\mathrm{{V/m}},\\quad \\mathbf{{B}} = {terms_tex(B)}\\ \\mathrm{{T}}$"
    choices = [
        ch(txt(E_ok, B_ok), True, r"Right: $\mathbf{E} = -\nabla\Phi-\partial\mathbf{A}/\partial t$ and $\mathbf{B} = \nabla\times\mathbf{A}$."),
        ch(txt(E_plus, B_ok), False, r"Sign: $\mathbf{E} = -\nabla\Phi\,-\,\partial\mathbf{A}/\partial t$; the induced part enters with a minus sign."),
        ch(txt(E_stat, B_ok), False, r"With time variation $\mathbf{E}\neq-\nabla\Phi$: the term $-\partial\mathbf{A}/\partial t$ is the induced, circulating part of $\mathbf{E}$."),
        ch(txt(E_ok, B_bad), False, f"Recheck the curl: $\\nabla\\times(A_{p}\\hat{{{p}}}) = \\nabla A_{p}\\times\\hat{{{p}}}$, and $\\hat{{{q}}}\\times\\hat{{{p}}} = {uhat(bax, bsg)}$."),
    ]
    A_tex = terms_tex([(beta, p, {q: 1, "t": 1})])
    stem = (f"In some region the potentials are $\\Phi = {alpha}\\,{x1}^{{2}}\\ \\mathrm{{V}}$ and $\\mathbf{{A}} = {A_tex}\\ \\mathrm{{Wb/m}}$ "
            "(coordinates in m, $t$ in s). The fields there are")
    sol = (f"$-\\nabla\\Phi = {terms_tex([(-2 * alpha, x1, {x1: 1})])}$ and $-\\partial\\mathbf{{A}}/\\partial t = {terms_tex([(-beta, p, {q: 1})])}$, "
           f"so $\\mathbf{{E}} = {terms_tex(E_ok)}\\ \\mathrm{{V/m}}$ (static, although $\\mathbf{{A}}$ changes). "
           f"$\\mathbf{{B}} = \\nabla\\times\\mathbf{{A}} = \\nabla({beta}\\,t\\,{q})\\times\\hat{{{p}}} = {beta}\\,t\\,\\hat{{{q}}}\\times\\hat{{{p}}} = {terms_tex(B_ok)}\\ \\mathrm{{T}}$, growing linearly in time.")
    data = {"alpha": alpha, "x1": x1, "beta": beta, "q": q, "p": p}
    return finish("potentials", stem, choices, sol, data)


def g_sheetH():
    while True:
        w = random.choice(AXES)
        tang = [x for x in AXES if x != w]
        Js = [0, 0, 0]
        Js[IDX[tang[0]]] = random.choice([-4, -3, -2, -1, 1, 2, 3, 4])
        Js[IDX[tang[1]]] = random.choice([-3, -2, 0, 0, 2, 3])
        Hg = [random.choice([-4, -3, -2, -1, 0, 1, 2, 3, 4]) for _ in range(3)]
        Hg[IDX[w]] = random.choice([-5, -3, -2, 2, 3, 5])
        given = random.choice(["below", "above"])
        exam_like = (w == "z" and Js == [0, 2, 0] and given == "below" and Hg == [0, 0, 3])
        if not exam_like:
            break
    nw = [0, 0, 0]
    nw[IDX[w]] = 1
    jx = [Js[1] * nw[2] - Js[2] * nw[1], Js[2] * nw[0] - Js[0] * nw[2], Js[0] * nw[1] - Js[1] * nw[0]]   # J_s x w_hat
    sg = 1 if given == "below" else -1            # asked = given + sg * (J_s x w_hat)
    H_ok = [Hg[i] + sg * jx[i] for i in range(3)]
    H_rev = [Hg[i] - sg * jx[i] for i in range(3)]
    H_half = [Hg[i] + sg * 0.5 * jx[i] for i in range(3)]
    H_J = [Hg[i] + sg * Js[i] for i in range(3)]
    asked = "above" if given == "below" else "below"
    rel = {"below": r"\lt", "above": r"\gt"}
    choices = [
        ch(f"$\\mathbf{{H}} = {vec_tex(H_ok, 'A/m')}$", True, r"Right: $\hat{n}\times(\mathbf{H}_1-\mathbf{H}_2) = \mathbf{J}_s$ with $\hat{n}$ from 2 into 1, and $H_n$ continuous."),
        ch(f"$\\mathbf{{H}} = {vec_tex(H_rev, 'A/m')}$", False,
           f"Sign slip: with $\\hat{{n}} = +\\hat{{{w}}}$ pointing from medium 2 (${w}\\lt 0$) into medium 1 (${w}\\gt 0$), $\\mathbf{{H}}_{{1t}} = \\mathbf{{H}}_{{2t}} + \\mathbf{{J}}_s\\times\\hat{{n}}$."),
        ch(f"$\\mathbf{{H}} = {vec_tex(H_half, 'A/m')}$", False,
           r"$\tfrac12\mathbf{J}_s\times\hat{n}$ is the field an isolated sheet makes on each side; the jump across the sheet is the full $\mathbf{J}_s\times\hat{n}$."),
        ch(f"$\\mathbf{{H}} = {vec_tex(H_J, 'A/m')}$", False,
           r"The jump in tangential $\mathbf{H}$ is perpendicular to $\mathbf{J}_s$ (it is $\mathbf{J}_s\times\hat{n}$), not along it."),
    ]
    stem = (f"The plane ${w} = 0$ carries the surface current $\\mathbf{{J}}_s = {vec_tex(Js, 'A/m')}$, and the same non-magnetic medium fills both sides. "
            f"Just {given} the plane (${w} {rel[given]} 0$), $\\mathbf{{H}} = {vec_tex(Hg, 'A/m')}$. Just {asked} it (${w} {rel[asked]} 0$),")
    sol = (f"Let medium 1 be ${w}\\gt 0$ and medium 2 be ${w}\\lt 0$, so $\\hat{{n}} = +\\hat{{{w}}}$. Then $\\hat{{n}}\\times(\\mathbf{{H}}_1-\\mathbf{{H}}_2) = \\mathbf{{J}}_s$ "
           f"gives $\\mathbf{{H}}_{{1t}}-\\mathbf{{H}}_{{2t}} = \\mathbf{{J}}_s\\times\\hat{{n}} = {vec_tex(jx, 'A/m')}$, and $H_{w}$ is continuous (same $\\mu$, $B_n$ continuous). "
           f"So just {asked}, $\\mathbf{{H}} = {vec_tex(H_ok, 'A/m')}$.")
    data = {"w": w, "Js": Js, "given": given, "Hg": Hg}
    return finish("sheetH", stem, choices, sol, data)


def g_phase():
    eps_r = random.choice([2.25, 4.0, 9.0])
    n = math.sqrt(eps_r)
    v = C0 / n
    a = random.choice([F(1, 2), F(1), F(3, 2), F(2), F(5, 2), F(3), F(4), F(5), F(6)])
    medium = f"a lossless, non-magnetic medium with $\\epsilon_r = {numstr(eps_r)}$"
    api = ("" if a == 1 else ftex(a)) + "\\pi"                  # a*pi as LaTeX (no "1\\pi")
    if random.random() < 0.5:
        while True:
            dtv, unit = random.choice([(2, "ns"), (4, "ns"), (5, "ns"), (10, "ns"), (20, "ns"), (25, "ns"),
                                       (2, "μs"), (5, "μs"), (8, "μs"), (10, "μs"), (20, "μs")])
            if not (a == 3 and dtv == 10 and unit == "μs"):
                break
        dt = dtv * (1e-9 if unit == "ns" else 1e-6)
        utex = r"\mathrm{ns}" if unit == "ns" else r"\mu\mathrm{s}"
        T = 2 * dt / float(a)                       # period
        lam = v * T
        cands = [(C0 * T, r"That is the wavelength in vacuum; in the medium the wave is slower, $v = c/\sqrt{\epsilon_r}$, and $\lambda = vT$."),
                 (v * dt / (float(a) * math.pi), r"$\Delta\phi/\Delta t$ is the angular frequency $\omega$ in rad/s; the frequency is $f = \omega/2\pi$."),
                 (v * dt, r"$\Delta t$ is not one period here: the phase advanced by " + f"${api}$" + r" rad, not $2\pi$."),
                 (lam / 2, r"A phase change of $2\pi$ (not $\pi$) is one period: $T = 2\pi\,\Delta t/\Delta\phi$."),
                 (C0 * dt / (float(a) * math.pi), r"Two slips: $\Delta\phi/\Delta t$ is $\omega$, not $f$, and the medium slows the wave to $c/\sqrt{\epsilon_r}$.")]
        right = lam
        unit_ans = "m"
        stem = (f"At a fixed point in {medium}, the phase of a sinusoidal plane wave advances by ${api}$ rad in ${dtv}\\ {utex}$. "
                "Its wavelength is")
        sol = (f"$\\omega = \\Delta\\phi/\\Delta t$, so the period is $T = 2\\pi\\,\\Delta t/\\Delta\\phi = {g3(T * 1e9)}\\ \\mathrm{{ns}}$. "
               f"In the medium $v = c/\\sqrt{{\\epsilon_r}} = {g3(v)}\\ \\mathrm{{m/s}}$, so $\\lambda = vT = {g3(lam)}\\ \\mathrm{{m}}$.")
        data = {"mode": "time", "a": str(a), "dt": dt, "eps_r": eps_r}
    else:
        dd = random.choice([0.5, 1, 1.5, 2, 3, 4, 5, 10, 20])
        lam = 2 * dd / float(a)
        f = v / lam
        cands = [(C0 / lam, r"That uses the vacuum speed; in the medium $f = v/\lambda$ with $v = c/\sqrt{\epsilon_r}$."),
                 (v * float(a) * math.pi / dd, r"$\Delta\phi/\Delta d$ is the phase constant $\beta$ in rad/m; $f = v\beta/2\pi$, not $v\beta$."),
                 (v / dd, r"The distance is not one wavelength here: the phase changed by " + f"${api}$" + r" rad, not $2\pi$."),
                 (v * float(a) / dd, r"A phase change of $2\pi$ (not $\pi$) is one wavelength: $\lambda = 2\pi\,\Delta d/\Delta\phi$."),
                 (C0 * float(a) * math.pi / dd, r"Two slips: $\Delta\phi/\Delta d$ is $\beta$ (so $f = v\beta/2\pi$), and the medium slows the wave to $c/\sqrt{\epsilon_r}$.")]
        right = f
        unit_ans = "MHz"
        stem = (f"At one instant, the phase of a sinusoidal plane wave in {medium} changes by ${api}$ rad over a distance of "
                f"${numstr(dd)}\\ \\mathrm{{m}}$ along its direction of travel. Its frequency is")
        sol = (f"$\\beta = \\Delta\\phi/\\Delta d$, so $\\lambda = 2\\pi\\,\\Delta d/\\Delta\\phi = {g3(lam)}\\ \\mathrm{{m}}$. "
               f"With $v = c/\\sqrt{{\\epsilon_r}} = {g3(v)}\\ \\mathrm{{m/s}}$, $f = v/\\lambda = {g3(f / 1e6)}\\ \\mathrm{{MHz}}$.")
        data = {"mode": "space", "a": str(a), "dd": dd, "eps_r": eps_r}
    scale = 1.0 if unit_ans == "m" else 1e-6
    choices = [ch(f"${g3(right * scale)}\\ \\mathrm{{{unit_ans}}}$", True, "Right.")]
    used = [right * scale]
    for val, fb in cands:
        add_numeric(choices, used, val * scale, f"${g3(val * scale)}\\ \\mathrm{{{unit_ans}}}$", fb)
    return finish("phase", stem, choices, sol, data)


def g_drude():
    while True:
        a = random.choice([F(2), F(3), F(4), F(1, 2)])
        b = random.choice([F(2), F(3), F(1, 2), F(1, 3)])
        c = random.choice([F(2), F(3), F(1, 2)])
        vals = [a * b / c, a / (b * c), a * b * c, b / c]
        if len(set(vals)) == 4:
            break
    word = lambda q: ftex(q)
    stem = (f"Two conductors A and B obey the Drude model, with carriers of the same charge. B has ${word(a)}$ times the carrier density of A, "
            f"${word(b)}$ times its collision time $\\tau$, and ${word(c)}$ times its carrier (effective) mass. "
            r"The ratio $\sigma_B/\sigma_A$ of their DC conductivities is")
    choices = [
        ch(f"${ftex(vals[0])}$", True, r"Right: $\sigma = Nq^2\tau/m$."),
        ch(f"${ftex(vals[1])}$", False, r"$\sigma = Nq^2\tau/m$ grows with the collision time: a longer free flight lets a carrier pick up more drift speed. You treated $\tau$ like the collision rate $\nu = 1/\tau$."),
        ch(f"${ftex(vals[2])}$", False, r"The mass is in the denominator: heavier carriers accelerate less between collisions."),
        ch(f"${ftex(vals[3])}$", False, r"That is the ratio of the mobilities $q\tau/m$; the conductivity also counts the carriers, $\sigma = Nq\times$ mobility."),
    ]
    sol = (f"$\\sigma = Nq^2\\tau/m$, so $\\sigma_B/\\sigma_A = ({ftex(a)})({ftex(b)})/({ftex(c)}) = {ftex(vals[0])}$.")
    data = {"a": str(a), "b": str(b), "c": str(c)}
    return finish("drude", stem, choices, sol, data)


def g_magcur():
    M0 = random.choice([50, 120, 200, 250, 400, 800])           # kA/m
    eax = random.choice(AXES)
    esg = random.choice([1, -1])
    parallel = random.random() < 0.3
    if parallel:
        nax, nsg = eax, random.choice([1, -1])
    else:
        nax, nsg = random.choice([x for x in AXES if x != eax]), random.choice([1, -1])
    Mv = [0, 0, 0]
    Mv[IDX[eax]] = esg * M0
    nv = [0, 0, 0]
    nv[IDX[nax]] = nsg
    J = [Mv[1] * nv[2] - Mv[2] * nv[1], Mv[2] * nv[0] - Mv[0] * nv[2], Mv[0] * nv[1] - Mv[1] * nv[0]]
    side = random.choice([2, 3, 5])
    stem = (f"A cube of side ${side}\\ \\mathrm{{cm}}$ is uniformly magnetized, $\\mathbf{{M}} = {vec_tex(Mv, 'kA/m')}$, and carries no free current. "
            f"On its face with outward normal $\\hat{{n}} = {uhat(nax, nsg)}$, the magnetization surface current density is")
    if not parallel:
        Jrev = [-x for x in J]
        Jalong = Mv
        choices = [
            ch(f"$\\mathbf{{J}}_{{sM}} = {vec_tex(J, 'kA/m')}$", True, r"Right: $\mathbf{J}_{sM} = \mathbf{M}\times\hat{n}$."),
            ch(f"$\\mathbf{{J}}_{{sM}} = {vec_tex(Jrev, 'kA/m')}$", False, r"Order: $\mathbf{J}_{sM} = \mathbf{M}\times\hat{n}$, not $\hat{n}\times\mathbf{M}$."),
            ch(r"$\mathbf{J}_{sM} = \mathbf{0}$, because a uniform $\mathbf{M}$ has $\nabla\times\mathbf{M} = \mathbf{0}$", False,
               r"There is no volume current, but $\mathbf{M}$ stops at the surface, and that edge carries $\mathbf{M}\times\hat{n}$."),
            ch(f"$\\mathbf{{J}}_{{sM}} = {vec_tex(Jalong, 'kA/m')}$", False, r"$\mathbf{M}\times\hat{n}$ is perpendicular to $\mathbf{M}$: the bound current circulates around the magnetization."),
        ]
    else:
        tax = random.choice([x for x in AXES if x != eax])
        t1v = [0, 0, 0]
        t1v[IDX[tax]] = M0
        t2v = [-x for x in t1v]
        nM = [M0 * x for x in nv]
        choices = [
            ch(r"$\mathbf{J}_{sM} = \mathbf{0}$", True, r"Right: on this face $\hat{n}\parallel\mathbf{M}$, so $\mathbf{M}\times\hat{n} = \mathbf{0}$."),
            ch(f"$\\mathbf{{J}}_{{sM}} = {vec_tex(t1v, 'kA/m')}$", False, r"$\mathbf{M}\times\hat{n} = \mathbf{0}$ when $\mathbf{M}$ is parallel to $\hat{n}$: faces perpendicular to $\mathbf{M}$ carry no bound current."),
            ch(f"$\\mathbf{{J}}_{{sM}} = {vec_tex(t2v, 'kA/m')}$", False, r"$\mathbf{M}\times\hat{n} = \mathbf{0}$ when $\mathbf{M}$ is parallel to $\hat{n}$: faces perpendicular to $\mathbf{M}$ carry no bound current."),
            ch(f"$\\mathbf{{J}}_{{sM}} = {vec_tex(nM, 'kA/m')}$", False, r"That is the bound-charge rule $\mathbf{P}\cdot\hat{n}$ carried over; the magnetization current is $\mathbf{M}\times\hat{n}$, which vanishes here."),
        ]
    sol = (f"$\\mathbf{{J}}_{{sM}} = \\mathbf{{M}}\\times\\hat{{n}} = ({vec_tex(Mv)})\\times({uhat(nax, nsg)}) = {vec_tex(J, 'kA/m')}$."
           + (" The face is perpendicular to $\\mathbf{M}$, so there is no bound current on it; the current flows on the four side faces." if parallel else ""))
    data = {"M0": M0, "eax": eax, "esg": esg, "nax": nax, "nsg": nsg}
    return finish("magcur", stem, choices, sol, data)


def g_disp():
    eps_r = random.choice([2.25, 3.0, 4.0, 5.0])
    E0 = random.choice([10, 20, 50, 100, 200, 500])
    f_MHz = random.choice([1, 10, 50, 100, 500, 1000])
    om = 2 * math.pi * f_MHz * 1e6
    Jd = om * EPS0 * eps_r * E0                                 # A/m^2
    sc, un = (1.0, "A/m^2") if Jd >= 1.0 else (1e3, "mA/m^2")     # one unit for all four choices
    cands = [(om * EPS0 * E0, r"In the dielectric $\mathbf{D} = \epsilon_r\epsilon_0\mathbf{E}$; you kept only the vacuum part $\epsilon_0\,\partial\mathbf{E}/\partial t$."),
             (om * EPS0 * (eps_r - 1) * E0, r"That is only the polarization current $\partial\mathbf{P}/\partial t$; the displacement current $\partial\mathbf{D}/\partial t$ also contains $\epsilon_0\,\partial\mathbf{E}/\partial t$."),
             (f_MHz * 1e6 * EPS0 * eps_r * E0, r"The time derivative of $\cos(2\pi ft)$ brings down $\omega = 2\pi f$, not $f$.")]
    choices = [ch(f"${g3(Jd * sc)}\\ \\mathrm{{{un}}}$", True, r"Right: $\lvert\partial\mathbf{D}/\partial t\rvert_{\max} = \omega\epsilon_r\epsilon_0E_0$.")]
    for val, fb in cands:
        choices.append(ch(f"${g3(val * sc)}\\ \\mathrm{{{un}}}$", False, fb))
    ftxt = f"{f_MHz}\\ \\mathrm{{MHz}}" if f_MHz < 1000 else "1\\ \\mathrm{GHz}"
    stem = (f"A lossless dielectric with $\\epsilon_r = {numstr(eps_r)}$ carries the uniform field $\\mathbf{{E}} = {E0}\\cos(2\\pi ft)\\,\\hat{{x}}\\ \\mathrm{{V/m}}$ "
            f"with $f = {ftxt}$. The amplitude of the displacement current density $\\partial\\mathbf{{D}}/\\partial t$ is")
    sol = (f"$\\mathbf{{D}} = \\epsilon_r\\epsilon_0\\mathbf{{E}}$, so $\\partial\\mathbf{{D}}/\\partial t = -\\omega\\epsilon_r\\epsilon_0E_0\\sin(\\omega t)\\,\\hat{{x}}$, of amplitude "
           f"$\\omega\\epsilon_r\\epsilon_0E_0 = (2\\pi\\times{g3(f_MHz * 1e6)})({numstr(eps_r)})(8.854\\times10^{{-12}})({E0}) = {g3(Jd * sc)}\\ \\mathrm{{{un}}}$.")
    data = {"eps_r": eps_r, "E0": E0, "f_MHz": f_MHz}
    return finish("disp", stem, choices, sol, data)


def g_etav():
    mur, epsr = random.choice([(1, 2.25), (1, 4), (1, 6.25), (1, 9), (2, 8), (4, 9), (3, 12), (2, 18)])
    eta = ETA0 * math.sqrt(mur / epsr)
    v = C0 / math.sqrt(mur * epsr)

    def txt(e, vv):
        return f"$\\eta = {g3(e)}\\ \\Omega,\\quad v = {g3(vv)}\\ \\mathrm{{m/s}}$"
    right = txt(eta, v)
    cands = [(ETA0 * math.sqrt(epsr / mur), v, r"Inverted: $\eta = \sqrt{\mu/\epsilon} = \eta_0\sqrt{\mu_r/\epsilon_r}$, so a larger $\epsilon_r$ lowers $\eta$."),
             (ETA0 / math.sqrt(mur * epsr), v, r"$\eta$ depends on the ratio $\mu_r/\epsilon_r$; it is $v$ that depends on the product."),
             (eta, C0 * math.sqrt(mur / epsr), r"$v = 1/\sqrt{\mu\epsilon} = c/\sqrt{\mu_r\epsilon_r}$: both $\mu_r$ and $\epsilon_r$ slow the wave."),
             (eta, C0 / (mur * epsr), r"$v = c/\sqrt{\mu_r\epsilon_r}$: keep the square root."),
             (eta, C0 * math.sqrt(mur * epsr), r"Matter slows the wave: $v = c/\sqrt{\mu_r\epsilon_r}$.")]
    choices = [ch(right, True, r"Right: $\eta = \eta_0\sqrt{\mu_r/\epsilon_r}$ and $v = c/\sqrt{\mu_r\epsilon_r}$.")]
    used = {right}
    for e, vv, fb in cands:
        t = txt(e, vv)
        if t not in used and len(choices) < 4:
            used.add(t)
            choices.append(ch(t, False, fb))
    stem = f"A lossless medium has $\\mu_r = {mur}$ and $\\epsilon_r = {numstr(epsr)}$. The intrinsic impedance and the speed of a plane wave in it are"
    sol = (f"$\\eta = \\eta_0\\sqrt{{\\mu_r/\\epsilon_r}} = 376.7\\sqrt{{{mur}/{numstr(epsr)}}} = {g3(eta)}\\ \\Omega$ and "
           f"$v = c/\\sqrt{{\\mu_r\\epsilon_r}} = (2.998\\times10^{{8}})/\\sqrt{{{numstr(mur * epsr)}}} = {g3(v)}\\ \\mathrm{{m/s}}$.")
    data = {"mur": mur, "epsr": epsr}
    return finish("etav", stem, choices, sol, data)


def g_poynting():
    if random.random() < 0.5:
        e1, e2 = random.sample(AXES, 2)
        sa, sb = random.choice([1, -1]), random.choice([1, -1])
        a = random.choice([2, 3, 5, 6, 8, 10])
        bmA = a / ETA0 * 1e3
        tax, tsg = cross_ax(e1, e2)
        tsg *= sa * sb
        Ev = [0, 0, 0]
        Ev[IDX[e1]] = sa * a
        stem = (f"At one point of a plane wave in free space, at one instant, $\\mathbf{{E}} = {vec_tex(Ev, 'V/m')}$ and "
                f"$\\mathbf{{H}} = {'-' if sb < 0 else ''}{g3(bmA)}\\,\\hat{{{e2}}}\\ \\mathrm{{mA/m}}$. The wave travels toward")
        choices = [
            ch(f"${uhat(tax, tsg)}$", True, r"Right: the wave travels along $\mathbf{E}\times\mathbf{H}$."),
            ch(f"${uhat(tax, -tsg)}$", False, r"That is $\mathbf{H}\times\mathbf{E}$: the order matters, and the travel is along $\mathbf{E}\times\mathbf{H}$."),
            ch(f"${uhat(e1, sa)}$ (along $\\mathbf{{E}}$)", False, r"A plane wave is transverse: it travels perpendicular to both $\mathbf{E}$ and $\mathbf{H}$."),
            ch(f"${uhat(e2, sb)}$ (along $\\mathbf{{H}}$)", False, r"A plane wave is transverse: it travels perpendicular to both $\mathbf{E}$ and $\mathbf{H}$."),
        ]
        sol = (f"The travel direction is that of $\\mathbf{{E}}\\times\\mathbf{{H}}$: $({uhat(e1, sa)})\\times({uhat(e2, sb)}) = {uhat(tax, tsg)}$. "
               f"(Check: $\\lvert\\mathbf{{E}}\\rvert/\\lvert\\mathbf{{H}}\\rvert = {a}/({g3(bmA)}\\times10^{{-3}}) \\approx 377\\ \\Omega = \\eta_0$.)")
        data = {"sub": "travel", "e1": e1, "sa": sa, "a": a, "e2": e2, "sb": sb}
    else:
        u, e = random.sample(AXES, 2)
        su, sa = random.choice([1, -1]), random.choice([1, -1])
        a = random.choice([2, 3, 5, 6, 8, 10])
        hax, hsg = cross_ax(u, e)
        hsg *= su * sa
        Ev = [0, 0, 0]
        Ev[IDX[e]] = sa * a
        stem = (f"A plane wave in free space travels toward ${uhat(u, su)}$. At one point and instant $\\mathbf{{E}} = {vec_tex(Ev, 'V/m')}$. "
                r"There, $\mathbf{H}$ points along")
        choices = [
            ch(f"${uhat(hax, hsg)}$", True, r"Right: $\mathbf{H} = \hat{u}\times\mathbf{E}/\eta_0$."),
            ch(f"${uhat(hax, -hsg)}$", False, r"Reversed: with this $\mathbf{H}$, $\mathbf{E}\times\mathbf{H}$ would point against the travel."),
            ch(f"${uhat(u, su)}$ (along the travel)", False, r"A plane wave has no field component along its direction of travel."),
            ch(f"${uhat(e, sa)}$ (along $\\mathbf{{E}}$)", False, r"In a plane wave $\mathbf{H}\perp\mathbf{E}$."),
        ]
        sol = (f"$\\mathbf{{H}} = \\hat{{u}}\\times\\mathbf{{E}}/\\eta_0$: $({uhat(u, su)})\\times({uhat(e, sa)}) = {uhat(hax, hsg)}$, with magnitude "
               f"${a}/376.7 = {g3(a / ETA0 * 1e3)}\\ \\mathrm{{mA/m}}$. Check: $\\mathbf{{E}}\\times\\mathbf{{H}}$ points along ${uhat(u, su)}$.")
        data = {"sub": "hdir", "u": u, "su": su, "e": e, "sa": sa, "a": a}
    return finish("poynting", stem, choices, sol, data)


GENERATORS = [g_wires, g_loop, g_magnet, g_dcond, g_potentials, g_sheetH, g_phase, g_drude, g_magcur, g_disp, g_etav, g_poynting]


def generate(data):
    gens = random.sample(GENERATORS, 6)
    items = []
    for i, gfun in enumerate(gens, start=1):
        it = gfun()
        it["name"] = f"q{i}"
        it["num"] = i
        items.append(it)
    data["params"]["items"] = items
