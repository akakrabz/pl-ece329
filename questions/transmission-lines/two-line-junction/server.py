"""Lecture 30 concept question: a step through R_g into line 1 (Z1, T1), a junction (optionally a shunt resistor R
across it), line 2 (Z2, T2) and a load R_L.

Coefficients (Lecture 30 page): the arriving wave sees Zeq (line k, in parallel with a shunt R if present)
    Gamma_12 = (Zeq12 - Z1)/(Zeq12 + Z1), Zeq12 = Z2 or R||Z2;  tau_12 = 1 + Gamma_12
    Gamma_21 = (Zeq21 - Z2)/(Zeq21 + Z2), Zeq21 = Z1 or R||Z1;  tau_21 = 1 + Gamma_21
    tau_g = Z1/(Rg+Z1), Gamma_g = (Rg-Z1)/(Rg+Z1) (line 1), Gamma_L = (RL-Z2)/(RL+Z2) (line 2; open +1)
The bounce lattice is enumerated exactly (fractions; integer one-way delays in us). The junction voltage jumps by
tau_12 w for a wave w arriving from line 1 and by tau_21 w for one arriving from line 2. DC: both lines are
wires, so V_L = V0 (R||RL)/(Rg + R||RL).
"""
import random
from collections import defaultdict
from fractions import Fraction as Fr

Z1S = [50, 60, 75, 100]
ZR = [Fr(1, 3), Fr(1, 2), Fr(2, 3), Fr(3, 2), Fr(2), Fr(3)]
RG_R = [Fr(1, 3), Fr(1, 2), Fr(1), Fr(2), Fr(3)]
RL_R = [Fr(1, 3), Fr(1, 2), Fr(1), Fr(2), Fr(3)]
SH_R = [Fr(1, 2), Fr(1), Fr(2), Fr(3)]               # shunt as a multiple of Z2 (Z1 also allowed)
VS = [100, 150, 200, 300]                             # m/us
V0S = [10, 12, 15, 18, 20, 24, 30, 36]
# (Rg/Z1, Z2/Z1, RL/Z2) of the notes' two-line circuit and the site's worked problem (no shunt)
BLACK = {(Fr(2), Fr(2), Fr(1)), (Fr(1), Fr(3), Fr(3))}
LINE = {"1f": "line 1, toward the junction", "1b": "line 1, toward the generator",
        "2f": "line 2, toward the load", "2b": "line 2, toward the junction"}


def num(x, sig=4):
    x = float(x)
    if abs(x) < 1e-12:
        return "0"
    s = f"{x:.{sig}g}"
    if "e" in s:
        m, e = s.split("e")
        return f"{m}\\times10^{{{int(e)}}}"
    return s


def r3(q):
    return float(f"{float(q):.3g}")


def tol_ok(terms, floor, rtol=0.01):
    """Tolerance for a sum of wave terms (display units). atol covers 3-s.f. rounding of each term; the variant
    is kept only if a 3-s.f. chain is accepted and dropping any one term is rejected."""
    a = sum(terms, Fr(0))
    atol = max(floor, 0.003 * float(sum(abs(t) for t in terms)))
    tol = atol + rtol * abs(float(a))
    chain = sum(r3(t) for t in terms)
    ok = abs(chain - float(a)) <= 0.8 * tol and all(abs(float(t)) > 1.5 * tol for t in terms if t != 0)
    return ok, atol


def tnum(x):
    """num() for HTML table cells: exponent forms go inside math."""
    s = num(x)
    return f"${s}$" if "times" in s else s


def ftex(q):
    q = Fr(q)
    if q.denominator == 1:
        return str(q.numerator)
    sg = "-" if q < 0 else ""
    return f"{sg}\\tfrac{{{abs(q.numerator)}}}{{{q.denominator}}}"


def par(a, b):
    return a * b / (a + b)


def lattice(V0, c, T1, T2, tmax):
    """Exact bounce lattice. Returns waves {(type, t_launch): weight} and junction arrivals
    [(t, side, w_arriving, coefficient name, jump)], both including zero-weight (structural) entries."""
    waves = defaultdict(Fr)
    waves[("1f", 0)] = c["tg"] * V0
    junc = []
    for t in range(0, tmax + 1):
        for typ in ("1f", "1b", "2f", "2b"):
            if (typ, t) not in waves:
                continue
            w = waves[(typ, t)]
            ta = t + (T1 if typ[0] == "1" else T2)
            if ta > tmax:
                continue
            if typ == "1f":
                junc.append((ta, "line 1", w, r"\tau_{12}", c["t12"] * w))
                waves[("1b", ta)] += c["G12"] * w
                waves[("2f", ta)] += c["t12"] * w
            elif typ == "2b":
                junc.append((ta, "line 2", w, r"\tau_{21}", c["t21"] * w))
                waves[("2f", ta)] += c["G21"] * w
                waves[("1b", ta)] += c["t21"] * w
            elif typ == "1b":
                waves[("1f", ta)] += c["Gg"] * w
            else:
                waves[("2b", ta)] += c["GL"] * w
    junc.sort(key=lambda r: (r[0], r[1]))
    return waves, junc


def generate(data):
    while True:
        Z1 = random.choice(Z1S)
        r = random.choice(ZR)
        Z2 = Z1 * r
        shunt = random.random() < 0.4
        Rg = Z1 * random.choice(RG_R)
        openL = random.random() < 0.12
        RL = None if openL else Z2 * random.choice(RL_R)
        R = None
        if shunt:
            R = random.choice([k * Z2 for k in SH_R] + [Fr(Z1)])
        vals = [Z2, Rg] + ([RL] if RL is not None else []) + ([R] if R is not None else [])
        if any(Fr(x).denominator != 1 for x in vals):
            continue
        if not shunt and RL is not None and (Rg / Z1, r, RL / Z2) in BLACK:
            continue
        Zeq12 = Z2 if R is None else par(R, Z2)
        Zeq21 = Fr(Z1) if R is None else par(R, Fr(Z1))
        if Zeq12 == Z1:
            continue                                   # matched junction: no echo, (e) trivial
        c = {"tg": Fr(Z1) / (Rg + Z1), "Gg": Fr(Rg - Z1) / (Rg + Z1),
             "GL": Fr(1) if RL is None else Fr(RL - Z2) / (RL + Z2),
             "G12": (Zeq12 - Z1) / (Zeq12 + Z1), "G21": (Zeq21 - Z2) / (Zeq21 + Z2)}
        c["t12"], c["t21"] = 1 + c["G12"], 1 + c["G21"]
        T1, T2 = random.sample([1, 2, 3], 2)
        s2 = max(3 * T1, T1 + 2 * T2)
        V0 = random.choice(V0S)
        waves, junc = lattice(V0, c, T1, T2, s2 + 2 * max(T1, T2) + 1)
        times = sorted({j[0] for j in junc})
        if len([t for t in times if t <= s2]) > 5:
            continue
        t_next = min(t for t in times if t > s2)
        tc = Fr(s2 + t_next, 2)
        VJ = sum(j[4] for j in junc if j[0] < tc)
        # avoid tiny nonzero junction voltage (rtol meaningful)
        if VJ != 0 and abs(VJ) < Fr(1, 50) * V0:
            continue
        if sum(1 for j in junc if j[0] < tc and j[2] != 0) < 2:
            continue                                   # (c) must involve at least one echo
        okJ, atolJ = tol_ok([j[4] for j in junc if j[0] < tc and j[4] != 0], 2e-4 * V0)
        if not okJ:
            continue
        Rdc = (RL if R is None else (R if RL is None else par(R, RL)))
        VLdc = Fr(V0) if Rdc is None else V0 * Rdc / (Rg + Rdc)
        VJn = VJ                                                    # next distinct junction level
        for tn in sorted(t for t in times if t > s2):
            VJn = sum(j[4] for j in junc if j[0] <= tn)
            if VJn != VJ:
                break
        tolJ = atolJ + 0.01 * abs(float(VJ))
        if any(b != VJ and abs(float(b - VJ)) <= 1.5 * tolJ for b in (VLdc, VJn)):
            continue                                   # no credit for the DC shortcut or an extra echo
        break

    v1, v2 = random.choice(VS), random.choice(VS)
    ell1, ell2 = v1 * T1, v2 * T2
    V2first = c["t12"] * c["tg"] * V0
    V1refl = c["G12"] * c["tg"] * V0

    # ---------------------------------------------------------------- (e) matching resistor MC
    D = Z2 - Z1
    Rsh_m = Fr(Z1) * Z2 / abs(D)
    Rse_m = abs(D)
    cand = [
        ("sh_ok", f"a resistor of {num(Rsh_m)} Ω connected across the junction (between the two conductors)", D > 0,
         (f"Right: the arriving wave sees $R\\parallel Z_2 = {num(Rsh_m)}\\parallel{num(Z2)} = {Z1}\\ \\Omega = Z_1$." if D > 0 else
          f"A shunt resistor can only lower the impedance below $Z_2 = {num(Z2)}\\ \\Omega$, which is already less than $Z_1$; here $R\\parallel Z_2 = {num(par(Rsh_m, Z2))}\\ \\Omega$.")),
        ("se_ok", f"a resistor of {num(Rse_m)} Ω inserted in series in the upper conductor at the junction", D < 0,
         (f"Right: the arriving wave sees $R + Z_2 = {num(Rse_m)} + {num(Z2)} = Z_1$." if D < 0 else
          f"A series resistor adds to $Z_2 = {num(Z2)}\\ \\Omega$, which already exceeds $Z_1$; the wave would see $R + Z_2 = {num(Rse_m + Z2)}\\ \\Omega$.")),
        ("sh_Z1", f"a resistor of {Z1} Ω connected across the junction (between the two conductors)", False,
         f"The wave sees $R\\parallel Z_2 = {num(par(Fr(Z1), Z2))}\\ \\Omega$, not $Z_1 = {Z1}\\ \\Omega$."),
        ("sh_diff", f"a resistor of {num(Rse_m)} Ω connected across the junction (between the two conductors)", False,
         f"Across the junction it gives $R\\parallel Z_2 = {num(par(Rse_m, Z2))}\\ \\Omega$, not $Z_1$; $\\lvert Z_2 - Z_1\\rvert$ is what a series resistor adds."),
        ("none", "none: no resistor at the junction can do it; only making $R_L = Z_2$ removes the echo toward the generator", False,
         "$R_L$ controls the echo from the far end only. The junction echo depends on the impedance the arriving wave sees at the junction, which a resistor there changes."),
    ]
    e_choices, seen = [], set()
    for cid, txt, ok, fb in cand:
        if txt in seen:
            continue
        seen.add(txt)
        e_choices.append({"id": cid, "text": txt, "correct": "true" if ok else "false", "feedback": fb,
                          "R": float(Rsh_m if cid == "sh_ok" else (Rse_m if cid in ("se_ok", "sh_diff") else (Z1 if cid == "sh_Z1" else -1))),
                          "conn": "shunt" if cid.startswith("sh") else ("series" if cid == "se_ok" else "none")})
    random.shuffle(e_choices)
    e_stem = ("Remove the shunt resistor. Which single resistor, added at the junction, would then make $\\Gamma_{12} = 0$, so that waves arriving from line 1 are not reflected there?"
              if shunt else
              "Which single resistor, added at the junction, would make $\\Gamma_{12} = 0$, so that waves arriving from line 1 are not reflected there?")

    # ---------------------------------------------------------------- (f) statements
    G12, G21, t12, t21 = c["G12"], c["G21"], c["t12"], c["t21"]
    inv_true = (G21 == -G12)
    facts = [
        ("inv", inv_true, "For this junction, $\\Gamma_{21} = -\\Gamma_{12}$.",
         ("Without a resistor, swapping $Z_1$ and $Z_2$ flips the sign of $Z_2 - Z_1$." if inv_true else
          "With the shunt, a wave from line 1 sees $R\\parallel Z_2$ and one from line 2 sees $R\\parallel Z_1$: the two coefficients are not simply opposite.")),
        ("tau1pG", True, "For this junction, $\\tau_{12} = 1 + \\Gamma_{12}$.",
         "The voltage is continuous across the junction" + (" (the shunt sits across the same node)" if shunt else "") + ", so $V^+ + V^- = V^{++}$."),
        ("tau_gt1", True, "A voltage transmission coefficient greater than 1 at a junction does not violate energy conservation.",
         "Power goes as $V^2/Z$: with $Z_2 \\gt Z_1$ a larger voltage carries less power."),
        ("power", not shunt, "For this junction, the power carried onto line 2 plus the reflected power equals the incident power.",
         ("Without a resistor, $\\Gamma_{12}^2 + \\tau_{12}^2Z_1/Z_2 = 1$." if not shunt else
          "The shunt resistor absorbs part of the incident power, so $\\Gamma_{12}^2 + \\tau_{12}^2Z_1/Z_2 \\lt 1$.")),
        ("tautau", False, "For every junction between two lines with no resistor, $\\tau_{12}\\tau_{21} = 1$.",
         "$\\tau_{12}\\tau_{21} = 1-\\Gamma_{12}^2$, which is 1 only if $Z_1 = Z_2$; e.g. $Z_2 = 3Z_1$ gives $\\tfrac32\\cdot\\tfrac12 = \\tfrac34$."),
        ("speeds", False, "The wave speeds $v_1$ and $v_2$ enter $\\Gamma_{12}$ and $\\tau_{12}$.",
         "Only the impedances enter; the speeds set the timing."),
        ("equalZ", True, "If $Z_1 = Z_2$ and there is no resistor at the junction, nothing reflects there, even if $v_1 \\ne v_2$.",
         "$\\Gamma_{12} = (Z_2-Z_1)/(Z_2+Z_1) = 0$ whatever the speeds."),
        ("gen_Z1", True, "$\\tau_g$ and $\\Gamma_g$ are computed with $Z_1$, and $\\Gamma_L$ with $Z_2$.",
         "Each end sees only the line attached to it."),
        ("back_tau12", False, "A wave returning from the load is transmitted into line 1 with the coefficient $\\tau_{12}$.",
         "It travels from line 2 into line 1, so it uses $\\tau_{21}$, which differs from $\\tau_{12}$ whenever $Z_1 \\ne Z_2$."),
        ("Itrans", False, "The current transmitted onto line 2 is $\\tau_{12}$ times the incident current.",
         "Transmitted over incident current is $\\tau_{12}V^+/Z_2$ over $V^+/Z_1$, i.e. $\\tau_{12}Z_1/Z_2$, not $\\tau_{12}$, since $Z_1 \\ne Z_2$."),
        ("dc_equal", True, "As $t\\to\\infty$ the junction voltage equals the load voltage.",
         "At DC both lines are just pairs of wires, so every point of line 2 is at the same voltage."),
        ("Irefl", True, "For a wave arriving from line 1, the reflected current is $-\\Gamma_{12}$ times the incident current.",
         "The reflected wave travels toward $-z$ on line 1, so its current is $-V^-/Z_1 = -\\Gamma_{12}V^+/Z_1$."),
    ]
    fact_list = [{"id": fid, "correct": "true" if t else "false", "text": txt, "feedback": ("True: " if t else "False: ") + why,
                  "verdict": "true" if t else "false", "why": why} for fid, t, txt, why in facts]
    random.shuffle(fact_list)

    # ---------------------------------------------------------------- answer-panel tables
    wave_rows = []
    for (typ, t), wgt in sorted(waves.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if t < tc and wgt != 0:
            wave_rows.append({"t": t, "where": LINE[typ], "w": tnum(wgt)})
    run = Fr(0)
    j_rows = []
    for ta, side, w_arr, cname, jump in junc:
        if ta < tc and w_arr != 0:
            run += jump
            j_rows.append({"t": ta, "side": side, "w": tnum(w_arr), "coef": cname, "jump": tnum(jump), "run": tnum(run)})

    p = data["params"]
    p.update({
        "Z1": Z1, "Z2": float(Z2), "Rg": float(Rg), "RL": (-1 if RL is None else float(RL)), "R": (-1 if R is None else float(R)),
        "shunt": shunt, "V0": V0, "v1": v1, "v2": v2, "ell1": ell1, "ell2": ell2, "T1": T1, "T2": T2,
        "Z2_s": num(Z2), "Rg_s": num(Rg), "R_s": ("" if R is None else num(R)), "ellsum": ell1 + ell2,
        "load_txt": ("is left open" if RL is None else f"ends in a resistor $R_L = {num(RL)}\\ \\Omega$"),
        "shunt_txt": ("" if R is None else f"A resistor $R = {num(R)}\\ \\Omega$ is connected across the junction, between the upper and lower conductors at $z = \\ell_1$. "),
        "tc_s": num(tc), "atolV": f"{2e-4 * V0:.2g}", "atolJ": f"{atolJ:.2g}",
        "e_stem": e_stem, "e_choices": e_choices, "facts": fact_list,
        # panel
        "Zeq12_s": num(Zeq12), "Zeq21_s": num(Zeq21), "G12_tex": ftex(G12), "t12_tex": ftex(t12), "G21_tex": ftex(G21), "t21_tex": ftex(t21),
        "tg_tex": ftex(c["tg"]), "Gg_tex": ftex(c["Gg"]), "GL_tex": ftex(c["GL"]), "Vplus_s": num(c["tg"] * V0),
        "V2f_s": num(V2first), "V1r_s": num(V1refl), "VJ_s": num(VJ), "VL_s": num(VLdc),
        "Rdc_s": ("\\infty" if Rdc is None else num(Rdc)),
        "zeq12_expr": ("Z_2" if R is None else "R\\parallel Z_2"), "zeq21_expr": ("Z_1" if R is None else "R\\parallel Z_1"),
        "dc_expr": ("R_L" if R is None else ("R" if RL is None else "R\\parallel R_L")),
        "s3T1": 3 * T1, "sT12T2": T1 + 2 * T2,
        "gen_zero": (" (zero here, since $\\Gamma_g = 0$)" if c["Gg"] == 0 else ""),
        "load_zero": (" (zero here, since $\\Gamma_L = 0$)" if c["GL"] == 0 else ""), "wave_rows": wave_rows, "j_rows": j_rows,
        "dc_open_note": ("With the load open and no shunt, no current flows at DC, so $V_L = V_0$." if Rdc is None else ""),
    })
    ca = data["correct_answers"]
    ca["G12"] = float(G12) + 0.0
    ca["t12"] = float(t12)
    ca["G21"] = float(G21) + 0.0
    ca["t21"] = float(t21)
    ca["V2f"] = float(V2first)
    ca["V1r"] = float(V1refl) + 0.0
    ca["VJ"] = float(VJ) + 0.0
    ca["VL"] = float(VLdc) + 0.0
