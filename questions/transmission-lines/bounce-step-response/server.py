"""Lecture 28 concept question: step response of a lossless line between a Thevenin generator and a resistive load.

Conventions (Lecture 28 page): z = 0 generator, z = ell load, +z toward the load, one-way delay T = ell/v.
V is the upper conductor (wire 2) relative to the lower (wire 1); I flows along +z on the upper conductor.
    tau_g = Z0/(Rg + Z0),  Gamma_g = (Rg - Z0)/(Rg + Z0),  Gamma_L = (RL - Z0)/(RL + Z0)  (open +1, short -1)
Forward segment n:  weight tau_g (GL Gg)^n,      passes z at  z/v + 2nT,       current +w/Z0
Backward segment n: weight tau_g GL (GL Gg)^n,   passes z at 2T - z/v + 2nT,   current -w/Z0
Step Vg u(t): V(z,t) = Vg * (sum of weights crossed before t); DC limit Vg RL/(Rg+RL), Vg/(Rg+RL).
All bookkeeping is in exact fractions with times in units of T.
"""
import math
import random
from fractions import Fraction as Fr

Z0S = [50, 60, 75, 100]
RG_RATIOS = [Fr(1, 3), Fr(1, 2), Fr(2), Fr(3)]                      # plus 0 (ideal) and 1 (matched)
RL_RATIOS = [Fr(1, 4), Fr(1, 3), Fr(1, 2), Fr(2), Fr(3), Fr(4)]     # plus 0 (short), 1 (matched), open
ERS = [1, Fr(9, 4), 4, 9]                                             # non-magnetic fillings: v = c/1, c/1.5, c/2, c/3
VGS = [6, 8, 10, 12, 15, 16, 18, 20, 24, 30]
XS = [Fr(1, 4), Fr(1, 3), Fr(1, 2), Fr(2, 3), Fr(3, 4)]
# (Z0, Rg, RL) triples used on the site's Lecture 28/29 pages, practice set and worked problems (RL None = open)
BLACKLIST = {(60, 20, 180), (100, 0, 300), (50, 30, 0), (50, 25, 200), (50, 50, 150), (60, 20, 300), (50, 450, None),
             (60, 180, 20), (50, 0, 100), (50, 50, 100), (50, 75, 200), (75, 75, Fr(75, 2)), (75, 25, Fr(75, 2))}


def num(x, sig=4):
    x = float(x)
    if abs(x) < 1e-12:
        return "0"
    s = f"{x:.{sig}g}"
    if "e" in s:
        m, e = s.split("e")
        return f"{m}\\times10^{{{int(e)}}}"
    return s


def tnum(x):
    """num() for HTML table cells: exponent forms go inside math."""
    s = num(x)
    return f"${s}$" if "times" in s else s


def ftex(q):
    """Exact fraction as LaTeX."""
    q = Fr(q)
    if q.denominator == 1:
        return str(q.numerator)
    sg = "-" if q < 0 else ""
    return f"{sg}\\tfrac{{{abs(q.numerator)}}}{{{q.denominator}}}"


def coeffs(Z0, Rg, RL):
    tg = Fr(Z0) / (Rg + Z0)
    Gg = Fr(Rg - Z0) / (Rg + Z0)
    GL = Fr(1) if RL is None else Fr(RL - Z0) / (RL + Z0)
    return tg, Gg, GL


def crossings(x, tg, Gg, GL, tmax):
    """(time/T, weight per volt of Vg, direction, segment index) of every segment passing x before tmax."""
    out = []
    n = 0
    while x + 2 * n < tmax or 2 - x + 2 * n < tmax:
        wf = tg * (GL * Gg) ** n
        wb = tg * GL * (GL * Gg) ** n
        if x + 2 * n < tmax:
            out.append((x + 2 * n, wf, 1, 2 * n))
        if 2 - x + 2 * n < tmax:
            out.append((2 - x + 2 * n, wb, -1, 2 * n + 1))
        n += 1
    out.sort(key=lambda r: (r[0], r[3]))
    return out


def level(x, t, tg, Gg, GL):
    cr = crossings(x, tg, Gg, GL, t)
    return sum(w for _, w, _, _ in cr), sum(d * w for _, w, d, _ in cr)


def event_times(x, tmax):
    ev = set()
    n = 0
    while x + 2 * n < tmax:
        ev.add(x + 2 * n)
        ev.add(2 - x + 2 * n)
        n += 1
    return sorted(e for e in ev if e < tmax)


def candidates(x, tmax=Fr(14)):
    """Times (units of T) on a T/2 grid at least T/8 from every crossing, with their interval index."""
    ev = event_times(x, tmax + 2)
    out = []
    for k in range(1, int(2 * tmax)):
        t = Fr(k, 2)
        if all(abs(t - e) >= Fr(1, 8) for e in ev):
            out.append((t, sum(1 for e in ev if e < t)))
    return out


def ok_size(q):
    return q == 0 or abs(q) >= Fr(1, 50)


def draw_circuit():
    while True:
        Z0 = random.choice(Z0S)
        gk = random.choices(["ideal", "matched", "res"], weights=[15, 20, 65])[0]
        lk = random.choices(["open", "short", "matched", "res"], weights=[15, 10, 10, 65])[0]
        if gk == "ideal" and lk in ("open", "short"):
            continue                                    # lossless resonator or infinite DC current
        Rg = Fr(0) if gk == "ideal" else (Fr(Z0) if gk == "matched" else Z0 * random.choice(RG_RATIOS))
        RL = None if lk == "open" else (Fr(0) if lk == "short" else (Fr(Z0) if lk == "matched" else Z0 * random.choice(RL_RATIOS)))
        if Rg.denominator != 1 or (RL is not None and RL.denominator != 1):
            continue
        if (Z0, Rg, RL) in BLACKLIST:
            continue
        return Z0, Rg, RL, gk, lk


def r3(q):
    return float(f"{float(q):.3g}")


def tol_ok(terms, floor, rtol=0.01):
    """Tolerance for a running sum of bounce terms (display units). atol covers 3-s.f. rounding of each term;
    the variant is kept only if a 3-s.f. chain is accepted and dropping any one term is rejected."""
    a = sum(terms, Fr(0))
    atol = max(floor, 0.003 * float(sum(abs(t) for t in terms)))
    tol = atol + rtol * abs(float(a))
    chain = sum(r3(t) for t in terms)
    ok = abs(chain - float(a)) <= 0.8 * tol and all(abs(float(t)) > 1.5 * tol for t in terms if t != 0)
    return ok, atol


def pick_times(x, groups, tg, Gg, GL, scale, floor, use_current=False, dc=None):
    """Pick one distinct time per interval group (returned in increasing order); scale converts weights
    (per volt of Vg) to display units. The value after the next crossing and the DC value dc (per volt) must be
    rejected by the tolerance unless equal to the answer (no credit for an extra crossing or the DC shortcut)."""
    cand = candidates(x)
    out, atols = [], []
    for g in groups:
        opts = [t for t, i in cand if i in g and t not in out]
        random.shuffle(opts)
        found = None
        for t in opts:
            v, zi = level(x, t, tg, Gg, GL)
            a = (zi if use_current else v) * scale
            terms = [(d * w if use_current else w) * scale for _, w, d, _ in crossings(x, tg, Gg, GL, t)]
            ok, atol = tol_ok(terms, floor)
            tol = atol + 0.01 * abs(float(a))
            e_next = min(e for e in event_times(x, t + 4) if e > t)
            vn, zn = level(x, e_next + Fr(1, 100), tg, Gg, GL)
            others = [(zn if use_current else vn) * scale] + ([] if dc is None else [dc * scale])
            far = all(b == a or abs(float(b - a)) > 1.5 * tol for b in others)
            if ok_size(zi if use_current else v) and ok and far:
                found = (t, atol)
                break
        if found is None:
            return None
        out.append(found[0])
        atols.append(found[1])
    order = sorted(range(len(out)), key=lambda i: out[i])
    return [out[i] for i in order], [atols[i] for i in order]


def generate(data):
    Vg = random.choice(VGS)
    er = random.choice(ERS)
    T = random.choice([1, 2, 3, 4])
    if random.random() < 0.5:
        tu, lu, c, tu_t = r"\mu\mathrm{s}", r"\mathrm{m}", 300, "μs"
    else:
        tu, lu, c, tu_t = r"\mathrm{ns}", r"\mathrm{cm}", 30, "ns"
    v = Fr(c) / Fr(math.isqrt(int(er * 4)), 2)                 # c/sqrt(er): sqrt of 1, 9/4, 4, 9
    ell = v * T
    xs = [x for x in XS if (x * ell * 100).denominator == 1]   # probe positions with short decimals
    while True:
        Z0, Rg, RL, gk, lk = draw_circuit()
        tg, Gg, GL = coeffs(Z0, Rg, RL)
        dcV = Fr(1) if RL is None else Fr(RL) / (Rg + RL)              # DC values per volt (V, Z0*I)
        dcI = Fr(0) if RL is None else Fr(Z0) / (Rg + RL)
        if lk == "matched":                                            # V is 0 then V+: ask once before arrival
            xs_p = [x for x in xs if x > Fr(1, 2)] + [Fr(1)]
            groups = [{0}, set(range(1, 60)), set(range(1, 60))]
        else:
            xs_p = xs + [Fr(1)] * (lk != "short") + [Fr(0)] * (gk != "ideal")
        xp = random.choice(xs_p)
        ends = xp in (0, 1)
        if lk != "matched":
            groups = [{1}, {2}, {3}] if ends else [{1, 2}, {3, 4}, {5, 6}]
        res = pick_times(xp, groups, tg, Gg, GL, Fr(Vg), 2e-4 * Vg, dc=dcV)
        if res is None:
            continue
        tb, atol_b = res
        if len({level(xp, t, tg, Gg, GL)[0] for t in tb}) == 1:
            continue                                    # (b) must depend on the timing
        xs_q = [x for x in xs + [Fr(0), Fr(1)] if x != xp and not (x == 1 and lk == "open")]
        xq = random.choice(xs_q)
        grp = [{1, 2}] if xq == 1 else ([{2, 3}] if xq == 0 else [{2, 3, 4}])
        res = pick_times(xq, grp, tg, Gg, GL, Fr(Vg * 1000, Z0), 2e-4 * Vg * 1000 / Z0, use_current=True, dc=dcI)
        if res is None:
            continue
        tq, atol_q = res[0][0], res[1][0]
        break

    def tdisp(tT):          # time in display units from units of T
        return num(tT * T)

    def zdisp(x):
        return num(x * ell)

    def zdesc(x, nm):
        if x == 0:
            return f"the generator end, ${nm} = 0$"
        if x == 1:
            return f"the load end, ${nm} = \\ell = {num(ell)}\\ {lu}$"
        return f"${nm} = {num(x * ell)}\\ {lu}$"

    # ---------------------------------------------------------------- answers
    Vplus = tg * Vg
    Vb = [level(xp, t, tg, Gg, GL)[0] * Vg for t in tb]
    Iq = level(xq, tq, tg, Gg, GL)[1] * Vg / Z0 * 1000          # mA
    if RL is None:
        Vdc, Idc = Fr(Vg), Fr(0)
    else:
        Vdc, Idc = Vg * RL / (Rg + RL), Fr(Vg) / (Rg + RL) * 1000
    # geometric-series check of the DC value (from the coefficients)
    Vdc_series = Vg * tg * (1 + GL) / (1 - GL * Gg)
    assert Vdc_series == Vdc

    # ---------------------------------------------------------------- text pieces
    Rg_txt = (r"no internal resistance (an ideal source, $R_g = 0$)" if gk == "ideal" else f"internal resistance $R_g = {num(Rg)}\\ \\Omega$")
    if lk == "open":
        load_txt = r"left open ($R_L \to \infty$)"
    elif lk == "short":
        load_txt = r"short-circuited ($R_L = 0$)"
    else:
        load_txt = f"terminated by a resistor $R_L = {num(RL)}\\ \\Omega$"
    er_txt = "air" if er == 1 else f"a non-magnetic dielectric with $\\epsilon_r = {num(er)}$"
    RL_sym = r"\infty" if RL is None else num(RL)

    # weights table
    tmax = max(max(tb), tq)
    nseg = max(4, math.ceil(tmax))
    seg_rows = []
    for k in range(nseg):
        n = k // 2
        fwd = k % 2 == 0
        w = tg * (GL * Gg) ** n * (1 if fwd else GL)
        def pw(sym, e):
            return "" if e == 0 else (sym if e == 1 else sym + "^{" + str(e) + "}")
        mult = r"\tau_g" + pw(r"\Gamma_L", n + (0 if fwd else 1)) + pw(r"\Gamma_g", n)
        if w == 0:
            continue
        seg_rows.append({
            "k": k + 1, "dir": "toward the load" if fwd else "toward the generator",
            "t0": tdisp(k), "t1": tdisp(k + 1), "mult": mult,
            "w": tnum(w * Vg), "i": tnum((1 if fwd else -1) * w * Vg / Z0 * 1000),
        })

    def cross_rows(x, tend, current):
        rows = []
        run = Fr(0)
        for t, w, d, k in crossings(x, tg, Gg, GL, tend):
            if w == 0:
                continue
            dv = (d * w if current else w) * Vg * (Fr(1000, Z0) if current else 1)
            run += dv
            rows.append({"t": tdisp(t), "k": k + 1, "dir": "forward" if d > 0 else "backward",
                         "dv": tnum(dv), "run": tnum(run)})
        return rows

    rows_p = cross_rows(xp, max(tb), False)
    rows_q = cross_rows(xq, tq, True)

    # ---------------------------------------------------------------- (e) what-if MC
    scen = []
    if gk != "matched":
        scen.append("matchgen")
    if lk != "open" and gk != "ideal":
        scen.append("open")
    if lk != "short" and gk != "ideal":
        scen.append("short")
    sc = random.choice(scen)
    Ts = num(T)
    if sc == "matchgen":
        e_stem = (r"Suppose $R_g$ were replaced by a resistor equal to $Z_0$ (same source voltage, line and load). "
                  r"Counting from $t = 0$, when does the voltage stop changing at <em>every</em> point of the line? "
                  r"($T = \ell/v$ is the one-way delay.)")
        settle = "T" if GL == 0 else "2T"
        e_choices = [
            {"id": "T", "claim": "settle", "tT": 1, "text": r"at $t = T$, when the first wave reaches the load",
             "fb": (r"Right: with $R_L = Z_0$ too, neither end reflects; nothing changes after the first wave reaches the load." if settle == "T" else
                    r"The first wave reaches the load at $T$, but this load is not matched ($\Gamma_L \ne 0$) and sends an echo back; the line keeps changing until the echo is absorbed at $z = 0$.")},
            {"id": "2T", "claim": "settle", "tT": 2, "text": r"at $t = 2T$, when the echo from the load reaches $z = 0$",
             "fb": (r"Right: now $\Gamma_g = 0$, so the load's echo is absorbed when it reaches $z = 0$ and nothing new is launched." if settle == "2T" else
                    r"With $R_L = Z_0$ there is no echo at all: the last change is the first wave's arrival at the load.")},
            {"id": "4T", "claim": "settle", "tT": 4, "text": r"at $t = 4T$, after two complete round trips",
             "fb": r"A second round trip needs a wave re-launched at $z = 0$; with $\Gamma_g = 0$ the returning echo is absorbed and launches nothing."},
            {"id": "never", "claim": "never", "tT": 0, "text": r"never exactly: the staircase only approaches its final value geometrically",
             "fb": r"The steps shrink as $(\Gamma_L\Gamma_g)^n$; with $\Gamma_g = 0$ every term after the first echo is exactly zero, so the line settles in finite time."},
        ]
        for ch in e_choices:
            ch["correct"] = "true" if ch["id"] == settle else "false"
    elif sc == "open":
        e_stem = (r"Suppose instead the end $z = \ell$ were left open (same source and line). Which statement about the new circuit is correct?")
        e_choices = [
            {"id": "ok", "correct": "true",
             "text": r"At $t = T$ the voltage at $z = \ell$ jumps to twice the arriving wave's voltage; as $t\to\infty$ the current dies out and the voltage approaches $V_g$ everywhere.",
             "fb": r"Right: $\Gamma_L = +1$ doubles the voltage and cancels the current at the open end; at DC no current flows, so $R_g$ drops nothing."},
            {"id": "zeroV", "correct": "false",
             "text": r"The voltage at $z = \ell$ stays zero, because no current can flow into an open circuit.",
             "fb": r"It is the current, not the voltage, that is zero at an open end. The echo equals the arriving wave ($\Gamma_L = +1$), so the voltages add."},
            {"id": "noecho", "correct": "false",
             "text": r"No echo comes back from an open end, so the line keeps the first wave's voltage $\tau_gV_g$ everywhere for all later times.",
             "fb": r"That describes a matched load. An open end reflects totally: $\Gamma_L = (R_L-Z_0)/(R_L+Z_0)\to+1$."},
            {"id": "dc", "correct": "false",
             "text": r"At $t = T$ the voltage at $z = \ell$ jumps to twice the arriving wave's voltage; as $t\to\infty$ the voltage approaches $\tau_gV_g$ everywhere.",
             "fb": r"The first half is right, but at DC the line is just two wires: with no current, $R_g$ drops nothing and the voltage tends to $V_g$, not $\tau_gV_g$."},
            {"id": "Idouble", "correct": "false",
             "text": r"At $t = T$ the current at $z = \ell$ jumps to twice the arriving wave's current.",
             "fb": r"That happens at a short. At an open end $I(\ell,t) = 0$ at every instant: the echo's current $-\Gamma_L$ times the arriving one cancels it."},
        ]
    else:
        e_stem = (r"Suppose instead the end $z = \ell$ were short-circuited (same source and line). Which statement about the new circuit is correct?")
        e_choices = [
            {"id": "ok", "correct": "true",
             "text": r"The voltage at $z = \ell$ is zero at every instant; at $t = T$ the current there jumps to twice the arriving wave's current, and as $t\to\infty$ the current approaches $V_g/R_g$.",
             "fb": r"Right: $\Gamma_L = -1$ cancels the voltage and doubles the current at the short; at DC the source sees only $R_g$."},
            {"id": "Vdouble", "correct": "false",
             "text": r"At $t = T$ the voltage at $z = \ell$ jumps to twice the arriving wave's voltage.",
             "fb": r"That happens at an open end. A short forces $V(\ell,t) = 0$: the echo is the arriving wave inverted."},
            {"id": "sign", "correct": "false",
             "text": r"The short reflects the arriving voltage with $\Gamma_L = +1$, so the echo has the same polarity as the arriving wave.",
             "fb": r"$\Gamma_L = (0-Z_0)/(0+Z_0) = -1$: the voltage echo is inverted (its current has the same sign as the arriving current)."},
            {"id": "dc", "correct": "false",
             "text": r"The voltage at $z = \ell$ is zero at every instant and the current there doubles at $t = T$; as $t\to\infty$ the current approaches $V_g/(R_g+Z_0)$, since the line still presents $Z_0$.",
             "fb": r"The first half is right, but at DC the line is just two wires: the current tends to $V_g/R_g$. $Z_0$ is not a resistance."},
            {"id": "noI", "correct": "false",
             "text": r"No current flows at $z = \ell$, because there is no voltage across the short.",
             "fb": r"Zero voltage across a short does not mean zero current: the arriving and reflected currents add, $(1-\Gamma_L) = 2$."},
        ]
    for ch in e_choices:
        ch["feedback"] = ch.pop("fb")
    random.shuffle(e_choices)
    if sc == "matchgen":
        e_why = (r"With $R_g = Z_0$: $\Gamma_g = 0$, so the generator absorbs every returning wave. "
                 + (r"Here also $\Gamma_L = 0$, so there is no echo at all and the line is final once the first wave reaches the load, at $t = T = " + Ts + r"\ " + tu + "$."
                    if GL == 0 else
                    r"The load's echo ($\Gamma_L = " + ftex(GL) + r"$) is absorbed when it reaches $z = 0$ at $t = 2T = " + num(2 * T) + r"\ " + tu + r"$; after that nothing travels, so every point keeps its value."))
    elif sc == "open":
        e_why = (r"An open end has $\Gamma_L = +1$: the echo equals the arriving wave, so $V(\ell)$ doubles and $I(\ell) = (1-\Gamma_L)\times$(arriving current) $= 0$. "
                 r"At DC no current flows anywhere, there is no drop across $R_g$, and $V \to V_g$ (the divider $V_gR_L/(R_g+R_L)$ with $R_L\to\infty$).")
    else:
        e_why = (r"A short has $\Gamma_L = -1$: $V(\ell) = (1+\Gamma_L)\times$(arriving) $= 0$ and $I(\ell) = (1-\Gamma_L)\times$(arriving current), twice it. "
                 r"At DC the line is two wires, so the current is $V_g/R_g$ (the divider with $R_L = 0$).")

    # ---------------------------------------------------------------- params
    p = data["params"]
    p.update({
        "Z0": Z0, "Rg": float(Rg), "RL": (-1 if RL is None else float(RL)), "gk": gk, "lk": lk, "Vg": Vg,
        "er": float(er), "c": c, "T": T, "ell": float(ell), "v": float(v), "xp": float(xp), "xq": float(xq),
        "tb": [float(t * T) for t in tb], "tq": float(tq * T), "scen": sc,
        "Rg_txt": Rg_txt, "load_txt": load_txt, "er_txt": er_txt, "tu": tu, "lu": lu, "tu_t": tu_t,
        "ell_s": num(ell), "zp_s": zdisp(xp), "zq_s": zdisp(xq), "zp_desc": zdesc(xp, "z_p"), "zq_desc": zdesc(xq, "z_q"),
        "t1_s": tdisp(tb[0]), "t2_s": tdisp(tb[1]), "t3_s": tdisp(tb[2]), "tq_s": tdisp(tq),
        "atolV": f"{2e-4 * Vg:.2g}", "atolI": f"{2e-4 * Vg / Z0 * 1000:.2g}",
        "atolV1": f"{atol_b[0]:.2g}", "atolV2": f"{atol_b[1]:.2g}", "atolV3": f"{atol_b[2]:.2g}", "atolIq": f"{atol_q:.2g}",
        # answer panel
        "v_s": num(v), "v_expr": ("c" if er == 1 else "c/" + num(math.sqrt(er))), "T_s": Ts, "twoT_s": num(2 * T),
        "tg_tex": ftex(tg), "Gg_tex": ftex(Gg), "GL_tex": ftex(GL), "GLGg_tex": ftex(GL * Gg),
        "Rg_s": num(Rg), "RL_sym": RL_sym, "Vplus_s": num(Vplus), "Iplus_s": num(Vplus / Z0 * 1000),
        "Vb1_s": num(Vb[0]), "Vb2_s": num(Vb[1]), "Vb3_s": num(Vb[2]), "Iq_s": num(Iq),
        "Vdc_s": num(Vdc), "Idc_s": num(Idc),
        "seg_rows": seg_rows, "zero_note": ("" if (GL != 0 and Gg != 0) else
                                            (r"Since $\Gamma_L = 0$, nothing comes back from the load. " if GL == 0 else
                                             r"Since $\Gamma_g = 0$, the generator absorbs the first echo and launches nothing more. ")), "rows_p": rows_p, "rows_q": rows_q,
        "zp_over_v": tdisp(xp), "zq_over_v": tdisp(xq),
        "tp_back": tdisp(2 - xp), "tq_back": tdisp(2 - xq),
        "dc_text": (r"With the end open no current flows at DC, so $R_g$ drops nothing: $V(z,\infty) = V_g = " + num(Vg) + r"$ V and $I(z,\infty) = 0$."
                    if RL is None else
                    r"$$V(z,\infty) = V_g\frac{R_L}{R_g+R_L} = " + num(Vg) + r"\cdot\frac{" + num(RL) + "}{" + num(Rg + RL) + "} = " + num(Vdc)
                    + r"\ \mathrm{V},\qquad I(z,\infty) = \frac{V_g}{R_g+R_L} = " + num(Idc) + r"\ \mathrm{mA}.$$"),
        "series_text": (r"Check with the bounce series: $V_g\tau_g\dfrac{1+\Gamma_L}{1-\Gamma_L\Gamma_g} = " + num(Vplus) + r"\cdot\dfrac{" + num(1 + GL)
                        + "}{" + num(1 - GL * Gg) + "} = " + num(Vdc) + r"$ V."),
        "end_note": ("" if not ends else
                     r"At an end, the arriving segment and the one it launches cross that point together; both are listed. "),
        "end_note_q": ("" if xq not in (0, 1) else
                       r" At an end, the arriving segment and the one it launches cross together; both are listed."),
        "e_stem": e_stem, "e_choices": e_choices, "e_why": e_why,
    })
    ca = data["correct_answers"]
    ca["Vplus"] = float(Vplus)
    ca["GL"] = float(GL) + 0.0
    ca["Gg"] = float(Gg) + 0.0
    ca["V1"] = float(Vb[0]) + 0.0
    ca["V2"] = float(Vb[1]) + 0.0
    ca["V3"] = float(Vb[2]) + 0.0
    ca["Iq"] = float(Iq) + 0.0
    ca["Vdc"] = float(Vdc) + 0.0
    ca["Idc"] = float(Idc) + 0.0
