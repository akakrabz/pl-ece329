"""Lecture 29 concept question: a short pulse on a line with a mismatched load; snapshots and probe records.

Conventions (Lecture 28 page): z = 0 generator, z = ell load, +z toward the load, T = ell/v.
    tau_g = Z0/(Rg+Z0), Gamma_g = (Rg-Z0)/(Rg+Z0), Gamma_L = (RL-Z0)/(RL+Z0) (open +1, short -1)
Horizontal cut at t (Lecture 29):  forward segment n:  w f_i(t - z/v - 2nT),  backward n:  w f_i(t + z/v - 2(n+1)T)
with weights tau_g (GL Gg)^n and tau_g GL (GL Gg)^n; V adds the copies, Z0 I subtracts the backward ones.
The pulse lasts w <= T/2, so copies are separate except near the ends.
    echo (first backward copy) alone on the line for T + w < t0 < 2T, on  ell - v(t0-T) < z < ell - v(t0-T-w)
    re-launched copy alone for 2T + w < t0 < 3T, on  v(t0-2T-w) < z < v(t0-2T)
    overlap at the load for T < t1 < T + w; the echo's peak passes z_p at 2T - z_p/v + t_pk.
Exact fractions throughout; times in us, lengths in m.
"""
import random
from fractions import Fraction as Fr

Z0S = [50, 60, 75, 100]
VS = [100, 150, 200, 300]                 # m/us
AS = [6, 8, 10, 12, 16, 20, 24]
XS = [Fr(1, 4), Fr(1, 3), Fr(1, 2), Fr(2, 3), Fr(3, 4)]
BLACKLIST = {(50, 25, 150, 200, 3), (50, 50, 150, 200, 3)}   # the site's ramp-pulse problem and practice 28.10


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
    atol = max(floor, 0.005 * float(sum(abs(t) for t in terms)))   # 3-s.f. factors, e.g. 1/6 -> 0.167
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


def shape(t, w, tp):
    """Normalized source pulse (peak 1): triangle peaking at tp, or rectangle if tp is None."""
    if t <= 0 or t >= w:
        return Fr(0)
    if tp is None:
        return Fr(1)
    return t / tp if t <= tp else (w - t) / (w - tp)


def generate(data):
    while True:
        Z0 = random.choice(Z0S)
        v = random.choice(VS)
        w = random.choice([1, 2])
        T = random.choice([t for t in (3, 4, 5, 6) if t >= 2 * w])
        ell = v * T
        rect = random.random() < 0.3
        tp = None if rect else Fr(random.choice([1, 3]), 4) * w
        A = random.choice(AS)
        gk = random.choices(["matched", "ideal", "res"], weights=[35, 10, 55])[0]
        Rg = Fr(Z0) if gk == "matched" else (Fr(0) if gk == "ideal" else Z0 * random.choice([Fr(1, 3), Fr(1, 2), Fr(2), Fr(3)]))
        lk = random.choices(["open", "short", "res"], weights=[12, 8, 80])[0]
        RL = None if lk == "open" else (Fr(0) if lk == "short" else Z0 * random.choice([Fr(1, 4), Fr(1, 3), Fr(1, 2), Fr(2), Fr(3), Fr(4)]))
        if Rg.denominator != 1 or (RL is not None and RL.denominator != 1):
            continue
        if gk == "ideal" and lk != "res":
            continue                                    # both ends totally reflecting: too few distinct choices
        if (Z0, Rg, RL, v, T) in BLACKLIST:
            continue
        tg = Fr(Z0) / (Rg + Z0)
        Gg = Fr(Rg - Z0) / (Rg + Z0)
        GL = Fr(1) if RL is None else Fr(RL - Z0) / (RL + Z0)
        # ---------- (a) snapshot time
        regimes = ["echo"] + (["second"] if Gg != 0 else [])
        reg = random.choice(regimes)
        if reg == "echo":
            lo, hi, bad = T + w, 2 * T, Fr(3 * T + w, 2)
        else:
            lo, hi, bad = 2 * T + w, 3 * T, Fr(5 * T + w, 2)
        t0s = [Fr(k, 2) for k in range(2 * lo + 1, 2 * hi) if Fr(k, 2) != bad]
        if not t0s:
            continue
        t0 = random.choice(t0s)
        # ---------- (b), (d) overlap point near the load
        pairs = [(u, d) for u in [Fr(k, 4) for k in range(1, 4 * w)] for d in [Fr(1, 4), Fr(1, 2)] if d < min(u, w - u)]
        u, dl = random.choice(pairs)
        t1 = T + u
        z1 = ell - v * dl
        af, ae = u + dl, u - dl                        # source times seen by the forward copy and the echo
        Vb = tg * A * (shape(af, w, tp) + GL * shape(ae, w, tp))
        Id = tg * A * (shape(af, w, tp) - GL * shape(ae, w, tp)) / Z0 * 1000      # mA
        small = Fr(1, 50) * tg * A
        if (Vb != 0 and abs(Vb) < small) or (Id != 0 and abs(Id) < small / Z0 * 1000):
            continue
        tv = [tg * A * shape(af, w, tp), tg * A * GL * shape(ae, w, tp)]
        okV, atolV = tol_ok(tv, 2e-4 * A)
        okI, atolI = tol_ok([tv[0] * Fr(1000, Z0), -tv[1] * Fr(1000, Z0)], 2e-4 * A * 1000 / Z0)
        if not (okV and okI):
            continue
        # ---------- (c) probe
        xs = [x for x in XS if (x * ell * 100).denominator == 1]
        xp = random.choice(xs)
        zp = xp * ell
        tc = 2 * T - zp / v + (w if rect else tp)
        break

    # ---------------------------------------------------------------- (a) snapshot choices
    def desc(val, za, zb, zpk):
        if val is None:
            return None
        if rect:
            return (f"a single rectangular pulse: $V = {num(val)}\\ \\mathrm{{V}}$ for ${num(za, 6)}\\ \\mathrm{{m}} \\lt z \\lt {num(zb, 6)}\\ \\mathrm{{m}}$, "
                    f"zero elsewhere")
        return (f"a single triangular pulse with its peak $V = {num(val)}\\ \\mathrm{{V}}$ at $z = {num(zpk, 6)}\\ \\mathrm{{m}}$, "
                f"nonzero only for ${num(za, 6)}\\ \\mathrm{{m}} \\lt z \\lt {num(zb, 6)}\\ \\mathrm{{m}}$")

    L = v * w
    if reg == "echo":
        wt = tg * GL
        za = ell - v * (t0 - T)
        zpk = None if rect else za + v * tp
        za_s = v * (t0 - T - w)
        cands = [
            ("ok", wt * A, za, zpk, f"Right: the echo $\\tau_g\\Gamma_Lf_i$ left the load at $t = T$ and has moved $v(t_0-T)$ back toward the generator; it lies in source order (first-sent part nearer $z = 0$)."),
            ("pol", -wt * A, za, zpk, f"Check the sign: the echo's weight is $\\tau_g\\Gamma_L$ with $\\Gamma_L = {ftex(GL)}$, so its peak is {'negative' if wt < 0 else 'positive'}."),
            ("mirror", wt * A, za, None if rect else 2 * za + L - zpk, "This is the echo reversed. An echo lies in the same order as $f_i(t)$: the part sent first leads, on the side nearer the generator."),
            ("slip", wt * A, za_s, None if rect else za_s + (zpk - za), "This puts the pulse $v(t_0-T)$ from the generator. The echo starts at the load at $t = T$ and travels back from $z = \\ell$."),
            ("ampL", (1 + GL) * tg * A, za, zpk, "$(1+\\Gamma_L)\\tau_g$ is the load voltage while the pulse is at the load; the echo alone carries $\\tau_g\\Gamma_L$."),
            ("ampA", GL * A, za, zpk, "$\\Gamma_L$ multiplies the launched pulse $\\tau_gf_i$, not the open-circuit source pulse $f_i$."),
            ("zero", None, None, None, ("A resistor $R_L \\ne Z_0$ absorbs only part of the pulse; the fraction $\\Gamma_L$ of its voltage comes back." if lk == "res" else
                                         "An open or shorted end absorbs nothing: the whole pulse comes back, with $\\Gamma_L = \\pm1$.")),
            ("ampT", tg * A, za, zpk, "This is the launched pulse's height; the echo is $\\Gamma_L$ times it."),
        ]
        zero_txt = ("$V = 0$ everywhere on the line: the load resistor has absorbed the pulse" if lk == "res" else
                    "$V = 0$ everywhere on the line: the pulse has left through the end $z = \\ell$")
    else:
        wt = tg * GL * Gg
        zb = v * (t0 - 2 * T)
        za = zb - L
        zpk = None if rect else zb - v * tp
        za_s = ell - v * (t0 - 2 * T)
        cands = [
            ("ok", wt * A, za, zpk, f"Right: the echo reached $z = 0$ at $2T$ and the generator re-launched $\\Gamma_g$ times it toward the load; a pulse moving toward the load lies reversed."),
            ("pol", -wt * A, za, zpk, f"Check the sign: the weight is $\\tau_g\\Gamma_L\\Gamma_g$ with $\\Gamma_L = {ftex(GL)}$ and $\\Gamma_g = {ftex(Gg)}$, so the peak is {'negative' if wt < 0 else 'positive'}."),
            ("mirror", wt * A, za, None if rect else 2 * za + L - zpk, "A pulse moving toward the load lies reversed compared with $f_i(t)$: the part sent first is farthest from the generator."),
            ("slip", wt * A, za_s, None if rect else za_s + (zpk - za), "This measures from the load. The re-launched pulse starts at $z = 0$ at $t = 2T$ and travels toward the load."),
            ("noGg", tg * GL * A, za, zpk, f"That is the first echo's height. It has since been reflected by the generator, which multiplies it by $\\Gamma_g = {ftex(Gg)}$."),
            ("zero", None, None, None, f"With $R_g \\ne Z_0$, $\\Gamma_g = {ftex(Gg)} \\ne 0$: the generator re-launches $\\Gamma_g$ times the echo toward the load."),
            ("noGL", tg * Gg * A, za, zpk, "The pulse has been reflected at both ends, so both $\\Gamma_L$ and $\\Gamma_g$ multiply it."),
            ("notg", GL * Gg * A, za, zpk, "The launched pulse is $\\tau_gf_i$, not $f_i$: $\\tau_g$ multiplies too."),
        ]
        zero_txt = "$V = 0$ everywhere on the line: the generator has absorbed the returning echo"
    chosen, seen = [], set()
    for cid, val, a_, pk, fb in cands:
        if rect and cid in ("mirror", "ampL"):          # ampL would display (b)'s answer for a rectangle
            continue
        key = ("zero",) if val is None else (val, a_, pk)
        if key in seen or (val is not None and val == 0):
            continue
        seen.add(key)
        chosen.append({"id": cid, "correct": "true" if cid == "ok" else "false", "feedback": fb,
                       "text": zero_txt if val is None else desc(val, a_, a_ + L, pk),
                       "val": None if val is None else float(val), "za": None if a_ is None else float(a_),
                       "zb": None if a_ is None else float(a_ + L), "zpk": None if pk is None else float(pk)})
        if len(chosen) == 5:
            break
    snap = chosen
    random.shuffle(snap)

    # ---------------------------------------------------------------- (e) statement pool
    if lk == "res":
        ohm = (f"At $z = \\ell$, $V(\\ell,t) = R_LI(\\ell,t)$ at every instant, also while the pulse is being reflected.",
               "Ohm's law holds at the resistor at all times; the incident and reflected waves together satisfy it.")
    elif lk == "open":
        ohm = ("At the open end, $I(\\ell,t) = 0$ at every instant, also while the pulse is being reflected.",
               "The echo's current, $-\\Gamma_L = -1$ times the arriving current, cancels it at $z = \\ell$.")
    else:
        ohm = ("At the short, $V(\\ell,t) = 0$ at every instant, also while the pulse is being reflected.",
               "The echo, $\\Gamma_L = -1$ times the arriving voltage, cancels it at $z = \\ell$.")
    facts = [
        ("probe_order", True, "A probe at a fixed $z$ records the echo with the same time order as $f_i(t)$: the part the source sent first passes first.",
         "At fixed $z$ each copy is $w\\,f_i(t-t_k)$, a delayed copy, never reversed in time."),
        ("probe_rev", False, "A probe at a fixed $z$ records the echo time-reversed compared with $f_i(t)$: the part the source sent last passes first.",
         "Only snapshots in $z$ can look reversed; in time every copy is $w\\,f_i(t-t_k)$."),
        ("snap_fwd", True, "On a snapshot $V(z,t_0)$, a pulse travelling toward the load lies reversed compared with $f_i(t)$: the part sent first is farthest from the generator.",
         "The leading edge points the way it travels, so the first-sent part is on the right."),
        ("snap_echo", False, "On a snapshot $V(z,t_0)$, the echo travelling toward the generator lies reversed compared with $f_i(t)$: the part sent first is farthest from the generator.",
         "The echo's first-sent part leads toward $z = 0$, on the left: same order as $f_i(t)$."),
        ("length", True, "The echo occupies the same length of line as the incident pulse, $v$ times the pulse duration.",
         f"Both copies travel at $v$ and last {num(w)} μs: {num(L)} m each."),
    ] + ([("shrink", False, "The echo's length on the line is $\\lvert\\Gamma_L\\rvert$ times the incident pulse's length.",
            "$\\Gamma_L$ scales the voltage, not the duration or the length.")] if lk == "res" else []) + [
        ("overlapZ0", False, "Where the incident pulse and its echo overlap, $V/I = Z_0$ still holds, because each wave on its own satisfies it.",
         "$V = f+g$ but $Z_0I = f-g$: the ratio is $Z_0(f+g)/(f-g)$, not $Z_0$, wherever $g\\ne0$."),
        ("load_ohm", True, ohm[0], ohm[1]),
        ("echoI", True, "The echo's current has the opposite sign to its voltage.",
         "A backward wave carries $I = -V/Z_0$."),
        ("echoI_same", False, "When $\\Gamma_L \\gt 0$, the echo's current has the same sign as its voltage.",
         "Any backward wave has $I = -V/Z_0$, whatever the sign of $\\Gamma_L$."),
        ("absorbed", Gg == 0, "In this circuit the echo is completely absorbed when it returns to $z = 0$.",
         ("$R_g = Z_0$, so $\\Gamma_g = 0$: the generator is matched." if Gg == 0 else
          "$R_g \\ne Z_0$, so $\\Gamma_g \\ne 0$ and part of the echo is re-launched toward the load.")),
        ("matched_load", True, "With a matched load ($R_L = Z_0$) there would be no echo at all, whatever the shape of the pulse.",
         "$\\Gamma_L = 0$ multiplies every part of the pulse: the line looks infinitely long."),
        ("amp_GL", False, "Doubling the height of the source pulse would double $\\Gamma_L$.",
         "$\\Gamma_L$ depends on $R_L$ and $Z_0$ only; the echo doubles, its ratio to the incident pulse does not."),
        ("closer", True, "A probe closer to the generator receives the echo later.",
         "The echo starts at the load and moves toward $z = 0$: it passes $z$ at $2T - z/v$ plus the delay within the pulse."),
    ]
    fact_list = [{"id": fid, "correct": "true" if t else "false", "text": txt, "feedback": ("True: " if t else "False: ") + why,
                  "verdict": "true" if t else "false", "why": why} for fid, t, txt, why in facts]
    random.shuffle(fact_list)

    # ---------------------------------------------------------------- statement text
    if rect:
        fi_txt = (f"a rectangular pulse: $f_i(t) = {A}\\ \\mathrm{{V}}$ for $0 \\lt t \\lt {num(w)}\\ \\mu\\mathrm{{s}}$ and $f_i(t) = 0$ otherwise")
        pk_word = "trailing edge"
        c_ask = "At what time has the echo (the pulse reflected by the load) completely passed the probe, i.e. when does its trailing edge reach it?"
    else:
        fi_txt = (f"a triangular pulse that rises linearly from $0$ at $t = 0$ to its peak ${A}\\ \\mathrm{{V}}$ at $t = {num(tp)}\\ \\mu\\mathrm{{s}}$, "
                  f"then falls linearly to $0$ at $t = {num(w)}\\ \\mu\\mathrm{{s}}$; $f_i(t) = 0$ at all other times")
        pk_word = "peak"
        c_ask = "At what time does the peak of the echo (the pulse reflected by the load) reach the probe?"
    Rg_txt = ("An ideal source ($R_g = 0$)" if gk == "ideal" else f"A source with internal resistance $R_g = {num(Rg)}\\ \\Omega$")
    load_txt = ("left open" if lk == "open" else ("short-circuited" if lk == "short" else f"terminated by $R_L = {num(RL)}\\ \\Omega$"))

    # answer-panel pieces
    sf, se = shape(af, w, tp), shape(ae, w, tp)
    p = data["params"]
    p.update({
        "Z0": Z0, "v": v, "ell": ell, "T": T, "w": w, "A": A, "rect": rect, "tp": (None if rect else float(tp)),
        "Rg": float(Rg), "RL": (-1 if RL is None else float(RL)), "gk": gk, "lk": lk, "reg": reg,
        "t0_s": num(t0), "t1_s": num(t1), "z1_s": num(z1), "zp_s": num(zp),
        "fi_txt": fi_txt, "Rg_txt": Rg_txt, "load_txt": load_txt, "c_ask": c_ask, "pk_word": pk_word,
        "atolV": f"{atolV:.2g}", "atolI": f"{atolI:.2g}",
        "snap": snap, "facts": fact_list,
        # panel
        "T_s": num(T), "L_s": num(L), "tg_tex": ftex(tg), "Gg_tex": ftex(Gg), "GL_tex": ftex(GL),
        "wecho_s": num(tg * GL * A), "Vl_s": num(tg * A),
        "gen_txt": (r"the matched generator ($\Gamma_g = 0$) absorbs it when it returns at $t = 2T$" if Gg == 0 else
                    r"the generator re-launches $\tau_g\Gamma_L\Gamma_g f_i$ (peak " + num(tg * GL * Gg * A) + r" V) at $t = 2T$"),
        "snap_why": (f"At $t_0 = {num(t0)}\\ \\mu\\mathrm{{s}}$ (between $T + {num(w)} = {num(T + w)}$ and $2T = {num(2 * T)}\\ \\mu\\mathrm{{s}}$) only the echo is on the line. "
                     f"Its term is $\\tau_g\\Gamma_Lf_i(t_0 + z/v - 2T)$, nonzero for $0 \\lt t_0 + z/v - 2T \\lt {num(w)}$, i.e. "
                     f"${num(za)} \\lt z \\lt {num(za + L)}$ m, with height $\\tau_g\\Gamma_L\\cdot{A} = {num(tg * GL * A)}$ V"
                     + ("" if rect else f" and its peak where $t_0 + z/v - 2T = {num(tp)}$, at $z = {num(zpk)}$ m") + "."
                     if reg == "echo" else
                     f"At $t_0 = {num(t0)}\\ \\mu\\mathrm{{s}}$ (between $2T + {num(w)} = {num(2 * T + w)}$ and $3T = {num(3 * T)}\\ \\mu\\mathrm{{s}}$) the first echo has been "
                     f"reflected by the generator and only the re-launched pulse is on the line. Its term is $\\tau_g\\Gamma_L\\Gamma_gf_i(t_0 - z/v - 2T)$, nonzero for "
                     f"$0 \\lt t_0 - z/v - 2T \\lt {num(w)}$, i.e. ${num(za)} \\lt z \\lt {num(za + L)}$ m, with height $\\tau_g\\Gamma_L\\Gamma_g\\cdot{A} = {num(tg * GL * Gg * A)}$ V"
                     + ("" if rect else f" and its peak where $t_0 - z/v - 2T = {num(tp)}$, at $z = {num(zpk)}$ m") + "."),
        "u_s": num(u), "dl_s": num(dl), "af_s": num(af), "ae_s": num(ae), "sf_s": num(sf), "se_s": num(se),
        "Vb_s": num(Vb), "Id_s": num(Id), "tc_s": num(tc), "zpv_s": num(zp / v), "twoT_s": num(2 * T),
        "tdel_s": num(w if rect else tp),
    })
    ca = data["correct_answers"]
    ca["Vov"] = float(Vb) + 0.0
    ca["tpk"] = float(tc)
    ca["Iov"] = float(Id) + 0.0
