"""Midterm 3 exam-style: three line sections in cascade, driven by a narrow pulse.

Family of FA23 Exam 3 #3 and SP19 Exam 3 #4 (re-parameterized). Sections Z1, Z2, Z3 with one-way delays T1, T2, T3
between a generator (Rg) and a load (RL). For a wave going from section j into section k the line beyond acts as a
load Zk:
    Gamma_jk = (Zk - Zj)/(Zk + Zj),   tau_jk = 1 + Gamma_jk = 2 Zk/(Zk + Zj)        (voltage)
    current coefficients:  reflection -Gamma_jk,  transmission 1 - Gamma_jk = 2 Zj/(Zk + Zj)
    launched pulse V1 = V0 Z1/(Rg + Z1)
    echo of the second junction back at the generator: time 2 (T1 + T2), height V1 tau_12 Gamma_23 tau_21
    first arrival at the load: time T1 + T2 + T3, load voltage V1 tau_12 tau_23 (1 + Gamma_L)
    generator end when the first echo (from junction 1) arrives at 2 T1:  Gamma_12 V1 (1 + Gamma_g)
The delays are chosen so that no other echo reaches the generator at 2 (T1 + T2).
"""
import random
from fractions import Fraction as Fr

DELAYS = [(2, 1), (2, 3), (1, 0.5), (1, 1.5), (3, 2), (1, 2.5), (3, 1), (2, 0.5)]


def fnum(x, sig=4):
    x = float(x)
    if abs(x) < 1e-12:
        return "0"
    s = f"{x:.{sig}g}"
    if "e" in s:
        e = int(s.split("e")[1])
        s = f"{x:.{max(0, sig - 1 - e)}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def ftex(q):
    q = Fr(q)
    if q.denominator == 1:
        return str(q.numerator)
    return f"{'-' if q < 0 else ''}\\tfrac{{{abs(q.numerator)}}}{{{q.denominator}}}"


def circuit(Rg, Z, T, RL):
    xs = [90, 180, 270, 360]
    out = ['<svg viewBox="0 0 450 104" width="450" style="max-width:100%" role="img" aria-label="Generator, three line sections in cascade, load">',
           '<g fill="none" stroke="currentColor" stroke-width="1.5">',
           '<circle cx="30" cy="56" r="13"/><path d="M30 43V24h12l4-6 8 12 8-12 8 12 4-6h16M30 69v19h60M90 24h270M90 88h270M360 24v20M360 68v20"/>',
           '<rect x="352" y="44" width="16" height="24"/>',
           '<path stroke-dasharray="3 3" d="M180 16V96M270 16V96"/></g><g fill="currentColor" font-size="12">',
           f'<text x="40" y="12">Rg = {Rg} &#937;</text><text x="372" y="60">RL = {RL} &#937;</text>']
    for i in range(3):
        cx = (xs[i] + xs[i + 1]) / 2
        out.append(f'<text x="{cx}" y="50" text-anchor="middle">Z{i + 1} = {Z[i]} &#937;</text>')
        out.append(f'<text x="{cx}" y="68" text-anchor="middle">T{i + 1} = {fnum(T[i])} &#956;s</text>')
    out.append('<text x="76" y="102">z = 0</text><text x="150" y="12">junction 1</text><text x="244" y="12">junction 2</text></g></svg>')
    return "".join(out)


def generate(data):
    while True:
        Z1, Z2, Z3 = (random.choice([25, 50, 75, 100, 150, 300]) for _ in range(3))
        if Z1 == Z2 or Z2 == Z3:
            continue
        rg = random.choice([Fr(1), Fr(1), Fr(1, 2), Fr(2), Fr(3), Fr(1, 3)])
        rl = random.choice([Fr(1), Fr(1), Fr(1, 2), Fr(2), Fr(3), Fr(1, 3)])
        Rg, RL = Z1 * rg, Z3 * rl
        V0 = random.choice([10, 20, 30, 60, 120])
        if Rg.denominator == 1 and RL.denominator == 1:
            break
    Rg, RL = int(Rg), int(RL)
    T1, T2 = random.choice(DELAYS)
    T3 = random.choice([0.5, 1, 2])
    tg = Fr(Z1, Rg + Z1)
    V1 = tg * V0
    G12, G23 = Fr(Z2 - Z1, Z2 + Z1), Fr(Z3 - Z2, Z3 + Z2)
    t12, t21, t23 = 1 + G12, 1 - G12, 1 + G23
    GL, Gg = Fr(RL - Z3, RL + Z3), Fr(Rg - Z1, Rg + Z1)
    echo = V1 * t12 * G23 * t21
    Vload = V1 * t12 * t23 * (1 + GL)
    Vgen = G12 * V1 * (1 + Gg)
    f = float

    cur = [
        {"val": f(1 - G12), "text": "$1-\\Gamma_{12} = \\dfrac{2Z_1}{Z_1+Z_2}$", "correct": "true", "feedback": "Correct."},
        {"val": f(1 + G12), "text": "$1+\\Gamma_{12} = \\dfrac{2Z_2}{Z_1+Z_2}$", "correct": "false",
         "feedback": "That is the voltage transmission coefficient. The transmitted current is the transmitted voltage divided by $Z_2$, the incident current the incident voltage divided by $Z_1$."},
        {"val": f(-G12), "text": "$-\\Gamma_{12} = \\dfrac{Z_1-Z_2}{Z_1+Z_2}$", "correct": "false",
         "feedback": "That is the current reflection coefficient: it describes the wave sent back, not the one that goes on."},
        {"val": f(-(1 + G12)), "text": "$-(1+\\Gamma_{12}) = -\\dfrac{2Z_2}{Z_1+Z_2}$", "correct": "false",
         "feedback": "The sign flip belongs to backward waves. The transmitted wave still travels toward $+z$, and its current is not the voltage coefficient with a minus sign."},
    ]
    random.shuffle(cur)

    data["params"].update({
        "svg": circuit(Rg, (Z1, Z2, Z3), (T1, T2, T3), RL), "V0": V0, "t2T1": fnum(2 * T1), "cur": cur,
        # solution strings
        "s_tg": ftex(tg), "s_V1": fnum(V1), "s_G12": ftex(G12), "s_t12": ftex(t12), "s_t21": ftex(t21), "s_G23": ftex(G23),
        "s_GI23": ftex(-G23), "s_t23": ftex(t23), "s_GL": ftex(GL), "s_Gg": ftex(Gg),
        "s_techo": fnum(2 * (T1 + T2)), "s_echo": fnum(echo), "s_tload": fnum(T1 + T2 + T3), "s_Vload": fnum(Vload),
        "s_Vgen": fnum(Vgen), "s_first": fnum(G12 * V1), "s_tau_i": fnum(1 - G12),
        "s_matched_g": "The generator is matched ($\\Gamma_g = 0$), so the echo is simply absorbed." if Gg == 0 else "The generator is not matched, so the echo and its own reflection are present together.",
        # raw data for the independent check
        "Z": [Z1, Z2, Z3], "T": [T1, T2, T3], "Rg": Rg, "RL": RL,
    })
    ca = data["correct_answers"]
    ca["tg"], ca["V1"] = f(tg), f(V1)
    ca["G12"], ca["t12"], ca["GI23"] = f(G12), f(t12), f(-G23)
    ca["techo"], ca["echo"] = 2 * (T1 + T2), f(echo)
    ca["tload"], ca["Vload"] = T1 + T2 + T3, f(Vload)
    ca["Vgen"] = f(Vgen)
