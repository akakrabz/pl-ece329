"""Lecture 17 concept question: magnetization current, B and H in a long rod magnet,
dia-/para-/ferromagnetism read from chi_m (or mu_r), and B across an interface between two linear media.

(a)-(b) A long rod along u, uniformly magnetized M = s*M0*u_hat, no free current anywhere.
        Side wall: outward normal r_hat, J_sM = M x r_hat = s*M0 (u_hat x r_hat) = s*M0 phi_hat.
        End faces: n_hat = +-u_hat is parallel to M, so M x n_hat = 0; interior: curl M = 0.
        The side wall is a solenoid winding with nI -> M0: B = mu0 M inside far from the ends,
        H = B/mu0 - M = 0 there, B ~ 0 just outside the side wall.  An air-core solenoid with n turns/m
        gives the same B with I = M0/n.
(c)     chi_m < 0: diamagnetic; 0 < chi_m << 1: paramagnetic; chi_m >> 1: ferromagnetic (handbook values).
(d)     Plane w = 0; medium 1 (mu_r1) fills w > 0, medium 2 (mu_r2 >= 2) fills w < 0; no free surface current;
        n_hat = +w_hat points from 2 into 1.  B2n = B1n;  H2t = H1t  =>  B2t = (mu_r2/mu_r1) B1t;
        M2 = chi_m2 H2 = (mu_r2 - 1) B2 / (mu_r2 mu0)  =>  M2n = (mu_r2 - 1) B1n / (mu_r2 mu0).
(e)     checkbox pool; the "angle" statement is true exactly when mu_r2 > mu_r1 (tan theta proportional to mu).
"""
import math
import random

MU0 = 4e-7 * math.pi
AXES = ["x", "y", "z"]

# Permanent-magnet materials and realistic magnetizations (kA/m); mu0*M0 = 0.38-1.38 T, i.e. typical remanence.
MAGNETS = [
    ("sintered NdFeB", [950, 1000, 1050, 1100]),
    ("samarium-cobalt", [800, 850, 900]),
    ("alnico", [950, 1000, 1030]),
    ("hard-ferrite (ceramic)", [300, 320, 340]),
]

# SI volume susceptibilities at room temperature (CRC Handbook / standard tables), 2 significant figures.
DIA_PARA = [
    ("bismuth", -1.7e-4, "dia"), ("water", -9.0e-6, "dia"), ("copper", -9.6e-6, "dia"),
    ("silver", -2.4e-5, "dia"), ("gold", -3.4e-5, "dia"), ("lead", -1.6e-5, "dia"),
    ("diamond", -2.2e-5, "dia"),
    ("aluminium", 2.1e-5, "para"), ("platinum", 2.7e-4, "para"), ("palladium", 8.0e-4, "para"),
    ("titanium", 1.8e-4, "para"), ("magnesium", 1.2e-5, "para"),
    ("air, because of its oxygen", 3.7e-7, "para"), ("liquid oxygen at 90 K", 3.5e-3, "para"),
]
# Typical small-field relative permeabilities of ferromagnets.
FERRO = [("cobalt", 250), ("nickel", 600), ("iron, 99.8 % pure", 5000), ("mumetal", 50000),
         ("permalloy", 8000), ("electrical silicon steel", 4000)]
CLASS_NAME = {"dia": "diamagnetic", "para": "paramagnetic", "ferro": "ferromagnetic"}


def sci2(x):
    """x to 2 significant figures as LaTeX 'a.b\\times10^{e}'."""
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    if round(abs(m), 1) >= 10:
        e += 1
        m = x / 10 ** e
    return f"{m:.1f}\\times10^{{{e}}}"


def group3(digits):
    return "\\,".join(digits[i:i + 3] for i in range(0, len(digits), 3))


def mur_near_one(chi):
    """1 + chi printed with just enough decimals to show chi to 2 significant figures, grouped in threes."""
    dec = -int(math.floor(math.log10(abs(chi)))) + 1
    ip, fp = f"{1 + chi:.{dec}f}".split(".")
    return ip + "." + group3(fp)


def int_tex(n):
    s = str(int(n))
    if len(s) <= 4:
        return s
    head = len(s) % 3 or 3
    return s[:head] + "\\," + "\\,".join(s[i:i + 3] for i in range(head, len(s), 3))


def num(x, sig=4):
    s = f"{x:.{sig}g}"
    if "e" in s:
        m, e = s.split("e")
        return f"{m}\\times10^{{{int(e)}}}"
    return s


def vec_tex(comp):
    """comp: dict axis -> number; LaTeX like '3\\,\\hat{x} - 2\\,\\hat{z}'."""
    out = ""
    for ax in AXES:
        v = comp[ax]
        if v == 0:
            continue
        mag = num(abs(v))
        if not out:
            out = ("-" if v < 0 else "") + f"{mag}\\,\\hat{{{ax}}}"
        else:
            out += (" - " if v < 0 else " + ") + f"{mag}\\,\\hat{{{ax}}}"
    return out or "0"


def generate(data):
    # ---------------------------------------------------------------- (a)-(b) the rod magnet
    u = random.choice(AXES)
    s = random.choice([1, -1])
    mat, m0_list = random.choice(MAGNETS)
    M0 = random.choice(m0_list)                      # kA/m
    a_mm = random.choice([2, 3, 4, 5])
    L_cm = random.choice([20, 25, 30])
    n_turns = random.choice([500, 1000, 2000, 2500])
    Ieq = M0 * 1e3 / n_turns                          # A
    B_in = s * MU0 * M0 * 1e3 * 1e3                   # mT, along +u
    sgn = "+" if s > 0 else "-"
    M_tex = ("" if s > 0 else "-") + f"{M0}\\,\\hat{{{u}}}"

    jdir = [
        {"id": "plus", "text": r"on the curved side wall, along $+\hat{\phi}$", "correct": "true" if s > 0 else "false"},
        {"id": "minus", "text": r"on the curved side wall, along $-\hat{\phi}$", "correct": "false" if s > 0 else "true"},
        {"id": "axial", "text": r"on the curved side wall, along the rod's axis (parallel or antiparallel to $\mathbf{M}$)",
         "correct": "false"},
        {"id": "ends", "text": r"only on the two flat end faces, which are perpendicular to $\mathbf{M}$", "correct": "false"},
        {"id": "none", "text": r"nowhere: a uniform $\mathbf{M}$ has $\nabla\times\mathbf{M} = \mathbf{0}$", "correct": "false"},
    ]
    good = "plus" if s > 0 else "minus"
    fb = {
        "plus": None, "minus": None,
        "axial": r"$\mathbf{M}\times\hat{n}$ is perpendicular to $\mathbf{M}$, so the bound current cannot run along the axis.",
        "ends": r"On an end face $\hat{n} = \pm\hat{" + u + r"}$ is parallel to $\mathbf{M}$, so $\mathbf{M}\times\hat{n} = \mathbf{0}$: no current there.",
        "none": r"Inside, yes: $\nabla\times\mathbf{M} = \mathbf{0}$. But $\mathbf{M}$ drops to zero across the side wall, and that edge carries the surface current $\mathbf{M}\times\hat{n}$.",
    }
    wrong_sign = (r"Sign slip: $\mathbf{J}_{sM} = \mathbf{M}\times\hat{n}$ with the outward normal $\hat{n} = \hat{r}$, and $\hat{"
                  + u + r"}\times\hat{r} = +\hat{\phi}$, so here $\mathbf{J}_{sM} = " + sgn + r"M_0\,\hat{\phi}$. Writing $\hat{n}\times\mathbf{M}$ (or using the inward normal) reverses it.")
    for c in jdir:
        if c["id"] == good:
            c["feedback"] = (r"Right: $\mathbf{M}\times\hat{r} = " + sgn + r"M_0\,\hat{" + u + r"}\times\hat{r} = " + sgn
                             + r"M_0\,\hat{\phi}$, like the winding of a solenoid.")
        elif c["id"] in ("plus", "minus"):
            c["feedback"] = wrong_sign
        else:
            c["feedback"] = fb[c["id"]]
    random.shuffle(jdir)

    inside = [
        {"id": "ok", "correct": "true",
         "text": r"$\mathbf{B}\approx\mu_0\mathbf{M}$ and $\mathbf{H}\approx\mathbf{0}$ inside; just outside the side wall $\mathbf{B}\approx\mathbf{0}$.",
         "feedback": r"Right: the side wall is a solenoid winding with $nI\to M$, so $\mathbf{B}\approx\mu_0\mathbf{M}$ inside and nearly zero outside, and $\mathbf{H} = \mathbf{B}/\mu_0-\mathbf{M}\approx\mathbf{0}$."},
        {"id": "HM", "correct": "false",
         "text": r"$\mathbf{B}\approx\mu_0\mathbf{M}$ and $\mathbf{H}\approx\mathbf{M}$ inside; just outside the side wall $\mathbf{B}\approx\mathbf{0}$.",
         "feedback": r"That uses the vacuum relation $\mathbf{H} = \mathbf{B}/\mu_0$ inside the magnet. In matter $\mathbf{H} = \mathbf{B}/\mu_0-\mathbf{M}$, which is about $\mu_0\mathbf{M}/\mu_0-\mathbf{M} = \mathbf{0}$ here."},
        {"id": "cancel", "correct": "false",
         "text": r"$\mathbf{B}\approx\mathbf{0}$ and $\mathbf{H}\approx-\mathbf{M}$ inside, because the field of the bound current cancels the magnetization.",
         "feedback": r"That carries the dielectric picture ($-\mathbf{P}/\epsilon_0$) over with the wrong sign: inside a current loop the loop's field points along its moment, so the bound current reinforces $\mathbf{B}$. ($\mathbf{H}\approx-\mathbf{M}$ belongs to a thin plate magnetized through its thickness, not to a long rod.)"},
        {"id": "end", "correct": "false",
         "text": r"$\mathbf{B}\approx\tfrac12\mu_0\mathbf{M}$ and $\mathbf{H}\approx-\tfrac12\mathbf{M}$ inside.",
         "feedback": r"Those are the values at the centre of an end face, where only half of the equivalent solenoid lies on one side. Far from the ends the rod looks infinite: $\mathbf{B}\approx\mu_0\mathbf{M}$."},
        {"id": "leak", "correct": "false",
         "text": r"$\mathbf{B}\approx\mu_0\mathbf{M}$ inside and $\mathbf{B}\approx\tfrac12\mu_0\mathbf{M}$ just outside the side wall, as on the two sides of a flat current sheet.",
         "feedback": r"A flat sheet gives $\pm\tfrac12 J_s$ on its two sides, but the side wall is a closed cylindrical sheet (a solenoid): the contributions add inside and cancel outside, so $\mathbf{B}\approx\mathbf{0}$ just outside, far from the ends."},
    ]
    random.shuffle(inside)

    # ---------------------------------------------------------------- (c) classification by chi_m or mu_r
    pool = [dict(name=n, chi=c, cls=k) for n, c, k in DIA_PARA] + \
           [dict(name=n, chi=float(mr - 1), cls="ferro", mur=mr) for n, mr in FERRO]
    picks = [random.choice([m for m in pool if m["cls"] == k]) for k in ("dia", "para", "ferro")]
    rest = [m for m in pool if m not in picks]
    picks += random.sample(rest, 2)
    random.shuffle(picks)
    cls_items = []
    for m in picks:
        form = random.choice(["chi", "mur"])
        if form == "chi":
            val = sci2(m["chi"])
            text = r"a material with $\chi_m\approx " + val + "$"
        else:
            val = int_tex(m["mur"]) if m["cls"] == "ferro" else mur_near_one(m["chi"])
            text = r"a material with $\mu_r\approx " + val + "$"
        verdict = {"dia": r" \lt 0$", "para": r" \gt 0$ and $\ll 1$", "ferro": r" \gg 1$"}[m["cls"]]
        if form == "mur":
            reason = r"$\chi_m = \mu_r-1\approx " + sci2(m["chi"]) + verdict
        else:
            reason = r"$\chi_m" + verdict
        cls_items.append({"key": m["cls"], "text": text, "form": form, "chi": m["chi"], "name": m["name"],
                          "cls_name": CLASS_NAME[m["cls"]], "reason": reason})

    # ---------------------------------------------------------------- (d) interface between mu_r1 and mu_r2
    while True:
        mr1 = random.choice([1, 2, 3, 5, 10, 20, 50, 100])
        mr2 = random.choice([2, 4, 5, 10, 20, 50, 100, 400])
        if mr1 != mr2 and 1 / 50 <= mr2 / mr1 <= 400:
            break
    w = random.choice(AXES)
    tang = [ax for ax in AXES if ax != w]
    random.shuffle(tang)
    B1 = {w: random.choice([-1, 1]) * random.randint(2, 12),
          tang[0]: random.choice([-1, 1]) * random.randint(1, 9),
          tang[1]: random.randint(-9, 9)}
    r = mr2 / mr1
    B2 = {ax: (B1[ax] if ax == w else r * B1[ax]) for ax in AXES}
    M2n = (mr2 - 1) * B1[w] * 1e-3 / (mr2 * MU0) / 1e3     # kA/m
    H1n = B1[w] * 1e-3 / (mr1 * MU0)                        # A/m (for the answer panel)
    H2n = B1[w] * 1e-3 / (mr2 * MU0)
    angle_true = mr2 > mr1

    # ---------------------------------------------------------------- (e) statement pool
    facts = [
        ("Hdef", True, r"$\mathbf{H} = \mathbf{B}/\mu_0-\mathbf{M}$ holds in every material, linear or not.",
         r"It is the definition of $\mathbf{H}$; only $\mathbf{B} = \mu\mathbf{H}$ needs a linear medium."),
        ("ampere_free", True, r"In magnetostatics, $\oint_C\mathbf{H}\cdot d\mathbf{l}$ equals the <em>free</em> current through $C$, even where $C$ runs through magnetized material.",
         r"$\nabla\times\mathbf{H} = \mathbf{J}_{\text{free}}$: the magnetization current has been absorbed into $\mathbf{H}$."),
        ("remanence", True, r"A magnetized ferromagnet can have $\mathbf{B}\neq\mathbf{0}$ where $\mathbf{H} = \mathbf{0}$ (on the hysteresis loop this is the remanence $B_r$).",
         r"That is a permanent magnet; the rod of parts (a) and (b) has $\mathbf{H}\approx\mathbf{0}$ and $\mathbf{B}\approx\mu_0\mathbf{M}$ inside."),
        ("coercivity", True, r"The coercivity $H_c$ is the size of the reverse field needed to bring $B$ back to zero after the material has been magnetized.",
         r"That is its definition on the hysteresis loop."),
        ("iron_nonlinear", True, r"For iron, $B/(\mu_0H)$ is not a material constant: it depends on $H$ and on the sample's magnetic history.",
         r"Ferromagnets are non-linear and hysteretic; a tabulated $\mu_r$ is a typical small-field slope, and $B$ saturates near 2 T."),
        ("dia_opposes", True, r"In a diamagnetic material, $\mathbf{M}$ points opposite to $\mathbf{H}$.",
         r"$\chi_m\lt 0$, so $\mathbf{M} = \chi_m\mathbf{H}$ is antiparallel to $\mathbf{H}$."),
        ("divJM", True, r"The magnetization current density $\nabla\times\mathbf{M}$ has zero divergence, so it never piles up charge anywhere.",
         r"The divergence of a curl vanishes identically: bound loop currents close on themselves."),
        ("mu0chi", False, r"In a linear medium $\mathbf{M} = \mu_0\chi_m\mathbf{H}$, just as $\mathbf{P} = \epsilon_0\chi_e\mathbf{E}$.",
         r"$\mathbf{M}$ and $\mathbf{H}$ are both in A/m, so $\mathbf{M} = \chi_m\mathbf{H}$ with no $\mu_0$; $\mu_0\chi_m\mathbf{H}$ would be in tesla."),
        ("Bmu_minus_M", False, r"In a linear medium $\mathbf{H} = \mathbf{B}/\mu-\mathbf{M}$.",
         r"$\mathbf{B}/\mu$ is already $\mathbf{H}$; subtracting $\mathbf{M}$ again gives $(1-\chi_m)\mathbf{H}$. The correct forms are $\mathbf{H} = \mathbf{B}/\mu_0-\mathbf{M} = \mathbf{B}/\mu$."),
        ("para_domains", False, r"A paramagnet responds to a field through domains that grow and turn, as a ferromagnet does but more weakly.",
         r"Domains belong to ferromagnets only; in a paramagnet individual atomic moments are partly aligned against thermal agitation."),
        ("divH", False, r"$\nabla\cdot\mathbf{H} = 0$ everywhere, including across an interface between media with different $\mu$.",
         r"The law is $\nabla\cdot\mathbf{B} = 0$. At an interface $B_n$ is continuous, so $H_n = B_n/\mu$ jumps."),
        ("Bt_cont", False, r"With no free surface current, the tangential component of $\mathbf{B}$ is continuous across an interface between media with different $\mu$.",
         r"Tangential $\mathbf{H}$ is continuous; tangential $\mathbf{B} = \mu\mathbf{H}_t$ jumps by the ratio of the permeabilities (part (d))."),
        ("dia_attracted", False, r"A diamagnetic sample is pulled toward the region of strongest field, as a paramagnet is.",
         r"Diamagnets are weakly pushed out of strong-field regions; paramagnets are weakly pulled in."),
        ("remove_H", False, r"Switching off the applied field always returns a ferromagnet to $\mathbf{M} = \mathbf{0}$.",
         r"Hysteresis: a remanent magnetization survives at $H = 0$, which is what makes permanent magnets possible."),
        ("angle", angle_true, r"In part (d), the field lines in medium 2 make a larger angle with the interface normal than those in medium 1.",
         r"$\tan\theta_1/\tan\theta_2 = \mu_1/\mu_2$ (same $B_n$, tangential part scaled by $\mu$); here $\mu_{r2} = " + str(mr2)
         + ("$ is larger than" if angle_true else "$ is smaller than") + r" $\mu_{r1} = " + str(mr1) + "$, so the lines in medium 2 are "
         + ("farther from" if angle_true else "closer to") + " the normal."),
    ]
    fact_list = []
    for fid, truth, text, why in facts:
        fact_list.append({"id": fid, "correct": "true" if truth else "false", "text": text,
                          "feedback": ("True: " if truth else "False: ") + why,
                          "verdict": "true" if truth else "false", "why": why})
    random.shuffle(fact_list)

    # ---------------------------------------------------------------- answer-panel strings
    B1_tex = vec_tex(B1)
    B2_tex = vec_tex(B2)
    data["params"].update({
        "u": u, "mat": mat, "M0": M0, "M_tex": M_tex, "a_mm": a_mm, "L_cm": L_cm, "n_turns": n_turns,
        "sgn": sgn, "s": s, "Ieq_str": num(Ieq), "Bin_str": num(B_in), "jdir_choices": jdir, "inside_choices": inside,
        "cls_items": cls_items,
        "w": w, "t1": tang[0], "t2": tang[1], "mr1": mr1, "mr2": mr2, "B1_tex": B1_tex, "B2_tex": B2_tex,
        "B1": B1, "ratio_str": num(r), "M2n_str": num(M2n), "H1n_str": num(H1n), "H2n_str": num(H2n),
        "B1n": B1[w], "chi2": mr2 - 1,
        "facts": fact_list,
    })
    data["correct_answers"]["JsM"] = float(M0)
    data["correct_answers"]["Ieq"] = float(Ieq)
    data["correct_answers"]["Bin"] = float(B_in)
    for ax in AXES:
        data["correct_answers"]["B2" + ax] = float(B2[ax])
    data["correct_answers"]["M2n"] = float(M2n)
