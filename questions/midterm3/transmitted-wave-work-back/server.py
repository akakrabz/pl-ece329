"""Midterm 3 exam-style: normal incidence, worked backwards from the transmitted wave.

Family of SP18 Exam 3 #2 (re-parameterized): free space on the incident side, a perfect non-magnetic dielectric beyond
the plane w = 0, and only the transmitted phasor is given,  E_t = a_hat E_t exp(-/+ j beta_t w)  at a known omega.
    beta_1 = omega / c,   n = beta_t / beta_1,   eps_r = n^2,   eta_t = omega mu0 / beta_t = 120 pi / n
    Gamma = (eta_t - eta_0)/(eta_t + eta_0) = (1 - n)/(1 + n),   tau = 1 + Gamma = 2/(1 + n)
    E_i0 = E_t / tau,   E_r0 = Gamma E_i0,   reflected phasor  a_hat E_r0 exp(+/- j beta_1 w)   (travels back, in region 1)
    H_r = (-u) x a_hat E_r0 / eta_0,     transmitted power fraction 1 - Gamma^2,   <S_t> = u E_t^2 / (2 eta_t)
Numbers are chosen so that every amplitude is an integer.
"""
import math
import random

AX = ["x", "y", "z"]
C0 = 3e8
ETA0 = 120 * math.pi
# n, (E_t unit, E_i unit, E_r unit) with E_i = E_t (1+n)/2 and E_r = E_i (1-n)/(1+n)
CASES = {1.2: (10, 11, -1), 1.5: (8, 10, -2), 2: (2, 3, -1), 3: (1, 2, -1), 4: (2, 5, -3)}


def fnum(x, sig=4):
    if abs(x) < 1e-12:
        return "0"
    s = f"{x:.{sig}g}"
    if "e" in s:
        e = int(s.split("e")[1])
        s = f"{x:.{max(0, sig - 1 - e)}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def cross(a, sa, b, sb):
    ia, ib = AX.index(a), AX.index(b)
    return AX[3 - ia - ib], sa * sb * (1 if (ia + 1) % 3 == ib else -1)


def hat(ax, sg=1):
    return ("+" if sg > 0 else "-") + "\\hat{" + ax + "}"


def frac_tex(num, den):
    g = math.gcd(abs(num), den)
    num, den = num // g, den // g
    return str(num) if den == 1 else f"{'-' if num < 0 else ''}\\tfrac{{{abs(num)}}}{{{den}}}"


def generate(data):
    n = random.choice(list(CASES))
    ut, ui, ur = CASES[n]
    k = random.choice([1, 2, 3, 5] if ut >= 8 else [1, 2, 3, 5, 10, 20])
    Et, Ei, Er = ut * k, ui * k, ur * k
    om_m, om_e = random.choice([(3, 9), (6, 9), (1.5, 9), (3, 8), (6, 8), (9, 8)])
    omega = om_m * 10 ** om_e
    beta1 = omega / C0
    betat = n * beta1
    eta_t = ETA0 / n
    w, a = random.sample(AX, 2)
    s = random.choice([1, -1])
    hax, hsg = cross(w, -s, a, 1)                 # reflected wave: (-u) x a_hat
    Gam, tau = (1 - n) / (1 + n), 2 / (1 + n)
    Hr = hsg * Er / ETA0 * 1e3                    # mA/m, signed along +hax
    pct = (1 - Gam ** 2) * 100
    St = s * Et ** 2 / (2 * eta_t)

    side1 = f"{w} \\lt 0" if s > 0 else f"{w} \\gt 0"
    side2 = f"{w} \\gt 0" if s > 0 else f"{w} \\lt 0"
    sg_t = "-" if s > 0 else "+"
    sg_r = "+" if s > 0 else "-"
    bt_s, b1_s = fnum(betat), fnum(beta1)
    field_tex = f"\\tilde{{\\mathbf{{E}}}}_t = \\hat{{{a}}}\\,{Et}\\,e^{{{sg_t}j{bt_s}{w}}}"

    def rtex(amp, sgn, bs):
        return f"$\\hat{{{a}}}\\,({amp})\\,e^{{{sgn}j{bs}{w}}}$" if amp < 0 else f"$\\hat{{{a}}}\\,{amp}\\,e^{{{sgn}j{bs}{w}}}$"

    refl = [
        {"text": rtex(Er, sg_r, b1_s), "correct": "true", "amp": Er, "k": -s * beta1, "feedback": "Correct."},
        {"text": rtex(Er, sg_t, b1_s), "correct": "false", "amp": Er, "k": s * beta1,
         "feedback": "This phasor travels the same way as the incident wave. The reflected wave travels back, so its space factor has the opposite sign."},
        {"text": rtex(Er, sg_r, bt_s), "correct": "false", "amp": Er, "k": -s * betat,
         "feedback": "The reflected wave lives in free space, so it uses the free-space phase constant, not the one of the dielectric."},
        {"text": rtex(-Er, sg_r, b1_s), "correct": "false", "amp": -Er, "k": -s * beta1,
         "feedback": "The sign is wrong: the dielectric has the smaller impedance, so the reflection coefficient is negative and the reflected field is opposite to the incident one at the interface."},
    ]
    random.shuffle(refl)

    data["params"].update({
        "w": w, "a": a, "hax": hax, "side1": side1, "side2": side2, "field_tex": field_tex,
        "om_tex": f"{fnum(om_m)}\\times10^{{{om_e}}}", "refl": refl,
        # solution strings
        "s_udir": hat(w, s), "s_b1": b1_s, "s_bt": bt_s, "s_n": fnum(n), "s_epsr": fnum(n * n), "s_etat": fnum(eta_t),
        "s_etat_tex": f"120\\pi/{fnum(n)}", "s_Gam": fnum(Gam), "s_tau": fnum(tau),
        "s_Gam_tex": frac_tex(int(round((1 - n) * 10)), int(round((1 + n) * 10))),
        "s_tau_tex": frac_tex(20, int(round((1 + n) * 10))),
        "s_Et": str(Et), "s_Ei": str(Ei), "s_Er": str(Er), "s_sgr": sg_r, "s_sgt": sg_t,
        "s_cross": f"({hat(w, -s)})\\times\\hat{{{a}}} = {hat(hax, hsg)}", "s_Hr": fnum(Hr), "s_Hrmag": fnum(abs(Er) / ETA0 * 1e3),
        "s_pct": fnum(pct), "s_St": fnum(St), "s_Stmag": fnum(abs(St)), "s_G2": fnum(Gam ** 2),
        # raw data for the independent check
        "s": s, "omega": omega, "betat": betat, "Et": Et,
    })
    ca = data["correct_answers"]
    ca["beta1"], ca["epsr"], ca["eta_t"] = beta1, n * n, eta_t
    ca["Gam"], ca["tau"] = Gam, tau
    ca["Ei"], ca["Er"] = Ei, Er
    ca["Hr"] = Hr
    ca["pct"], ca["St"] = pct, St
