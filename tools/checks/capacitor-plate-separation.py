"""Independent check for questions/electrostatics/capacitor-plate-separation.

Route: C from the given-Q chain (D = Q/A, E = D/eps0, V = integral of E across the gap); stored energies from the
field energy density 1/2 eps0 E^2 times the gap volume; the force on a plate as its charge times the OTHER plate's
sheet field rho_s/(2 eps0); your work as the numerical integral of that force over the displacement; the battery's
work as the integral of V0 dQ. Energy conservation is then checked, and the MC / select-all keys are re-derived
from these numbers.
"""
import numpy as np

EPS0 = 8.854e-12


def phys(d, check):
    p, c = d["params"], d["correct_answers"]
    mode, A, V0 = p["mode"], p["A_cm2"] * 1e-4, float(p["V0"])
    d1, d2 = p["d1"] * 1e-3, p["d2"] * 1e-3
    check(d2 > d1 > 0, "plates: d2 must exceed d1")

    def C_of(x):                                   # given-Q chain with Q = 1 C
        zs = np.linspace(0.0, x, 101)
        E = np.full_like(zs, (1.0 / A) / EPS0)
        return 1.0 / np.trapezoid(E, zs)

    def simpson(f, x):                             # composite Simpson, len(x) odd, uniform grid
        h = x[1] - x[0]
        return h / 3 * (f[0] + f[-1] + 4 * np.sum(f[1:-1:2]) + 2 * np.sum(f[2:-1:2]))

    C1, C2 = C_of(d1), C_of(d2)
    check(abs(C1 * 1e12 - c["C1"]) <= 1e-9 * C1 * 1e12, f"plates: C1 {c['C1']} vs {C1*1e12}")
    check(abs(C2 * 1e12 - c["C2"]) <= 1e-9 * C2 * 1e12, f"plates: C2 {c['C2']} vs {C2*1e12}")

    Q1 = C1 * V0                                    # charged to V0 at d1 in both setups

    def Q_at(x):                                    # plate charge at gap x
        return Q1 if mode == "Q" else C_of(x) * V0

    def state(x):
        Q = Q_at(x)
        E = Q / (EPS0 * A)                          # Gauss: D = rho_s = Q/A
        W = 0.5 * EPS0 * E ** 2 * A * x             # energy density times volume
        F = Q * (Q / A) / (2 * EPS0)                # charge times the other plate's field
        return Q, E, W, F

    Qa, Ea, Wa, Fa = state(d1)
    Qb, Eb, Wb, Fb = state(d2)
    check(abs(Wa * 1e6 - c["W1"]) <= 1e-9 * Wa * 1e6, f"plates: W1 {c['W1']} vs {Wa*1e6}")
    check(abs(Wb * 1e6 - c["W2"]) <= 1e-9 * Wb * 1e6, f"plates: W2 {c['W2']} vs {Wb*1e6}")
    check(abs(Fa * 1e3 - c["F1"]) <= 1e-9 * Fa * 1e3, f"plates: F1 {c['F1']} vs {Fa*1e3}")
    check(abs(Fb * 1e3 - c["F2"]) <= 1e-9 * Fb * 1e3, f"plates: F2 {c['F2']} vs {Fb*1e3}")

    # your work: integrate the (attractive) force over the displacement; battery: integrate V0 dQ
    xs = np.linspace(d1, d2, 2001)
    Fx = np.array([state(x)[3] for x in xs])
    W_you = float(simpson(Fx, xs))
    if mode == "V":
        Qx = np.array([state(x)[0] for x in xs])
        W_batt = float(np.sum(V0 * np.diff(Qx)))
    else:
        W_batt = 0.0
    check(abs(W_you * 1e6 - c["Wagent"]) <= 2e-6 * abs(W_you * 1e6), f"plates: W_you {c['Wagent']} vs integral {W_you*1e6}")
    check(abs(W_you + W_batt - (Wb - Wa)) <= 2e-6 * Wa, "plates: energy conservation W_you + W_batt = dW fails")

    # (e) MC key from the energy flow
    ch = p["energy_choices"]
    check(sum(x["correct"] == "true" for x in ch) == 1 and len(ch) == 4, "plates: (e) needs 4 choices, one correct")
    key = [x["text"] for x in ch if x["correct"] == "true"][0]
    if Wb > Wa and W_batt == 0.0:
        check(key.startswith("All of your work is stored"), "plates: (e) key wrong for isolated plates")
    else:
        check(W_batt < 0 and Wb < Wa and abs(-W_batt - 2 * (Wa - Wb)) <= 1e-6 * Wa, "plates: battery should absorb twice the energy drop")
        check(key.startswith("The stored energy falls while charge flows back"), "plates: (e) key wrong for battery case")

    # (f) pool truths from the computed states
    F_d1_other_setup = (C1 * V0) ** 2 / (2 * EPS0 * A)
    F_d1_2V = (C1 * 2 * V0) ** 2 / (2 * EPS0 * A)
    truth = {
        "sameE": abs(Eb - Ea) <= 1e-9 * Ea,
        "sameQ": abs(Qb - Qa) <= 1e-9 * Qa,
        "sameV": abs(Qb / C2 - Qa / C1) <= 1e-9 * V0,
        "Wup": Wb > Wa,
        "density": abs(0.5 * EPS0 * Eb ** 2 - 0.5 * EPS0 * Ea ** 2) <= 1e-9 * EPS0 * Ea ** 2,
        "backflow": mode == "V" and Qb < Qa,
        "positive": W_you > 0,
        "Cdown": C2 < C1,
        "sameforce": abs(Fa - F_d1_other_setup) <= 1e-9 * Fa,
        "fullfield": abs(Fa - Qa * Ea) <= 1e-9 * Fa,
        "growforce": Fb > Fa * (1 + 1e-9),
        "doubleV": abs(F_d1_2V - 2 * Fa) <= 1e-9 * Fa,
        "workisdW": abs(W_you - (Wb - Wa)) <= 1e-6 * Wa,
    }
    ids = [x["id"] for x in p["pool"]]
    check(sorted(ids) == sorted(truth), f"plates: pool ids {ids}")
    for x in p["pool"]:
        if x["id"] in truth:
            check((x["correct"] == "true") == truth[x["id"]], f"plates: pool statement {x['id']} keyed {x['correct']} ({mode})")
    ntrue = sum(x["correct"] == "true" for x in p["pool"])
    check(ntrue >= 4 and len(ids) - ntrue >= 4 and 8 <= len(ids) <= 14, "plates: pool balance/size")
    # realism: field below air breakdown (3 MV/m), small gap compared with the plate size
    check(max(Ea, Eb) <= 1.0e6 + 1 and d2 <= 0.1 * np.sqrt(A), "plates: field or fringing out of the intended range")
