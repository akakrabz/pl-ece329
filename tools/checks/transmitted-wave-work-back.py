"""Independent check for midterm3/transmitted-wave-work-back.

Route: no reflection formula is used. (1) eps_r from the dispersion relation of the dielectric, beta_t^2 = w^2 mu0 eps;
(2) the incident and reflected amplitudes from the two boundary conditions (tangential E and tangential H continuous)
solved as a 2x2 linear system, with each wave's H obtained from Faraday's law on its phasor by central differences;
(3) Gamma and tau as ratios of those amplitudes; (4) power from Re(E x H*)/2 of each wave; (5) each reflected-phasor
choice is tested for: Helmholtz equation of free space, power flowing away from the interface, and the boundary
condition on tangential E.
"""
import cmath
import math

import numpy as np

MU0 = 4e-7 * math.pi
C0 = 3e8
EPS0 = 1 / (MU0 * C0 ** 2)
IDX = {"x": 0, "y": 1, "z": 2}


def unit(ax):
    v = np.zeros(3); v[IDX[ax]] = 1.0; return v


def close(a, b, rel=1e-6, ab=1e-9):
    return abs(a - b) <= rel * max(abs(a), abs(b)) + ab


def phys(d, check):
    p, ca = d["params"], d["correct_answers"]
    q = "transmitted-wave-work-back"
    w, a, s, om, bt, Et = p["w"], p["a"], p["s"], p["omega"], p["betat"], p["Et"]
    ea, ew, eh = unit(a), unit(w), unit(p["hax"])
    iw = IDX[w]

    def wave(amp, k):                               # phasor a_hat amp exp(-j k w)
        return lambda r: ea * amp * cmath.exp(-1j * k * r[iw])

    def Hof(E, k):
        h = 1e-4 / abs(k)
        def H(r):
            J = np.zeros((3, 3), dtype=complex)
            for j in range(3):
                dr = np.zeros(3); dr[j] = h
                J[:, j] = (E(r + dr) - E(r - dr)) / (2 * h)
            curl = np.array([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]])
            return -curl / (1j * om * MU0)
        return H

    r0 = np.zeros(3)
    b1 = om * math.sqrt(MU0 * EPS0)
    epsr = bt ** 2 / (om ** 2 * MU0 * EPS0)
    check(close(b1, ca["beta1"], 1e-5) and close(epsr, ca["epsr"], 1e-5), f"{q}: beta1 or eps_r")
    check(close(math.sqrt(MU0 / (epsr * EPS0)), ca["eta_t"], 1e-5), f"{q}: eta_t")
    # unit-amplitude waves and their H on the interface, component along the H axis
    Etf = wave(Et, s * bt)
    ht = Hof(Etf, s * bt)(r0) @ eh
    hi1 = Hof(wave(1.0, s * b1), s * b1)(r0) @ eh
    hr1 = Hof(wave(1.0, -s * b1), -s * b1)(r0) @ eh
    sol = np.linalg.solve(np.array([[1.0, 1.0], [hi1, hr1]], dtype=complex), np.array([Et, ht], dtype=complex))
    Ei, Er = sol[0].real, sol[1].real
    check(abs(sol[0].imag) < 1e-6 and abs(sol[1].imag) < 1e-6, f"{q}: complex amplitudes?")
    check(close(Ei, ca["Ei"], 1e-5) and close(Er, ca["Er"], 1e-5), f"{q}: E_i0, E_r0 = {Ei}, {Er} vs {ca['Ei']}, {ca['Er']}")
    check(close(Er / Ei, ca["Gam"], 1e-5) and close(Et / Ei, ca["tau"], 1e-5), f"{q}: Gamma, tau")
    check(close((Hof(wave(Er, -s * b1), -s * b1)(r0) @ eh).real * 1e3, ca["Hr"], 1e-5), f"{q}: H_r")

    def Sw(amp, k):
        E = wave(amp, k)
        return 0.5 * np.real(np.cross(E(r0), np.conj(Hof(E, k)(r0))))[iw]

    Si, St = Sw(Ei, s * b1), Sw(Et, s * bt)
    check(close(St, ca["St"], 1e-5) and St * s > 0, f"{q}: <S_t> {St} vs {ca['St']}")
    check(close(100 * St / Si, ca["pct"], 1e-5), f"{q}: transmitted percentage {100 * St / Si} vs {ca['pct']}")
    check(close(Si + Sw(Er, -s * b1), St, 1e-5), f"{q}: power not conserved at the interface")
    # (d) reflected-phasor choices
    good = 0
    for c in p["refl"]:
        helm = close(abs(c["k"]), b1, 1e-9)
        away = Sw(c["amp"], c["k"]) * s < 0
        bc = close(Ei + c["amp"], Et, 1e-6)
        ok = helm and away and bc
        check((c["correct"] == "true") == ok, f"{q}: reflected choice {c['text']} mis-marked")
        good += ok
    check(good == 1 and len({c["text"] for c in p["refl"]}) == 4, f"{q}: reflected MC")
