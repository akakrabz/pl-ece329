"""Independent check for midterm3/damped-wave-magnetic-medium.

Route: (1) get sigma and eps from gamma by complex algebra on gamma^2 = j w mu (sigma + j w eps); (2) rebuild gamma and
eta as principal square roots from that medium; (3) build the E phasor as a vector field and get H from Faraday's law
by central differences (curl E = -j w mu H), with no use of eta or of a direction rule; (4) <S> = Re(E x H*)/2 as a
vector; (5) loss tangents from sigma and eps directly. Every multiple-choice option is judged from these fields.
"""
import cmath
import math

import numpy as np

MU0 = 4e-7 * math.pi
EPS0 = 1e-9 / (36 * math.pi)
IDX = {"x": 0, "y": 1, "z": 2}


def unit(key):
    v = np.zeros(3)
    v[IDX[key[-1]]] = -1.0 if key[0] == "-" else 1.0
    return v


def close(a, b, rel=1e-6, ab=1e-12):
    return abs(a - b) <= rel * max(abs(a), abs(b)) + ab


def phys(d, check):
    p, ca = d["params"], d["correct_answers"]
    q = "damped-wave-magnetic-medium"
    al, be, om, mu, s = p["alpha"], p["beta"], p["omega"], p["mur"] * MU0, p["s"]
    w, a = p["w"], p["a"]
    gam = complex(al, be)
    y = gam ** 2 / (1j * om * mu)                      # sigma + j omega eps
    sig, eps = y.real, y.imag / om
    check(sig > 0 and eps > 0.999 * EPS0, f"{q}: unphysical medium sigma={sig}, eps_r={eps / EPS0}")
    g2 = cmath.sqrt(1j * om * mu * (sig + 1j * om * eps))
    eta = cmath.sqrt(1j * om * mu / (sig + 1j * om * eps))
    check(close(g2.real, ca["alpha"]) and close(g2.imag, ca["beta"]), f"{q}: alpha/beta do not belong to a medium")
    check(close(cmath.exp(-1j * g2.imag * ca["lam"]), 1.0, ab=1e-9), f"{q}: lambda is not one full phase turn")
    check(close(om / g2.imag, ca["vp"] * 1e8), f"{q}: v_p")
    check(close(math.exp(-g2.real * ca["delta"]), math.exp(-1)), f"{q}: delta")
    check(close(abs(eta), ca["eta_mag"]) and close(math.degrees(cmath.phase(eta)), ca["tau"]), f"{q}: eta polar form")

    E0c = p["E0"] * cmath.exp(1j * math.radians(p["phiE"]))
    ea, iw = unit("+" + a), IDX[w]

    def E(r):
        return ea * E0c * cmath.exp(-s * gam * r[iw])

    def H(r):
        h = 1e-4 / abs(gam)
        J = np.zeros((3, 3), dtype=complex)            # J[i][j] = dE_i/dx_j
        for j in range(3):
            dr = np.zeros(3); dr[j] = h
            J[:, j] = (E(r + dr) - E(r - dr)) / (2 * h)
        curl = np.array([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]])
        return -curl / (1j * om * mu)

    r0 = np.zeros(3)
    H0v = H(r0)
    # (a) direction: the side on which |E| shrinks
    for c in p["dir_choices"]:
        u = unit(c["key"])
        shrinks = abs(E(r0 + 0.1 * u / al) @ ea) < abs(E(r0) @ ea) * 0.95
        check((c["correct"] == "true") == shrinks, f"{q}: direction choice {c['key']} mis-marked")
    # (d) H phasor: direction, size, phase
    for c in p["h_choices"]:
        proj = H0v @ unit(c["key"])
        good = abs(proj) > 0.999 * np.linalg.norm(H0v) and close(cmath.phase(proj / cmath.exp(1j * math.radians(ca["phiH"]))), 0, ab=1e-5)
        check((c["correct"] == "true") == bool(good), f"{q}: h choice {c['key']} mis-marked")
    check(close(np.linalg.norm(H0v) * 1e3, ca["H0"], rel=1e-5), f"{q}: H0 {np.linalg.norm(H0v) * 1e3} vs {ca['H0']}")
    check(-180 < ca["phiH"] <= 180, f"{q}: phiH outside (-180, 180]")
    # (e) average Poynting vector on the plane w = w1
    r1 = np.zeros(3); r1[iw] = p["w1"]
    S = 0.5 * np.real(np.cross(E(r1), np.conj(H(r1))))
    check(close(S[iw] * 1e3, ca["Sw"], rel=1e-5), f"{q}: <S_w> {S[iw] * 1e3} vs {ca['Sw']}")
    check(abs(S[iw]) > 0.999 * np.linalg.norm(S) and S[iw] * s > 0 and p["w1"] * s > 0, f"{q}: <S> not along the travel / plane not downstream")
    # (f) loss tangents and class
    check(close(sig / (om * eps), ca["tanl"]), f"{q}: loss tangent")
    t2 = sig / (p["K"] * om * eps)
    check(close(t2, ca["tanl2"]), f"{q}: loss tangent at the new frequency")
    check(not (0.05 <= t2 <= 0.2 or 5 <= t2 <= 20), f"{q}: new loss tangent {t2} too close to a threshold")
    truth = "good" if t2 > 10 else "diel" if t2 < 0.1 else "mid"
    for c in p["cls_choices"]:
        check((c["correct"] == "true") == (c["key"] == truth), f"{q}: class choice {c['key']} mis-marked")
