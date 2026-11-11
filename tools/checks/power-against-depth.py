"""Independent check for midterm3/power-against-depth.

Route: (1) take the frequency the question gives as the answer and compute the exact gamma and eta of the medium
(sigma, eps_r, mu0) as principal complex roots: they must reproduce alpha = beta = a and |eta|, 45 deg to well under
1 % (so the good-conductor reading is legitimate for this medium and frequency); (2) H from Faraday's law on the given
phasor, <S> = Re(E x H*)/2; (3) the heat in the layer from the volume integral of sigma |E|^2 / 2 by quadrature;
(4) decibels from the ratio of two computed power densities; (5) the scaling item from the exact skin depth of the
changed medium/frequency.
"""
import cmath
import math

import numpy as np

MU0 = 4e-7 * math.pi
EPS0 = 8.8541878128e-12


def close(a, b, rel=1e-6):
    return abs(a - b) <= rel * max(abs(a), abs(b)) + 1e-15


def exact(f, sigma, epsr):
    om = 2 * math.pi * f
    y = sigma + 1j * om * epsr * EPS0
    return cmath.sqrt(1j * om * MU0 * y), cmath.sqrt(1j * om * MU0 / y)


def phys(d, check):
    p, ca = d["params"], d["correct_answers"]
    q = "power-against-depth"
    a, sigma, E0, d1, fac = p["a"], p["sigma"], p["E0"], p["d"], p["Sfac"]
    f = ca["f"] * 1e3
    om = 2 * math.pi * f
    gam, eta = exact(f, sigma, p["epsr"])
    check(sigma / (om * p["epsr"] * EPS0) > 100, f"{q}: loss tangent only {sigma / (om * p['epsr'] * EPS0):.1f}; not a clear good conductor")
    check(close(gam.real, a, 5e-3) and close(gam.imag, a, 5e-3), f"{q}: exact gamma {gam} is not a(1+j), a={a}")
    check(close(abs(eta), ca["eta_mag"], 5e-3) and abs(math.degrees(cmath.phase(eta)) - 45) < 0.3, f"{q}: eta {eta}")
    check(close(math.exp(-a * ca["delta"]), math.exp(-1)), f"{q}: delta")
    check(close(cmath.exp(-1j * a * ca["lam"]), 1.0, 1e-9), f"{q}: lambda")

    g = a * (1 + 1j)
    ia = "xy".index(p["ax"])

    def E(z):
        v = np.zeros(3, dtype=complex); v[ia] = E0 * cmath.exp(-g * z); return v

    def H(z):                                   # curl E = -j w mu H, fields depend on z only
        h = 1e-5 / a
        dE = (E(z + h) - E(z - h)) / (2 * h)
        curl = np.array([-dE[1], dE[0], 0.0])
        return -curl / (1j * om * MU0)

    def Sz(z):
        return 0.5 * np.real(np.cross(E(z), np.conj(H(z))))[2]

    check(close(Sz(0.0) * fac, ca["S0"], 1e-5), f"{q}: S(0) {Sz(0.0) * fac} vs {ca['S0']}")
    check(close(Sz(d1) * fac, ca["Sd"], 1e-5), f"{q}: S(d) {Sz(d1) * fac} vs {ca['Sd']}")
    check(close(10 * math.log10(Sz(0.3 / a) / Sz(0.8 / a)) / (0.5 / a), ca["dBm"], 1e-5), f"{q}: dB per metre")
    check(close(10 * math.log10(Sz(0.0) / Sz(ca["zX"])), p["XdB"], 1e-5), f"{q}: depth for {p['XdB']} dB")
    zs = np.linspace(0, d1, 20001)
    heat = np.trapezoid(0.5 * sigma * np.abs(E0 * np.exp(-g * zs)) ** 2, zs)
    check(close(heat * fac, ca["Qd"], 1e-5), f"{q}: heat in the layer {heat * fac} vs {ca['Qd']}")
    # (a) regime
    check(sum(c["correct"] == "true" for c in p["regime"]) == 1 and all((c["correct"] == "true") == (c["key"] == "good") for c in p["regime"]), f"{q}: regime MC")
    # (f) scaling from the exact skin depth
    f2, s2 = f, sigma
    if p["what"] == "f_up":
        f2 = f * p["mfac"] ** 2
    elif p["what"] == "f_down":
        f2 = f / p["mfac"] ** 2
    else:
        s2 = sigma * p["mfac"] ** 2
    ratio = (1 / exact(f2, s2, p["epsr"])[0].real) / (1 / gam.real)
    check(s2 / (2 * math.pi * f2 * p["epsr"] * EPS0) > 50, f"{q}: changed medium is no longer a good conductor")
    for c in p["scal"]:
        check((c["correct"] == "true") == (abs(c["key"] - ratio) < 0.02 * ratio), f"{q}: scaling choice {c['key']} mis-marked (true ratio {ratio})")
    check(len(p["scal"]) == 4, f"{q}: scaling MC needs 4 choices")
