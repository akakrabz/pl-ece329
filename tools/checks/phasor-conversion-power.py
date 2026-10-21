"""Independent check for waves/phasor-conversion-power (Lecture 21).

(a) every displayed field and every displayed phasor is parsed from its LaTeX; each field is evaluated as a real
    function of (u, t) and each phasor through Re{E~(u) e^{j w t}} on a grid of samples. Each field must agree with
    exactly one option (the keyed one); the three extra options must agree with no field; all options must differ.
(b) E~ is parsed from the LaTeX; beta is checked against 2 pi f sqrt(mu eps). H~ is NOT built from u x E/eta: it is
    -curl(E~)/(j w mu) by central differences of the complex field (phasor Faraday), and phasor Ampere
    curl(H~) = j w eps E~ is checked with nested differences. <S> is the period average of np.cross of the real
    fields Re{E~ e^{jwt}}, Re{H~ e^{jwt}} (trapezoid quadrature).
(c) the template is parsed; each MC direction is tested by riding along it at the phase speed w/|k| and requiring the
    real field to stay the same (only the true direction of travel passes); lambda from zero-crossing spacing.
(d) the plasma: N from f_p (N = w_p^2 m eps0/e^2); the electrons' velocity is time-stepped from m dv/dt = -e E at a fixed
    point; beta^2 is solved by least squares from the wave equation d2E/dz2 = mu0 dJ/dt + mu0 eps0 d2E/dt2 with
    E = cos(wt - beta z); the MC key follows from the sign of beta^2 at the second frequency.
"""
import cmath
import math
import re

import numpy as np

C = 3e8
MU0 = 4e-7 * math.pi
EPS0 = 1.0 / (MU0 * C * C)
QE, ME = 1.602e-19, 9.109e-31
IDX = {"x": 0, "y": 1, "z": 2}
E3 = np.eye(3)


def pifrac(s):
    """'2\\pi/3' -> 2/3 ; '\\pi' -> 1 ; '\\pi/4' -> 1/4 (units of pi)."""
    m = re.fullmatch(r"(\d*)\\pi(?:/(\d+))?", s)
    return (int(m.group(1)) if m.group(1) else 1) / (int(m.group(2)) if m.group(2) else 1)


def parse_field(tex):
    m = re.fullmatch(r"\$(-?)(\d+)\\(sin|cos)\((.*)\)\\,\\hat\{([xyz])\}\$", tex)
    amp = int(m.group(2)) * (-1 if m.group(1) else 1)
    ct = cu = 0
    ph = 0.0
    uax = None
    for tok in re.findall(r"[+-]?[^+-]+", m.group(4).replace(" ", "")):
        sg = -1 if tok.startswith("-") else 1
        body = tok.lstrip("+-")
        if body == r"\omegat":
            ct += sg
        elif body.startswith(r"\beta"):
            cu += sg
            uax = body[-1]
        else:
            ph += sg * pifrac(body) * math.pi
    return amp, m.group(3), ct, cu, uax, ph, m.group(5)


def parse_coef(s):
    s = s.strip()
    m = re.fullmatch(r"(-?)(\d+)(j?)", s)
    if m:
        return int(m.group(2)) * (-1 if m.group(1) else 1) * (1j if m.group(3) else 1)
    m = re.fullmatch(r"(\d+)\\,e\^\{(-?)j(.*)\}", s)
    return int(m.group(1)) * cmath.exp(1j * (-1 if m.group(2) else 1) * pifrac(m.group(3)) * math.pi)


def parse_option(tex):
    m = re.fullmatch(r"\$(.*)\\,e\^\{([+-])j\\beta ([xyz])\}\\,\\hat\{([xyz])\}\$", tex)
    return parse_coef(m.group(1)), (-1 if m.group(2) == "-" else 1), m.group(3), m.group(4)


def curl_c(F, r, h):
    J = np.zeros((3, 3), complex)
    for j in range(3):
        dr = np.zeros(3); dr[j] = h
        J[:, j] = (F(r + dr) - F(r - dr)) / (2 * h)
    return np.array([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]])


def unit_from_choice(text):
    m = re.fullmatch(r"\$\((-?)(\d+)\\,\\hat\{([xyz])\} ([+-]) (\d+)\\,\\hat\{([xyz])\}\)/(\d+)\$", text)
    if m:
        v = np.zeros(3)
        v[IDX[m.group(3)]] = int(m.group(2)) * (-1 if m.group(1) else 1)
        v[IDX[m.group(6)]] = int(m.group(5)) * (-1 if m.group(4) == "-" else 1)
        return v / int(m.group(7))
    m = re.match(r"\$([+-])\\hat\{([xyz])\}\$", text)
    return (1 if m.group(1) == "+" else -1) * E3[IDX[m.group(2)]]


def phys(d, check):
    p, c = d["params"], d["correct_answers"]

    # ================================================================ (a)
    om, be = 2 * math.pi * 1.0, 1.7
    us = np.linspace(-2.3, 3.1, 37)
    ts = np.linspace(0, 1.0, 41)
    U, T = np.meshgrid(us, ts)
    fields = []
    for it in p["conv_items"]:
        amp, fn, ct, cu, uax, ph, eax = parse_field(it["text"])
        check(uax == p["u"] and eax == p["e"], "phasor (a): axis of a field")
        check(abs(ct) == 1 and abs(cu) == 1, "phasor (a): argument coefficients")
        arg = ct * om * T + cu * be * U + ph
        fields.append((it["key"], amp * (np.sin(arg) if fn == "sin" else np.cos(arg))))
    opts = {}
    for o in p["conv_opts"]:
        cf, sexp, uax, eax = parse_option(o["text"])
        check(uax == p["u"] and eax == p["e"], "phasor (a): axis of an option")
        check(abs(abs(cf) - p["A0"]) < 1e-9, "phasor (a): option magnitude")
        opts[o["name"]] = np.real(cf * np.exp(1j * sexp * be * U) * np.exp(1j * om * T))
    check(len(opts) == 8 and len(p["conv_items"]) == 5, "phasor (a): counts")
    names = list(opts)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            check(np.max(np.abs(opts[names[i]] - opts[names[j]])) > 1e-3 * p["A0"], "phasor (a): two options are the same field")
    matched = set()
    for key, f in fields:
        hits = [n for n in names if np.max(np.abs(opts[n] - f)) < 1e-9 * p["A0"]]
        check(hits == [key], f"phasor (a): field matches options {hits}, key {key}")
        matched.update(hits)
    check(len(matched) == 5, "phasor (a): keys not distinct")

    # ================================================================ (b)
    m = re.fullmatch(r"\\tilde\{\\mathbf\{E\}\} = \\big\((.*)\\big\)\\,e\^\{([+-])j\\,(\d*)\\pi ([xyz])\}\\ \\ \\mathrm\{V/m\}", p["Eb_tex"])
    check(m is not None, "phasor (b): cannot parse E~")
    E0 = np.zeros(3, complex)
    terms = re.findall(r"([+-]?)\s*(\d+)(j?)\\,\\hat\{([xyz])\}", m.group(1))
    check(len(terms) == 2, "phasor (b): two components")
    for sg, mag, jj, ax in terms:
        E0[IDX[ax]] += (-1 if sg == "-" else 1) * int(mag) * (1j if jj else 1)
    sexp = -1 if m.group(2) == "-" else 1
    beta = (int(m.group(3)) if m.group(3) else 1) * math.pi
    ub = E3[IDX[m.group(4)]]
    eps = float(p["epsb_str"]) * EPS0
    w = 2 * math.pi * float(p["fb_str"]) * 1e6
    check(abs(beta - w * math.sqrt(MU0 * eps)) < 1e-9 * beta, "phasor (b): beta inconsistent with f and eps_r")
    check(abs(np.dot(E0, ub)) == 0, "phasor (b): E~ not transverse")
    check(sum(abs(E0.real) > 0) == 1 and sum(abs(E0.imag) > 0) == 1, "phasor (b): one real and one imaginary component")

    def Et(r):
        return E0 * np.exp(1j * sexp * beta * np.dot(r, ub))
    lam = 2 * math.pi / beta
    h = lam * 1e-5

    def Ht(r):
        return -curl_c(Et, r, h) / (1j * w * MU0)
    H0 = Ht(np.zeros(3))
    ax1, ax2 = p["ax1"], p["ax2"]
    for key, val in (("H1re", H0[IDX[ax1]].real), ("H1im", H0[IDX[ax1]].imag), ("H2re", H0[IDX[ax2]].real), ("H2im", H0[IDX[ax2]].imag)):
        check(abs(val * 1e3 - c[key]) < 1e-6 * max(1.0, np.max(np.abs(H0)) * 1e3), f"phasor (b): {key} {val * 1e3} vs {c[key]}")
    check(abs(np.dot(H0, ub)) < 1e-6 * np.max(np.abs(H0)), "phasor (b): H~ has a component along the travel")
    r1 = np.array([0.21, -0.37, 0.13]) * lam
    amp_res = curl_c(Ht, r1, lam * 1e-3) - 1j * w * eps * Et(r1)
    check(np.max(np.abs(amp_res)) < 1e-4 * w * eps * np.max(np.abs(E0)), "phasor (b): phasor Ampere-Maxwell fails")
    tt = np.linspace(0, 2 * math.pi / w, 4001)
    Ereal = np.real(E0[None, :] * np.exp(1j * w * tt)[:, None])
    Hreal = np.real(H0[None, :] * np.exp(1j * w * tt)[:, None])
    Savg = np.trapezoid(np.cross(Ereal, Hreal), tt, axis=0) * w / (2 * math.pi)
    check(abs(np.dot(Savg, ub) * 1e3 - c["Sb"]) < 1e-6 * abs(c["Sb"]), f"phasor (b): <S_u> {np.dot(Savg, ub) * 1e3} vs {c['Sb']}")
    check(np.dot(Savg, ub) * sexp < 0, "phasor (b): power not along the travel read from the exponent")

    # ================================================================ (c)
    m = re.fullmatch(r"\\tilde\{\\mathbf\{E\}\} = (\d+)\\,\\hat\{([xyz])\}\\,e\^\{j\((.*)\)\}\\ \\ \\mathrm\{V/m\}", p["tmpl_tex"])
    check(m is not None, "phasor (c): cannot parse the template")
    Ec, pol = int(m.group(1)), E3[IDX[m.group(2)]]
    kv = np.zeros(3)
    for sg, cf, ax in re.findall(r"([+-]?)\s*(\d*\.?\d*)\\pi\\,([xyz])", m.group(3)):
        kv[IDX[ax]] += (-1 if sg == "-" else 1) * (float(cf) if cf else 1.0) * math.pi
    kmag = np.linalg.norm(kv)
    wc = C * kmag

    def ec(r, t):
        return Ec * math.cos(wc * t + np.dot(kv, r))
    rng = np.random.default_rng(3)
    pts = [(rng.uniform(-1, 1, 3), rng.uniform(0, 1e-8), rng.uniform(1e-10, 3e-9)) for _ in range(6)]
    good = []
    for ch in p["dir_list"]:
        uv = unit_from_choice(ch["text"])
        check(abs(np.linalg.norm(uv) - 1) < 1e-12, "phasor (c): choice not a unit vector")
        rides = all(abs(ec(r + uv * C * dt, t + dt) - ec(r, t)) < 1e-7 * Ec for r, t, dt in pts)
        good.append(rides)
        check((ch["correct"] == "true") == rides, f"phasor (c): choice {ch['text']} key vs riding test")
    check(sum(good) == 1, "phasor (c): not exactly one direction of travel")
    uc = unit_from_choice(next(ch["text"] for ch in p["dir_list"] if ch["correct"] == "true"))
    check(abs(np.dot(uc, pol)) < 1e-12, "phasor (c): travel not perpendicular to the field")
    xs = np.linspace(0, 6.0, 600001)
    vals = Ec * np.cos(np.outer(xs, uc) @ kv)
    zc = xs[:-1][np.sign(vals[:-1]) != np.sign(vals[1:])]
    lam_num = 2 * float(np.mean(np.diff(zc)))
    check(abs(lam_num - c["lam_c"]) < 1e-3 * c["lam_c"], f"phasor (c): lambda {lam_num} vs {c['lam_c']}")
    check(abs(C / lam_num / 1e9 - c["f_c"]) < 1e-3 * c["f_c"], "phasor (c): frequency (GHz)")

    # ================================================================ (d)
    fp = float(p["fp_str"]) * 1e6
    wp = 2 * math.pi * fp
    N = wp ** 2 * ME * EPS0 / QE ** 2

    def beta_sq(f):
        wv = 2 * math.pi * f
        T0 = 2 * math.pi / wv
        tq = np.linspace(0, 3 * T0, 30001)
        Eq = np.cos(wv * tq)                                    # E at z = 0
        acc = -QE * Eq / ME
        v = np.concatenate([[0.0], np.cumsum(0.5 * (acc[1:] + acc[:-1]) * np.diff(tq))])
        v = v - np.trapezoid(v[10000:], tq[10000:]) / (2 * T0)  # steady state: no drift
        J = -N * QE * v
        dJ = np.gradient(J, tq)
        d2E = np.gradient(np.gradient(Eq, tq), tq)
        sl = slice(10000, 29000)
        # d2E/dz2 = -beta^2 E  =>  -beta^2 E = MU0 dJ/dt + MU0 EPS0 d2E/dt2
        rhs = MU0 * dJ[sl] + MU0 * EPS0 * d2E[sl]
        return -float(np.dot(rhs, Eq[sl]) / np.dot(Eq[sl], Eq[sl]))
    f1 = float(p["f1_str"]) * 1e6
    b2 = beta_sq(f1)
    check(b2 > 0, "phasor (d): first wave does not propagate")
    bnum = math.sqrt(b2)
    check(abs(bnum - c["beta_p"]) < 2e-3 * c["beta_p"], f"phasor (d): beta {bnum} vs {c['beta_p']}")
    vp = 2 * math.pi * f1 / bnum
    check(abs(vp / 1e8 - c["vp8"]) < 2e-3 * c["vp8"] and vp > C, f"phasor (d): v_p {vp / 1e8} vs {c['vp8']}")
    f2 = float(p["f2_str"]) * 1e6
    b22 = beta_sq(f2)
    check(abs(b22) > 1e-3 * (2 * math.pi * f2 / C) ** 2, "phasor (d): second frequency too close to f_p")
    want = "none" if b22 < 0 else "fast"
    if b22 > 0:
        check(2 * math.pi * f2 / math.sqrt(b22) > C, "phasor (d): v_p at f2 not above c")
    chs = {x["id"]: x for x in p["plasma_list"]}
    check(sum(x["correct"] == "true" for x in p["plasma_list"]) == 1 and chs[want]["correct"] == "true", "phasor (d): MC key")
    check(p["below"] == (b22 < 0), "phasor (d): below flag")
