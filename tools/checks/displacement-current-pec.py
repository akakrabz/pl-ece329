"""Independent check for displacement-current-pec (Lecture 16).

Routes:
  * dD_z/dt: D_z from the source data (bottom-plate charge Q_b(t) via Gauss, or the plate potentials via a
    numerical gradient of the linear V(z)), differentiated in time by central differences;
  * H_phi(r1), H_phi(r2): Biot-Savart summed over the whole closed current system -- semi-infinite leads, radial
    surface currents on both plates, and the displacement current as vertical filaments -- built as a
    Kirchhoff-consistent network (no Ampere's law); plus a numerical-flux / circulation evaluation with np.cross;
  * the two-surface MC: conduction current through S1 and the D-flux rate through S2 (Gauss on the plate);
  * the perfect conductor: outward normal from a finite-difference gradient of the region function,
    J_s = np.cross(n, H1); checkbox truths recomputed (dot/cross products, finite-difference divergence and curl,
    Biot-Savart below the plate).
"""
import math

import numpy as np

EPS0 = 8.854e-12


def seg_H(P, A, B, I):
    """H at P from straight segments A[i] -> B[i] carrying I[i] (closed form of the Biot-Savart integral)."""
    P = np.asarray(P, float)
    u = B - A
    L = np.linalg.norm(u, axis=1)
    u = u / L[:, None]
    a1, a2 = P - A, P - B
    c1 = np.einsum("ij,ij->i", a1, u)
    c2 = np.einsum("ij,ij->i", a2, u)
    rho = a1 - c1[:, None] * u
    d = np.linalg.norm(rho, axis=1)
    fac = c1 / np.linalg.norm(a1, axis=1) - c2 / np.linalg.norm(a2, axis=1)
    return np.sum((I * fac / (4 * np.pi * d))[:, None] * (np.cross(u, rho) / d[:, None]), axis=0)


def network(i_lead, a, d, n_r=60, n_phi=120, far=1e4):
    """Closed current system of the charging capacitor, discretized so that Kirchhoff's law holds at every node."""
    dr, dphi = a / n_r, 2 * np.pi / n_phi
    rj = (np.arange(n_r) + 0.5) * dr
    Jd = i_lead / (np.pi * a * a)
    dI = Jd * rj * dr * dphi                              # displacement current of each cell (one rail)
    I_rad = np.cumsum(dI[::-1])[::-1]                     # current in the radial segment arriving at node j
    A, B, I = [np.array([[0, 0, -far], [0, 0, d]])], [np.array([[0, 0, 0.0], [0, 0, far]])], [np.array([i_lead,
                                                                                                         i_lead])]
    for k in range(n_phi):
        ph = (k + 0.5) * dphi
        c, s = math.cos(ph), math.sin(ph)
        r_from = np.concatenate([[0.0], rj[:-1]])
        for z, outward in ((0.0, True), (d, False)):
            p0 = np.stack([r_from * c, r_from * s, np.full(n_r, z)], axis=1)
            p1 = np.stack([rj * c, rj * s, np.full(n_r, z)], axis=1)
            A.append(p0 if outward else p1)
            B.append(p1 if outward else p0)
            I.append(I_rad)
        A.append(np.stack([rj * c, rj * s, np.zeros(n_r)], axis=1))
        B.append(np.stack([rj * c, rj * s, np.full(n_r, d)], axis=1))
        I.append(dI)
    return np.concatenate(A), np.concatenate(B), np.concatenate(I)


def H_phi(net, r, z):
    H = seg_H([r, 0.0, z], *net)
    return float(H[1]), H                                 # at phi = 0, phi_hat = +y


def Dz_of_t(p, t, er=None):
    """D_z in the gap from the source data: Gauss on the bottom plate (constant current) or the plate potentials."""
    er = float(p["er"]) if er is None else er
    a, d = float(p["a_cm"]) * 1e-2, float(p["d_mm"]) * 1e-3
    if p["mode"] == "const":
        Qb = 1e-9 + p["s_I"] * p["I_mA"] * 1e-3 * t       # bottom-plate charge (any initial value)
        return Qb / (math.pi * a * a)                     # rho_s = n . D with n = +z out of the bottom plate
    w = 2 * math.pi * p["f_kHz"] * 1e3
    v_bt = p["s_v"] * p["V0"] * math.cos(w * t)           # V_bottom - V_top
    zs = np.linspace(0, d, 11)
    V = v_bt * (1 - zs / d)                               # Laplace in 1-D between the plates (V_top = 0)
    Ez = -np.gradient(V, zs)
    return EPS0 * er * float(np.mean(Ez))


def t_eval(p):
    return 0.0 if p["mode"] == "const" else p["t1_us"] * 1e-6


def Jd_numeric(p, er=None):
    t = t_eval(p)
    h = 1e-9 if p["mode"] == "const" else 1e-6 / (2 * math.pi * p["f_kHz"] * 1e3)
    return (Dz_of_t(p, t + h, er) - Dz_of_t(p, t - h, er)) / (2 * h)


def phys(d, check):
    p, c = d["params"], d["correct_answers"]
    a, dd = float(p["a_cm"]) * 1e-2, float(p["d_mm"]) * 1e-3
    r1, r2 = float(p["r1_cm"]) * 1e-2, float(p["r2_cm"]) * 1e-2

    # (a)
    Jd = Jd_numeric(p)
    check(abs(Jd - c["Jd"]) <= 1e-6 * abs(Jd), f"capacitor: dD/dt {Jd} vs {c['Jd']}")
    i_lead = Jd * math.pi * a * a

    # (b) numerical flux through the disk + circulation with np.cross orientation
    def H_by_flux(r, n=400):
        reff = min(r, a)                                   # displacement current only where the plates are
        rr = (np.arange(n) + 0.5) * reff / n
        flux = float(np.sum(Jd * 2 * np.pi * rr) * reff / n)                           # disk normal +z
        th = 2 * np.pi * (np.arange(720) + 0.5) / 720
        pts = np.stack([r * np.cos(th), r * np.sin(th), np.zeros_like(th)], axis=1)
        tang = np.cross(np.array([0, 0, 1.0]), pts) / r                                  # counterclockwise from +z
        dl = tang * (2 * np.pi * r / 720)
        normal = np.cross(pts[0], pts[1])                                              # right-hand rule normal
        sgn = np.sign(normal[2])
        return sgn * flux / float(np.sum(np.einsum("ij,ij->i", tang, dl)))
    check(abs(H_by_flux(r1) * 1e3 - c["H1"]) <= 1e-4 * abs(c["H1"]), f"capacitor: H(r1) flux route vs {c['H1']}")
    check(abs(H_by_flux(r2) * 1e3 - c["H2"]) <= 1e-4 * abs(c["H2"]), f"capacitor: H(r2) flux route vs {c['H2']}")

    net = network(i_lead, a, dd)
    H2bs, H2vec = H_phi(net, r2, dd / 2)
    check(abs(H2bs * 1e3 - c["H2"]) <= 2e-4 * abs(c["H2"]), f"capacitor: Biot-Savart H(r2) {H2bs * 1e3} vs {c['H2']}")
    check(abs(H2vec[0]) + abs(H2vec[2]) < 1e-6 * abs(H2bs), "capacitor: H(r2) not purely azimuthal")
    # Inside the gap the field point is only d/2 from the plates, closer than the spacing of the discretized radial
    # plate currents, so the real gap would need an enormous grid. For a given i, H_phi at r < a in the mid-plane
    # does not depend on d (Ampere-Maxwell), so evaluate the same network with a thick gap d' = a, at two
    # resolutions, and Richardson-extrapolate the second-order discretization error away.
    Hc = H_phi(network(i_lead, a, a, n_r=60, n_phi=120), r1, a / 2)[0]
    Hf = H_phi(network(i_lead, a, a, n_r=120, n_phi=240), r1, a / 2)[0]
    H1bs = (4 * Hf - Hc) / 3
    check(abs(H1bs * 1e3 - c["H1"]) <= 1e-6 * abs(c["H1"]), f"capacitor: Biot-Savart H(r1) {H1bs * 1e3} vs {c['H1']}")

    # (c) two surfaces: conduction through S1, D-flux rate through S2 (Gauss on the enclosed bottom plate)
    I_S1 = i_lead                                          # the lead pierces the flat disk
    t, h = t_eval(p), 1e-9
    Id_S2 = (Dz_of_t(p, t + h) - Dz_of_t(p, t - h)) / (2 * h) * math.pi * a * a if p["mode"] == "const" else \
        Jd * math.pi * a * a
    check(abs(I_S1 - Id_S2) <= 1e-6 * abs(I_S1), "capacitor MC: the two surfaces disagree numerically")
    ch = p["paradox"]
    check(sum(x["correct"] == "true" for x in ch) == 1, "capacitor MC: exactly one correct")
    check([x["key"] for x in ch if x["correct"] == "true"][0] == "both", "capacitor MC: correct key")
    er = float(p["er"])
    check(("vacuum" in [x["key"] for x in ch]) == (er > 1), "capacitor MC: vacuum distractor only with a dielectric")

    # (d) perfect conductor
    axis, side = p["pec_axis"], p["pec_side"]
    k = "xyz".index(axis)
    f = lambda x: side * x[k]  # noqa: E731   conductor where f > 0
    x0, hh = np.array([0.0, 0.0, 0.0]), 1e-6
    grad = np.array([(f(x0 + e) - f(x0 - e)) / (2 * hh) for e in np.eye(3) * hh])
    n = -grad / np.linalg.norm(grad)                       # out of the conductor, into free space
    check(np.allclose(n, p["n"]), "PEC: normal")
    check(p["pec_region"].startswith(axis) and (r"\gt" in p["pec_region"]) == (side > 0), "PEC: region text")
    H1 = np.array(p["Hpec"], float)
    check(abs(np.dot(n, H1)) < 1e-12 and np.linalg.norm(H1) > 0, "PEC: H1 must be tangential and nonzero")
    Js = np.cross(n, H1)
    for comp, key in zip(Js, ("Jsx", "Jsy", "Jsz")):
        check(abs(comp - c[key]) <= 1e-12, f"PEC: {key} {c[key]} vs {comp}")
    check(np.allclose(np.cross(Js, n), H1), "PEC: J_s x n should give back H1")

    # (e) checkbox truths
    def truth(fct):
        key, ex = fct["key"], fct.get("extra")
        if key == "vecE":
            v = np.array(ex["vec"], float)
            return np.linalg.norm(np.cross(n, v)) < 1e-12 and np.linalg.norm(v) > 0
        if key == "vecH":
            v = np.array(ex["vec"], float)
            return abs(np.dot(n, v)) < 1e-12 and np.linalg.norm(v) > 0
        if key == "rho_sign":
            return EPS0 * np.dot(n, np.array(ex["vec"], float)) * 1e3 > 0
        if key == "js_perp":
            return abs(np.dot(Js, n)) < 1e-12 and abs(np.dot(Js, H1)) < 1e-12
        if key == "no_dBdt":       # E = 0 everywhere inside: curl E = 0, so dB/dt = -curl E = 0
            E0 = lambda x: np.zeros(3)  # noqa: E731
            J = np.array([(E0(x0 + e) - E0(x0 - e)) / 2e-3 for e in np.eye(3) * 1e-3]).T
            return np.allclose([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]], 0)
        if key == "Bn":
            return True            # B_n continuous (pillbox) and zero time-varying B inside
        if key == "Et_dynamic":
            return False           # E_t continuous for any time variation (thin-loop flux -> 0)
        if key == "continuity":    # random smooth H(x,t), D(x,t): J = curl H - dD/dt, rho = div D
            def Hf(x, tt):
                return np.array([x[1] * x[2] * tt, x[0] ** 2 * tt ** 2, np.sin(x[0] + tt) * x[1]])

            def Df(x, tt):
                return np.array([x[0] * x[1] * tt ** 2, np.cos(x[2]) * tt, x[0] * x[1] * x[2] * tt])

            def jac(F, x, tt, s=1e-3):
                return np.array([(F(x + e, tt) - F(x - e, tt)) / (2 * s) for e in np.eye(3) * s]).T

            def Jf(x, tt):
                Jm = jac(Hf, x, tt)
                curl = np.array([Jm[2, 1] - Jm[1, 2], Jm[0, 2] - Jm[2, 0], Jm[1, 0] - Jm[0, 1]])
                return curl - (Df(x, tt + 1e-4) - Df(x, tt - 1e-4)) / 2e-4

            def rho(x, tt):
                return float(np.trace(jac(Df, x, tt)))
            xx, tt = np.array([0.3, -0.4, 0.7]), 0.8
            divJ = float(np.trace(jac(Jf, xx, tt, 1e-2)))
            drho = (rho(xx, tt + 1e-3) - rho(xx, tt - 1e-3)) / 2e-3
            return abs(divJ + drho) < 1e-4
        if key == "charges_cross":
            return False           # sigma = 0 in a perfect dielectric: no conduction current in the gap
        if key == "divB_faraday":  # B = B0 (x, y, z), static, with E = 0 satisfies Faraday but div B = 3 B0
            return False
        if key == "H_axis":
            return abs(H_by_flux(1e-6 * a)) > abs(H_by_flux(0.999 * a))
        if key == "er_effect":
            return abs(Jd_numeric(p, er=2 * er)) > 1.5 * abs(Jd_numeric(p, er=er))
        if key == "lead_equal":
            below, _ = H_phi(net, r2, -0.3 * a)
            return abs(below - H2bs) <= 2e-4 * abs(H2bs)
        raise KeyError(key)

    facts = p["facts"]
    nt = sum(fc["correct"] == "true" for fc in facts)
    check(len(facts) == 5 and 1 <= nt <= 4, f"capacitor checkbox: {len(facts)} statements, {nt} true")
    for fc in facts:
        tv = bool(truth(fc))
        check(tv == (fc["correct"] == "true"), f"capacitor checkbox: {fc['key']} marked {fc['correct']}, recomputed {tv}")
