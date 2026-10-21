"""Independent check for waves/poynting-power-ledger (Lecture 20).

Wire: the cross-section is gridded (fine square cells), the uniform J is fixed by sum(J dA) = I, H_phi(rho) comes from
Ampere's law with the *numerically* enclosed current, E = J/sigma; S = np.cross(E, H) at the point P and on a ring of
points around the surface; the inflow is the ring sum of -S.n dA and is compared with sum(sigma E^2 dV) and with I^2 R.
Wave: the field is parsed from the displayed LaTeX; beta/omega is checked against the stated eps_r. The partner field
is NOT built from u x E/eta: it is obtained by integrating Faraday's (or Ampere-Maxwell's) law in time,
dH/dt = -(1/mu) curl E (central-difference curl), over two periods at the probe point, mean removed. <S> is the
trapezoid average of np.cross(E, H) over a period; the window normal is built at angle theta to <S>.
Checkbox: every statement re-decided by a computation (finite-difference Poynting residuals, quadrature averages,
np.cross for directions, a radiating sheet's pillbox balance, a time-varying-epsilon counterexample).
"""
import math
import re

import numpy as np

C = 3e8
MU0 = 4e-7 * math.pi
EPS0 = 1.0 / (MU0 * C * C)
IDX = {"x": 0, "y": 1, "z": 2}
E3 = np.eye(3)


def unit_of(tex):
    m = re.fullmatch(r"\$([+-])\\hat\{([xyz])\}\$", tex.strip())
    return None if m is None else (1 if m.group(1) == "+" else -1) * E3[IDX[m.group(2)]]


def curl_fd(F, r, h):
    J = np.zeros((3, 3))
    for j in range(3):
        dr = np.zeros(3); dr[j] = h
        J[:, j] = (F(r + dr) - F(r - dr)) / (2 * h)
    return np.array([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]])


def div_fd(F, r, h):
    out = 0.0
    for j in range(3):
        dr = np.zeros(3); dr[j] = h
        out += (F(r + dr)[j] - F(r - dr)[j]) / (2 * h)
    return out


def kcoef(s):
    return 1.0 if s == "" else float(s)


def phys(d, check):
    p, c = d["params"], d["correct_answers"]

    # =========================================================== wire
    sigma = p["sigma"]
    sig_disp = float(re.sub(r"\\times10\^\{(-?\d+)\}", r"e\1", p["sigma_str"]))
    check(abs(sig_disp - sigma) < 1e-9 * sigma, "ledger: displayed sigma differs from the value used")
    a = float(p["a_str"]) * 1e-3
    ell = float(p["ell_str"])
    I = float(p["I_str"])
    wv = E3[IDX[p["w"]]]
    eI = p["sI"] * wv                                   # current direction
    check(np.allclose(unit_of("$" + p["Idir"] + "$"), eI), "ledger: displayed current direction")
    # grid the cross-section in local coordinates (q1, q2) perpendicular to the wire
    q1 = E3[(IDX[p["w"]] + 1) % 3]
    q2 = np.cross(wv, q1)
    Ng = 801
    xs = (np.arange(Ng) + 0.5) / Ng * 2 * a - a
    X, Y = np.meshgrid(xs, xs)
    inside = X ** 2 + Y ** 2 < a * a
    dA = (2 * a / Ng) ** 2
    J0 = I / (inside.sum() * dA)                         # uniform J with sum(J dA) = I
    Ein = J0 / sigma * eI                                # Ohm's law, axial
    rcells = np.sort(np.sqrt(X[inside] ** 2 + Y[inside] ** 2))

    def H_at(r):                                         # Ampere with the numerically enclosed current
        rp = r - np.dot(r, wv) * wv
        rho = np.linalg.norm(rp)
        Ienc = J0 * dA * np.searchsorted(rcells, rho) if rho < a else I
        phi = np.cross(eI, rp / rho)
        return Ienc / (2 * math.pi * rho) * phi

    def E_at(r):
        return Ein

    # (b): ring of points just inside the surface
    rr = a * (1 - 1e-9)
    phis = (np.arange(720) + 0.5) / 720 * 2 * math.pi
    inflow = 0.0
    Smag = []
    for ph in phis:
        rhat = math.cos(ph) * q1 + math.sin(ph) * q2
        S = np.cross(E_at(rr * rhat), H_at(rr * rhat))
        Smag.append(np.linalg.norm(S))
        inflow += -np.dot(S, rhat) * rr * (2 * math.pi / 720) * ell
    Smag = np.array(Smag)
    check(np.ptp(Smag) < 1e-6 * Smag.mean(), "ledger: |S| not uniform around the wire")
    check(abs(Smag.mean() / 1e3 - c["S_surf"]) < 3e-3 * c["S_surf"], f"ledger: |S| surface {Smag.mean()/1e3} vs {c['S_surf']}")
    joule = sigma * np.dot(Ein, Ein) * inside.sum() * dA * ell
    R = ell / (sigma * inside.sum() * dA)
    check(abs(inflow - joule) < 3e-3 * joule, f"ledger: inflow {inflow} vs Joule {joule}")
    check(abs(inflow - c["P_in"]) < 3e-3 * c["P_in"], f"ledger: P_in {inflow} vs {c['P_in']}")
    check(abs(I * I * R - c["P_in"]) < 3e-3 * c["P_in"], "ledger: P_in vs I^2 R")
    check(I / (math.pi * a * a) <= 2.6e7, "ledger: unrealistic current density")

    # (c): direction at P
    m = re.fullmatch(r"\((.*)\)", p["P_tex"])
    comps = [s.strip() for s in m.group(1).split(",")]
    Pv = np.array([0.0 if s == "0" else (-1.0 if s == "-a" else 1.0) for s in comps]) * a * (1 - 1e-9)
    check(abs(np.linalg.norm(Pv) - a) < 1e-9 and abs(np.dot(Pv, wv)) < 1e-15, "ledger: P not on the surface")
    SP = np.cross(E_at(Pv), H_at(Pv))
    sdir = SP / np.linalg.norm(SP)
    ch = {x["id"]: x for x in p["sdir_list"]}
    check(sum(x["correct"] == "true" for x in p["sdir_list"]) == 1 and ch["in"]["correct"] == "true", "ledger: sdir key")
    check(np.allclose(unit_of(ch["in"]["text"]), sdir), "ledger: S direction at P")
    check(np.dot(sdir, Pv) < 0, "ledger: S at P not inward")
    check(np.allclose(unit_of(ch["out"]["text"]), -sdir), "ledger: 'out' distractor")
    check(np.allclose(unit_of(ch["along"]["text"]), eI) and np.allclose(unit_of(ch["against"]["text"]), -eI), "ledger: axial distractors")
    HP = H_at(Pv)
    check(np.allclose(unit_of(ch["alongH"]["text"]), HP / np.linalg.norm(HP)), "ledger: 'alongH' distractor")
    check(len({x["text"] for x in p["sdir_list"]}) == 5, "ledger: sdir choices not distinct")

    # =========================================================== wave
    m = re.fullmatch(r"\\mathbf\{(E|H)\} = (-?)([\d.]+)\\cos\\big\((.*)\\big\)\\,\\hat\{([xyz])\}\\ \\ \\mathrm\{(V/m|mA/m)\}",
                     p["field_tex"])
    check(m is not None, "ledger: cannot parse field_tex")
    kind = m.group(1)
    amp = float(m.group(3)) * (-1 if m.group(2) == "-" else 1) * (1e-3 if m.group(6) == "mA/m" else 1.0)
    check((kind == "E") == (m.group(6) == "V/m"), "ledger: unit of the given field")
    fhat = E3[IDX[m.group(5)]]
    ma = re.fullmatch(r"(\d*\.?\d*)\\pi\\times10\^\{8\}\\,t ([+-]) (\d*\.?\d*)\\pi\\,([xyz])", m.group(4))
    check(ma is not None, "ledger: cannot parse the argument")
    om = kcoef(ma.group(1)) * math.pi * 1e8
    sgn = -1 if ma.group(2) == "-" else 1
    beta = kcoef(ma.group(3)) * math.pi
    uax = E3[IDX[ma.group(4)]]
    eps_r = float(p["eps_str"])
    eps, mu = EPS0 * eps_r, MU0
    check(abs(beta - om * math.sqrt(mu * eps)) < 1e-9 * beta, "ledger: displayed beta/omega inconsistent with eps_r")
    check(abs(np.dot(fhat, uax)) == 0, "ledger: field not transverse")
    T = 2 * math.pi / om
    lam = 2 * math.pi / beta

    def F_given(r, t):
        return amp * math.cos(om * t + sgn * beta * np.dot(r, uax)) * fhat

    # partner by time-integrating the curl law at a probe point
    r0 = np.array([0.137, -0.291, 0.443]) * lam
    hx = lam * 1e-5
    tt = np.linspace(0, 2 * T, 4001)
    curl = np.array([curl_fd(lambda rr: F_given(rr, t), r0, hx) for t in tt])
    factor = (-1 / mu) if kind == "E" else (1 / eps)
    integ = np.concatenate([[np.zeros(3)], np.cumsum(0.5 * (curl[1:] + curl[:-1]) * np.diff(tt)[:, None], axis=0)]) * factor
    partner = integ - np.trapezoid(integ[2000:], tt[2000:], axis=0) / T     # steady state: no DC part
    given = np.array([F_given(r0, t) for t in tt])
    Ev, Hv = (given, partner) if kind == "E" else (partner, given)
    Sinst = np.cross(Ev, Hv)
    seg = slice(2000, 4001)
    Savg_vec = np.trapezoid(Sinst[seg], tt[seg], axis=0) / T
    Smax = np.max(np.linalg.norm(Sinst[seg], axis=1))
    check(abs(np.dot(Savg_vec, uax) - c["S_avg"]) < 3e-3 * abs(c["S_avg"]), f"ledger: <S_u> {np.dot(Savg_vec, uax)} vs {c['S_avg']}")
    check(np.linalg.norm(Savg_vec - np.dot(Savg_vec, uax) * uax) < 1e-3 * np.linalg.norm(Savg_vec), "ledger: <S> not along u")
    check(abs(Smax - 2 * abs(c["S_avg"])) < 5e-3 * Smax, "ledger: peak S is not twice the average")
    trav = np.sign(np.dot(Savg_vec, uax)) * uax
    check(np.sign(np.dot(Savg_vec, uax)) == -sgn, "ledger: travel direction vs argument sign")
    th = math.radians(p["theta"])
    tperp = np.cross(trav, fhat)
    nwin = math.cos(th) * trav + math.sin(th) * tperp
    Pw = abs(np.dot(Savg_vec, nwin)) * p["A_cm2"] * 1e-4 * 1e3
    check(abs(Pw - c["P_win"]) < 3e-3 * c["P_win"], f"ledger: window power {Pw} vs {c['P_win']}")

    # closed-form partner (for the statement experiments), validated against the time-integrated one
    eta = math.sqrt(mu / eps)
    if kind == "E":
        def Ef(r, t): return F_given(r, t)
        def Hf(r, t): return np.cross(trav, F_given(r, t)) / eta
    else:
        def Hf(r, t): return F_given(r, t)
        def Ef(r, t): return eta * np.cross(F_given(r, t), trav)
    chk = np.array([Hf(r0, t) if kind == "E" else Ef(r0, t) for t in tt[2000::10]])
    check(np.max(np.abs(chk - partner[2000::10])) < 2e-3 * np.max(np.abs(chk)), "ledger: closed-form partner disagrees with Faraday/Ampere integration")

    def S_at(r, t): return np.cross(Ef(r, t), Hf(r, t))
    def w_at(r, t):
        e, h = Ef(r, t), Hf(r, t)
        return 0.5 * eps * np.dot(e, e) + 0.5 * mu * np.dot(h, h)

    rng = np.random.default_rng(7)
    ht = T * 1e-5
    samples = [(rng.uniform(-1, 1, 3) * lam, rng.uniform(0, T)) for _ in range(12)]
    div_ok, resid_ok = True, True
    for r, t in samples:
        dS = div_fd(lambda rr: S_at(rr, t), r, hx)
        dw = (w_at(r, t + ht) - w_at(r, t - ht)) / (2 * ht)
        scale = max(abs(dS), abs(dw), 1e-3 * 2 * abs(c["S_avg"]) * beta)
        resid_ok &= abs(dS + dw) < 1e-5 * scale
        if dS > 1e-3 * scale:
            div_ok &= dw < 0
    # wire interior: steady, so div S + sigma E^2 = 0 (H inside from Ampere for uniform J: J0 rho/2)
    def Hw(r):
        rp = r - np.dot(r, wv) * wv
        return J0 / 2 * np.cross(eI, rp)
    wire_ok = True
    for _ in range(5):
        rr0 = (rng.uniform(-0.5, 0.5) * q1 + rng.uniform(-0.5, 0.5) * q2) * a + rng.uniform(0, ell) * wv
        dSw = div_fd(lambda rr: np.cross(Ein, Hw(rr)), rr0, a * 1e-4)
        wire_ok &= abs(dSw + sigma * np.dot(Ein, Ein)) < 1e-6 * sigma * np.dot(Ein, Ein)
    # vectorized time series (helpers cross-checked against the scalar fields above)
    def g_t(r, ts, back=False):
        ph = om * ts + (-sgn if back else sgn) * beta * np.dot(r, uax)
        return amp * np.cos(ph)[:, None] * fhat[None, :]

    def EH_t(r, ts, back=False):
        uu = -trav if back else trav
        g = g_t(r, ts, back)
        if kind == "E":
            return g, np.cross(np.broadcast_to(uu, g.shape), g) / eta
        return eta * np.cross(g, np.broadcast_to(uu, g.shape)), g

    tser = np.linspace(0, T, 2001)
    E_s, H_s = EH_t(r0, tser)
    check(np.allclose(E_s[37], Ef(r0, tser[37])) and np.allclose(H_s[37], Hf(r0, tser[37])), "ledger: vectorized helpers")
    Sser = np.cross(E_s, H_s) @ trav
    wser = 0.5 * eps * np.sum(E_s ** 2, axis=1) + 0.5 * mu * np.sum(H_s ** 2, axis=1)
    v = 1 / math.sqrt(mu * eps)
    S_av = np.trapezoid(Sser, tser) / T
    w_av = np.trapezoid(wser, tser) / T
    dw_max = np.max(np.abs(np.gradient(wser, tser)))
    # standing wave from the same parameters: a second wave with the same field vector, travelling the other way
    def E2(r, t): return EH_t(r, np.array([t]), back=True)[0][0]
    def H2(r, t): return EH_t(r, np.array([t]), back=True)[1][0]
    Es = lambda r, t: Ef(r, t) + E2(r, t)
    Hs_ = lambda r, t: Hf(r, t) + H2(r, t)
    rs1 = r0 + 0.1 * lam * trav
    Eamp = abs(amp) if kind == "E" else eta * abs(amp)
    fara = curl_fd(lambda rr: Es(rr, 0.3 * T), rs1, hx) + mu * (Hs_(rs1, 0.3 * T + ht) - Hs_(rs1, 0.3 * T - ht)) / (2 * ht)
    ampe = curl_fd(lambda rr: Hs_(rr, 0.3 * T), rs1, hx) - eps * (Es(rs1, 0.3 * T + ht) - Es(rs1, 0.3 * T - ht)) / (2 * ht)
    stand_ok = np.linalg.norm(fara) < 1e-5 * beta * Eamp and np.linalg.norm(ampe) < 1e-5 * beta * Eamp / eta
    st_S, st_w = [], []
    for k in range(7):
        rk = r0 + (k / 7) * lam * 0.5 * trav
        e1, h1 = EH_t(rk, tser)
        e2, h2 = EH_t(rk, tser, back=True)
        Et, Ht = e1 + e2, h1 + h2
        st_S.append(abs(np.trapezoid(np.cross(Et, Ht) @ trav, tser) / T))
        st_w.append(np.trapezoid(0.5 * eps * np.sum(Et ** 2, axis=1) + 0.5 * mu * np.sum(Ht ** 2, axis=1), tser) / T)
    standing_truth = stand_ok and max(st_S) < 1e-6 * abs(S_av) and min(st_w) > 0.1 * w_av
    # closed box in the travelling wave: two faces normal to the travel carry flux, the side faces none
    L = 0.37 * lam
    eo, ho = EH_t(r0 + L * trav, tser)
    face_in = np.trapezoid(Sser, tser) / T * L * L
    face_out = np.trapezoid(np.cross(eo, ho) @ trav, tser) / T * L * L
    side = np.trapezoid(np.cross(E_s, H_s) @ fhat, tser) / T * L * L
    closed_truth = abs(face_out - face_in) < 1e-6 * abs(face_in) and abs(side) < 1e-9 * abs(face_in)
    # time-varying epsilon counterexample
    tq = np.linspace(0, 6, 60001)
    Eq = np.cos(3 * tq)
    for epsq, tag in ((2.0 + 0 * tq, "const"), (2.0 + np.sin(tq), "vary")):
        lhs = Eq * np.gradient(epsq * Eq, tq)
        rhs = np.gradient(0.5 * epsq * Eq * Eq, tq)
        if tag == "const":
            const_ok = np.max(np.abs(lhs - rhs)[5:-5]) < 1e-5
        else:
            vary_fails = np.max(np.abs(lhs - rhs)[5:-5]) > 1e-2
    # radiating sheet in vacuum: pillbox balance
    eta0 = math.sqrt(MU0 / EPS0)
    def Jsh(t): return 1.3 * math.cos(om * t) * E3[0]
    def Esh(z, t): return -(eta0 / 2) * Jsh(t - abs(z) / C)
    def Hsh(z, t): return 0.5 * np.cross(Jsh(t - abs(z) / C), np.sign(z) * E3[2])
    src_ok = True
    for t in np.linspace(0.05, 0.95, 9) * T:
        JE = np.dot(Jsh(t), Esh(0.0, t))
        out = np.dot(np.cross(Esh(1e-12, t), Hsh(1e-12, t)), E3[2]) - np.dot(np.cross(Esh(-1e-12, t), Hsh(-1e-12, t)), E3[2])
        src_ok &= (JE < 0) and abs(out + JE) < 1e-6 * abs(JE)
    # inner cylinder of the wire
    ft = re.search(r"\\tfrac\{1\}\{(\d+)\}", next(f["text"] for f in p["facts_list"] if f["id"] == "inner"))
    frac_shown = 1.0 / int(ft.group(1))
    rin = a / p["k_in"]
    flow_in = 0.0
    for ph in phis:
        rhat = math.cos(ph) * q1 + math.sin(ph) * q2
        flow_in += -np.dot(np.cross(E_at(rin * rhat), H_at(rin * rhat)), rhat) * rin * (2 * math.pi / 720) * ell
    inner_truth = abs(flow_in / (I * I * R) - frac_shown) < 2e-2 * frac_shown
    # axial S anywhere in the wire?
    ax_max = max(abs(np.dot(np.cross(Ein, Hw(rr)), wv)) for rr in [(rng.uniform(-.7, .7) * q1 + rng.uniform(-.7, .7) * q2) * a for _ in range(6)])
    # magnetic force does no work
    vv, BB = rng.normal(size=3), rng.normal(size=3)
    mag_work = abs(np.dot(np.cross(vv, BB), vv))
    # units: exponents of (kg, m, s, A)
    V_, A_, m_, W_ = np.array([1, 2, -3, -1]), np.array([0, 0, 0, 1]), np.array([0, 1, 0, 0]), np.array([1, 2, -3, 0])
    S_rs = Sser
    truth = {
        "units": np.array_equal(V_ - m_ + A_ - m_, W_ - 2 * m_),
        "source": src_ok,
        "div_out": div_ok and resid_ok,
        "instant": resid_ok and wire_ok,
        "lti": const_ok and vary_fails,
        "standing": standing_truth,
        "closed_avg": closed_truth,
        "inner": inner_truth,
        "S_alongE": np.linalg.norm(np.cross(S_at(r0, 0.1 * T), Ef(r0, 0.1 * T))) < 1e-9 * np.linalg.norm(S_at(r0, 0.1 * T)) * np.linalg.norm(Ef(r0, 0.1 * T)),
        "ends": ax_max > 1e-12 * np.linalg.norm(Ein) * I / a,
        "magnetic_work": mag_work > 1e-12,
        "S_over_w": abs(S_av - w_av / v) < 1e-3 * abs(S_av),
        "dwdt_zero": dw_max < 1e-9 * w_av * om,
        "S_reverses": np.min(S_rs) < -1e-9 * np.max(S_rs),
    }
    check(abs(S_av - v * w_av) < 1e-3 * abs(S_av), "ledger: <S> = v <w> fails (self-test)")
    expect = {"units": True, "source": True, "div_out": True, "instant": True, "lti": True, "standing": True,
              "closed_avg": True, "S_alongE": False, "ends": False, "magnetic_work": False, "S_over_w": False,
              "dwdt_zero": False, "S_reverses": False}
    for k, val in expect.items():
        check(bool(truth[k]) == val, f"ledger: computed truth of {k} is {truth[k]}, expected {val} (self-test)")
    facts = p["facts_list"]
    check({f["id"] for f in facts} == set(truth) and len(facts) == len(truth), "ledger: statement ids")
    for f in facts:
        check((f["correct"] == "true") == bool(truth[f["id"]]), f"ledger: statement {f['id']} key")
        check(f["verdict"] == f["correct"], "ledger: verdict text")
