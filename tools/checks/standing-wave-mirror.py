"""Independent check for waves/standing-wave-mirror (Lecture 26).

The incident field is parsed from its displayed LaTeX into a real function E_i(r, t). Its travel direction is
found numerically (which shift r -> r + u c dt, t -> t + dt leaves E unchanged). The reflected wave is a plane
wave travelling the other way whose phasor is fixed by a direct solve of the perfect-conductor condition
(tangential E = 0 on the surface). Each wave's H = u x E/eta0 is built with np.cross; Faraday's and Ampere's laws
are checked on the total field by finite differences, and normal B = 0 on the mirror.
Nulls: scan max_t |E(D, t)| and max_t |H(D, t)| in D and refine the minima. Hand: numeric rotation of E at a fixed
point against the numeric travel direction. J_s = n x H(0, t). Pressure: time average of J_s x B_avg (B_avg the
mean of the fields on the two faces of the sheet), compared with 2<S_i>/c from quadrature of E_i x H_i, and the
absorber-versus-mirror ratio from the Maxwell stress tensor. Every checkbox statement is re-decided numerically.
"""
import math
import re

import numpy as np

IDX = {"x": 0, "y": 1, "z": 2}
E3 = np.eye(3)
C0 = 3e8                                   # as stated
MU0 = 4e-7 * math.pi
EPS0 = 1 / (MU0 * C0 ** 2)
ETA0 = math.sqrt(MU0 / EPS0)               # = 120 pi

ARG = r"\\omega t ([+-]) \\beta ([xyz])"
LIN = r"(-?)(\d+)\\cos\(" + ARG + r"\)\\,\\hat\{([xyz])\}"
CIRC = (r"(-?)(\d+)\\left\[\\cos\(" + ARG + r"\)\\,\\hat\{([xyz])\} ([+-]) \\sin\(" + ARG +
        r"\)\\,\\hat\{([xyz])\}\\right\]")


def parse_incident(tex):
    """-> list of terms (amplitude vector, 'cos'|'sin', sign of the beta term, axis index)."""
    m = re.fullmatch(LIN, tex)
    if m:
        sg, amp, pm, ax, eax = m.groups()
        A = (-1 if sg else 1) * int(amp)
        return [(A * E3[IDX[eax]], "cos", 1 if pm == "+" else -1, IDX[ax])]
    m = re.fullmatch(CIRC, tex)
    assert m, "cannot parse incident field: " + tex
    sg, amp, pm1, ax1, e1, op, pm2, ax2, e2 = m.groups()
    assert pm1 == pm2 and ax1 == ax2
    A = (-1 if sg else 1) * int(amp)
    s = 1 if pm1 == "+" else -1
    return [(A * E3[IDX[e1]], "cos", s, IDX[ax1]), (A * (1 if op == "+" else -1) * E3[IDX[e2]], "sin", s, IDX[ax1])]


def as_pts(R):
    R = np.asarray(R, float)
    return R.reshape(-1, 3)


def make_wave(terms, om, beta):
    """E(R, t): R points (N, 3) or (3,), t array (M,) or scalar -> (N, M, 3)."""
    def E(R, t):
        R = as_pts(R)
        t = np.atleast_1d(np.asarray(t, float))
        out = np.zeros((R.shape[0], t.size, 3))
        for vec, fn, s, ax in terms:
            ph = om * t[None, :] + s * beta * R[:, ax][:, None]
            g = np.cos(ph) if fn == "cos" else np.sin(ph)
            out += g[:, :, None] * vec[None, None, :]
        return out
    return E


def phasor_wave(Rc, u, om, beta):
    """Real field of the phasor Rc e^{-j beta u.r} (travel along u)."""
    def E(R, t):
        R = as_pts(R)
        t = np.atleast_1d(np.asarray(t, float))
        ph = np.exp(1j * (om * t[None, :] - beta * (R @ u)[:, None]))
        return np.real(ph[:, :, None] * Rc[None, None, :])
    return E


def h_of(E, u):
    return lambda R, t: np.cross(u, E(R, t)) / ETA0


def one(F, r, t):
    return F(r, t)[0, 0]


def travel_dir(E, om, v):
    """Unit vector u (one of +-axes) with E(r + u v dt, t + dt) = E(r, t)."""
    rng = np.random.default_rng(7)
    found = []
    for k in range(3):
        for sg in (1, -1):
            u = sg * E3[k]
            ok = True
            for _ in range(4):
                r = rng.uniform(-1, 1, 3) * (v / om)
                t = rng.uniform(0, 2 * math.pi / om)
                dt = 0.31 / om
                e1, e2 = one(E, r, t), one(E, r + u * v * dt, t + dt)
                if np.linalg.norm(e1 - e2) > 1e-9 * max(1.0, np.linalg.norm(e1)):
                    ok = False
            if ok:
                found.append(u)
    assert len(found) == 1, f"travel direction ambiguous: {found}"
    return found[0]


def hand_of(E, u, r0, om):
    """+1 right, -1 left, 0 linear: sign of u . (E(t) x E(t + dt)) over a period (right thumb along u)."""
    T = 2 * math.pi / om
    ts = np.linspace(0, T, 400, endpoint=False)
    dt = T / 1000
    e1, e2 = E(r0, ts)[0], E(r0, ts + dt)[0]
    vals = np.cross(e1, e2) @ u
    scale = np.max(np.sum(e1 * e1, axis=1)) * (2 * math.pi * dt / T)
    if np.max(np.abs(vals)) < 1e-9 * scale:
        return 0
    assert vals.min() > 0 or vals.max() < 0, "rotation sense changes within a period"
    return 1 if vals[0] > 0 else -1


def curl(F, r, h):
    J = np.zeros((3, 3))
    for j in range(3):
        dr = np.zeros(3); dr[j] = h
        J[:, j] = (F(r + dr) - F(r - dr)) / (2 * h)
    return np.array([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]])


def golden_min(fn, lo, hi, it=70):
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = fn(c), fn(d)
    for _ in range(it):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a); fc = fn(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a); fd = fn(d)
    return 0.5 * (a + b)


def reflect(Ei, ui, om, beta, T):
    """Reflected phasor from tangential E_total = 0 on the plane through the origin (all E_i tangential)."""
    Rc = -one(Ei, np.zeros(3), 0.0) + 1j * one(Ei, np.zeros(3), T / 4)
    return phasor_wave(Rc, -ui, om, beta), Rc


TERM = re.compile(r"([+-]?) ?\\(sin|cos)\(\\omega t\)\\,\\hat\{([xyz])\}")


def parse_time_terms(rest):
    """'\\left[-\\sin(\\omega t)\\,\\hat{a} + ...\\right]' or a single term -> list of (sign, fn, axis index)."""
    if rest.startswith(r"\left["):
        assert rest.endswith(r"\right]"), rest
        rest = rest[len(r"\left["):-len(r"\right]")]
    out, pos = [], 0
    for m in TERM.finditer(rest):
        assert rest[pos:m.start()].strip() == "", rest
        out.append((-1 if m.group(1) == "-" else 1, m.group(2), IDX[m.group(3)]))
        pos = m.end()
    assert rest[pos:].strip() == "" and out, rest
    return out


def parse_standing(tex, unit, spatial=True):
    """Answer-panel field 'c sin(beta w)[...] unit' -> function F(r, t) (vector)."""
    pat = r"(-?[\d.]+)" + (r"\\(sin|cos)\(\\beta ([xyz])\)" if spatial else "") + r"(.*)\\ \\mathrm\{" + re.escape(unit) + r"\}"
    m = re.fullmatch(pat, tex)
    assert m, "cannot parse " + tex
    c = float(m.group(1))
    terms = parse_time_terms(m.group(4) if spatial else m.group(2))
    def F(r, t, om, beta):
        sp = 1.0
        if spatial:
            sp = (math.sin if m.group(2) == "sin" else math.cos)(beta * r[IDX[m.group(3)]])
        v = np.zeros(3)
        for sg, fn, ax in terms:
            v[ax] += sg * (math.sin(om * t) if fn == "sin" else math.cos(om * t))
        return c * sp * v
    return F


def phys(d, check):
    p, ca = d["params"], d["correct_answers"]
    f = float(p["f_Hz"])
    om = 2 * math.pi * f
    beta = om / C0
    lam = C0 / f
    T = 1 / f
    terms = parse_incident(p["E_tex"])
    Ei = make_wave(terms, om, beta)

    # region and normal (out of the conductor into the air)
    m = re.fullmatch(r"([xyz]) \\(gt|lt) 0", p["region"])
    m2 = re.fullmatch(r"([xyz]) \\(gt|lt) 0", p["cond"])
    check(m is not None and m2 is not None and m2.group(1) == m.group(1) and m2.group(2) != m.group(2),
          "mirror: region / conductor side text")
    wax = IDX[m.group(1)]
    n = (1 if m.group(2) == "gt" else -1) * E3[wax]

    ui = travel_dir(Ei, om, C0)
    check(np.allclose(ui, -n), "mirror: incident wave does not travel toward the mirror")
    check(abs(one(Ei, np.zeros(3), 0.2 * T) @ n) < 1e-12, "mirror: incident E not tangential")

    Er, Rc = reflect(Ei, ui, om, beta, T)
    ur = travel_dir(Er, om, C0)
    check(np.allclose(ur, -ui), "mirror: reflected travel direction")
    tt = np.linspace(0, T, 13)
    tot0 = Ei(np.zeros(3), tt)[0] + Er(np.zeros(3), tt)[0]
    tang = tot0 - np.outer(tot0 @ n, n)
    check(np.max(np.abs(tang)) < 1e-9 * np.max(np.abs(Rc)), "mirror: PEC tangential E not zero")

    Hi, Hr = h_of(Ei, ui), h_of(Er, ur)
    E = lambda R, t: Ei(R, t) + Er(R, t)
    H = lambda R, t: Hi(R, t) + Hr(R, t)

    # ---- Maxwell by finite differences in the air; normal B on the mirror
    rng = np.random.default_rng(11)
    for _ in range(3):
        r = n * rng.uniform(0.05, 1.3) * lam + rng.uniform(-1, 1, 3) * lam * (1 - np.abs(n))
        t = rng.uniform(0, T)
        hs, ht = lam * 1e-5, T * 1e-5
        cE = curl(lambda rr: one(E, rr, t), r, hs)
        dH = (one(H, r, t + ht) - one(H, r, t - ht)) / (2 * ht)
        check(np.linalg.norm(cE + MU0 * dH) < 1e-5 * max(np.linalg.norm(cE), 1e-12), "mirror: Faraday's law fails")
        cH = curl(lambda rr: one(H, rr, t), r, hs)
        dE = (one(E, r, t + ht) - one(E, r, t - ht)) / (2 * ht)
        check(np.linalg.norm(cH - EPS0 * dE) < 1e-5 * max(np.linalg.norm(cH), 1e-15), "mirror: Ampere's law fails")
    check(np.max(np.abs(H(np.zeros(3), tt)[0] @ n)) < 1e-15, "mirror: normal B on the mirror not zero")

    ts = np.linspace(0, T, 720, endpoint=False)
    tc = ts[::6]

    def amp(F, Ds):
        Ds = np.atleast_1d(Ds)
        v = F(np.outer(Ds, n), tc)
        return np.max(np.linalg.norm(v, axis=2), axis=1)

    E0 = np.max(np.linalg.norm(Ei(np.zeros(3), ts)[0], axis=1))

    def nulls(F, scale):
        Ds = np.linspace(0, 1.3 * lam, 521)
        vals = amp(F, Ds)
        out = [0.0] if vals[0] < 1e-9 * scale else []
        for i in range(1, len(Ds) - 1):
            if vals[i] <= vals[i - 1] and vals[i] <= vals[i + 1] and vals[i] < 0.1 * scale:
                Dm = golden_min(lambda D: amp(F, D)[0], Ds[i - 1], Ds[i + 1])
                if amp(F, Dm)[0] < 1e-6 * scale:
                    out.append(Dm)
        return out

    En = nulls(E, 2 * E0)
    Hn = nulls(H, 2 * E0 / ETA0)
    check(len(En) >= 3 and En[0] == 0.0, f"mirror: E nulls {En}")
    check(len(Hn) >= 2 and Hn[0] > 0, f"mirror: H nulls {Hn}")
    check(abs(En[1] * 100 - ca["dE"]) < 1e-6 * ca["dE"], f"mirror: first E null {En[1]*100} vs {ca['dE']}")
    check(abs(Hn[0] * 100 - ca["dH"]) < 1e-6 * ca["dH"], f"mirror: first H null {Hn[0]*100} vs {ca['dH']}")

    # ---- probe
    Dp = float(p["dp_str"]) / 100
    check(abs(float(p["wp_str"]) / 100 - Dp * (n @ np.ones(3))) < 1e-12, "mirror: probe coordinate vs distance")
    Ep = np.max(np.linalg.norm(E(Dp * n, ts)[0], axis=1))
    Hp = np.max(np.linalg.norm(H(Dp * n, ts)[0], axis=1)) * 1e3
    check(abs(Ep - ca["Emax"]) < 1e-6 * max(1, Ep), f"mirror: probe |E|max {Ep} vs {ca['Emax']}")
    check(abs(Hp - ca["Hmax"]) < 1e-6 * max(1, Hp), f"mirror: probe |H|max {Hp} vs {ca['Hmax']}")
    check(Ep > 1e-3 * E0 and Hp > 1e-6, "mirror: probe sits on a node")

    # ---- (c) reflected wave
    r0 = np.zeros(3)
    hi = hand_of(Ei, ui, r0, om)
    hr = hand_of(Er, ur, r0, om)
    corr = [c for c in p["refl_list"] if c["correct"] == "true"]
    check(len(corr) == 1, "mirror: refl MC needs one correct")
    name = {1: "right", -1: "left"}
    if hi == 0:
        check(hr == 0, "mirror: linear incident gave non-linear reflection")
        ei, er = Ei(r0, ts)[0], Er(r0, ts)[0]
        axE = int(np.argmax(np.abs(ei[0])))
        off = [k for k in range(3) if k != axE]
        check(np.max(np.abs(er[:, off])) < 1e-12, "mirror: reflected E leaves the axis")
        inv = np.allclose(er, -ei, atol=1e-12)
        same = np.allclose(er, ei, atol=1e-12)
        check(inv and not same and corr[0]["id"] == "lin-inv", "mirror: linear reflection key")
    else:
        check(hr == -hi, "mirror: reflected circular hand did not flip")
        check(corr[0]["id"] == f"{name[hi]}-{name[hr]}", f"mirror: hand key {corr[0]['id']} vs {name[hi]}-{name[hr]}")
        check(p["hand_i"] == name[hi] and p["hand_r"] == name[hr], "mirror: answer-panel hands")
    check(len({c["text"] for c in p["refl_list"]}) == len(p["refl_list"]), "mirror: duplicate refl choices")

    # ---- (d) surface current
    H0 = H(r0, ts)[0]
    Jall = np.cross(n, H0)
    Jm = np.max(np.linalg.norm(Jall, axis=1)) * 1e3
    check(abs(Jm - ca["Js"]) < 1e-6 * Jm, f"mirror: |J_s|max {Jm} vs {ca['Js']}")
    t1 = {"0": 0.0, "T/4": 0.25, "T/2": 0.5, "3T/4": 0.75}[p["t1_tex"]] * T
    H1 = one(H, r0, t1)
    J1 = np.cross(n, H1)
    check(np.linalg.norm(J1) * 1e3 > 0.5 * Jm, "mirror: J_s small at the asked instant")
    k = int(np.argmax(np.abs(J1)))
    check(np.linalg.norm(J1 - J1[k] * E3[k]) < 1e-9 * np.linalg.norm(J1), "mirror: J_s(t1) not along an axis")
    jc = [c for c in p["js_list"] if c["correct"] == "true"]
    check(len(jc) == 1 and jc[0]["ax"] == "xyz"[k] and jc[0]["sg"] == int(np.sign(J1[k])),
          f"mirror: J_s direction key vs {'xyz'[k]}{np.sign(J1[k])}")
    kh = int(np.argmax(np.abs(H1)))
    for c in p["js_list"]:
        if c["id"] == "H":
            check(c["ax"] == "xyz"[kh] and c["sg"] == int(np.sign(H1[kh])), "mirror: 'H' distractor is not along H")
    check(len({c["text"] for c in p["js_list"]}) == len(p["js_list"]), "mirror: duplicate J_s choices")
    check(p["J_dir"] == ("+" if J1[k] > 0 else "-") + r"\hat{" + "xyz"[k] + "}" and
          p["H_dir"] == ("+" if H1[kh] > 0 else "-") + r"\hat{" + "xyz"[kh] + "}", "mirror: answer-panel directions")

    # ---- (e) pressure: J_s x B_avg, B_avg = mu0 H(0)/2 (field is zero inside the conductor)
    Pvec = np.mean(np.cross(Jall, 0.5 * MU0 * H0), axis=0)
    Pmag = np.linalg.norm(Pvec) * 1e9
    check(np.allclose(Pvec / np.linalg.norm(Pvec), -n), "mirror: pressure not into the conductor")
    Si = np.mean(np.cross(Ei(r0, ts)[0], Hi(r0, ts)[0]), axis=0)
    check(abs(2 * np.linalg.norm(Si) / C0 * 1e9 - Pmag) < 1e-6 * Pmag, "mirror: J x B pressure != 2<S_i>/c")
    check(abs(Pmag - ca["P"]) < 1e-6 * Pmag, f"mirror: pressure {Pmag} vs {ca['P']}")
    check(abs(Pmag * float(p["A_m2"]) - ca["F"]) < 1e-6 * ca["F"], "mirror: force = P A")

    # ---- answer-panel formulas against the fields built here
    Ft = parse_standing(p["Etot_tex"], "V/m")
    Fh = parse_standing(p["Htot_tex"], "mA/m")
    Fj = parse_standing(p["Ji_tex"], "mA/m", spatial=False)
    er_tex = p["Er_tex"]
    check(er_tex.endswith(r"\ \ \mathrm{V/m}"), "mirror: E_r display units")
    Er_disp = make_wave(parse_incident(er_tex[:-len(r"\ \ \mathrm{V/m}")]), om, beta)
    for _ in range(4):
        rr = n * rng.uniform(0, 1.2) * lam + rng.uniform(-1, 1, 3) * lam * (1 - np.abs(n))
        tq = rng.uniform(0, T)
        sc = 2 * E0
        check(np.linalg.norm(Ft(rr, tq, om, beta) - one(E, rr, tq)) < 2e-3 * sc, "mirror: answer-panel total E")
        check(np.linalg.norm(Fh(rr, tq, om, beta) - one(H, rr, tq) * 1e3) < 2e-3 * sc / ETA0 * 1e3, "mirror: answer-panel total H")
        check(np.linalg.norm(Er_disp(rr, tq)[0, 0] - one(Er, rr, tq)) < 1e-9 * sc, "mirror: answer-panel E_r")
        check(np.linalg.norm(Fj(rr, tq, om, beta) - np.cross(n, one(H, r0, tq)) * 1e3) < 2e-3 * sc / ETA0 * 1e3, "mirror: answer-panel J_s")

    # ---- (f) statements, each re-decided by a computation
    Ssc = np.linalg.norm(Si)
    truth = {}
    Dset = np.array([0.07, 0.19, 0.33, 0.61]) * lam
    Ev, Hv = E(np.outer(Dset, n), ts[::3]), H(np.outer(Dset, n), ts[::3])
    truth["avgS0"] = np.max(np.linalg.norm(np.mean(np.cross(Ev, Hv), axis=1), axis=1)) < 1e-9 * Ssc
    truth["Enull_half"] = abs((En[2] - En[1]) - lam / 2) < 1e-6 * lam and abs(En[1] - lam / 2) < 1e-6 * lam
    truth["Etan0"] = np.max(np.abs(tang)) < 1e-9 * E0

    def flips(hsign):
        a_ax = [kk for kk in range(3) if kk != wax][0]
        b_ax = [kk for kk in range(3) if kk not in (wax, a_ax)][0]
        s = int(round(-ui[wax]))            # phase w t + s beta w travels along -s w_hat = ui
        Et = make_wave([(E3[a_ax], "cos", s, wax), (hsign * E3[b_ax], "sin", s, wax)], om, beta)
        u_t = travel_dir(Et, om, C0)
        Ert, _ = reflect(Et, u_t, om, beta, T)
        ur_t = travel_dir(Ert, om, C0)
        h1, h2 = hand_of(Et, u_t, r0, om), hand_of(Ert, ur_t, r0, om)
        return h1 != 0 and h2 == -h1
    truth["circ_flip"] = flips(1) and flips(-1)

    def stress_p(e, hh):          # -T_nn averaged: pressure on a body with these fields on its face
        en, hn = e @ n, hh @ n
        return np.mean(0.5 * EPS0 * np.sum(e * e, 1) + 0.5 * MU0 * np.sum(hh * hh, 1) - EPS0 * en ** 2 - MU0 * hn ** 2)
    P_mirror = stress_p(E(r0, ts)[0], H0)
    P_abs = stress_p(Ei(r0, ts)[0], Hi(r0, ts)[0])
    check(abs(P_mirror * 1e9 - Pmag) < 1e-6 * Pmag, "mirror: stress-tensor pressure != J x B pressure")
    truth["absorber_half"] = abs(P_abs / P_mirror - 0.5) < 1e-9

    Dz = np.linspace(0.013 * lam, 1.2 * lam, 2000)

    def zeros_at(t):
        v = E(np.outer(Dz, n), np.array([t]))[:, 0, :]
        comp = int(np.argmax(np.max(np.abs(v), axis=0)))
        s = np.sign(v[:, comp])
        return Dz[np.where(s[:-1] != s[1:])[0]]
    z1, z2 = zeros_at(0.11 * T), zeros_at(0.37 * T)
    truth["fixed_nodes"] = len(z1) == len(z2) and len(z1) > 0 and np.allclose(z1, z2, atol=2 * lam / 2000)

    Dg = np.linspace(0.03, 1.1, 23) * lam
    Sg = np.cross(E(np.outer(Dg, n), ts[::15]), H(np.outer(Dg, n), ts[::15]))
    truth["S_inst0"] = np.max(np.linalg.norm(Sg, axis=2)) < 1e-9 * Ssc
    Pt = 0.5 * MU0 * np.sum(H0 * H0, axis=1)
    truth["P_const"] = (Pt.max() - Pt.min()) < 1e-9 * Pt.mean()
    Dn = En[1]
    eN, hN = E(Dn * n, ts)[0], H(Dn * n, ts)[0]
    wt = 0.5 * EPS0 * np.sum(eN * eN, 1) + 0.5 * MU0 * np.sum(hN * hN, 1)
    truth["w0_Enode"] = wt.min() < 1e-9 * wt.mean()
    Dq = lam / 8
    eq, hq = E(Dq * n, ts)[0], H(Dq * n, ts)[0]
    iq = int(np.argmax(np.linalg.norm(hq, axis=1)))
    truth["E0_Hmax"] = np.linalg.norm(eq[iq]) < 1e-9 * E0
    truth["Htan0"] = np.max(np.linalg.norm(Jall, axis=1)) < 1e-12
    truth["nodes_same"] = any(abs(a - b) < 1e-6 * lam for a in En for b in Hn)
    truth["avgP0"] = Pmag < 1e-9
    big = np.linalg.norm(Jall, axis=1) > 1e-3 * Jm * 1e-3
    par = np.linalg.norm(np.cross(Jall, H0), axis=1) < 1e-9 * np.linalg.norm(Jall, axis=1) * np.linalg.norm(H0, axis=1)
    truth["Js_alongH"] = bool(np.all(par[big]))
    truth["avgw0"] = wt.mean() < 1e-12
    Him = np.max(np.linalg.norm(Hi(r0, ts)[0], axis=1)) * 1e3
    truth["Js_eqH"] = abs(Jm - Him) < 1e-6 * Jm
    truth["Emax_mirror"] = int(np.argmax(amp(E, np.linspace(0, lam / 2, 201)))) == 0
    ids = {c["id"] for c in p["facts_list"]}
    check(ids == set(truth), f"mirror: statement ids differ {ids ^ set(truth)}")
    for c in p["facts_list"]:
        check((c["correct"] == "true") == bool(truth.get(c["id"])),
              f"mirror: statement {c['id']} marked {c['correct']}, computed {truth.get(c['id'])}")
    check(len({c["text"] for c in p["facts_list"]}) == len(p["facts_list"]), "mirror: duplicate statements")
