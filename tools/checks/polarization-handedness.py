"""Independent check for waves/polarization-handedness (Lecture 24).

(a) Every matching statement is parsed from its displayed LaTeX (phasor or real form) into a function E(q, t)
    (omega = beta = 1). The travel direction is found numerically (which shift q -> q +- d, t -> t + d leaves E
    unchanged), and the polarization by sampling E over one period at a fixed point: linear if E(t) x E(t+dt) = 0,
    circular if |E| is constant, hand from the sign of u . (E x dE/dt) (right thumb along the travel).
(b) The wave is parsed from its text and built in SI units; H comes from Faraday's law (finite-difference curl of E,
    integrated in time, mean removed), then <S> by quadrature of E x H over a period.
(c) beta(omega) is solved by bisection from omega^2 = omega_p^2 + c^2 beta^2; v_p = omega/beta, v_g by a central
    difference of omega(beta); crossing time L/v_g.
(d) every statement is re-decided by a computation (numeric rotation, quadrature, sheet fields).
"""
import math
import re

import numpy as np

IDX = {"x": 0, "y": 1, "z": 2}
E3 = np.eye(3)
CN = {"lin": "linear", "rc": "right-circular", "lc": "left-circular",
      "re": "right-handed elliptical", "le": "left-handed elliptical"}
QE, ME, EPS0N, C = 1.602e-19, 9.109e-31, 8.854e-12, 3e8
MU0 = 4e-7 * math.pi
ETA0 = 120 * math.pi
EPS0 = MU0 / ETA0 ** 2          # consistent with eta0 = 120 pi and mu0

PH = r"(?: ([+-]) (\d*)\\pi(?:/(\d+))?)?"
TT = r"(\d+)\\(cos|sin)\(\\omega t ([+-]) \\beta ([xyz])" + PH + r"\)\\,\\hat\{([xyz])\}"
PT = r"(\d+)(j|e\^\{(-?)j(\d*)\\pi(?:/(\d+))?\})?\\,\\hat\{([xyz])\}"


def phase(sg, n, d):
    if sg is None:
        return 0.0
    v = (int(n) if n else 1) * math.pi / (int(d) if d else 1)
    return v if sg == "+" else -v


def parse_field(tex):
    """-> (E(q, t) as a 3-vector, axis letter of q). omega = beta = 1."""
    tex = tex.strip().strip("$")
    m = re.fullmatch(r"\\mathbf\{E\} = (-?)" + TT + r" ([+-]) " + TT, tex)
    if m:
        g = m.groups()
        t1 = (g[0], g[1:9])
        t2 = (g[9], g[10:18])
        terms, axes = [], set()
        for sgn, (amp, fn, asg, ax, psg, pn, pd, eax) in (t1, t2):
            terms.append(((-1 if sgn == "-" else 1) * int(amp), fn, -1 if asg == "-" else 1,
                          phase(psg, pn, pd), E3[IDX[eax]]))
            axes.add(ax)
        assert len(axes) == 1

        def Ef(q, t):
            out = np.zeros(3)
            for amp, fn, asg, th, e in terms:
                arg = t + asg * q + th
                out += amp * (math.cos(arg) if fn == "cos" else math.sin(arg)) * e
            return out
        return Ef, axes.pop()
    m = re.fullmatch(r"\\tilde\{\\mathbf\{E\}\} = (j|-j|-)?\((-?)" + PT + r" ([+-]) " + PT +
                     r"\)\\,e\^\{(-?)j\\beta ([xyz])\}", tex)
    assert m, "cannot parse " + tex
    g = m.groups()
    F = {None: 1, "j": 1j, "-j": -1j, "-": -1}[g[0]]

    def coef(amp, kind, esg, en, ed):
        v = complex(int(amp))
        if kind == "j":
            v *= 1j
        elif kind:
            ph = (int(en) if en else 1) * math.pi / (int(ed) if ed else 1)
            v *= complex(math.cos(ph), (-1 if esg == "-" else 1) * math.sin(ph))
        return v
    c1 = (-1 if g[1] == "-" else 1) * coef(*g[2:7])
    c2 = (-1 if g[8] == "-" else 1) * coef(*g[9:14])
    vec = F * (c1 * E3[IDX[g[7]]] + c2 * E3[IDX[g[14]]])
    ks = -1 if g[15] == "-" else 1

    def Ef(q, t):
        return np.real(vec * np.exp(1j * (ks * q + t)))
    return Ef, g[16]


def travel(Ef, check, tag):
    """Numerical direction of travel along the argument axis: +1 or -1."""
    rng = np.random.default_rng(7)
    res = set()
    for _ in range(4):
        q0, t0, dd = rng.uniform(-2, 2), rng.uniform(0, 6), rng.uniform(0.2, 1.0)
        ref = Ef(q0, t0)
        plus = np.linalg.norm(Ef(q0 + dd, t0 + dd) - ref)
        minus = np.linalg.norm(Ef(q0 - dd, t0 + dd) - ref)
        sc = max(np.linalg.norm(ref), 1e-9)
        if plus < 1e-9 * sc and minus > 1e-6 * sc:
            res.add(1)
        elif minus < 1e-9 * sc and plus > 1e-6 * sc:
            res.add(-1)
        else:
            res.add(0)
    check(len(res) == 1 and 0 not in res, f"pol: travel direction undetermined for {tag}")
    return res.pop()


def rotation(Ef_t, uvec, n=721):
    """Sample E over one period (omega = 1): returns (kind, hand, total turned angle about u, |E| list)."""
    ts = np.linspace(0, 2 * math.pi, n)
    Es = np.array([Ef_t(t) for t in ts])
    mags = np.linalg.norm(Es, axis=1)
    sc = float(np.max(mags) ** 2)
    cr = np.cross(Es[:-1], Es[1:])
    if float(np.max(np.linalg.norm(cr, axis=1))) < 1e-9 * sc:
        return "lin", 0, 0.0, mags
    L = cr @ uvec
    hand = 1 if float(np.sum(L)) > 0 else -1
    mono = bool(np.all(np.sign(L) == hand))
    p = E3[int(np.argmin(np.abs(uvec)))]
    p = p - np.dot(p, uvec) * uvec
    p /= np.linalg.norm(p)
    q = np.cross(uvec, p)
    ang = np.unwrap(np.arctan2(Es @ q, Es @ p))
    turned = float(ang[-1] - ang[0])
    circ = float(np.max(mags) - np.min(mags)) < 1e-6 * float(np.max(mags))
    if not mono:
        return "bad", hand, turned, mags
    if circ:
        return ("rc" if hand > 0 else "lc"), hand, turned, mags
    return ("re" if hand > 0 else "le"), hand, turned, mags


def classify_text(tex, check, tag):
    Ef, ax = parse_field(tex)
    s = travel(Ef, check, tag)
    uvec = s * E3[IDX[ax]]
    q0 = 0.37
    check(abs(np.dot(Ef(q0, 0.4), uvec)) < 1e-12 * max(1, np.linalg.norm(Ef(q0, 0.4))), f"pol: {tag} not transverse")
    kind, hand, turned, mags = rotation(lambda t: Ef(q0, t), uvec)
    return kind, uvec, Ef, ax


def classify_phasor(vec, uvec):
    """Synthetic phasor at its reference plane, travel along uvec."""
    return rotation(lambda t: np.real(vec * np.exp(1j * t)), uvec)[0]


def phys(d, check):
    p, c = d["params"], d["correct_answers"]

    # ---------------------------------------------------------------- (a) matching
    items = p["match_items"]
    check(len(items) == 5 and len({it["text"] for it in items}) == 5, "pol: need 5 distinct statements")
    keys = set()
    for it in items:
        kind, uvec, Ef, ax = classify_text(it["text"], check, it["text"])
        check(kind == it["key"], f"pol: statement {it['text']} is {kind}, keyed {it['key']}")
        check(it["verdict"] == CN[it["key"]], "pol: verdict text")
        check(f"Travel ${'+' if uvec.sum() > 0 else '-'}\\hat{{{ax}}}$" in it["why"], f"pol: travel text in why for {it['text']}")
        keys.add(it["key"])
    check({"rc", "lc"} <= keys and len(keys) >= 3, "pol: matching keys not varied enough")

    # ---------------------------------------------------------------- (b) circular wave power
    if p["medium_b"].startswith("vacuum"):
        eps_r = 1.0
    else:
        eps_r = float(re.search(r"\\epsilon_r = ([\d.]+)\$", p["medium_b"]).group(1))
    check(abs(eps_r - p["eps_r"]) < 1e-12, "pol: medium text vs eps_r")
    Ef, ax = parse_field(p["wave_tex"])
    s = travel(Ef, check, "wave_b")
    uvec = s * E3[IDX[ax]]
    check(ax == p["ub"], "pol: wave axis vs ub")
    kind = rotation(lambda t: Ef(0.2, t), uvec)[0]
    check(kind in ("rc", "lc"), "pol: wave of (b) is not circular")
    om = 2 * math.pi * 1e8
    beta = om * math.sqrt(MU0 * EPS0 * eps_r)

    def E_SI(r, t):     # Ef uses omega = beta = 1: phase = om t +- beta q
        return Ef(beta * np.dot(r, E3[IDX[ax]]), om * t)

    def curlE(r, t, h=1e-6):
        J = np.zeros((3, 3))
        for j in range(3):
            dr = np.zeros(3); dr[j] = h
            J[:, j] = (E_SI(r + dr, t) - E_SI(r - dr, t)) / (2 * h)
        return np.array([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]])
    r0 = np.array([0.13, -0.27, 0.41])
    T = 2 * math.pi / om
    ts = np.linspace(0, T, 4001)
    dHdt = np.array([-curlE(r0, t) / MU0 for t in ts])
    Hs = np.zeros_like(dHdt)
    Hs[1:] = np.cumsum(0.5 * (dHdt[1:] + dHdt[:-1]) * (ts[1] - ts[0]), axis=0)
    Hs -= np.mean(Hs[:-1], axis=0)
    Es = np.array([E_SI(r0, t) for t in ts])
    Ss = np.cross(Es, Hs)
    Savg = np.mean(Ss[:-1], axis=0)
    check(abs(np.dot(Savg, E3[IDX[ax]]) * 1e3 - c["S_avg"]) < 2e-4 * abs(c["S_avg"]),
          f"pol: <S> {np.dot(Savg, E3[IDX[ax]]) * 1e3} vs {c['S_avg']}")
    Hm = np.linalg.norm(Hs, axis=1) * 1e3
    check(abs(float(np.max(Hm)) - c["H_mag"]) < 2e-4 * c["H_mag"] and abs(float(np.min(Hm)) - c["H_mag"]) < 2e-4 * c["H_mag"],
          f"pol: |H| {np.min(Hm)}..{np.max(Hm)} vs {c['H_mag']}")
    Sn = np.linalg.norm(Ss, axis=1)
    S_trans = np.linalg.norm(Ss - np.outer(Ss @ uvec, uvec), axis=1)
    inst = {"const": float(np.max(Sn) - np.min(Sn)) < 1e-3 * float(np.mean(Sn)) and abs(float(np.mean(Sn)) - np.linalg.norm(Savg)) < 1e-3 * float(np.mean(Sn)),
            "pulse": float(np.max(Sn) - np.min(Sn)) > 0.5 * float(np.mean(Sn)),
            "rotate": float(np.max(S_trans)) > 1e-3 * float(np.mean(Sn)),
            "zeroavg": np.linalg.norm(Savg) < 1e-3 * float(np.mean(Sn))}
    for ch in p["s_inst"]:
        check((ch["correct"] == "true") == inst[ch["id"]], f"pol: (b) MC choice {ch['id']} key")
    check(sum(ch["correct"] == "true" for ch in p["s_inst"]) == 1, "pol: (b) MC needs one key")

    # ---------------------------------------------------------------- (c) plasma
    m = re.fullmatch(r"(?:([\d.]+)\\times)?10\^\{(\d+)\}", p["N_tex"])
    N = (float(m.group(1)) if m.group(1) else 1.0) * 10.0 ** int(m.group(2))
    check(abs(N - p["N"]) < 1e-6 * N, "pol: N text")
    f = float(p["f_str"]) * 1e6
    L = float(p["L_km"]) * 1e3
    wp = math.sqrt(N * QE ** 2 / (ME * EPS0N))
    w = 2 * math.pi * f
    check(w > wp, "pol: carrier below f_p")
    wofb = lambda b: math.sqrt(wp ** 2 + (C * b) ** 2)
    lo, hi = 0.0, 10 * w / C
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if wofb(mid) < w else (lo, mid)
    b0 = 0.5 * (lo + hi)
    vp = w / b0
    db = b0 * 1e-6
    vg = (wofb(b0 + db) - wofb(b0 - db)) / (2 * db)
    check(abs(wp / (2 * math.pi) / 1e6 - c["fp"]) < 1e-6 * c["fp"], f"pol: f_p {wp / 2 / math.pi / 1e6} vs {c['fp']}")
    check(abs(vp / 1e8 - c["vp8"]) < 1e-6 * c["vp8"], f"pol: v_p {vp / 1e8} vs {c['vp8']}")
    check(abs(vg / 1e8 - c["vg8"]) < 1e-6 * c["vg8"], f"pol: v_g {vg / 1e8} vs {c['vg8']}")
    check(abs(L / vg * 1e3 - c["delay"]) < 1e-6 * c["delay"], f"pol: delay {L / vg * 1e3} vs {c['delay']}")
    check(vg < C < vp, "pol: v_g < c < v_p fails")
    sig_true = {"vg": True, "vp": False, "c": False, "mean": False, "none": False}
    for ch in p["sig"]:
        check((ch["correct"] == "true") == sig_true[ch["id"]], f"pol: (c) MC {ch['id']} key")

    # ---------------------------------------------------------------- (d) statements
    z, x, y = E3[2], E3[0], E3[1]
    rng = np.random.default_rng(11)
    tests = [(3 * x + 3j * y, z), (2 * y - 5j * z, -x), (4 * z + 4 * np.exp(1j * math.pi / 3) * x, y),
             (3 * x + 2 * y, -z), (2j * x - 2 * y, z)]
    ftxt = next(f["text"] for f in p["facts"] if f["id"] == "factor")
    fm = re.search(r"by \$(.+?)\$ changes", ftxt).group(1)
    F = {"j": 1j, "-1": -1, "-j": -1j, r"e^{j\pi/3}": np.exp(1j * math.pi / 3)}[fm]
    factor_ok = all(classify_phasor(v, u) == classify_phasor(F * v, u) for v, u in tests)
    # H of a circular wave rotates with E
    def hand_of(fn, u):
        return rotation(fn, u)[1]
    H_same = True
    for v, u in [(3 * x - 3j * y, z), (2 * y + 2j * z, -x), (5 * z - 5j * x, -y)]:
        Ef_ = lambda t, v=v: np.real(v * np.exp(1j * t))
        Hf_ = lambda t, v=v, u=u: np.cross(u, np.real(v * np.exp(1j * t))) / 100.0
        H_same &= hand_of(Ef_, u) == hand_of(Hf_, u) != 0
    # linear = RC + LC of equal amplitude (solve the 2x2 complex system for random linear fields)
    B = np.array([[1, 1], [-1j, 1j]])           # columns: x - j y (RC for +z), x + j y (LC for +z)
    lin_ok = classify_phasor(x - 1j * y, z) == "rc" and classify_phasor(x + 1j * y, z) == "lc"
    for _ in range(4):
        th = rng.uniform(0, math.pi)
        cf = np.linalg.solve(B, np.array([math.cos(th), math.sin(th)]) * rng.uniform(1, 5))
        lin_ok &= abs(abs(cf[0]) - abs(cf[1])) < 1e-12
    # twice the power: quadrature of E x H for a circular wave and for one component
    tq = np.linspace(0, 2 * math.pi, 2001)[:-1]
    def avgS(v, u, eta=100.0):
        Ev = np.real(np.outer(np.exp(1j * tq), v))
        Hv = np.cross(np.broadcast_to(u, Ev.shape), Ev) / eta
        return float(np.mean(np.cross(Ev, Hv) @ u))
    twice = abs(avgS(4 * x - 4j * y, z) / avgS(4 * x + 0j * y, z) - 2) < 1e-9
    # clockwise for a viewer looking along the travel
    def screen_turn(v, u):
        up = E3[int(np.argmin(np.abs(u)))]
        up = up - np.dot(up, u) * u
        up /= np.linalg.norm(up)
        right = np.cross(u, up)
        Ev = np.real(np.outer(np.exp(1j * tq), v))
        ang = np.unwrap(np.arctan2(Ev @ up, Ev @ right))
        return ang[-1] - ang[0]
    cw = all(screen_turn(v, u) < 0 for v, u in [(x - 1j * y, z), (y - 1j * x, -z), (y - 1j * z, x), (2j * z + 2 * x, y)]
             if classify_phasor(v, u) == "rc")
    cw &= sum(classify_phasor(v, u) == "rc" for v, u in [(x - 1j * y, z), (y - 1j * x, -z), (y - 1j * z, x), (2j * z + 2 * x, y)]) >= 2
    same_fn = all(classify_phasor(a * x + b * y, z) == "lin" for a, b in [(3, 4), (2, -5), (6, 6)])
    quarter_any = all(classify_phasor(a * x - 1j * b * y, z) in ("rc", "lc") for a, b in [(3, 3), (3, 2)])
    k1 = classify_phasor(x - 1j * y, z)
    k2 = classify_phasor(x - 1j * y, -z)
    w1 = np.cross(np.real(x - 1j * y), np.real((x - 1j * y) * np.exp(1j * 0.01)))
    reverse_keeps = (k1 == k2)          # same rotation at the point (w1 is the same field), opposite travel
    tr = rotation(lambda t: np.real((x - 1j * y) * np.exp(1j * t)), z)[2]
    half_period = abs(abs(tr) - 4 * math.pi) < 1e-6
    check(abs(abs(tr) - 2 * math.pi) < 1e-6 and np.linalg.norm(w1) > 0, "pol: circular wave should turn once per period")
    stxt = next(f["text"] for f in p["facts"] if f["id"] == "sheet_same")
    sm = re.search(r"J_0\[\\cos\(\\omega t\)\\,\\hat\{([xyz])\} \+ \\sin\(\\omega t\)\\,\\hat\{([xyz])\}\]\$ on the plane \$([xyz]) = 0", stxt)
    sp, sq, su = sm.groups()
    un = E3[IDX[su]]
    def Esheet(r, t, eta=100.0, v=1.0):      # E = -(eta/2) J_s(t - |u|/v)
        tr_ = t - abs(np.dot(r, un)) / v
        return -(eta / 2) * (math.cos(tr_) * E3[IDX[sp]] + math.sin(tr_) * E3[IDX[sq]])
    hp = rotation(lambda t: Esheet(0.8 * un, t), un)[0]
    hm = rotation(lambda t: Esheet(-0.8 * un, t), -un)[0]
    check(hp in ("rc", "lc") and hm in ("rc", "lc"), "pol: sheet waves should be circular")
    sheet_same = hp == hm
    unequal_lin = classify_phasor(3 * (x - 1j * y) + (x + 1j * y), z) == "lin"
    rtxt = next(f["text"] for f in p["facts"] if f["id"] == "rc_form")
    rm = re.search(r"travel along \$\+\\hat\{([xyz])\}\$, the phasor \$\(\\hat\{([xyz])\} ([+-]) j\\hat\{([xyz])\}\)e\^\{-j\\beta ([xyz])\}\$", rtxt)
    ru, rp, rs, rq, ru2 = rm.groups()
    check(ru == ru2, "pol: rc_form axes")
    rvec = E3[IDX[rp]] + (1j if rs == "+" else -1j) * E3[IDX[rq]]
    rc_form = classify_phasor(rvec, E3[IDX[ru]]) == "rc"
    truth = {"factor": factor_ok, "H_same": H_same, "lin_sum": lin_ok, "twice": twice, "clockwise": cw,
             "same_fn": same_fn, "quarter_any": quarter_any, "reverse": reverse_keeps, "half_period": half_period,
             "sheet_same": sheet_same, "unequal_lin": unequal_lin, "rc_form": rc_form}
    expect = {"factor": True, "H_same": True, "lin_sum": True, "twice": True, "clockwise": True, "same_fn": True,
              "quarter_any": False, "reverse": False, "half_period": False, "sheet_same": False, "unequal_lin": False}
    for k, val in expect.items():
        check(bool(truth[k]) == val, f"pol: computed truth of {k} is {truth[k]}, expected {val} (self-test)")
    facts = p["facts"]
    check({f["id"] for f in facts} == set(truth) and len(facts) == len(truth), "pol: statement ids")
    for fct in facts:
        check((fct["correct"] == "true") == bool(truth[fct["id"]]), f"pol: statement {fct['id']} key")
        check(fct["verdict"] == fct["correct"], "pol: verdict text")
