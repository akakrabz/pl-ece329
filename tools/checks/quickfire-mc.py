"""Independent check for midterm2/quickfire-mc (six items from twelve generators).

For every item the key is recomputed from the item's parameters (and, where possible, from numbers parsed out of the
stem) by a route different from server.py, and the choice marked correct is identified by parsing the choice texts:
  wires      Biot-Savart sum over a long discretized wire 1, force I2 dl x B with np.cross, before and after
  loop       flux of dB/dt through the loop by 2-D quadrature over its area
  magnet     point-dipole field integrated over the ring's disk at successive magnet positions; Lenz from dPsi/dt
  dcond      divergence of each displayed D by finite differences at two times
  potentials E = -grad Phi - dA/dt and B = curl A by finite differences; each displayed (E, B) evaluated and compared
  sheetH     n x (H1 - H2) = J_s and H_n continuity checked with np.cross on the parsed vectors
  phase      each offered answer converted back into a phase change and compared with the stated one
  drude      Drude equation of motion time-stepped to steady state for both samples; sigma = N q v / E
  magcur     curl of a smoothed M across the face, integrated through the surface layer
  disp       numerical time derivative of D(t) over a period
  etav       eta = sqrt(mu/eps), v = 1/sqrt(mu eps) from mu0, eps0
  poynting   np.cross of the parsed E and H (or u x E)
"""
import math
import re
from fractions import Fraction

import numpy as np

MU0 = 4e-7 * math.pi
EPS0 = 8.8541878128e-12
C0 = 1 / math.sqrt(MU0 * EPS0)
ETA0 = math.sqrt(MU0 / EPS0)
IDX = {"x": 0, "y": 1, "z": 2}
E3 = np.eye(3)


# ------------------------------------------------------------------ parsers
def parse_num(tex):
    t = tex.strip().replace(" ", "")
    m = re.fullmatch(r"(-?[\d.]+)(?:\\times10\^\{(-?\d+)\})?", t)
    return float(m.group(1)) * 10 ** int(m.group(2) or 0)


def parse_frac(tex):
    t = tex.strip()
    m = re.fullmatch(r"(-?)\\tfrac\{(\d+)\}\{(\d+)\}", t)
    if m:
        return Fraction(int(m.group(2)), int(m.group(3))) * (-1 if m.group(1) else 1)
    return Fraction(int(t))


def parse_terms(tex):
    """'2\\,t\\,y\\,\\hat{x} - x^{2}\\,\\hat{z}' -> list of (coef, axis, {var: power})."""
    tex = tex.strip()
    if tex == r"\mathbf{0}":
        return []
    out = []
    for tok in re.findall(r"[+-]?\s*[^+-]+", tex):
        tok = tok.replace(" ", "")
        sgn = -1.0 if tok.startswith("-") else 1.0
        tok = tok.lstrip("+-")
        m = re.fullmatch(r"(.*?)\\hat\{([xyz])\}", tok)
        assert m, tok
        body, ax = m.group(1), m.group(2)
        coef, pw = 1.0, {}
        for f in [f for f in body.split("\\,") if f]:
            mm = re.fullmatch(r"([txyz])(?:\^\{(\d+)\})?", f)
            if mm:
                pw[mm.group(1)] = pw.get(mm.group(1), 0) + int(mm.group(2) or 1)
            else:
                coef *= float(f)
        out.append((sgn * coef, ax, pw))
    return out


def eval_terms(terms, r, t):
    v = np.zeros(3)
    env = {"x": r[0], "y": r[1], "z": r[2], "t": t}
    for k, ax, pw in terms:
        val = k
        for var, n in pw.items():
            val *= env[var] ** n
        v[IDX[ax]] += val
    return v


def parse_vec(tex):
    """'-4\\,\\hat{x} + 7\\,\\hat{y}\\ \\mathrm{A/m}' or '\\mathbf{0}' -> np.array."""
    t = re.sub(r"\\ \\mathrm\{[^}]*\}", "", tex).strip()
    v = np.zeros(3)
    for k, ax, pw in parse_terms(t):
        assert not pw, tex
        v[IDX[ax]] += k
    return v


def one_correct(it, check):
    chs = it["choices"]
    check(len(chs) == 4 and len({c["text"] for c in chs}) == 4, f"quickfire {it['gen']}: need 4 distinct choices")
    check(sum(c["correct"] == "true" for c in chs) == 1, f"quickfire {it['gen']}: exactly one correct")
    check(all(c.get("feedback") for c in chs), f"quickfire {it['gen']}: every choice needs feedback")
    return [c for c in chs if c["correct"] == "true"][0]


# ------------------------------------------------------------------ per-generator checks
def chk_wires(it, check):
    dd = it["data"]
    a, b, c = (Fraction(dd[k]) for k in ("a", "b", "c"))
    ax = E3[IDX[dd["ax"]]]
    perp = E3[(IDX[dd["ax"]] + 1) % 3]

    def force(I1, I2, d):
        L = 4000 * d
        u = np.sinh(np.linspace(-np.arcsinh(L / d), np.arcsinh(L / d), 40001)) * d
        mid, dl = 0.5 * (u[1:] + u[:-1]), np.diff(u)
        src = mid[:, None] * ax[None, :]
        R = d * perp[None, :] - src
        Rn = np.linalg.norm(R, axis=1)[:, None]
        B = np.sum(MU0 / (4 * math.pi) * I1 * np.cross(np.broadcast_to(ax, R.shape), R) * dl[:, None] / Rn ** 3, axis=0)
        return I2 * np.cross(ax, B)                      # force per unit length on wire 2
    s0 = 1 if dd["same0"] else -1
    d0 = dd["d_cm"] * 1e-2
    f0 = force(1.0, float(s0), d0)
    f1 = force(float(a), float(b) * s0 * (-1 if dd["rev2"] else 1), float(c) * d0)
    ratio = np.linalg.norm(f1) / np.linalg.norm(f0)
    attract = np.dot(f1, -perp) > 0                      # toward wire 1 (at the origin)
    good = one_correct(it, check)
    m = re.fullmatch(r"\$(.*)F_0\$, (attractive|repulsive)", good["text"])
    check(m is not None, "quickfire wires: cannot parse key")
    k = parse_frac(m.group(1)) if m.group(1) else Fraction(1)
    check(abs(float(k) - ratio) < 1e-5 * ratio, f"quickfire wires: factor {k} vs Biot-Savart {ratio}")
    check((m.group(2) == "attractive") == attract, "quickfire wires: attract/repel")
    st = it["stem"]
    check(f"${dd['d_cm']}\\ \\mathrm{{cm}}$ apart" in st and f"changed to ${float(c) * dd['d_cm']:g}\\ \\mathrm{{cm}}$" in st,
          "quickfire wires: separations in the stem")
    check(("in the same direction" in st) == dd["same0"] and ("reversed" in st) == dd["rev2"], "quickfire wires: stem wording")


def chk_loop(it, check):
    dd = it["data"]
    n = E3[IDX[dd["nax"]]]
    dB = np.array(dd["dBdt"], float)
    vm = re.search(r"d\\mathbf\{B\}/dt = \((.*?)\)\\ \\mathrm\{T/s\}", it["stem"])
    check(vm is not None and np.allclose(parse_vec(vm.group(1)), dB), "quickfire loop: dB/dt in stem")
    plane = "".join(sorted(k for k in "xyz" if k != dd["nax"]))
    check(f"${plane}$ plane" in it["stem"], "quickfire loop: plane in stem")
    check(f"R = {dd['R']:g}" in it["stem"], "quickfire loop: R in stem")
    N = 400
    if dd["shape"] == "square":
        check(f"side ${dd['size_cm']}\\ \\mathrm{{cm}}$" in it["stem"], "quickfire loop: size in stem")
        s = dd["size_cm"] * 1e-2
        cells = np.full((N, N), (s / N) ** 2)                          # midpoint grid over the square
    else:
        check(f"radius ${dd['size_cm']}\\ \\mathrm{{cm}}$" in it["stem"], "quickfire loop: size in stem")
        rr = dd["size_cm"] * 1e-2
        rho = (np.arange(N) + 0.5) / N * rr                             # polar midpoint grid over the disk
        cells = np.outer(rho * (rr / N), np.full(N, 2 * math.pi / N))
    flux_rate = float(np.sum(cells * np.dot(dB, n)))                    # integral of (dB/dt).n dS (uniform integrand)
    I_mA = abs(flux_rate) / dd["R"] * 1e3
    good = one_correct(it, check)
    val = parse_num(re.fullmatch(r"\$(.*)\\ \\mathrm\{mA\}\$", good["text"]).group(1))
    check(abs(val - I_mA) < 6e-3 * I_mA, f"quickfire loop: {val} vs quadrature {I_mA}")
    for c in it["choices"]:
        if c is not good:
            v2 = parse_num(re.fullmatch(r"\$(.*)\\ \\mathrm\{mA\}\$", c["text"]).group(1))
            check(abs(v2 - I_mA) > 0.1 * I_mA, "quickfire loop: a distractor is within 10 % of the key")


def chk_magnet(it, check):
    dd = it["data"]
    # moment direction: S -> N inside the magnet; the leading pole points along the motion
    lead_end_up = dd["motion"] == "up"
    N_up = lead_end_up if dd["lead"] == "N" else not lead_end_up
    m = np.array([0, 0, 1.0 if N_up else -1.0])
    a = 1.0
    rho = (np.arange(400) + 0.5) / 400 * a
    phi = (np.arange(64) + 0.5) / 64 * 2 * math.pi
    P, Q = np.meshgrid(rho, phi)
    pts = np.stack([P * np.cos(Q), P * np.sin(Q), 0 * P], axis=-1)
    dS = (a / 400) * (2 * math.pi / 64) * P

    def flux(zm):
        r = pts - np.array([0, 0, zm])
        rn = np.linalg.norm(r, axis=-1)[..., None]
        B = MU0 / (4 * math.pi) * (3 * (r @ m)[..., None] * r / rn ** 5 - m / rn ** 3)
        return float(np.sum(B[..., 2] * dS))
    path = [3.0, 2.5, 2.0] if dd["motion"] == "down" else [-3.0, -2.5, -2.0]       # approaching
    after = [-2.0, -2.5, -3.0] if dd["motion"] == "down" else [2.0, 2.5, 3.0]      # leaving
    dpsi_app = flux(path[-1]) - flux(path[0])
    dpsi_lv = flux(after[-1]) - flux(after[0])
    ccw_above = [-np.sign(dpsi_app), -np.sign(dpsi_lv)]      # E = -dPsi/dt, circulation ccw from above <-> dS = +z
    flip = 1 if dd["view"] == "above" else -1
    words = ["counterclockwise" if s * flip > 0 else "clockwise" for s in ccw_above]
    expect = f"{words[0]}, then {words[1]}" if words[0] != words[1] else f"{words[0]} both times"
    good = one_correct(it, check)
    check(good["text"] == expect, f"quickfire magnet: key {good['text']!r}, dipole model says {expect!r}")
    check(f"Seen from {dd['view']}" in it["stem"] and ("dropped" in it["stem"]) == (dd["motion"] == "down"), "quickfire magnet: stem")


def chk_dcond(it, check):
    rng = np.random.default_rng(7)
    allowed = []
    for c in it["choices"]:
        m = re.fullmatch(r"\$\\mathbf\{D\} = D_0\\left\((.*)\\right\)\$", c["text"])
        check(m is not None, "quickfire dcond: parse")
        terms = parse_terms(m.group(1))
        ok = True
        for _ in range(3):
            r = rng.uniform(-2, 2, 3)
            h = 1e-4

            def div(t):
                return sum((eval_terms(terms, r + h * E3[j], t)[j] - eval_terms(terms, r - h * E3[j], t)[j]) / (2 * h) for j in range(3))
            for t in (0.7, 1.9):
                ddt = (div(t + 1e-3) - div(t - 1e-3)) / 2e-3
                if abs(ddt) > 1e-6:
                    ok = False
        allowed.append(ok)
    check(sum(allowed) == 1, f"quickfire dcond: {sum(allowed)} fields have time-independent divergence")
    good = one_correct(it, check)
    check(allowed[it["choices"].index(good)], "quickfire dcond: the key's divergence changes in time")


def chk_potentials(it, check):
    dd = it["data"]
    al, be, x1, p, q = dd["alpha"], dd["beta"], dd["x1"], dd["p"], dd["q"]
    check(f"\\Phi = {al}\\,{x1}^{{2}}" in it["stem"], "quickfire potentials: Phi in stem")
    mA = re.search(r"\\mathbf\{A\} = (.*?)\\ \\mathrm\{Wb/m\}", it["stem"])
    Aterms = parse_terms(mA.group(1))

    def Phi(r):
        return al * r[IDX[x1]] ** 2

    def A(r, t):
        return eval_terms(Aterms, r, t)
    check(np.allclose(A(np.array([0.3, -1.1, 0.7]), 0.9), be * 0.9 * np.array([0.3, -1.1, 0.7])[IDX[q]] * E3[IDX[p]]), "quickfire potentials: A in stem vs data")
    h = 1e-5
    good = one_correct(it, check)
    matches = []
    for c in it["choices"]:
        m = re.fullmatch(r"\$\\mathbf\{E\} = (.*)\\ \\mathrm\{V/m\},\\quad \\mathbf\{B\} = (.*)\\ \\mathrm\{T\}\$", c["text"])
        check(m is not None, "quickfire potentials: parse choice")
        Et, Bt = parse_terms(m.group(1)), parse_terms(m.group(2))
        ok = True
        for r, t in ((np.array([0.4, -0.3, 1.2]), 0.5), (np.array([-1.1, 0.8, 0.2]), 2.0)):
            gradPhi = np.array([(Phi(r + h * E3[j]) - Phi(r - h * E3[j])) / (2 * h) for j in range(3)])
            dAdt = (A(r, t + h) - A(r, t - h)) / (2 * h)
            E = -gradPhi - dAdt
            J = np.array([[(A(r + h * E3[j], t)[i] - A(r - h * E3[j], t)[i]) / (2 * h) for j in range(3)] for i in range(3)])
            B = np.array([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]])
            if not (np.allclose(eval_terms(Et, r, t), E, atol=1e-6) and np.allclose(eval_terms(Bt, r, t), B, atol=1e-6)):
                ok = False
        matches.append(ok)
    check(sum(matches) == 1 and matches[it["choices"].index(good)], "quickfire potentials: key does not match the computed E and B")


def chk_sheetH(it, check):
    dd = it["data"]
    n = E3[IDX[dd["w"]]]
    Js = np.array(dd["Js"], float)
    Hg = np.array(dd["Hg"], float)
    mj = re.search(r"\\mathbf\{J\}_s = (.*?)\$", it["stem"])
    mh = re.search(r"\\mathbf\{H\} = (.*?)\$", it["stem"])
    check(np.allclose(parse_vec(mj.group(1)), Js) and np.allclose(parse_vec(mh.group(1)), Hg), "quickfire sheetH: stem vectors")
    good = one_correct(it, check)
    sat = []
    for c in it["choices"]:
        Hc = parse_vec(re.fullmatch(r"\$\\mathbf\{H\} = (.*)\$", c["text"]).group(1))
        H1, H2 = (Hc, Hg) if dd["given"] == "below" else (Hg, Hc)      # 1: w > 0, 2: w < 0, n from 2 into 1
        sat.append(np.allclose(np.cross(n, H1 - H2), Js) and abs(np.dot(H1 - H2, n)) < 1e-12)
    check(sum(sat) == 1 and sat[it["choices"].index(good)], "quickfire sheetH: boundary conditions select a different choice")


def chk_phase(it, check):
    dd = it["data"]
    eps_r = dd["eps_r"]
    v = 1 / math.sqrt(MU0 * EPS0 * eps_r)
    dphi = float(Fraction(dd["a"])) * math.pi
    check(f"\\epsilon_r = {eps_r:g}" in it["stem"], "quickfire phase: eps_r in stem")
    good = one_correct(it, check)
    hits = []
    for c in it["choices"]:
        m = re.fullmatch(r"\$(.*)\\ \\mathrm\{(m|MHz)\}\$", c["text"])
        val = parse_num(m.group(1))
        if dd["mode"] == "time":
            f = v / val                                          # offered wavelength -> frequency
            implied = 2 * math.pi * f * dd["dt"]                 # phase advance at a fixed point in dt
        else:
            lam = v / (val * 1e6)
            implied = 2 * math.pi * dd["dd"] / lam               # phase change over dd at a fixed instant
        hits.append(abs(implied - dphi) < 6e-3 * dphi)
    check(sum(hits) == 1 and hits[it["choices"].index(good)], f"quickfire phase: {sum(hits)} choices reproduce the stated phase change")


def chk_drude(it, check):
    dd = it["data"]
    a, b, c = (Fraction(dd[k]) for k in ("a", "b", "c"))

    def sigma(N, tau, m, q=1.0, E=1.0):
        v, dt = 0.0, tau / 1000
        for _ in range(25000):                    # explicit Euler on m dv/dt = qE - m v/tau, 25 collision times
            v += dt * (q * E / m - v / tau)
        return N * q * v / E
    ratio = sigma(float(a), float(b), float(c)) / sigma(1.0, 1.0, 1.0)
    good = one_correct(it, check)
    k = parse_frac(good["text"].strip("$"))
    check(abs(float(k) - ratio) < 1e-6 * ratio, f"quickfire drude: key {k} vs time-stepped {ratio}")
    others = [parse_frac(cc["text"].strip("$")) for cc in it["choices"] if cc is not good]
    check(all(abs(float(o) - ratio) > 1e-3 * ratio for o in others), "quickfire drude: a distractor equals the key")


def chk_magcur(it, check):
    dd = it["data"]
    M = dd["esg"] * dd["M0"] * E3[IDX[dd["eax"]]]
    n = dd["nsg"] * E3[IDX[dd["nax"]]]
    check(np.allclose(parse_vec(re.search(r"\\mathbf\{M\} = (.*?)\$", it["stem"]).group(1)), M), "quickfire magcur: M in stem")
    w = 1e-3

    def Mf(r):
        return M * 0.5 * (1 - np.tanh(np.dot(r, n) / w))     # magnetized for r.n < 0 (inside), zero outside
    h = w / 50
    K = np.zeros(3)
    ss = np.linspace(-15 * w, 15 * w, 301)
    r0 = 0.37 * E3[(IDX[dd["nax"]] + 1) % 3] + 0.21 * E3[(IDX[dd["nax"]] + 2) % 3]
    for i in range(len(ss) - 1):
        r = r0 + 0.5 * (ss[i] + ss[i + 1]) * n
        J = np.array([[(Mf(r + h * E3[j])[k] - Mf(r - h * E3[j])[k]) / (2 * h) for j in range(3)] for k in range(3)])
        K += np.array([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]]) * (ss[i + 1] - ss[i])
    good = one_correct(it, check)
    t = good["text"]
    Kc = np.zeros(3) if t.startswith(r"$\mathbf{J}_{sM} = \mathbf{0}") else parse_vec(re.fullmatch(r"\$\\mathbf\{J\}_\{sM\} = (.*)\$", t).group(1))
    check(np.allclose(Kc, K, atol=1e-3 * dd["M0"]), f"quickfire magcur: key {Kc} vs numerical curl {K}")


def chk_disp(it, check):
    dd = it["data"]
    om = 2 * math.pi * dd["f_MHz"] * 1e6
    tt = np.linspace(0, 2 * math.pi / om, 20001)
    D = EPS0 * dd["eps_r"] * dd["E0"] * np.cos(om * tt)
    dDdt = np.gradient(D, tt)
    amp = float(np.max(np.abs(dDdt)))                       # A/m^2

    def val_of(text):
        m = re.fullmatch(r"\$(.*)\\ \\mathrm\{(m?)A/m\^2\}\$", text)
        return parse_num(m.group(1)) * (1e-3 if m.group(2) else 1.0)
    good = one_correct(it, check)
    units = {re.search(r"\\mathrm\{(m?A/m\^2)\}", c["text"]).group(1) for c in it["choices"]}
    check(len(units) == 1, "quickfire disp: mixed units among the choices")
    val = val_of(good["text"])
    check(abs(val - amp) < 6e-3 * amp, f"quickfire disp: {val} vs numerical {amp}")
    for c in it["choices"]:
        if c is not good:
            check(abs(val_of(c["text"]) - amp) > 0.1 * amp, "quickfire disp: distractor within 10 %")


def chk_etav(it, check):
    dd = it["data"]
    mu, eps = MU0 * dd["mur"], EPS0 * dd["epsr"]
    eta, v = math.sqrt(mu / eps), 1 / math.sqrt(mu * eps)
    good = one_correct(it, check)
    ok = []
    for c in it["choices"]:
        m = re.fullmatch(r"\$\\eta = (.*)\\ \\Omega,\\quad v = (.*)\\ \\mathrm\{m/s\}\$", c["text"])
        e2, v2 = parse_num(m.group(1)), parse_num(m.group(2))
        ok.append(abs(e2 - eta) < 6e-3 * eta and abs(v2 - v) < 6e-3 * v)
    check(sum(ok) == 1 and ok[it["choices"].index(good)], "quickfire etav: key does not match sqrt(mu/eps), 1/sqrt(mu eps)")


def parse_unit(tex):
    m = re.match(r"\$([+-])\\hat\{([xyz])\}\$", tex)
    return (1 if m.group(1) == "+" else -1) * E3[IDX[m.group(2)]]


def chk_poynting(it, check):
    dd = it["data"]
    good = one_correct(it, check)
    if dd["sub"] == "travel":
        Ev = parse_vec(re.search(r"\\mathbf\{E\} = (.*?)\$", it["stem"]).group(1))
        mh = re.search(r"\\mathbf\{H\} = (-?)([\d.]+)\\,\\hat\{([xyz])\}\\ \\mathrm\{mA/m\}", it["stem"])
        Hv = (-1 if mh.group(1) else 1) * float(mh.group(2)) * 1e-3 * E3[IDX[mh.group(3)]]
        check(abs(np.linalg.norm(Ev) / np.linalg.norm(Hv) - ETA0) < 0.01 * ETA0, "quickfire poynting: |E|/|H| is not eta0")
        S = np.cross(Ev, Hv)
        check(np.allclose(parse_unit(good["text"]), S / np.linalg.norm(S)), "quickfire poynting: travel direction")
    else:
        mu_ = re.search(r"travels toward \$([+-])\\hat\{([xyz])\}\$", it["stem"])
        u = (1 if mu_.group(1) == "+" else -1) * E3[IDX[mu_.group(2)]]
        Ev = parse_vec(re.search(r"\\mathbf\{E\} = (.*?)\$", it["stem"]).group(1))
        H = np.cross(u, Ev) / ETA0
        check(np.allclose(parse_unit(good["text"]), H / np.linalg.norm(H)), "quickfire poynting: H direction")
        check(np.allclose(np.cross(Ev, H) / np.linalg.norm(np.cross(Ev, H)), u), "quickfire poynting: E x H along u")


CHECKS = {"wires": chk_wires, "loop": chk_loop, "magnet": chk_magnet, "dcond": chk_dcond, "potentials": chk_potentials,
          "sheetH": chk_sheetH, "phase": chk_phase, "drude": chk_drude, "magcur": chk_magcur, "disp": chk_disp,
          "etav": chk_etav, "poynting": chk_poynting}


def phys(d, check):
    items = d["params"]["items"]
    check(len(items) == 6, "quickfire: need six items")
    check([it["name"] for it in items] == [f"q{i}" for i in range(1, 7)], "quickfire: answers-names q1..q6")
    check([it["num"] for it in items] == list(range(1, 7)), "quickfire: numbering")
    check(len({it["gen"] for it in items}) == 6, "quickfire: generators repeat within a variant")
    for it in items:
        check(it["gen"] in CHECKS, f"quickfire: unknown generator {it['gen']}")
        check(bool(it.get("solution")) and bool(it.get("stem")), "quickfire: stem/solution missing")
        CHECKS[it["gen"]](it, check)
