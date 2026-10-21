"""Independent check for transmission-lines/pulse-snapshot-record (Lecture 29).

The circuit and the source pulse are parsed from the displayed statement. The line is simulated exactly by the
method of characteristics (dt = 1/48 us, one cell per step). At z = 0 the outgoing wave is solved from KVL through
R_g, at z = ell from the load law (V = R_L I, I = 0 or V = 0), each together with V = f + g and Z0 I = f - g;
no reflection or injection coefficient is used. Every snapshot choice is turned into a predicted profile V(z) and
compared with the simulated snapshot; (b)-(d) are read off the simulation; every checkbox statement is re-decided
from simulations (orientation of pulses on snapshots and in probe records, lengths, signs, absorption, scaling).
"""
import re

import numpy as np

DT = 1.0 / 48            # us


def fnum(s):
    return float(s)


class Line:
    def __init__(self, Z0, Rg, RL, src, N, nsteps):
        self.Z0, self.N = Z0, N
        Ag = np.linalg.inv(np.array([[1.0, Rg], [1.0, -Z0]]))                  # V + Rg I = Vs, V - Z0 I = 2 g_in
        if RL is None:
            Al = np.linalg.inv(np.array([[1.0, Z0], [0.0, 1.0]]))              # V + Z0 I = 2 f_in, I = 0
        else:
            Al = np.linalg.inv(np.array([[1.0, Z0], [1.0, -RL]]))              # V + Z0 I = 2 f_in, V = RL I
        self.f0 = np.zeros(nsteps + 1)
        self.gN = np.zeros(nsteps + 1)
        for n in range(nsteps + 1):
            g_in = self.gN[n - N] if n >= N else 0.0
            V0, I0 = Ag @ np.array([src(n * DT), 2 * g_in])
            self.f0[n] = (V0 + Z0 * I0) / 2
            f_in = self.f0[n - N] if n >= N else 0.0
            VL, IL = Al @ np.array([2 * f_in, 0.0])
            self.gN[n] = (VL - Z0 * IL) / 2

    def F(self, k, n):
        return self.f0[n - k] if n >= k else 0.0

    def G(self, k, n):
        m = n - (self.N - k)
        return self.gN[m] if m >= 0 else 0.0

    def V(self, k, n):
        return self.F(k, n) + self.G(k, n)

    def I(self, k, n):
        return (self.F(k, n) - self.G(k, n)) / self.Z0


def parse_choice(text):
    t = text.replace(" ", "")
    m = re.fullmatch(r"asingletriangularpulsewithitspeak\$V=(-?[\d.]+)\\\\mathrm\{V\}\$at\$z=([\d.]+)\\\\mathrm\{m\}\$,"
                     r"nonzeroonlyfor\$([\d.]+)\\\\mathrm\{m\}\\ltz\\lt([\d.]+)\\\\mathrm\{m\}\$", t)
    if m:
        return ("tri", float(m.group(1)), float(m.group(3)), float(m.group(4)), float(m.group(2)))
    m = re.fullmatch(r"asinglerectangularpulse:\$V=(-?[\d.]+)\\\\mathrm\{V\}\$for\$([\d.]+)\\\\mathrm\{m\}\\ltz\\lt([\d.]+)\\\\mathrm\{m\}\$,zeroelsewhere", t)
    if m:
        return ("rect", float(m.group(1)), float(m.group(2)), float(m.group(3)), None)
    if text.startswith("$V = 0$ everywhere on the line"):
        return ("zero", 0.0, None, None, None)
    return None


def profile(desc, z):
    kind, val, za, zb, zpk = desc
    if kind == "zero":
        return np.zeros_like(z), np.ones_like(z, dtype=bool)
    out = np.zeros_like(z)
    inside = (z > za) & (z < zb)
    if kind == "rect":
        out[inside] = val
        use = (np.abs(z - za) > 1e-6) & (np.abs(z - zb) > 1e-6)
        return out, use
    up = inside & (z <= zpk)
    dn = inside & (z > zpk)
    out[up] = val * (z[up] - za) / (zpk - za)
    out[dn] = val * (zb - z[dn]) / (zb - zpk)
    return out, np.ones_like(z, dtype=bool)


def orientation(vals, zs):
    """For a single triangular bump: (peak - left end)/(right end - left end)."""
    nz = np.nonzero(np.abs(vals) > 1e-12 * np.max(np.abs(vals)))[0]
    a, b = zs[nz[0] - 1], zs[nz[-1] + 1]
    pk = zs[np.argmax(np.abs(vals))]
    return (pk - a) / (b - a), b - a


def phys(d, check):
    p, c = d["params"], d["correct_answers"]
    Z0, v, ell = float(p["Z0"]), float(p["v"]), float(p["ell"])
    if p["Rg_txt"].startswith("An ideal source"):
        Rg = 0.0
    else:
        Rg = float(re.search(r"R_g = ([\d.]+)\\ \\Omega", p["Rg_txt"]).group(1))
    if p["load_txt"] == "left open":
        RL = None
    elif p["load_txt"] == "short-circuited":
        RL = 0.0
    else:
        RL = float(re.search(r"R_L = ([\d.]+)\\ \\Omega", p["load_txt"]).group(1))
    check(Rg == p["Rg"] and (RL is None) == (p["RL"] == -1) and (RL is None or RL == p["RL"]), "psr: parsed circuit vs params")
    fi = p["fi_txt"]
    m = re.fullmatch(r"a rectangular pulse: \$f_i\(t\) = (\d+)\\ \\mathrm\{V\}\$ for \$0 \\lt t \\lt ([\d.]+)\\ \\mu\\mathrm\{s\}\$ and \$f_i\(t\) = 0\$ otherwise", fi)
    if m:
        A, w, tp = float(m.group(1)), float(m.group(2)), None
        src = lambda t: A if 0 <= t < w - 1e-12 else 0.0
    else:
        m = re.fullmatch(r"a triangular pulse that rises linearly from \$0\$ at \$t = 0\$ to its peak \$(\d+)\\ \\mathrm\{V\}\$ at \$t = ([\d.]+)\\ \\mu\\mathrm\{s\}\$, "
                         r"then falls linearly to \$0\$ at \$t = ([\d.]+)\\ \\mu\\mathrm\{s\}\$; \$f_i\(t\) = 0\$ at all other times", fi)
        check(m is not None, "psr: cannot parse the source pulse")
        A, tp, w = float(m.group(1)), float(m.group(2)), float(m.group(3))
        src = lambda t: float(np.interp(t, [0, tp, w], [0, A, 0], left=0, right=0))
    check(A == p["A"] and w == p["w"] and (tp is None) == p["rect"], "psr: pulse text vs params")
    T = ell / v
    N = int(round(T / DT))
    check(abs(N * DT - T) < 1e-12, "psr: T not on the grid")
    check(w <= T / 2, "psr: pulse too long for separate copies")
    nsteps = int(round(4 * T / DT)) + 200
    L = Line(Z0, Rg, RL, src, N, nsteps)
    zs = np.arange(N + 1) * DT * v

    def kz(z):
        k = z / (DT * v)
        check(abs(k - round(k)) < 1e-9 and 0 <= round(k) <= N, f"psr: z = {z} off the grid")
        return int(round(k))

    def nt(t):
        n = t / DT
        check(abs(n - round(n)) < 1e-9, f"psr: t = {t} off the grid")
        return int(round(n))

    # (a) snapshot
    n0 = nt(fnum(p["t0_s"]))
    snapV = np.array([L.V(k, n0) for k in range(N + 1)])
    snapF = np.array([L.F(k, n0) for k in range(N + 1)])
    snapG = np.array([L.G(k, n0) for k in range(N + 1)])
    check(np.max(np.abs(snapV)) > 0, "psr: empty snapshot")
    if p["reg"] == "echo":
        check(np.max(np.abs(snapF)) == 0, "psr: echo regime but a forward wave is on the line")
    else:
        check(np.max(np.abs(snapG)) == 0, "psr: second regime but a backward wave is on the line")
    scale = np.max(np.abs(snapV))
    matches = []
    for ch in p["snap"]:
        desc = parse_choice(ch["text"])
        check(desc is not None, f"psr: cannot parse choice {ch['text'][:60]}")
        if desc is None:
            continue
        pred, use = profile(desc, zs)
        ok = np.max(np.abs(pred[use] - snapV[use])) < 2e-3 * scale
        matches.append(ok)
        check(ok == (ch["correct"] == "true"), f"psr: snapshot choice {ch['id']} key wrong (matches={ok})")
    check(sum(matches) == 1, "psr: snapshot needs exactly one matching choice")
    check(len(p["snap"]) == 5, "psr: snapshot needs 5 choices")

    # (b), (d) overlap point
    k1, n1 = kz(fnum(p["z1_s"])), nt(fnum(p["t1_s"]))
    check(L.F(k1, n1) != 0 and L.G(k1, n1) != 0, "psr: (z1, t1) is not in an overlap")
    Vs, Is = L.V(k1, n1), L.I(k1, n1) * 1e3
    check(abs(Vs - c["Vov"]) < 1e-9 * A, f"psr: V overlap {Vs} vs {c['Vov']}")
    check(abs(Is - c["Iov"]) < 1e-9 * A / Z0 * 1e3, f"psr: I overlap {Is} vs {c['Iov']}")
    npk = int(round((tp if tp is not None else w / 2) / DT))
    tg_sim = L.F(0, npk) / src(npk * DT)                 # launched / open-circuit, before any echo
    check(c["Vov"] == 0 or abs(c["Vov"]) >= 0.0199 * abs(tg_sim) * A, "psr: tiny V overlap")
    check(c["Iov"] == 0 or abs(c["Iov"]) >= 0.0199 * abs(tg_sim) * A / Z0 * 1e3, "psr: tiny I overlap")
    # tolerance: the two wave components at (z1, t1) give the 3-s.f. rounding noise
    def r3(x):
        return float(f"{x:.3g}")
    fw, bw = L.F(k1, n1), L.G(k1, n1)
    for key, akey, terms, floor in (("Vov", "atolV", [fw, bw], 2e-4 * A), ("Iov", "atolI", [fw / Z0 * 1e3, -bw / Z0 * 1e3], 2e-4 * A / Z0 * 1e3)):
        a = float(p[akey])
        expect = max(floor, 0.005 * sum(abs(x) for x in terms))
        check(abs(a - expect) <= 0.06 * expect, f"psr: {akey} {a} vs expected {expect}")
        tol = a + 0.01 * abs(c[key])
        check(abs(sum(r3(x) for x in terms) - c[key]) <= tol, f"psr: 3-s.f. chain rejected for {key}")
        check(all(abs(x) > 1.2 * tol for x in terms), f"psr: dropping one wave of {key} would be accepted")

    # (c) echo's peak / trailing edge at the probe, from the backward wave alone (t < 3T: first echo only)
    kp = kz(fnum(p["zp_s"]))
    nmax = int(round(3 * T / DT))
    gp = np.array([L.G(kp, n) for n in range(nmax + 1)])
    nzr = np.nonzero(np.abs(gp) > 1e-12 * A)[0]
    check(len(nzr) > 0, "psr: no echo at the probe")
    if tp is None:
        t_c = (nzr[-1] + 1) * DT
    else:
        t_c = np.argmax(np.abs(gp)) * DT
    check(abs(t_c - c["tpk"]) < 1e-9, f"psr: echo time {t_c} vs {c['tpk']}")

    # (e) statements
    truth = {}
    ne = nt((3 * T + w) / 2)                               # echo alone, wholly on the line
    check(max(abs(L.F(k, ne)) for k in range(N + 1)) == 0, "psr: forward wave present at the echo-alone time")
    # probe record order (triangle only can show it): fraction of the rise in the echo record at kp
    if tp is not None:
        frac, _ = orientation(gp, np.arange(nmax + 1) * DT)
        truth["probe_order"] = abs(frac - tp / w) < 1e-6
        truth["probe_rev"] = abs(frac - (1 - tp / w)) < 1e-6
        # snapshots: forward pulse alone at t in (w, T); echo alone at t in (T + w, 2T)
        nf = nt((w + T) / 2)
        fr_f, len_f = orientation(np.array([L.V(k, nf) for k in range(N + 1)]), zs)
        fr_e, len_e = orientation(np.array([L.V(k, ne) for k in range(N + 1)]), zs)
        truth["snap_fwd"] = abs((1 - fr_f) - tp / w) < 1e-6           # first-sent part farthest from z = 0
        truth["snap_echo"] = abs((1 - fr_e) - tp / w) < 1e-6          # would mean the echo is reversed
        truth["length"] = abs(len_e - len_f) < 1e-9 and abs(len_e - v * w) < 1e-9
        GLsim = L.G(N, N + int(round(tp / DT))) / L.F(N, N + int(round(tp / DT)))
        if "shrink" in {x["id"] for x in p["facts"]}:
            truth["shrink"] = abs(len_e - abs(GLsim) * len_f) < 1e-9
    else:
        row = np.array([L.V(k, ne) for k in range(N + 1)])
        nzz = np.nonzero(np.abs(row) > 1e-12)[0]
        len_e = (nzz[-1] - nzz[0] + 1) * DT * v
        check(abs(len_e - v * w) < 1e-9, "psr: rect echo length")
        GLsim = L.G(N, N + 2) / L.F(N, N + 2)
        truth.update({"probe_order": True, "probe_rev": False, "snap_fwd": True, "snap_echo": False, "length": True})
        if "shrink" in {x["id"] for x in p["facts"]}:
            truth["shrink"] = abs(abs(GLsim) - 1) < 1e-9
    VovI = L.I(k1, n1)
    truth["overlapZ0"] = VovI != 0 and abs(L.V(k1, n1) / VovI - Z0) < 1e-9 * Z0
    ns = range(N, N + int(round(w / DT)) + 2)
    if RL is None:
        truth["load_ohm"] = all(abs(L.I(N, n)) < 1e-12 for n in ns)
    else:
        truth["load_ohm"] = all(abs(L.V(N, n) - RL * L.I(N, n)) < 1e-9 * A for n in ns)
    # echo alone: sign of its current against its voltage (this circuit, and one with Gamma_L > 0)
    def echo_sign(LL):
        row = np.array([LL.V(k, ne) for k in range(N + 1)])
        km = int(np.argmax(np.abs(row)))
        return LL.V(km, ne) * LL.I(km, ne)
    truth["echoI"] = echo_sign(L) < 0
    L3 = Line(Z0, Rg, 3 * Z0, src, N, ne + 2)
    check(L3.G(N, N + npk) / L3.F(N, N + npk) > 0, "psr: test load does not give Gamma_L > 0")
    truth["echoI_same"] = echo_sign(L3) > 0
    # absorption at z = 0: forward wave launched while the echo arrives?
    relaunch = max(abs(L.f0[n]) for n in range(2 * N, 2 * N + int(round(w / DT)) + 2))
    truth["absorbed"] = relaunch < 1e-12 * A
    L4 = Line(Z0, Rg, Z0, src, N, ne + 2)
    truth["matched_load"] = max(abs(L4.G(k, n)) for k in range(N + 1) for n in range(N, ne + 1, 4)) < 1e-12 * A
    L2 = Line(Z0, Rg, RL, lambda t: 2 * src(t), N, 2 * N + 50)
    nr = N + int(round((tp if tp is not None else w / 2) / DT))
    truth["amp_GL"] = abs(L2.G(N, nr) / L2.F(N, nr) - 2 * GLsim) < 1e-9
    a1 = next(n for n in range(4 * N) if abs(L.G(kz(ell / 4), n)) > 1e-12 * A)
    a3 = next(n for n in range(4 * N) if abs(L.G(kz(3 * ell / 4), n)) > 1e-12 * A)
    truth["closer"] = a1 > a3
    for x in p["facts"]:
        check(x["id"] in truth, f"psr: statement {x['id']} not re-decided")
        if x["id"] in truth:
            check((x["correct"] == "true") == bool(truth[x["id"]]), f"psr: statement {x['id']} keyed {x['correct']}, computed {truth[x['id']]}")
    nt_ = sum(x["correct"] == "true" for x in p["facts"])
    nf_ = len(p["facts"]) - nt_
    check(nt_ >= 4 and nf_ >= 4, "psr: pool cannot fill 5 with 1-4 correct")
    # panel strings
    for s, k in (("Vb_s", "Vov"), ("Id_s", "Iov"), ("tc_s", "tpk")):
        check(abs(float(p[s]) - c[k]) <= 5e-4 * max(1, abs(c[k])), f"psr: panel {s} vs {k}")
