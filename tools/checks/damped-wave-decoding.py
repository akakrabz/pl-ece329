"""Independent check for waves/damped-wave-decoding (Lecture 22).

The given field is rebuilt from the *displayed* LaTeX (time-domain or phasor form) and the displayed omega and mu.
Its phasor is obtained by numerical demodulation (DFT over one period at each point). Then, without the closed forms:
  * alpha, beta and the travel direction from the phasor's log-amplitude and phase gradients (finite differences);
  * the other field from Maxwell's equations by finite differences: H~ = -curl E~/(j w mu) (E given), or
    gamma^2 = lap H~ / H~, sigma + j w eps = gamma^2/(j w mu), E~ = curl H~/(sigma + j w eps) (H given);
    for E given, sigma + j w eps = (curl H~).e/(E~.e);
  * eta as the ratio of the two numerically obtained fields; the direction MC from the real fields at the
    instant the given field peaks along the stated direction; the amplitude at u1 from the phasor there;
  * every checkbox statement re-decided by a computation (time sampling for power and lag, random media for alpha < beta).
"""
import math
import re

import numpy as np

MU0 = 4e-7 * math.pi
EPS0 = 1e-9 / (36 * math.pi)
IDX = {"x": 0, "y": 1, "z": 2}
E3 = np.eye(3)
SQ3 = math.sqrt(3.0)

_T = (r"^\\mathbf\{([EH])\}\(([xyz]),t\) = (-?)\\hat\{([xyz])\}\\,([\d.]+)\\,e\^\{(-?)([\d.]*)([xyz])\}"
      r"\\cos\(([\d.]+)\\times10\^\{(\d+)\}\\,t ([+-]) ([\d.]*(?:\\sqrt\{3\})?)(?:\\,)?([xyz])(?: ([+-]) ([\d.]+)\^\\circ)?\)$")
_P = (r"^\\tilde\{\\mathbf\{([EH])\}\}\(([xyz])\) = (-?)\\hat\{([xyz])\}\\,([\d.]+)(?:\\,e\^\{(-?)j(\d+)\^\\circ\})?"
      r"\\,e\^\{(-?)\(([\d.]+) \+ j([\d.]*(?:\\sqrt\{3\})?)\)([xyz])\}$")


def coef(s):
    """'2\\sqrt{3}' -> 2 sqrt3, '\\sqrt{3}' -> sqrt3, '' -> 1, '1.5' -> 1.5."""
    r3 = s.endswith(r"\sqrt{3}")
    s = s.replace(r"\sqrt{3}", "")
    v = float(s) if s else 1.0
    return v * (SQ3 if r3 else 1.0)


def unit_of(tex):
    m = re.search(r"([+-])\\hat\{([xyz])\}", tex)
    return (1 if m.group(1) == "+" else -1) * E3[IDX[m.group(2)]]


def phys(d, check):
    p, c = d["params"], d["correct_answers"]
    om_m = re.fullmatch(r"([\d.]+)\\times10\^\{(\d+)\}", p["om_tex"])
    om = float(om_m.group(1)) * 10 ** int(om_m.group(2))
    mm = re.fullmatch(r"\\mu = (\d*)\\mu_0", p["mu_tex"])
    mu = (float(mm.group(1)) if mm.group(1) else 1.0) * MU0
    scale = 1e-3 if p["unit_tex"] == r"\mathrm{mA/m}" else 1.0
    tt = p["field_tex"]
    mt, mp = re.fullmatch(_T, tt), re.fullmatch(_P, tt)
    check((mt is not None) != (mp is not None), f"damped: cannot parse field_tex {tt}")
    if mt:
        fld, u, sg, ax, A = mt.group(1), mt.group(2), mt.group(3), mt.group(4), float(mt.group(5))
        ca = (-1 if mt.group(6) == "-" else 1) * (float(mt.group(7)) if mt.group(7) else 1.0)
        check(mt.group(8) == u and mt.group(13) == u, "damped: axis mismatch in field_tex")
        om_t = float(mt.group(9)) * 10 ** int(mt.group(10))
        check(abs(om_t - om) < 1e-6 * om, "damped: omega in the cosine differs from the stated omega")
        cb = (-1 if mt.group(11) == "-" else 1) * coef(mt.group(12))
        ph = 0.0 if mt.group(14) is None else (-1 if mt.group(14) == "-" else 1) * float(mt.group(15))
    else:
        fld, u, sg, ax, A = mp.group(1), mp.group(2), mp.group(3), mp.group(4), float(mp.group(5))
        ph = 0.0 if mp.group(7) is None else (-1 if mp.group(6) == "-" else 1) * float(mp.group(7))
        s = -1 if mp.group(8) == "-" else 1
        ca, cb = s * float(mp.group(9)), s * coef(mp.group(10))
        check(mp.group(11) == u, "damped: axis mismatch in phasor tex")
    check(fld == p["field"], "damped: field letter")
    eF = (-1 if sg == "-" else 1) * E3[IDX[ax]]
    uax = E3[IDX[u]]
    T = 2 * math.pi / om
    N = 32
    tk = np.arange(N) * T / N

    def F_real(r, t):                                   # SI units (V/m or A/m), vector; t may be an array
        uu = float(np.dot(r, uax))
        return np.outer(A * scale * math.exp(ca * uu) * np.cos(om * np.asarray(t, float) + cb * uu + math.radians(ph)), eF)

    def F_ph(r):                                        # numerical demodulation over one period
        return (2.0 / N) * np.sum(F_real(r, tk) * np.exp(-1j * om * tk)[:, None], axis=0)

    r0 = np.array([0.31, -0.17, 0.23]) * (1 - uax)
    # ---- alpha, beta, direction (finite differences of log-amplitude and phase along u)
    hh = 1e-5
    f0, f1 = np.dot(F_ph(r0), eF), np.dot(F_ph(r0 + hh * uax), eF)
    ca_n = math.log(abs(f1) / abs(f0)) / hh
    cb_n = float(np.angle(f1 / f0)) / hh
    alpha, beta = abs(ca_n), abs(cb_n)
    dn = -1 if cb_n > 0 else 1                         # phase decreasing along +u -> travel toward +u
    uhat = dn * uax
    check(ca_n * dn < 0, "damped: the displayed wave grows along its direction of travel")
    check(abs(alpha - c["alpha"]) < 1e-5 * alpha and abs(beta - c["beta"]) < 1e-5 * beta, f"damped: alpha/beta {alpha},{beta} vs {c['alpha']},{c['beta']}")
    lam, dlt = 2 * math.pi / beta, 1 / alpha
    check(abs(lam - c["lam"]) < 1e-5 * lam and abs(dlt - c["delta"]) < 1e-5 * dlt, "damped: lambda or delta")
    check(abs(abs(np.dot(F_ph(r0 + dlt * uhat), eF)) / abs(f0) - math.exp(-1)) < 1e-9, "damped: amplitude not 1/e after delta")
    dmap = {x["id"]: x for x in p["dir_choices"]}
    good = "plus" if dn > 0 else "minus"
    check(dmap[good]["correct"] == "true" and sum(x["correct"] == "true" for x in p["dir_choices"]) == 1, "damped: direction MC key")
    check(dmap["plus"]["text"] == f"toward $+{u}$" and dmap["minus"]["text"] == f"toward $-{u}$", "damped: direction texts")
    check(np.allclose(unit_of(dmap["alongF"]["text"]), eF), "damped: 'along F' distractor direction")

    # ---- the other field from Maxwell's equations (finite differences of the demodulated phasor)
    h = 2e-4 * lam

    def curl(Ff, r):
        J = np.zeros((3, 3), complex)
        for j in range(3):
            dr = np.zeros(3); dr[j] = h
            J[:, j] = (Ff(r + dr) - Ff(r - dr)) / (2 * h)
        return np.array([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]])

    def lap(Ff, r):
        out = -6 * Ff(r)
        for j in range(3):
            dr = np.zeros(3); dr[j] = h
            out = out + Ff(r + dr) + Ff(r - dr)
        return out / h ** 2

    if fld == "E":
        Eph = F_ph

        def Hph(r):
            return -curl(Eph, r) / (1j * om * mu)
        Y = np.dot(curl(Hph, r0), eF) / np.dot(Eph(r0), eF)
        g2 = np.dot(lap(Eph, r0), eF) / np.dot(Eph(r0), eF)
    else:
        Hph = F_ph
        g2 = np.dot(lap(Hph, r0), eF) / np.dot(Hph(r0), eF)
        Y = g2 / (1j * om * mu)

        def Eph(r):
            return curl(Hph, r) / Y
    check(abs(g2 - (1j * om * mu) * Y) < 1e-5 * abs(g2), "damped: wave equation gamma^2 vs j w mu (sigma + j w eps)")
    check(abs(g2 - complex(ca_n, cb_n) ** 2) < 1e-5 * abs(g2), "damped: Laplacian gamma^2 vs gradient gamma")
    sig, eps = Y.real, Y.imag / om
    check(abs(sig * 1e3 - c["sigma"]) < 1e-5 * c["sigma"], f"damped: sigma {sig*1e3} vs {c['sigma']} mS/m")
    check(abs(eps / EPS0 - c["epsr"]) < 1e-5 * c["epsr"], f"damped: eps_r {eps/EPS0} vs {c['epsr']}")
    E0v, H0v = Eph(r0), Hph(r0)
    ehat = eF if fld == "E" else np.cross(eF, uhat)
    hhat = np.cross(uhat, eF) if fld == "E" else eF
    check(np.linalg.norm(E0v - np.dot(E0v, ehat) * ehat) < 1e-6 * np.linalg.norm(E0v)
          and np.linalg.norm(H0v - np.dot(H0v, hhat) * hhat) < 1e-6 * np.linalg.norm(H0v), "damped: fields not along the expected axes")
    eta = np.dot(E0v, ehat) / np.dot(H0v, hhat)
    check(abs(abs(eta) - c["eta_mag"]) < 1e-5 * abs(eta), f"damped: |eta| {abs(eta)} vs {c['eta_mag']}")
    check(abs(math.degrees(np.angle(eta)) - c["tau_deg"]) < 1e-4, f"damped: tau {math.degrees(np.angle(eta))} vs {c['tau_deg']}")
    Gph = Hph if fld == "E" else Eph
    gsc = 1e3 if fld == "E" else 1.0                   # mA/m for H, V/m for E
    u1 = float(p["u1_str"])
    G1 = np.linalg.norm(Gph(r0 + u1 * uax)) * gsc
    check(abs(G1 - c["Oamp"]) < 1e-5 * G1, f"damped: other-field amplitude {G1} vs {c['Oamp']}")
    check(u1 * dn > 0, "damped: u1 is not downstream")

    # the answer panel's time-domain expression of the other field
    mo = re.fullmatch(_T, p["o_expr"])
    check(mo is not None and mo.group(1) != fld and mo.group(2) == u, f"damped: cannot parse o_expr {p['o_expr']}")
    o_vec = (-1 if mo.group(3) == "-" else 1) * E3[IDX[mo.group(4)]] * float(mo.group(5)) / gsc
    o_ca = (-1 if mo.group(6) == "-" else 1) * (float(mo.group(7)) if mo.group(7) else 1.0)
    o_om = float(mo.group(9)) * 10 ** int(mo.group(10))
    o_cb = (-1 if mo.group(11) == "-" else 1) * coef(mo.group(12))
    o_ph = 0.0 if mo.group(14) is None else (-1 if mo.group(14) == "-" else 1) * float(mo.group(15))
    for uu in (0.0, 0.4 * dlt * dn, 1.3 * dlt * dn):
        rr = r0 + uu * uax
        for tq in (0.0, 0.21 * T, 0.67 * T):
            shown = o_vec * math.exp(o_ca * uu) * math.cos(o_om * tq + o_cb * uu + math.radians(o_ph))
            true = np.real(Gph(rr) * np.exp(1j * om * tq))
            check(np.linalg.norm(shown - true) < 2e-3 * np.linalg.norm(Gph(rr)), "damped: answer-panel expression of the other field")

    # direction of the other field at the instant the given field peaks along ref_hat
    eref = unit_of(p["ref_hat"])
    check(abs(abs(np.dot(eref, eF)) - 1) < 1e-12, "damped: ref direction not along the given field")
    wt = -float(np.angle(np.dot(F_ph(r0), eref)))
    Freal = np.real(F_ph(r0) * np.exp(1j * wt))
    check(abs(np.dot(Freal, eref) - np.linalg.norm(F_ph(r0))) < 1e-9 * np.linalg.norm(F_ph(r0)), "damped: instant of peak")
    Greal = np.real(Gph(r0) * np.exp(1j * wt))
    check(np.linalg.norm(Greal) > 0.5 * np.linalg.norm(Gph(r0)), "damped: other field nearly zero at that instant")
    gdir = Greal / np.linalg.norm(Greal)
    omap = {x["id"]: x for x in p["odir_choices"]}
    check(sum(x["correct"] == "true" for x in p["odir_choices"]) == 1 and omap["ok"]["correct"] == "true", "damped: odir MC key")
    check(np.allclose(unit_of(omap["ok"]["text"]), gdir, atol=1e-6), "damped: odir correct choice disagrees with Maxwell")
    check(np.allclose(unit_of(omap["rev"]["text"]), -gdir, atol=1e-6), "damped: reversed distractor")
    check(np.allclose(unit_of(omap["travel"]["text"]), uhat) and np.allclose(unit_of(omap["par"]["text"]), eref), "damped: odir distractors")

    # ---- statements
    def EH_real(r, ts):
        ts = np.asarray(ts, float)
        ex = np.exp(1j * om * ts)[:, None]
        return np.real(Eph(r)[None, :] * ex), np.real(Hph(r)[None, :] * ex)

    ts = np.arange(64) * T / 64

    def S_avg(r):
        Er, Hr = EH_real(r, ts)
        return np.mean(np.cross(Er, Hr), axis=0)
    S0 = S_avg(r0)
    S1 = S_avg(r0 + dlt * uhat)
    S_half = S_avg(r0 + 0.5 * dlt * uhat)
    rat1, rath = np.dot(S1, uhat) / np.dot(S0, uhat), np.dot(S_half, uhat) / np.dot(S0, uhat)
    power = abs(rat1 - 0.135) < 1.5e-3 and abs(rath - math.exp(-1)) < 1e-6
    power_delta = abs(rat1 - math.exp(-1)) < 1e-2
    txt = {f["id"]: f["text"] for f in p["facts"]}
    # lag of |H| behind |E| at a fixed point, by dense time sampling
    tg = np.linspace(0, T, 200001)
    Er, Hr = EH_real(r0, tg)
    tE = tg[int(np.argmax(np.linalg.norm(Er, axis=1)))]
    tH = tg[int(np.argmax(np.linalg.norm(Hr, axis=1)))]
    lag = (tH - tE) % (T / 2)
    lm = re.search(r"peaks \$([\d.]+)\\ \\mathrm\{ns\}\$ (after|before)", txt["lag"])
    X = float(lm.group(1)) * 1e-9
    tol = max(4 * T / 200000, 3e-3 * X)
    lag_ok = abs(lag - X) < tol if lm.group(2) == "after" else abs(lag - (T / 2 - X)) < tol
    pm = re.search(r"falls to \$([\d.]+)\\ \\%\$", txt["per_lambda"])
    ratio_lam = abs(np.dot(F_ph(r0 + lam * uhat), eF)) / abs(f0) * 100
    per_ok = abs(float(pm.group(1)) - ratio_lam) < 3e-3 * ratio_lam
    km = re.search(r"is \$([\d.]+)\$ times", txt["loss_tan"])
    Ev = Eph(r0)
    cond_disp = np.linalg.norm(sig * Ev) / np.linalg.norm(1j * om * eps * Ev)
    lt_ok = abs(float(km.group(1)) - cond_disp) < 3e-3 * cond_disp
    Y2 = g2 / (1j * om * 2 * mu)
    Y1 = g2 / (1j * om * mu)
    mu_cancel = abs(Y2.real / Y2.imag - Y1.real / Y1.imag) < 1e-9 * Y1.real / Y1.imag
    vp_slower = om / beta < 1 / math.sqrt(mu * eps)
    gam2 = np.sqrt(1j * 2 * om * mu * (sig + 1j * 2 * om * eps))
    double_f = abs(abs(gam2.imag) - 2 * beta) < 1e-6 * beta
    rng = np.random.default_rng(12345)
    sgs, ers, mrs, oms = (10 ** rng.uniform(-8, 8, 20000), rng.uniform(1, 100, 20000), rng.uniform(1, 1000, 20000),
                          10 ** rng.uniform(2, 11, 20000))
    gg = np.sqrt(1j * oms * mrs * MU0 * (sgs + 1j * oms * ers * EPS0))
    alpha_gt_beta = bool(np.any(np.abs(gg.real) > np.abs(gg.imag)))
    t3 = np.linspace(0, 3 * T, 3001)
    Fr3 = F_real(r0, t3) @ eF
    amps = [np.max(np.abs(Fr3[i * 1000:(i + 1) * 1000])) for i in range(3)]
    steady = np.allclose(F_real(r0, t3[:1000] + T) @ eF, Fr3[:1000], atol=1e-9 * max(amps)) and max(amps) - min(amps) < 1e-6 * max(amps)
    eta_real = abs(abs(eta) - math.sqrt(mu / eps)) < 1e-3 * abs(eta)
    grows = abs(np.dot(F_ph(r0 + 0.3 * dlt * uhat), eF)) > abs(f0)
    Sdir = float(np.dot(S0, uhat)) > 0 and np.linalg.norm(S0 - np.dot(S0, uhat) * uhat) < 1e-9 * np.linalg.norm(S0)
    truth = {"power": power, "power_delta": power_delta, "lag": lag_ok, "per_lambda": per_ok, "loss_tan": lt_ok,
             "mu_cancel": mu_cancel, "vp_slower": vp_slower, "double_f": double_f, "alpha_gt_beta": alpha_gt_beta,
             "steady": steady, "eta_real": eta_real, "grows": grows, "Sdir": Sdir}
    expect = {"power": True, "power_delta": False, "mu_cancel": True, "vp_slower": True, "double_f": False,
              "alpha_gt_beta": False, "steady": True, "eta_real": False, "grows": False, "Sdir": True}
    for k, v in expect.items():
        check(bool(truth[k]) == v, f"damped: computed truth of {k} is {truth[k]}, expected {v} (self-test)")
    facts = p["facts"]
    check({f["id"] for f in facts} == set(truth) and len(facts) == len(truth), "damped: statement ids")
    for f in facts:
        check((f["correct"] == "true") == bool(truth[f["id"]]), f"damped: statement {f['id']} key")
        check(f["verdict"] == f["correct"], "damped: verdict text")
