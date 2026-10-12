"""Independent check for waves/magnetization-materials (Lecture 17).

Routes used (none copies the closed forms in server.py):
  (a) bound current: finite-difference curl of a smoothed magnetization M(r) = M S(rho) across the side wall,
      integrated radially -> surface current vector; the same across an end face -> zero.
  (b) B at the centre: vector Biot-Savart sum (np.cross) over the discretized bound surface current, finite rod;
      the long-rod limit from a ring sum with l/a = 4000; H = B/mu0 - M from the definition.
  (c) classification recomputed from the number printed in each statement (parsed from the LaTeX), and the value
      compared with chi_v(SI) = 4 pi chi_mol(cgs) rho / M_molar from CRC molar susceptibilities.
  (d) boundary conditions solved as a least-squares linear system (n.B continuous, n x H continuous).
  (e) statement truths: small computations where possible, otherwise this module's own truth table.
"""
import math
import re

import numpy as np

MU0 = 4e-7 * math.pi
AX = {"x": 0, "y": 1, "z": 2}
E3 = np.eye(3)

# CRC Handbook molar susceptibilities (cgs, 1e-6 cm^3/mol), density (g/cm^3), molar mass (g/mol).
CRC = {
    "bismuth": (-280.1, 9.78, 208.98), "water": (-12.97, 0.997, 18.015), "copper": (-5.46, 8.96, 63.546),
    "silver": (-19.5, 10.49, 107.87), "gold": (-28.0, 19.3, 196.97), "lead": (-23.0, 11.34, 207.2),
    "diamond": (-5.9, 3.51, 12.011), "aluminium": (16.5, 2.70, 26.98), "platinum": (193.0, 21.45, 195.08),
    "palladium": (567.4, 12.02, 106.42), "titanium": (151.0, 4.506, 47.867), "magnesium": (13.1, 1.738, 24.305),
    "liquid oxygen at 90 K": (7699.0, 1.141, 32.00),
}
# Air: 21 % O2 at 293 K, 1 atm, chi_mol(O2) = +3449e-6 cm^3/mol.
AIR_CHI = 4 * math.pi * 3449e-6 * 0.21 * 101325 / (8.314 * 293) * 1e-6
FERRO_RANGE = {"cobalt": (70, 300), "nickel": (100, 700), "iron, 99.8 % pure": (2000, 10000),
               "mumetal": (2e4, 1e5), "permalloy": (5000, 1e5), "electrical silicon steel": (1500, 8000)}


def ref_chi(name):
    if name.startswith("air"):
        return AIR_CHI
    cm, rho, mm = CRC[name]
    return 4 * math.pi * cm * 1e-6 * rho / mm


def parse_number(tex):
    """'-1.7\\times10^{-4}', '0.999\\,83', '50\\,000', '600' -> float."""
    t = tex.replace("\\,", "").replace(" ", "")
    m = re.fullmatch(r"(-?[\d.]+)\\times10\^\{(-?\d+)\}", t)
    if m:
        return float(m.group(1)) * 10 ** int(m.group(2))
    return float(t)


def parse_vec(tex):
    t = tex.replace(" ", "")
    v = np.zeros(3)
    for sgn, mag, ax in re.findall(r"([+-]?)([\d.]+)\\,\\hat\{([xyz])\}", t):
        v[AX[ax]] += (-1 if sgn == "-" else 1) * float(mag)
    return v


def smooth_step(x, w):
    return 0.5 * (1 - np.tanh(x / w))


def curl_fd(F, p, h):
    J = np.zeros((3, 3))           # J[i, j] = dF_i/dx_j
    for j in range(3):
        dp = np.zeros(3)
        dp[j] = h
        J[:, j] = (F(p + dp) - F(p - dp)) / (2 * h)
    return np.array([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]])


def phys(d, check):
    p, c = d["params"], d["correct_answers"]
    u = E3[AX[p["u"]]]
    M0 = p["M0"] * 1e3                       # A/m
    Mvec = p["s"] * M0 * u
    check(np.allclose(parse_vec(p["M_tex"]) * 1e3, Mvec), "magnetization: M_tex does not match M0, s, u")
    a = p["a_mm"] * 1e-3
    L = p["L_cm"] * 1e-2

    # ---- (a) side wall: smoothed M, numerical curl, integrate across the wall
    others = [k for k in "xyz" if k != p["u"]]
    rng = np.random.default_rng(d.get("variant_seed", 0))
    ang = rng.uniform(0, 2 * math.pi)
    rhat = math.cos(ang) * E3[AX[others[0]]] + math.sin(ang) * E3[AX[others[1]]]
    phihat = np.cross(u, rhat)               # r x phi = u  <=>  phi = u x r
    check(np.allclose(np.cross(rhat, phihat), u), "magnetization: phi-hat convention")
    wdt = a / 200

    def Mfield(r):
        rho = np.linalg.norm(r - np.dot(r, u) * u)
        return Mvec * smooth_step(rho - a, wdt)

    rs = np.linspace(a - 12 * wdt, a + 12 * wdt, 241)
    K = np.zeros(3)
    for i in range(len(rs) - 1):
        rm = 0.5 * (rs[i] + rs[i + 1])
        K += curl_fd(Mfield, rm * rhat, wdt / 20) * (rs[i + 1] - rs[i])
    check(abs(np.linalg.norm(K) / 1e3 - c["JsM"]) < 2e-3 * c["JsM"], f"magnetization: |J_sM| {np.linalg.norm(K)/1e3} vs {c['JsM']}")
    check(abs(np.dot(K, u)) < 1e-6 * M0, "magnetization: bound surface current has an axial part")
    plus = np.dot(K, phihat) > 0
    jd = {x["id"]: x["correct"] for x in p["jdir_choices"]}
    check(sum(v == "true" for v in jd.values()) == 1 and len(jd) == 5, "magnetization: jdir needs exactly one correct of 5")
    check(jd["plus" if plus else "minus"] == "true", "magnetization: direction of J_sM (numerical curl) disagrees with key")

    # end face: smoothed in the axial coordinate -> no surface current
    def Mend(r):
        return Mvec * smooth_step(np.dot(r, u) - L / 2, wdt)
    zs = np.linspace(L / 2 - 12 * wdt, L / 2 + 12 * wdt, 121)
    Kend = sum(curl_fd(Mend, 0.5 * (zs[i] + zs[i + 1]) * u + 0.3 * a * rhat, wdt / 20) * (zs[i + 1] - zs[i])
               for i in range(len(zs) - 1))
    check(np.linalg.norm(Kend) < 1e-6 * M0, "magnetization: end face should carry no bound current")

    # ---- (b) B at the centre from the bound current, by vector Biot-Savart (finite rod)
    nz, nphi = 1200, 48
    xi = np.linspace(-1, 1, nz + 1)
    zgrid = (L / 2) * np.sinh(4 * xi) / math.sinh(4)          # denser near the centre
    zc = 0.5 * (zgrid[1:] + zgrid[:-1])
    dz = np.diff(zgrid)
    phis = (np.arange(nphi) + 0.5) * 2 * math.pi / nphi
    e1, e2 = E3[AX[others[0]]], E3[AX[others[1]]]
    if np.dot(np.cross(e1, e2), u) < 0:
        e2 = -e2
    B = np.zeros(3)
    Ksurf = np.linalg.norm(K) * (1 if plus else -1)            # signed, along phi-hat (from the numerical curl above)
    for ph in phis:
        rp = a * (math.cos(ph) * e1 + math.sin(ph) * e2)
        ph_hat = -math.sin(ph) * e1 + math.cos(ph) * e2
        src = zc[:, None] * u[None, :] + rp[None, :]
        R = -src                                               # from source to the field point (origin)
        Rn = np.linalg.norm(R, axis=1)[:, None]
        dl = ph_hat * a * 2 * math.pi / nphi
        dI = Ksurf * dz[:, None]                               # current in each strip
        B += np.sum(MU0 / (4 * math.pi) * dI * np.cross(np.broadcast_to(dl, R.shape), R) / Rn ** 3, axis=0)
    Bu_finite = np.dot(B, u)
    check(abs(Bu_finite * 1e3 - c["Bin"]) < 3e-3 * abs(c["Bin"]), f"magnetization: B inside (Biot-Savart, finite rod) {Bu_finite*1e3} vs {c['Bin']}")
    check(np.linalg.norm(B - Bu_finite * u) < 1e-6 * abs(Bu_finite), "magnetization: B at the centre not axial")
    # long-rod limit with a ring sum (l/a = 4000)
    zz = a * np.sinh(np.linspace(-np.arcsinh(2000), np.arcsinh(2000), 400001))
    zm, dzz = 0.5 * (zz[1:] + zz[:-1]), np.diff(zz)
    B_long = np.sum(MU0 * Ksurf * dzz * a * a / (2 * (a * a + zm * zm) ** 1.5))
    check(abs(B_long * 1e3 - c["Bin"]) < 1e-5 * abs(c["Bin"]), f"magnetization: long-rod B {B_long*1e3} vs {c['Bin']}")
    H_in = B_long / MU0 - np.dot(Mvec, u)
    check(abs(H_in) < 1e-4 * M0, "magnetization: H inside should vanish")
    ins = {x["id"]: x["correct"] for x in p["inside_choices"]}
    check(ins.get("ok") == "true" and sum(v == "true" for v in ins.values()) == 1, "magnetization: inside MC key")
    check(abs(c["Ieq"] - abs(B_long) / (MU0 * p["n_turns"])) < 1e-4 * c["Ieq"], "magnetization: equivalent solenoid current")

    # ---- (c) classification
    items = p["cls_items"]
    check(len(items) == 5 and {it["key"] for it in items} == {"dia", "para", "ferro"}, "magnetization: matching needs 5 items, all classes")
    for it in items:
        mm = re.search(r"\\(chi_m|mu_r)\\approx (.+)\$", it["text"])
        check(mm is not None, f"magnetization: cannot parse {it['text']}")
        if not mm:
            continue
        val = parse_number(mm.group(2))
        chi = val - 1 if mm.group(1) == "mu_r" else val
        cls = "dia" if chi < 0 else ("para" if chi < 1e-2 else ("ferro" if chi > 10 else "??"))
        check(cls == it["key"], f"magnetization: {it['text']} classified {cls}, key {it['key']}")
        check(abs(chi - it["chi"]) <= 0.06 * abs(it["chi"]), f"magnetization: printed value {chi} vs chi {it['chi']}")
        if it["name"] in FERRO_RANGE:
            lo, hi = FERRO_RANGE[it["name"]]
            check(lo <= it["chi"] + 1 <= hi, f"magnetization: mu_r of {it['name']} unrealistic")
        else:
            ref = ref_chi(it["name"])
            check(abs(it["chi"] - ref) <= 0.10 * abs(ref), f"magnetization: chi of {it['name']} {it['chi']} vs CRC {ref:.3g}")

    # ---- (d) interface by least squares
    n = E3[AX[p["w"]]]
    B1 = np.array([p["B1"][k] for k in "xyz"], float)
    check(np.allclose(parse_vec(p["B1_tex"]), B1), "magnetization: B1_tex does not match B1")
    mu1, mu2 = p["mr1"] * MU0, p["mr2"] * MU0
    Nx = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])     # n x (.)
    A = np.vstack([n[None, :], Nx / mu2])
    rhs = np.concatenate([[n @ B1], Nx @ B1 / mu1])
    B2 = np.linalg.lstsq(A, rhs, rcond=None)[0]
    for k in "xyz":
        check(abs(B2[AX[k]] - c["B2" + k]) <= 1e-9 + 1e-9 * abs(B2[AX[k]]), f"magnetization: B2{k} {B2[AX[k]]} vs {c['B2'+k]}")
    check(np.allclose(parse_vec(p["B2_tex"]), B2, rtol=1e-3, atol=1e-9), "magnetization: B2_tex in the answer panel")
    H2 = B2 * 1e-3 / mu2
    M2 = B2 * 1e-3 / MU0 - H2                                   # H = B/mu0 - M
    check(abs(M2 @ n / 1e3 - c["M2n"]) < 1e-9 * max(1, abs(c["M2n"])), f"magnetization: M2n {M2 @ n / 1e3} vs {c['M2n']}")

    # ---- (e) statement truths
    t1 = math.atan2(np.linalg.norm(B1 - (B1 @ n) * n), abs(B1 @ n))
    t2 = math.atan2(np.linalg.norm(B2 - (B2 @ n) * n), abs(B2 @ n))
    curlM_div = 0.0
    def Mtest(r):
        x, y, z = r
        return np.array([math.sin(y * z), x * x * z, math.cos(x + 2 * y)])
    hh = 1e-4
    r0 = np.array([0.3, -0.7, 0.5])
    def curlM(r):
        return curl_fd(Mtest, r, hh)
    for j in range(3):
        dp = np.zeros(3); dp[j] = 1e-3
        curlM_div += (curlM(r0 + dp)[j] - curlM(r0 - dp)[j]) / 2e-3
    chi_t, Ht = 99.0, 1.0
    units_mu0chiH = np.array([1, 1, -2, -2]) + np.array([0, -1, 0, 1])     # (kg, m, s, A): mu0 * H
    units_M = np.array([0, -1, 0, 1])
    truth = {
        "Hdef": True, "ampere_free": True, "remanence": True, "coercivity": True, "iron_nonlinear": True,
        "dia_opposes": (-1e-5 * Ht) * Ht < 0,
        "divJM": abs(curlM_div) < 1e-4,
        "mu0chi": bool(np.array_equal(units_mu0chiH, units_M)),
        "Bmu_minus_M": abs((MU0 * (1 + chi_t) * Ht) / (MU0 * (1 + chi_t)) - chi_t * Ht - Ht) < 1e-12,
        "para_domains": False,
        "divH": abs(B1 @ n / mu1 - B2 @ n / mu2) < 1e-15,
        "Bt_cont": np.allclose(B1 - (B1 @ n) * n, B2 - (B2 @ n) * n),
        "dia_attracted": (-1e-5) > 0,          # force ~ chi grad(B^2): toward strong field only for chi > 0
        "remove_H": False,
        "angle": t2 > t1,
    }
    facts = p["facts"]
    check(len(facts) == len(truth) and {f["id"] for f in facts} == set(truth), "magnetization: statement ids")
    for f in facts:
        check((f["correct"] == "true") == truth[f["id"]], f"magnetization: statement {f['id']} truth key")
        check(f["verdict"] == f["correct"], "magnetization: verdict text")
