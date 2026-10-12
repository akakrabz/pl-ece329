"""Independent check for midterm2/current-slabs-ampere.

Route: chop each slab into thin sheets (thickness 1/400 mm, cell edges on every quarter-millimetre, so the
midpoint sum is exact at the listed points), give each thin sheet the surface current J*du, and add the
sheet fields 0.5 * Js x n_hat with np.cross (n_hat from that thin sheet toward the field point). Then:
  * every H answer (component along +h) and the vanishing of the other two components;
  * Ampere's law on rectangles between pairs of points: H_h(uj) - H_h(ui) = s_h * (current per unit length
    enclosed), with the enclosed current integrated directly from J;
  * the reversed/removed-sheet answer;
  * both MC keys, by evaluating every choice's claim against the numerical field profile.
"""
import numpy as np

AX = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]), "z": np.array([0.0, 0.0, 1.0])}
IDX = {"x": 0, "y": 1, "z": 2}
DU = 1.0 / 400.0          # mm


def _sheets(p, Ksheet):
    pos, cur = [], []
    for a, b, J in ((p["a1"], p["b1"], p["J1"]), (p["a2"], p["b2"], p["J2"])):
        n = int(round((b - a) / DU))
        pos.append(a + (np.arange(n) + 0.5) * (b - a) / n)
        cur.append(np.full(n, J * 1e3 * ((b - a) / n) * 1e-3))       # A/m^2 x m = A/m per thin sheet
    pos.append(np.array([0.0]))
    cur.append(np.array([float(Ksheet)]))
    return np.concatenate(pos), np.concatenate(cur)


def _H(p, x, Ksheet):
    ec, eu = AX[p["c"]], AX[p["u"]]
    pos, cur = _sheets(p, Ksheet)
    nhat = np.sign(x - pos)[:, None] * eu[None, :]
    Js = cur[:, None] * ec[None, :]
    return 0.5 * np.cross(Js, nhat).sum(axis=0)


def _I_between(p, ui, uj, Ksheet):
    """current per unit length (A/m) along +c crossing the strip ui < u < uj, by direct integration of J."""
    n = int(round((uj - ui) / DU))                       # cell edges on the quarter-mm grid: midpoint rule exact for steps
    xs = ui + (np.arange(n) + 0.5) * (uj - ui) / n
    J = np.where((xs > p["a1"]) & (xs < p["b1"]), p["J1"] * 1e3, 0.0) + np.where((xs > p["a2"]) & (xs < p["b2"]), p["J2"] * 1e3, 0.0)
    tot = np.sum(J) * (uj - ui) / n * 1e-3
    if ui < 0 < uj:
        tot += Ksheet
    return tot


def phys(d, check):
    p, c = d["params"], d["correct_answers"]
    name = "current-slabs-ampere"
    ih, ic, iu = IDX[p["h"]], IDX[p["c"]], IDX[p["u"]]
    s_h = float(np.dot(np.cross(AX[p["c"]], AX[p["u"]]), AX[p["h"]]))
    check(s_h == p["s_h"], f"{name}: s_h bookkeeping")
    pts = p["pts"]
    check(pts[0] < p["a1"] < pts[1] < p["b1"] < pts[2] < p["a2"] < pts[3] < p["b2"] < pts[4] and pts[2] != 0,
          f"{name}: points not in the five regions {pts}")
    Hv = [_H(p, x, p["K"]) for x in pts]
    for i, H in enumerate(Hv):
        key = f"H{i + 1}"
        check(abs(H[ih] - c[key]) <= 1e-9 * max(1.0, abs(H[ih])), f"{name}: {key} superposition {H[ih]} vs {c[key]}")
        check(abs(H[ic]) < 1e-12 and abs(H[iu]) < 1e-12, f"{name}: H has components along c or u at P{i + 1}: {H}")
    # Ampere rectangles between consecutive points (integrated J, not the slab formula)
    for i in range(4):
        dH = Hv[i + 1][ih] - Hv[i][ih]
        I = _I_between(p, pts[i], pts[i + 1], p["K"])
        check(abs(dH - s_h * I) <= 1e-6 * max(1.0, abs(I)), f"{name}: Ampere loop P{i + 1}-P{i + 2}: dH {dH} vs I {I}")
    # (d)
    Knew = -p["K"] if p["dmode"] == "reversed" else 0.0
    Hd = _H(p, pts[p["kd"]], Knew)[ih]
    check(abs(Hd - c["Hd"]) <= 1e-9 * max(1.0, abs(Hd)), f"{name}: Hd {Hd} vs {c['Hd']} ({p['dmode']})")

    # (b) which components vanish everywhere: sample densely
    samples = np.linspace(pts[0] - 3, pts[4] + 3, 61) + 0.01
    Hs = np.array([_H(p, x, p["K"]) for x in samples])
    zero = {a for a in "xyz" if np.max(np.abs(Hs[:, IDX[a]])) < 1e-12}
    for ch in p["comp_choices"]:
        truth = set(ch["zero"]) == zero
        check(truth == (ch["correct"] == "true"), f"{name}: component MC claim {ch['zero']} vs numerical zero set {zero}")

    # (c) shape across slab ks
    a, b = (p["a1"], p["b1"]) if p["ks"] == 1 else (p["a2"], p["b2"])
    J = p["J1"] if p["ks"] == 1 else p["J2"]
    w = b - a
    hq = 0.25
    inner = [a + 0.25 * w, a + 0.5 * w, a + 0.75 * w]
    slopes = [(_H(p, x + hq, p["K"])[ih] - _H(p, x - hq, p["K"])[ih]) / (2 * hq) for x in inner]
    jumps = [abs(_H(p, f + 1e-3, p["K"])[ih] - _H(p, f - 1e-3, p["K"])[ih]) for f in (a, b)]
    m = 0.5 * (a + b)
    inv = [(_H(p, m + dd, p["K"])[ih] - _H(p, m - dd, p["K"])[ih]) * dd for dd in (0.25 * w, 0.5 * w)]
    for ch in p["shape_choices"]:
        k = ch["kind"]
        if k == "linear":
            truth = all(abs(s - ch["slope"]) < 1e-9 for s in slopes) and max(jumps) < 1e-9
        elif k == "const_jump":
            truth = all(abs(s) < 1e-9 for s in slopes) and all(abs(j - abs(J) * w) < 1e-9 for j in jumps)
        elif k == "linear_jump":
            truth = abs(slopes[0] - slopes[1]) < 1e-9 and all(abs(j - abs(J) * w / 2) < 1e-9 for j in jumps)
        else:   # 1/distance from the centre plane: (H(m+d) - H(m-d)) * d would be the same for both d
            truth = abs(inv[0] - inv[1]) < 1e-9 * max(1.0, abs(inv[0]))
        check(truth == (ch["correct"] == "true"), f"{name}: shape MC claim {k} {ch.get('slope')} truth {truth} vs key {ch['correct']} (slopes {slopes}, jumps {jumps})")
    check(abs(slopes[0] - s_h * J) < 1e-9, f"{name}: slope inside slab {slopes[0]} vs s_h J {s_h * J}")

    # figure: drawn to scale along u (one linear map for faces, sheet and points), points labelled in order
    xs = {k: float(p[k]) for k in ("x_a1", "x_b1", "x_0", "x_a2", "x_b2")}
    us = {"x_a1": p["a1"], "x_b1": p["b1"], "x_0": 0.0, "x_a2": p["a2"], "x_b2": p["b2"]}
    xp = [float(q["x"]) for q in p["pts_svg"]]
    allx, allu = list(xs.values()) + xp, [us[k] for k in xs] + list(pts)
    k, c0 = np.polyfit(allu, allx, 1)
    check(k > 0 and max(abs(k * uu + c0 - xx) for uu, xx in zip(allu, allx)) < 0.11, f"{name}: figure not to scale")
    check([q["lab"] for q in p["pts_svg"]] == ["1", "2", "3", "4", "5"], f"{name}: figure point labels")
    check(abs(float(p["w_s1"]) - (xs["x_b1"] - xs["x_a1"])) < 0.11 and abs(float(p["w_s2"]) - (xs["x_b2"] - xs["x_a2"])) < 0.11,
          f"{name}: slab rectangles do not match their faces")
