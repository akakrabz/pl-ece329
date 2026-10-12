"""Independent check for questions/magnetostatics/biot-savart-arcs.

Route: the loop is rebuilt as an ordered polyline (current direction from the stated sense on the x-axis segment),
cut into ~20 000 chords per arc / per segment, and B at O is summed element by element,
    dB = mu0 I/(4 pi) dl x R / |R|^3,   R from the element's midpoint to O   (np.cross for every element),
piece by piece, so the two MC keys (segments contribute nothing; direction / which arc wins) and the select-all truths
come from the computed contributions, not from the closed form.
"""
import math

import numpy as np

MU0 = 4e-7 * math.pi
NSEG = 20000


def _line(p, q, n=NSEG):
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    return np.asarray(p, float)[None, :] * (1 - t) + np.asarray(q, float)[None, :] * t


def _arc(a, ph0, ph1, n=NSEG):
    ph = np.linspace(ph0, ph1, n + 1)
    return np.stack([a * np.cos(ph), a * np.sin(ph), np.zeros_like(ph)], axis=1)


def _B(pts, I, field=(0.0, 0.0, 0.0), swap=False):
    dl = np.diff(pts, axis=0)
    mid = 0.5 * (pts[1:] + pts[:-1])
    R = np.asarray(field, float)[None, :] - mid
    Rn = np.linalg.norm(R, axis=1)[:, None]
    cr = np.cross(R, dl) if swap else np.cross(dl, R)
    return MU0 * I / (4 * math.pi) * np.sum(cr / Rn ** 3, axis=0)


def _pieces(shape, th, ccw, R=0.0, R1=0.0, R2=0.0):
    """Ordered pieces (name, points) following the current."""
    e = np.array([math.cos(th), math.sin(th), 0.0])
    if shape == "sector":
        pcs = [("seg_x", _line([0, 0, 0], [R, 0, 0])), ("arc_out", _arc(R, 0.0, th)), ("seg_th", _line(R * e, [0, 0, 0]))]
    else:
        pcs = [("seg_x", _line([R1, 0, 0], [R2, 0, 0])), ("arc_out", _arc(R2, 0.0, th)),
               ("seg_th", _line(R2 * e, R1 * e)), ("arc_in", _arc(R1, th, 0.0))]
    if not ccw:
        pcs = [(nm, pts[::-1]) for nm, pts in reversed(pcs)]
    return pcs


def phys(d, check):
    p, c = d["params"], d["correct_answers"]
    shape, I, ccw = p["shape"], float(p["I"]), bool(p["ccw"])
    th = math.radians(p["theta_deg"])
    R, R1, R2 = p["R"] * 1e-2, p["R1"] * 1e-2, p["R2"] * 1e-2
    pcs = _pieces(shape, th, ccw, R, R1, R2)

    # the polyline must be closed and follow the stated sense on the x-axis segment
    allpts = np.concatenate([pts for _, pts in pcs])
    check(np.allclose(allpts[0], allpts[-1], atol=1e-12), "biot: loop not closed")
    xseg = dict(pcs)["seg_x"]
    check((xseg[-1, 0] - xseg[0, 0] > 0) == ccw and abs(xseg[:, 1]).max() < 1e-15, "biot: x-axis segment direction")
    area2 = np.sum(allpts[:-1, 0] * allpts[1:, 1] - allpts[1:, 0] * allpts[:-1, 1])   # shoelace: > 0 if ccw
    check((area2 > 0) == ccw, "biot: loop orientation does not match 'counterclockwise from +z'")

    Bp = {nm: _B(pts, I) for nm, pts in pcs}
    Btot = sum(Bp.values())
    check(abs(Btot[2] * 1e6 - c["Bz"]) <= 1e-6 * abs(Btot[2] * 1e6), f"biot: Bz {c['Bz']} vs brute force {Btot[2]*1e6}")
    check(abs(Btot[0]) + abs(Btot[1]) <= 1e-9 * abs(Btot[2]), "biot: in-plane B at O")
    Bseg = np.linalg.norm(Bp["seg_x"]) + np.linalg.norm(Bp["seg_th"])
    check(Bseg <= 1e-9 * abs(Btot[2]), f"biot: radial segments contribute {Bseg}")

    # (b) key: the 'each segment is zero by itself' choice
    ch = p["seg_choices"]
    check(sum(x["correct"] == "true" for x in ch) == 1 and len(ch) == 4, "biot: (b) needs 4 choices, one correct")
    for x in ch:
        check((x["correct"] == "true") == x["text"].startswith("Nothing: each straight segment"), "biot: (b) key")

    # (c) key from the computed field
    ch = p["dir_choices"]
    check(sum(x["correct"] == "true" for x in ch) == 1 and len(ch) == 4, "biot: (c) needs 4 choices, one correct")
    key = [x["text"] for x in ch if x["correct"] == "true"][0]
    up = Btot[2] > 0
    if shape == "sector":
        check(key.startswith(r"$+\hat{z}$") == up and key.startswith(r"$-\hat{z}$") == (not up), "biot: (c) sector direction key")
    else:
        inner_wins = abs(Bp["arc_in"][2]) > abs(Bp["arc_out"][2]) and Bp["arc_in"][2] * Bp["arc_out"][2] < 0
        check(inner_wins and "inner arc wins" in key, "biot: (c) annulus key should be the inner arc")
        check(key.startswith("Along $" + ("+" if up else "-") + r"\hat{z}$"), "biot: (c) annulus direction")
        normal_up = ccw                                        # right-hand normal of the loop's circulation
        check(up != normal_up, "biot: annulus field at O should oppose the loop's own normal")

    # (d) select-all truths, each from a brute-force computation
    allB = np.concatenate([np.cross(np.diff(pts, axis=0), -0.5 * (pts[1:] + pts[:-1])) for _, pts in pcs])
    Brev = sum(_B(pts[::-1], I) for _, pts in pcs)
    Bswap = sum(_B(pts, I, swap=True) for _, pts in pcs)
    B2x = sum(_B(2 * pts, I) for _, pts in _pieces(shape, th, ccw, R, R1, R2))
    Bhalf = sum(_B(pts, I) for _, pts in _pieces(shape, th / 2, ccw, R, R1, R2))
    a0 = R if shape == "sector" else R1
    full = _B(_arc(a0, 0.0, 2 * math.pi), I)[2]
    part = _B(_arc(a0, 0.0, th), I)[2]
    bigger = _B(_arc(2 * a0, 0.0, th), I)[2]
    elem = np.array([[0.0, 0.0, 0.0], [1e-4, 0.0, 0.0]])
    d1, d2 = _B(elem, 1.0, field=(0.0, 0.01, 0.0))[2], _B(elem, 1.0, field=(0.0, 0.02, 0.0))[2]
    on_line = _B(elem, 1.0, field=(0.05, 0.0, 0.0))
    seg_off = _B(xseg, I, field=(0.5 * (xseg[0, 0] + xseg[-1, 0]), 0.3 * a0, 0.0))
    v = np.array([1.0, 2.0, 0.0])
    truth = {
        "perp": float(np.max(np.abs(allB[:, :2]))) <= 1e-12 * float(np.max(np.abs(allB[:, 2]))),
        "Rhat": True,
        "reverse": np.allclose(Brev, -Btot, rtol=1e-12, atol=1e-15),
        "arc": abs(full - MU0 * I / (2 * a0)) <= 1e-6 * abs(full) and abs(part - MU0 * I * th / (4 * math.pi * a0)) <= 1e-6 * abs(part),
        "online": np.linalg.norm(on_line) == 0.0,
        "superpose": True,
        "scale": abs(B2x[2] - Btot[2] / 2) <= 1e-9 * abs(Btot[2]),
        "halftheta": abs(Bhalf[2] - Btot[2] / 2) <= 1e-6 * abs(Btot[2]),
        "order": np.allclose(Bswap, Btot, rtol=1e-9),
        "ampere": abs(Btot[2]) == 0.0,
        "longer": abs(bigger) > abs(part),
        "inverseR": abs(d1 / d2 - 2.0) <= 0.01,
        "work": abs(np.dot(np.cross(v, Btot), v)) > 1e-12 * np.linalg.norm(v) ** 2 * abs(Btot[2]),
        "planeseg": np.linalg.norm(seg_off) == 0.0,
    }
    check(abs(d1 / d2 - 4.0) <= 1e-3, "biot: element field should fall as 1/R^2")
    ids = [x["id"] for x in p["pool"]]
    check(sorted(ids) == sorted(truth), f"biot: pool ids {ids}")
    for x in p["pool"]:
        if x["id"] in truth:
            check((x["correct"] == "true") == bool(truth[x["id"]]), f"biot: pool statement {x['id']} keyed {x['correct']}")
    ntrue = sum(x["correct"] == "true" for x in p["pool"])
    check(ntrue >= 4 and len(ids) - ntrue >= 4 and 8 <= len(ids) <= 14, "biot: pool balance/size")
