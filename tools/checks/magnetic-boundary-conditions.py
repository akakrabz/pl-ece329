"""Independent check for magnetic-boundary-conditions (Lecture 6).

(a) Superposition instead of the jump formula: the sheet's own field is H_s = 1/2 Js x n_p (n_p from the sheet
    toward the field point); a uniform background H_bg is fixed from the given side, and the other side is
    H_bg + H_s(other side).  Then n.(B1 - B2) = 0 and Ampere's law on thin loops are verified numerically.
(b) Js from the given fields by numerically evaluating the circulation of H around two thin rectangles that
    straddle the plane (long sides along each in-plane axis): I_enc = Js.(t x n) L.
(c), (d): truth / match tables written here.
"""
import numpy as np

AX = "xyz"
E3 = {a: np.eye(3)[i] for i, a in enumerate(AX)}
CB_TRUTH = {"Et": True, "Dn": True, "Bn": True, "Ht": True, "par": True, "swap": True,
            "En": False, "Dt": False, "Bt": False, "Hn": False, "order": False, "units": False}
MATCH = {"gaussD": "Dn", "gaussB": "Bn", "circE": "Et", "ampere": "Ht"}
MU0 = 4e-7 * np.pi


def circulation(Hfun, t, n, L=2.0, h=1e-3, N=400):
    """Numerically integrate H.dl around a rectangle in the (t, n) plane centred on the interface:
    +t along the medium-2 side (n-coordinate -h/2), up +n, -t along the medium-1 side, down -n. Midpoint rule."""
    corners = [-h / 2 * n - L / 2 * t, -h / 2 * n + L / 2 * t, h / 2 * n + L / 2 * t, h / 2 * n - L / 2 * t]
    total = 0.0
    for k in range(4):
        a, b = corners[k], corners[(k + 1) % 4]
        s = (np.arange(N) + 0.5) / N
        pts = a[None, :] + s[:, None] * (b - a)[None, :]
        H = np.array([Hfun(r) for r in pts])
        total += float(np.sum(H @ (b - a)) / N)
    return total


def phys(d, check):
    p, c = d["params"], d["correct_answers"]
    z = E3["z"]

    # ---------------------------------------------------------------- (a)
    Js = np.array([p["Jx"], p["Jy"], 0.0])
    H1, H2 = np.array(p["H1"], float), np.array(p["H2"], float)
    sheet = lambda side: 0.5 * np.cross(Js, side * z)          # side = +1 above, -1 below
    if p["given_above"]:
        Hbg = H1 - sheet(+1)
        Hother = Hbg + sheet(-1)
        check(np.allclose(Hother, H2), f"magnetic-bc (a): superposition gives {Hother}, server H2 {H2}")
    else:
        Hbg = H2 - sheet(-1)
        Hother = Hbg + sheet(+1)
        check(np.allclose(Hother, H1), f"magnetic-bc (a): superposition gives {Hother}, server H1 {H1}")
    ans = np.array([c["Hax"], c["Hay"], c["Haz"]])
    check(np.allclose(ans, Hother, atol=1e-12), f"magnetic-bc (a): answers {ans} vs {Hother}")
    check(abs(MU0 * (H1[2] - H2[2])) < 1e-15, "magnetic-bc (a): normal B not continuous")
    Hfun_a = lambda r: H1 if r[2] > 0 else H2
    for t in (E3["x"], E3["y"]):
        I_loop = circulation(Hfun_a, t, z)
        I_enc = float(Js @ np.cross(t, z)) * 2.0
        check(abs(I_loop - I_enc) < 1e-9, f"magnetic-bc (a): Ampere loop along {t}: {I_loop} vs {I_enc}")
    check(p["asked"] == ("2" if p["given_above"] else "1"), "magnetic-bc (a): label of the asked side")

    # ---------------------------------------------------------------- (b)
    u, t1, t2 = p["u"], p["t1"], p["t2"]
    n = E3[u]
    Hpos, Hneg = np.array(p["Hpos"], float), np.array(p["Hneg"], float)
    check(abs((Hpos - Hneg) @ n) < 1e-12, "magnetic-bc (b): given data violate normal-B continuity")
    Hfun_b = lambda r: Hpos if r @ n > 0 else Hneg
    # the loop with long sides along t encloses current Js.(t x n) per unit length; t x n is the other in-plane axis
    comp = {}
    for t in (E3[t1], E3[t2]):
        m = np.cross(t, n)
        k = int(np.argmax(np.abs(m)))
        comp[AX[k]] = circulation(Hfun_b, t, n) / 2.0 / m[k]
    check(abs(comp[t1] - c["Jb1"]) < 1e-9 and abs(comp[t2] - c["Jb2"]) < 1e-9,
          f"magnetic-bc (b): loops give J_s{t1}={comp[t1]}, J_s{t2}={comp[t2]} vs {c['Jb1']}, {c['Jb2']}")
    check(sorted([u, t1, t2]) == ["x", "y", "z"] and u in "xy" and t1 < t2, "magnetic-bc (b): axes")
    check(abs(c["Jb1"]) + abs(c["Jb2"]) > 0, "magnetic-bc (b): zero surface current")

    # ---------------------------------------------------------------- (c), (d)
    cb = p["cb"]
    check(len(cb) == len(CB_TRUTH) and {x["key"] for x in cb} == set(CB_TRUTH), "magnetic-bc: checkbox keys")
    for x in cb:
        check((x["correct"] == "true") == CB_TRUTH[x["key"]], f"magnetic-bc: statement {x['key']} marked {x['correct']}")
        check(x["tf"] == ("True" if CB_TRUTH[x["key"]] else "False"), f"magnetic-bc: tf label {x['key']}")
    for s in p["statements"]:
        check(MATCH.get(s["key"]) == s["match"], f"magnetic-bc: matching {s['key']} -> {s['match']}")
    names = [o["name"] for o in p["options"]]
    check(len(names) == len(set(names)) and set(MATCH.values()) <= set(names) and len(names) >= 6,
          "magnetic-bc: matching options")

    # a numerical illustration of the 'par' and 'order' statements with this variant's sheet in (a)
    jump = H1 - H2
    check(abs(jump @ Js) < 1e-12 and np.allclose(jump, np.cross(Js, z)) and
          (not np.allclose(jump, np.cross(z, Js))), "magnetic-bc: jump must be Js x n, perpendicular to Js")
