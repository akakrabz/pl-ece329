"""Independent check for pn-junction-poisson (Lecture 7).

Shooting with Gauss's law instead of the neutrality formula: start at the outer edge of the given layer with
E = 0, integrate dE/ds = rho/eps numerically (midpoint rule, cell by cell) through the junction into the other
side, and find where E returns to zero -- that is the other layer's edge.  Then rebuild rho(x) in the variant's
real orientation, integrate E_x(x) and V(x) = -int E_x dx numerically, and read off |E|max, where it occurs, the
sign of E_x (direction MC), the curvature of V in each layer and V_bi = V(n edge) - V(p edge).
Checkbox: static truths tabled here, the n-wider statement from the numerical widths.
"""
import numpy as np

EPS = 11.7 * 8.854e-12
E_CH = 1.602e-19
STATIC_TRUTH = {"poisson": True, "parab": True, "Econt": True, "neutralE": True, "concave_n": True, "lighter": True,
                "area": True, "mono": True, "linear": False, "Ejump": False, "concave_p": False, "heavier": False,
                "Vequal": False}


def integrate_midpoint(rho_of, x):
    """E at the nodes x: E(x0) = 0, E_{i+1} = E_i + rho(cell midpoint) * dx / eps."""
    mid = 0.5 * (x[1:] + x[:-1])
    E = np.zeros_like(x)
    E[1:] = np.cumsum(rho_of(mid) * np.diff(x)) / EPS
    return E


def phys(d, check):
    p, c = d["params"], d["correct_answers"]
    Na, Nd = p["Na"] * 1e6, p["Nd"] * 1e6                       # m^-3
    rho_p, rho_n = -E_CH * Na, E_CH * Nd
    Wg = p["W_given_nm"] * 1e-9
    rho_g, rho_o = (rho_p, rho_n) if p["given"] == "p" else (rho_n, rho_p)

    # ---- shooting: s from the given layer's outer edge (E = 0) through the junction (s = Wg) and beyond
    s = np.concatenate([np.linspace(0, Wg, 2001), Wg + np.geomspace(1e-15, 200.0 * Wg, 200000)])
    E = integrate_midpoint(lambda m: np.where(m < Wg, rho_g, rho_o), s)
    beyond = np.nonzero((s > Wg) & (np.sign(E) != np.sign(E[2000])))[0]
    check(len(beyond) > 0, "pn: E never returns to zero")
    if not len(beyond):
        return
    k = beyond[0]
    s0 = s[k - 1] - E[k - 1] * (s[k] - s[k - 1]) / (E[k] - E[k - 1])     # E is linear inside the cell
    W_other = (s0 - Wg) * 1e9
    check(abs(W_other - c["W_other"]) <= 1e-6 * c["W_other"], f"pn: other width {W_other} nm vs {c['W_other']}")

    # ---- full profile in the variant's orientation, junction exactly on a node
    Wp = (p["Wp_nm"] if p["given"] == "p" else W_other) * 1e-9
    Wn = (p["Wn_nm"] if p["given"] == "n" else W_other) * 1e-9
    if p["p_left"]:
        xl, xr, rho_l, rho_r = -Wp, Wn, rho_p, rho_n
    else:
        xl, xr, rho_l, rho_r = -Wn, Wp, rho_n, rho_p
    n = 100000
    x = np.concatenate([np.linspace(xl, 0.0, n + 1), np.linspace(0.0, xr, n + 1)[1:]])
    rho_of = lambda m: np.where(m < 0, rho_l, rho_r)
    Ex = integrate_midpoint(rho_of, x)
    check(abs(Ex[-1]) <= 1e-6 * np.max(np.abs(Ex)), "pn: E_x does not vanish at the far edge (not neutral)")
    imax = int(np.argmax(np.abs(Ex)))
    Emax = abs(Ex[imax])
    check(abs(Emax / 1e6 - c["Emax"]) <= 1e-6 * c["Emax"], f"pn: |E|max {Emax/1e6} vs {c['Emax']}")
    check(imax == n, f"pn: |E| peaks at x = {x[imax]}, not at the junction")
    V = np.zeros_like(x)
    V[1:] = -np.cumsum(0.5 * (Ex[1:] + Ex[:-1]) * np.diff(x))          # E is piecewise linear: trapezoid is exact
    Vbi = (V[-1] - V[0]) if p["p_left"] else (V[0] - V[-1])          # V(n edge) - V(p edge)
    check(abs(Vbi - c["Vbi"]) <= 1e-6 * c["Vbi"], f"pn: V_bi {Vbi} vs {c['Vbi']}")
    check(0.55 < Vbi < 0.9 and 0.5 < Emax / 1e6 < 15, f"pn: unrealistic V_bi {Vbi} or |E|max {Emax}")
    inner = Ex[1:-1]
    sign = np.sign(np.mean(inner))
    check(np.all(np.sign(inner) == sign), "pn: E_x changes sign inside the depletion region")
    # curvature: nonuniform second difference (exact for a parabola) has the sign of -rho, junction node excluded
    h1, h2 = np.diff(x)[:-1], np.diff(x)[1:]
    V2 = 2 * ((V[2:] - V[1:-1]) / h2 - (V[1:-1] - V[:-2]) / h1) / (h1 + h2)
    xi = x[1:-1]
    keep = xi != 0.0
    expect = -rho_of(xi[keep] - 1e-30 * np.sign(xi[keep])) / EPS
    check(np.allclose(V2[keep], expect, rtol=1e-3, atol=0), "pn: V'' != -rho/eps in the layers (not a parabola?)")
    v_mono = np.diff(V) * (1 if (p["p_left"]) else -1)               # going from p edge to n edge, V rises
    check(np.all(v_mono > 0), "pn: V not monotonic from p to n")

    # ---- MC (b): location; MC (d): direction
    loc = {x_["key"]: x_["correct"] for x_ in p["loc"]}
    check(loc == {"junction": "true", "p_edge": "false", "n_edge": "false", "uniform": "false", "mid_wide": "false"},
          f"pn: location MC keys {loc}")
    dirs = {x_["key"]: x_["correct"] for x_ in p["dirs"]}
    want = "plus" if sign > 0 else "minus"
    check(dirs.get(want) == "true" and sum(v == "true" for v in dirs.values()) == 1 and len(dirs) == 5,
          f"pn: direction MC, numerical sign {sign}, keys {dirs}")
    check(not any("nm" in x_["text"] or "W_p =" in x_["text"] for x_ in p["loc"]), "pn: location choices leak the widths")
    check(not any(ch.isdigit() for x_ in p["cb"] if x_["key"] == "nwider" for ch in x_["feedback"]), "pn: numbers in feedback")
    for lst in (p["loc"], p["dirs"]):
        check(len({x_["text"] for x_ in lst}) == len(lst) and all(x_["feedback"] for x_ in lst), "pn: MC texts/feedback")

    # ---- checkbox
    for x_ in p["cb"]:
        tv = (Wn > Wp) if x_["key"] == "nwider" else STATIC_TRUTH.get(x_["key"])
        check(tv is not None, f"pn: unknown statement {x_['key']}")
        check((x_["correct"] == "true") == bool(tv) and x_["tf"] == ("True" if tv else "False"),
              f"pn: statement {x_['key']} marked {x_['correct']}")
    check(len(p["cb"]) == 14, "pn: pool size")
    check((Wn > Wp) == (Nd < Na), "pn: the lighter-doped side is not the wider one")

    # ---- answer-panel numbers
    for key, sv in (("Emax", p["Emax_str"]), ("Vbi", p["Vbi_str"])):
        check(abs(float(sv) - c[key]) <= 1e-3 * c[key], f"pn: panel {key} {sv} vs {c[key]}")
    shown_other = p["Wn_str"] if p["given"] == "p" else p["Wp_str"]
    check(abs(float(shown_other) - c["W_other"]) <= 1e-3 * c["W_other"], "pn: panel width")
