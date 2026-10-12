"""Independent check for midterm2/moving-loop-gradient.

Rebuild the full field B = sig B0 [(1 + g/L) b_hat + (b/L) g_hat] (both components) from the params and
  * integrate B . dS over the square (40 x 40 midpoint grid; a viewer frame turns "counterclockwise as seen
    from +b" into an explicit dS direction) at t1 and t1 +- h: Psi(t1), emf = -dPsi/dt, I = emf/R;
  * recompute the emf a second way, as the line integral of (v x B) . dl around the moving edges (np.cross);
  * check that the emf is the same at another time (time-dependence MC) and verify the sense MC from its sign;
  * for every displayed orientation/motion statement, integrate the flux through that (possibly tilted)
    square at two times and decide "no current" from whether it changes;
  * verify the figure: the view is from +b (right x up = +b_hat), the C arrow and the v arrow point the right
    way, the field markers are dots (out of the page) iff the field points along +b, and they grow toward +g.
"""
import numpy as np

IDX = {"x": 0, "y": 1, "z": 2}
EYE = np.eye(3)


def _B(p, r):
    """uT at points r (N x 3), metres."""
    ib, ig = IDX[p["b"]], IDX[p["g"]]
    out = np.zeros_like(r)
    out[:, ib] = p["sig"] * p["B0"] * (1 + r[:, ig] / p["L"])
    out[:, ig] = p["sig"] * p["B0"] * r[:, ib] / p["L"]
    return out


def _plane_basis(nhat):
    a = EYE[0] if abs(nhat[0]) < 0.9 else EYE[1]
    e1 = np.cross(nhat, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(nhat, e1)


def _flux(p, centre, nhat, N=40):
    e1, e2 = _plane_basis(nhat)
    s = p["s"]
    u = (np.arange(N) + 0.5) / N * s - s / 2
    U, V = np.meshgrid(u, u)
    pts = centre + U.reshape(-1, 1) * e1 + V.reshape(-1, 1) * e2
    return float((_B(p, pts) @ nhat).sum() * (s / N) ** 2)          # uT m^2 = uWb


def _viewer_frame(axis_vec):
    """viewer on the +axis side looking back: returns (right, up) with right x up = +axis."""
    up = EYE[0] if abs(axis_vec[0]) < 0.9 else EYE[1]
    up = up - np.dot(up, axis_vec) * axis_vec
    up /= np.linalg.norm(up)
    right = np.cross(-axis_vec, up)
    return right, up


def _motional(p, centre, vel, ccw_about, N=400):
    """loop integral of (v x B) . dl (uV) around the square in the plane normal to ccw_about, traversed
    counterclockwise as seen from the +ccw_about side."""
    right, up = _viewer_frame(ccw_about)
    s = p["s"]
    corners = [centre + s / 2 * (sx * right + sy * up) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    tot = 0.0
    for k in range(4):                                               # (-,-) -> (+,-) -> (+,+) -> (-,+): ccw on screen
        a, b = corners[k], corners[(k + 1) % 4]
        f = (np.arange(N) + 0.5) / N
        pts = a + f[:, None] * (b - a)
        dl = (b - a) / N
        vxB = np.cross(np.broadcast_to(vel, pts.shape), _B(p, pts))
        tot += float((vxB @ dl).sum())
    return tot


def phys(d, check):
    p, c = d["params"], d["correct_answers"]
    name = "moving-loop-gradient"
    bh, gh, th = EYE[IDX[p["b"]]], EYE[IDX[p["g"]]], EYE[IDX[p["t"]]]
    check(len({p["b"], p["g"], p["t"]}) == 3, f"{name}: axes not distinct")
    # field sanity: divergence and curl vanish (finite differences)
    r0 = np.array([0.3, -0.2, 0.4])
    J = np.zeros((3, 3))
    for j in range(3):
        J[:, j] = (_B(p, (r0 + 1e-4 * EYE[j])[None])[0] - _B(p, (r0 - 1e-4 * EYE[j])[None])[0]) / 2e-4
    check(abs(np.trace(J)) < 1e-9 and np.allclose(J, J.T, atol=1e-9), f"{name}: field not div- and curl-free")
    vel = p["v"] * gh
    centre = lambda t: (p["g0"] + p["v"] * t) * gh
    ccw = p["Ccw"] == "counterclockwise"
    # dS from the circulation by the right-hand rule, via the viewer frame: ccw seen from +b <-> dS along +b
    right, up = _viewer_frame(bh)
    dS = np.cross(right, up) * (1 if ccw else -1)
    t1, h = p["t1"], 1e-3
    Psi = _flux(p, centre(t1), dS)
    check(abs(Psi - c["Psi"]) <= 1e-7 * max(1.0, abs(Psi)), f"{name}: Psi {Psi} vs {c['Psi']}")
    emf = -(_flux(p, centre(t1 + h), dS) - _flux(p, centre(t1 - h), dS)) / (2 * h)
    check(abs(emf - c["emf"]) <= 1e-6 * max(1.0, abs(emf)), f"{name}: emf {emf} vs {c['emf']}")
    emf_mot = _motional(p, centre(t1), vel, bh) * (1 if ccw else -1)
    check(abs(emf_mot - c["emf"]) <= 1e-6 * max(1.0, abs(emf_mot)), f"{name}: motional emf {emf_mot} vs {c['emf']}")
    check(abs(c["I"] - emf / p["R"]) <= 1e-6 * max(1.0, abs(c["I"])), f"{name}: I {c['I']} vs {emf / p['R']}")
    # time dependence: emf at another time
    t2 = t1 + 0.3
    emf2 = -(_flux(p, centre(t2 + h), dS) - _flux(p, centre(t2 - h), dS)) / (2 * h)
    Bc = lambda t: float(_B(p, centre(t)[None])[0] @ bh)
    for ch in p["time_choices"]:
        truth = {"constant": abs(emf2 - emf) < 1e-9 * max(1.0, abs(emf)) and abs(emf) > 0,
                 "follows_B": abs(emf2 / emf - Bc(t2) / Bc(t1)) < 1e-9,
                 "zero": abs(emf) < 1e-12,
                 "reverses": False}[ch["claim"]]
        if ch["claim"] == "reverses":                               # compare the emf on both sides of g = 0
            ta, tb = (-0.3 - p["g0"]) / p["v"], (0.3 - p["g0"]) / p["v"]
            ea = -(_flux(p, centre(ta + h), dS) - _flux(p, centre(ta - h), dS)) / (2 * h)
            eb = -(_flux(p, centre(tb + h), dS) - _flux(p, centre(tb - h), dS)) / (2 * h)
            truth = ea * eb < 0
        check(truth == (ch["correct"] == "true"), f"{name}: time MC claim {ch['claim']} truth {truth}")
    # sense of the induced current, seen from +b
    emf_ccw = _motional(p, centre(t1), vel, bh)
    for ch in p["sense_choices"]:
        truth = {"ccw": emf_ccw > 0, "cw": emf_ccw < 0, "none": emf_ccw == 0, "reverses": False}[ch["claim"]]
        check(truth == (ch["correct"] == "true"), f"{name}: sense MC claim {ch['claim']} truth {truth}")
    # (a) orientation statements
    ntrue = 0
    for ch in p["orient"]:
        n = np.array(ch["n"], float)
        vv = np.array(ch["vel"], float)
        cen = lambda t: p["g0"] * gh + vv * t
        dpsi = _flux(p, cen(0.5), n) - _flux(p, cen(0.0), n)
        nocur = abs(dpsi) < 1e-9
        ntrue += nocur
        check(nocur == (ch["correct"] == "true"), f"{name}: orientation statement truth {nocur} vs key {ch['correct']}: {ch['text'][:70]}")
        check(abs(np.linalg.norm(vv) - abs(p["v"])) < 1e-12, f"{name}: statement speed differs from |v|")
    check(1 <= ntrue <= 4 and len(p["orient"]) == 5, f"{name}: checkbox needs 5 statements with 1-4 true ({ntrue})")
    check(p["k_true"] == ntrue, f"{name}: max-correct ({p['k_true']}) must equal the number of true statements ({ntrue})")
    # figure semantics: right = +g, up = +-t; the view must be from +b
    fig_up = th if p["fig_up_is_plus_t"] else -th
    check(np.allclose(np.cross(gh, fig_up), bh), f"{name}: figure is not the view from +b")
    pts = [tuple(map(float, q.split(","))) for q in p["c_arrow"].split()]
    check((pts[0][0] < pts[1][0]) == ccw, f"{name}: C arrow on the top edge points the wrong way")
    vt = float(p["v_head"].split()[0].split(",")[0])
    check((vt > float(p["v_x2"])) == (p["v"] > 0), f"{name}: velocity arrow points the wrong way")
    tt = float(p["t_head"].split()[0].split(",")[1])
    check((tt < min(float(p["t_y1"]), float(p["t_y2"]))) == p["fig_up_is_plus_t"], f"{name}: t-axis arrow wrong")
    for mk in p["marks"]:
        check(bool(mk["dot"]) == (p["sig"] > 0) and bool(mk["cross"]) == (p["sig"] < 0), f"{name}: field markers show the wrong direction")
    radii = [float(mk["r"]) for mk in sorted(p["marks"], key=lambda m: m["x"])]
    check(all(r2 >= r1 for r1, r2 in zip(radii, radii[1:])) and radii[-1] > radii[0], f"{name}: markers do not grow toward +g")
