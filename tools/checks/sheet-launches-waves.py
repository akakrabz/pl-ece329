"""Independent check for midterm2/sheet-launches-waves.

Route (the slides' one, not the server's recipe): on each side an outgoing plane wave E = A g(t -+ n/v),
H = (1/eta) k x E with k = +-n-hat (radiation condition). The tangential amplitude vectors A+ and A- (4 unknowns) are
found with np.linalg.solve from the two boundary conditions at the sheet, tangential E continuous and
n-hat x (H+ - H-) = J_s, the coefficients being built with np.cross. Then:
  * signed components at the two points, compared with the answers;
  * Faraday and Ampere-Maxwell by centred finite differences of the 3-D curl (mu0 = 4 pi 1e-7, eps = eps_r/(mu0 c^2)
    with c = 3e8, consistent with eta0 = 120 pi);
  * the probe record scanned on a 1 ps grid for its first and last nonzero instants;
  * MC (a) from the numerical field directions on each side; MC (e) from the numerical snapshot at t1
    (support on each side, location of the maximum), and every choice's text checked against its numbers.
"""
import math
import re

import numpy as np

MU0 = 4e-7 * math.pi
C = 3.0e8
AX = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}


def _g(t, tr, th, tf):
    t = np.asarray(t, float)
    T = tr + th + tf
    out = np.where(t < tr, t / tr, np.where(t <= tr + th, 1.0, (T - t) / tf))
    return np.where((t <= 0) | (t >= T), 0.0, out)


def phys(d, check):
    p, c = d["params"], d["correct_answers"]
    n_hat, a_hat, h_hat = AX[p["n"]], AX[p["a"]], AX[p["h"]]
    sigma, eps_r, J0 = p["sigma"], p["eps_val"], p["J0_val"] * 1e-3      # J0 in A/m
    tr, th, tf = p["wave"]
    T = tr + th + tf
    eps = eps_r / (MU0 * C * C)
    v = 1 / math.sqrt(MU0 * eps)                  # m/s
    eta = math.sqrt(MU0 / eps)
    v_ns = v * 1e-9                               # m/ns
    check(abs(float(p["v"]) - v_ns) < 1e-9 and abs(float(p["eta"]) - eta) <= 1e-3 * eta, "sheet: v or eta")
    check((p["n"], p["a"]) not in (("y", "x"), ("x", "z")), "sheet: geometry repeats the exam or the site problem")
    check(tr != tf, "sheet: waveform must be asymmetric")

    # --- amplitudes from the boundary conditions
    tang = [AX[k] for k in "xyz" if k != p["n"]]
    M = np.zeros((4, 4))
    rhs = np.zeros(4)
    for j, tb in enumerate(tang):                 # unknown j: A+ along tb (col j), A- along tb (col 2 + j)
        for i, ti in enumerate(tang):
            M[i, j] += tb @ ti                    # E continuity:  A+ - A- = 0
            M[i, 2 + j] -= tb @ ti
            Hp = np.cross(n_hat, tb) / eta        # H+ per unit A+ (k = +n)
            Hm = np.cross(-n_hat, tb) / eta       # H- per unit A- (k = -n)
            M[2 + i, j] += np.cross(n_hat, Hp) @ ti
            M[2 + i, 2 + j] -= np.cross(n_hat, Hm) @ ti
    for i, ti in enumerate(tang):
        rhs[2 + i] = sigma * J0 * (a_hat @ ti)     # n x (H+ - H-) = J_s  (per unit g)
    sol = np.linalg.solve(M, rhs)
    Ap = sol[0] * tang[0] + sol[1] * tang[1]
    Am = sol[2] * tang[0] + sol[3] * tang[1]

    def EH(r, t_s):
        un = r @ n_hat
        if un > 0:
            E = Ap * float(_g((t_s - un / v) * 1e9, tr, th, tf))
            return E, np.cross(n_hat, E) / eta
        E = Am * float(_g((t_s + un / v) * 1e9, tr, th, tf))
        return E, np.cross(-n_hat, E) / eta

    # --- the two points
    for k, (un, t_ns) in enumerate(p["pts"]):
        r = un * n_hat + 0.3 * a_hat - 0.2 * h_hat
        E, H = EH(r, t_ns * 1e-9)
        Ea, Hh = float(E @ a_hat), float(H @ h_hat) * 1e3
        check(abs(Ea - c[f"E{k + 1}"]) <= 1e-9 * max(1, abs(Ea)), f"sheet: E at point {k + 1}: {Ea} vs {c[f'E{k + 1}']}")
        check(abs(Hh - c[f"H{k + 1}"]) <= 1e-9 * max(1, abs(Hh)), f"sheet: H at point {k + 1}: {Hh} vs {c[f'H{k + 1}']}")
        check(abs(Ea) > 0 and abs(Hh) > 0, f"sheet: point {k + 1} has no field")
        check(float(np.linalg.norm(E - Ea * a_hat)) < 1e-12 and float(np.linalg.norm(H * 1e3 - Hh * h_hat)) < 1e-9,
              "sheet: field has components the question does not ask for")
        check(float(np.cross(E, H) @ (np.sign(un) * n_hat)) > 0, "sheet: E x H not away from the sheet")
        # Maxwell by finite differences
        dz, dt = 1e-5, 1e-5 / v
        cE, cH = np.zeros(3), np.zeros(3)
        dEd, dHd = {}, {}
        for i in range(3):
            e = np.zeros(3); e[i] = dz
            Ep, Hp_ = EH(r + e, t_ns * 1e-9)
            Em, Hm_ = EH(r - e, t_ns * 1e-9)
            dEd[i], dHd[i] = (Ep - Em) / (2 * dz), (Hp_ - Hm_) / (2 * dz)
        for cu, dF in ((cE, dEd), (cH, dHd)):
            cu[0] = dF[1][2] - dF[2][1]
            cu[1] = dF[2][0] - dF[0][2]
            cu[2] = dF[0][1] - dF[1][0]
        E2_, H2_ = EH(r, t_ns * 1e-9 + dt)
        E0_, H0_ = EH(r, t_ns * 1e-9 - dt)
        dEdt, dHdt = (E2_ - E0_) / (2 * dt), (H2_ - H0_) / (2 * dt)
        sc = J0 * eta / (2 * v * 1e-9) + 1e-30
        check(np.allclose(cE, -MU0 * dHdt, rtol=1e-5, atol=1e-7 * sc), f"sheet: Faraday fails at point {k + 1}")
        check(np.allclose(cH, eps * dEdt, rtol=1e-5, atol=1e-7 * sc / eta), f"sheet: Ampere-Maxwell fails at point {k + 1}")
    check(np.sign(p["pts"][0][0]) == -np.sign(p["pts"][1][0]), "sheet: the two points must be on opposite sides")

    # --- boundary conditions reproduced at the sheet
    for tq in (0.3 * tr, tr + 0.5 * th + 0.01, T - 0.3 * tf):
        Ep, Hp_ = EH(1e-12 * n_hat, tq * 1e-9)
        Em, Hm_ = EH(-1e-12 * n_hat, tq * 1e-9)
        Js = sigma * J0 * float(_g(tq, tr, th, tf)) * a_hat
        check(np.allclose(Ep, Em, atol=1e-9) and np.allclose(np.cross(n_hat, Hp_ - Hm_), Js, atol=1e-9), "sheet: BCs")

    # --- probe record
    un_q = p["probe"]
    tt = np.arange(0, 60000) * 1e-3               # ns, 1 ps grid up to 60 ns
    dist = abs(un_q)
    gq = _g(tt - dist / v_ns, tr, th, tf)
    nz = np.nonzero(gq)[0]
    check(len(nz) > 0, "sheet: probe never sees the pulse")
    if len(nz):
        check(abs(tt[nz[0]] - c["t_on"]) <= 2e-3 and abs(tt[nz[-1]] - c["t_off"]) <= 2e-3,
              f"sheet: probe interval ({tt[nz[0]]}, {tt[nz[-1]]}) vs ({c['t_on']}, {c['t_off']})")

    # --- MC (a): directions while the fields are nonzero
    t_mid = 0.5 * tr
    dirs_num = []
    for side in (1, -1):
        dd = 0.4
        E, H = EH(side * dd * n_hat, (t_mid + dd / v_ns) * 1e-9)
        dirs_num += [int(np.sign(H @ h_hat)), int(np.sign(E @ a_hat))]
    ch = p["dir_choices"]
    check(sum(x["correct"] == "true" for x in ch) == 1 and len({x["text"] for x in ch}) == len(ch), "sheet: MC (a)")
    for x in ch:
        check((list(x["dirs"]) == dirs_num) == (x["correct"] == "true"), f"sheet: direction choice {x['dirs']} vs {dirs_num}")
        found = re.findall(r"\\mathbf\{H\}\$ along \$([+-])\\hat\{(\w)\}\$, \$\\mathbf\{E\}\$ along \$([+-])\\hat\{(\w)\}\$", x["text"])
        txt = []
        for hs, ha, es, ea in found:
            check(ha == p["h"] and ea == p["a"], "sheet: direction choice uses the wrong axes")
            txt += [1 if hs == "+" else -1, 1 if es == "+" else -1]
        check(txt == list(x["dirs"]), f"sheet: direction choice text {x['text']} vs data {x['dirs']}")

    # --- MC (e): the snapshot at t1
    t1 = p["pts"][0][1]
    ugrid = np.arange(-40000, 40001) * 5e-4        # m, +-20 m
    mag = np.where(ugrid > 0, _g(t1 - ugrid / v_ns, tr, th, tf), _g(t1 + ugrid / v_ns, tr, th, tf))
    geom = {}
    for side in (1, -1):
        sel = (np.sign(ugrid) == side) & (mag > 0)
        dists = np.abs(ugrid[sel])
        mx = mag.max()
        top = np.abs(ugrid[(np.sign(ugrid) == side) & (mag >= mx - 1e-9)])
        geom[side] = (dists.min(), dists.max(), top.min(), top.max()) if len(dists) else None
    check(geom[1] is not None and geom[-1] is not None, "sheet: snapshot empty on one side")
    sk = p["sketch_choices"]
    check(sum(x["correct"] == "true" for x in sk) == 1 and len({x["text"] for x in sk}) == len(sk), "sheet: MC (e)")
    tol = 2e-3
    for x in sk:
        gm = x["geom"]
        if gm.get("zero"):
            match = False
            check("zero everywhere" in x["text"], "sheet: zero choice text")
        else:
            ok_side = lambda gs: gs is not None and all(abs(a - b) <= tol for a, b in
                                                        zip(gs, (gm["lo"], gm["hi"], gm["pk_lo"], gm["pk_hi"])))
            match = ok_side(geom[1]) and (ok_side(geom[-1]) if gm["both"] else geom[-1] is None)
            nums = [float(s) for s in re.findall(r"(-?\d+(?:\.\d+)?)\\ \\mathrm\{m\}", x["text"])]
            want = [gm["lo"], gm["hi"]] + ([gm["pk_lo"]] if gm["pk_lo"] == gm["pk_hi"] else [gm["pk_lo"], gm["pk_hi"]])
            check(len(nums) == len(want) and all(abs(a - b) < 1e-4 for a, b in zip(nums, want)),
                  f"sheet: sketch text {x['text']} vs {want}")
        check(match == (x["correct"] == "true"), f"sheet: sketch choice {x['text'][:60]} (numerical {geom})")
