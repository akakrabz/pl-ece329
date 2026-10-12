"""Independent check for questions/electrostatics/lorentz-force-work.

Recomputes, in SI units, from the integer data in params:
  * F with np.cross (server: hand-expanded integer cross product, unit shortcut mN);
  * P as the rate of change of kinetic energy over a short step of Newton's law with an arbitrary mass
    (central difference of KE, not a dot product formula);
  * the truth of every displayed checkbox statement, from small numerical experiments (RK4 orbits in a pure
    magnetic field, cross products of random vectors, ...) rather than from the server's labels;
  * the velocity selector: scans all six injection directions, solves for the speed that zeroes the force by
    least squares, and requires exactly one direction (with positive speed) to work.
"""
import math

import numpy as np

_rng = np.random.default_rng(12345)


def _rk4_pure_B(q, m, v0, B, T, n=4000):
    """Integrate m dv/dt = q v x B (uniform B) with RK4; return the velocity history."""
    h = T / n
    v = np.array(v0, float)
    hist = [v.copy()]
    f = lambda vv: (q / m) * np.cross(vv, B)
    for _ in range(n):
        k1 = f(v); k2 = f(v + 0.5 * h * k1); k3 = f(v + 0.5 * h * k2); k4 = f(v + h * k3)
        v = v + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        hist.append(v.copy())
    return np.array(hist)


def _experiments():
    """Physics facts established numerically once (statement id -> truth)."""
    t = {}
    vs, Bs = _rng.normal(size=(50, 3)), _rng.normal(size=(50, 3))
    cr = np.cross(vs, Bs)
    t["perp_v"] = bool(np.all(np.abs(np.einsum("ij,ij->i", cr, vs)) < 1e-12))
    t["perp_B"] = bool(np.all(np.abs(np.einsum("ij,ij->i", cr, Bs)) < 1e-12))
    t["along_B"] = bool(np.allclose(np.cross(cr, Bs), 0))      # is q v x B parallel to B?
    t["no_work"] = t["perp_v"]                                # power of q v x B is q (v x B).v
    t["mag_power"] = not t["no_work"]                         # "part of P is magnetic"
    # orbit in a pure, strong magnetic field: speed fixed, direction changes
    q, m, B = 1.6e-19, 9.1e-31, np.array([0.3, -1.2, 2.0])
    hist = _rk4_pure_B(q, m, [2e5, 1e5, -3e5], B, T=2 * math.pi * m / (q * np.linalg.norm(B)) * 0.37)
    speeds = np.linalg.norm(hist, axis=1)
    speed_const = np.ptp(speeds) < 1e-8 * speeds[0]
    dir_changes = np.linalg.norm(hist[-1] - hist[0]) > 1e-2 * speeds[0]
    t["dir_not_speed"] = bool(speed_const and dir_changes)
    t["speeds_up"] = bool(not speed_const)
    t["no_change_v"] = bool(not dir_changes)
    t["rest"] = bool(np.allclose(np.cross(np.zeros(3), Bs[0] * 1e6), 0))
    t["parallel"] = bool(np.allclose(np.cross(2.5 * Bs[1], Bs[1]), 0) and np.allclose(np.cross(-Bs[2], Bs[2]), 0))
    # |v x B| for v parallel to B versus perpendicular
    b = Bs[3]; vperp = np.cross(b, vs[3]); vperp *= np.linalg.norm(b) / np.linalg.norm(vperp)
    t["max_parallel"] = bool(np.linalg.norm(np.cross(b, b)) >= np.linalg.norm(np.cross(vperp, b)))
    # negative charge: force . (v x B) < 0 for every non-parallel pair
    t["neg_along"] = bool(np.all(np.einsum("ij,ij->i", -1.0 * cr, cr) > 0))
    # definitions: F/q at v = 0 is E; for a moving charge F/q differs from E whenever v x B != 0
    E = _rng.normal(size=3)
    t["E_def_rest"] = bool(np.allclose((E + np.cross(np.zeros(3), Bs[4])), E))
    t["E_def_moving"] = bool(np.allclose(E + np.cross(vs[4], Bs[4]), E))
    return t


_FACTS = _experiments()


def _unit(sign, ax):
    u = np.zeros(3); u["xyz".index(ax)] = sign
    return u


def phys(d, check):
    p, c = d["params"], d["correct_answers"]
    # ------------------------------------------------ (a) force, SI units
    q = p["q"] * 1e-6
    E = np.array(p["E_list"], float) * 1e3
    v = np.array(p["v_list"], float) * 1e3
    B = np.array(p["B_list"], float)
    F = q * (E + np.cross(v, B))
    for i, k in enumerate(("Fx", "Fy", "Fz")):
        check(abs(F[i] * 1e3 - c[k]) <= 1e-9 * max(1, abs(F[i] * 1e3)), f"lorentz: {k} {c[k]} vs {F[i]*1e3}")
    check(np.any(np.abs(np.cross(v, B)) > 0), "lorentz: v x B is zero (statements about the magnetic force become vacuous)")
    # ------------------------------------------------ (b) power from the kinetic energy change of a short Newton step
    m, h = 1e-9, 1e-6                      # arbitrary mass (kg) and step (s); KE is quadratic, so the central difference is exact
    a = F / m
    KE = lambda vv: 0.5 * m * float(np.dot(vv, vv))
    P_num = (KE(v + a * h) - KE(v - a * h)) / (2 * h)
    check(abs(P_num - c["P"]) <= 1e-6 * max(1, abs(P_num)), f"lorentz: P {c['P']} vs dKE/dt {P_num}")
    check(abs(float(np.dot(q * np.cross(v, B), v))) <= 1e-9, "lorentz: magnetic power is not zero")
    check(abs(c["P"] - p["q"] * p["Edotv"]) < 1e-9, "lorentz: P != q E.v")
    # ------------------------------------------------ (c) statements
    facts = p["facts"]
    check(len(facts) == 5, f"lorentz: {len(facts)} statements shown, expected 5")
    ntrue = sum(f["correct"] == "true" for f in facts)
    check(1 <= ntrue <= 4, f"lorentz: {ntrue} true statements (need 1..4)")
    check(len({f["text"] for f in facts}) == 5, "lorentz: duplicate statements")
    for f in facts:
        if f["id"] == "speeding_up":
            truth = P_num > 1e-6
        elif f["id"] == "slowing_down":
            truth = P_num < -1e-6
        else:
            truth = _FACTS.get(f["id"])
        check(truth is not None, f"lorentz: statement id {f['id']} unknown to the check")
        check((f["correct"] == "true") == bool(truth), f"lorentz: statement {f['id']} keyed {f['correct']}, physics says {truth}")
        check(f["verdict"] == ("True" if truth else "False"), f"lorentz: verdict text wrong for {f['id']}")
    # ------------------------------------------------ (d) velocity selector: scan the six injection directions
    E0, B0 = p["E0"] * 1e3, p["B0"] * 1e-3
    Es, Bs = E0 * _unit(p["sE"], p["axE"]), B0 * _unit(p["sB"], p["axB"])
    working = []
    for ax in "xyz":
        for sgn in (1, -1):
            dvec = _unit(sgn, ax)
            w = np.cross(dvec, Bs)                     # v x B per unit speed
            if np.linalg.norm(w) < 1e-15:
                continue
            s = -float(np.dot(Es, w)) / float(np.dot(w, w))       # least-squares speed
            if s > 0 and np.linalg.norm(Es + s * w) < 1e-9 * E0:
                working.append((sgn, ax, s))
    check(len(working) == 1, f"lorentz: {len(working)} injection directions work, expected exactly 1")
    if len(working) == 1:
        sgn, ax, s = working[0]
        check(abs(s / 1e3 - c["vsel"]) <= 1e-9 * s, f"lorentz: selected speed {c['vsel']} vs {s/1e3}")
        want = ("+" if sgn > 0 else "-") + "\\hat{" + ax + "}"
        corr = [x for x in p["dir_choices"] if x["correct"] == "true"]
        check(len(corr) == 1 and corr[0]["text"] == f"${want}$", f"lorentz: direction key {corr} vs {want}")
        for x in p["dir_choices"]:
            if x["text"].startswith("either"):
                check(x["correct"] == "false", "lorentz: 'either direction' keyed true")
        # (e): the same velocity gives zero force for any charge and sign (mass never enters F)
        vsel = s * _unit(sgn, ax)
        for qq in (1.6e-19, -1.6e-19, 3.2e-19, -5e-18):
            check(np.linalg.norm(qq * (Es + np.cross(vsel, Bs))) < 1e-9 * abs(qq) * E0, "lorentz: selector depends on q")
    check(len(p["dir_choices"]) == 5 and len({x["text"] for x in p["dir_choices"]}) == 5, "lorentz: direction choices")
