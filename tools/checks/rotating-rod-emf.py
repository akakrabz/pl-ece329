"""Independent check for rotating-rod-emf (Lecture 14).

Routes:
  * emf: midpoint-rule line integral of (v x B) . dl along the rod, with v = w x r and every product taken with
    np.cross, at several angular positions (pivot -> tip);
  * second route: the flux rule for the closed path rod -> rail arc -> lead back to the pivot, with the sector flux
    computed as a polygon area at two times and differenced;
  * power: torque of the forces I dl x B on the rod's current (np.cross) dotted with w, compared with I^2 R;
  * MC and checkbox keys recomputed from those numbers (and from variations of L, w, R, B).
"""
import math

import numpy as np

AX = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}


def basis(axis):
    e = AX[axis]
    e1 = np.roll(e, 1)                 # z -> x, x -> y, y -> z
    e2 = np.cross(e, e1)               # e1 x e2 = e
    return e, e1, e2


def emf_line(wv, Bv, L, rhat, n=4000):
    s = (np.arange(n) + 0.5) * L / n
    r = s[:, None] * rhat
    v = np.cross(wv, r)
    f = np.cross(v, Bv)
    return float(np.sum(f @ rhat) * L / n)


def torque_on_current(Bv, L, rhat, I_along_rhat, n=4000):
    s = (np.arange(n) + 0.5) * L / n
    dl = np.tile(rhat * (L / n), (n, 1))
    dF = I_along_rhat * np.cross(dl, Bv)
    return np.sum(np.cross(s[:, None] * rhat, dF), axis=0)


def sector_flux(Bv, e, e1, e2, L, th_c, th, n=20000):
    """Flux through the closed path pivot -> tip (angle th) -> rail back to angle th_c -> pivot, by the
    right-hand rule: polygon (shoelace) area with orientation, times B . e."""
    angs = np.concatenate([[th], np.linspace(th, th_c, n)])
    pts = np.stack([np.zeros(2)] + [L * np.array([math.cos(a), math.sin(a)]) for a in angs[1:]])
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)      # > 0 if counterclockwise about +e
    return float(np.dot(Bv, e)) * area


def phys(d, check):
    p, c = d["params"], d["correct_answers"]
    e, e1, e2 = basis(p["axis"])
    L = p["L_cm"] * 1e-2
    wv = p["sw"] * p["omega"] * e
    Bv = p["sb"] * p["B_mT"] * 1e-3 * e
    R = float(p["R"])

    emfs = []
    for th in (0.0, 0.7, 2.1, 3.9, 5.5):
        rhat = math.cos(th) * e1 + math.sin(th) * e2
        emfs.append(emf_line(wv, Bv, L, rhat))
    emf = emfs[0]
    check(max(emfs) - min(emfs) < 1e-12 * abs(emf), "rod: emf depends on the angle?")
    check(abs(abs(emf) * 1e3 - c["emf"]) <= 1e-9 * abs(emf) * 1e3, f"rod: |emf| {abs(emf) * 1e3} vs {c['emf']} mV")

    # flux rule on the closed path, contact point of the lead on the rail at angle th_c
    th_c, th, dt = 0.3, 1.2, 1e-6
    w_e = float(np.dot(wv, e))                     # angular velocity about +e
    psi1 = sector_flux(Bv, e, e1, e2, L, th_c, th)
    psi2 = sector_flux(Bv, e, e1, e2, L, th_c, th + w_e * dt)
    emf_flux = -(psi2 - psi1) / dt                 # around the path, i.e. pivot -> tip along the rod
    check(abs(emf_flux - emf) < 1e-6 * abs(emf), f"rod: flux rule {emf_flux} vs line integral {emf}")

    tip_plus = emf > 0
    ch = p["choices"]
    check(sum(x["correct"] == "true" for x in ch) == 1, "rod MC: exactly one correct")
    corr = [x["key"] for x in ch if x["correct"] == "true"][0]
    check(corr == ("tip" if tip_plus else "pivot"), f"rod MC: correct is {corr}, physics says tip_plus={tip_plus}")
    check(p["tip_plus"] == tip_plus, "rod: params tip_plus inconsistent")

    I = abs(emf) / R
    check(abs(I * 1e3 - c["I"]) <= 1e-9 * I * 1e3, "rod: current")
    rhat = e1
    tau = torque_on_current(Bv, L, rhat, I if tip_plus else -I)    # current flows toward the + end inside the rod
    P_mech = float(np.dot(-tau, wv))                                 # applied torque = -magnetic torque
    check(P_mech > 0 and float(np.dot(tau, wv)) < 0, "rod: magnetic torque should brake the rotation")
    check(abs(P_mech * 1e3 - c["P"]) <= 1e-6 * P_mech * 1e3, f"rod: power from torque {P_mech * 1e3} vs {c['P']} mW")
    check(abs(I * I * R - P_mech) <= 1e-6 * P_mech, "rod: I^2 R != mechanical power")

    # checkbox truths
    def emf_of(Lx=L, wx=wv, Bx=Bv, th=0.0):
        return emf_line(wx, Bx, Lx, math.cos(th) * e1 + math.sin(th) * e2)

    truths = {
        "motional": abs(emf_flux - emf) < 1e-6 * abs(emf),     # static B: the flux change is all motional
        "quadruple": abs(emf_of(Lx=2 * L) / emf - 4) < 1e-9,
        "tip_speed": abs((np.linalg.norm(wv) * L) * np.linalg.norm(Bv) * L / abs(emf) - 2) < 1e-9,
        "reverse": np.sign(emf_of(wx=-wv)) == -np.sign(emf),
        "swept": abs(np.linalg.norm(Bv) * 0.5 * L * L * np.linalg.norm(wv) - abs(emf)) < 1e-9 * abs(emf),
        "axial_only": abs(emf_of(Bx=Bv + 0.37 * e1 - 0.21 * e2) - emf) < 1e-9 * abs(emf),
        "open_no_torque": np.allclose(torque_on_current(Bv, L, rhat, 0.0), 0)
        and np.allclose(np.cross(0.3 * L * rhat, 1.7 * rhat), 0),
        "rod_dir": tip_plus,
        "zero_uniform": abs(emf) < 1e-12,
        "lenz_faster": float(np.dot(tau, wv)) > 0,
        "ac": len({np.sign(x) for x in emfs}) > 1,
        "more_R": (emf ** 2 / (2 * R)) > (emf ** 2 / R),
        "angle": max(emfs) - min(emfs) > 1e-9 * abs(emf),
    }
    facts = p["facts"]
    nt = sum(f["correct"] == "true" for f in facts)
    check(len(facts) == 5 and 1 <= nt <= 4, f"rod checkbox: {len(facts)} statements, {nt} true")
    for f in facts:
        check(bool(truths[f["key"]]) == (f["correct"] == "true"),
              f"rod checkbox: {f['key']} marked {f['correct']}, recomputed {truths[f['key']]}")
