"""Independent check for midterm2/vector-potential-sheet.

Rebuild A(r, t) in SI units from the statement's data (k, beta, mode, axes), then
  * B = curl A by central finite differences at P1 and P2 (all three components);
  * div A by finite differences at P1, P2 and on the plane u = 0 -> the MC key must be the "zero" claim;
  * the current: |u| case -> H just above/below u = 0 from FD curls, J_s = n x (H1 - H2) with np.cross,
    and curl H = 0 away from the plane; u^2 case -> J = curl(curl A)/mu0 by nested finite differences;
    magnitude vs the answer, direction vs every direction-MC claim;
  * E = -dA/dt (Phi = 0) by a finite difference in time, and Faraday's law curl E = -dB/dt as a sanity check.
"""
import math
import numpy as np

MU0 = 4e-7 * math.pi
IDX = {"x": 0, "y": 1, "z": 2}
EYE = np.eye(3)


def _A(p, r, t=0.0):
    k = (p["k"] + p["beta"] * t) * 1e-3                      # T (|u|) or T/m (u^2)
    uu = r[IDX[p["u"]]]
    f = abs(uu) if p["mode"] == "abs" else uu * uu
    return k * f * EYE[IDX[p["a"]]]                           # Wb/m


def _jac(F, r, h):
    J = np.zeros((3, 3))                                      # J[i, j] = dF_i / dx_j
    for j in range(3):
        dr = h * EYE[j]
        J[:, j] = (F(r + dr) - F(r - dr)) / (2 * h)
    return J


def _curl(F, r, h):
    J = _jac(F, r, h)
    return np.array([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]])


def _div(F, r, h):
    return float(np.trace(_jac(F, r, h)))


def phys(d, check):
    p, c = d["params"], d["correct_answers"]
    name = "vector-potential-sheet"
    iu, ia, ic = IDX[p["u"]], IDX[p["a"]], IDX[p["cax"]]
    check(len({iu, ia, ic}) == 3, f"{name}: axes not distinct")
    P1 = np.array([p["P1"][x] for x in "xyz"], float)
    P2 = np.array([p["P2"][x] for x in "xyz"], float)
    check(P1[iu] > 0 > P2[iu], f"{name}: P1/P2 not on the two sides")
    A = lambda r: _A(p, r)
    h = 1e-6
    # (b) B at both points, mT
    for tag, P in (("1", P1), ("2", P2)):
        B = _curl(A, P, h) * 1e3
        for x in "xyz":
            key = f"B{tag}{x}"
            check(abs(B[IDX[x]] - c[key]) <= 1e-6 * max(1.0, abs(B[IDX[x]])), f"{name}: {key} FD curl {B[IDX[x]]} vs {c[key]}")
    # (a) div A
    on_plane = P1.copy(); on_plane[iu] = 0.0
    divs = [_div(A, r, h) for r in (P1, P2, on_plane)]
    div_zero = all(abs(v) < 1e-12 for v in divs)
    check(div_zero, f"{name}: div A not zero: {divs}")
    for ch in p["div_choices"]:
        truth = (ch["claim"] == "zero") and div_zero
        check(truth == (ch["correct"] == "true"), f"{name}: div MC claim {ch['claim']} vs key {ch['correct']}")
    # (c) the current
    if p["mode"] == "abs":
        eps = 1e-3
        above, below = P1.copy(), P1.copy()
        above[iu], below[iu] = eps, -eps
        H1 = _curl(A, above, 1e-5) / MU0
        H2 = _curl(A, below, 1e-5) / MU0
        Jvec = np.cross(EYE[iu], H1 - H2) / 1e3               # kA/m
        Hcurl = lambda r: _curl(A, r, 1e-6) / MU0
        for P in (P1, P2):                                    # no volume current away from the plane
            vol = _curl(Hcurl, P, 1e-3)
            check(np.max(np.abs(vol)) < 1e-3, f"{name}: volume current away from the sheet {vol}")
        Jother = Jvec
    else:
        Hcurl = lambda r: _curl(A, r, 1e-3) / MU0
        Jvec = _curl(Hcurl, P1, 1e-3) / 1e3                    # kA/m^2
        Jother = _curl(Hcurl, P2, 1e-3) / 1e3
        check(np.allclose(Jvec, Jother, rtol=1e-6, atol=1e-9), f"{name}: J not uniform {Jvec} vs {Jother}")
    Jmag = float(np.linalg.norm(Jvec))
    check(abs(Jmag - c["Jmag"]) <= 1e-6 * Jmag, f"{name}: |J| {Jmag} vs {c['Jmag']}")
    jhat = Jvec / Jmag
    check(abs(jhat[iu]) < 1e-9 and abs(jhat[ic]) < 1e-9, f"{name}: J not along the a axis {jhat}")
    Aal = _A(p, P1)[ia]
    check(np.sign(jhat[ia]) == -np.sign(Aal), f"{name}: J should be antiparallel to A")
    for ch in p["dir_choices"]:
        v = ch["vec"]
        if isinstance(v, list):
            truth = np.allclose(np.array(v, float), jhat, atol=1e-9)
        elif v == "normal":
            truth = abs(jhat[iu]) > 0.5
        elif v == "alongB":                                   # "+-c_hat: along B": J would have to lie along c
            truth = abs(jhat[ic]) > 0.5
        elif v == "none":
            truth = Jmag < 1e-12
        else:                                                 # "reverses": J at P1 and P2 opposite
            truth = float(np.dot(Jvec, Jother)) < 0
        check(truth == (ch["correct"] == "true"), f"{name}: direction MC claim {v} truth {truth} vs key {ch['correct']}")
    # (e) E = -dA/dt at P1, mV/m
    ht = 1e-3
    Evec = -(_A(p, P1, ht) - _A(p, P1, -ht)) / (2 * ht) * 1e3
    check(abs(Evec[ia] - c["E1"]) <= 1e-6 * max(1.0, abs(Evec[ia])), f"{name}: E1 {Evec[ia]} vs {c['E1']}")
    check(abs(Evec[iu]) < 1e-12 and abs(Evec[ic]) < 1e-12, f"{name}: E has extra components")
    # Faraday sanity: curl E = -dB/dt at P1
    Ef = lambda r: -(_A(p, r, ht) - _A(p, r, -ht)) / (2 * ht)
    curlE = _curl(Ef, P1, 1e-6)
    dBdt = (_curl(lambda r: _A(p, r, ht), P1, 1e-6) - _curl(lambda r: _A(p, r, -ht), P1, 1e-6)) / (2 * ht)
    check(np.allclose(curlE, -dBdt, atol=1e-9), f"{name}: Faraday check failed {curlE} vs {-dBdt}")
