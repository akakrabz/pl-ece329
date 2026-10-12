"""Independent check for midterm2/nested-solenoids-lenz.

No Ampere's law here. Each ideal solenoid is a stack of rings (n per metre) in the plane of the field point
and along the axis:
  * H by Biot-Savart: dl x R with np.cross, integrated along the axis analytically
    (int dz'/(rho^2 + z'^2)^{3/2} = 2/rho^2; the z'-odd part of dl x R vanishes) and around the ring by the
    periodic trapezoid rule;
  * the flux through a coaxial loop by Stokes, Psi = loop integral of A, with A from the Coulomb integral
    mu0/(4 pi) int I dl/|R| over a long coil (half-length 1 km: int dz'/|R| = 2 asinh(L/rho));
  * "counterclockwise as seen from the +axis side" is turned into a ring direction by an explicit viewer frame
    (forward = -axis, right = forward x up).
Checks H in the core with the given I2, the cancelling I2, H in the annulus (and zero outside), the Lenz key from
the sign of -dPsi/dt through an inner turn, and the loop emf by a finite difference in time.
"""
import math
import numpy as np

MU0 = 4e-7 * math.pi
AX = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]), "z": np.array([0.0, 0.0, 1.0])}
M = 2048
HALF = 1000.0                                                        # m, half-length used for A


def _frame(axis):
    ea = AX[axis]
    up = AX[{"x": "y", "y": "z", "z": "x"}[axis]]
    right = np.cross(-ea, up)                                          # viewer on the +axis side looks along -axis
    return ea, right, up


def _ring(axis, a, ccw):
    ea, right, up = _frame(axis)
    phi = (np.arange(M) + 0.5) * 2 * np.pi / M
    pts = a * (np.cos(phi)[:, None] * right + np.sin(phi)[:, None] * up)
    dl = (1.0 if ccw else -1.0) * a * (2 * np.pi / M) * (-np.sin(phi)[:, None] * right + np.cos(phi)[:, None] * up)
    return pts, dl


def _H(axis, n, I, a, ccw, r):
    """H (A/m) at distance r from the axis (on the viewer's 'right' direction) from an infinite solenoid."""
    ea, right, up = _frame(axis)
    pts, dl = _ring(axis, a, ccw)
    R = r * right - pts
    rho2 = (R * R).sum(1)
    return n * I / (4 * np.pi) * (2 * np.cross(dl, R) / rho2[:, None]).sum(0)


def _A(axis, n, I, a, ccw, r):
    ea, right, up = _frame(axis)
    pts, dl = _ring(axis, a, ccw)
    rho = np.sqrt(((r * right - pts) ** 2).sum(1))
    return MU0 * n * I / (4 * np.pi) * (dl * 2 * np.arcsinh(HALF / rho)[:, None]).sum(0)


def _flux(axis, n, I, a, ccw, r0, loop_ccw):
    """Psi through a coaxial loop of radius r0 traversed ccw/cw seen from +axis (Stokes: Psi = 2 pi r0 A_phi)."""
    ea, right, up = _frame(axis)
    t = up if loop_ccw else -up                                        # loop direction at the point r0*right
    return 2 * np.pi * r0 * float(np.dot(_A(axis, n, I, a, ccw, r0), t))


def phys(d, check):
    p, c = d["params"], d["correct_answers"]
    name = "nested-solenoids-lenz"
    ax = p["axis"]
    ccw1 = p["sense1"] == "counterclockwise"
    n1, n2, I1, I2 = p["n1"], p["n2"], p["I1"], p["I2"]
    a1, a2 = p["a1"] * 1e-2, p["a2"] * 1e-2
    ea = AX[ax]
    # I2 counted positive in I1's sense: a negative value is a ring of the same sense with negative current
    Hcore = _H(ax, n1, I1, a1, ccw1, 0.5 * a2) + _H(ax, n2, I2, a2, ccw1, 0.5 * a2)
    check(abs(np.dot(Hcore, ea) / 1e3 - c["Ha"]) <= 1e-7 * max(1.0, abs(c["Ha"])), f"{name}: Ha Biot-Savart {np.dot(Hcore, ea) / 1e3} vs {c['Ha']}")
    check(np.linalg.norm(Hcore - np.dot(Hcore, ea) * ea) < 1e-6, f"{name}: core field not axial")
    Hcore0 = _H(ax, n1, I1, a1, ccw1, 0.0) + _H(ax, n2, I2, a2, ccw1, 0.0)
    check(np.allclose(Hcore0, Hcore, atol=1e-6), f"{name}: core field not uniform")
    # (b) the cancelling current
    Hcancel = _H(ax, n1, I1, a1, ccw1, 0.5 * a2) + _H(ax, n2, c["I2c"], a2, ccw1, 0.5 * a2)
    check(np.linalg.norm(Hcancel) < 1e-6 * n1 * I1, f"{name}: I2c does not cancel the core field: {Hcancel}")
    # (c) annulus, and zero outside
    ran = 0.5 * (a1 + a2)
    Hann = _H(ax, n1, I1, a1, ccw1, ran) + _H(ax, n2, c["I2c"], a2, ccw1, ran)
    check(abs(np.dot(Hann, ea) / 1e3 - c["Hc"]) <= 1e-7 * max(1.0, abs(c["Hc"])), f"{name}: Hc Biot-Savart {np.dot(Hann, ea) / 1e3} vs {c['Hc']}")
    Hann2 = _H(ax, n1, I1, a1, ccw1, ran) + _H(ax, n2, I2, a2, ccw1, ran)
    check(np.allclose(Hann, Hann2, atol=1e-6), f"{name}: annulus field depends on I2")
    Hout = _H(ax, n1, I1, a1, ccw1, 1.5 * a1) + _H(ax, n2, I2, a2, ccw1, 1.5 * a1)
    check(np.linalg.norm(Hout) < 1e-6, f"{name}: field outside not zero {Hout}")
    # (d) Lenz: emf around one inner turn (in I1's sense) while I1 changes
    rate = 1.0 if p["change"] == "increases" else -1.0
    h = 1e-3
    dpsi = (_flux(ax, n1, I1 + rate * h, a1, ccw1, a2, ccw1) - _flux(ax, n1, I1 - rate * h, a1, ccw1, a2, ccw1)) / (2 * h)
    emf_turn = -dpsi                                                   # > 0 drives current in I1's sense
    for ch in p["lenz_choices"]:
        truth = {"opposite": emf_turn < 0, "same": emf_turn > 0}.get(ch["claim"], emf_turn == 0)
        check(truth == (ch["correct"] == "true"), f"{name}: Lenz claim {ch['claim']} truth {truth} vs key {ch['correct']} (emf {emf_turn})")
    # (e) loop emf, loop traversed in I1's sense, both currents ramping
    r0 = p["r0"] * 1e-2
    psi = lambda t: (_flux(ax, n1, I1 + p["al1"] * t, a1, ccw1, r0, ccw1) + _flux(ax, n2, I2 + p["al2"] * t, a2, ccw1, r0, ccw1))
    emf = -(psi(h) - psi(-h)) / (2 * h) * 1e6
    check(abs(emf - c["emf"]) <= 1e-6 * max(1.0, abs(emf)), f"{name}: loop emf {emf} vs {c['emf']} (region {p['region']})")
    check({"core": r0 < a2, "annulus": a2 < r0 < a1, "outside": r0 > a1}[p["region"]], f"{name}: r0 not in its region")
    # figure: end view from +axis (screen right x screen up = toward the viewer = +axis); radii to scale;
    # the arrowhead sits on top of the outer circle, and on screen a counterclockwise current moves left there
    rp = [float(p[k]) for k in ("r_px1", "r_px2", "r_px0")]
    check(abs(rp[1] - rp[0] * a2 / a1) < 0.11 and abs(rp[2] - rp[0] * r0 / a1) < 0.11, f"{name}: figure radii not to scale {rp}")
    tip, base = [tuple(map(float, q.split(","))) for q in p["arrow"].split()[:2]]
    check(abs(tip[1] - (110 - rp[0])) < 0.11, f"{name}: arrowhead not on top of the outer circle")
    screen_right, screen_up = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    toward_viewer = np.cross(screen_right, screen_up)               # the +axis direction in screen coordinates
    v_ccw_top = np.cross(toward_viewer, screen_up)                   # velocity of a ccw (right-handed) rotation at the top
    moves_left = tip[0] < base[0]
    check(moves_left == (ccw1 == (v_ccw_top[0] < 0)), f"{name}: figure arrow does not show I1's sense")
