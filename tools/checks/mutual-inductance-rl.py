"""Independent check for mutual-inductance-rl (Lecture 15).

Routes:
  * M from Neumann's formula, M = (mu0/4pi) sum over loop pairs of the double line integral of dl1.dl2/|r1 - r2|:
    for coaxial circular loops this is (mu0 a b / 2) int cos(psi) / sqrt(a^2 + b^2 + z^2 - 2ab cos psi) dpsi
    (periodic trapezoid rule), summed over the solenoid's winding as n1 * integral dz over an infinitely long coil
    (Gauss-Legendre after z = a1 tan(theta)). No flux-times-area formula is used. Reciprocity: the same sum with
    the roles of the two coils swapped.
  * L1 from the field energy 1/2 mu0 H^2 x volume (checks the value printed in the statement);
  * the RL switch-on by RK4 time stepping of L1 dI/dt = V0 - R I (no exponential formula); tau read off as the
    time at which I = (1 - 1/e) V0/R; the stored energy as the time integral of the inductor's power L1 I dI/dt;
  * MC keys and checkbox truths recomputed from those (plus a non-coaxial Neumann integral for the shifted coil).
"""
import math

import numpy as np

MU0 = 4e-7 * math.pi
_GL_X, _GL_W = np.polynomial.legendre.leggauss(200)


def m_coax(a, b, z, n=512):
    """Neumann mutual inductance of two coaxial circular loops of radii a, b, axial separation z (array)."""
    psi = 2 * np.pi * np.arange(n) / n
    z = np.atleast_1d(z)[:, None]
    integ = np.cos(psi) / np.sqrt(a * a + b * b + z * z - 2 * a * b * np.cos(psi))
    return MU0 * a * b / 2 * integ.sum(axis=1) * (2 * np.pi / n)


def m_solenoid_coil(n1, a1, N2, a2):
    """n1 turns/m of radius a1 over -inf < z < inf, against N2 coaxial turns of radius a2 at z = 0."""
    th = _GL_X * math.pi / 2
    z = a1 * np.tan(th)
    dz = a1 / np.cos(th) ** 2 * (math.pi / 2)
    return N2 * n1 * float(np.sum(_GL_W * m_coax(a1, a2, z) * dz))


def m_shifted(n1, a1, N2, a2, shift, n=256):
    """Same, but with the coil's axis displaced sideways by `shift` (general double line integral)."""
    th = _GL_X * math.pi / 2
    zs = a1 * np.tan(th)
    dzs = a1 / np.cos(th) ** 2 * (math.pi / 2)
    p1 = 2 * np.pi * np.arange(n) / n
    P1, P2 = np.meshgrid(p1, p1, indexing="ij")
    dot = a1 * a2 * np.cos(P1 - P2)                           # dl1 . dl2 / (dphi1 dphi2)
    dx = a1 * np.cos(P1) - (shift + a2 * np.cos(P2))
    dy = a1 * np.sin(P1) - a2 * np.sin(P2)
    rho2 = dx * dx + dy * dy
    tot = 0.0
    for z, dz, w in zip(zs, dzs, _GL_W):
        tot += w * dz * float(np.sum(dot / np.sqrt(rho2 + z * z)))
    return N2 * n1 * MU0 / (4 * np.pi) * tot * (2 * np.pi / n) ** 2


def rk4_current(L, R, V0, t_end, nstep):
    f = lambda I: (V0 - R * I) / L  # noqa: E731
    dt = t_end / nstep
    I = 0.0
    ts, Is = [0.0], [0.0]
    for k in range(nstep):
        k1 = f(I)
        k2 = f(I + 0.5 * dt * k1)
        k3 = f(I + 0.5 * dt * k2)
        k4 = f(I + dt * k3)
        I += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        ts.append((k + 1) * dt)
        Is.append(I)
    return np.array(ts), np.array(Is)


def tau_by_stepping(L, R, V0, t_guess):
    ts, Is = rk4_current(L, R, V0, 3 * t_guess, 6000)
    target = (1 - math.exp(-1)) * V0 / R
    k = int(np.argmax(Is >= target))
    return ts[k - 1] + (target - Is[k - 1]) / (Is[k] - Is[k - 1]) * (ts[k] - ts[k - 1])


_SHIFT_CACHE = {}


def phys(d, check):
    p, c = d["params"], d["correct_answers"]
    n1, N2, rate = p["n1"], p["N2"], p["rate"]
    a1, a2, ell = float(p["a1_cm"]) * 1e-2, float(p["a2_cm"]) * 1e-2, p["ell_cm"] * 1e-2
    V0, R = float(p["V0"]), float(p["R"])

    # (a) M by Neumann, emf
    M = m_solenoid_coil(n1, a1, N2, a2)
    check(abs(M * 1e6 - c["M"]) <= 1e-7 * M * 1e6, f"mutual: Neumann M {M * 1e6} vs {c['M']} uH")
    check(abs(M * rate * 1e3 - c["emf"]) <= 1e-7 * M * rate * 1e3, "mutual: |emf| = M |dI/dt|")
    check(ell >= 10 * a1 and a2 < a1, "mutual: geometry not 'long solenoid, coil inside'")

    # (b) reciprocity: the same double sum with the roles of the coils exchanged
    M_swap = n1 * N2 * float(np.sum(_GL_W * m_coax(a2, a1, a1 * np.tan(_GL_X * math.pi / 2)) * a1 /
                                    np.cos(_GL_X * math.pi / 2) ** 2 * (math.pi / 2)))
    check(abs(M_swap - M) <= 1e-9 * M, "mutual: reciprocity failed numerically")
    rc = p["rec_choices"]
    check(sum(x["correct"] == "true" for x in rc) == 1, "mutual MC: exactly one correct")
    for x in rc:
        is_M = x["val"] is not None and abs(x["val"] - M_swap) <= 1e-6 * M_swap
        check(is_M == (x["correct"] == "true"), f"mutual MC: choice {x['key']} marked {x['correct']}")
        if x["val"] is not None and x["correct"] == "false":
            check(abs(x["val"] - M_swap) > 0.04 * M_swap, f"mutual MC: distractor {x['key']} too close to M")

    # L1 printed in the statement, from the field energy
    I_test = 1.0
    W_field = 0.5 * MU0 * (n1 * I_test) ** 2 * math.pi * a1 ** 2 * ell
    L1 = 2 * W_field / I_test ** 2
    check(abs(float(p["L1_mH"]) - L1 * 1e3) <= 6e-4 * L1 * 1e3, f"mutual: printed L1 {p['L1_mH']} vs {L1 * 1e3} mH")

    # (c) RL by time stepping
    t1 = float(p["t1_ms"]) * 1e-3
    ts, Is = rk4_current(L1, R, V0, t1, 4000)
    check(abs(Is[-1] - c["I1"]) <= 1e-8 * max(1e-9, abs(Is[-1])), f"RL: I(t1) {Is[-1]} vs {c['I1']}")
    tau = tau_by_stepping(L1, R, V0, L1 / R)
    check(abs(tau * 1e3 - c["tau"]) <= 1e-5 * tau * 1e3, f"RL: tau by stepping {tau * 1e3} vs {c['tau']} ms")
    ts, Is = rk4_current(L1, R, V0, 40 * tau, 20000)
    dIdt = (V0 - R * Is) / L1
    P_L = L1 * dIdt * Is
    W_int = float(np.sum(0.5 * (P_L[1:] + P_L[:-1]) * np.diff(ts)))
    check(abs(W_int * 1e3 - c["W"]) <= 1e-5 * W_int * 1e3, f"RL: energy integral {W_int * 1e3} vs {c['W']} mJ")
    W_field_inf = 0.5 * MU0 * (n1 * V0 / R) ** 2 * math.pi * a1 ** 2 * ell
    check(abs(W_field_inf * 1e3 - c["W"]) <= 1e-9 * W_field_inf * 1e3, "RL: energy vs 1/2 mu0 H^2 x volume")

    # (d) t = 0+ or long after
    I0p, dI0p = 0.0, (V0 - R * 0.0) / L1
    emf0 = M * dI0p
    late_I = Is[-1]
    sc = p["state_choices"]
    check(sum(x["correct"] == "true" for x in sc) == 1, "state MC: exactly one correct")
    want = "start_state" if p["moment"] == "start" else "late_state"
    corr = [x for x in sc if x["correct"] == "true"][0]
    check(corr["key"] == want, f"state MC: correct is {corr['key']}, want {want}")
    st = {x["key"]: x for x in sc}
    check(abs(st["start_state"]["val"] - emf0) <= 1e-7 * emf0 and I0p == 0.0, "state MC: emf at 0+")
    check(abs(st["late_state"]["val"] - late_I) <= 1e-9 * late_I, "state MC: final current")

    # (e) checkbox truths
    def truth(key):
        if key == "m_N2":
            return abs(m_solenoid_coil(n1, a1, 2 * N2, a2) / M - 2) < 1e-9
        if key == "m_shift":
            if "m_shift" not in _SHIFT_CACHE:
                Ms = m_shifted(n1, a1, N2, a2, 0.5 * (a1 - a2))
                _SHIFT_CACHE["m_shift"] = abs(Ms / M - 1) < 1e-4
            return _SHIFT_CACHE["m_shift"]
        if key == "m_I":     # flux linkage for two different currents, per ampere
            lam = lambda I: m_solenoid_coil(n1, a1, N2, a2) * I  # noqa: E731  (Neumann: linkage = sum ... * I)
            return abs(lam(3.7) / 3.7 - lam(0.2) / 0.2) < 1e-12 * M
        if key == "scale_n1":
            L1b = 2 * (0.5 * MU0 * (2 * n1) ** 2 * math.pi * a1 ** 2 * ell)
            return abs(L1b / L1 - 4) < 1e-9 and abs(m_solenoid_coil(2 * n1, a1, N2, a2) / M - 2) < 1e-9
        if key == "steady_zero":
            return M * 0.0 == 0.0
        if key == "tau63":
            tsx, Isx = rk4_current(L1, R, V0, L1 / R, 4000)
            return abs(Isx[-1] / (V0 / R) - (1 - math.exp(-1))) < 1e-6
        if key == "energy_field":
            return abs(0.5 * L1 * (V0 / R) ** 2 - W_field_inf) < 1e-9 * W_field_inf
        if key == "m_area1":
            return abs(m_solenoid_coil(n1, 1.5 * a1, N2, a2) / M - 2.25) < 1e-3
        if key == "flux_emf":   # flux largest (t >> tau) vs emf largest (t = 0+)
            emf_late = M * (V0 - R * late_I) / L1
            return emf_late >= emf0
        if key == "tau_V0":
            return abs(tau_by_stepping(L1, R, 2 * V0, L1 / R) / tau - 2) < 1e-3
        if key == "more_R_slower":
            return tau_by_stepping(L1, 2 * R, V0, L1 / (2 * R)) > tau
        if key == "energy_half":
            tsx, Isx = rk4_current(L1, R, V0, L1 / R, 4000)
            return 0.5 * L1 * Isx[-1] ** 2 > 0.5 * c["W"] * 1e-3
        if key == "energy_zero":
            return 0.5 * L1 * late_I ** 2 == 0.0
        raise KeyError(key)

    facts = p["facts"]
    nt = sum(f["correct"] == "true" for f in facts)
    check(len(facts) == 5 and 1 <= nt <= 4, f"mutual checkbox: {len(facts)} statements, {nt} true")
    for f in facts:
        t = bool(truth(f["key"]))
        check(t == (f["correct"] == "true"), f"mutual checkbox: {f['key']} marked {f['correct']}, recomputed {t}")
