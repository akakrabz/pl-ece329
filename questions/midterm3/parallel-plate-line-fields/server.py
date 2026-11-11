"""Midterm 3 exam-style: a parallel-plate transmission line, from the fields to V, I and power.

Family of SP19 Exam 3 #2 (re-parameterized). Plates in the planes x = 0 (bottom) and x = d (top), width W along y
(W >> d), dielectric eps_r between them, TEM fields E_x and H_y, travel along +z or -z.
    Z0 = eta d / W,   v = c / sqrt(eps_r),   eta = 120 pi / sqrt(eps_r)
    plate currents from J_s = n x H with n pointing into the field region:
        top plate (n = -x):  I_top(+z) = -H_y W        bottom plate (n = +x):  I_bot(+z) = +H_y W
    travel along s z_hat: E x H along s z  ->  E_x = s eta H_y
    V = V_top - V_bottom = -E_x d,   rho_s(top) = n . D = -eps E_x,   and V / I_top = s Z0
    sinusoidal wave E = x_hat E0 exp(-j beta z):  V0 = E0 d,  <P> = E0^2 W d / (2 eta) = V0^2 / (2 Z0)
"""
import math
import random

C0 = 3e8
ETA0 = 120 * math.pi
EPS0 = 1e-9 / (36 * math.pi)
GEOM = [(50, 10), (50, 5), (20, 4), (20, 2), (10, 2), (10, 1), (5, 1), (4, 1), (25, 5)]     # W in cm, d in mm
DIEL = [(1, "air", 1), (2.25, "polyethylene", 1.5), (4, "a dielectric", 2), (9, "a ceramic", 3), (16, "a ceramic", 4)]


def fnum(x, sig=4):
    if abs(x) < 1e-12:
        return "0"
    s = f"{x:.{sig}g}"
    if "e" in s:
        e = int(s.split("e")[1])
        s = f"{x:.{max(0, sig - 1 - e)}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


SVG = """<svg viewBox="0 0 430 162" width="430" style="max-width:100%" role="img" aria-label="Cross-section of the parallel-plate line: bottom plate at x = 0, top plate at x = d, width W along y, z into the page">
  <g fill="none" stroke="currentColor" stroke-width="1.5">
    <rect x="110" y="38" width="220" height="7" fill="currentColor" fill-opacity="0.35"/>
    <rect x="110" y="98" width="220" height="7" fill="currentColor" fill-opacity="0.35"/>
    <path d="M110 136h220M110 132v8M330 132v8M348 45v53M344 45h8M344 98h8"/>
    <path d="M40 120V70M40 70l-4 8M40 70l4 8M40 120h50M90 120l-8-4M90 120l-8 4"/>
    <circle cx="40" cy="120" r="7"/><path d="M35 115l10 10M45 115l-10 10"/>
  </g>
  <g fill="currentColor" font-size="13">
    <text x="214" y="154">W</text><text x="356" y="76">d</text>
    <text x="36" y="62">x</text><text x="95" y="124">y</text><text x="4" y="146">z (into page)</text>
    <text x="176" y="30">top plate, x = d</text><text x="166" y="121">bottom plate, x = 0</text>
    <text x="220" y="76" text-anchor="middle">%s</text>
  </g>
</svg>"""


def generate(data):
    Wcm, dmm = random.choice(GEOM)
    epsr, fill, n = random.choice(DIEL)
    W, dd = Wcm * 1e-2, dmm * 1e-3
    eta = ETA0 / n
    Z0 = eta * dd / W
    v = C0 / n
    s = random.choice([1, -1])                                   # travel direction along z
    plate = random.choice(["top", "bottom"])
    isg = random.choice([1, -1])                                 # given current flows along isg * z
    I1 = random.choice([0.5, 1, 2, 4, 5])
    Itop = isg * I1 if plate == "top" else -isg * I1             # current along +z in the top plate
    Hy = -Itop / W
    Ex = s * eta * Hy
    V = -Ex * dd
    rho = -epsr * EPS0 * Ex                                      # C/m^2 on the top plate's inner face

    zdir = "+z" if s > 0 else "-z"
    ratio = [
        {"key": 1.0, "text": "$+Z_0$", "feedback": "$V/I = +Z_0$ belongs to a wave travelling toward $+z$."},
        {"key": -1.0, "text": "$-Z_0$", "feedback": "$V/I = -Z_0$ belongs to a wave travelling toward $-z$."},
        {"key": None, "text": "$+\\eta$, the intrinsic impedance of the filling", "feedback": "$\\eta$ is the ratio of the fields, $E/H$. The ratio of $V = E d$ to $I = H W$ is $\\eta\\,d/W = Z_0$."},
        {"key": None, "text": "It depends on which plate's current was specified.", "feedback": "The two plates always carry equal and opposite currents, so $I$ (defined on the top plate) is fixed either way."},
    ]
    for c in ratio:
        c["correct"] = "true" if c["key"] == float(s) else "false"
        if c["correct"] == "true":
            c["feedback"] = "Correct."
    random.shuffle(ratio)

    kE = random.choice([60, 120, 240, 30])
    E0 = kE * math.pi
    V0 = E0 * dd
    P = E0 ** 2 * W * dd / (2 * eta)

    data["params"].update({
        "svg": SVG.replace("%s", f"&#949;r = {fnum(epsr)}"), "W_s": fnum(Wcm), "d_s": fnum(dmm), "epsr_s": fnum(epsr), "fill": fill,
        "zdir": zdir, "plate": plate, "I1_s": fnum(I1), "idir": "+z" if isg > 0 else "-z", "ratio": ratio,
        "E0_tex": f"{kE}\\pi",
        # solution strings
        "s_eta": fnum(eta), "s_eta_tex": "120\\pi" if n == 1 else f"120\\pi/{fnum(n)}", "s_Z0": fnum(Z0), "s_v": fnum(v / 1e8),
        "s_dW": f"{fnum(dmm)}\\ \\mathrm{{mm}}/{fnum(Wcm * 10)}\\ \\mathrm{{mm}}",
        "s_Itop": fnum(Itop), "s_Hy": fnum(Hy), "s_Ex": fnum(Ex), "s_V": fnum(V), "s_rho": fnum(rho * 1e9),
        "s_s": "+" if s > 0 else "-", "s_sZ0": ("" if s > 0 else "-") + fnum(Z0),
        "s_plate_rule": ("top plate: $\\hat n = -\\hat x$, so $J_{sz} = -H_y$ and $I = -H_yW$" if plate == "top"
                         else "bottom plate: $\\hat n = +\\hat x$, so $J_{sz} = +H_y$ and $I = +H_yW$"),
        "s_Igiven": fnum(isg * I1), "s_V0": fnum(V0), "s_P": fnum(P),
        # raw data for the independent check
        "W": W, "d": dd, "epsr": epsr, "s": s, "isg": isg, "I1": I1, "E0": E0,
    })
    ca = data["correct_answers"]
    ca["Z0"], ca["v"] = Z0, v / 1e8
    ca["Hy"], ca["Ex"], ca["V"], ca["rho"] = Hy, Ex, V, rho * 1e9
    ca["V0"], ca["P"] = V0, P
