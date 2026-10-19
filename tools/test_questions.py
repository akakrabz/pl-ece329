#!/usr/bin/env python3
"""Offline checks for the ECE 329 PrairieLearn questions (no PrairieLearn needed).

For every question directory:
  * import server.py and run generate() on N seeds (random seeded like PL does);
  * data must be JSON-serializable; every {{params.x}} in question.html must exist;
    every answers-name must have a correct_answers entry (or be a choice element);
  * multiple-choice: exactly one correct choice, all choice texts distinct;
  * physics: closed-form answers re-checked numerically (finite differences / quadrature).
Run:  python3 tools/test_questions.py [N]
"""
import importlib.util, json, math, os, random, re, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QDIR = os.path.join(ROOT, "questions")
import argparse
_ap = argparse.ArgumentParser(description="Offline checks for the ECE 329 PrairieLearn questions")
_ap.add_argument("n", nargs="?", type=int, default=300, help="seeds per question (default 300)")
_ap.add_argument("--only", default="", help="comma-separated question names (directory names) to test")
_ap.add_argument("--katex", default="", help="path to katex.min.js: also compile every $...$ with KaTeX (needs node)")
_ap.add_argument("--katex-seeds", type=int, default=3, help="seeds per question for the KaTeX pass (default 3)")
ARGS = _ap.parse_args()
N = ARGS.n
EPS0 = 8.8541878128e-12


def load(qpath):
    spec = importlib.util.spec_from_file_location("server", os.path.join(qpath, "server.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def run(mod, seed):
    random.seed(seed); np.random.seed(seed % (2**32 - 1))
    data = {"params": {}, "correct_answers": {}, "variant_seed": seed, "options": {}}
    mod.generate(data)
    json.dumps(data)   # must be serializable
    return data


def mustache_refs(html):
    refs = set(re.findall(r"\{\{\{?\s*params\.([A-Za-z0-9_]+)\s*\}?\}\}", html))
    loops = set(re.findall(r"\{\{#params\.([A-Za-z0-9_]+)\}\}", html))
    names = re.findall(r'answers-name="([^"]+)"', html)
    choice_elems = re.findall(r"<pl-(multiple-choice|checkbox)[^>]*answers-name=\"([^\"]+)\"", html)
    return refs, loops, names, {n for _, n in choice_elems}


def sym_eval(expr, **vals):
    """Evaluate a sympy-style string with numpy (only +,-,*,/,** and names)."""
    if not re.fullmatch(r"[0-9A-Za-z_+\-*/().\s]+", expr):
        raise ValueError("unexpected characters in expression: " + expr)
    return eval(expr, {"__builtins__": {}}, vals)


errors = []
def check(cond, msg):
    if not cond:
        errors.append(msg)


# ------------------------------------------------------------------ per-question physics checks
def phys_conservative(d):
    p = d["params"]; alpha, beta, m, n, pp = p["alpha"], p["beta"], p["m"], p["n"], p["p"]
    E0, eps = 1.7, 2.3
    def E(x, y, z):
        return np.array([alpha*m*x**(m-1)*y**n, alpha*n*x**m*y**(n-1), beta*pp*z**(pp-1)]) * E0
    Vs, rhos = d["correct_answers"]["V"], d["correct_answers"]["rho"]
    h = 1e-5
    for _ in range(5):
        x, y, z = np.random.uniform(0.5, 2, 3)
        V = lambda x, y, z: sym_eval(Vs, x=x, y=y, z=z, E0=E0)
        grad = np.array([(V(x+h,y,z)-V(x-h,y,z)), (V(x,y+h,z)-V(x,y-h,z)), (V(x,y,z+h)-V(x,y,z-h))])/(2*h)
        check(np.allclose(-grad, E(x,y,z), rtol=1e-4, atol=1e-6), f"conservative: -grad V != E at {(x,y,z)}")
        divE = ((E(x+h,y,z)-E(x-h,y,z))[0] + (E(x,y+h,z)-E(x,y-h,z))[1] + (E(x,y,z+h)-E(x,y,z-h))[2])/(2*h)
        rho = sym_eval(rhos, x=x, y=y, z=z, E0=E0, epsilon=eps)
        check(abs(rho - eps*divE) < 1e-4*max(1, abs(rho)), f"conservative: rho != eps*div E ({rho} vs {eps*divE})")
        curlz = ((E(x+h,y,z)-E(x-h,y,z))[1] - (E(x,y+h,z)-E(x,y-h,z))[0])/(2*h)
        check(abs(curlz) < 1e-6, "conservative: curl_z != 0")
    # line integral: V(0)-V(B) in units of E0
    bx, by, bz = [int(v) for v in re.findall(r"-?\d+", p["B_tex"])]
    val = (sym_eval(Vs, x=0, y=0, z=0, E0=1.0) - sym_eval(Vs, x=bx, y=by, z=bz, E0=1.0))
    check(abs(val - d["correct_answers"]["lineint"]) < 1e-9, "conservative: line integral mismatch")
    check(sum(c["correct"] == "true" for c in p["curl_choices"]) == 1, "conservative: MC needs exactly one correct")
    check(len({c["text"] for c in p["curl_choices"]}) == 4, f"conservative: duplicate MC choices {[c['text'] for c in p['curl_choices']]}")


def phys_slab(d):
    p = d["params"]; k = p["k"]; rho0 = p["rho0_nC"]*1e-9; a = p["a_mm"]*1e-3; b = p["b_over_a"]*a
    xin = a*p["xin_num"]/p["xin_den"]
    xs = np.linspace(-a, a, 200001); rho = rho0*np.abs(xs/a)**k
    sigma = np.trapezoid(rho, xs)
    check(abs(sigma*1e9 - d["correct_answers"]["sigma"]) < 1e-6*sigma*1e9, "slab: sigma")
    def Ex(x):   # Gauss with symmetric pillbox: 2 eps0 E = int_{-x}^{x} rho
        xx = np.linspace(-x, x, 20001); return np.trapezoid(rho0*np.abs(xx/a)**k*(np.abs(xx) <= a), xx)/(2*EPS0)
    check(abs(Ex(xin) - d["correct_answers"]["E_in"]) < 1e-5*abs(Ex(xin)), "slab: E_in")
    E_out = sigma/(2*EPS0)   # pillbox enclosing the whole slab
    check(abs(E_out - d["correct_answers"]["E_out"]) < 1e-5*abs(E_out), "slab: E_out")
    xs2 = np.linspace(0, b, 4001); Vb = -np.trapezoid([Ex(x) for x in xs2], xs2)
    check(abs(Vb - d["correct_answers"]["V_b"]) < 2e-4*abs(Vb), f"slab: V_b {Vb} vs {d['correct_answers']['V_b']}")


def phys_interface(d):
    p = d["params"]; er1, er2 = p["er1"], p["er2"]; c = d["correct_answers"]
    check(c["E2x"] == p["E1x"], "interface: E2x")
    check(abs(er2*c["E2z"] - er1*p["E1z"]) < 1e-9, "interface: normal D continuity")
    check(abs(c["D1z"] - er1*p["E1z"]) < 1e-9 and abs(c["D2x"] - er2*p["E1x"]) < 1e-9, "interface: D")
    check(abs(c["P1x"] - (er1-1)*p["E1x"]) < 1e-9 and abs(c["P2z"] - (er2-1)*c["E2z"]) < 1e-9, "interface: P")
    check(abs(c["rho_sb"] - (p["E1z"] - c["E2z"])) < 1e-9, "interface: bound charge = E1z - E2z")
    check(float(c["E2z"]).is_integer(), "interface: E2z not integer")


def phys_coax(d):
    p = d["params"]; a, c, b = p["a_mm"]*1e-3, p["c_mm"]*1e-3, p["b_mm"]*1e-3; er1, er2 = p["er1"], p["er2"]; lam = p["lam_nC"]*1e-9
    r1, r2 = 0.5*(a+c), 0.5*(c+b)
    check(abs(d["correct_answers"]["D_r1"] - lam/(2*math.pi*r1)*1e9) < 1e-9, "coax: D")
    check(abs(d["correct_answers"]["E_r2"] - lam/(2*math.pi*EPS0*er2*r2)) < 1e-6, "coax: E")
    rs = np.linspace(a, b, 200001); E = np.where(rs < c, lam/(2*math.pi*EPS0*er1*rs), lam/(2*math.pi*EPS0*er2*rs))
    Vab = np.trapezoid(E, rs)
    check(abs(Vab - d["correct_answers"]["Vab"]) < 1e-4*Vab, f"coax: Vab {Vab} vs {d['correct_answers']['Vab']}")
    Cseries = 1/(math.log(c/a)/(2*math.pi*EPS0*er1) + math.log(b/c)/(2*math.pi*EPS0*er2))
    check(abs(Cseries*1e12 - d["correct_answers"]["Cp"]) < 1e-6*Cseries*1e12, "coax: C' vs series formula")
    check(d["correct_answers"]["q_outer"] == -p["lam_nC"], "coax: outer charge")


def phys_flux(d):
    p = d["params"]; ch = p["choices"]
    check(sum(c["correct"] == "true" for c in ch) == 1, "flux: exactly one correct")
    check(len({c["text"] for c in ch}) == len(ch) and len(ch) >= 4, f"flux: choices not distinct or fewer than 4: {[c['text'] for c in ch]}")
    up = p["normal_tex"].startswith("+")
    psi = (p["q_below"] - p["q_above"])/2 * (1 if up else -1)
    correct = [c["text"] for c in ch if c["correct"] == "true"][0]
    # parse the correct text back to a number
    m = re.search(r"\$(-?)(?:\\dfrac\{(\d*)Q\}\{(\d+)\}|(\d*)Q|0)\$", correct)
    val = 0.0
    if m and m.group(3): val = int(m.group(2) or 1)/int(m.group(3))
    elif m and m.group(4) is not None: val = float(m.group(4) or 1)
    if m and m.group(1) == "-": val = -val
    check(abs(val - psi) < 1e-9, f"flux: correct choice {correct} != {psi}")


def _one_correct(ch, name):
    check(sum(c["correct"] == "true" for c in ch) == 1, f"{name}: MC needs exactly one correct")
    check(len({c["text"] for c in ch}) == len(ch), f"{name}: duplicate MC choices")


def phys_two_layer(d):
    p = d["params"]; c = d["correct_answers"]; er1, er2, dd, t2 = p["er1"], p["er2"], p["d"], p["t2"]
    check(abs(er1*p["E1z"] - er2*c["E2z"]) < 1e-9, "two-layer: D_z continuity")
    check(float(c["E2z"]).is_integer() and p["E1z"] < 0, "two-layer: E2z integer / E1z sign")
    n = (dd + t2)*10000; h = (dd + t2)/n; zm = (np.arange(n) + 0.5)*h; Ez = np.where(zm < dd, p["E1z"], c["E2z"])   # midpoint rule, cells aligned to z = d
    check(abs(-np.sum(Ez)*h - c["Vp"]) < 1e-9*abs(c["Vp"]), "two-layer: Vp = -int E dz")
    check(abs(c["Vd"] + p["E1z"]*dd) < 1e-9 and c["Vp"] > c["Vd"] > 0, "two-layer: V(d), monotone potential")
    check(abs(c["rho_top"] + er2*c["E2z"]) < 1e-9 and c["rho_top"] > 0, "two-layer: top plate charge")
    C_series = 1/(dd/(er1*EPS0) + t2/(er2*EPS0))
    check(abs(C_series*1e12 - c["C_per_A"]) < 1e-6*C_series*1e12, "two-layer: C/A series")
    check(abs(c["rho_top"]*EPS0/c["Vp"] - C_series) < 1e-9*C_series, "two-layer: C/A = rho_s/Vp")


def phys_sheet(d):
    p = d["params"]; c = d["correct_answers"]; er1, er2, dd, t2, s = p["er1"], p["er2"], p["d"], p["t2"], p["s"]
    check(abs(er2*c["E2z"] - er1*c["E1z"] - s) < 1e-9, "sheet: D jump = rho_s")
    check(abs(c["E1z"]*dd + c["E2z"]*t2) < 1e-9, "sheet: V(z0) = 0 (int E over both regions)")
    check(abs(-c["E1z"]*dd - c["V0"]) < 1e-9, "sheet: V0 = -E1z d")
    check(abs(c["rho_0"] + c["rho_z0"] + s) < 1e-9, "sheet: plate charges sum to -rho_s")
    check(abs(c["rho_0"] - er1*c["E1z"]) < 1e-9 and abs(c["rho_z0"] + er2*c["E2z"]) < 1e-9, "sheet: plate BCs")
    check(np.sign(c["V0"]) == np.sign(s), "sheet: sign of V0")


def phys_insertion(d):
    p = d["params"]; c = d["correct_answers"]; er, E0 = p["er"], p["E0"]
    if p["mode"] == "Q":
        check(c["D"] == E0 and abs(c["E"] - E0/er) < 1e-12 and abs(c["Wratio"] - 1/er) < 1e-12, "insertion Q: D fixed")
    else:
        check(c["E"] == E0 and c["D"] == er*E0 and c["Wratio"] == er, "insertion V: E fixed")
    check(abs(c["P"] - (c["D"] - c["E"])) < 1e-9, "insertion: P = D - eps0 E")
    check(abs(c["D"] - er*c["E"]) < 1e-9, "insertion: D = eps E")
    check(abs(c["Wratio"] - (c["D"]*c["E"])/(E0*E0)) < 1e-9, "insertion: energy ratio = DE/D0E0")
    _one_correct(p["choices"], "insertion")
    correct = [x["text"] for x in p["choices"] if x["correct"] == "true"][0]
    check(("= 0$" in correct) == (p["mode"] == "Q"), "insertion: MC correct choice does not match the mode")


def phys_lossy(d):
    p = d["params"]; c = d["correct_answers"]; eps = p["er"]*EPS0; sig = p["sigma"]
    g = p["geometry"]
    if g == "plates":
        area = lambda x: p["A_cm2"]*1e-4 + 0*x; lo, hi = 0.0, p["d_mm"]*1e-3
    elif g == "coax":
        area = lambda r: 2*math.pi*r*p["L_m"]; lo, hi = p["a_mm"]*1e-3, p["b_mm"]*1e-3
    else:
        area = lambda r: 4*math.pi*r*r; lo, hi = p["a_cm"]*1e-2, p["b_cm"]*1e-2
    xs = np.linspace(lo, hi, 400001)
    inv_area = 1/area(xs)
    R = np.trapezoid(inv_area/sig, xs)          # sum of shell resistances dR = dr/(sigma area)
    Vunit = np.trapezoid(inv_area/eps, xs)      # V for unit charge, from D = Q/area
    check(abs(1/Vunit*1e12 - c["C"]) < 1e-4*c["C"], f"lossy {g}: C by integration {1/Vunit*1e12} vs {c['C']}")
    check(abs(1/R*1e9 - c["G"]) < 1e-4*c["G"], f"lossy {g}: G by shell resistances")
    check(abs(c["tau"] - eps/sig*1e6) < 1e-9*c["tau"] and abs(c["C"]*1e-12/(c["G"]*1e-9) - eps/sig) < 1e-9*eps/sig, "lossy: tau = C/G = eps/sigma")
    check(abs(c["Qt"] - p["Q0"]*math.exp(-p["t1"]/c["tau"])) < 1e-9, "lossy: Q(t1)")
    check(abs(c["I0"] - p["Q0"]*1e-9/(c["tau"]*1e-6)*1e6) < 1e-9*c["I0"], "lossy: I0 = Q0/tau")


def phys_ampere_coax(d):
    p = d["params"]; c = d["correct_answers"]; a, b, cc = p["a_mm"]*1e-3, p["b_mm"]*1e-3, p["c_mm"]*1e-3; I = p["I"]
    Ja, Jb = I/(math.pi*a*a), -I/(math.pi*(cc*cc - b*b))
    def Ienc(r):        # integrate J(r') 2 pi r' dr' by the midpoint rule; cells (1 um) align with a, b, c (integer mm)
        n = int(round(r/1e-6)); h = r/n; rm = (np.arange(n) + 0.5)*h; J = np.where(rm < a, Ja, np.where((rm > b) & (rm < cc), Jb, 0.0))
        return np.sum(J*2*math.pi*rm)*h
    for key, r in (("H1", a/2), ("H2", (a+b)/2), ("H3", (b+cc)/2), ("H4", 2*cc)):
        H = Ienc(r)/(2*math.pi*r)
        check(abs(H - c[key]) < 1e-6*I/(2*math.pi*r), f"ampere-coax: {key} {H} vs {c[key]}")
    check(abs(c["H4same"] - 2*I/(2*math.pi*2*cc)) < 1e-9, "ampere-coax: H4same")
    check(c["H1"] > 0 and c["H2"] > c["H3"] > 0 and c["H4"] == 0, "ampere-coax: ordering")


def phys_sheets(d):
    p = d["params"]; c = d["correct_answers"]; J1, J2 = p["Js1"], p["Js2"]; MU0 = 4e-7*math.pi
    for key, x in (("Hleft", -1.0), ("Hmid", p["d"]/2), ("Hright", p["d"] + 1.0)):
        H = 0.5*J1*np.sign(x) + 0.5*J2*np.sign(x - p["d"])      # H = 1/2 Js x n per sheet
        check(abs(H - c[key]) < 1e-9, f"sheets: {key}")
    f = np.cross(np.array([0, 0, J2]), np.array([0, MU0*J1/2, 0]))   # Js2 x B1(x=d)
    check(abs(f[0]*1e6 - c["fx"]) < 1e-9 and abs(f[1]) + abs(f[2]) < 1e-15, "sheets: force")
    check((c["fx"] < 0) == (J1*J2 > 0), "sheets: parallel currents attract")
    check(("+" in p["dir_correct"]) == (J1 > 0), "sheets: direction MC")


def phys_solenoid(d):
    p = d["params"]; c = d["correct_answers"]; MU0 = 4e-7*math.pi
    n = p["N"]/(p["L_cm"]*1e-2); A = math.pi*(p["a_mm"]*1e-3)**2
    check(abs(c["H"] - n*p["I"]) < 1e-9, "solenoid: H")
    check(abs(c["Psi"] - MU0*n*p["I"]*A*1e6) < 1e-9, "solenoid: Psi")
    check(abs(c["Lind"] - p["N"]*c["Psi"]*1e-6/p["I"]*1e6) < 1e-9, "solenoid: L = N Psi / I")
    check(abs(c["W"] - 0.5*MU0*c["H"]**2*A*p["L_cm"]*1e-2*1e6) < 1e-6*c["W"], "solenoid: W = energy density x volume")
    check(abs(c["tau"] - c["Lind"]/p["R"]) < 1e-9, "solenoid: tau = L/R")
    check(p["a_mm"]*1e-3 <= p["L_cm"]*1e-2/5, "solenoid: not long")


def phys_faraday(d):
    p = d["params"]; c = d["correct_answers"]; A = p["a_cm"]*p["b_cm"]*1e-4
    Psi = lambda t: (p["B0"] + p["k"]*t)*A
    h = 1e-4
    check(abs(c["Psi"] - Psi(p["t1"])*1e3) < 1e-9, "faraday: Psi(t1)")
    check(abs(c["emf"] + p["N"]*(Psi(0.5+h) - Psi(0.5-h))/(2*h)) < 1e-6, "faraday: emf = -N dPsi/dt")
    check(abs(c["I"] - c["emf"]/p["R"]*1e3) < 1e-9, "faraday: I = emf/R")
    _one_correct(p["sense_choices"], "faraday sense"); _one_correct(p["lenz_choices"], "faraday lenz")
    sense = [x["text"] for x in p["sense_choices"] if x["correct"] == "true"][0]
    lenz = [x["text"] for x in p["lenz_choices"] if x["correct"] == "true"][0]
    check(sense.startswith("counterclockwise") == (c["emf"] > 0), "faraday: sense MC vs sign of emf")
    check(("+" in lenz) == (p["k"] < 0), "faraday: Lenz — induced field +z iff flux decreasing")


PHYS = {"conservative-field-analysis": phys_conservative, "nonuniform-slab": phys_slab,
        "dielectric-interface": phys_interface, "coax-two-layer": phys_coax, "flux-through-plane": phys_flux,
        "two-layer-parallel-plates": phys_two_layer, "sheet-between-grounded-plates": phys_sheet,
        "dielectric-insertion": phys_insertion, "lossy-capacitor-relaxation": phys_lossy,
        "ampere-coax-fields": phys_ampere_coax, "current-sheets-superposition": phys_sheets,
        "solenoid-inductance-energy": phys_solenoid, "faraday-loop-emf": phys_faraday}


# ====================================================================== generic checks (all questions)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mustache_lite import render as mustache_render  # noqa: E402
import html as _html  # noqa: E402
import subprocess  # noqa: E402

TOOLS = os.path.dirname(os.path.abspath(__file__))
CHECKS_DIR = os.path.join(TOOLS, "checks")
STANDARD_TAGS = {"numeric", "symbolic", "drawing", "MC", "code", "multianswer", "graph", "concept", "calculate",
                 "compute", "software", "estimation", "secret", "nontest"}
STANDARD_SETS = {"Homework", "Machine Problem", "Quiz", "Practice Quiz", "Exam", "Practice Exam", "Prep", "Worksheet"}
ALLOWED_IMPORTS = {"random", "math", "cmath", "fractions", "itertools", "json", "numpy", "io", "matplotlib",
                   "matplotlib.pyplot", "collections", "functools"}

# Attribute whitelists, from docs.prairielearn.com element pages (checked October 2026).
_MC = {"all-of-the-above", "all-of-the-above-feedback", "allow-blank", "answers-name", "aria-label", "builtin-grading",
       "display", "hide-letter-keys", "hide-score-badge", "none-of-the-above", "none-of-the-above-feedback",
       "number-answers", "order", "placeholder", "size", "weight"}
_CB = {"allow-blank", "answers-name", "detailed-help-text", "display", "hide-help-text", "hide-letter-keys",
       "hide-score-badge", "max-correct", "max-select", "min-correct", "min-select", "number-answers", "order",
       "partial-credit", "show-number-correct", "weight"}
ATTRS = {
    "pl-question-panel": set(), "pl-answer-panel": set(), "pl-submission-panel": set(), "pl-hidden-hints": set(),
    "pl-hint": {"hint-name", "show-after-submission"},
    "pl-multiple-choice": _MC, "pl-checkbox": _CB, "pl-answer": {"correct", "feedback", "score"},
    "pl-matching": {"allow-blank", "answers-name", "blank", "counter-type", "fixed-options-order", "fixed-order",
                    "hide-score-badge", "none-of-the-above", "number-options", "number-statements",
                    "options-placement", "partial-credit", "weight"},
    "pl-statement": {"match"}, "pl-option": {"name"},
    "pl-number-input": {"answers-name", "weight", "correct-answer", "label", "aria-label", "suffix", "display",
                        "comparison", "rtol", "atol", "digits", "allow-complex", "allow-blank", "blank-value",
                        "allow-fractions", "size", "show-help-text", "placeholder", "custom-format", "show-score"},
    "pl-integer-input": {"allow-blank", "answers-name", "aria-label", "base", "blank-value", "correct-answer",
                         "display", "initial-value", "label", "placeholder", "show-help-text", "show-score", "size",
                         "suffix", "weight"},
    "pl-symbolic-input": {"additional-simplifications", "allow-blank", "allow-complex", "allow-sets",
                          "allow-trig-functions", "answers-name", "aria-label", "blank-value", "correct-answer",
                          "custom-functions", "display", "display-log-as-ln", "display-simplified-expression",
                          "formula-editor", "imaginary-unit-for-display", "initial-value", "label", "placeholder",
                          "show-help-text", "show-score", "size", "suffix", "variables", "weight"},
    "pl-figure": {"alt", "directory", "display", "file-name", "type", "width"},
}
_ATTR_RE = r"(?:\s+[\w:-]+(?:\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s\"'>]+))?)*"
_TAG_RE = re.compile(r"<(pl-[a-z-]+)(" + _ATTR_RE + r")\s*/?>")
_ONE_ATTR = re.compile(r"([\w:-]+)(?:\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s\"'>]+)))?")


def attrs_of(s):
    out = {}
    for m in _ONE_ATTR.finditer(s):
        out[m.group(1)] = next((g for g in m.group(2, 3, 4) if g is not None), "")
    return out


def blocks(h, tag):
    rx = re.compile(r"<" + tag + r"(" + _ATTR_RE + r")\s*>(.*?)</" + tag + r">", re.S)
    return [(attrs_of(m.group(1)), m.group(2)) for m in rx.finditer(h)]


def _norm(t):
    return re.sub(r"\s+", " ", t).strip()


def _true(v):
    return str(v).strip().lower() == "true"


def lint_raw_html(q, raw):
    for m in _TAG_RE.finditer(raw):
        tag, a = m.group(1), attrs_of(m.group(2))
        if tag not in ATTRS:
            check(False, f"{q}: element <{tag}> is not on the checked list (typo, or add it to ATTRS after reading its docs)")
            continue
        bad = set(a) - ATTRS[tag]
        check(not bad, f"{q}: <{tag}> has unknown/deprecated attributes {sorted(bad)}")
    check("<pl-answer-panel>" in raw, f"{q}: no <pl-answer-panel> (every question explains its solution)")
    names = re.findall(r'answers-name="([^"]+)"', raw)
    check(len(names) == len(set(names)), f"{q}: duplicate answers-name {sorted(n for n in set(names) if names.count(n) > 1)}")


def lint_rendered(q, h, d, seed):
    """Checks on one rendered variant: choice blocks, matching, numeric answers."""
    ca = d["correct_answers"]
    for a, inner in blocks(h, "pl-multiple-choice"):
        ans = blocks(inner, "pl-answer")
        texts = [_norm(t) for _, t in ans]
        ncor = sum(_true(x.get("correct", "false")) for x, _ in ans)
        special = any(a.get(k, "false") not in ("false", "") for k in ("all-of-the-above", "none-of-the-above"))
        n_ans = int(a["number-answers"]) if a.get("number-answers") else None
        if not special:
            if n_ans is None:
                check(ncor == 1, f"{q} seed {seed}: multiple-choice {a.get('answers-name')} has {ncor} correct answers (need exactly 1)")
            else:
                check(ncor >= 1 and len(ans) - ncor >= n_ans - 1,
                      f"{q} seed {seed}: multiple-choice {a.get('answers-name')} cannot fill number-answers={n_ans}")
        check(len(set(texts)) == len(texts), f"{q} seed {seed}: multiple-choice {a.get('answers-name')} has duplicate choices")
        check(all(t for t in texts), f"{q} seed {seed}: multiple-choice {a.get('answers-name')} has an empty choice")
        for x, _ in ans:
            if "score" in x:
                check(0.0 <= float(x["score"]) <= 1.0, f"{q}: pl-answer score outside [0, 1]")
    for a, inner in blocks(h, "pl-checkbox"):
        ans = blocks(inner, "pl-answer")
        check(not any("score" in x for x, _ in ans), f"{q}: pl-answer inside pl-checkbox cannot have score=")
        texts = [_norm(t) for _, t in ans]
        ncor = sum(_true(x.get("correct", "false")) for x, _ in ans)
        ninc = len(ans) - ncor
        n_ans = int(a.get("number-answers") or len(ans))
        min_c = int(a.get("min-correct") or (ncor if "number-answers" not in a else 1))
        max_c = int(a.get("max-correct") or ncor)
        check(ncor >= max(1, min_c), f"{q} seed {seed}: checkbox {a.get('answers-name')} has {ncor} correct, needs >= {max(1, min_c)}")
        check(n_ans <= len(ans) and ninc >= n_ans - min(max_c, ncor),
              f"{q} seed {seed}: checkbox {a.get('answers-name')} cannot fill number-answers={n_ans} (correct {ncor}, incorrect {ninc})")
        check(len(set(texts)) == len(texts), f"{q} seed {seed}: checkbox {a.get('answers-name')} has duplicate choices")
    for a, inner in blocks(h, "pl-matching"):
        st = blocks(inner, "pl-statement")
        op = blocks(inner, "pl-option")
        check(len(st) >= 2, f"{q}: matching {a.get('answers-name')} has fewer than 2 statements")
        check(all(x.get("match") for x, _ in st), f"{q}: matching {a.get('answers-name')}: every pl-statement needs match=")
        stexts = [_norm(t) for _, t in st]
        check(len(set(stexts)) == len(stexts), f"{q} seed {seed}: matching {a.get('answers-name')} has duplicate statements")
        keys = [x.get("name") or _norm(t) for x, t in op]
        check(len(set(keys)) == len(keys), f"{q} seed {seed}: matching {a.get('answers-name')} has duplicate option names")
        if a.get("number-statements"):
            check(int(a["number-statements"]) <= len(st), f"{q}: matching number-statements > statements")
    for a in [attrs_of(m.group(1)) for m in re.finditer(r"<pl-number-input(" + _ATTR_RE + r")\s*/?>", h)]:
        nm = a.get("answers-name")
        if "correct-answer" in a or _true(a.get("allow-complex", "false")):
            continue
        v = ca.get(nm)
        check(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v),
              f"{q} seed {seed}: number-input {nm} correct answer {v!r} is not a finite real number")
    for a in [attrs_of(m.group(1)) for m in re.finditer(r"<pl-integer-input(" + _ATTR_RE + r")\s*/?>", h)]:
        v = ca.get(a.get("answers-name"))
        check(isinstance(v, int) and not isinstance(v, bool), f"{q} seed {seed}: integer-input {a.get('answers-name')} correct answer {v!r} is not an int")


def math_segments(h):
    segs = []
    h = re.sub(r"<script.*?</script>", " ", h, flags=re.S)
    for m in re.finditer(r"\$\$(.+?)\$\$", h, re.S):
        segs.append((m.group(1), True))
    h2 = re.sub(r"\$\$(.+?)\$\$", " ", h, flags=re.S).replace("\\$", " ")
    for m in re.finditer(r"\$(.+?)\$", h2, re.S):
        segs.append((m.group(1), False))
    for m in re.finditer(r"\\\((.+?)\\\)", h2, re.S):
        segs.append((m.group(1), False))
    for m in re.finditer(r"\\\[(.+?)\\\]", h2, re.S):
        segs.append((m.group(1), True))
    return [(_html.unescape(x), disp) for x, disp in segs]


def katex_check(items):
    """items: list of (question, expr, display). Returns list of (question, expr, error)."""
    js = os.path.join(TOOLS, "katex_check.js")
    payload = json.dumps([{"q": q, "e": e, "d": d} for q, e, d in items])
    r = subprocess.run(["node", js, ARGS.katex], input=payload, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        return [("(katex)", "", r.stderr.strip()[:500])]
    return [(x["q"], x["e"], x["err"]) for x in json.loads(r.stdout)]


def check_server_imports(q, src):
    mods = []
    for m in re.finditer(r"^\s*import\s+([\w., ]+)", src, re.M):
        mods += [x.strip().split(" as ")[0].strip() for x in m.group(1).split(",")]
    mods += re.findall(r"^\s*from\s+([\w.]+)\s+import", src, re.M)
    for mod in mods:
        check(mod in ALLOWED_IMPORTS or mod.split(".")[0] in {"numpy", "matplotlib"},
              f"{q}: server.py imports {mod} (keep questions dependency-free: random, math, fractions, numpy, matplotlib)")


def load_check_module(q):
    path = os.path.join(CHECKS_DIR, q + ".py")
    if not os.path.exists(path):
        return None
    spec = importlib.util.spec_from_file_location("check_" + q.replace("-", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.phys


# ---------------------------------------------------------------- course-level checks
course = json.load(open(os.path.join(ROOT, "infoCourse.json")))
TOPICS = {t["name"] for t in course.get("topics", [])}
TAGS = {t["name"] for t in course.get("tags", [])} | STANDARD_TAGS
SETS = {s["name"] for s in course.get("assessmentSets", [])} | STANDARD_SETS
check(len({t["name"] for t in course.get("tags", [])}) == len(course.get("tags", [])), "infoCourse: duplicate tag names")
uuids = {}

ONLY = {x.strip() for x in ARGS.only.split(",") if x.strip()}
qids = []
katex_items = []
for topic in sorted(os.listdir(QDIR)):
    for q in sorted(os.listdir(os.path.join(QDIR, topic))):
        qpath = os.path.join(QDIR, topic, q)
        if not os.path.isdir(qpath):
            continue
        qids.append(f"{topic}/{q}")
        if ONLY and q not in ONLY:          # other questions (maybe half-written): only remember their uuid
            try:
                uuids.setdefault(json.load(open(os.path.join(qpath, "info.json"))).get("uuid", "").lower(), q)
            except Exception:  # noqa: BLE001
                pass
            continue
        try:
            info = json.load(open(os.path.join(qpath, "info.json")))
        except Exception as e:  # noqa: BLE001
            check(False, f"{q}: info.json unreadable: {e}")
            continue
        check(info.get("type") == "v3" and "uuid" in info and info.get("title"), f"{q}: info.json needs type v3, uuid, title")
        u = info.get("uuid", "").lower()
        check(u not in uuids, f"{q}: uuid duplicates {uuids.get(u)}")
        uuids[u] = q
        check(info.get("topic") in TOPICS, f"{q}: topic {info.get('topic')!r} not declared in infoCourse.json")
        for t in info.get("tags", []):
            check(t in TAGS, f"{q}: tag {t!r} not declared in infoCourse.json")
        raw = open(os.path.join(qpath, "question.html")).read()
        lint_raw_html(q, raw)
        stripped = re.sub(r"\{\{\{\s*[A-Za-z0-9_.]+\s*\}\}\}|\{\{[#/^]?\s*[A-Za-z0-9_.]+\s*\}\}", "", raw)
        check("{{" not in stripped, f"{q}: stray Mustache braces after removing valid tags (LaTeX brace collision?)")
        for block in re.findall(r"<pl-multiple-choice.*?</pl-multiple-choice>", raw, re.S):
            if "{{" not in block and "number-answers" not in block:      # fully static choices: exactly one correct
                check(block.count('correct="true"') == 1, f"{q}: static multiple-choice block needs exactly one correct answer")
        src = open(os.path.join(qpath, "server.py")).read()
        check_server_imports(q, src)
        refs, loops, names, choice_names = mustache_refs(raw)
        try:
            mod = load(qpath)
        except Exception as e:   # noqa: BLE001 - report and move on
            check(False, f"{q}: server.py failed to import: {e!r}")
            print(f"{q:36s} IMPORT ERROR")
            continue
        physfn = PHYS.get(q) or load_check_module(q)
        check(physfn is not None, f"{q}: no physics check (add tools/checks/{q}.py with phys(d, check))")
        dyn_figs = [attrs_of(m.group(1)) for m in re.finditer(r"<pl-figure(" + _ATTR_RE + r")\s*/?>", raw)]
        dyn_figs = [a for a in dyn_figs if a.get("type") == "dynamic"]
        variants = set()
        nerr0 = len(errors)
        for seed in range(1, N + 1):
            d = run(mod, seed)
            for r in refs | loops:
                check(r in d["params"], f"{q}: question.html references params.{r} which generate() did not set (seed {seed})")
            for nme in names:
                # pl-multiple-choice / pl-checkbox / pl-matching write data["params"][answers-name] themselves and
                # raise "duplicate params variable name" if generate() already set that key.
                if nme in choice_names or re.search(r'<pl-matching[^>]*answers-name="' + re.escape(nme) + '"', raw):
                    check(nme not in d["params"], f"{q}: params['{nme}'] collides with a choice element's answers-name (PrairieLearn raises 'duplicate params variable name')")
                if nme not in choice_names and not re.search(r'<pl-matching[^>]*answers-name="' + re.escape(nme) + '"', raw):
                    check(nme in d["correct_answers"], f"{q}: answers-name {nme} has no correct_answers entry")
            try:
                h, missing = mustache_render(raw, d)
            except ValueError as e:
                check(False, f"{q}: {e}"); break
            check(not missing, f"{q} seed {seed}: template names not set by generate(): {sorted(missing)}")
            lint_rendered(q, h, d, seed)
            if ARGS.katex and seed <= ARGS.katex_seeds:
                katex_items += [(q, e, disp) for e, disp in math_segments(h)]
            if dyn_figs and seed <= 2 and hasattr(mod, "file"):
                for a in dyn_figs:
                    fname = mustache_render(a.get("file-name", ""), d)[0]
                    try:
                        out = mod.file(dict(d, filename=fname))
                        if hasattr(out, "getvalue"):
                            out = out.getvalue()
                        check(out is not None and len(out) > 100, f"{q}: file({fname}) returned nothing useful")
                    except ImportError as e:
                        print(f"  note: {q}: dynamic figure not tested ({e})")
            if physfn:
                physfn(d) if physfn in PHYS.values() else physfn(d, check)
            variants.add(json.dumps(d["params"], sort_keys=True))
            if len(errors) > nerr0 + 20:
                break
        print(f"{q:36s} {N} seeds, {len(variants)} distinct variants, {'OK' if len(errors) == nerr0 else 'ERRORS'}")

# ---------------------------------------------------------------- assessments (skipped with --only)
for dirpath, _, files in ([] if ONLY else os.walk(os.path.join(ROOT, "courseInstances"))):
    for fn in files:
        if fn not in ("infoAssessment.json", "infoCourseInstance.json"):
            continue
        a = json.load(open(os.path.join(dirpath, fn)))
        u = a.get("uuid", "").lower()
        rel = os.path.relpath(os.path.join(dirpath, fn), ROOT)
        check(u and u not in uuids, f"{rel}: missing or duplicate uuid")
        uuids[u] = rel
        if fn != "infoAssessment.json":
            continue
        check(a.get("set") in SETS, f"{rel}: assessment set {a.get('set')!r} not declared")
        exam = a.get("type") == "Exam"
        for z in a.get("zones", []):
            for qq in z.get("questions", []):
                ids = [qq["id"]] if "id" in qq else [x["id"] for x in qq.get("alternatives", [])]
                for i in ids:
                    check(i in qids, f"{rel}: question id {i!r} does not exist")
                ap = qq.get("autoPoints")
                if exam:
                    check(isinstance(ap, (int, float)) or (isinstance(ap, list) and all(isinstance(x, (int, float)) for x in ap)),
                          f"{rel}: Exam autoPoints must be a number or list")
                    check("maxAutoPoints" not in qq and "triesPerVariant" not in qq, f"{rel}: maxAutoPoints/triesPerVariant are Homework-only")
                else:
                    check(isinstance(ap, (int, float)), f"{rel}: Homework autoPoints must be a number")
                    if "maxAutoPoints" in qq:
                        check(qq["maxAutoPoints"] >= ap, f"{rel}: maxAutoPoints < autoPoints")

# ---------------------------------------------------------------- JSON schemas (PrairieLearn's own, vendored in tools/schemas)
try:
    import jsonschema
    _S = {k: json.load(open(os.path.join(TOOLS, "schemas", k + ".json")))
          for k in ("infoQuestion", "infoAssessment", "infoCourse", "infoCourseInstance")}
    def _val(kind, path):
        try:
            jsonschema.validate(json.load(open(path)), _S[kind])
        except jsonschema.ValidationError as e:
            errors.append(f"{os.path.relpath(path, ROOT)}: schema: {e.message[:200]} at {list(e.absolute_path)}")
    _val("infoCourse", os.path.join(ROOT, "infoCourse.json"))
    for dirpath, _, files in os.walk(ROOT):
        if os.path.relpath(dirpath, ROOT).split(os.sep)[0] not in ("questions", "courseInstances"):
            continue
        for fn in files:
            kind = {"info.json": "infoQuestion", "infoAssessment.json": "infoAssessment",
                    "infoCourseInstance.json": "infoCourseInstance"}.get(fn)
            if kind and (fn != "info.json" or os.path.relpath(dirpath, ROOT).startswith("questions")):
                if ONLY and (kind != "infoQuestion" or os.path.basename(dirpath) not in ONLY):
                    continue
                _val(kind, os.path.join(dirpath, fn))
    print("JSON schemas: checked against PrairieLearn's schemas (tools/schemas)")
except ImportError:
    print("note: pip install jsonschema to validate the JSON files against PrairieLearn's schemas")

if ARGS.katex and katex_items:
    bad = katex_check(katex_items)
    print(f"KaTeX: {len(katex_items)} expressions compiled, {len(bad)} failed")
    for q, e, err in bad[:40]:
        errors.append(f"{q}: KaTeX cannot compile ${e[:80]}$ -> {err[:160]}")

if errors:
    print("\n".join(errors[:60])); sys.exit(1)
print("all checks passed")
