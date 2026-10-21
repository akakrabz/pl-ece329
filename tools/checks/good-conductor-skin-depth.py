"""Independent check for waves/good-conductor-skin-depth (Lecture 23).

Everything is parsed from the displayed text (sigma, eps_r, f of every pool item and of parts (b)-(e), the distance,
the numbers inside the choices) and recomputed with the EXACT lossy-medium solution, not the good-conductor shortcuts:
  * gamma = sqrt(j w mu0 (sigma + j w eps)) (complex numpy), eta = j w mu0/gamma;
  * delta = 1/Re(gamma), |eta|; the shortcut answers must agree within the shortcut's own error bound (~1/(2p));
  * the power ratio over l from time-averaged E x H of the exact fields (time sampling), then dB and percent;
  * the lag of H behind E from dense time sampling of the exact fields; the wavelength from Im(gamma);
  * matching keys from p = sigma/(w eps) with a margin of 2 from the thresholds.
"""
import math
import re

import numpy as np

MU0 = 4e-7 * math.pi
EPS0 = 8.854e-12
C0 = 1 / math.sqrt(MU0 * EPS0)
F_U = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}
U = {r"\mathrm{km}": 1e3, r"\mathrm{m}": 1.0, r"\mathrm{mm}": 1e-3, r"\mu\mathrm{m}": 1e-6,
     r"\mathrm{ms}": 1e-3, r"\mu\mathrm{s}": 1e-6, r"\mathrm{ns}": 1e-9, r"\mathrm{ps}": 1e-12,
     r"\Omega": 1.0, r"\mathrm{m}\Omega": 1e-3, r"\mu\Omega": 1e-6}
_ITEM = (r"^(.+) \(\$\\sigma = ([^$]+)\\ \\mathrm\{S/m\}\$, \$\\epsilon_r = ([\d.]+)\$\) at "
         r"\$f = ([\d.]+)\\ \\mathrm\{(GHz|MHz|kHz|Hz)\}\$$")
_QTY = r"= ([\d.]+(?:\\times10\^\{-?\d+\})?)\\ (\\mathrm\{[a-z]+\}|\\mu\\mathrm\{[a-z]\})"


def tex_num(s):
    m = re.fullmatch(r"(?:([\d.]+)\\times)?10\^\{(-?\d+)\}", s)
    if m:
        return (float(m.group(1)) if m.group(1) else 1.0) * 10 ** int(m.group(2))
    return float(s)


def qty(text, which=0):
    hits = re.findall(_QTY, text)
    v, u = hits[which]
    return tex_num(v) * U[u]


def exact(sig, er, f):
    om = 2 * math.pi * f
    gam = np.sqrt(1j * om * MU0 * (sig + 1j * om * er * EPS0))
    if gam.real < 0:
        gam = -gam
    return om, gam, 1j * om * MU0 / gam


def phys(d, check):
    p, c = d["params"], d["correct_answers"]
    # ---- (a) matching pool
    items = p["cls_items"]
    check(len(items) == 8 and len({it["text"] for it in items}) == 8, "skin: pool size / duplicates")
    seen = set()
    for it in items:
        m = re.fullmatch(_ITEM, it["text"])
        check(m is not None, f"skin: cannot parse pool item {it['text']}")
        sig, er, f = tex_num(m.group(2)), float(m.group(3)), float(m.group(4)) * F_U[m.group(5)]
        om, gam, eta = exact(sig, er, f)
        pr = sig / (om * er * EPS0)                     # |conduction| / |displacement| current density
        check(abs(math.atan(pr) / 2 - np.angle(eta)) < 1e-9, "skin: angle of eta vs loss tangent")
        if pr > 10:
            key = "good"
            check(pr >= 20, f"skin: {it['text']} too close to the good-conductor threshold ({pr})")
        elif pr < 0.1:
            key = "imperfect"
            check(pr <= 0.05, f"skin: {it['text']} too close to the imperfect-dielectric threshold ({pr})")
        else:
            key = "neither"
            check(0.2 <= pr <= 5, f"skin: {it['text']} too close to a threshold ({pr})")
        check(it["key"] == key, f"skin: matching key for {it['text']} is {it['key']}, physics says {key}")
        p0 = pr * er
        k0 = "good" if p0 > 10 else ("imperfect" if p0 < 0.1 else "neither")
        check(bool(it["trap"]) == (k0 != key), f"skin: eps0-trap remark for {it['text']}")
        seen.add(key)
    check(seen == {"good", "imperfect", "neither"}, "skin: pool lacks a category")

    # ---- (b)
    m = re.fullmatch(r"([\d.]+)\\ \\mathrm\{(GHz|MHz|kHz|Hz)\}", p["f_tex"])
    f = float(m.group(1)) * F_U[m.group(2)]
    sig, er = tex_num(p["sig_tex"]), float(p["er"])
    om, gam, eta = exact(sig, er, f)
    pr = sig / (om * er * EPS0)
    check(pr > 1000, f"skin: part (b) medium is not safely a good conductor ({pr})")
    bound = 1.0 / pr + 1e-9                             # shortcut error is about 1/(2p)
    check(abs(c["lt"] - pr) < 1e-9 * pr, f"skin: loss tangent {c['lt']} vs {pr}")
    dlt = 1 / gam.real
    dl_ans = c["delta"] * U[p["d_u"]]
    check(abs(dl_ans - dlt) < bound * dlt, f"skin: delta {dl_ans} vs exact {dlt}")
    eta_ans = c["eta"] * U[p["e_u"]]
    check(abs(eta_ans - abs(eta)) < bound * abs(eta), f"skin: |eta| {eta_ans} vs exact {abs(eta)}")
    check(abs(math.degrees(np.angle(eta)) - 45) < 0.05, "skin: angle of eta not ~45 deg")

    # ---- (c) power ratio over l from time-averaged E x H of the exact fields
    ell = float(p["ell_str"]) * U[p["ell_u"]]
    T = 1 / f
    ts = np.arange(64) * T / 64

    def S_avg(z):
        Et = np.real(np.exp(-gam * z) * np.exp(1j * om * ts))
        Ht = np.real(np.exp(-gam * z) / eta * np.exp(1j * om * ts))
        return float(np.mean(Et * Ht))                 # x_hat E times y_hat H -> z component
    ratio = S_avg(ell) / S_avg(0.0)
    dB_ex, pct_ex = -10 * math.log10(ratio), 100 * ratio
    check(abs(c["dB"] - dB_ex) < bound * dB_ex, f"skin: dB {c['dB']} vs exact {dB_ex}")
    check(abs(math.log(c["pct"] / pct_ex)) < 0.2303 * dB_ex * bound + 1e-9, f"skin: percent {c['pct']} vs exact {pct_ex}")
    check(abs(c["pct"] - 100 * 10 ** (-c["dB"] / 10)) < 1e-9 * c["pct"], "skin: percent and dB inconsistent")
    check(0.7 <= ell / dlt <= 6.5, "skin: distance outside the intended range")
    rt = float(p["pct_rtol"])                          # per-variant tolerance for the percentage
    check(2 * (ell / dlt) * 0.0074 <= rt <= 0.1, f"skin: pct tolerance {rt} vs 3-s.f. delta drift")
    check(abs(100 * math.exp(-ell / dlt) - c["pct"]) > 3 * rt * c["pct"], "skin: field-ratio slip would pass")

    # ---- (d) lag of H behind E (dense sampling of the exact fields at a fixed depth)
    tg = np.linspace(0, T, 400001)
    Et = np.abs(np.real(np.exp(1j * om * tg)))
    Ht = np.abs(np.real(np.exp(1j * om * tg) / eta))
    lag = (tg[int(np.argmax(Ht))] - tg[int(np.argmax(Et))]) % (T / 2)
    tol = 6 * T / 400000
    lch = {x["id"]: x for x in p["lag_choices"]}
    check(sum(x["correct"] == "true" for x in p["lag_choices"]) == 1 and lch["ok"]["correct"] == "true", "skin: lag MC key")
    X = qty(lch["ok"]["text"])
    check(abs(X - lag) < max(tol, 5e-3 * lag), f"skin: displayed T/8 {X} vs sampled lag {lag}")
    check("after" in lch["ok"]["text"] and "before" in lch["lead"]["text"], "skin: lag wording")
    check(abs(qty(lch["lead"]["text"]) - X) < 1e-12 and abs(lag - (T / 2 - X)) > 0.1 * T, "skin: 'leads' distractor would be true")
    Y = qty(lch["quarter"]["text"])
    check(abs(Y - T / 4) < 5e-3 * T / 4 and abs(lag - Y) > 0.05 * T, "skin: 'quarter period' distractor")
    check(lag > 0.05 * T, "skin: 'in phase' distractor would be true")

    # ---- (e) wavelength inside
    lam = 2 * math.pi / gam.imag
    mch = {x["id"]: x for x in p["lam_choices"]}
    check(sum(x["correct"] == "true" for x in p["lam_choices"]) == 1 and mch["ok"]["correct"] == "true", "skin: lambda MC key")
    check(abs(qty(mch["ok"]["text"]) - lam) < 5e-3 * lam, f"skin: displayed lambda vs exact {lam}")
    amp_ratio = abs(np.exp(-gam * lam)) * 100
    check(abs(amp_ratio - 0.19) < 0.01 and "0.19" in mch["ok"]["text"], f"skin: decay over a wavelength {amp_ratio}%")
    check(abs(qty(mch["weak"]["text"]) - lam) < 5e-3 * lam and abs(amp_ratio - 37) > 30, "skin: 'weak' distractor")
    check(abs(qty(mch["delta"]["text"]) - dlt) < 5e-3 * dlt and abs(dlt - lam) > 0.5 * lam, "skin: 'delta' distractor")
    lam_ll = 2 * math.pi / (om * math.sqrt(MU0 * er * EPS0))
    check(abs(qty(mch["free"]["text"]) - lam_ll) < 5e-3 * lam_ll and lam_ll > 10 * lam, "skin: lossless-wavelength distractor")
