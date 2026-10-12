"""Field at the centre O of a planar loop made of circular arcs and radial segments (Lecture 12, Biot-Savart).

Loop in the plane z = 0, one straight edge on the positive x axis (phi = 0), the other on the ray phi = theta.
  "sector":  O -> (R,0) -> arc of radius R to angle theta -> back to O        (O is the sector's vertex)
  "annulus": (R1,0) -> (R2,0) -> outer arc to theta -> inward along phi = theta -> inner arc back to phi = 0
Current I; "ccw" means counterclockwise seen from +z, i.e. +x-hat on the segment that lies on the x axis.
Biot-Savart, R-hat from the element to O: an arc of radius a and angle theta gives mu0 I theta/(4 pi a) along +z if it
circulates counterclockwise about O (seen from +z), along -z if clockwise. Radial segments: dl || R-hat -> 0.
    sector:  B_z = s mu0 I theta/(4 pi R)                        (s = +1 for ccw)
    annulus: B_z = s mu0 I theta/(4 pi) (1/R2 - 1/R1)            (outer arc ccw, inner arc cw for ccw: inner wins)
"""
import math
import random

MU0 = 4e-7 * math.pi


def num(v, sig=4):
    if v == 0:
        return "0"
    s = f"{v:.{sig}g}"
    if "e" in s:
        m, ex = s.split("e")
        return m + r"\times10^{" + str(int(ex)) + "}"
    return s


def generate(data):
    shape = random.choice(["sector", "annulus"])
    theta_deg = random.choice([45, 60, 90, 120, 135, 150, 180, 210, 240, 270, 300])
    I = random.choice([1.5, 2, 2.5, 3, 4, 5, 6, 8, 10])          # A
    ccw = random.choice([True, False])
    s = 1 if ccw else -1
    th = math.radians(theta_deg)

    if shape == "sector":
        R = random.choice([2, 3, 4, 5, 6, 8, 10])                   # cm
        R1 = R2 = 0
        B_arc_out = s * MU0 * I * th / (4 * math.pi * R * 1e-2)
        B_arc_in = 0.0
        Bz = B_arc_out
    else:
        R1 = random.choice([2, 3, 4, 5])
        R2 = R1 * random.choice([1.5, 2, 2.5, 3])
        R = 0
        B_arc_out = s * MU0 * I * th / (4 * math.pi * R2 * 1e-2)    # outer arc circulates with the loop
        B_arc_in = -s * MU0 * I * th / (4 * math.pi * R1 * 1e-2)    # inner arc circulates against it
        Bz = B_arc_out + B_arc_in
    up = Bz > 0
    zdir = r"+\hat{z}" if up else r"-\hat{z}"
    zopp = r"-\hat{z}" if up else r"+\hat{z}"
    sense = "counterclockwise" if ccw else "clockwise"

    # ---------------------------------------------------------------- (b) radial segments, MC (same for both shapes)
    a_near = "R" if shape == "sector" else "R_1"
    seg_choices = [
        {"text": r"Nothing: each straight segment lies on a line through O, so every element has $d\mathbf{l}$ parallel or antiparallel "
                 r"to $\hat{R}$ and $d\mathbf{l}\times\hat{R} = 0$.",
         "correct": "true", "feedback": r"Right: the cross product kills every element of a segment that points at the field point."},
        {"text": r"Nothing in total: the two segments carry current in opposite radial directions (one toward O, one away), so their nonzero fields cancel.",
         "correct": "false",
         "feedback": r"There is nothing to cancel: each segment gives zero by itself, element by element, because $d\mathbf{l}\parallel\hat{R}$. "
                     r"The two segments are not on the same line either (unless $\theta = 180^\circ$)."},
        {"text": r"Each acts like half of a long straight wire, adding $\mu_0I/(4\pi " + a_near + r")$, so together they add $\mu_0I/(2\pi " + a_near + r")$ along $\hat{z}$.",
         "correct": "false",
         "feedback": r"The half-infinite-wire result $\mu_0I/(4\pi a)$ is for a point at perpendicular distance $a$ from the wire's end. "
                     r"O lies on the line of each segment: perpendicular distance zero, and $d\mathbf{l}\times\hat{R} = 0$ throughout."},
        {"text": r"A field in the plane of the loop, along the bisector of the sector, which the arcs' contributions do not cancel.",
         "correct": "false",
         "feedback": r"For a planar loop and a field point in its plane, $d\mathbf{l}$ and $\hat{R}$ both lie in the plane, so any nonzero "
                     r"$d\mathbf{l}\times\hat{R}$ is along $\pm\hat{z}$. Here it is zero anyway."},
    ]
    random.shuffle(seg_choices)

    # ---------------------------------------------------------------- (c) direction / which arc dominates, MC
    if shape == "sector":
        dir_choices = [
            {"text": r"$+\hat{z}$, out of the page", "correct": "true" if up else "false",
             "feedback": (r"Right: a counterclockwise arc gives $d\mathbf{l}\times\hat{R}$ along $+\hat{z}$." if up else
                          r"Check $d\mathbf{l}\times\hat{R}$ at $(R, 0, 0)$: the clockwise arc has $d\mathbf{l}\propto-\hat{y}$ there and "
                          r"$\hat{R} = -\hat{x}$ (from the element to O), so $(-\hat{y})\times(-\hat{x}) = -\hat{z}$.")},
            {"text": r"$-\hat{z}$, into the page", "correct": "false" if up else "true",
             "feedback": (r"Right: a clockwise arc gives $d\mathbf{l}\times\hat{R}$ along $-\hat{z}$." if not up else
                          r"Check $d\mathbf{l}\times\hat{R}$ at $(R, 0, 0)$: the counterclockwise arc has $d\mathbf{l}\propto+\hat{y}$ there and "
                          r"$\hat{R} = -\hat{x}$ (from the element to O), so $\hat{y}\times(-\hat{x}) = +\hat{z}$.")},
            {"text": r"In the plane of the loop, along the bisector of the sector", "correct": "false",
             "feedback": r"Every $d\mathbf{l}$ and every $\hat{R}$ lie in the plane $z = 0$, so each $d\mathbf{B}$ at O is along $\pm\hat{z}$."},
            {"text": r"Nowhere: $\mathbf{B} = 0$ at O, because O lies on the loop itself", "correct": "false",
             "feedback": r"Only the straight segments, which point at O, contribute nothing; the arc, at distance $R$ all along, contributes $\mu_0I\theta/(4\pi R)$."},
        ]
    else:
        dir_choices = [
            {"text": r"Along $" + zdir + r"$: the inner arc wins, because each arc contributes $\mu_0I\theta/(4\pi a)$ and $R_1 \lt R_2$.",
             "correct": "true", "feedback": r"Right: the nearer arc dominates; the loop's sense only fixes which way each arc circulates about O."},
            {"text": r"Along $" + zopp + r"$: the outer arc wins, because it is longer ($\theta R_2 \gt \theta R_1$).",
             "correct": "false",
             "feedback": r"The length grows like $a$, but each element's contribution falls like $1/a^2$, so an arc gives $\mu_0I\theta/(4\pi a)$: "
                         r"the inner arc wins."},
            {"text": r"Nowhere: the two arcs subtend the same angle, so their contributions cancel and $\mathbf{B} = 0$ at O.",
             "correct": "false",
             "feedback": r"Same angle but different radii: $\mu_0I\theta/(4\pi R_1) \neq \mu_0I\theta/(4\pi R_2)$."},
            {"text": r"Along $" + zopp + r"$, the right-hand normal of the loop's circulation, as at the centre of a circular loop.",
             "correct": "false",
             "feedback": r"O is outside this loop. The thumb rule gives the field direction inside a loop; at O the inner arc, which circulates "
                         r"against the loop's sense about O, dominates."},
        ]
    random.shuffle(dir_choices)

    # ---------------------------------------------------------------- (d) select-all pool
    def st(sid, text, truth, why):
        return {"id": sid, "text": text, "correct": "true" if truth else "false",
                "feedback": ("True: " if truth else "False: ") + why}

    pool = [
        st("perp", r"At O, every element of this loop contributes a $d\mathbf{B}$ that is perpendicular to the plane of the loop, or zero.",
           True, r"$d\mathbf{l}$ and $\hat{R}$ both lie in the plane $z = 0$, so $d\mathbf{l}\times\hat{R}$ is along $\pm\hat{z}$."),
        st("Rhat", r"In $d\mathbf{B} = \mu_0I\,d\mathbf{l}\times\hat{R}/(4\pi R^2)$, the unit vector $\hat{R}$ points from the current element to the field point.",
           True, r"that is the convention; the opposite choice flips every $d\mathbf{B}$."),
        st("reverse", r"Reversing the current reverses $\mathbf{B}$ at O.", True, r"$d\mathbf{B}$ is linear in $I\,d\mathbf{l}$."),
        st("arc", r"An arc of radius $a$ and angle $\theta$ contributes $\mu_0I\theta/(4\pi a)$ at its centre; for $\theta = 2\pi$ this is the full-loop value $\mu_0I/(2a)$.",
           True, r"every element is at distance $a$ with $d\mathbf{l}\perp\hat{R}$: $\dfrac{\mu_0I}{4\pi a^2}\cdot a\theta$."),
        st("online", r"A current element contributes nothing at points on the line through it along $d\mathbf{l}$.",
           True, r"there $d\mathbf{l}\parallel\hat{R}$, so $d\mathbf{l}\times\hat{R} = 0$."),
        st("superpose", r"The fields of the arcs and of the straight segments add as vectors (superposition).",
           True, r"Biot–Savart is linear in the current; the loop's field is the sum over its pieces."),
        st("scale", r"Scaling every length of the loop by 2 (same current, same angle) halves $|\mathbf{B}|$ at O.",
           True, r"each element gives $\propto dl/R^2$, which scales as $1/(\text{length})$."),
        st("halftheta", r"Halving the angle $\theta$ (same current and radii) halves $|B_z|$ at O.",
           True, r"each arc's contribution is proportional to $\theta$, and the straight segments add nothing."),
        st("order", r"Writing $\hat{R}\times d\mathbf{l}$ instead of $d\mathbf{l}\times\hat{R}$ gives the same $\mathbf{B}$.",
           False, r"the cross product is antisymmetric: swapping the order reverses every $d\mathbf{B}$."),
        st("ampere", r"Ampère's law applied to a small circle in the plane $z = 0$ centred on O shows that $\mathbf{B} = 0$ at O, since no current crosses the flat disk inside that circle.",
           False, r"$\mathbf{B}$ at points of that circle is along $\pm\hat{z}$, perpendicular to $d\mathbf{l}$, so $\oint\mathbf{B}\cdot d\mathbf{l} = 0$ "
                  r"holds whatever $B$ is; without symmetry Ampère's law does not give $B$."),
        st("longer", r"For the same angle, an arc of larger radius gives a larger field at its centre, because it is longer.",
           False, r"length $\propto a$ but each element's field $\propto 1/a^2$: the arc gives $\mu_0I\theta/(4\pi a)$, smaller for larger $a$."),
        st("inverseR", r"Each current element's field falls off as $1/R$ with distance, like the field of a long straight wire.",
           False, r"an element's field falls as $1/R^2$; the $1/r$ of a long wire comes from integrating over its infinite length."),
        st("work", r"The magnetic force on a charge moving through O speeds it up or slows it down, depending on the sign of $B_z$.",
           False, r"$q\mathbf{v}\times\mathbf{B}$ is perpendicular to $\mathbf{v}$ and does no work; it only changes the direction of motion."),
        st("planeseg", r"A straight current segment produces zero field at every point in the plane that contains the loop.",
           False, r"it gives zero only on its own line; elsewhere in the plane its field is along $\pm\hat{z}$ and nonzero."),
    ]

    # ---------------------------------------------------------------- statement and answer-panel strings
    if shape == "sector":
        geom = (r"<b>Shape (i), a sector:</b> two straight segments, one from O to $(R, 0, 0)$ on the positive $x$ axis and one from O along "
                r"the ray $\phi = \theta$, joined by the circular arc of radius $R = " + num(R) + r"\ \mathrm{cm}$ centred on O that spans "
                r"$0 \le \phi \le \theta$. O is the corner where the two segments meet.")
        a_expl = (r"Only the arc contributes. Each of its elements is at distance $R$ with $d\mathbf{l}\perp\hat{R}$, so "
                  r"$|B_z| = \dfrac{\mu_0I}{4\pi R^2}\,(R\theta) = \dfrac{\mu_0I\theta}{4\pi R} = \dfrac{(4\pi\times10^{-7})(" + num(I) + r")(" +
                  num(th, 4) + r")}{4\pi(" + num(R / 100) + r")}\ \mathrm{T} = " + num(abs(Bz) * 1e6) + r"\ \mu\mathrm{T}$. The arc circulates " +
                  sense + r" about O (seen from $+z$), so $B_z = " + num(Bz * 1e6) + r"\ \mu\mathrm{T}$.")
        c_expl = (r"At $(R, 0, 0)$ the " + sense + r" arc has $d\mathbf{l}\propto" + (r"+" if ccw else r"-") + r"\hat{y}$ and $\hat{R} = -\hat{x}$, so "
                  r"$d\mathbf{l}\times\hat{R}\propto" + (r"\hat{y}\times(-\hat{x}) = +\hat{z}" if ccw else r"(-\hat{y})\times(-\hat{x}) = -\hat{z}") +
                  r"$: $\mathbf{B}$ at O is along $" + zdir + r"$, as the right-hand rule for the arc's circulation says.")
    else:
        geom = (r"<b>Shape (ii), an annular sector:</b> two circular arcs centred on O, of radii $R_1 = " + num(R1) + r"\ \mathrm{cm}$ and $R_2 = " +
                num(R2) + r"\ \mathrm{cm}$, both spanning $0 \le \phi \le \theta$, joined by two straight segments along the positive $x$ axis "
                r"($R_1 \le x \le R_2$) and along the ray $\phi = \theta$. O itself is not on the loop.")
        a_expl = (r"Each arc gives $\mu_0I\theta/(4\pi a)$. With the current " + sense + r" around the loop, the outer arc circulates " + sense +
                  r" about O and the inner arc the opposite way, so $B_z = " + ("+" if ccw else "-") + r"\dfrac{\mu_0I\theta}{4\pi}\Big(\dfrac{1}{R_2} - \dfrac{1}{R_1}\Big)"
                  r" = " + ("+" if ccw else "-") + r"(10^{-7})(" + num(I) + r")(" + num(th, 4) + r")\Big(\dfrac{1}{" + num(R2 / 100) +
                  r"} - \dfrac{1}{" + num(R1 / 100) + r"}\Big)\ \mathrm{T} = " + num(Bz * 1e6) + r"\ \mu\mathrm{T}$ "
                  r"(outer arc $" + num(B_arc_out * 1e6) + r"$, inner arc $" + num(B_arc_in * 1e6) + r"\ \mu\mathrm{T}$).")
        c_expl = (r"The inner arc's term is $R_2/R_1 = " + num(R2 / R1) + r"$ times the outer one's, so $\mathbf{B}$ at O follows the inner arc: along $" +
                  zdir + r"$. Note that this is <em>opposite</em> to the right-hand normal of the loop's own circulation: O lies outside the loop.")

    data["params"].update({
        "shape": shape, "theta_deg": theta_deg, "I": I, "ccw": ccw, "R": R, "R1": R1, "R2": R2,
        "geom": geom, "sense": sense, "xdir": r"+\hat{x}" if ccw else r"-\hat{x}",
        "I_str": num(I), "a_expl": a_expl, "c_expl": c_expl,
        "seg_choices": seg_choices, "dir_choices": dir_choices, "pool": pool,
    })
    data["correct_answers"]["Bz"] = float(Bz * 1e6)       # uT
