#!/usr/bin/env python3
"""Print a question's question.html rendered for one seed, the way PrairieLearn's Mustache pass would see it.

    python3 tools/render_question.py <question-name> [seed]      # e.g. render_question.py faraday-loop-emf 7

Shows the generated params/correct answers, then the rendered HTML (elements are not executed, so you see
the raw <pl-...> markup with every {{params.x}} filled in). Use it to proofread wording, numbers and LaTeX.
"""
import glob
import importlib.util
import json
import os
import random
import sys

import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS)
from mustache_lite import render  # noqa: E402

name = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
hits = glob.glob(os.path.join(ROOT, "questions", "*", name))
if not hits:
    sys.exit(f"no question named {name}")
qpath = hits[0]
spec = importlib.util.spec_from_file_location("server", os.path.join(qpath, "server.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
random.seed(seed)
np.random.seed(seed % (2**32 - 1))
data = {"params": {}, "correct_answers": {}, "variant_seed": seed, "options": {}}
mod.generate(data)
print("=== correct_answers:", json.dumps(data["correct_answers"], indent=1, default=str))
html, missing = render(open(os.path.join(qpath, "question.html")).read(), data)
if missing:
    print("=== MISSING template names:", sorted(missing))
print("=== rendered question.html (seed %d)" % seed)
print(html)
