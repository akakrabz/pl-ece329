"""A small Mustache renderer for checking PrairieLearn question.html templates offline.

Supports what PrairieLearn questions use: {{name}} (HTML-escaped), {{{name}}} and {{& name}} (raw),
{{#name}}...{{/name}} sections (lists iterate, dicts push context, truthy values render once),
{{^name}}...{{/name}} inverted sections, {{! comments}} and dotted names such as params.x.
Lookups that fail anywhere in the context stack are recorded in `missing` (PrairieLearn would render
them as empty strings, which is almost always a bug in a question).
"""
import html
import re

_TAG = re.compile(r"\{\{\{\s*([^}]+?)\s*\}\}\}|\{\{\s*([#^/&!]?)\s*([^}]*?)\s*\}\}", re.S)
_MISSING = object()


def _lookup(name, stack):
    if name == ".":
        return stack[-1]
    parts = name.split(".")
    for ctx in reversed(stack):
        if isinstance(ctx, dict) and parts[0] in ctx:
            val = ctx[parts[0]]
            for p in parts[1:]:
                if isinstance(val, dict) and p in val:
                    val = val[p]
                else:
                    return _MISSING
            return val
    return _MISSING


def _falsy(v):
    """Mustache.js semantics: missing, null, false, empty list, empty string and 0 skip a section."""
    if v is _MISSING or v is None or v is False:
        return True
    if isinstance(v, (list, str)) and len(v) == 0:
        return True
    if isinstance(v, (int, float)) and not isinstance(v, bool) and v == 0:
        return True
    return False


def _str(v):
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
        # JavaScript prints 3.0 as "3"; mimic that so the offline render matches PrairieLearn.
        return str(int(v))
    return str(v)


def _parse(tpl):
    """Return a nested token list: ('text', s) | ('var', name, raw) | ('sec', name, inverted, children)."""
    tokens, stack = [], [("root", [])]
    pos = 0
    for m in _TAG.finditer(tpl):
        if m.start() > pos:
            stack[-1][1].append(("text", tpl[pos:m.start()]))
        pos = m.end()
        if m.group(1) is not None:
            stack[-1][1].append(("var", m.group(1).strip(), True))
            continue
        sigil, name = m.group(2), m.group(3).strip()
        if sigil == "!":
            continue
        if sigil in ("#", "^"):
            node = ("sec", name, sigil == "^", [])
            stack[-1][1].append(node)
            stack.append((name, node[3]))
        elif sigil == "/":
            if len(stack) == 1 or stack[-1][0] != name:
                raise ValueError(f"mustache: unbalanced section close {{{{/{name}}}}}")
            stack.pop()
        else:
            stack[-1][1].append(("var", name, sigil == "&"))
    if len(stack) != 1:
        raise ValueError(f"mustache: unclosed section {{{{#{stack[-1][0]}}}}}")
    if pos < len(tpl):
        stack[0][1].append(("text", tpl[pos:]))
    return stack[0][1]


def _render(nodes, stack, missing):
    out = []
    for node in nodes:
        kind = node[0]
        if kind == "text":
            out.append(node[1])
        elif kind == "var":
            v = _lookup(node[1], stack)
            if v is _MISSING:
                missing.add(node[1])
                continue
            if v is None:
                continue
            s = _str(v)
            out.append(s if node[2] else html.escape(s, quote=True))
        else:
            _, name, inverted, children = node
            v = _lookup(name, stack)
            if v is _MISSING:
                missing.add(name)
            if inverted:
                if _falsy(v):
                    out.append(_render(children, stack, missing))
                continue
            if _falsy(v):
                continue
            if isinstance(v, list):
                for item in v:
                    out.append(_render(children, stack + [item], missing))
            elif isinstance(v, dict):
                out.append(_render(children, stack + [v], missing))
            else:
                out.append(_render(children, stack, missing))
    return "".join(out)


def render(template, data):
    """Render `template` with `data` (a dict). Returns (html, set_of_missing_names)."""
    missing = set()
    return _render(_parse(template), [data], missing), missing


if __name__ == "__main__":  # tiny self-test
    t = "{{#params.ch}}<a c=\"{{correct}}\">{{{text}}}</a>{{/params.ch}}{{^params.none}}N{{/params.none}}{{params.x}}{{params.y}}"
    h, miss = render(t, {"params": {"ch": [{"correct": "true", "text": "$a<b$"}], "x": 3.0, "y": "<i>"}})
    assert h == '<a c="true">$a<b$</a>N3&lt;i&gt;', h
    assert miss == {"params.none"}, miss
    print("mustache_lite OK")
