#!/usr/bin/env python3
"""Noms utilisés mais jamais liés — le genre de bug qui tue un run Kaggle après
deux heures (LM_LINES, .nbytes). Sur-approximation : on collecte TOUT nom lié
n'importe où, donc pas de faux positif de portée, quitte à en manquer."""
import ast, builtins, sys

src = open(sys.argv[1], encoding="utf-8").read()
tree = ast.parse(src)
bound = set(dir(builtins)) | {"__name__", "__file__"}

for n in ast.walk(tree):
    if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
        bound.add(n.id)
    elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        bound.add(n.name)
        for a in getattr(n, "args", None).args if hasattr(n, "args") else []:
            bound.add(a.arg)
        if hasattr(n, "args"):
            for a in (n.args.posonlyargs + n.args.kwonlyargs
                      + ([n.args.vararg] if n.args.vararg else [])
                      + ([n.args.kwarg] if n.args.kwarg else [])):
                bound.add(a.arg)
    elif isinstance(n, (ast.Import, ast.ImportFrom)):
        for a in n.names:
            bound.add((a.asname or a.name).split(".")[0])
    elif isinstance(n, ast.ExceptHandler) and n.name:
        bound.add(n.name)
    elif isinstance(n, (ast.comprehension,)):
        for t in ast.walk(n.target):
            if isinstance(t, ast.Name):
                bound.add(t.id)
    elif isinstance(n, ast.Lambda):
        for a in n.args.args:
            bound.add(a.arg)

bad = {}
for n in ast.walk(tree):
    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in bound:
        bad.setdefault(n.id, n.lineno)

for name, line in sorted(bad.items(), key=lambda kv: kv[1]):
    print(f"  ligne {line}: {name} — jamais lié")
print(f"{len(bad)} nom(s) non liés" if bad else "aucun nom non lié")
