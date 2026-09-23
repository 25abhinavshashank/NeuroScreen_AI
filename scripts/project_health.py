
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".venv", ".git", "__pycache__"}


def main():
    py_files = [p for p in ROOT.rglob("*.py") if not any(part in SKIP for part in p.parts)]
    empty = [str(p.relative_to(ROOT)) for p in py_files if p.stat().st_size == 0]
    syntax_errors = []
    for path in py_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except Exception as exc:
            syntax_errors.append({"file": str(path.relative_to(ROOT)), "error": str(exc)})

    registry = ROOT / "models" / "registry" / "registry.json"
    registry_data = json.loads(registry.read_text(encoding="utf-8")) if registry.exists() else {}
    active = registry_data.get("active_model", {})

    print("NEUROSCREEN AI PROJECT HEALTH")
    print(f"Python files: {len(py_files)}")
    print(f"Empty Python files: {len(empty)}")
    print(f"Syntax errors: {len(syntax_errors)}")
    print(f"Active model: {active.get('name')} {active.get('version')}")
    if empty:
        print("\nEmpty files:")
        for item in empty:
            print(" -", item)
    if syntax_errors:
        print("\nSyntax errors:")
        for item in syntax_errors:
            print(" -", item)

if __name__ == "__main__":
    main()
