#!/usr/bin/env python3
"""Notebook workflow utilities for Colab/JupyterLab stabilization and sync."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent
INVENTORY_PATH = ROOT / "resources" / "notebook_inventory.md"
DEFAULT_TEXT_FORMAT = "md"
PAIR_FORMATS = f"ipynb,{DEFAULT_TEXT_FORMAT}"
FIRST_RUN_MARKDOWN = (
    "## Run This First\n"
    "\n"
    "Run the next code cell before the rest of the notebook.\n"
    "\n"
    "- In Google Colab it installs any missing notebook-only packages and enables widget support.\n"
    "- In local JupyterLab it only verifies imports against your active environment.\n"
    "- Local setup: create a virtual environment and install the packages in `requirements-notebooks.txt`.\n"
)
SETUP_CELL_TAG = "notebook-runtime-setup"
SETUP_CELL_SOURCE_TEMPLATE = """# Notebook runtime setup for Google Colab and local JupyterLab.
import importlib
import subprocess
import sys

try:
    from google.colab import output as colab_output
    IS_COLAB = True
except ImportError:
    colab_output = None
    IS_COLAB = False


def ensure_notebook_packages(requirements):
    if not IS_COLAB:
        return

    missing = []
    for package_name, module_name in requirements:
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)

    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


NOTEBOOK_REQUIREMENTS = {requirements}
ensure_notebook_packages(NOTEBOOK_REQUIREMENTS)

if IS_COLAB:
    colab_output.enable_custom_widget_manager()
"""

STD_LIB_MODULES = {
    "collections",
    "copy",
    "csv",
    "dataclasses",
    "datetime",
    "functools",
    "glob",
    "importlib",
    "io",
    "itertools",
    "json",
    "math",
    "numbers",
    "operator",
    "os",
    "pathlib",
    "pickle",
    "random",
    "re",
    "statistics",
    "string",
    "subprocess",
    "sys",
    "time",
    "typing",
    "warnings",
}

PACKAGE_BY_MODULE = {
    "altair": "altair",
    "brian2": "brian2",
    "brian2tools": "brian2tools",
    "imageio": "imageio",
    "ipywidgets": "ipywidgets",
    "matplotlib": "matplotlib",
    "mpl_toolkits": "matplotlib",
    "nengo": "nengo",
    "nengo_extras": "nengo_extras",
    "nengo_spa": "nengo_spa",
    "netgraph": "netgraph",
    "networkx": "networkx",
    "numpy": "numpy",
    "pandas": "pandas",
    "PIL": "Pillow",
    "requests": "requests",
    "scipy": "scipy",
    "seaborn": "seaborn",
    "skimage": "scikit-image",
    "sympy": "sympy",
    "vega_datasets": "vega_datasets",
}

NON_NOTEBOOK_PACKAGES = {
    "google",
    "IPython",
}


@dataclass
class NotebookInfo:
    path: Path
    md_path: Path | None
    has_widgets: bool
    has_brian2: bool
    has_runtime_setup: bool
    imports: set[str]


def iter_notebooks() -> Iterable[Path]:
    for path in sorted(ROOT.rglob("*.ipynb")):
        if ".ipynb_checkpoints" in path.parts:
            continue
        yield path


def load_notebook(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_notebook(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, ensure_ascii=False)
        fh.write("\n")


def cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def set_cell_source(cell: dict, source: str) -> None:
    cell["source"] = source


def parse_imports(source: str) -> set[str]:
    modules: set[str] = set()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
    except SyntaxError:
        return modules

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def classify_notebook(path: Path) -> NotebookInfo:
    data = load_notebook(path)
    imports: set[str] = set()
    text_parts: list[str] = []
    has_runtime_setup = False

    for cell in data.get("cells", []):
        source = cell_source(cell)
        text_parts.append(source)
        imports.update(parse_imports(source))
        if SETUP_CELL_TAG in cell.get("metadata", {}).get("tags", []):
            has_runtime_setup = True

    text = "\n".join(text_parts)
    md_path = path.with_suffix(f".{DEFAULT_TEXT_FORMAT}")
    has_widgets = "ipywidgets" in imports or "interact(" in text or "interactive(" in text or "widgets." in text
    has_brian2 = "brian2" in imports or "from brian2" in text or "import brian2" in text
    return NotebookInfo(
        path=path,
        md_path=md_path if md_path.exists() else None,
        has_widgets=has_widgets,
        has_brian2=has_brian2,
        has_runtime_setup=has_runtime_setup,
        imports=imports,
    )


def notebook_dependencies(imports: set[str], has_widgets: bool, has_brian2: bool) -> list[tuple[str, str]]:
    requirements: list[tuple[str, str]] = []
    seen: set[str] = set()

    if has_widgets:
        requirements.append(("ipywidgets", "ipywidgets"))
        seen.add("ipywidgets")
    if has_brian2:
        requirements.append(("brian2", "brian2"))
        seen.add("brian2")

    for module in sorted(imports):
        if module in NON_NOTEBOOK_PACKAGES or module in STD_LIB_MODULES:
            continue
        package = PACKAGE_BY_MODULE.get(module)
        if not package or package in seen:
            continue
        requirements.append((package, module))
        seen.add(package)
    return requirements


def update_notebook_metadata(data: dict) -> None:
    metadata = data.setdefault("metadata", {})
    jupytext = metadata.setdefault("jupytext", {})
    jupytext["formats"] = PAIR_FORMATS
    widgets_meta = metadata.get("widgets")
    if widgets_meta is not None:
        metadata.pop("widgets", None)


def clear_execution_state(data: dict) -> None:
    for cell in data.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []


def ensure_cell_ids(data: dict, notebook_path: Path | None = None) -> None:
    seen: set[str] = set()
    notebook_key = str(notebook_path.relative_to(ROOT)) if notebook_path is not None else "notebook"
    for index, cell in enumerate(data.get("cells", []), start=1):
        source = cell_source(cell)
        digest = hashlib.sha1(f"{notebook_key}:{index}:{source}".encode("utf-8")).hexdigest()[:12]
        cell_id = cell.get("id") or f"cell-{digest}"
        while cell_id in seen:
            digest = hashlib.sha1(f"{cell_id}:{index}".encode("utf-8")).hexdigest()[:12]
            cell_id = f"cell-{digest}"
        cell["id"] = cell_id
        seen.add(cell_id)


def make_markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def make_setup_cell(requirements: list[tuple[str, str]]) -> dict:
    req_repr = repr(requirements)
    source = SETUP_CELL_SOURCE_TEMPLATE.format(requirements=req_repr)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": [SETUP_CELL_TAG]},
        "outputs": [],
        "source": source,
    }


def needs_runtime_setup(info: NotebookInfo) -> bool:
    return info.has_widgets or info.has_brian2 or "nengo" in info.imports or "nengo_extras" in info.imports or "nengo_spa" in info.imports


def strip_old_bootstrap(cells: list[dict], requirements: list[tuple[str, str]]) -> list[dict]:
    cleaned: list[dict] = []
    package_names = {package for package, _ in requirements}

    for cell in cells:
        source = cell_source(cell)
        tags = cell.get("metadata", {}).get("tags", [])
        if SETUP_CELL_TAG in tags:
            continue
        if cell.get("cell_type") == "code":
            lines = source.splitlines()
            filtered: list[str] = []
            removed_entire_cell = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("!pip install"):
                    lower = stripped.lower()
                    if any(name.lower() in lower for name in package_names | {"ipympl"}):
                        continue
                if "get_ipython().kernel.do_shutdown" in stripped:
                    continue
                if stripped == "%matplotlib widget":
                    continue
                if stripped == "%matplotlib notebook":
                    continue
                if stripped == "from google.colab import output":
                    continue
                if stripped == "output.enable_custom_widget_manager()":
                    continue
                filtered.append(line)

            if not filtered and ("!pip install" in source or "%matplotlib widget" in source or "google.colab" in source):
                removed_entire_cell = True
            if removed_entire_cell:
                continue

            new_source = "\n".join(filtered).strip("\n")
            set_cell_source(cell, new_source + ("\n" if new_source else ""))
        cleaned.append(cell)
    return cleaned


def upsert_runtime_cells(data: dict, requirements: list[tuple[str, str]]) -> None:
    cells = data.get("cells", [])
    cells = strip_old_bootstrap(cells, requirements)

    first_code_index = next((idx for idx, cell in enumerate(cells) if cell.get("cell_type") == "code"), None)
    if first_code_index is None:
        return

    if first_code_index == 0 or cell_source(cells[first_code_index - 1]).strip() != FIRST_RUN_MARKDOWN.strip():
        cells.insert(first_code_index, make_markdown_cell(FIRST_RUN_MARKDOWN))
        first_code_index += 1

    if SETUP_CELL_TAG not in cells[first_code_index].get("metadata", {}).get("tags", []):
        cells.insert(first_code_index, make_setup_cell(requirements))
    else:
        cells[first_code_index] = make_setup_cell(requirements)

    data["cells"] = cells


def stabilize_notebook(path: Path) -> bool:
    data = load_notebook(path)
    info = classify_notebook(path)
    if not needs_runtime_setup(info):
        update_notebook_metadata(data)
        clear_execution_state(data)
        ensure_cell_ids(data, path)
        save_notebook(path, data)
        return False

    requirements = notebook_dependencies(info.imports, info.has_widgets, info.has_brian2)
    upsert_runtime_cells(data, requirements)
    update_notebook_metadata(data)
    clear_execution_state(data)
    ensure_cell_ids(data, path)
    save_notebook(path, data)
    return True


def run_command(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def sync_pair(ipynb_path: Path, md_path: Path) -> None:
    if not ipynb_path.exists() and not md_path.exists():
        raise FileNotFoundError(f"Neither pair exists: {ipynb_path} / {md_path}")
    if not ipynb_path.exists():
        newer = md_path
        direction = "md_to_ipynb"
    elif not md_path.exists():
        newer = ipynb_path
        direction = "ipynb_to_md"
    else:
        direction = "ipynb_to_md" if ipynb_path.stat().st_mtime >= md_path.stat().st_mtime else "md_to_ipynb"
        newer = ipynb_path if direction == "ipynb_to_md" else md_path

    if direction == "ipynb_to_md":
        run_command([sys.executable, "-m", "jupytext", "--to", DEFAULT_TEXT_FORMAT, str(ipynb_path), "--output", str(md_path)])
    else:
        run_command([sys.executable, "-m", "jupytext", "--to", "notebook", str(md_path), "--output", str(ipynb_path)])

    data = load_notebook(ipynb_path)
    update_notebook_metadata(data)
    ensure_cell_ids(data, ipynb_path)
    save_notebook(ipynb_path, data)
    os.utime(ipynb_path, (newer.stat().st_atime, newer.stat().st_mtime))
    os.utime(md_path, (newer.stat().st_atime, newer.stat().st_mtime))


def validate_notebook(path: Path) -> tuple[bool, str]:
    try:
        data = load_notebook(path)
    except Exception as exc:  # pragma: no cover - diagnostic path
        return False, f"{path}: failed to parse JSON: {exc}"

    if data.get("nbformat") is None:
        return False, f"{path}: missing nbformat"
    if not isinstance(data.get("cells"), list):
        return False, f"{path}: missing cells list"

    for idx, cell in enumerate(data["cells"], start=1):
        if "cell_type" not in cell:
            return False, f"{path}: cell {idx} missing cell_type"
        if "source" not in cell:
            return False, f"{path}: cell {idx} missing source"
    return True, f"{path}: ok"


def build_inventory() -> str:
    widget_brian2: list[str] = []
    widget_only: list[str] = []
    brian2_only: list[str] = []

    for path in iter_notebooks():
        info = classify_notebook(path)
        rel = str(path.relative_to(ROOT))
        if info.has_widgets and info.has_brian2:
            widget_brian2.append(rel)
        elif info.has_widgets:
            widget_only.append(rel)
        elif info.has_brian2:
            brian2_only.append(rel)

    def render_section(title: str, items: list[str]) -> list[str]:
        lines = [f"## {title}", ""]
        if not items:
            lines.append("- None")
        else:
            lines.extend(f"- `{item}`" for item in items)
        lines.append("")
        return lines

    lines = [
        "# Notebook Inventory",
        "",
        "Generated by `scripts/notebook_workflow.py inventory`.",
        "",
    ]
    lines.extend(render_section("ipywidgets + brian2", widget_brian2))
    lines.extend(render_section("ipywidgets only", widget_only))
    lines.extend(render_section("brian2 only", brian2_only))
    return "\n".join(lines).rstrip() + "\n"


def cmd_inventory(_: argparse.Namespace) -> int:
    INVENTORY_PATH.write_text(build_inventory(), encoding="utf-8")
    print(f"Wrote {INVENTORY_PATH.relative_to(ROOT)}")
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    notebooks = [classify_notebook(path) for path in iter_notebooks()]
    paired = [info for info in notebooks if info.md_path is not None or args.ensure_md]

    for info in paired:
        md_path = info.path.with_suffix(f".{DEFAULT_TEXT_FORMAT}")
        if not md_path.exists() and not args.ensure_md:
            continue
        sync_pair(info.path, md_path)
        print(f"Synced {info.path.relative_to(ROOT)}")
    return 0


def cmd_stabilize(args: argparse.Namespace) -> int:
    changed = 0
    for path in iter_notebooks():
        if stabilize_notebook(path):
            changed += 1
            print(f"Stabilized {path.relative_to(ROOT)}")
    if args.inventory:
        INVENTORY_PATH.write_text(build_inventory(), encoding="utf-8")
        print(f"Wrote {INVENTORY_PATH.relative_to(ROOT)}")
    print(f"Updated runtime cells in {changed} notebooks")
    return 0


def cmd_validate(_: argparse.Namespace) -> int:
    failures = 0
    for path in iter_notebooks():
        ok, message = validate_notebook(path)
        print(message)
        if not ok:
            failures += 1
    return 1 if failures else 0


def cmd_check(args: argparse.Namespace) -> int:
    rc = cmd_validate(args)
    return rc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync = subparsers.add_parser("sync", help="Sync ipynb/md pairs using jupytext.")
    sync.add_argument("--ensure-md", action="store_true", help="Create markdown pairs for notebooks that do not have one yet.")
    sync.set_defaults(func=cmd_sync)

    stabilize = subparsers.add_parser("stabilize", help="Patch notebooks for Colab/JupyterLab runtime compatibility.")
    stabilize.add_argument("--inventory", action="store_true", help="Rewrite the inventory document after stabilization.")
    stabilize.set_defaults(func=cmd_stabilize)

    inventory = subparsers.add_parser("inventory", help="Generate the notebook inventory document.")
    inventory.set_defaults(func=cmd_inventory)

    validate = subparsers.add_parser("validate", help="Validate notebook JSON structure.")
    validate.set_defaults(func=cmd_validate)

    check = subparsers.add_parser("check", help="Alias for validate, suitable for hooks.")
    check.set_defaults(func=cmd_check)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
