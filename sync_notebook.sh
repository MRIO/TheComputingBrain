#!/bin/zsh
set -euo pipefail

# Jupytext uses the most recently modified representation as the source when
# synchronizing. Edit only one representation of a pair between syncs.
readonly FORMATS='ipynb,md,py:percent'
readonly FALLBACK_PYTHON='/Users/mrio/Sync/Courses/NB2181/.venv/bin/python'

if [[ -n "${JUPYTEXT_PYTHON:-}" ]]; then
  readonly PYTHON="$JUPYTEXT_PYTHON"
elif command -v jupytext >/dev/null 2>&1; then
  readonly JUPYTEXT=(jupytext)
  readonly PYTHON="$(command -v python3)"
elif [[ -x "$FALLBACK_PYTHON" ]]; then
  readonly PYTHON="$FALLBACK_PYTHON"
else
  print -u2 'Jupytext was not found. Install it or set JUPYTEXT_PYTHON.'
  exit 2
fi

if (( ${+JUPYTEXT} == 0 )); then
  readonly JUPYTEXT=("$PYTHON" -m jupytext)
fi

mode=sync
if [[ "${1:-}" == '--check' ]]; then
  mode=check
  shift
fi

typeset -a notebooks
if (( $# )); then
  for notebook in "$@"; do
    if [[ ! -f "$notebook" || "$notebook" != *.ipynb ]]; then
      print -u2 -- "Not an ipynb file: $notebook"
      exit 2
    fi
    notebooks+=("$notebook")
  done
else
  while IFS= read -r -d '' notebook; do
    notebooks+=("$notebook")
  done < <(find . -type f -name '*.ipynb' \
    ! -path '*/.ipynb_checkpoints/*' \
    ! -path './tmp/*' \
    ! -path './.venv/*' \
    -print0)
fi

if [[ "$mode" == sync ]]; then
  for notebook in "${notebooks[@]}"; do
    "$JUPYTEXT[@]" --sync "$notebook" --quiet
  done
  exit 0
fi

result=0

for notebook in "${notebooks[@]}"; do
  markdown="${notebook%.ipynb}.md"
  script="${notebook%.ipynb}.py"

  if [[ ! -f "$markdown" || ! -f "$script" ]]; then
    print -u2 -- "Missing paired file for: $notebook"
    result=1
    continue
  fi

  if ! "$PYTHON" -c '
import sys
import jupytext

def cells(path):
    notebook = jupytext.read(path)
    # Text formats do not preserve a final newline inside every cell. Jupytext
    # treats that as an expected round-trip difference.
    return [(cell.cell_type, cell.source.rstrip("\n")) for cell in notebook.cells]

reference = cells(sys.argv[1])
if cells(sys.argv[2]) != reference or cells(sys.argv[3]) != reference:
    raise SystemExit(1)
' "$notebook" "$markdown" "$script"; then
    print -u2 -- "Out of sync: ${notebook%.ipynb}.{ipynb,md,py}"
    result=1
  fi
done

if (( result == 0 )); then
  print -- 'All Markdown and Python representations are in sync.'
fi
exit "$result"
