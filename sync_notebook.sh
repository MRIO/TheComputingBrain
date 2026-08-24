#!/bin/zsh
set -euo pipefail

# The ipynb file is always authoritative. This avoids Jupytext's --sync
# timestamp behavior accidentally copying an older Markdown file into it.
readonly FORMATS='ipynb,md,py:percent'
readonly FALLBACK_PYTHON='/Users/mrio/Sync/Courses/NB2181/.venv/bin/python'

if [[ -n "${JUPYTEXT_PYTHON:-}" ]]; then
  readonly PYTHON="$JUPYTEXT_PYTHON"
elif command -v jupytext >/dev/null 2>&1; then
  readonly JUPYTEXT=(jupytext)
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
  done < <(find . -type f -name '*.ipynb' ! -path '*/.ipynb_checkpoints/*' -print0)
fi

if [[ "$mode" == sync ]]; then
  for notebook in "${notebooks[@]}"; do
    # Update pairing metadata without --set-formats/--sync: those commands may
    # select an existing text representation as their input based on mtimes.
    "$JUPYTEXT[@]" --update-metadata \
      '{"jupytext":{"formats":"ipynb,md,py:percent"}}' "$notebook" --quiet
    "$JUPYTEXT[@]" --to md --output "${notebook%.ipynb}.md" "$notebook" --quiet
    "$JUPYTEXT[@]" --to py:percent --output "${notebook%.ipynb}.py" "$notebook" --quiet
  done
  exit 0
fi

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/jupytext-check.XXXXXX")"
trap 'rm -rf "$tmp_dir"' EXIT
result=0
index=0

for notebook in "${notebooks[@]}"; do
  (( index += 1 ))
  markdown="${notebook%.ipynb}.md"
  script="${notebook%.ipynb}.py"
  generated_md="$tmp_dir/$index.md"
  generated_py="$tmp_dir/$index.py"

  "$JUPYTEXT[@]" --to md --output "$generated_md" "$notebook" --quiet
  "$JUPYTEXT[@]" --to py:percent --output "$generated_py" "$notebook" --quiet

  if [[ ! -f "$markdown" ]] || ! cmp -s "$generated_md" "$markdown"; then
    print -u2 -- "Out of sync: $markdown"
    result=1
  fi
  if [[ ! -f "$script" ]] || ! cmp -s "$generated_py" "$script"; then
    print -u2 -- "Out of sync: $script"
    result=1
  fi
done

if (( result == 0 )); then
  print -- 'All Markdown and Python representations are in sync.'
fi
exit "$result"
