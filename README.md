# Python Notebooks for the Computing Brain

## Notebook workflow

Local notebook execution now targets JupyterLab.

1. Create a local environment: `python3 -m venv .venv`
2. Install notebook dependencies: `.venv/bin/pip install -r requirements-notebooks.txt`
3. Start JupyterLab: `.venv/bin/jupyter lab`

Interactive notebooks use a shared runtime setup cell:

- In Google Colab it installs any missing notebook-only packages and enables the custom widget manager.
- In local JupyterLab it leaves the environment unchanged and relies on the packages from `requirements-notebooks.txt`.

## Text-first authoring

Markdown notebooks are the canonical source format for paired notebooks.

- Sync pairs manually with `.venv/bin/python scripts/notebook_workflow.py sync --ensure-md`
- Rebuild the inventory with `.venv/bin/python scripts/notebook_workflow.py inventory`
- Validate notebooks with `.venv/bin/python scripts/notebook_workflow.py validate`

New `.ipynb` files are ignored by default. Existing tracked `.ipynb` files remain in git until you explicitly remove them from the index in a separate cleanup pass.

To enable commit-time sync and validation, configure the repo hooks once:

`git config core.hooksPath .githooks`

## Learning Goals per Project



### 00 Python









![Overview_of_Projects](./resources/Overview_of_Projects.png)
