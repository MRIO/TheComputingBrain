# The Computing Brain Notebooks

This repository has been cleaned so the notebook set runs top-to-bottom in a local Python environment instead of relying on Google Colab setup cells.

## Recommended setup

Use a local virtual environment in the project root:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install "numpy<2" scipy matplotlib brian2 networkx netgraph ipywidgets imageio sympy symengine ipython ipykernel nbformat nbclient
```

The `numpy<2` pin matters for Brian-based notebooks.

## Open notebooks interactively

If you want to work in Jupyter:

```bash
.venv/bin/python -m ipykernel install --user --name computing-brain
.venv/bin/python -m jupyter lab
```

If `jupyter` is not installed yet, add it with:

```bash
.venv/bin/pip install jupyterlab
```

## Run notebooks headlessly

This is the same style of execution used for validation:

```bash
.venv/bin/python - <<'PY'
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient

os.environ.update({
    "MPLBACKEND": "Agg",
    "IPYTHONDIR": "/tmp/ipython-codex",
    "JUPYTER_RUNTIME_DIR": "/tmp/jupyter-runtime-codex",
    "JUPYTER_CONFIG_DIR": "/tmp/jupyter-config-codex",
    "JUPYTER_DATA_DIR": "/tmp/jupyter-data-codex",
    "XDG_CACHE_HOME": "/tmp/xdg-cache-codex",
})

notebooks = [
    Path("01_The Resting Membrane Potential_/[Solutions] Simulation of Membrane Potential.ipynb"),
    Path("02_Biophysics and HH (two parts)/[solutions] DYI Neuron Model - Part 1.ipynb"),
    Path("02_Biophysics and HH (two parts)/[solutions] DIY Neuron Model - Part 1 - Interactive.ipynb"),
    Path("02_Biophysics and HH (two parts)/[Solutions] DIY Neuron Model - Part 2.ipynb"),
    Path("03_Neurodynamics/[solutions] Simplified neuron models (AdEx).ipynb"),
    Path("03_Neurodynamics/[solutions] Simplified neuron models(v.2).ipynb"),
    Path("04_SpikingNetworks/[solutions] Spiking Networks.ipynb"),
    Path("05_Networks and Graphs/[solutions] Networks and Graphs Tutorial.ipynb"),
    Path("06_FFNN and Receptive Fields/DIY Receptive Field.ipynb"),
    Path("07_Simple Neural Network/Simple Neural Network.ipynb"),
    Path("09_Hebbian Learning/Hebbian Learning and Receptive Fields.ipynb"),
    Path("09_Hebbian Learning/Elias Version of Self-organisation_ANSWERS.ipynb"),
    Path("09_Hebbian Learning/Self-organisation_ANSWERS.ipynb"),
    Path("10_RNN and Hopfield Network/Hopfield-NeckerCube_ANSWERS.ipynb"),
]

for key in ["IPYTHONDIR", "JUPYTER_RUNTIME_DIR", "JUPYTER_CONFIG_DIR", "JUPYTER_DATA_DIR", "XDG_CACHE_HOME"]:
    Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

for path in notebooks:
    print(f"Running {path}")
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()

print("All notebooks executed successfully.")
PY
```

## Notes

- The notebooks were normalized for local execution and no longer depend on Colab-only helpers.
- Some unfinished exercise cells were replaced with runnable example code.
- The `03_Neurodynamics` notebooks were trimmed to the runnable example sections where the later cells were incomplete drafts.
