# The Computing Brain Projects

This repository contains notebook-based projects for learning computational neuroscience and neural computation by building working simulations. The projects move from single-cell biophysics to network structure, simple neural networks, receptive fields, and Hebbian learning.

The overarching goal is to connect biological mechanisms to computational ideas. Students start with membrane potentials and ion-channel models, then use those foundations to reason about graph connectivity, activity propagation, manually tuned classifiers, and learning rules.

![Overview of the projects](resources/Overview_of_Projects.png)

## Project Index

For the detailed project-by-project learning goals, see the [Project Index and Learning Goals](Project%20Index%20and%20Learning%20Goals.md).

That index collects the explicit learning objectives from the notebooks, including goals such as simulating membrane-potential ODEs in Brian2, assembling Hodgkin-Huxley-style ionic currents, interpreting adjacency matrices as graphs, tuning feed-forward networks, understanding receptive fields, and exploring unsupervised Hebbian learning.

## What Is In This Repository

The active notebook set in this branch is:

- [Project Zero: Resting Membrane Potential](01_The%20Resting%20Membrane%20Potential_/Project_Zero_The_Membrane_potential.ipynb)
- [Simulation of Membrane Potential](01_The%20Resting%20Membrane%20Potential_/%5BSolutions%5D%20Simulation%20of%20Membrane%20Potential.ipynb)
- [DIY Neuron Model - Part 1 - Interactive](02_Biophysics%20and%20HH%20%28two%20parts%29/%5Bsolutions%5D%20DIY%20Neuron%20Model%20-%20Part%201%20-%20Interactive.ipynb)
- [Networks and Graphs Tutorial](05_Networks%20and%20Graphs/%5Bsolutions%5D%20Networks%20and%20Graphs%20Tutorial.ipynb)
- [DIY Receptive Field](06_FFNN%20and%20Receptive%20Fields/DIY%20Receptive%20Field.ipynb)
- [Simple Neural Network](07_Simple%20Neural%20Network/Simple%20Neural%20Network.ipynb)
- [Hebbian Learning and Receptive Fields](09_Hebbian%20Learning/Hebbian%20Learning%20and%20Receptive%20Fields.ipynb)
- [Self-organisation Answers](09_Hebbian%20Learning/Self-organisation_ANSWERS.ipynb)

## Learning Arc

The projects are intended to be read as a sequence:

1. Build intuition for membrane potentials, equilibrium potentials, and ODE-based simulation.
2. Connect passive and active membrane properties to ion-channel currents and neuron models.
3. Treat connectivity matrices as graphs and reason about network structure.
4. Build simple feed-forward networks and receptive-field-like classifiers.
5. Explore activity propagation through feed-forward and recurrent networks.
6. Introduce Hebbian and related unsupervised learning rules as mechanisms for extracting structure from inputs.

## Running The Notebooks

Create a local virtual environment and install the notebook dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements-notebooks.txt
```

Open the notebooks in Jupyter:

```bash
.venv/bin/python -m ipykernel install --user --name computing-brain
.venv/bin/python -m jupyter lab
```

The notebooks also contain lightweight setup cells for Colab-style environments. When opened in a fresh runtime, those cells install only missing packages such as `brian2`, `ipywidgets`, or `netgraph`.

## Colab Links

The GitHub mirror can be opened directly in Google Colab using this pattern:

```text
https://colab.research.google.com/github/MRIO/TheComputingBrain/blob/main/<notebook-path>
```

For example:

```text
https://colab.research.google.com/github/MRIO/TheComputingBrain/blob/main/07_Simple%20Neural%20Network/Simple%20Neural%20Network.ipynb
```

## Repository Notes

- `requirements-notebooks.txt` lists the Python packages needed for local execution.
- `.venv/`, notebook checkpoints, generated markdown exports, and local Obsidian files are intentionally ignored.
- The current active notebooks were validated to run top-to-bottom after the reproducibility cleanup.
