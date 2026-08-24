# NeuroAI: The Computing Brain Codign Projects

This repository contains notebook-based projects for learning computational neuroscience and neural computation by building working simulations. The projects move from single-cell biophysics to network structure, simple neural networks, receptive fields, and Hebbian learning.

The overarching goal is to connect biological mechanisms to computational ideas. Students start with membrane potentials and ion-channel models, then use those foundations to reason about graph connectivity, activity propagation, manually tuned classifiers, and learning rules.

![Overview of the projects](resources/Overview_of_Projects.png)

## Project Index

In this repo you'll find the coding projects for the NeuroAI courses at the Erasmus University. For the detailed project-by-project learning goals, see the [Project Index and Learning Goals](Project%20Index%20and%20Learning%20Goals.md).

For beginning students the projects are intended to be read as a sequence, but students with some background can pick up any project as entry point. The basic topic sequence for a student that wants to learn basics of computational neuroscience would be:

1. Build intuition for membrane potentials, equilibrium potentials, and ODE-based simulation.
2. Connect passive and active membrane properties to ion-channel currents and neuron models.
3. Treat connectivity matrices as graphs and reason about network structure.
4. Build simple feed-forward networks and receptive-field-like classifiers.
5. Explore activity propagation through feed-forward and recurrent networks.
6. Introduce Hebbian and related unsupervised learning rules as mechanisms for extracting structure from inputs.

[The project index](Project%20Index%20and%20Learning%20Goals.md) collects the explicit learning objectives from the notebooks, including goals such as simulating membrane-potential ODEs in Brian2, assembling Hodgkin-Huxley-style ionic currents, interpreting adjacency matrices as graphs, tuning feed-forward networks, understanding receptive fields, and exploring unsupervised Hebbian learning.



## Running The Notebooks

### In Google Colab, direct links
The projects here can be opened in google colab (a jupyter server in the cloud, provided by google).

The projects live in GitHub and can be opened directly in Google Colab using this pattern:

```text
https://colab.research.google.com/github/MRIO/TheComputingBrain/blob/main/<notebook-path>
```

For example:

```text
https://colab.research.google.com/github/MRIO/TheComputingBrain/blob/main/07_Simple%20Neural%20Network/Simple%20Neural%20Network.ipynb
```

### Locally, in your computer

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


## Repository Notes

- `requirements-notebooks.txt` lists the Python packages needed for local execution.
