# Project Index and Learning Goals

This document lists the available notebook-based projects in this collection and collects the explicit learning goals stated in those notebooks.![Overview_of_Projects](/Users/m/Sync/Code/TheComputingBrainProjects/resources/Overview_of_Projects.png)

## Table of Contents

- [01_The Resting Membrane Potential_](#01-the-resting-membrane-potential)
- [02_Biophysics and HH (two parts)](#02-biophysics-and-hh-two-parts)
- [03_Neurodynamics](#03-neurodynamics)
- [04_SpikingNetworks](#04-spikingnetworks)
- [05_Networks and Graphs](#05-networks-and-graphs)
- [06_FFNN and Receptive Fields](#06-ffnn-and-receptive-fields)
- [07_Simple Neural Network](#07-simple-neural-network)
- [09_Hebbian Learning](#09-hebbian-learning)
- [10_RNN and Hopfield Network](#10-rnn-and-hopfield-network)



## 01_The Resting Membrane Potential_

Available notebooks:
- [Simulation of Membrane Potential](01_The Resting Membrane Potential_/[Solutions] Simulation of Membrane Potential.ipynb)

Learning goals:
- Know how to use a **solver** to calculate the solutions of  differential equations (ODE's)
- Know how to simulate a membrane potential model (an ODE) in Brian 2
- Understand the role of a time constant in an differential equation
- Control the behaviour of a model via a voltage step

## 02_Biophysics and HH (two parts)

Available notebooks:
- [DIY Neuron Model - Part 2](02_Biophysics and HH (two parts)/[Solutions] DIY Neuron Model - Part 2.ipynb)
- [DIY Neuron Model - Part 1 - Interactive](02_Biophysics and HH (two parts)/[solutions] DIY Neuron Model - Part 1 - Interactive.ipynb)
- [DYI Neuron Model - Part 1](02_Biophysics and HH (two parts)/[solutions] DYI Neuron Model - Part 1.ipynb)

Learning goals:
- Explain how **passive ion channels** and membrane properties lead to the **membrane time constant**.
- Explain what is the **driving force**.
- Differentiate between **active** and **passive** ion channels.
- Explain what is a **gating variable**.
- Compute the **steady state** of **voltage gating variables** in the voltage clamp.
- Explain the **current flows** across the cell as a function of **maximal conductances** and state of the gating variables.
- Explain **conductance** in your own words and why it is different for different ion types.
- Compute the **currents** entering the cell for different ion channels.
- **Assemble** the different ionic currents in the HH model to produce action potentials.
- Know the physical **units** of conductances (Siemens), current (Ampere) and membrane potential (Volts) relate to each other.

Notes:
- `DIY Neuron Model - Part 2` does not contain an explicit learning goals/objectives section.

## 03_Neurodynamics

Available notebooks:
- [Simplified neuron models (AdEx)](03_Neurodynamics/[solutions] Simplified neuron models (AdEx).ipynb)
- [Simplified neuron models(v.2)](03_Neurodynamics/[solutions] Simplified neuron models(v.2).ipynb)

Learning goals:
- Recognize the state variables of the AdEx model.
- Implement an AdEx model in Brian and perform simulations with different parameters.
- Visualize the role of different parameters of the AdEx model on the dynamics of spiking.
- Keep your neurons unit-consistent (SI units: Ampere, Volts, Farads, Siemens).
- Calculate and display F x I curves and use them to compare different neuronal models.
- Use F x I curves to distinguish between 'integrators (type 1) and 'resonators' (type 2).

## 04_SpikingNetworks

Available notebooks:
- [Spiking Networks](04_SpikingNetworks/[solutions] Spiking Networks.ipynb)

Learning goals:
- create a neuronal group with a population of IF neurons.
- add simple excitatory and inhibitory synapses to that group.
- stimulate the network via random events (a Poisson Input).
- create a randomly connected network.
- calculate basic statistics of network activity, such as average firing rate.
- display network activity via raster plots and histograms.
- manually tune a network to be balanced and produce different kinds of network activity.

## 05_Networks and Graphs

Available notebooks:
- [Networks and Graphs Tutorial](05_Networks and Graphs/[solutions] Networks and Graphs Tutorial.ipynb)

Learning goals:
- Interpret an adjacency (connectivity) matrix as a graph (network).
- Classify different kinds of connectivity matrices on the basis of type of entries (i.e., weighted, unweighted, directed, undirected, random, self-connections, recurrences)
- Produce a network via a connectivity matrix according to specifications of their properties.
- **Programming Bonuses:**
- Learn to create a figure in python with multiple axes (subplots) and specified size
- Add plots to an axes
- Display a matrix as an image via `pcolor` or `imshow`
- Set a colormap to the plot
- Add `text` to an axis
- Display adjacency matrices as networks via the package `networkx`

## 06_FFNN and Receptive Fields

Available notebooks:
- [Project: DIY Receptive Field (FFNN)](06_FFNN and Receptive Fields/DIY Receptive Field.ipynb)

Learning goals:
- Students can code a simple "McCulloch-Pitts" binary neuron in python.
- Students understand how the operation of a simple neuron is represented by the dot product and a threshold ("Heaviside") function.
- Students are able to manually tune weights for a feed forward network to recognize oriented bars.
- Students develop intuition about the XOR problem and why feed forward neural networks with only one layer cannot solve it.
- Students learn how  to use python dictionaries to 'contain' stimuli.
- Students learn about a 'confusion matrix' to examine the outputs of their manually tuned classifiers.
- Students can provide reasons for the increase of receptive field complexity (and of receptive field size) in multilayer neural networks;
- Students can explain the role of convolutional layers in image recognition.
- Students can use an error to train a perceptron to recognize a set of patterns.
- Students know how to use backpropagation to train a network of classifiers.
- Understanding the role of multiple layers in solving the XOR problem.

## 07_Simple Neural Network

Available notebooks:
- [Learning Goals](07_Simple Neural Network/Simple Neural Network.ipynb)

Learning goals:
- Understand activity propagation in discrete networks via vector-matrix multiplication.
- Understand propagation of activity in **feed forward** (FFNN) and **recurrent neural networks** (RNN).
- Acquire intuition about the evolution of the **activity state** for different network configurations.
- Understand the use of **saturating non-linearities** -- so called, 'transfer functions', such as logistic sigmoids (logsig),  the hyperbolic tangent (tanh) and rectifying linear units (ReLu), to represent neuronal ativity.

## 09_Hebbian Learning

Available notebooks:
- [Elias Version of Self-organisation_ANSWERS](09_Hebbian Learning/Elias Version of Self-organisation_ANSWERS.ipynb)
- [Self-organisation_ANSWERS](09_Hebbian Learning/Self-organisation_ANSWERS.ipynb)

Learning goals:
- How to compute an activity dependent plasticity rule
- How Hebbian plasticity rule extracts regularities from inputs
- How to systematically create input patterns
- How to train a network by presenting stimuli
- How to inspect and interpret network weights
- How to balance the weight change.

## 10_RNN and Hopfield Network

Available notebooks:
- [Hopfield-NeckerCube_ANSWERS](10_RNN and Hopfield Network/Hopfield-NeckerCube_ANSWERS.ipynb)

Learning goals:
- Implement a Hopfield recurrent neural network (RNN)
- Train the network with 'one shot' Hebbian learning
- Relate the activity of the network with fixed point attractors of the newtork
- Explain what is an energy function
- Reason about the energy landscape of Hopfield Networks
