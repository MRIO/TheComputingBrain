
This document lists the available notebook-based projects in this collection and collects the explicit learning goals stated in those notebooks.
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
- Differentiate between **active** and **passive** ion channels.
- Explain what is the **driving force**.
- Explain what is a **gating variable**.
- Explain why a gating variable is a sigmoidal function.
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
- Interpret the neuron as a dynamical system
- Know what are state variables (of a dynamical system)
- Know what is a state space
- Know the difference between parameters and variables
- Recognize the state variables of spiking models such as the LIF and the AdEx model.
- Acquire intuition about the equations of the AdEx model 
- Perform simulations with different parameters.
- Visualize the role of different parameters of the AdEx model on the dynamics of spiking.
- Understand why we need to pay attention to units and how to keep neurons unit-consistent (SI units: Ampere, Volts, Farads, Siemens).
- Calculate and display F x I curves and use them to compare different neuronal models types.
- Use F x I curves to distinguish between 'integrators (type 1) and 'resonators' (type 2).

## 04_SpikingNetworks and Criticality

Available notebooks:
- [Spiking Networks](04_SpikingNetworks/[solutions] Spiking Networks.ipynb)

Learning goals:
- Create a neuronal group with a population of IF neurons
- Know the meaning of the parameters of a standard LIF neurons
- Create a randomly connected network of neurons (Erdos-Reiny).
- Know how to add simple excitatory and inhibitory synapses to a spiking network
- Explain what is Dale's rule
- Know how to stimulate a spiking network via random events (a Poisson Input).
- Calculate basic statistics of spiking network activity, such as average firing rate.
- Interpret network activity via raster plots and histograms.
- Manually tune a network with meta parameters to balanced and produce different kinds of network activity.
- Know what is criticality in the context of spiking networks
- Understand what is network heterogeneity

## 05_Networks and Graphs

Available notebooks:
- [Networks and Graphs Tutorial](05_Networks and Graphs/[solutions] Networks and Graphs Tutorial.ipynb)

Learning goals:
- Interpret an adjacency (connectivity) matrix as a graph (network)
- Classify different kinds of connectivity matrices on the basis of type of entries (i.e., weighted, unweighted, directed, undirected, random, self-connections, recurrences)
- Create network via a connectivity matrix according to specifications
  - Feed forward neural networks (convergent and divergent)
  - Ring Networks
  - Star Networks

- Interpret adjacency matrices as a 
- Know how to compute basic graph measures from adjacency matrices (degree, clusterization)
- Create F

## 06_FFNN and Receptive Fields

Available notebooks:
- [Project: DIY Receptive Field (FFNN)](06_FFNN and Receptive Fields/DIY Receptive Field.ipynb)

Learning goals:
- Students can explain a simple "McCulloch-Pitts" binary neuron
- Students understand how the operation of a simple neuron is represented by the dot product between two vectors, and a threshold ("Heaviside") function.
- Students can write the equation of a simple binary neuron
- **Students are able to manually tune weights for a feed forward network to recognize oriented bars.**
- Students learn how represent 2d stimuli with 1d vectors
- Students learn about a 'confusion matrix' to examine the outputs of their manually tuned classifiers.
- Students can provide reasons for the increase of receptive field complexity (and of receptive field size) in multilayer neural networks;
- Students can explain the role of convolutional layers in image recognition
- Students can use an error to train a perceptron to recognize a set of patterns
- Students know how to use backpropagation to train a network of classifiers.
- Students develop intuition about the XOR problem and why feed forward neural networks with only one layer cannot solve it.
- Understanding the role of multiple layers in solving the XOR problem.

## 07_Simple Neural Network

Learning goals:
- Understand activity propagation in discrete networks via vector-matrix multiplication.
- Understand propagation of activity in **feed forward** (FFNN) and **recurrent neural networks** (RNN).
- Acquire intuition about the evolution of the **activity state** for different network configurations.
- Understand the use of **saturating non-linearities** -- so called, 'transfer functions', such as logistic sigmoids (logsig),  the hyperbolic tangent (tanh) and rectifying linear units (ReLu), to represent neuronal ativity.

## 08_Hebbian Learning

Available notebooks:
- [Self-organisation_ANSWERS](09_Hebbian Learning/Self-organisation_ANSWERS.ipynb)

Learning goals:
- Understand the difference between supervised and unsupervised learning

- Know what is Hebbian plasticity

- Understand why standard Hebbian plasticity is unstable

- Be able to explain what are statistical regularities in stimuli

- Learn how to compute an activity dependent plasticity rule

- Understand how Hebbian plasticity rule extracts regularities from inputs

- Understand how to train an unsupervised network by presenting stimuli

- Know how to inspect and interpret network weights before and after learning

- Know how to normalize weights for stable learning rules (Oja, BCM)

- Know about the spiking version of Hebbian Learning via STDP


## 09_RNN and Attractor Networks

Available notebooks:
- [Hopfield-NeckerCube_ANSWERS](10_RNN and Hopfield Network/Hopfield-NeckerCube_ANSWERS.ipynb)

Learning goals:
- Implement a Hopfield recurrent neural network (RNN)
- Train the network with 'one shot' Hebbian learning
- Relate the activity of the network with fixed point attractors of the newtork
- Explain what is an energy function
- Reason about the energy landscape of Hopfield Networks

## 10 Decision Making and the Matching Law

- Reproduce the basic formulation of 'linear matching'.
- Understand how the perceptron models value in a simple decision making task.
- Randomly initialize a set of perceptron weights.
- Generate random payoffs of a multiarm bandit given reward probabilities
- Implement weight changes according to payoff outcomes.
- Display the time course of weight change averages for multiple initial networks.

## 11 Decoding Spikes

- Be able to define encoding into and decoding from networks
- Learn how to implement a LIF network
- Probe and display network inputs and outputs
- Encode an input quantity in a LIF population
- Decode the quantity from these LIF neurons
- Change properties of Networks, LIF neurons and synapses

