
This document lists the available notebook-based projects and collects the explicit learning goals stated in those notebooks.
## Table of Contents

### Basics
- [00 A Neural Simulator](#00_a-neural-simulator)
- [01_The Resting Membrane Potential](#01-the-resting-membrane-potential)
- [02_Biophysics and HH (two parts)](#02-biophysics-and-hh-two-parts)
- [03_Neurodynamics](#03-neurodynamics)
- [04_SpikingNetworks](#04-spikingnetworks)
- [05_Networks and Graphs](#05-networks-and-graphs)
- [06_FFNN and Receptive Fields](#06-ffnn-and-receptive-fields)
- [07_Simple Neural Network](#07-simple-neural-network)
- [09_Hebbian Learning](#09-hebbian-learning)
- [10_RNN and Hopfield Network](#10-rnn-and-hopfield-network)
- [11_Decoding Spikes](#11-decoding-spikes)
- [12_Decision Making and Reward Learning](#12-decision-making-and-reward-learning)
- [13_Basal Ganglia (Gurney Model)](#13-basal-ganglia-gurney-model)
- [14_Echo State Network (Reservoir Network)](#14-echo-state-network-reservoir-network)


## 00_A Neural Simulator
Available notebooks:
- [A Neural Simulator](https://colab.research.google.com/github/MRIO/TheComputingBrain/blob/main/00_A%20Neural%20Simulator/A%20Neural%20Simulator.ipynb)
- [A Quick Tutorial of the Brian2 Simulator](https://colab.research.google.com/github/MRIO/TheComputingBrain/blob/main/00_A%20Neural%20Simulator/A%20Quick%20Tutorial%20of%20the%20Brian2%20Simulator.ipynb)

*Learning goals:*

- Explain the role of simulation in computational neuroscience.
- Distinguish analytical solutions from numerical approximations.
- Describe why state variables, such as membrane potential, must be updated over time.
- Explain how numerical integration advances a dynamical system through discrete time steps.


## 01_The Resting Membrane Potential_

Available notebooks:
- [Project Zero: Resting Membrane Potential](https://colab.research.google.com/github/MRIO/TheComputingBrain/blob/main/01_The%20Resting%20Membrane%20Potential_/Project_Zero_The_Membrane_potential.ipynb)
- [Simulation of Membrane Potential](https://colab.research.google.com/github/MRIO/TheComputingBrain/blob/main/01_The%20Resting%20Membrane%20Potential_/Simulation%20of%20Membrane%20Potential.ipynb)

*Learning goals:*

- Use a numerical solver to approximate solutions to ordinary differential equations (ODEs).
- Simulate a membrane-potential model in Brian2.
- Explain how a time constant shapes the dynamics of a differential equation.
- Analyze a model's response to a voltage-step input.

## 02_Biophysics and HH (two parts)

Available notebooks:
- [DIY Neuron Model - Part 1](https://colab.research.google.com/github/MRIO/TheComputingBrain/blob/main/02_Biophysics%20and%20HH%20%28two%20parts%29/DIY%20Neuron%20Model%20-%20Part%201.ipynb)
- [DIY Neuron Model - Part 2](https://colab.research.google.com/github/MRIO/TheComputingBrain/blob/main/02_Biophysics%20and%20HH%20%28two%20parts%29/DIY%20Neuron%20Model%20-%20Part%202.ipynb)

*Learning goals:*

- Explain how passive ion channels and membrane properties determine the membrane time constant.
- Distinguish active ion channels from passive ion channels.
- Explain the driving force acting on an ion.
- Define a gating variable and explain its sigmoidal voltage dependence.
- Compute the steady states of voltage-dependent gating variables under voltage clamp.
- Explain how maximal conductances and gating variables determine ionic currents.
- Compute the currents carried by different ion channels.
- Assemble ionic currents into a Hodgkin–Huxley model that produces action potentials.
- Relate conductance, current, and membrane potential using their physical units.

## 03_Neurodynamics

Available notebooks:
- [Simplified neuron models (AdEx)](https://colab.research.google.com/github/MRIO/TheComputingBrain/blob/main/03_Neurodynamics/Simplified%20neuron%20models%20%28AdEx%29.ipynb)

*Learning goals:*

- Interpret a neuron as a dynamical system.
- Identify the state variables of LIF and AdEx neuron models.
- Describe a state space and distinguish variables from parameters.
- Explain the terms in the AdEx model equations.
- Simulate AdEx neurons across different parameter settings.
- Analyze how AdEx parameters shape spiking dynamics.
- Maintain consistent SI units for current, voltage, capacitance, and conductance.
- Calculate and display firing-rate–current (f–I) curves to compare neuron models.
- Use f–I curves to distinguish type I integrators from type II resonators.

## 04_SpikingNetworks and Criticality

Available notebooks:
- [Spiking Networks](https://colab.research.google.com/github/MRIO/TheComputingBrain/blob/main/04_SpikingNetworks/Spiking%20Networks.ipynb)

*Learning goals:*

- Construct a population of leaky integrate-and-fire (LIF) neurons.
- Explain the parameters of a standard LIF neuron model.
- Build a randomly connected Erdős–Rényi network.
- Add excitatory and inhibitory synapses that conform to Dale's principle.
- Drive a spiking network with Poisson input.
- Calculate network-activity statistics, including mean firing rate.
- Interpret spiking activity using raster plots and histograms.
- Tune network-level parameters to produce balanced and distinct activity regimes.
- Explain criticality and heterogeneity in spiking networks.

## 05_Networks and Graphs

Available notebooks:
- [Networks and Graphs Tutorial](https://colab.research.google.com/github/MRIO/TheComputingBrain/blob/main/05_Networks%20and%20Graphs/Networks%20and%20Graphs%20Tutorial.ipynb)

*Learning goals:*

- Interpret an adjacency matrix as a graph.
- Classify networks as weighted or unweighted, directed or undirected, recurrent or feedforward, and with or without self-connections.
- Construct adjacency matrices for convergent, divergent, ring, star, and random networks.
- Calculate graph measures, including degree and clustering coefficient, from adjacency matrices.
- Visualize adjacency matrices with Matplotlib.
- Visualize networks with NetworkX.

## 06_FFNN and Receptive Fields

Available notebooks:
- [Project: DIY Receptive Field (FFNN)](https://colab.research.google.com/github/MRIO/TheComputingBrain/blob/main/06_FFNN%20and%20Receptive%20Fields/DIY%20Receptive%20Field.ipynb)

*Learning goals:*

- Implement a McCulloch–Pitts binary neuron in Python.
- Explain how a dot product and Heaviside threshold determine a binary neuron's output.
- Represent two-dimensional stimuli as one-dimensional vectors and Python data structures.
- Tune feedforward-network weights to recognize oriented bars.
- Evaluate classifier outputs using a confusion matrix.
- Explain why receptive-field size and complexity increase across network layers.
- Explain the role of convolutional layers in image recognition.
- Train a perceptron from classification errors and describe how backpropagation extends this process.
- Explain why a single-layer network cannot solve XOR and how additional layers address the problem.

## 07_Simple Neural Network

Available notebooks:
- [Simple Neural Network](https://colab.research.google.com/github/MRIO/TheComputingBrain/blob/main/07_Simple%20Neural%20Network/Simple%20Neural%20Network.ipynb)

*Learning goals:*

- Compute activity propagation in discrete neural networks using vector–matrix multiplication.
- Compare activity propagation in feedforward and recurrent neural networks.
- Analyze how network configuration shapes the evolution of activity states.
- Explain how transfer functions, including logistic sigmoid, hyperbolic tangent, and rectified linear unit (ReLU), transform neuronal activity.

## 09_Hebbian Learning

Available notebooks:
- [Self-organisation](https://colab.research.google.com/github/MRIO/TheComputingBrain/blob/main/09_Hebbian%20Learning/Self-organisation.ipynb)
- [Hebbian Learning and Receptive Fields](https://colab.research.google.com/github/MRIO/TheComputingBrain/blob/main/09_Hebbian%20Learning/Hebbian%20Learning%20and%20Receptive%20Fields.ipynb)

*Learning goals:*

- Distinguish supervised learning from unsupervised learning.
- Explain Hebbian plasticity and why its basic form is unstable.
- Identify statistical regularities in stimulus ensembles.
- Compute activity-dependent synaptic weight updates.
- Explain how Hebbian learning extracts regularities from inputs.
- Construct stimulus patterns for unsupervised training.
- Train an unsupervised network by repeatedly presenting stimuli.
- Compare and interpret network weights before and after learning.
- Normalize weights using stable learning rules such as Oja's rule and BCM theory.
- Relate Hebbian learning in spiking networks to spike-timing-dependent plasticity (STDP).


## 10_RNN and Attractor Networks

Available notebooks:
- [Hopfield-NeckerCube](https://colab.research.google.com/github/MRIO/TheComputingBrain/blob/main/10_RNN%20and%20Hopfield%20Network/Hopfield-NeckerCube.ipynb)

*Learning goals:*

- Implement a Hopfield recurrent neural network.
- Store patterns using one-shot Hebbian learning.
- Relate network activity to fixed-point attractors.
- Explain the role of an energy function in Hopfield-network dynamics.
- Analyze pattern retrieval using the network's energy landscape.

## 11_Decoding Spikes

Available notebooks:
- [Decoding Horizontal Eye Position](https://colab.research.google.com/github/MRIO/TheComputingBrain/blob/main/11_Decoding%20Spikes/Decoding%20Horizontal%20Eye%20Position.ipynb)

*Learning goals:*

- Implement a leaky integrate-and-fire network in Nengo.
- Probe and visualize network inputs and outputs.
- Encode an input quantity in a population of LIF neurons.
- Decode the represented quantity from population activity.
- Analyze how neuron, network, and synapse properties affect decoding.

## 12_Decision Making and Reward Learning

Available notebooks:
- [Matching Law and the Perceptron](https://colab.research.google.com/github/MRIO/TheComputingBrain/blob/main/12_Decision%20Making%20and%20Reward%20Learning/%5BMNEU%5D%20Matching%20Law%20and%20the%20Perceptron.ipynb)

*Learning goals:*

- Reproduce the basic formulation of action and reward matching.
- Explain how a perceptron represents value in a decision-making task.
- Initialize perceptron weights randomly.
- Generate stochastic payoffs for a "multi-armed bandit" from specified reward probabilities.
- Update weights according to payoff outcomes.
- Visualize average weight trajectories across multiple initialized networks.

## 13_Basal Ganglia (Gurney Model)

Available notebooks:
- [Action Selection in the Basal Ganglia](https://colab.research.google.com/github/MRIO/TheComputingBrain/blob/main/13_Basal%20Ganglia%20%28Gurney%20Model%29/Action%20Selection%20in%20the%20Basal%20Ganglia.ipynb)

*Learning goals:*

- Identify the principal anatomical connections of the basal ganglia.
- Explain how spiking neurons implement competition between direct and indirect pathways.
- Represent abstract concepts in spiking-neuron populations using semantic pointer architecture.
- Relate action selection to action sequencing.

## 14_Echo State Network (Reservoir Network)

Available notebooks:
- [A Minimalistic Echo State Network Demo](https://colab.research.google.com/github/MRIO/TheComputingBrain/blob/main/14_Echo%20State%20Network%20%28Reservoir%20Network%29/ESN.ipynb)

*Learning goals:*

- Describe the reservoir and readout components of an echo state network.
- Generate input and target data for training an echo state network.
- Construct a recurrent reservoir with a specified leaking rate and spectral radius.
- Train output weights from collected reservoir states using ridge regression.
- Evaluate autonomous signal generation using prediction error.
