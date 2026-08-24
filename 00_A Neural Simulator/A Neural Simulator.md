
## In Short:

A **simulator** in computational neuroscience is software that approximates how neural state  variables change over time. Instead of solving equations exactly, a simulator usually advances the system in small time steps, repeatedly computing the next state from the current one.
## Learning goals

After reading this page, you should be able to:

- explain what a simulator does in computational neuroscience
- distinguish analytical from numerical solving in broad terms
- describe why state variables such as membrane potential must be updated over time
- explain the basic intuition behind step-by-step numerical integration
## What is a simulator in computational neuroscience?

A **simulator** is a piece of software that helps us compute how a model changes over time.

In computational neuroscience, simulators are often used to model neurons, synapses, and neural networks. Many of these models are written in the language of **differential equations**: equations that describe how a quantity changes, rather than directly giving its value at all times.

For example, instead of saying:

> “the membrane potential is exactly this at time \( t \),”

a model usually says something more like:

> “given the current membrane potential, input current, and other variables, here is how the membrane potential is changing right now.”

A simulator takes that rule and repeatedly computes what happens **a tiny time step later**, then again, and again, and again.

---

## The big idea

A simulator is a way of turning:

- a model (how states change with time), and
- an initial state,

into a **time-evolving trajectory**.

In computational neuroscience, this often means that the simulator keeps track of **state variables**, such as:

- membrane potential
- synaptic conductance
- adaptation current
- channel activation variables
- intracellular calcium

At each small time step, the simulator uses the current state to estimate the **future state**.

---

## Why differential equations appear

Neurons are dynamical systems: their behavior changes continuously over time.

A very common idea in neuroscience is that the current state of a neuron determines how it will change next. For instance:

- if the membrane potential is far below threshold, it may drift upward or downward slowly
- if synaptic input arrives, the membrane potential may rise
- if ion channels open, currents change
- if a spike occurs, some variables may reset abruptly

Differential equations are a natural way to describe these time-dependent changes.

You do **not** need to solve those equations by hand in order to simulate them. That is one of the main jobs of a simulator.

---

## Analytical solving versus numerical solving

This distinction is important.

### Analytical solving

An **analytical solution** is an exact mathematical expression for the behavior of the system.

For example, for some simple equations, mathematics allows us to write down a formula that directly tells us the value of a variable at any time \( t \).

That is elegant and powerful. But many realistic neuroscience models are too complex to solve this way.

### Numerical solving

A **numerical solution** does not usually produce one neat exact formula.

Instead, it approximates the future step by step:

1. start from the current state
2. estimate the change over a very small time interval
3. update the state
4. repeat many times

This is what simulators usually do.

So, in practice:

- **analytical solving** asks: “Can I solve this equation exactly?”
- **numerical solving** asks: “Can I approximate the system’s behavior accurately enough by taking many small steps?”

In computational neuroscience, numerical solving is the norm.

---

## A simple intuition: following the slope

Suppose you know the current membrane potential \( V \), and your model tells you its rate of change:

$$
\frac{dV}{dt} = f(V, t, \text{input}, \ldots)
$$

This equation says:

> “the change in membrane potential depends on the current state and input.”

A simulator does not magically jump to the full future solution.

Instead, it says:

- “Right now, what is the slope (rate of change)?”
- “If I move forward by a tiny amount of time, where do I end up?”
- “Now that I am there, what is the new slope?”
- “Repeat.”

This is the basic logic behind numerical integration.

---

## The Euler idea

One of the simplest numerical methods is the **Euler method**.

If a variable \( x \) changes according to

$$
\frac{dx}{dt} = f(x, t),
$$

then a very simple update rule is:

$$
x_{t+\Delta t} \approx x_t + \Delta t \, f(x_t, t)
$$

This says:

- take the current value \( x_t \)
- compute its current rate of change
- multiply that rate by a small time step \( \Delta t \)
- add that change to get the next value

This is often the first stepping method students encounter, because it makes the idea of simulation concrete.

You can read more here: [Euler method (Wikipedia)](https://en.wikipedia.org/wiki/Euler_method)

---

## What a simulator is really doing

In practical terms, a simulator usually does something like this:

1. **Store the current state** of all variables
2. **Compute derivatives or updates** from the model equations
3. **Advance time by a small step**
4. **Update the variables**
5. **Repeat until the simulation ends**

So if we simulate a neuron, the simulator may repeatedly compute:

- the new membrane potential
- the opening and closing of channels
- synaptic effects
- spike generation and resets
- changes in plasticity variables

The output is often a time series: for example, membrane potential as a function of time.

---

## Why this matters in neuroscience

Real neurons and networks can be too complicated for exact pen-and-paper solutions.

Simulators let us:

- test mechanistic hypotheses
- compare models with experiments
- explore what a model predicts
- study systems with many interacting variables
- simulate large neural networks efficiently

In that sense, a simulator is not “just software.”  
It is a **computational microscope** for dynamical models.

---

## A useful mental picture

A neuroscience simulator is often best understood as a machine that repeatedly answers the question:

> “Given the current state of the system, what should the state be a tiny moment later?”

That is the core idea.

---

## Examples of computational neuroscience simulators

Here are a few important examples:

### NEURON

**NEURON** is a widely used simulator for modeling individual neurons and networks, especially when morphology and biophysical detail matter.

Reference:
- [NEURON documentation](https://nrn.readthedocs.io/)

### Brian2

**Brian2** is a popular simulator for spiking neural networks, especially appreciated for its flexibility and its Python-based workflow.

Reference:
- [Brian2 documentation](https://brian2.readthedocs.io/)

### EDEN

**EDEN** is a NeuroML-based neural simulator designed to combine flexibility with high performance.

Reference:
- [EDEN paper / record](https://pmc.ncbi.nlm.nih.gov/articles/PMC9167055/)

---

## Final takeaway

A simulator in computational neuroscience is usually a tool that:

- represents a neural model as a set of state variables and update rules
- uses numerical methods to approximate how those variables evolve over time
- produces the future behavior of the model step by step

So when people say that a simulator “runs a model,” they usually mean:

> it numerically computes the time evolution of the model’s equations.

---

## Glossary

**Analytical solution**  
An exact mathematical expression for the solution of an equation.

**Approximation**  
A value that is not exact, but is close enough to be useful.

**Differential equation**  
An equation that describes how a quantity changes over time or with respect to another variable.

**Dynamical system**  
A system whose state changes over time according to rules.

**Euler method**  
A simple numerical method that estimates the future value of a variable using its current value and current rate of change.

**Initial condition**  
The starting value of a variable at the beginning of a simulation.

**Membrane potential**  
The electrical voltage across a neuron’s membrane.

**Numerical method**  
A step-by-step computational procedure used to approximate the solution of a mathematical problem.

**Numerical solution**  
An approximate solution obtained by computation rather than by an exact formula.

**Rate of change**  
How fast a variable is changing at a given moment.

**Simulator**  
Software that computes the behavior of a model over time.

**Slope**
The rate of change of a function at a given moment.

**State**  
The collection of variable values that fully describes the system at a given moment.

**State variable**  
A variable that helps describe the current condition of the system, such as membrane potential or synaptic conductance.

**Time step $\Delta t$**  
The small interval of time used when advancing a simulation.

**Trajectory**  
The evolution of a system’s state over time.

---

## References

1. [Euler method — Wikipedia](https://en.wikipedia.org/wiki/Euler_method)  
2. [NEURON documentation](https://nrn.readthedocs.io/)  
3. [Brian2 documentation](https://brian2.readthedocs.io/)  
4. [Panagiotou et al., EDEN: A High-Performance, General-Purpose, NeuroML-Based Neural Simulator](https://pmc.ncbi.nlm.nih.gov/articles/PMC9167055/)