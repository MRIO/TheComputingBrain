---
jupyter:
  jupytext:
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.5
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="9SkJDNRemMc8" -->
# This is a (python) notebook.
To be able to use it, go through this:


1. in the file menu (top left), click ```open in playground```
3. still in the file menu, click ```save copy in drive```, to make your own personalized and editable copy of this file.
4. edit as you like. If something breaks irreparably, go back to step 1.

More information about jupyter notebooks and colab is here:

0. Learn about how to use google colaboratory [video](https://www.google.com/search?client=opera&q=introduction+to+google+colab&sourceid=opera&ie=UTF-8&oe=UTF-8#kpvalbx=_gYFDX5-jEcv0kwWY_YGgCg113)
1. Have a look at this [example notebook](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l01c01_introduction_to_colab_and_python.ipynb) that introduces python and colab.
<!-- #endregion -->

<!-- #region id="ImczRANuXV3c" -->
# How to use this tutorial:

You are expected to read this document sequentially, and answer questions you will find in the comments and at the bottom of the document. You will present your answers at our next encounter.

> To edit and to run, go to the menu `File> Save a Copy in Drive` or `File > Open in Playground Mode`

*Tip: open the table of contents, on the icons on the top-right corner menu bar (look like bullet points).*

<!-- #endregion -->

<!-- #region id="Fcf9BzdGjizf" -->
# Tutorial: Representing Eye Position in Populations of Spiking Neurons
<!-- #endregion -->

<!-- #region id="UHTW0oDDjpjw" -->
In this [colab](https://colab.research.google.com/notebooks/intro.ipynb) we will use the simulator [NENGO](www.nengo.ai) to implement a neural population of leaky integrate and fire neurons that represents eye position (a continuous variable). This is an ultra simple example for you to **get a sense of what it means for populations of neurons to encode physical quantities**.

We will **create a population of encoding neurons**, which effectively will be basis functions, and will respond monotonically but non-linearly.

The response of these LIF neurons are derived from response properties found  in the NPH and MVN, which have tuning curves for specific eye positions.

We will then **decode the eye position information** from the population of neurons. We will then verify the quality of the decoding and observe how it changes with the size of the neural population and other biophysical properties of the modeled neurons such as membrane time constant.

<!-- #endregion -->

<!-- #region id="STNe4I_hl2IL" -->
## Learning Goals
<!-- #endregion -->

<!-- #region id="S0MCkNmJl5e2" -->
- Implement a leaky integrate-and-fire network in Nengo.
- Probe and visualize network inputs and outputs.
- Encode an input quantity in a population of LIF neurons.
- Decode the represented quantity from population activity.
- Analyze how neuron, network, and synapse properties affect decoding.

<!-- #endregion -->

<!-- #region id="JgFFQmu9jX1l" -->
For reference code snippets check: https://www.nengo.ai/nengo/examples/advanced/nef-summary.html
<!-- #endregion -->

<!-- #region id="TGJRtXPONr1x" -->
## Requirements
<!-- #endregion -->

<!-- #region id="lWD81kHONtx-" -->
- [Basic knowledge about integrate and fire neurons](https://lcnwww.epfl.ch/gerstner/SPNM/node26.html)
  - membrane time constant
  - response function
- Useful but not essential: [Basic notions of object oriented programming](https://www.youtube.com/watch?v=pTB0EiLXUC8)
<!-- #endregion -->

<!-- #region id="LDmF_cDmhZyV" -->
## Resources

- Nengo uses optimal linear decoding to find weights that reconstruct a signal from neuronal activity. [Here is a video I recorded with the analytical derivation of how to find the optimal decoding weights](https://youtu.be/A8Mc_IsVSTE)
- [A nice set of applets to get a sense of models of spiking](http://jackterwilliger.com/biological-neural-networks-part-i-spiking-neurons/). For the purposes herein, you are interested in the activity of integrate and fire neurons.
- [Wulfram Gerstner explains the dynamics of integrate and fire neurons](https://youtu.be/KGxVwJJC9zs?si=prIKnJAUBg5QNVYi).
- [Learn more about the Nengo simulator here](https://www.nengo.ai/nengo/).


<!-- #endregion -->

<!-- #region id="UVuGLdPrRoKZ" -->
## Glossary
<!-- #endregion -->

<!-- #region id="GKF3u-KkRpzD" -->
- **Input space**: the set of possible values that input can take. For example, the input space of a single photo receptor is the possible luminance values that the photosensor receives.
- **Sensitivity**: the sensitivity of a (typically sensory) neuron measures the modulation of the neuron to some input
<!-- #endregion -->

<!-- #region id="RyB5_LH9mLRD" -->
## Initialization
<!-- #endregion -->

<!-- #region id="7StXGltumPUl" -->
In the cells below we install the nengo simulator and import relevant python packages.
<!-- #endregion -->

```python id="cSTJu3u9Bi83"
# Install Nengo into the colab via pip
!pip install nengo
```

```python id="-729YdSaL15N"
import nengo # our simulator
import pandas as pd # to collect and handle data frames
import matplotlib.pyplot as plt # for plotting
import numpy as np # for numerical / mathematical functions
import seaborn as sns # for nice plotting
```

```python id="9PvuEPNpMnZm"
from nengo.dists import Uniform # a nice uniform distribution
from nengo.processes import WhiteSignal # a noise source
from nengo.utils.ensemble import tuning_curves # tuning curves
# from nengo.utils.ipython import hide_input
from nengo.utils.matplotlib import rasterplot # for a convenient way of displaying raster plots
```

```python id="ijmVPY0phMKd"
# to add the plots directly to the jupyter document:
%matplotlib inline
```

```python id="7qVZv-CShCwy"
from nengo.dists import Uniform # a nice uniform distribution
```

<!-- #region id="L_uyZYPcL0B1" -->
# Modeling Population Encoding in Nengo

A recipe for a model in Nengo is as follows:

1. Create a model with `nengo.Network()` object.
2. Create and insert inputs in the model via `nengo.Node()`.
3. Add a number of neuronal populations via `nengo.Ensembles()`. Here we can also set the properties of our neurons, such as membrane time constant.
4. Connect those ensembles via synapses via `nengo.Connection()`. Here we can also define synaptic properties. It is here as well, that we may want to define the **decoding functions**.
5. Add some **probes** to the model with `nengo.Probe()` to record variables.
6. Add the model to a `nengo.Simulator()` object and run.
7. Plot selected variables using your package of choice (e.g., seaborn, matplotlib).

<!-- #endregion -->

<!-- #region id="d2UeQJ8x4FUm" -->
## 1. Create a model.
<!-- #endregion -->

<!-- #region id="CTzcd1ANQJIm" -->
This is the **object** that we will populate with inputs, neurons and probes, and that we will simulate.
<!-- #endregion -->

```python id="ngEoCu5e4JoI"
model = nengo.Network(label='NPH and VN') # Label is simply a name (string).

# NPH : nucleus prepositus hippoglossi
# VN  : vestibular nucleus

```

<!-- #region id="QKzzCeJU2-q1" -->
## 2. Create an input:

Nengo's function `Node()` is used to create inputs to networks. [Lambda notation](https://realpython.com/python-lambda/) is often used for the purpose.



- for example, a function that returns a ramping input:

$$ input(t) = t*2-1 $$

is written in nengo as:

> `input = nengo.Node(lambda t: t * 2 - 1)`

- a function that returns 1 during a certain interval.

> `input = nengo.Node(lambda t: 1 0] if t < 0.1 else 0)`

- a function that produces a sine wave with a certain frequency

> `input = nengo.Node(lambda t: sin(2*pi*t)`

- combine the two last statements

`input = nengo.Node(lambda t: sin(2*pi*t) if t > 1 else 0)`
<!-- #endregion -->

```python id="wsNnM09a319N"
with model: # to add things to model, we use 'with'
  input = nengo.Node(lambda t: np.sin(2*1*np.pi * t)) # the input

  # you cal also try other functions as input.
  # input = nengo.Node(lambda t: t * 2 - 1)


  input_probe = nengo.Probe(input) # with Probe we record a variable
```

<!-- #region id="vd_0KA-jJ5GF" -->
In order to display our input, we need to run the model (this populates the output variables). We do that by adding our model to `Simulator()` on `model`  and running for `t` seconds.
<!-- #endregion -->

```python id="cF01dZhjKlBZ"
with nengo.Simulator(model) as sim:
    sim.run(1.0)
```

<!-- #region id="cxqhGGn5QlJ2" -->
Our model only has an input so far, no neurons. But we can plot it for sanity checking:
<!-- #endregion -->

```python id="NBAJjSITQxAp"
plt.figure()
plt.plot(sim.trange(), sim.data[input_probe], lw=2) #note 'input_probe' is what we created above. lw is line width.
plt.title("Input signal")
plt.xlabel("Time (s)")
plt.xlim(0, 2)
```

<!-- #region id="1LGYND-omihN" -->
## 3. Create some Neurons that Cover the Input Space
<!-- #endregion -->

<!-- #region id="I_ev_QBamu7K" -->
We will add an ensemble that encodes a continuous variable (representing some physical input to sensory neurons such as pressure, or light intensity). We do this by creating a set of neurons. Each of these neurons has a different tuning curve, such that together the neurons 'cover' the entire 'input space'. These tuning curves will have a mostly-linear relationship with an encoded quantity. In NENGO we refer to neurons with tuning curves as  'encoders'.
<!-- #endregion -->

<!-- #region id="aKioogaWeQYl" -->
---

Firing characteristics of the Nucleus Prepositus Hipoglossi and Medial Vestibular Nucleus

**System Specification:**
>
- Encode for horizontal eye position --
  - Humans: 50deg horizontal eye motion (Davson 1990, p.657).
  - In the model we can transform [-25 to 25] to a range between [-1, 1], without loss of generality!
- Average background firing rates in the NPH: 0-150 Hz (Moschovakis 1997)
- Maximum firing rate ~ 300Hz
- Tuning curve sensitivities (mostly linear!) : 0.1 to 7 Hz / deg





---
<!-- #endregion -->

<!-- #region id="Zw_m7d1y0HDA" -->
### Tuning curves as basis functions
We distribute the encoder neurons over the input space by setting neurons with intercepts along the encoded dimension, in this case from [-1 to 1]. We do this with a helper function `aligned()`, which we define below. We will use the output of this function to parameterize our encoding neurons to cover the base space.
<!-- #endregion -->

<!-- #region id="OE6wWo-MlE_o" -->
Example: Write a fuction called 'aligned', that returns two lists of N values, where:
- **intercepts**: the value at which the neuron starts firing. Equally spaced points in an interval between [-radius, radius], where radius represents the possible values of an input around zero.
- **encoders**: the slope of the tuning curves of the neuron. Here we simply use a vector populated with equal numbers of -1 (off neurons) and 1 (on neurons). The reasons for this choice are explained in Neuroengineering chapter 4).

<!-- #endregion -->

```python id="pwncuJXQlGwm"
def aligned(n_neurons, radius=0.9):
    intercepts = np.linspace(-radius, radius, n_neurons)
    encoders = np.tile([[1], [-1]], (n_neurons // 2, 1))
    intercepts *= encoders[:, 0]
    return intercepts, encoders
```

<!-- #region id="rsMxGZY1VaiX" -->
#### Warm-up Exercise:
**Call the function** defined above with 8 neurons and a radius of 1 and **inspect the output**.
<!-- #endregion -->

```python id="2_WrBywd0g5t"
# your code here
```

<!-- #region id="fzq0Bjy8jR5v" -->
### Check your answer below
<!-- #endregion -->

```python id="6VWJWJaBjF-I"
intercepts, encoders = aligned(8, 1)
print(intercepts)
```

<!-- #region id="K_StMume2vBs" -->
### Make an IF population of encoders
<!-- #endregion -->

<!-- #region id="W9agO0IeW2Zr" -->
Below we create an ensemble of 8 IF neurons to encode the 1-dimensional input (e.g., the eye position) according to specified intercepts, encoders and with specified max_rates.
<!-- #endregion -->

```python id="gxmeVn9tMZT0"
with model:
    NPH = nengo.Ensemble( #NPH stands for 'nucleus prepositus hipoglossi'
        8, # number of neurons in the ensemble
        dimensions=1, # encoded stimulus dimensions ("capacity")
        intercepts=intercepts,
        max_rates=Uniform(80, 100), # the maximum firing rate of the neurons are drawn from the uniform distribution
        encoders=encoders,
    )
```

<!-- #region id="gksKPQ7tZEX1" -->
We can plot the tuning curves of our neurons, with the function `tuning_curves` for population (In this case NPH).
<!-- #endregion -->

```python id="scUex73aNE7D"
with nengo.Simulator(model) as sim:
    eval_points, activities = tuning_curves(NPH, sim)

plt.figure()
plt.plot(eval_points, activities, lw=2)
plt.xlabel("Input signal")
plt.ylabel("Firing rate (Hz)")
```

```python id="k3yg4a1N2eLW"
help(tuning_curves)
```

<!-- #region id="8idrCw6DalS-" -->
### 5. Connect the Inputs with Neurons
<!-- #endregion -->

<!-- #region id="7NJ_h45GbRC5" -->
We distribute the input to all neurons in the ensemble via `nengo.Connection()`. By default we connect the input to all postsynaptic neurons with equal weights*. Let's also create a probe for the spikes the model is producing.
<!-- #endregion -->

```python id="IKceg-3hauyb"
with model:
    nengo.Connection(input, NPH)
    NPH_spikes = nengo.Probe(NPH.neurons)
```

<!-- #region id="pXqcZzvocZvh" -->
### 5. Run the Network
<!-- #endregion -->

```python id="yRP6GoRqdRdf"
with nengo.Simulator(model) as sim:
    sim.run(1)
```

<!-- #region id="RlrBEXVaZpHb" -->
## 6. How does the activity of the neurons look like??
<!-- #endregion -->

<!-- #region id="H_Ai_KnUZxud" -->
Now that we have the encoding neurons, the input and the probes, we can run the simulation and check the firing behavior of the neurons.
<!-- #endregion -->

<!-- #region id="xK7BemRYfhgA" -->
#### Raster Plot:
<!-- #endregion -->

```python id="ZRG_uOr6aVxw"
plt.figure(figsize=[15,4])
ax = plt.subplot(1, 1, 1)
rasterplot(sim.trange(), sim.data[NPH_spikes], ax)
ax.set_xlim(0, 1)
ax.set_ylabel("Neuron")
ax.set_xlabel("Time (s)")
```

<!-- #region id="U95aObpffjvv" -->
#### Membrane Potential
<!-- #endregion -->

```python id="QhtB9wvKfmYs"
with model:
  nengo.Connection(input, NPH)
  NPH_spikes = nengo.Probe(NPH.neurons, synapse=0.05)

with nengo.Simulator(model) as sim:
    sim.run(1)

scale = 180
plt.figure()
for i in range(NPH.n_neurons):
    plt.plot(sim.trange(), sim.data[NPH_spikes][:, i] - i * scale)
plt.xlim(0, 1)
plt.ylim(scale * (-NPH.n_neurons + 1), scale)
plt.ylabel("Neuron");
plt.yticks(
    np.arange(scale / 1.8, (-NPH.n_neurons + 1) * scale, -scale), np.arange(NPH.n_neurons)
);
```

<!-- #region id="HFu_x_KNd6qt" -->
## 7. Check your decoding skills!

<!-- #endregion -->

<!-- #region id="MxrXbJWWeDYj" -->
Can we decode those spikes and obtain our signal back? A decoder that produces the identity function can be obtained via a simple Probe. To find the decoding weights, the probe of a population uses the encoded value.
<!-- #endregion -->

```python id="h8M9e3lhRniD"
with model:
    NPH_probe = nengo.Probe(NPH, synapse=0.2)
    # 5ms PSC filter (AMPA like)
    # 10ms PSC filter (GABA like)
    # 100ms PSC filter (NMDA like)

simtime = 5 #seconds

with nengo.Simulator(model) as sim:
    sim.run(simtime)

plt.figure()
plt.plot(sim.trange(), sim.data[NPH_probe], label="Decoded estimate")
plt.plot(sim.trange(), sim.data[input_probe], label="Input signal")
plt.legend(loc="best")
plt.xlim(0, simtime)
```

<!-- #region id="UH3bpl0gm2zk" -->
# Questions and Exercises
<!-- #endregion -->

```python id="d7d9ewb5IkS2"

```

<!-- #region id="r5okN1gcm4wE" -->
**1.(Question)** What significant assumption have we made about the distribution of tuning curves (hint: what is particular about our chosen intercepts / encoders)?

**2. (Exercise).** Our network is rather small. Increase the number of neurons and observe the impact on decoding quality.

**3. (Check).** Verify that the parameters of our network are specified according to the experimental data. Particularly, make sure that the values for individual neuronal tuning curves in the code above are specified as in the system parameters (section 3).

**4. (Experiment)** Is the decoding quality a function of the input oscillation? In other words, does the network encode all frequencies equally well? Change the frequency of the sinusoidal input function and inspect the decoding quality. Are all possible input frequencies decoded equally well? Is there a dependency between quality of decoding and synapse time constant (determined in line 2 in the code block above).

**5. (Challenge).** Create an 'efference copy' from the encoding population. That is: produce another population downstream from the encoding population that conveys the same 'quantity' (same number of dimensions). Now decode the original value from the second population. Try to spot the main differences in the activity of the two populations. Note: this exercise needs you to code a new Model, with an extra 'ensemble'.

<!-- #endregion -->

<!-- #region id="y6ZjeZZb1ACV" -->
# Bonus: Influence of Membrane Time Constant on Signal Reconstruction

Code gracefully scraped from: https://github.com/ctn-waterloo/modelling_ideas/issues/91
<!-- #endregion -->

```python id="5J81DQPa1JaD"
def go(freq, tau_rc, n_neurons=10, tau_probe=0.005, t=1.0, dt=0.001, seed=0):
    with nengo.Network() as model:
        u = nengo.Node(output=lambda t: np.sin(2*np.pi*freq*t))
        x = nengo.Ensemble(n_neurons, 1, seed=seed,
                           neuron_type=nengo.LIF(tau_rc=tau_rc))
        nengo.Connection(u, x, synapse=None)

        p_u = nengo.Probe(u, synapse=tau_probe)
        p_x = nengo.Probe(x, synapse=tau_probe)

    with nengo.Simulator(model, dt=dt, progress_bar=False) as sim:
        sim.run(t, progress_bar=False)

    return nengo.utils.numpy.rmse(sim.data[p_u], sim.data[p_x])

data = []
for seed in range(5):
    for freq in np.linspace(0, 50, 20):
        for tau_rc in [0.001, 0.005, 0.01, 0.02, 0.1]:
            print(freq, tau_rc)
            data.append((freq, tau_rc, seed, go(freq, tau_rc, seed=seed)))
df = pd.DataFrame(data, columns=("Frequency", "tau_rc", "Seed", "RMSE"))
```

```python id="ktc1f6Xm1bQM"
plt.figure(figsize=[20,20])
for tau_rc in df.tau_rc.unique():
    sns.regplot(data=df[df['tau_rc'] == tau_rc], x_jitter=1.5,
                x="Frequency", y="RMSE", label=str(tau_rc))
plt.legend()
plt.show()
```

<!-- #region id="vpK0cHMI4FcO" -->
# Example: Encoding a two dimensional vector
<!-- #endregion -->

```python id="cmEzn3LN3_Oz"
# Setup the envirnment
import numpy as np
import nengo
from nengo.dists import Uniform

model = nengo.Network(label='2D Representation')
with model:
    #Two represent possible two dimensional input values, we choose sin and cos
    sin = nengo.Node(output=np.sin)
    cos = nengo.Node(output=np.cos)

    # Create here an ensemble with 100 LIF neurons which represents a 2-dimensional signal
    x = nengo.Ensemble(100, dimensions=2, max_rates=Uniform(100, 200))

    #Get the neuron encoders
    encoders = x.encoders.sample(100,2)

    # Connecnting input to ensemble
    # The indices in ensemble 'x' define which dimension the input will project to
    nengo.Connection(sin, x[0])
    nengo.Connection(cos, x[1])


#place a probe to record a selected variable
# Q: what changes with different synaptic time constants?
with model:
    probe1 = nengo.Probe(x.neurons, synapse=0.01)

simtime = 5 #seconds
# run the simulator for five seconds
with nengo.Simulator(model) as sim:
    sim.run(simtime)


# plot the decoded 2D variables
plt.figure()
plt.plot(sim.trange(), sim.data[probe1], label="Decoded estimate")
plt.legend(loc="best")
plt.xlim(0, simtime)

# challenge: can you plot the input signal as well?
```
