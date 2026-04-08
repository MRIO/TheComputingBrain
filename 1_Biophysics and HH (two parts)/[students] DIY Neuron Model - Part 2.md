---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="B287VpGfYWG9" -->
> # To be able to edit and use this Notebook:
> 0. Learn about how to use google colaboratory [video]()
> 1. in the file menu (top left), click ```open in playground```
> 3. still in the file menu, click ```save copy in drive```, to make your own > personalized and editable copy of this file.
> 4. edit as you like. If something breaks irreparably, go back to step 1.
<!-- #endregion -->

<!-- #region id="43TskcthQiMg" -->
# Project 1 : A DIY Neuron Model -  Part 2 


<!-- #endregion -->

<!-- #region id="JWxy67mxS8iW" -->
### Previously:
In the first part of the  project we synthetized the **Hodgkin Huxley model of action potential**, piece by piece, by adding individual currents. Now it is time to get empirical and experiment with parameters that represent experimental conditions, to gain intuition about what causes spiking.

In other words, we will be running some **protocols** for in-silico **experiments**, and reasoning about what we observe.

<!-- #endregion -->

<!-- #region id="Vz3qG1oghAmR" -->
# Initialization Code
<!-- #endregion -->

<!-- #region id="hh25znGpgZNN" -->
- In the code cell below we install the simulator [Brian2](https://brian2.readthedocs.io/) and import relevant python.modules.
- It installs and imports Brian2, the simulator we will be using.
- **Note that you have to run this every time that colab 'disconnects' from the kernel.**
- Documentation of brian2 can be found [here](https://brian2.readthedocs.io/en/stable/user/index.html)
<!-- #endregion -->

## Run This First

Run the next code cell before the rest of the notebook.

- In Google Colab it installs any missing notebook-only packages and enables widget support.
- In local JupyterLab it only verifies imports against your active environment.
- Local setup: create a virtual environment and install the packages in `requirements-notebooks.txt`.


```python tags=["notebook-runtime-setup"]
# Notebook runtime setup for Google Colab and local JupyterLab.
import importlib
import subprocess
import sys

try:
    from google.colab import output as colab_output
    IS_COLAB = True
except ImportError:
    colab_output = None
    IS_COLAB = False


def ensure_notebook_packages(requirements):
    if not IS_COLAB:
        return

    missing = []
    for package_name, module_name in requirements:
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)

    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


NOTEBOOK_REQUIREMENTS = [('ipywidgets', 'ipywidgets'), ('brian2', 'brian2'), ('matplotlib', 'matplotlib')]
ensure_notebook_packages(NOTEBOOK_REQUIREMENTS)

if IS_COLAB:
    colab_output.enable_custom_widget_manager()

```

```python id="88Qmb5nEPYq6" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1631033759325, "user_tz": -120, "elapsed": 15930, "user": {"displayName": "Elias Mateo Fernandez Santoro", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiN3naitVynkmvN89-tczt7o_TwYkXrydU6xGHOKQ=s64", "userId": "07472471926015090759"}} outputId="49324755-848f-40f3-e8b9-8e32b245bf19"
# install brian2 

# import necesary packages
from IPython.display import display
from brian2 import * # our simulator of choice
import brian2.numpy_ as np # the numpy that comes bundled with it
from ipywidgets import interact, interactive # for some neat interactions
from IPython.display import display 
import ipywidgets as widgets
import matplotlib.pyplot as plt # for neat plots
import time # for time basis conversions

```

<!-- #region id="tecRFUPUUQQL" -->
# The Anatomy of a Spike
<!-- #endregion -->

<!-- #region id="DU75qwXIE8mh" -->
## Coding Exercise: Reproduce Izhikevich's Nice Figure
<!-- #endregion -->

<!-- #region id="omInLJiJUY2-" -->
To warm up we will want to reproduce the figure below from Izhikevich (2007).

First study the entire figure, panel by panel. Do you know what is plotted?

Then note: there are two current injections at 2 and at 10ms. We observe a spike upon the second injection and not the first. We would like to plot all of the relevant variables to get a qualitative understanding of the interaction between the different conductances, currents and activation gates.

- you should guess on the parameters for the current injection based on the graphs below.
- parameters and equations are copied from the previous project for your 
convenience, but you should create your Brian model.
- to record the variables use Brian's state monitor.
- use plt.subplot just like in matlab to produce stacked subplots.
- for axis labels, simply use the physical units.
- in the plot with multiple traces use colors and a legend.

[![HH Action Potential](https://i.postimg.cc/kgzyfgzN/image.png)](https://postimg.cc/7517Lwxb)

> Nota Bene: In the graph below the $V_{rest}=0mV$. This is because in the original version of the model, Hodgkin and Huxley decided to shift the potential by +65mV "to make things easier". Over the years this led to much confusion! We want to reproduce the same figure as below, but we want the 'correct' resting membrane potential of -65mV.

<!-- #endregion -->

<!-- #region id="9aHhVTsNaThe" -->
#### Define Parameters and Equations
<!-- #endregion -->

```python id="a1QijtWiaKrZ" executionInfo={"status": "ok", "timestamp": 1631033759326, "user_tz": -120, "elapsed": 15, "user": {"displayName": "Elias Mateo Fernandez Santoro", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiN3naitVynkmvN89-tczt7o_TwYkXrydU6xGHOKQ=s64", "userId": "07472471926015090759"}}
### The Hodgkin Huxley Equations and Parameters

start_scope()

##################### Parameters:

# Reversal Potentials
E_leak = -54.4 * mV
E_Na   =  55.   * mV
E_K    = -77.   * mV

# Conductances 
## attention to UNITS! For instance uS << mS
g_leak =   300. * uS / cm ** 2  
gbar_Na   = 120. * mS / cm ** 2 # ('bar' means maximal conductance)
gbar_K    =  36. * mS / cm ** 2

# Membrane Capacitance
Cm = 1. * uF / cm ** 2

##################### Equations:
eqs_V ='''
dv/dt = (I -I_leak - I_Na - I_K )/Cm : volt
'''

# note: below we broke down the equations in conductance and current equations
# we need this to have access to the conductances via Brian's state monitor function

eqs_cond = '''
g_Na = gbar_Na*(m**3)*h           : siemens / meter ** 2
g_K  = gbar_K*(n**4)              : siemens / meter ** 2
'''

eqs_I = '''
I_leak = g_leak * (v - E_leak)   : amp / meter ** 2
I_Na =   g_Na   * (v - E_Na)     : amp / meter ** 2
I_K =    g_K    * (v - E_K)      : amp / meter ** 2
I                                : amp / meter ** 2
'''

# here you add the equations defining the potassium activation gates
eqs_activation= '''
n_inf =  1/(1+exp((-53*mV -v)/(15*mV))) : 1
m_inf =  1/(1+exp((-40*mV -v)/(9*mV)))  : 1 
h_inf =  1/(1+exp((-62*mV -v)/(-7*mV))) : 1
taun  =  1.1*ms + 4.7*exp(-(-79*mV-v)**2/(50*mV)**2) *ms : second
taum  =  .04*ms + .46*exp(-(-38*mV-v)**2/(30*mV)**2) *ms : second
tauh  =  1.2*ms + 7.4*exp(-(-67*mV-v)**2/(20*mV)**2) *ms : second
dm/dt = (m_inf - m)/taum : 1
dh/dt = (h_inf - h)/tauh : 1
dn/dt = (n_inf - n)/taun : 1
'''

# notice that we simply "concatenate" the strings with all equations
# ( with the operator+=).
eqs = eqs_V
eqs += eqs_cond
eqs += eqs_I
eqs += eqs_activation

```

<!-- #region id="1sHmIkHVkK_O" -->
#### Your code here
<!-- #endregion -->

```python id="HVKItlSCkN3G" executionInfo={"status": "ok", "timestamp": 1631033759327, "user_tz": -120, "elapsed": 14, "user": {"displayName": "Elias Mateo Fernandez Santoro", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiN3naitVynkmvN89-tczt7o_TwYkXrydU6xGHOKQ=s64", "userId": "07472471926015090759"}}
# your code here

```

<!-- #region id="OjhNiV7Iiw9q" -->
# What Causes A Spike?
<!-- #endregion -->

<!-- #region id="oB5Va5W6jHIH" -->
There are **two essential components** for the **action potential generation**: a negative feedback which reduces the membrane potential when it’s high (voltage activated potassium) — i.e., hyperpolarizes the membrane when it’s depolarized, and a positive feedback which increases the membrane potential when it’s increased, based on voltage sensitive sodium channels. Spikes appear due to the imbalance/disequilibrium in conductances and their time courses. 

> A spike is the product of an imbalance of the current flux

Note that the time courses and contributions to the membrane potential of the ion channels are very different: **potassium and sodium de-inactivation are slow** (slow time constant) while **sodium activates very quickly**. 

In what follows we will be running experiments to build intuition about how activation variables and membrane potential interact under different stimulation conditions.
<!-- #endregion -->

<!-- #region id="AxU6cuqojN04" -->
## A Spiking Analogy
---

 **Here's an analogy:**. In a house, the number of open windows and air conditioners has to do with the difference between outside and inside temperature, and the temperature preference of people inside the house, **Na**dia, **K**onrad and **Cl**aire.
 
- **Na**dia likes it hot and when the temperature picks up she starts to dance, increasing the temperature in the room. 

- The hotter **Na**dia gets, the more windows Konrad opens, but that takes time (tau), so the temperature increases faster than Konrad manages to open windows. 

- But then **Na**dia overheats (spikes) and has to rest for a while (inactivated state). **K**onrad gets activated when he sees that Nadia is dancing, so when she starts to franctically dance, he opens the windows. 

- Because **Cl**aire likes it cold, she always has an air conditioning on, so temperature is always going lower (heat is constantly leaking at a steady rate).


---
<!-- #endregion -->

<!-- #region id="0Vy3ZbJSi9-6" -->
# Experiments, interpretation and parameter exploration
<!-- #endregion -->

<!-- #region id="-sLDCqzTg1Ns" -->
Using the standard HH equations (pre-defined in the code for your convenience), conduct the experiments suggested. **For every experiment, explain what you see in terms of activation variables, conductances and the resting membrane potential**. 

If you have doubts or remarks, they will be very welcome in our forums!

<!-- #endregion -->

<!-- #region id="cm3hUCH8ZR0y" -->
## Current Injection Protocols
<!-- #endregion -->

<!-- #region id="DOotO8Q6ZV85" -->

#### Protocol 1. Produce a 'transient' until relaxation for 300s. Start the model with the following initial values:
- v = -65 mV
- m = 0
- h = 0 
- n = 0

Plot activation variables and membrane potential.
<!-- #endregion -->

```python id="PKmnK3QDesOd" executionInfo={"status": "ok", "timestamp": 1631033759327, "user_tz": -120, "elapsed": 13, "user": {"displayName": "Elias Mateo Fernandez Santoro", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiN3naitVynkmvN89-tczt7o_TwYkXrydU6xGHOKQ=s64", "userId": "07472471926015090759"}}
# Your code here

```

<!-- #region id="1LnXbIc1lAHl" -->
#### Review and analyze:

- explain what you see!
<!-- #endregion -->

<!-- #region id="R2bxEU_ocmnQ" -->
### Protocol 2. Inject pulses of different amplitudes

  - 50ms relaxation time (no injected current)
  - 10 uA/cm^2 for 5 ms 
  - relax for 50ms (no injected current, set it to zero)
  - apply -10 uA/cm^2 for 5 ms 
  - relax for 100ms
  
<!-- #endregion -->

<!-- #region id="DuuHZN5GsSZd" -->
#### Your Code Here
<!-- #endregion -->

```python id="Z41H26kjsUOK" executionInfo={"status": "ok", "timestamp": 1631033759328, "user_tz": -120, "elapsed": 13, "user": {"displayName": "Elias Mateo Fernandez Santoro", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiN3naitVynkmvN89-tczt7o_TwYkXrydU6xGHOKQ=s64", "userId": "07472471926015090759"}}
# Your code here

```

<!-- #region id="e-z7CY8_ZauU" -->
## Explore the **parameter space** of conductance values
<!-- #endregion -->

<!-- #region id="tFv1TNmGZerd" -->
For the following items, run simulation, observe results and explain about what ion channel drive the different observed behaviors.

4. Double the value of the Sodium conductance in the original model and apply protocol 2. 

5. Back in the original version double the Potassium conductance, and apply protocol 2.

5. Back in the original version double the leak conductance, and apply protocol 2.
<!-- #endregion -->

<!-- #region id="JUcBG8rWd8wJ" -->
### Your Solution
<!-- #endregion -->

```python id="zkZX_x0teGP-" executionInfo={"status": "error", "timestamp": 1631033759636, "user_tz": -120, "elapsed": 320, "user": {"displayName": "Elias Mateo Fernandez Santoro", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiN3naitVynkmvN89-tczt7o_TwYkXrydU6xGHOKQ=s64", "userId": "07472471926015090759"}} outputId="c7a8cbe7-946f-4381-9a07-66833efece4c" colab={"base_uri": "https://localhost:8080/", "height": 324}
# Exercise 4
restore('setup')

```

```python id="443ExA1HldRI" executionInfo={"status": "aborted", "timestamp": 1631033759633, "user_tz": -120, "elapsed": 315, "user": {"displayName": "Elias Mateo Fernandez Santoro", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiN3naitVynkmvN89-tczt7o_TwYkXrydU6xGHOKQ=s64", "userId": "07472471926015090759"}}
# Exercise 5
restore('setup')

```

```python id="JLM3qCcLlc4L" executionInfo={"status": "aborted", "timestamp": 1631033759633, "user_tz": -120, "elapsed": 315, "user": {"displayName": "Elias Mateo Fernandez Santoro", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiN3naitVynkmvN89-tczt7o_TwYkXrydU6xGHOKQ=s64", "userId": "07472471926015090759"}}
# Exercise 6
restore('setup')

```

<!-- #region id="R8pfaspzsHfk" -->
## Challenges

Challenge 1. Find two current injection protocols that produces exactly 3 spikes changing both the amplitude and duration of a single current square pulse.

<!-- #endregion -->

```python id="vD01mR7qin4O" executionInfo={"status": "aborted", "timestamp": 1631033759634, "user_tz": -120, "elapsed": 315, "user": {"displayName": "Elias Mateo Fernandez Santoro", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiN3naitVynkmvN89-tczt7o_TwYkXrydU6xGHOKQ=s64", "userId": "07472471926015090759"}}
# Challenge 1
start_scope() # initialize brian2 workspace
restore('setup')

```

<!-- #region id="lhoca_aLgQcw" -->
Challenge 2. Can you tune the conductance values for the model to spike spontaneously (no injected current)?
<!-- #endregion -->

```python id="i2iCeUd764vZ" executionInfo={"status": "aborted", "timestamp": 1631033759634, "user_tz": -120, "elapsed": 315, "user": {"displayName": "Elias Mateo Fernandez Santoro", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiN3naitVynkmvN89-tczt7o_TwYkXrydU6xGHOKQ=s64", "userId": "07472471926015090759"}}
# Challenge 2:
restore('freshly_setup') # restore state of simulator

```

<!-- #region id="NXgnBFk52Pro" -->
# Final Challenge
<!-- #endregion -->

<!-- #region id="SvdxCijO2TSg" -->
For the model in Challenge 2, in a single panel:
- plot the trajectories of the m and n state variables against each other? 
- plot the trajectories of the m and h state variables against each other? 
(for both these plots use the horizontal coordinate (x) to plot m and the  vertical coordinate (y) to plot n and h). 
<!-- #endregion -->

```python id="AnUOOqxV2-P3" executionInfo={"status": "aborted", "timestamp": 1631033759635, "user_tz": -120, "elapsed": 316, "user": {"displayName": "Elias Mateo Fernandez Santoro", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiN3naitVynkmvN89-tczt7o_TwYkXrydU6xGHOKQ=s64", "userId": "07472471926015090759"}}

```

<!-- #region id="p7iOB2Ng7nLE" -->
# Going Further
<!-- #endregion -->

<!-- #region id="BtBHe4lWg7Bp" -->
## Cable Equation and Compartmental Models

[![Saltatory-Conduction.gif](https://i.postimg.cc/0jv5hTFh/Saltatory-Conduction.gif)](https://postimg.cc/zbd1gcJk)

We have scratched the surface of action potential generation. Our primary goal in this course is to simplify the complexity of experiment into models that explain a wider phenomenology. But there are many interesting aspects that are left for you to discover, such as models with multiple compartments.

Here's my suggested list:
- Read about saltatory conductance in myelinated axons! https://en.wikipedia.org/wiki/Saltatory_conduction
- Watch Wulfram Gerstner's MOOC (chapter 3.b) :
  - https://www.edx.org/course/neuronal-dynamics
- Compute the propagation of an action potential in an axon in brian
- Play with the tutorial in pyramidal cells in opensourcebrain.org ("simulate electrophysiologically detailed cell models")
- Read and Run example of a multicompartmental cable in brian:
  - multicompartmental axon: https://brian.readthedocs.io/en/stable/examples-misc_cable.html
  

<!-- #endregion -->

<!-- #region id="k9aJkXljN8hm" -->

# Further Online Resources
<!-- #endregion -->

<!-- #region id="r01JnlUJZPR8" -->
[Nernst Simulator + Resting Membrane Potential](http://www.nernstgoldman.physiology.arizona.edu/#download)

[Resting membrane potential](https://www.physiologyweb.com/lecture_notes/resting_membrane_potential/resting_membrane_potential.html)

[Action Potential](https://www.physiologyweb.com/lecture_notes/neuronal_action_potential/neuronal_action_potential.html)

[Bilipid Layer Permeability](https://www.physiologyweb.com/lecture_notes/biological_membranes/lipid_bilayer_permeability.html)

[Derivation of the Nernst Equation](https://www.physiologyweb.com/lecture_notes/resting_membrane_potential/derivation_of_the_nernst_equation.html)

<!-- #endregion -->

<!-- #region id="r-rG3CjMCHe7" -->
#License

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

Mario Negrello, Daphne Cornelise, Elias Santoro. Figure sources: Geometry of Bursting, Eugene Izhikevich (2007). Saltatory conductance gif by By Dr. Jana - http://docjana.com/saltatory-conduction/

<!-- #endregion -->
