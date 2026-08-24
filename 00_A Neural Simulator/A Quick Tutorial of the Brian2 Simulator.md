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

<!-- #region id="aQoA0fOfIjWr" colab_type="text" -->
# Workshop Day 1: A Quick Tutorial of the Brian2 Simulator

---
---


<!-- #endregion -->

<!-- #region id="5DUVMJiRIpAx" colab_type="text" -->
# Part 1: A Quick Tutorial of the Brian2 Simulator
<!-- #endregion -->

<!-- #region id="URJworgssnv9" colab_type="text" -->
In this workshop, we shall use a neuron simulator called Brian 2, which is based on Python and enables the instantiation of single or networks of neurons, their simulation and the plotting of elaborate results. Brian 2 is a popular simulator as well as a good representative of a large class of neuronal simulators available out there.

We assembled a small Brian tutorial], that you will go over during Day 1 of the workshop.
<!-- #endregion -->

<!-- #region id="nqn7zGeRqPBP" colab_type="text" -->
---
## The Brian2 Simulator

Romain Brette and Dan Goodman developed the Brian simulator for spiking neural networks as a response to the lack of hegemony between different softwares used in neural network simulation (Goodman and Brette, 2008) as each of these softwares requires learning different scripting languages. Brian presents the following advantages: 

*   As it is written on Python, it offers a tighter integration with the various tools and libraries of Python, resulting in a lot of flexibility.
*   Differential equations can be defined at the highest level using standard mathematical notation.
*   For linear differential equations, exact updates are used while for non linear differential equations, Euler and exponential Euler methods are used. 
*   Contrary to other simulators, Brian is unit-consistent. This reduces errors when modeling.
*   Further features can be easily controlled; e.g. network connectivity offers a lot of flexibility (all-to all random connectivity, specific connectivity, delays, synaptic-weight functions, among others).

The following section is a step-by-step guide of how to get started with  Brian and model a simple neuron; more information can be found [here](https://brian2.readthedocs.io/en/latest/index.html\#). In this example, the neuron is modelled by the following differential equation:

$$\frac{dV_m(t)}{dt} = \frac{I(t)}{C_m}$$

where $V_m$ is the membrane potential, $C_m$ is the membrane capacitance and $I$ is the current input.  

<!-- #endregion -->

<!-- #region id="10Jgxfv3gcTr" colab_type="text" -->
---
## Importing the Brian Toolbox
First, every Brian script in Python begins by importing the Brian toolbox. Other toolboxes can also be imported, such as *matplotlib*  for plotting tools and *numpy* to help with mathematical computations. After importing the libraries, it is useful to invoke the *start_scope* function as it starts a new scope for the magic functions. In other words, it resets the simulations by clearing any preceding neurongroup defininitions.

<!-- #endregion -->

```python id="gzka8xx-guCy" colab_type="code" outputId="64323d4b-c896-485d-a418-77a875cea84b" executionInfo={"status": "ok", "timestamp": 1588986113408, "user_tz": -180, "elapsed": 7229, "user": {"displayName": "Christos Strydis", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gi63S46dCgrA3KHL62sJZLEG5Cg67r2Yf1pTEwLLw=s64", "userId": "08542955873475941983"}} colab={"base_uri": "https://localhost:8080/", "height": 245}
%%time

!pip install brian2 # this installs brian2 in the colaboratory (! is a shell commmand)

from brian2 import *
import matplotlib.pyplot as plt # for plotting in python
import brian2.numpy_ as np
import matplotlib.gridspec as gridspec

start_scope() # resets the simulations and clears any preceding model definitions
```

<!-- #region id="r8b_9kSxZmC9" colab_type="text" -->
<font color=green>*Hint: Let your mouse pointer rest on top of the various Python or Brian functions, modules etc. and a quick reference entry should pop up.*</font>
<!-- #endregion -->

<!-- #region id="6uyKczJFXGIV" colab_type="text" -->
## Notebook magic commands

In the above code cell, preceding the Python and Brian commands, you may have noticed that there is a ```%%time``` command, specially highlighted. This is known as a Notebook *magic command* which help add more functionality to the code cell than the "user code", in this case Python and Brian.

Specifically ```%%time``` measures the time it takes to execute the whole code cell and separates the time between so-called CPU (pure compute) and Wall-clock (human-perceptible) time.

You can read more about magic commands that you can try in this tutorial and in the next assignments [here](https://ipython.readthedocs.io/en/stable/interactive/magics.html#cell-magics).

The magic command ```%%timeit``` can also be used. The notable difference of this command is that it runs the specified code many times and computes an average. You can specify the number of runs with the ```-n``` option, but if nothing is passed a fitting value will be chosen based on computation time. More advanced magic-command uses can be found [here](https://www.dataquest.io/blog/advanced-jupyter-notebooks-tutorial/).

Keep in mind that a single-percent symbol ```%``` will make any magic command run for a single line of the code, while a double-percent symbol ```%%``` will make the command run for the whole code cell.


> <font color=yellow>QC1.1: As you may have noticed, CPU and Wall-clock times are never exactly the same. Can you imagine why that is?</font>

> <font color=yellow>QC1.2: Try running the same code cell a few times. Do you get the same time measurement each time? Why (not)?</font>
 
<!-- #endregion -->

<!-- #region id="wZnQOAEog-so" colab_type="text" -->
---
## Defining Parameters and Dealing with Units
Second, the parameters of the neurons are defined with the correct units. Brian accepts all basic SI units accompanied by all standard prefixes. 
<!-- #endregion -->

```python id="lRdzz7EhhFTB" colab_type="code" outputId="0ff10eb5-a4f2-4f0f-dd71-370060146108" executionInfo={"status": "ok", "timestamp": 1588986113414, "user_tz": -180, "elapsed": 7047, "user": {"displayName": "Christos Strydis", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gi63S46dCgrA3KHL62sJZLEG5Cg67r2Yf1pTEwLLw=s64", "userId": "08542955873475941983"}} colab={"base_uri": "https://localhost:8080/", "height": 52}
%%time

Cm = 1.0*uF/cm**2  #The membrane capacitance
```

<!-- #region id="LzS_-l_ShHdS" colab_type="text" -->
---
## Defining an Equation
Equations are defined **as strings** using standard mathematical notations; units also have to be defined as shown below. In this example, the voltage of the membrane $V$ is being computed, thus the unit is in Volts. The input for this example is the current $I$ and thus, is defined as shown below (with unit amp/m$^2$). Note that Brian recommends using triple single quotes when definining equations.
<!-- #endregion -->

```python id="7jrsx66khJN6" colab_type="code" outputId="7da89621-8ad6-4459-c103-1f8fcd3bc9c3" executionInfo={"status": "ok", "timestamp": 1588986113415, "user_tz": -180, "elapsed": 7016, "user": {"displayName": "Christos Strydis", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gi63S46dCgrA3KHL62sJZLEG5Cg67r2Yf1pTEwLLw=s64", "userId": "08542955873475941983"}} colab={"base_uri": "https://localhost:8080/", "height": 52}
%%time

eqs_IF = '''
    dV/dt = I/Cm : volt
    I : amp*meter**-2
'''
```

<!-- #region id="-5M6pf3ohOy2" colab_type="text" -->
---
## Creating a Neuron
Neuron models are generated using the ```NeuronGroup()``` function, which requires the specification of the number of neurons *N*, the neuron equation *eqs* as well as the integration *method*, for example ['forward Euler'](https://en.wikipedia.org/wiki/Euler_method).
<!-- #endregion -->

```python id="fBoawKRChPD7" colab_type="code" outputId="ec2c3c87-ffac-4c2e-854a-a4177e53660a" executionInfo={"status": "ok", "timestamp": 1588986113415, "user_tz": -180, "elapsed": 6976, "user": {"displayName": "Christos Strydis", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gi63S46dCgrA3KHL62sJZLEG5Cg67r2Yf1pTEwLLw=s64", "userId": "08542955873475941983"}} colab={"base_uri": "https://localhost:8080/", "height": 52}
%%time

N_IF = 1
Neuron_IF = NeuronGroup(N_IF, eqs_IF, method = 'euler')
```

<!-- #region id="x6cxDlVUhoGV" colab_type="text" -->
---
## Recording
At this point, the objective of the simulation must be specified through the ```StateMonitor()``` function for monitoring during simulation. For this example, both the membrane potential $V$ and the input current $I$ are measured at every simulation step and saved in the ```Neuron_statemon``` for later usage.
<!-- #endregion -->

```python id="BrDrFIZLhpzq" colab_type="code" outputId="bfc1e35b-5f10-48b2-890f-3f69890961a7" executionInfo={"status": "ok", "timestamp": 1588986113416, "user_tz": -180, "elapsed": 6954, "user": {"displayName": "Christos Strydis", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gi63S46dCgrA3KHL62sJZLEG5Cg67r2Yf1pTEwLLw=s64", "userId": "08542955873475941983"}} colab={"base_uri": "https://localhost:8080/", "height": 52}
%%time

Neuron_statemon_IF = StateMonitor(Neuron_IF, variables=['V','I'], record = True) 
  # "record = True" can also give "record = [0, 10] where the numbers are the indices of neurons to be recorded.
store('IF_Neuron') # stores the neuron and state monitor

# quick validation
Neuron_validation_IF = Neuron_IF.equations # return LaTeX code that can be copied and rendered below for reporting but also for validation purposes.
f"{Neuron_validation_IF}" # use a simple Python f-string to print the LaTeX math formula inline 
```

<!-- #region id="KuA11Xo8hsuZ" colab_type="text" -->
---
## Simulating a Neuron
To start the simulation, the runtime (100 milliseconds in this example) and the input variables (applied current in this example) are defined. Then, the results can be plotted. The figure below shows the voltage response of this neuron. As expected, the membrane potential increases linearly with a constant current. 

<!-- #endregion -->

```python id="dOME3tVKhtaB" colab_type="code" outputId="4d3f5db3-a791-4892-fa20-bcd315736296" executionInfo={"status": "ok", "timestamp": 1588986114042, "user_tz": -180, "elapsed": 7562, "user": {"displayName": "Christos Strydis", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gi63S46dCgrA3KHL62sJZLEG5Cg67r2Yf1pTEwLLw=s64", "userId": "08542955873475941983"}} colab={"base_uri": "https://localhost:8080/", "height": 428}
%%time

restore('IF_Neuron')
# We set the experiment runtime and initial input
runtime = 100*ms # units are added via the multiplication symbol
Neuron_IF.I = 1*uamp*cm**-2 # remember "I" is a variable contained in object "Neuron"
run(runtime) # run the neuron 

# Plot
fig = plt.figure(figsize=(15, 6))

gs1 = gridspec.GridSpec(10, 5)
gs1.update(left=0.01, right=0.5, wspace=9)
ax1 = plt.subplot(gs1[:-1, :])
ax1.plot(Neuron_statemon_IF.t/ms, Neuron_statemon_IF.V[0]/mvolt,'C1',lw='2')
ylabel('Voltage (mV)')

ax2 = plt.subplot(gs1[-1, :])
ax2.plot(Neuron_statemon_IF.t/ms, Neuron_statemon_IF.I[0]/(uamp/cm**2),'C2',lw='2')
xlabel('t (msec)')
ylabel(u"Current (\u03bcamp/cm\u00B2)")

show()
```

<!-- #region id="eNvfKi6Ocm9E" colab_type="text" -->
> <font color=yellow>QC1.3: Try running the same simulation for runtimes of 100 ms (original), 1000 ms and 10000 ms. Plot CPU and Wall-clock times as a function of the different runtimes. What do you observe? Why?</font>

<font color=green>*Hint: These are known as performance-scalability plots and you will be asked to plot a few of them throughout this workshop. You can plot directly in a notebook via matplot lib or e.g. through Excel in your computer using Line plots or Scatter plots with Markers.*</font>
<!-- #endregion -->

<!-- #region id="JSfQaGLOhZHo" colab_type="text" -->
---
## Simulating Different Currents Over Time
To see the response of the membrane potential to a varying current input, it is only needed to change the ```Neuron.I``` value and run the code again. For instance, a 50-millisecond current pulse can be created as shown below. 
<!-- #endregion -->

```python id="t716a5fVh7Yl" colab_type="code" outputId="f707a522-4a8a-4cbf-ecbc-59c9d283f49a" executionInfo={"status": "ok", "timestamp": 1588986114764, "user_tz": -180, "elapsed": 8268, "user": {"displayName": "Christos Strydis", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gi63S46dCgrA3KHL62sJZLEG5Cg67r2Yf1pTEwLLw=s64", "userId": "08542955873475941983"}} colab={"base_uri": "https://localhost:8080/", "height": 423}
%%time

restore('IF_Neuron')

# We run the code in blocks:
# First choose a runtime for the first part
runtime = 100*ms # Choose a runtime
Neuron_IF.I = 0*uamp*cm**-2 # set a new input current
run(runtime) # run the neuron for the specified runtime
# Second block:
Neuron_IF.I = 1*uamp*cm**-2 # set a new input current
run(50*ms) # run the neuron
# Third block:
Neuron_IF.I = 0*uamp*cm**-2 # set a new input current
run(100*ms) # run the neuron

# Plot
fig = plt.figure(figsize=(15, 6))

gs1 = gridspec.GridSpec(10, 5)
gs1.update(left=0.01, right=0.5, wspace=9)
ax1 = plt.subplot(gs1[:-1, :])
ax1.plot(Neuron_statemon_IF.t/ms, Neuron_statemon_IF.V[0]/mvolt,'C1',lw='2')
ylabel('V mV')

ax2 = plt.subplot(gs1[-1, :])
ax2.plot(Neuron_statemon_IF.t/ms, Neuron_statemon_IF.I[0]/(uamp/cm**2),'C2',lw='2')
xlabel('t (msec)')
ylabel('I (\u03bcamp/cm\u00B2)')

show()
```

<!-- #region id="IJApsU-ch6XC" colab_type="text" -->
By implementing a threshold potential at which a spike occurs, the neuron model in this example becomes a spiking-neuron model known as the **Integrate-and-Fire model** (IaF or I&F). It is the basis for modern spiking-neuron models which will be discussed next. 
<!-- #endregion -->

<!-- #region id="hsxIWNwiqKuT" colab_type="text" -->
---
## Leaky Integrate-and-Fire Model
This model reflects the diffusion of ions occurring through the membrane while the cell is not in equilibrium (Dayan and Abbott, 2005). This is achieved by introducing a leak term $I_l(t) = \frac{V_m(t)}{R_m}$. The introduction of this leak term causes the cell to fire only when $I(t)$ is bigger than a new threshold $I_{th} = \frac{V_{th}}{R_m}$ (Izhikevich,  2007). Hence, if the current does not reach this new threshold current, the leak term will leak out any change in potential. 
$$ C_m\frac{dV_m(t)}{dt} = I(t)-\frac{V_m(t)}{R_m}$$
where $R_m$ is the membrane resistance.

<!-- #endregion -->

<!-- #region id="29D-NHF3Krk3" colab_type="text" -->
---
## Implementing the Leaky I&F Model on the Brian2 Simulator
The membrane resistance parameter needs to be defined as well as the new leak current $I_l$. This model can be improved by including a refractory period that prevents the neuron to fire for a certain amount of time after the spike. This period can be manually added to the *NeuronGroup* function as follows:
<!-- #endregion -->

```python id="NY_fvtu7kTQ9" colab_type="code" outputId="916e6237-3a2c-4b87-eaf9-349737bd6128" executionInfo={"status": "ok", "timestamp": 1588986119814, "user_tz": -180, "elapsed": 13302, "user": {"displayName": "Christos Strydis", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gi63S46dCgrA3KHL62sJZLEG5Cg67r2Yf1pTEwLLw=s64", "userId": "08542955873475941983"}} colab={"base_uri": "https://localhost:8080/", "height": 423}
%%time

start_scope()

Cm = 1.0*uF/cm**2   
Vt = -40*mvolt      #Threshold
Vr = -65*mvolt      #Reset Voltage
Rm =  10*Mohm*cm**2 #Membrane resistance
eqs_LIF = '''
    dV/dt = (I-I_l)/Cm : volt
    I_l = V/Rm : amp*meter**-2
    I : amp*meter**-2
'''
N = 1
Neuron_LIF = NeuronGroup(N, eqs_LIF, threshold='V>Vt', reset='V=Vr', refractory = 2*ms, method = 'euler', name='IF_Neuron')
Neuron_statemon_LIF = StateMonitor(Neuron_LIF, variables=['V','I'], record = True)


Neuron_LIF.V = -70*mvolt
Neuron_LIF.I = 0*uamp*cm**-2
run(100*ms)

Neuron_LIF.I = 1*uamp*cm**-2
run(50*ms)

Neuron_LIF.I = 0*uamp*cm**-2
run(100*ms)

Neuron_LIF.I = 1*uamp*cm**-2
run(100*ms)

Neuron_LIF.I = 0*uamp*cm**-2
run(50*ms)

Neuron_LIF.I = 1*uamp*cm**-2
run(50*ms)

Neuron_LIF.I = 0*uamp*cm**-2
run(runtime)

# Plot
fig = plt.figure(figsize=(15, 6)) 

gs1 = gridspec.GridSpec(10, 5)
gs1.update(left=0.01, right=0.5, wspace=9)
ax1 = plt.subplot(gs1[:-1, :])
ax1.plot(Neuron_statemon_LIF.t/ms, Neuron_statemon_LIF.V[0]/mvolt,'C1',lw='2')
ylabel('V mV')

ax2 = plt.subplot(gs1[-1, :])
ax2.plot(Neuron_statemon_LIF.t/ms, Neuron_statemon_LIF.I[0]/(uamp/cm**2),'C2',lw='2')
xlabel('t (msec)')
ylabel('I (\u03bcamp/cm\u00B2)')

show()
```

<!-- #region id="aHsxtKoHWOGv" colab_type="text" -->
In tomorrow's lab, we will occupy ourselves with single-neuron simulations and the respective increases in computational effort in response to introducing advanced modeling features. In view of Brian's multithreading capabilities still not working as intended, we have resorted to the underlying libraries of Python to demonstrate multithreading execution and conduct some performance experiments.

Stay tuned and see you tomorrow!
<!-- #endregion -->
