---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region id="SPzlot5BMSad" -->
# Dynamics of the Adaptive Exponential Integrate and Fire
<!-- #endregion -->

<!-- #region id="5rPvYTvJX5m_" -->
In the last project we dove into the equations of the Hodgkin Huxley model of spike generation. That model is described by the temporal evolution of four state variables (quiz yourself: do you know which are the state variables of the HH?), and hence its dynamics are quite complex to analyze. 

In this project we will learn how to simulate and parameterize the **Adaptive Exponential Integrate and Fire model** (AdEx).  You will observe how the parameters of the AdEx model can lead it to produce a vast variety of spiking patterns. We will review type 1 and type 2 neuronal firing (introduced in the lecture) and will calculate **I x F** curves (i.e., injected current **I** vs spiking frequency  **F**) of the different types, and use the IxF curves to distinguish neuronal behavior.

<!-- #endregion -->

<!-- #region id="CeJCpYbPayzM" -->
# Learning Objectives
<!-- #endregion -->

<!-- #region id="ZTpIeABra1zF" -->
**After this project you will be able to:**
- Recognize the state variables of the AdEx model.
- Implement an AdEx model in Brian and perform simulations with different parameters.
- Visualize the role of different parameters of the AdEx model on the dynamics of spiking.
- Keep your neurons unit-consistent (SI units: Ampere, Volts, Farads, Siemens).
- Calculate and display F x I curves and use them to compare different neuronal models.
- Use F x I curves to distinguish between 'integrators (type 1) and 'resonators' (type 2).

<!-- #endregion -->

<!-- #region id="rOkay8Q1bmWz" -->
# Terminology
<!-- #endregion -->

<!-- #region id="RHLNnEz-bpea" -->
- **Dynamical system**: A mathematical description of the rules governing the state evolution of a system.
- **State variable**: one of the variables that is needed to describe the current state of a system.
- **Phase space**: The space occupied by the all the state variables of a given system.
- **Parameter**: A property of a sytem that influences its dynamics, usually a variable that changes slower than the state variables and is not directly influenced by them.
- **F x I curves**: Curves that measure the relationship between spiking rate  (Frequency) and injected current (I).
- **Equilibria**: Points in phase space where the state maps onto itself.
- **Nullclines**: Curves where the gradient for one particular state variable is zero (dx/dt = 0).
- **Resonators and Integrators** (Type II and I): Descriptions of neuronal types as a function of their spiking properties.
<!-- #endregion -->

<!-- #region id="b0olo4r3MZze" -->
## Initialization
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="WGYa5AZU_YbP" executionInfo={"status": "ok", "timestamp": 1651221592538, "user_tz": -120, "elapsed": 4895, "user": {"displayName": "Anna-Maria K.", "userId": "15616814866073248468"}} outputId="19bef0a1-233d-425b-a0fe-2658aa683028"
!pip install brian2

from brian2 import *
import time
from scipy import optimize
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from sympy.solvers import solve
from sympy import symbols
import sympy as sp
%matplotlib inline

cl=Clock(dt=0.1*ms)
```

<!-- #region id="iTM1MP4JmJ2F" -->
## The AdEx Integrate and Fire Model
<!-- #endregion -->

<!-- #region id="VboRtXCpBp_B" -->
The Hodkgin-Huxley model, in an attempt to capture much physiological detail has a multitude of equations. That is both computationally expensive and cumbersome. Mathematical work to understand the essential aspects of the HH dynamics have led to significant simplifications, capable of reproducing a nearly complete spectrum of spiking phenomena including tonic and phasic spiking, bursting, chattering, rebound spikes, and more. Simplified equations are more amenable to analysis and computation and may retain the spike generation behaviors of the more complex model, at a very moderate cost. In fact simplified models can reproduce more complex spiking phenomenology than the more complex models such as the Hodgkin-Huxley or the Goldman-Hodgkin-Katz models.

 In modern computational neuroscience one of the most used simplified models is the **Adaptive Exponential** neuronal model, or *AdEx* for short. It is of the 'integrate and fire' type, because unlike the HH model where all the state variables are continuous. It has a 'threshold value', which produces a spike and resets to a given reset potential. Its dynamics are defined by **two state variables**, the membrane potential and a *refractory* variable. We call it a **2D Integrate and Fire** model. The latter indicating the combined state of the refractory mechanisms, which repolarize the neuron (such as sodium inactivation and potassium activation). In the AdEx model, the positive feedback of Sodium channels that lead to the spike is replaced by an exponential function with a fast rise, able to correctly reproduce the spike shapes of many types of cortical neurons.



<!-- #endregion -->

<!-- #region id="4KtOVSJ57biH" -->
Below we present the AdEx equations, with parameters and state variables.

For the dynamics of the **membrane potential** we have the following variables, common base units and parameters:

> $V_m$ - Membrane potential, a __state variable__ (milivolts)  \\
> $g_L$ - Leak conductance, a __parameter__ passive decay through the membrane (siemems) \\
> $E_L$ - The reversal potential of the passive leak, a __parameter__ (volts) \\
> $V_T$ - The spike 'threshold' (volts) \\
>$\Delta_T$ - A 'slope' factor, a __parameter__ controling the rise time of the spike (ms) \\
> $I$ - Injected current, a __parameter__ either entering synaptically or through a current clamp (Amperes). \\
> $C$ - Capacitance, a __parameter__ determines how fast the membrane potential changes \\
> $w$ - Refractory current (Ampere), the other __state variable__. Notice that it has the opposite sign of $I$.

For the dynamics of the **refractory variable** $w$ representing the potassium currents and inactivations, in addition to some of the above, we have:

> $\tau_w$ - adaptation time constant (Ampere) \\
> $a$ - subthreshold adaptation \\
> $b$ - spike triggered adaptation \\



<!-- #endregion -->

<!-- #region id="qP0MFzSKxCgn" -->
**Membrane Dynamics**


\begin{align*}\frac{\mathrm{d}V_m}{\mathrm{d}t} &= \frac{ \overbrace{g_L \Delta_T e^{\frac{- V_T + Vm}{\Delta_T}}}^{spike \ rise} + \overbrace{g_L \left(E_L - V_m\right)}^{passive \ component} - w + I}{C} && \text{(unit of $V_m$: $\mathrm{Volt}$)}\\ \frac{\mathrm{d}w}{\mathrm{d}t} &= \frac{a \left(- E_L + V_m\right) - w}{\tau_w} && \text{(unit of $w$: $\mathrm{Ampere}$)}\end{align*} 


**On spiking:**

- Threshold condition:

$$V_m>V_{cut}$$

- Reset statements:

$$  
 \{ \begin{array}{ll}
  V_m = V_r \\
  w  \  +=b \\
 \end{array}
$$

<!-- #endregion -->

<!-- #region id="IdVnDGVWnM-k" -->
As you will notice in the equations above, the AdEx has a number of parameters (gL, EL, VT, a, tauw, Vcut, Vr, b). It also has two **state variables**, $V_m$ and $w$. 

Different combinations of values for the **parameters** will lead to different types of neuronal responses, including spike frequency, spike shape and response to input. In other words, these parameters will influence the dynamics of the neuronal model. By selecting the parameters judiciously, one can reproduce a variety of spike forms. These parameter sets can then be analyzed, and give insight into the physiology and spike generation mechanisms of real neurons.
<!-- #endregion -->

<!-- #region id="2V1BKJiT1KOx" -->
### Warm up: Write the AdEx equations

In order to create the AdEx model, we start by defining the differential equations for membrane dynamics in variable ```state_eqs```. 

We write the AdEx equations below, in a format that Brian understands. Remember to add the 'base units', ```volt``` and ```amp```.
<!-- #endregion -->

```python id="WVQxfLkhYqfH"
# your code
state_eqs = '''
dV_m= (1/C) * ((g_L*d_t*exp((-V_T+V_m)/d_t) + (g_L*(E_L-V_m) - W +I) : volt 
dW= (1/tau_W)* (a*(-E_L + V_m) -W): amp
'''
def linear(dV_m, d_t, g_L, d_T, V_T, V_m, E_L, W, I, C, tau_W):
  return (g_L * d_T * e **((-V_T + V_m)/ d_T) + g_L (E_L + V_m) - W + I) / C





```

```python id="n8snvQ3fJnPa" cellView="form"

```

<!-- #region id="25tOxwrjJ4jp" -->
Additionally, we can define conditions for the spiking model. These can be passed on to the model later when we define NeuronGroup, as seperate parameters. The conditions of interest are about spiking threshold ```threshold_eq``` and resetting the model after a spike ```reset_eqs```. The equations for both parameters can be found above.

We also need to define the unit of the input current ```I```, ```amp```, and we add those to the variable ```input_eqs```.

<!-- #endregion -->

```python id="0FzgEuGQjHP3"
# to keep unit consistency we must  define the unit of I ('input', the applied current)
input_eqs= 5
I : amp
threshold_eq='''
V_m> Vcut
'''

reset_eqs= '''
vm=Vr 
W+=b
'''
```

<!-- #region id="LmnK0DhmjHP-" -->
# Parameterizing Models
<!-- #endregion -->

<!-- #region id="34QzPtSZdJgM" -->
## A function that takes a neuron and returns a parameterized neuron
<!-- #endregion -->

<!-- #region id="XTdfbg8nzftP" -->
--- 
#### Dictionaries for Parameters

In the code below we define some default values for neurons parameters and an experiment, and create a **function** that returns the parameters of a given neuron type in a **dictionary**. This dictionary can then be passed to the ```neuronGroup()``` object, via it's **namespace**, which contains all the runtime parameters of the model we will run.

The advantage of using dictionaries for parameterizations are many:
1. orderliness: we keep all parameter sets in one place;
2. cleanliness: we can change the parameters of the neuron at any point in the code below without having to copy the values;
3. flexibility: we can easily change all the parameters at once;
4. clarity: we can easily access all user variables at once via the namespace;

---
<!-- #endregion -->

<!-- #region id="-NpaJSoBdfpn" -->
We wrote the function below for your convenience. The function ```set_parameters(neuron, name)``` takes an initialized ```neuron``` from Brian and loads it up with ```parameters``` specified by the neuron type we want (tonic, bursting and so on). You can think of this function as differentiating a stem cell ```neuron``` into a specific neuron.

**Task: Read the function below. Try to spot which are the 'default parameters' and which are the parameters that are set when you select a particular neuronal type.**

_Important: in the function below we are not testing if the parameters into the function are meaningful or correct (for example, if 'neuron' is indeed a Brian's neuron.). These kinds of checks are useful in larger coding projects. One could test if the parameters are sensible and only execute the function if they are, otherwise returning an error or warning._
<!-- #endregion -->

<!-- #region id="6IquVG_OW7Qn" -->
#### A python function to ```set_parameters(neuron)```
<!-- #endregion -->

```python id="d_e6HAMciXxt" cellView="form" colab={"base_uri": "https://localhost:8080/", "height": 130} executionInfo={"status": "error", "timestamp": 1651226106753, "user_tz": -120, "elapsed": 362, "user": {"displayName": "Anna-Maria K.", "userId": "15616814866073248468"}} outputId="0cf8d0a7-a048-4c2a-cb20-3749f1f8b988"
def set_parameters(neuron)
```

<!-- #region id="q9Cnx6pwJPiV" -->
### Example:  Parameterize a Neuron
In the field below, create a neuron and parameterize it as a *regularly spiking* neuron ('regular' from the above function).
<!-- #endregion -->

```python id="jIBBmSsBr7MT" colab={"base_uri": "https://localhost:8080/", "height": 252} executionInfo={"status": "error", "timestamp": 1651226288347, "user_tz": -120, "elapsed": 593, "user": {"displayName": "Anna-Maria K.", "userId": "15616814866073248468"}} outputId="d9815c19-e27d-44d0-beae-f7d79f287ea4"
start_scope()
# create a neuron groupp containing 1 neuron defined by equations above
neuron = NeuronGroup(1,model=state_eqs+input_eqs,threshold=threshold_eq, reset=reset_eqs, method='euler',name='neuron')

# set the parameters
neuron = set_parameters(neuron, 'regular') #Return parameters, applied current, and inital vm
```

```python id="D-y2XKj6Oqif"
# check the parameters of your differentiated neuron
print(neuron.namespace)
```

<!-- #region id="Vdh6msbDFQ53" -->
### Record Neuronal Behavior
<!-- #endregion -->

<!-- #region id="5zGD19fW0kB-" -->
A ```StateMonitor()``` is a function that 'records' the selected output variables of the neurons. We create StateMonitors to be able to observe the outputs of the neuron. 

**Note:** In general, simulators do not save every variable computed, as this can lead to enormous memory requirements.
<!-- #endregion -->

```python id="VzofjSsvjHQJ" colab={"base_uri": "https://localhost:8080/", "height": 269} executionInfo={"status": "error", "timestamp": 1651226493557, "user_tz": -120, "elapsed": 145, "user": {"displayName": "Anna-Maria K.", "userId": "15616814866073248468"}} outputId="4be980cf-791a-4ef9-d4f4-4327ae45d1f5"
# to retrieve the values of vm of the first neuron, write 'mon_v.vm[0] or mon_v.vm
mon = StateMonitor(neuron,['vm', 'I', 'w'],record=0) 

# the spike monitor records the times of the spikes in the array 'spikes.t'
spikes=SpikeMonitor(neuron)

```

<!-- #region id="eG6dqqysjHQM" -->
### Run the Simulation (according to parameters)

Note: after parameterizing the neuron also contains the parameters necessary to run the simulation. E.g., I0 is the initial current applied, T0 is the duration for I0.
<!-- #endregion -->

```python id="Bfsclzn4jHQN"
# Set Initial vm and I accordign to the parameters saved in our 'neuron'
neuron.vm = neuron.namespace['dv0']
neuron.I = neuron.namespace['I0']
```

```python id="q_tDMcoTLCSL"
# initial duration (relaxation)
run(neuron.namespace['T0'])

# apply current and run for duration T1
neuron.I = neuron.namespace['I1']
run(neuron.namespace['T1'])

# back to no current and relax
neuron.vm = neuron.namespace['dv0']
neuron.I = neuron.namespace['I0']
run(50*ms)


# we run the neuron for T2 seconds
run(neuron.namespace['T2'])
```

<!-- #region id="_oUwL1PPjHQQ" -->
## Plot the results
<!-- #endregion -->

<!-- #region id="uUV-JNcWLdMv" -->
Here we plot the results of our simulation. Note that to successfully pass the recorded variables to the plot functions, we need to remove the units from our monitors. We can do this by dividing the variable array by the appropriate unit. 
<!-- #endregion -->

```python id="DGe9perpL2X6" colab={"base_uri": "https://localhost:8080/", "height": 249} executionInfo={"status": "error", "timestamp": 1631504311824, "user_tz": -420, "elapsed": 560, "user": {"displayName": "Ph\u01b0\u01a1ng Th\u1ee7y Nguy\u1ec5n H\u1ed3", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gh1iyfUlZFSA0eUkg4R5TKs_EKxVHs4rROi5lTFGA=s64", "userId": "14934147373382570915"}} outputId="61f56969-149b-4709-d9ca-7698bfe1bd88"
# Plot the neuron's membrane potential
figure(figsize=(12,4))
plot(mon.t/ms,mon.vm[0]/mV)
ylabel('V (mV)')
xlabel('t (ms)')
```

<!-- #region id="saqR3OtTL3n2" -->
### Example: Plot all the state variables in separate subplots, including:
1. Membrane potential trace
2. Refractory variable trace
3. Current injection
4. State space (membrane potential vs refractory variable)
<!-- #endregion -->

```python id="o0zNskD_jHQR" cellView="form"
#membrane potential trace
figure(figsize=(12,4))
plot(mon.t/ms,mon.vm[0]/mV)
ylabel('V (mV)')
xlabel('t (ms)')

#refractory variable trace
figure(figsize=(12,4))
plot(mon.t/ms,mon.tauw[0])
ylabel('tauw [0]')
xlabel('t (ms)')

#current injection
figure(figsize=(12,4))
plot(mon.t/ms,mon.I_input/nA)
xlabel('t (ms)')
ylabel('I_input (nA)')

#state space (membrane potential vs refractory variable)
figure(figsize=(12,4))
plot(mon.w[0]/nA,mon.vm[0]/mV)
ylabel('V (mV)')
xlabel('w (nA)')
```

<!-- #region id="jXUVVddsjHQW" -->
# Exercises:

<!-- #endregion -->

<!-- #region id="resPn3I6XQKV" -->
## Exercise 1.
Go back to the line that parameterizes the neuron ```set_parameters(neuron, type)```, choose yourself a neuron type by its name and re-run the simulation. Can you see why the neuron has the name it has? Bonus: Attempt to relate the parameters with the behavior the neuron model displays.
<!-- #endregion -->

<!-- #region id="onN4rDhxkB7u" -->
## Exercise 2. 

Create F x I curves for the following neuronal types: **regular**, and **fast**. Display the FxI curves in the same axes and add legends. Determine which of them is of "Type I" and which one is "Type II".

The idea is that you select a range for input currents (say from 0*nA to 1.5*nA, in 40 steps)

You can check how to calculate and display F x I curves at https://brian2.readthedocs.io/en/stable/examples/IF_curve_LIF.html

Brain's way to do it is to **sweep over the parameter space** of currents (I)via a group of unconnected neurons, where each is given a current pulse of increasing amplitude. 

Note: To compute the firing rate, be sure to only take the spikes that are due to the period of stimulation.
<!-- #endregion -->

<!-- #region id="8YpUI5Wpmkcs" -->
### Your Code
<!-- #endregion -->

```python id="zl1eGJFxjHQX" cellView="form"

start_scope()

# general parameters:
C=281*pF # Can be fixed
gL=30*nS
taum=C/gL
EL=-70.6*mV # Same as changing I
VT=-50.4*mV
DeltaT=2*mV
Vcut=VT+5*DeltaT


n = 1000
duration = 50*second


# set the parameters
namespace_params = set_parameters('regular') #Return parameters, applied current, and inital vm
neuron_regular = NeuronGroup(n,model=state_eqs+input_eqs,threshold=threshold_eq, reset=reset_eqs, namespace=namespace_params, method='euler',name='neuron')

namespace_params = set_parameters('fast') #Return parameters, applied current, and inital vm
neuron_fast = NeuronGroup(n,model=state_eqs+input_eqs,threshold=threshold_eq, reset=reset_eqs, namespace=namespace_params, method='euler',name='neuron')


# to retrieve the values of vm of the first neuron, write 'mon_v.vm[0]
mon = StateMonitor(neuron,['vm', 'I_input', 'w'],record=0) 

# the spike monitor records the times of the spikes in the array 'spikes.t'
spikes=SpikeMonitor(neuron)

# Set Initial vm and I accordign to the parameters saved in our 'neuron'
neuron.vm = neuron.namespace['dv0']
neuron.I_input = neuron.namespace['I0']

# initial duration (relaxation)
run(neuron.namespace['T0'])

# apply current and run for duration T1
neuron.I_input = neuron.namespace['I1']
run(neuron.namespace['T1'])

# back to no current and relax
neuron.vm = neuron.namespace['dv0']
neuron.I_input = neuron.namespace['I0']
run(50*ms)

# we run the neuron for T2 seconds
run(neuron.namespace['T2'])

# Plot the neuron's membrane potential
figure(figsize=(12,4))
plot(mon.t/ms,mon.vm[0]/mV)
ylabel('V (mV)')
xlabel('t (ms)')


```

<!-- #region id="BTRtbRe7VboX" -->
## References
[0] [Dimensionality Reduction and Phase Plane Analysis (ebook)](https://neuronaldynamics.epfl.ch/online/Ch4.html)

[1] [Adaptive Exponential Integrate and Fire model (Scholarpedia)](http://www.scholarpedia.org/article/Adaptive_exponential_integrate-and-fire_model)

[2]	J. Touboul and R. Brette, “Dynamics and bifurcations of the adaptive exponential integrate-and-fire model.,” Biol Cybern, vol. 99, no. 4, pp. 319–334, Nov. 2008.

[3] E. Izhikevich. "Dynamical Systems in Neuroscience". MIT press. 2017.

[4] [M. Stimberg, R. Brette, D. Goodman. "Brian 2, an intuitive and efficient neural simulator" eLife, 2019.](https://elifesciences.org/articles/47314)
<!-- #endregion -->

<!-- #region id="vxtD_sIcNRxN" -->
# To Know More:

To deepen your understanding of mathematics behind the neurodynamics of simplified neuron models you can watch these episodes of Wulfram Gerstners' MOOC. Thereafter you should be equipped to go through the "advanced neurodynamics" addendum to this project.

- Week 4 from [Wulfram Gerstner's MOOC](https://lcnwww.epfl.ch/gerstner/NeuronalDynamics-MOOCall.html)
<!-- #endregion -->

<!-- #region id="SMT8pP9WJxFU" -->
> The section below is provided for reference on more advanced concepts. It can be safely skipped over!
<!-- #endregion -->

<!-- #region id="8HucpNUhVs8M" -->
# ADVANCED NEURODYNAMICS:
<!-- #endregion -->

<!-- #region id="6Fh26EsfmbQJ" -->
This section is for advanced students with a solid calculus foundation. If you want to know more about how equations lead to diverse types of spike generation, you can use the code snippets below to analyze the dynamics of the different models in detail.
<!-- #endregion -->

<!-- #region id="YY_MOQduEMb5" -->
 ## Examining the State Space and Fixed Point Stability

 The code below performs in depth analysis of the stability of 2D systems.
<!-- #endregion -->

<!-- #region id="edGfohvcrlMj" -->
## Find the nullcline equations
<!-- #endregion -->

<!-- #region id="fBJqQOWklS-R" -->
nullclines: the curves in phase space where the gradient is zero. In a 2D system with state variables V and W, the nullclines are given by two equations $\dot{V} = F(V,W) = 0$ and $\dot{W} = G(V,W) = 0$.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="GgBLcvpOruRw" executionInfo={"status": "ok", "timestamp": 1651225535092, "user_tz": -120, "elapsed": 974, "user": {"displayName": "Anna-Maria K.", "userId": "15616814866073248468"}} outputId="e61fd3f4-35a7-444f-dea0-5c7c6369c139"
### Define the symbos you will use to solve the equations
vm1,w1,C1,gL1,taum1,EL1,VT1,DeltaT1,Vcut1,tauw1,a1,b1,I_input1 = symbols('vm w C gL taum EL VT DeltaT Vcut tauw a b I_input')
# Use solve
vm_eq = solve((gL1*(EL1-vm1)+gL1*DeltaT1*sp.exp((vm1-VT1)/DeltaT1)+I_input1-w1)/C1,w1)
w_eq = solve((a1*(vm1-EL1)-w1)/tauw1,w1)
  
y_vm = vm_eq[0]
y_w = w_eq[0]

### Equations 
print('\n Symbolic Equations:\n','\nVm Nullcline:\n', " w = ", y_vm,'\nW Nullcline:\n', " w = ", y_w)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 235} id="S45oo1oQj9wN" executionInfo={"status": "error", "timestamp": 1651225535147, "user_tz": -120, "elapsed": 142, "user": {"displayName": "Anna-Maria K.", "userId": "15616814866073248468"}} outputId="4bd2ed73-7090-402c-8133-a8137904c69b"
### Substitute values of the constants to simplify the equations. 
def nl(gL,EL,VT,DeltaT,a,I):
    yvm = y_vm.subs(gL1,gL/nsiemens)
    yvm = yvm.subs(EL1,EL/mvolt)
    yvm = yvm.subs(VT1,VT/mvolt)
    yvm = yvm.subs(DeltaT1,DeltaT/mvolt)
    yvm = yvm.subs(I_input1, I/namp)
    yw = y_w.subs(a1,a/nsiemens)
    yw = yw.subs(EL1,EL/mvolt)
    return [yvm, yw]
  
I_in = 1*nA

eqs_nl = nl(gL,EL,VT,DeltaT,a,I_in)
vm_0 = eqs_nl[0]
w_0 = eqs_nl[1]

### Simplified Equations for plotting
print('\n Simplified Equations:\n','\nVm Nullcline:\n', " w = ", vm_0,'\nW Nullcline:\n', " w = ", w_0)
```

<!-- #region id="c8aNVBg4YRvw" -->
## Find the intersection points
<!-- #endregion -->

<!-- #region id="fzNxxer1lrio" -->
At the intersection of nullclines we have fixed points, which may be attractors, repellors or saddle nodes.
<!-- #endregion -->

<!-- #region id="cppNmlR1n3q2" -->
### Write the equations of the nullclines from your results
<!-- #endregion -->

```python id="x9ta0Lalw4VP"
### Choose range for vm and w  
min_lin = -100.0 #Min value for vm
max_lin = 20.0 #Max value for vm
resol = 120000 #Resolution for the linspace. Choose the number of points in your linspace. In this case we need a high resolution to find the intersections. 
vm = np.linspace(min_lin, max_lin, resol)

### Get a list with the solution to the equations. Substitute values of vm in the equations vm_0 and w_0. 
y1 = []
for i in range(len(vm)):
  y1.append(vm_0.subs(vm1,vm[i]))
y2 = []
for i in range(len(vm)):
  y2.append(w_0.subs(vm1,vm[i]))
```

<!-- #region id="kZnMWD2ayX2C" -->
### Find Intersections
<!-- #endregion -->

```python id="yhC7J8KKyb3V"
### Find the index where y1 and y2 have a difference of 0. In other words, y1-y2 = 0.
idx=np.argwhere(np.diff(np.sign(np.subtract(y1,y2))) != 0).reshape(-1) + 0

### Position of the intersections.
intersec = np.zeros((len(idx),2))
for i in range(len(idx)):
    x_pos = (vm[idx[i]]+vm[idx[i]+1])/2.
    y_pos = (y1[idx[i]]+y1[idx[i]+1])/2.
    intersec[i] = x_pos,y_pos
```

<!-- #region id="aO3tks8-okUK" -->
### Plot intersections
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 52} id="tQkcRmpYokvd" executionInfo={"status": "ok", "timestamp": 1631086472377, "user_tz": -120, "elapsed": 1676, "user": {"displayName": "Mario Negrello", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gh9vnOzDqUD2QacfGPwd13jMPmwn1hzZzBnVURjO4E=s64", "userId": "10136788594790905986"}} outputId="5cc2dc03-2c72-43bd-d5d1-5cfb751bf269"
fig, ax = plt.subplots(figsize=(15, 8), dpi= 80, facecolor='w', edgecolor='k')
ax.vlines(Vcut/mV, (intersec[1][0]-100), (intersec[1][-1]+100), lw=2, color='k')
plt.plot(vm,y1, label="Vm Nullcline")
plt.plot(vm,y2, label="W Nullcline")
for i in range(len(idx)):
    plt.plot(intersec[i][0],intersec[i][1], 'ro')
xt = ax.get_xticks() 
xt=np.append(xt,Vcut/mV)
xtl=xt.tolist()
xtl[-1]="$V_{threshold}$"
ax.set_xticks(xt)
ax.set_xticklabels(xtl)
xlabel('$v_m (mV)$')
ylabel('$w (nA)$')
plt.xlim([(intersec[0][0]-20), (intersec[1][0]+20)])
plt.ylim([(intersec[0][1]-100), (intersec[1][-1]+100)])
legend();
plt.show() 
```

<!-- #region id="3MlCV2nFS_fo" -->
## Verify your nullclines
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="gQ7lite_kzOD" executionInfo={"status": "ok", "timestamp": 1631086472377, "user_tz": -120, "elapsed": 8, "user": {"displayName": "Mario Negrello", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gh9vnOzDqUD2QacfGPwd13jMPmwn1hzZzBnVURjO4E=s64", "userId": "10136788594790905986"}} outputId="c94de6fb-1ffa-4439-aa99-416b730adacd"
### Equations of dvm/dt and dw/dt
fvm = vm_0-w1
fw = w_0-w1

### Intersection point 1
f1 = fvm.subs(vm1,intersec[0][0])
f1 = f1.subs(w1,intersec[0][1])
print('\ndvm/dt = \n',round(f1,1))

### Intersection point 2
f2 = fw.subs(vm1,intersec[0][0])
f2 = f2.subs(w1,intersec[0][1])
print('\ndw/dt = \n',round(f2,1))
```

<!-- #region id="WsBmTsfcrtDG" -->
## Define the ODEs to plot the streamlines
<!-- #endregion -->

```python id="1iHj9DnNrvzX"
def f(Y, t, C,gL,taum,EL,VT,DeltaT,Vcut,tauw,a,b,I):
    C=C/pfarad # Can be fixed
    gL=gL/nsiemens
    taum=taum/msecond
    EL=EL/mvolt # Same as changing I
    VT=VT/mvolt
    DeltaT=DeltaT/mvolt
    Vcut=Vcut/mvolt
    I_input = I/nA
    tauw = tauw/msecond
    a = a/nsiemens
    b = b/nA
    vm, w = Y
    return [(gL*(EL-vm)+gL*DeltaT*exp((vm-VT)/DeltaT)+I_input-w)/C, (a*(vm-EL)-w)/tauw]

### Choose range for vm and w   
vm_stream = np.linspace(int((intersec[0][0]-20)), int((intersec[1][0]+20)), int((2*(intersec[1][0]-intersec[0][0]))))
w_stream = np.linspace(int((intersec[0][1]-500)), int((intersec[1][1]+200)), int((2*(intersec[1][1]-intersec[0][1]))))

### Colve the ODE
Y1, Y2 = np.meshgrid(vm_stream, w_stream)
t = 0
u, v = np.zeros(Y1.shape), np.zeros(Y2.shape)
NI, NJ = Y1.shape
for i in range(NI):
    for j in range(NJ):
        x = Y1[i, j]
        y = Y2[i, j]
        yprime = f([x, y], t, C,gL,taum,EL,VT,DeltaT,Vcut,tauw,a,b,1.*namp)
        u[i,j] = yprime[0]
        v[i,j] = yprime[1]
```

<!-- #region id="9HOu3colr6V9" -->
## Plot nullclines with streamlines
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 52} id="crGQILffr5fI" executionInfo={"status": "ok", "timestamp": 1631086475100, "user_tz": -120, "elapsed": 2425, "user": {"displayName": "Mario Negrello", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gh9vnOzDqUD2QacfGPwd13jMPmwn1hzZzBnVURjO4E=s64", "userId": "10136788594790905986"}} outputId="b6f0f313-70f9-4d8d-ce47-e554ccb96b81"
### Get a list with the solution to the equations. Substitute values of vm in the equations vm_0 and w_0. 
# This time use stream values.
y1_stream = []
for i in range(len(vm_stream)):
  y1_stream.append(vm_0.subs(vm1,vm_stream[i]))
y2_stream = []
for i in range(len(vm_stream)):
  y2_stream.append(w_0.subs(vm1,vm_stream[i]))

fig, ax = plt.subplots(figsize=(20, 10), dpi= 80, facecolor='w', edgecolor='k')
Q = ax.streamplot(Y1, Y2, u, v)
ax.vlines(Vcut/mV, (intersec[1][0]-200), (intersec[1][-1]+200), lw=2, color='k')
plt.plot(vm_stream,y1_stream, label="Vm Nullcline")
plt.plot(vm_stream,y2_stream, label="W Nullcline")
xt = ax.get_xticks() 
xt=np.append(xt,Vcut/mV)
xtl=xt.tolist()
xtl[-1]="$V_{threshold}$"
ax.set_xticks(xt)
ax.set_xticklabels(xtl)
xlabel('$v_m (mV)$')
ylabel('$w (nA)$')
plt.xlim([(intersec[0][0]-20), (intersec[1][0]+20)])
plt.ylim([(int(min(y1_stream))-30), (intersec[1][-1]+200)])
legend();
plt.show()
```

<!-- #region id="Em2nvvTekTER" -->
## Plot nullclines, streamlines and orbit of spiking neuron
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 52} id="ICWFc6aZkTok" executionInfo={"status": "ok", "timestamp": 1631086476802, "user_tz": -120, "elapsed": 1708, "user": {"displayName": "Mario Negrello", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gh9vnOzDqUD2QacfGPwd13jMPmwn1hzZzBnVURjO4E=s64", "userId": "10136788594790905986"}} outputId="3e64b6b2-1c38-4355-b3a0-4a1513935f75"
vm_orbit = mon.vm[0]/mV
w_orbit = mon.w[0]/nA

fig, ax = plt.subplots(figsize=(20, 10), dpi= 80, facecolor='w', edgecolor='k')
Q = ax.streamplot(Y1, Y2, u, v)
ax.vlines(Vcut/mV, (intersec[1][0]-200), (intersec[1][-1]+200), lw=2, color='k')
plt.plot(vm_orbit,w_orbit, label="Orbit", lw=4, color='r')
plt.plot(vm_stream,y1_stream, label="Vm Nullcline")
plt.plot(vm_stream,y2_stream, label="W Nullcline")
xt = ax.get_xticks() 
xt=np.append(xt,Vcut/mV)
xtl=xt.tolist()
xtl[-1]="$V_{threshold}$"
ax.set_xticks(xt)
ax.set_xticklabels(xtl)
xlabel('$v_m (mV)$')
ylabel('$w (nA)$')
plt.xlim([(intersec[0][0]-20), (intersec[1][0]+20)])
plt.ylim([(int(min(y1_stream))-30), (intersec[1][-1]+200)])
legend();
plt.show()
```

<!-- #region id="ct89DYW57LQ0" -->
## Compute Jacobian Matrix to determine type of the Equilibria
<!-- #endregion -->

<!-- #region id="opkmu3srl58v" -->
The Jacobian matrix at the fixed point equilibria give the stability of the fixed points.
<!-- #endregion -->

<!-- #region id="7NUIyYji7K_f" -->
Let's consider the function **f**. You may recognize the AdEx equations. 

\begin{align*}
\textbf{f}(v_m,w) =& 
\begin{bmatrix} 
\frac{1}{C} \left(\Delta_T\cdot g_L \cdot e^{\frac{1}{\Delta_T} \left(- V_T + vm\right)} + I_{input} + g_L \left(E_L - v_m\right) - w\right) \\ 
\frac{1}{tau_w} \left(a \left(- E_L + v_m\right) - w\right)
\end{bmatrix}\\
\end{align*}

Then we have 
\begin{align*}
f_1(v_m,w) =& \frac{1}{C} \left(\Delta_T\cdot g_L\cdot e^{\frac{1}{\Delta_T} \left(- V_T + v_m\right)} + I_{input} + g_L \left(E_L - v_m\right) - w\right)\\
\end{align*}

and
\begin{align*}
f_2(v_m,w) =& \frac{1}{tau_w} \left(a \left(- E_L + v_m\right) - w\right)\\
\end{align*}

The Jacobian matrix of **f** is:
\begin{align*}
\textbf{J}_{\textbf{f}}(v_m,w) =& \begin{bmatrix} 
\frac{\partial f_1}{\partial v_m} & \frac{\partial }{\partial w}\\
\frac{\partial f_2}{\partial v_m} & \frac{\partial f_2}{\partial w}
\end{bmatrix}
 = \begin{bmatrix} 
\frac{\partial \left(\frac{1}{C} \left(\Delta_T\cdot g_L \cdot e^{\frac{1}{\Delta_T} \left(- V_T + v_m\right)} + I_{input} + g_L \left(E_L - v_m\right) - w\right)\right)}{\partial v_m} & \frac{\partial \left(\frac{1}{C} \left(\Delta_T\cdot g_L \cdot e^{\frac{1}{\Delta_T} \left(- V_T + vm\right)} + I_{input} + g_L \left(E_L - v_m\right) - w\right)\right)}{\partial w}\\
\frac{\partial \left(\frac{1}{tau_w} \left(a \left(- E_L + v_m\right) - w\right)\right)}{\partial v_m} & \frac{\partial \left(\frac{1}{tau_w} \left(a \left(- E_L + v_m\right) - w\right)\right)}{\partial w}
\end{bmatrix}\\
\end{align*}
<!-- #endregion -->

<!-- #region id="9rBPiysoS3-L" -->
### Install Symengine
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="aBR25GVrG9P_" executionInfo={"status": "ok", "timestamp": 1651221636714, "user_tz": -120, "elapsed": 7436, "user": {"displayName": "Anna-Maria K.", "userId": "15616814866073248468"}} outputId="0f1bfa27-ca3f-46af-bbb1-890d789705d7"
pip install symengine
```

<!-- #region id="NMBs9BDCIwUE" -->
## State Function for the Nullclines Symbolically 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 235} id="egAT9XDlIz61" executionInfo={"status": "error", "timestamp": 1651221636781, "user_tz": -120, "elapsed": 147, "user": {"displayName": "Anna-Maria K.", "userId": "15616814866073248468"}} outputId="201b5817-0095-4566-8b45-51a9d2e6f26f"
### Write equations in symbol format
fvm = vm_0 - w1
fw = w_0 - w1

### Find the function f
print('\n f(vm,w) = \n', np.matrix([fvm,fw]))
print('\n f1(vm,w) = \n', fvm)
print('\n f2(vm,w) = \n', fw)
```

<!-- #region id="4YJ1Hrd-I1-D" -->
## Compute the Jacobian
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 235} id="3SSc9hvRHREb" executionInfo={"status": "error", "timestamp": 1651221639329, "user_tz": -120, "elapsed": 266, "user": {"displayName": "Anna-Maria K.", "userId": "15616814866073248468"}} outputId="9e0332b0-0fad-4bc5-da70-1899ecaf1d2f"
def Jacobian(v_str, f_list):
    vars = sp.symbols(v_str)
    f = sp.sympify(f_list)
    J = sp.zeros(len(f),len(vars))
    for i, fi in enumerate(f):
        for j, s in enumerate(vars):
            J[i,j] = sp.diff(fi, s)
    return J

J = Jacobian('vm w', [fvm,fw])

print('\nJacobian Matrix:\n',J)
```

<!-- #region id="341SaaXTmKzb" -->
Take the 2x2 matrix of first derivatives at each fixed point (the Jacobian) and compute its eigenvalues:


*   Two negative eigenvalues at a fix point implies that the fix point is stable (trajectories starting from neighboring points converge to it).
*   Two positive eigenvalues indicates an unstable fix point (trajectories starting from neighboring points diverge from it).
*   One eigenvalue of each sign corresponds to a saddle point (trajectories from some neighboring points converge to it and others diverge).

<!-- #endregion -->

<!-- #region id="1PUyGgA4m2z9" -->
## Eigenvalue at each point



<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="dOhfEjQZmvE-" executionInfo={"status": "ok", "timestamp": 1631086484115, "user_tz": -120, "elapsed": 9, "user": {"displayName": "Mario Negrello", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gh9vnOzDqUD2QacfGPwd13jMPmwn1hzZzBnVURjO4E=s64", "userId": "10136788594790905986"}} outputId="462dd75f-2d3d-447e-db10-1aa7bf3ffcbb"
## Point 1
J1 = J.subs(vm1,intersec[0][0])
JJ1 = np.array([[float(J1[0]),float(J1[1])],[float(J1[2]),float(J1[3])]])
e1 = np.linalg.eigvals(JJ1)

print('\nEigenvalues point 1:\n',e1)

## Point 2
J2 = J.subs(vm1,intersec[1][0])
JJ2 = np.array([[float(J2[0]),float(J2[1])],[float(J2[2]),float(J2[3])]])
e2 = np.linalg.eigvals(JJ2)

print('\nEigenvalues point 2:\n',e2)
```

<!-- #region id="Gp37-MX6AdPy" -->
### What type of fix points do we have?


*   For point 1:
*   For point 2:


<!-- #endregion -->

<!-- #region id="fIIFQ_xzlEQn" -->
#License

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

Mario Negrello, Elias Santoro. 

<!-- #endregion -->
