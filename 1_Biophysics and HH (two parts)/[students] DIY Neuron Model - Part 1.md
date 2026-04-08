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
# To be able to edit and use this Notebook:

0. Learn about how to use google colaboratory [video](https://www.youtube.com/watch?v=inN8seMm7UI)
1. in the file menu (top left), click ```open in playground```
3. still in the file menu, click ```save copy in drive```, to make your own personalized and editable copy of this file.
4. edit as you like. If something breaks irreparably, either:
  1. restart the ```Runtime```
  2. or go back to step 1.

<!-- #endregion -->

<!-- #region id="43TskcthQiMg" -->
# Project 1: A DIY Neuron Model


<!-- #endregion -->

<!-- #region id="t4I9GWWuw77P" -->
### The Hodgkin Huxley Model:
The origin of the action potential was a mystery to early neuroscientists when Hodgkin and Huxley proposed their model in 1952. Aided by the voltage clamp technique, and the persuasion that they could model the physics of the neuron as an electrical circuit, they set out to understand the mechanism underlying the generation of the action potential. In their [concluding paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1392413/pdf/jphysiol01442-0106.pdf) in 1952, they formulated the mathematical framework most often used in computational neuroscience. Many fundamental building blocks in modelling excitable cells were laid down in this framework, which is of foundational importance to date.



<!-- #endregion -->

<!-- #region id="JWxy67mxS8iW" -->
<!-- ### Previous Project:
In the previous project, you have learned that:
- individual ions lead to individual **equilibrium potentials** or **Nernst** potential.
- when different ion concentrations exist inside and outside the cell, the cells settles in a ***resting potential***, which is given by the weighted average of the different conductances. -->

### In this Project:
You will synthetize the **Hodgkin Huxley model of action potential** piece by piece. We will be building understanding as we present the components of the model along with with some simulated experiments (using the 'brian2' simulator), sample code and simple coding exercises. We begin by describing the **passive properties** of the membrane (leak and capacitance) then introduce the **active** components and the **voltage gated** ion channels. In the second part of the project we will examine the completed Hodgkin Huxley model and conduct some experiements using the model. 

Questions during the project are designed as self-evaluation (for you to test your own comprehension). If you have doubts about particular answers, ask your neighbor, your TA or your professor (in that order).

### The landmarks in the road map:

1. (Project 0) The electrical potential across membrane is due to differences in chemical concentration.
2. The membrane potential is altered by flux of ions across the membrane via ion channels.
3. Membrane permeability to different ions varies as a function of the membrane potential (active channels).
4. Positive feedback creates an accelerating rise of positive currents due to sodium channels.
5. High membrane potential actives potassium channels (a negative feedback).
6. Spikes are due to the different time scales of ion permeability.
<!-- #endregion -->

<!-- #region id="dibGcKwuhV6O" -->
# Key Terms
<!-- #endregion -->

<!-- #region id="GQN4DZfLhaXV" -->

- **Permeability**. How easy it is for an ion to cross the membrane. Plastic is 'impermeable to water'. Expressed as a propotionality. It is proportional to the number of open channels on the membrane for an ion. Commonly expressed as relative permeability of the ion channels compared to one another.
- **Current, _I_**. A flow of electric charge. Measured in amperes ($\mathbf{A}$). Commonly seen also as a current density ($\mu A/cm^2$)
- **Capacitance, _C_**. How well a material can *store* charge. Measured in farads ($\mathbf{F}$).
- **Resistance, _R_**. A measure of the opposition to the flow of charge (current).  Measured in ohms ($\mathbf{Ω}$).
- **Conductance, _g_**. The reciprocal of resistance (1/R), measures how easily the electricity flows through parts of the circuit for a given difference in voltage. Is an electrical property and is influenced by permeability. It is measured in Siemens (‎$S‎ = Ω^{-1} $). It often appears as a density ($mS/cm^2$).
- **Ion Channels**. Pores through the membrane that allow for selective permeability of different ion types. There are 'active' and 'passive' ion channels. Active channels change their conductance as a function of, for example, membrane potential. The conductance of passive channels is constant.
- **Voltage Clamp**. A method that creates a constant potential difference across the membrane and measures the current flowing through it.
- **Activation Gates**. Variables that represent the proportion of channels in a certain state (open or closed).
- **Depolarization**. A positive change in the membrane voltage (cell becomes more positive).
- **Hyperpolarization**. A negative change in the membrane voltage (cell becomes more negative).


<!-- #endregion -->

<!-- #region id="3aT0aRHn8eON" -->
# Learning Objectives
<!-- #endregion -->

<!-- #region id="eaPl2oV1088V" -->
After this project you'll be able to:

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
<!-- #endregion -->

<!-- #region id="509DzcGxeXcI" -->
# Pre-requisites
<!-- #endregion -->

<!-- #region id="0-j_Dp8zeZ_R" -->
Before you go through the project watch the videos suggested below. If you find a better video, you can contribute with the whole "computing brain" crowd by suggesting links to be uploaded to the [computing brain youtube channel](https://www.youtube.com/channel/UCU2BRdfg49st7ZdFMZbDScg)!


- How to measure neurons via clamps (current clamp, voltage clamp) [video](https://www.youtube.com/watch?v=mVbkSD5FHOw&t=4s)
- Ions and electrical charge (Ohm's law) : [video](https://www.youtube.com/watch?v=G3H5lKoWPpY) or [video](https://www.youtube.com/watch?v=fGI9d0CjI8s)
- Capacitance :  [video](https://youtu.be/f_MZNsEqyQw) or [video](https://www.youtube.com/watch?v=u-jigaMJT10)
- Basic understanding of the electrochemical gradient [video](https://www.youtube.com/watch?v=Ba02v7eoVWQ).
  
<!-- #endregion -->

<!-- #region id="Vz3qG1oghAmR" -->
# Initialization Code
<!-- #endregion -->

<!-- #region id="hh25znGpgZNN" -->
- In the code cell below we install the simulator [Brian2](https://brian2.readthedocs.io/) and import relevant python.modules.
- It installs and imports Brian2, the simulator we will be using.
- **Note that you have to run this every time that colab 'disconnects' from the kernel.**
- Documentation of brian2 can be found [here](https://brian2.readthedocs.io/en/stable/user/index.html)
- Many errors will be related to your units
- You can safely ignore Warning messages. 
- If you don't know what to do anymore, restart the "Runtime" in colab.
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

```python id="88Qmb5nEPYq6" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629798014692, "user_tz": -120, "elapsed": 16150, "user": {"displayName": "Aaron Benson Wong", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiHL2W1r23pEiPL77ZG_up8Kld-3_uSnF5MnlHqFw=s64", "userId": "12009993929995728596"}} outputId="3c2dfb13-bf74-405a-a8a3-ddcf06577252"
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

<!-- #region id="k-DV4AGMZbMA" -->
# 1. The Passive Membrane
<!-- #endregion -->

<!-- #region id="ZBuIHB-3mpV3" -->
## Differences of Potential Due to Differences in Ion Concentration
<!-- #endregion -->

<!-- #region id="blOekBtwiixa" -->
As we learned in the previous project, fatty membrane separating the outside and the inside. Insert some  negatively charged ions and molecules inside (such as Chloride or other anionic proteins). If this compartment is placed in a more positively charged environment, there's a negative difference of potential across the membrane.

**The membrane is not perfectly permeable**. There are ion channels in the membrane through which ions can cross. Nature likes an equilibrium, and in principle ions would flow out of the membrane until charge balance is achieved.  However, some ions are too large to get across the membrane, thus they get stuck inside, keeping that charge in the cell. Many proteins inside the cell fit that description, and that makes it so that the inside of the cell is negatively charged. How negative? That's a function of how much charge per volume (charge density) there is. The thing to remember is that even if all channels in the membrane are open, the cell would still retain considerable negative charge.

The smaller the ion, the more charge can be packed in a volume. Also, in general there are more ions than ion channels, and so it takes time until these ions can find a channel to leak through, until it comes back to the resting membrane potential.

We can calculate this rate of decay via the **passive properties** of the membrane, the **capacitance** and the **leak conductance**.

<!-- #endregion -->

<!-- #region id="FGny2Rr0XA1Z" -->
### The Driving Force

<!-- #endregion -->

<!-- #region id="mryNTXlWmuB6" -->
The same way as a ball on a hill has potential energy proportional to the height of the hill, the passive membrane that is taken away from $V_{rest}$ has a potential called the **driving force**.


---


$$V - V_{rest}$$


---


The farther from $V_{rest}$, the higher the potential, and the willingness of the potential to return to its equilibrium state (usually between -50 to -70mV). Take note of the sign change for when V is smaller than or larger than V_rest. This means that the force *changes direction* depending on the membrane potential of the cell. Note that the driving force pulls the neurons toward their equilibrium potential.

Two parameters determine the rate with which the membrane returns to the resting potential, the **membrane capacitance** ($C_{membrane}$, measured in Farads) and the **leak resistance** ($R_{leak}$, measured in Siemens, the reciprocal of Ohm). While the leak resistance relates to the number of pores for ions to cross (only those that are permeable), capacitance is equivalent to the amount of charge that the cell membrane can store and is relatively constant for most cells.
<!-- #endregion -->

<!-- #region id="iJO_7MnOZszM" -->
### Charge Loss Due to Leak Conductance and Capacitance


<!-- #endregion -->

<!-- #region id="7YlrRuxzm8zE" -->
  When ions cross the membrane through pores (ion channels) we speak of a **current leak**. The **leak conductance** is a measure of how permeable the membrane is to ions, or, in other words, the number of **passive ion channels** and how many ions can cross over time (recall that charge flow is current).
 
The membrane potential change per unit time is represented by ($\frac{dv}{dt}$), or equivalently $\dot{V}$, and is an ordinary differential equation. In the absence of any other influences, $V$ tends to $V_{rest}$ (in millivolts (mV)), with a rate determined by the **leak conductance** ($g_{leak}$), which is a constant value, and therefore a **"passive conductance"**. Conductance is a measure of the number of leak channels available for the crossing of ions. 

---


$$ \dot V = - \frac{1}{R_{leak}}(V-V_{rest}) $$

is the same as 

$$ \Leftrightarrow \dot{V} = - g_{leak}(V-V_{rest})$$

---


Through passive ion channels, the charge leaks out of the neuron until the electrochemical equilibrium, i.e., the resting potential is reached. That is, the driving force gets smaller and smaller until zero. The conductance determines the rate of the return to equilbrium (resting potential).

The negative sign explains that the direction of potential change is opposite to the driving force. That is, if we have a net positive driving force, we have a negative rate of change.

<!-- #endregion -->

<!-- #region id="1KKBWPm08jPG" -->
Furthermore, cell membranes have [**capacitance**](https://www.youtube.com/watch?v=G3H5lKoWPpY), which is a measurement of how much charge it can store (review the video on capacitance in pre-requisites). The thicker a membrane, the lower the membrane capacitance $C_m$. **The capacitance dampens a current flow**, reducing the rate of change of the membrane potential. The lower the capacitance, the faster the potential difference comes to the resting potential. 
 
---
From charge conservation considerations we know that if the total current flowing through the circuit is equal to the sum of the capacitive and the resistive currents. 

$$I_{injected} = I_c + I_{leak}$$

If there is no net current ($I_{injected} = 0$), the capacitive current flowing in the cell is equal to the leak current, with an inverted sign.

$$0 = I_c + I_{leak}$$
$$I_c = - I_{leak}$$

If we spell out the currents , we have that:

$$C_{m} \dot{V}  =  I_{injected} - g_{leak}(V-V_{rest}) $$

and therefore:

$$\dot{V} =  \frac{I_{injected} - g_{leak}(V-V_{rest})}{C_{m} }$$


or by using leak resistance instead of conductance,


$$\dot{V} =  \frac{I_{injected}}{C_{m}} - \frac{(V-V_{rest})}{R_{leak} C_{m} }$$

---

The denominator of the second equation $R_{leak}C_m$ is often substituted by $\tau_m$, the so called **time constant** for reasons that will become clear below.

Notice also that injected current creates voltage deflections with the same sign, whereas the leak induces a negative rate (i.e., returning to rest).

<!-- #endregion -->

<!-- #region id="AcEZHGCicWjD" -->
## Interactive: Visualizing the Time Constant via Current Injection into the Passive Membrane
<!-- #endregion -->

<!-- #region id="grA6VfBtizT6" -->
How fast do we get to the resting membrane potential if we only have passive elements in the membrane (leak ion channels and capacitance)?

To see how capacitance and resistance influence the passive decay time of the membrane potential back to resting potential, we create a live widget where we can give a voltage step ('clamp') and see how changes of $C_m$ and $g_{leak}$ change the slope of the decay. 

**In the code below, pay particular attention to the equation being computed.**
<!-- #endregion -->

```python id="v3yKjghdZfGX" colab={"base_uri": "https://localhost:8080/", "height": 375, "referenced_widgets": ["251d710bf3614e8592381f6dfe06c670", "321a204788cf4fb2ab445b6ff07a6a98", "d01d6917365b40a8b1996247e5a0f685", "95c45be70bd04d3d9d0910da27176fa9", "1efe81d2c7c74273930fc4ea1c245dc7", "5985a9a9391d47478ee028d90c037fbf", "a92fb5d19efb4bcf95a6a719d7c65b6b", "2d768f6ca0ae47a6a7ac004721bba854", "1db94a682e5f4af1bed10eec701f2ab6", "8f71d1ef92c84f008170fa304cefdc53", "517b8824a791479ea20e64b40ffd7adc", "e0d40d8b49d5446bb7d80a77037dadc9", "499b50aa18ad4927982be69322715b1c"]} executionInfo={"status": "ok", "timestamp": 1629798026332, "user_tz": -120, "elapsed": 5905, "user": {"displayName": "Aaron Benson Wong", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiHL2W1r23pEiPL77ZG_up8Kld-3_uSnF5MnlHqFw=s64", "userId": "12009993929995728596"}} outputId="e6f3c86d-d825-4ab4-d001-b312c97f6ca5"
def PassiveMembrane(C_m, g_leak, I):
    start_scope()
    # this is a function that computes the membrane potential over time 
    # for a given capacitance and leak conductance

    E_leak = -54.5 * mV            # We define the reversal potential as a constant (typical value for HH)
    dtt = 0.025*msecond            # simulation parameters (integration time)
    g_leak = g_leak* mS/cm**2     # attribute unit [mS/cm^2] to function argument g_leak
    Cm = C_m * uF/cm**2           # attribute unit [pF] to function argument C_m

    # Here we define our model of the passive membrane in Brian2
    # note 1: for Brian, the equations are defined as a string, 
    #         that's why you see the equation surrounded by ''' 
    # note 2: the unit of the variable is mandatory and specified
    #          in the string. It is given as a 'base unit' of the ISU.
    # https://en.wikipedia.org/wiki/International_System_of_Units


    eqs ='''
    I_leak =  g_leak * (v - E_leak)   : ampere*meter**-2 
    dv/dt = I/Cm - I_leak/Cm          : volt
    I                                 : ampere*meter**-2 
    '''
    

    # with Brian2 we create a neuron group G with 1 neuron and equations defined above
    # note: we use 'euler' method for integrating.
    passive_membrane = NeuronGroup(1, eqs, 'euler') 
    
    # we introduce a StateMonitor in G to record 'v'
    M = StateMonitor(passive_membrane, ['v' , 'I'], record=0)
    
    # we set an initial value for membrane potential (G.v) 
    passive_membrane.v = E_leak # initial condition
    
    # we run the model (e.g., integrate the differential equations via Brian) for 50ms without input.
    run(50*ms)
    
    # CURRENT INJECTION (OR CURRENT CLAMP):
    # here we inject current
    passive_membrane.I = I*mA/cm**2 # take voltage to v_pulse 
    
    # and let the membrane potential 'relax', that is, 
    # we run 50ms more of simulation, to see the difference in decay time
    # due to the change of voltage.
    run(50*ms)

    passive_membrane.I = 0*mA/cm**2 # take voltage to v_pulse 
    run(50*ms)

    # finally, the function Mem_func is able to plot its output.
    axes = plt.gca()
    plt.plot(M.t/ms, M.v[0]/mV)
    plt.xlabel('Time (ms)')
    plt.ylabel('V (mV)')
    
    return E_leak

# make the function interactive
# note: we have to use numpy floats for continuous variables

w = interactive(PassiveMembrane, g_leak=(0.0,1.0), C_m=(0.1,10.0), I=(-1., 1));

#w.children
display(w)

```

<!-- #region id="DHYeM4GDU3XC" -->
## Questions:
<!-- #endregion -->

<!-- #region id="YxYPNj3Ni56B" -->
0. Describe to yourself what you see in the plot above. Relate that to the code.

1. A current pulse is injected into the membrane. What is its duration? What is its unit?

1. What is the **time scale** of the decay due to passive properties of the membrane? 

2. What is the influence of capacitance on decay time to resting potential? What is the influence on voltage rise time?

3. What is the relationship between leak conductance and capacitance on decay dynamics?

4. What is the **Driving Force**? What direction does it point towards?

5. What happens if leak conductance is set to zero?

7. Did you notice that the unit of the membrane potential is not a density ( divided by an area)? Explain why via (1) the equations, and (2) intuitively.

<!-- #endregion -->

<!-- #region id="HkJrdvXaMn7h" -->
### Answers:

<!-- #endregion -->

<!-- #region id="uQxyPpy5fCQF" -->
0. A simulation of the response of the passive membrane for a current injection, depending on the resistance and capacitance of the membrane. 
1. A passive membrane has a resting potential of -70mV. At about 50ms, it is momentarily 'clamped' to a higher voltage (-45mV). After release of the holding potential, the membrane potential returns to resting value.

2. Judging by the time for the voltage to decay to baseline in the plots above, the time scale of the decay is about 10's of milliseconds.

3. The larger the capacitance, the slower the decay. This makes sense, as capacitance is a measure of 'charge storage capacity'.

4. They are reciprocal of each other.

5. The Driving force is (Vpulse - Vleak). Positive if Vpulse is larger than Vleak. The direction is then the opposite of the voltage deflection.

6. After the step, the membrane potential no longer changes, because current is not leaking passively through the leak channels. The entire charge delivered by the current is kept.

7. The membrane potential is a property of the entire cell. While the densities cancel out for capacitance and conductance, giving area to voltage would result on unit discrepancy.
<!-- #endregion -->

<!-- #region id="cQ3YHcsAP6iV" -->
## Units of Conductance: mS or mS/cm^2?
<!-- #endregion -->

<!-- #region id="UWbCsdGLQHrf" -->

**The base unit of conductance** is the $S$, the *Siemens*, the reciprocal of resistance, given in *Ohm* ($\Omega^{-1} = S$). The leak **conductance** reflects the number of leak channels in the membrane (i.e., pores). 

A single ion channel, will have a certain conductance (called 'unitary' conductance, because it is a single ion channel). The membrane has many of these channels per unit area, so when describing the channel conductance of the neuronal membrane we often use 'density'. The ion channel conductances per unit of membrane area is often given as milisiemens per centimeter square ($mS/cm^2$). 

In this project we use *conductance density* and also *current density* (often in $mA/cm^2$). Other models can use *total conductance* for the entire cell, or *total current injected* in which case we are effectively multiplying a density times a membrane area. While the later value is largely invariant to cell size, the former choice is very dependent on the cell's surface area. Therefore, when we measure conductance in experiment we often prefer to use express conductances as densities.

While this may seem like a digression, a *modeler has to be very aware of his choices for units* because cell dynamics are very sensitive to values. This is also the reason why we selected Brian as our neural simulator, as it forces us to make unit choices explicit.

Henceforth, every time you see a value, stare at the unit for a moment and try to visualize what it is measuring. Attend to scales!
<!-- #endregion -->

<!-- #region id="Sm0WcCxDfYX3" -->
# 2. The Excitable Membrane
<!-- #endregion -->

<!-- #region id="okOtlv5znlu_" -->
### Active Ion Channels
<!-- #endregion -->

<!-- #region id="fm1jDFA-jHf1" -->

**Active Channels are those where ion permeability changes as a function of something, usually voltage** or some chemical. In contrast, passive channels such as the **leak** have constant conductance. Active essentially means that the state of the channels changes as a function of some other variable, such as the membrane potential. 

Ion channels come in grate variety depending on ion selectivity, whether their activation is voltage dependent, or on whether they depend on binding to a particular chemical (ligand gated channels).

We begin our exposition with active (but simple) ion channels whose activity is dependent on the membrane potential.

<!-- #endregion -->

<!-- #region id="E-WavOTajAmE" -->
### Ion Selectivity 
<!-- #endregion -->

<!-- #region id="0leWqWhNjQLn" -->
Channels can be more or less **selective** for particular ions, and we generally name the current that is due to a certain ion channel with the name of it's preferred ion. For example, Sodium ions go through the **Sodium channel** and become the **Sodium Current**. 

Other channel types exist that are less selective, and have currents  due to multiple kinds of ions, with prominent examples being the Sodium/Potassium pump and the **NMDA** receptor, which exchanges both Calcium and Magnesium. There are also Chloride channels and many others. We focus on the ones that are crucial for the production of action potentials.
<!-- #endregion -->

<!-- #region id="UG9QIPlWjDPB" -->
### Currents
<!-- #endregion -->

<!-- #region id="lcFRw1nRjM8p" -->
The current that crosses the membrane is a function of the number of permeable ion channels available at the membrane at a given time. For example, **the Sodium current ($I_{Na}$) is the total amount of current that is due to the flow of Sodium ions through the sodium channels**. The current is determined by the multiplication between the **maximum conductance**, the **gating variables** representing the proportion of open ($m$) and active ($h$) chanels and the **driving force** ($V-V_{rest}$).


The following paragraphs define **gating variables**, their effect on permeability and how they change over time, i.e., their **dynamics**.

For an explanation on single channels [see this video](https://youtu.be/lpkaXwtAt7E)
<!-- #endregion -->

<!-- #region id="mYxdMOQ_kD49" -->
### Gating Variables


<!-- #endregion -->

<!-- #region id="WDLD1AfHkpPA" -->
Installed in the membrane are large numbers of active ion channels (varying from tens to tens of thousands). Many ion channels are rather selective. Ion channels can be open or closed state. The ion channels are statistically independent of each other. 

From these assumptions, we can represent the state of active ion channels via **gating variables, which represent the proportion of channels that are open or closed** (or sometimes 'activated and inactivated'). As they are a **proportion** of open channels of all available channels, they take values from 0 (no channels opened) to 1 (all channels opened).

(If we have 100,000 channels, and 80,000 are currently open, then gating variable $x$ has value 0.8, that is 80% of the channels are open).

Each channel opens and closes probabilistically and independently ([see explanation on [single channel patch clamp](our video). The probability of opening and closing is a function of the context, which could be, for example, the momentary membrane potential of the cell. When there are large numbers of channels, this probability of the single channel can be taken as the proportion of channels open. For example, for a given membrane potential $V$, the **proportion of  channels in an open state** is given by a function of activation gate $m_{\infty}$ (a **gating variable**):

<img src="https://i.postimg.cc/h4TDpRN2/m-inf.png" width="350x">

In this figure we have the experimentally measured proportion of open channels, i.e., the *steady state* of the **gating variable** (the error bars represent the outcoume of multiple experiments).
<!-- #endregion -->

<!-- #region id="tCko7TGRbxoq" -->
#### Quiz Questions:

1. Analyze the graph above and determine what is the potential at which all channels are closed?
1. What is the difference between an active and a passive ion channel?
1. What is a gating variable?
1. What is selectivity?
<!-- #endregion -->

<!-- #region id="XrgFWFOsWoiG" -->
### Activation and Inactivation Gates 


<!-- #endregion -->

<!-- #region id="qXtr1Iy6o3N9" -->
Note that there can be more than one gating variable per channel type, depending on the mechanisms of the ion channel. For example, the Sodium ion channel has two gates, the **activation gate** $m$ and the **inactivation gate** $h$. A function that determines the activation gate is called an activation function.

The flow of sodium through all channels (permeability)  is dependent on both these gating variables. As m and h represent proportion, they can also be thought of as probabilities. Hence, the probability of an ion channel being permeable is their multiplication.

----

$$p = m^a h^b$$

----

The exponents $a$ and $b$ are also related to the mechanical action of the ion channel, essentially expressing the number of  activation gates (a) and inactivation gates (b) (for a more in-depth explanation consult chapter 2 of Dynamical Systems in neuroscience, or Aaron).
<!-- #endregion -->

<!-- #region id="2XziB8_HXz6i" -->
### Sample Code: Plotting a function with python
<!-- #endregion -->

<!-- #region id="sqVTvIy_1I_K" -->
An arbitrary **activation function** $x$ takes from 0 to 1 as a function of V. It is commonly represented by a  [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function) such as the one below. The sigmoid is chosen because it is a function that is bounded, symmetric and smooth.

---

$$x_{\infty}(V) = \frac{1}{1+e^{(V_{1/2}-V)/k}}$$

---
This type of function is often used to model the **steady state** (represented by the $\infty$ subscript) of the activation gate, because it is symmetrical, rises smoothly and can be parameterized to match experimental data.

The function takes two parameters, $V_{1/2}$ and $k$ ($V_{1/2}$ defines the position of the midpoint of the sigmoid, and k defines the slope, see figure). These parameters are obtained experimentally via patch clamp experiments. For example, the activation variable $m_{\infty}$ of the squid axon, we have that $V_{1/2} = -40, k = 15$.


The parameters determine the shape of the sigmoid function, as such:

[![boltzmann-function.png](https://i.postimg.cc/Tw2b5MsG/boltzmann-function.png)](https://postimg.cc/R38h289b)

The python code below defines and plots a mathematical function, in this case, the sigmoid (also called a Boltzmann function).


<!-- #endregion -->

```python id="8FuejXdS1Q4x" colab={"base_uri": "https://localhost:8080/", "height": 71} executionInfo={"status": "ok", "timestamp": 1619004532212, "user_tz": -120, "elapsed": 4644, "user": {"displayName": "Mario Negrello", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gh9vnOzDqUD2QacfGPwd13jMPmwn1hzZzBnVURjO4E=s64", "userId": "10136788594790905986"}} outputId="89f8857d-005b-4676-88d2-c2942633db63"
# define a function m_inf that returns the
# steady state of m gate for any value of the potential V.

def sigmoid(v_at_half, k, V):
    
    #np.exp means we are using the function exp from the package numpy
    return (1./(1.+np.exp((v_at_half-V)/k)))

# define a voltage range for calculating the function
v = np.arange(-100.0, 0.0, 1.) # from -100 to 0mV at steps of 1mV

m_inf = sigmoid(-40,15, v)

plt.figure(1) # create a figure

plt.plot(v, m_inf) # plot the range x vs the function tau(x) domain 
plt.xlabel(r'$mV$') # add labels to the plot
plt.ylabel(r'$m_{\infty}(V)$') 

```

<!-- #region id="FvWE2n1KWnGM" -->
##### Quiz Question:

- Why does a sigmoid shape is well suited to represent gating variables?

<!-- #endregion -->

<!-- #region id="ZOD6theSV4tl" -->
### Activation Time Constants
<!-- #endregion -->

<!-- #region id="JYlMJ9tZ1Rb1" -->
When the membrane potential changes, the opening and closing of channels **is not instantaneous**. The gating variable tends to the steady state that is dependent on  voltage (see plot under gating variables) with a rate inversely proportional to $\tau_x$, which is also voltage dependent. In other words, to determine this rate of change (from open to close or vice-versa) we can measure $\tau_x$. 
Note that $\tau_x$ is the time constant for the rate of change of gating variables and is different from the membrane time constant.

![tau-m.png](https://i.postimg.cc/FFZq9PS9/tau-m.png)

The function that determines the rate of change of m is given by

---

$\tau_x(V) = C_{base} + C_{amp}  e^\frac{-(V_{max}-V)^2}{\sigma^2}$

---

<!-- #endregion -->

<!-- #region id="7XRTzA4N_J1W" -->
#### Code Exercise: Plot the function for $\tau_m(V)$

Using the sample code above, plot the function defined above.

Where $C_{base}, C_{amp}, V_{max}, \sigma$ are given parameters (see below).

[![gaussian.png](https://i.postimg.cc/XJFf9SKz/gaussian.png)](https://postimg.cc/JDrHWF2N)
<!-- #endregion -->

<!-- #region id="IZ3Eo8B0cuUJ" -->
### Your Code
<!-- #endregion -->

```python id="M7HGg_xhcwys"
# Your solution goes below

```

<!-- #region id="VTijklfBDwWx" -->
### Maximal conductance ($\bar{g}$)

<!-- #endregion -->

<!-- #region id="HipY8FZreHhh" -->
While the gating variable stands for the proportion of channels open, the **maximum conductance** $\bar{g}_{some\ channel}$ represents the total number of channels available of a given type. We read $\bar{g}$ as 'g bar'. As the channels can be in different states, the actual conductance is a product of the gating variable and of the maximum conductance.

---
$$g_{channel} = \bar{g}_{channel} x$$

---


Where x is the activation gate, that indicates the proportion of channels letting ions through.

Active ion channels change their conductance as a function of other variables, such as the momentary membrane potential. For the sodium ion channel for example, conductance is low when the cell is negative and high when the cell is depolarized.


<!-- #endregion -->

<!-- #region id="Acm24dTE3Azl" -->
### Permeability of the Sodium Ion Channel 
<!-- #endregion -->

<!-- #region id="8fscJsLleBfZ" -->
Now we can write the  conductance of sodium ion as a function of its activation and inactivation gates m and h:

$$
g_{Na}=\bar{g}_{Na}m^3h 
$$

Where $\bar{g}_{Na}$ is the maxiumum conductance, $m$ is an activation gate, and $h$ is an inactivation gate. Essentially, if all channels are open $(m=1)$ and de-inactivated $(h=1)$, we have that the conductance is maximal.


The variables in this equation can be interpreted. *m* is the probability of the activation gate to be open,  *h* is the probability of the inactivation gate to be 'de-inactivated', that is 'available'. Their powers, $a=3$ and $b=1$, respectively are the degrees of freedom of (in)activation gates in the Sodium channel. This figure represents the situation for the Sodium channel:

[![sodium-channel.png](https://i.postimg.cc/Gh7HXSR9/sodium-channel.png)](https://postimg.cc/cr3xHhrN)

<!-- #endregion -->

<!-- #region id="iOKY4vmKgAnB" -->
### General model for conductances based currents:

The vast majority of voltage-gated ion channel are modeled similarly (like potassium for example, see below). In general, the activation and inactivation variables ( $m$ and $h$) are raised to powers $a$ and $b$ respectively. The exponents appear through fitting procedures, but can be taken to mean the "degrees of freedom" of the channel (how many distinct conformations that make the channel permeable). 

$$
g_i = \bar{g_i} m^a h^b 
$$

<!-- #endregion -->

<!-- #region id="NRRSO6e5u--8" -->
#### Quiz Question:
- What is the difference between the membrane time constant and the time constant for the ion channel gating variables?
- How does inactivated channels differ from closed channels? 

<!-- #endregion -->

<!-- #region id="s0ndCk70kZCd" -->
### Dynamics of Sodium gating variables
<!-- #endregion -->

<!-- #region id="bDQ_hIwCkexz" -->
Gating variables in general take values from 0 (impermeable) to 1 (permeable). Changes in the membrane potential lead to changes in the proportion of channels open or inactivated. 


- partially activated $(0<m<1)$
- completely activated $(m=1)$
- deactivated $(m=0)$
- inactivated $(h=0)$
- deinactivated$(h=1)$

The change in state (for m or h) is given by a differential equation that is a function of $V$. Note that it is very similar to the one for the dynamics of the membrane potential of passive membranes.

---

$$
\dot m = dm/dt = \frac{m_\infty(V) - m}{\tau_m(V)}
$$

---

where $m_{\infty}$ is the steady state  activation or inactivation (the equilibrium) and $\tau(V)$ is the activation or inactivation time constant (a measure of the time to reach a steady state, or the speed). 

Think of $m_\infty$ as the value that is inevitably obtained as we keep the membrane potential fixed for a very long times (i.e., infinity). 

$k$ is just a factor that determines the steepness of the slope, the smaller $k$ is, the steeper $m_\infty(V)$ is.

$h$ is described by an equivalent equation:

---

$$
\dot h = dh/dt =  \frac{h_\infty(V) - h}{\tau_h(V)}
$$

---



<!-- #endregion -->

<!-- #region id="FkMjEMYvW25u" -->
#### Exercises:

1. Below you encounter parameters for the steady state and time constant of the different ion channel gating functions. These are the specific parameters according to Hodgkin and Huxley for the squid axon. Plot the steady state activation functions and time constants for each of the ions!


| gating function | $V_{\frac{1}{2}}$ | k | time constant ($\tau_{gate}$) | $V_{max}$ | $\sigma$ | $C_{base}$ | $C_{amp}$ |
|-|-|-|-|-|-|-|-|
| $m_\infty(V)$ | -40 | 9 |$\tau_m(V)$|-38     |30| 0.04 |0.46|
| $h_\infty(V)$ | -62 | -7 |$\tau_h(V)$| -67   |20| 1.2  |7.4|

<!-- | $n_\infty(V)$ | -53 | 15 |$\tau_n(V)$|-79 |50|1.1  |4.7| -->

**Important Note**: Hodgkin and Huxley in their original paper summed +65mV to the variables, so that the so that their resting potential would appear to be at 0mV. Here we use the value for the actual reversal potential of about -65mV. 
<!-- #endregion -->

```python id="eGeuWokMc9mK"
# your code

```

<!-- #region id="oNhiP9htlxOj" -->
#### Comprehension Questions:
<!-- #endregion -->

<!-- #region id="fQZQVD0H8ogW" -->
0. What is the unit for the steady state activation variables?
1. What is the activation variable that changes the fastest?
2. What is the status of the m and h gates for low membrane potentials?
3. The curves for h and m appear reflected across the vertical axes. What does that mean?
4. Where is the rate of change of activation variables the fastest? Why?
<!-- #endregion -->

<!-- #region id="4BWAHEX-7eao" -->
#### Check your Answers
<!-- #endregion -->

<!-- #region id="fhJgPLAh8siB" -->
Make sure you can produce a reason for these answers, else ask your lecturers!

0. Unitless.
1. 'm'.
2. m = 0 (closed), h = 1 (de-inactivated).
3. That they do the opposite of each other.
4. They are the fastest closest to the activation value of 0.5. This is because that is the value at which the function has the maximum slope.
<!-- #endregion -->

<!-- #region id="gRt4YQICWLix" -->
### Interactive: Dynamics of Activation Variables

In this example we will plot the dynamics (change over time) of **INDIVIDUAL** activation variables (m and h) after a change of membrane potential. **For the moment we assume that the membrane has no leak and capacitance is very small**, so that we can isolate the effect of the gating variables as a function of voltage. We assume an initial membrane potential of V=-80mV is held for 100ms, and then instantaneously changed it to a new holding potential of V=0mV. What you observe is the m and h variable tending to their steady potential values. 
<!-- #endregion -->

```python id="WTO0vzMiKvae"
start_scope()

def ActivationGates(V):
    # this is a function that computes the membrane potential over time 
    # for a given capacitance and leak conductance

  eqs_activation= '''
    m_inf =  1/(1+exp((-40*mV -v)/(9*mV))) : 1
    h_inf =  1/(1+exp((-62*mV -v)/(-7*mV))) : 1
    taum  =  .46*ms + .04*exp(-(-38*mV-v)**2/(30*mV)**2) *ms : second
    tauh  =  7.4*ms + 1.2*exp(-(-67*mV-v)**2/(20*mV)**2) *ms : second
    dm/dt = (m_inf - m)/taum : 1
    dh/dt = (h_inf - h)/tauh : 1
    v : volt
    '''
  # with Brian2 we create a neuron group G with 1 neuron and equations defined above
  N = NeuronGroup(1, eqs_activation, method='euler', dt=0.025*ms)

  # we introduce a StateMonitor
  M = StateMonitor(N, ['v','m','h'], record=0)
    
  # we set an initial value for membrane potential
  N.v = -90*mV # initial condition
  
  #we run the model for 50ms without input
  run(50*ms)

  # VOLTAGE CLAMP:
  # here we set the initial voltage value (we 'clamp the voltage to a constant', also called, 'holding potential')
  N.v = V*mV # voltage clamp 
  
  # and let the membrane potential 'relax', that is, 
  # we run 50ms more of simulation, to see the difference in decay time
  # due to the change of voltage.
  run(50*ms)

  # finally, the function Mem_func is able to plot its output.
  axes = plt.gca()

  plt.plot(M.t/ms, M.m[0])
  plt.plot(M.t/ms, M.h[0])
  plt.xlabel('time (ms)')
  plt.ylabel('activation')
  plt.legend(['m' , 'h'])

```

```python id="A-8A50z4kwkl" colab={"base_uri": "https://localhost:8080/", "height": 311, "referenced_widgets": ["9c84450566224ad68e82c658f570257b", "634ff63c338c4ea588d1708994cb5569", "093c90468e824fcd9093b9c78eba69bc", "03fd8fec9ec74ed5bc54bc4419230e9e", "6dfc8b3e26f54d17bb8b630d89dabfad", "ee2d2b7f9e3741138a70516d7bbd2b52", "298bbb7718e54d8899f6487d949cd690"]} executionInfo={"status": "ok", "timestamp": 1626449258377, "user_tz": -120, "elapsed": 5042, "user": {"displayName": "Karlijn Hers", "photoUrl": "", "userId": "13078358567483151064"}} outputId="094f9c33-8338-4aa6-8662-26d0d22739f8"
w = interactive(ActivationGates, V=(-100.0,10.0))
display(w)

```

<!-- #region id="w86XgppQ8aqx" -->
#### Comprehension Questions:

<!-- #endregion -->

<!-- #region id="beTO3cf6mUsf" -->
Experiment with the slider above and reason about the following questions:

1. At the initial stage, why does h grow and m does not?
2. Why do m and h have opposite behaviors after the membrane potential step?
3. Which variable takes the longest to reach steady state?
<!-- #endregion -->

<!-- #region id="RxyjJCJnny2p" -->
## The Sodium Current

<!-- #endregion -->

<!-- #region id="hD0bp92E88aS" -->
Now we have all the ingredients to understand how (1) gating variables, (2) conductance and (3) driving force becomes a **current**. We begin our exploration with the Sodium current.

For the Sodium current to flow into the cell, three factors must obtain:
1. There must be a driving force for the current to enter the cell $(V-E_{Na}) \neq 0$;
2. The Sodium channels must be open (m>0);
3. The Sodium innactivation gate must be letting ions through(h>0).

The current due to Sodium is then determined by the driving force for the ion, **times** the maximum conductance for the channel (representing the total number of available channels times the conductance of a single channel) **times** the proportion of channels open **times** how many channels are _not innactive_.

For Sodium, this is the equation that determines the current that flows in the cell:

------------

$I_{Na} = \bar{g}_{Na} m ^3 h (V-E_{Na})$

------------

Note the exponent at $m$. Its existence is related to the number of independent subunits that have to be open for the ion channel to let ion channels through.
<!-- #endregion -->

<!-- #region id="vixMjhGVdGJh" -->
### Simulating a Membrane with the Sodium Current
<!-- #endregion -->

<!-- #region id="Q9IqAWQkKuvT" -->
What would happen to the membrane potential if there were **only** Sodium channels in the membrane? Let us assume that, for the moment:
- No leak conductance.
- No other channels.

<!-- #endregion -->

```python id="LSmZZqgzf0PL"
start_scope()

# Reversal Potentials
E_Na = 55*mV

# Conductances
g_Na =  120 * mS / cm**2

# Membrane Capacitance
Cm = 1 * uF / cm**2

# define our differential equations
eqs_V ='''
dv/dt = (I - I_Na )/Cm : volt
'''

eqs_I = '''
I_Na = g_Na*(m*m*m)*h*(v-E_Na) : amp / meter ** 2
I : amp / meter ** 2
'''

eqs_activation= '''
m_inf =  1/(1+exp((-40*mV -v)/(9*mV))) : 1
h_inf =  1/(1+exp((-68*mV -v)/(-7*mV))) : 1
taum  =  .04*ms + .46*exp(-(-38*mV-v)**2/(30*mV)**2) *ms : second
tauh  =  1.2*ms + 7.4*exp(-(-67*mV-v)**2/(20*mV)**2) *ms : second
dm/dt = (m_inf - m)/taum : 1
dh/dt = (h_inf - h)/tauh : 1
'''

eqs = eqs_V
eqs += eqs_I 
eqs += eqs_activation

G = NeuronGroup(1,eqs, method='euler', dt=0.001*ms)

M = StateMonitor(G, ['v','m','h'], record=0)

## Run the simulation starting from a hyperpolarized membrane.
G.v = 0 * mV
# start with deinactivated (active) channels
G.h = 1 
# run for 100ms
run(100*ms)

# briefly change the potential to a very hyperpolarized potential
G.v = -150 * mV
run(100*ms)

```

```python id="-S7u_v53k11H" colab={"base_uri": "https://localhost:8080/", "height": 400} executionInfo={"status": "ok", "timestamp": 1626447910789, "user_tz": -120, "elapsed": 778, "user": {"displayName": "Karlijn Hers", "photoUrl": "", "userId": "13078358567483151064"}} outputId="a51d4737-cb3f-4d96-a63a-cf55800a3dac"
# Plot results

# prepare a grid to plot graphs  (like function subplot in matlab)
grid = plt.GridSpec(5, 1, wspace=0.4, hspace=1)

# create figure
figure(figsize=(10, 10), dpi= 80, facecolor='w', edgecolor='k')

# in the first viewport plot membrane potential vs time
subplot(grid[:2, 0])
plot(M.t/ms, M.v[0]/mV, label='v')
xlabel('Time (ms)')
ylabel('V (mV)')
legend();

# plot the activation variables
subplot(grid[2, 0])
plot(M.t/ms, M.h[0], label='h')
plot(M.t/ms, M.m[0], label='m')
xlabel('Time (ms)')
ylabel('Gating variables')
legend();

show()

```

<!-- #region id="pySR6R5dtizG" -->
#### Questions: 
1. Describe what you see making reference to the code above.
2. Why does the blue curve goes down slowly and the orange curve fast?
3. Why does the membrane potential rise fast at about 15ms?
3. What happened at 100ms?
4. At the end of the experiment is the channel (dein)active or inactive? Which state variable represents that?
5. At the end of the experiment is the channel open or closed? Which state variable represents that?
6. Experiment with the voltage. What do you observe?

<!-- #endregion -->

<!-- #region id="tl7T8R7JAAH4" -->
#### Answers: 
<!-- #endregion -->

<!-- #region id="7YlBhpeAl0ZH" -->
1. We are measuring the value of the activation variables while we 'clamp' the voltage to different values.
1. The state of the gating variables is membrane potential dependent. When we raise the membrane potential, Sodium channels open (m tends to 1) and inactivate (h tends to 0). A fast flick of the membrane potential leads the channels to close (fast) and de-inactivate (more slowly). Interestingly, the membrane potential settles on one of two possible potentials, either a depolarized one (first 100ms) or a hyperpolarized one (100-200ms). 
2. Because of the time constants.
3. Because m gates are open.
4. The membrane potential was momenterily set to a very low value (-150mV).
5. Active, h.
6. Closed, m.
<!-- #endregion -->

<!-- #region id="ZBREeA1TdTm8" -->
## The Potassium Current
<!-- #endregion -->

<!-- #region id="_oEZ7pv69Imx" -->

The potassium **current** can be expressed equivalently to the Sodium Current, by the following equation:

----

$$ I_k = \bar{g}_{K}n^4(V-E_K)$$ 

----

Where $(V-E_K)$ is the Potassium driving force (the momentary membrane potential minus the potassium reversal potential),  $\bar{g}_{K}$ is the maxiumum conductance of potassium ions. The $n$ is the activation variable. Just like $m$ and $h$ variables, the $n$ activation variable follows the same kind of dynamics as in Sodium, where $n$ tends to its **steady state**.

----
$$ \frac{dn}{dt} = \dot{n} = \frac{n_{\infty}(V) - n}{\tau_n} 
$$

> Just like the other activation variables $n_{\infty}(V)$ has a sigmoidal shape and $\tau_n(V)$ has a gaussian shape.

--- 

We can now inspect the activation gates and add these equations to the model.

<!-- #endregion -->

<!-- #region id="X8cuqE_25tCl" -->
### Coding Exercise: All Activation Functions

Before we go into adding the Potassium (K) current to our membrane, it is illustrative to compare all the activation variables and their respective time constants. Using your sigmoid and tau functions above, plot all the activation variables alongside in the same panel. In another figure or panel do the same for the time constants.
<!-- #endregion -->

<!-- #region id="IR9nLcC8_rK3" -->
| gating function | $V_{\frac{1}{2}}$ | k | time constant ($\tau_{gate}$) | $V_{max}$ | $\sigma$ | $C_{base}$ | $C_{amp}$ |
|-|-|-|-|-|-|-|-|
| $m_\infty(V)$ | -40 | 9 |$\tau_m(V)$|-38     |30| 0.04 |0.46|
| $h_\infty(V)$ | -62 | -7 |$\tau_h(V)$| -67   |20| 1.2  |7.4|
| $n_\infty(V)$ | -53 | 15 |$\tau_n(V)$|-79 |50|1.1  |4.7|

<!-- #endregion -->

```python id="yC6rdTHY5sHe" colab={"base_uri": "https://localhost:8080/", "height": 450} executionInfo={"status": "ok", "timestamp": 1626448013716, "user_tz": -120, "elapsed": 1081, "user": {"displayName": "Karlijn Hers", "photoUrl": "", "userId": "13078358567483151064"}} outputId="ccda2e7f-653b-4d52-eee2-5092ba8a4f27"
# your code goes here

```

<!-- #region id="t2KYsxINaQEO" -->
# 3. The Hodgkin Huxley Model

<!-- #endregion -->

<!-- #region id="DcOxqq27HFIm" -->
### The Hodgkin Huxley Model
<!-- #endregion -->

<!-- #region id="Azy_X383jhQr" -->
Embodying the facts above is the **Hodgkin Huxley model** —a **single compartment model** of the giant squid axon — where the **action potential** (or spike) is a due to **voltage mediated** **interactions between multiple ion channels** which can lead current out or in the cell, that is, had time and voltage dependence, i.e., **active conductances**. 

The  **HH is a four-dimensional model** where the momentary membrane potential ($V_m$) depends on the **state variables** (m, h, n) describing the different states of the ion channels (e.g., activation and inactivation variables). The evolution of the state variables are represented by differential equations.
<!-- #endregion -->

<!-- #region id="f7jOgg_pA2x3" -->
Here we summarize the mechanisms in sections 1 and 2 above. According to the Kirchhoff’s law, the total electrical current $I$ flowing across the membrane may be divided into the capacitive current $I_C$, plus all the ionic currents. We considered three types of ion channels in this model: the sodium channel ($I_{Na}$), the potassium channel ($I_{K}$), and unspecified passive leakage  ($I_{L}$). 

We arrive at the final equation defining the model (also called the 'master equation'):

----

$$ 
I = I_C + I_{Na}+ I_K + I_{leak}
$$


with

$$
I_C = C_m \dot{V}
$$

and where $C_m$ is the membrane capacitance and $\dot{V}$ is the change of voltage across the membrane as a function of time. Therefore, we can express the change of voltage accross the membrane as:

----

$$ 
C_m \dot{V} = I - I_{Na}- I_K - I_{leak}
$$

or more explicitly:

$$ 
  \dot{V} = \frac{I - \bar{g}_{Na} m^3 h (V -E_{Na}) - \bar{g}_{K} n^4 (V - E_k) - g_{leak} (V - E_{leak})} {C_m}
$$

----

Where we also compute $\dot{m}$, $\dot{n}$ and $\dot{h}$ via all the auxiliary equations for \tau and the steady states.
<!-- #endregion -->

<!-- #region id="o2lG9Gz1CmYA" -->
## Coding Exercise: Complete The Hodgkin Huxley Model
<!-- #endregion -->

<!-- #region id="mdEH4WOzCtaJ" -->
Now put all the pieces together: 

1. The passive components of the membrane (capacitance and leak)
2. The Sodium current (conductance, activations, driving force)
3. The Potassium current (conductance, activations, driving force)

<!-- #endregion -->

<!-- #region id="Oo60tvfbaT-0" -->
Below we give you the equations for the dynamics of the potassium current.  

----
> $$ I_k = \bar{g}_K n^4 (V - E_k)$$
> $$ \frac{dn}{dt} = \dot{n} = \frac{n_{\infty}(V) - n}{\tau_n} $$
> $$\tau_n(V) = 1.1 + 4.7  e^\frac{-(-79mV-V)}{(50mV)^2}$$
> $$ n_\infty(V) = \frac{1}{1+e^{(-53mV-V)/15mV}} $$

with 

> $$ \bar{g}_K = 36 \mu /cm^2 S$$
> $$ E_K = -77 mV$$

----


<!-- #endregion -->

```python id="dOi-SHjpeowv"
 start_scope()

# Reversal Potentials
E_leak = -54.4 * mV
E_Na   =  55   * mV
E_K    = -77   * mV

# Conductances
## attention to UNITS! For instance uS << mS
g_leak =   300 * uS / cm ** 2  
g_Na   = 120.0 * mS / cm ** 2
g_K    =  36.0 * mS / cm ** 2

# Membrane Capacitance
Cm = 1 * uF / cm ** 2

# here we define our master differential equation. Note that I_k is already here.
eqs_V ='''
dv/dt = (I -I_leak - I_Na - I_K )/Cm : volt
'''


# here we add a definition for the potassium current I_K
eqs_I = '''
I_leak = g_leak * (v - E_leak)   : amp / meter ** 2
I_Na = g_Na*(m*m*m)*h*(v - E_Na) : amp / meter ** 2
I_K = g_K*(n*n*n*n)*(v - E_K)    : amp / meter ** 2
I                                : amp / meter ** 2
'''


# here you add the equations defining the potassium activation gates
eqs_activation= '''
n_inf =  1/(1+exp((-53*mV -v)/(15*mV))) : 1
m_inf =  1/(1+exp((-40*mV -v)/(9*mV)))  : 1 
h_inf =  1/(1+exp((-62*mV -v)/(-7*mV))) : 1
taum  =  .04*ms + .46*exp(-(-38*mV-v)**2/(30*mV)**2) *ms : second
tauh  =  1.2*ms + 7.4*exp(-(-67*mV-v)**2/(20*mV)**2) *ms : second
taun  =  1.1*ms + 4.7*exp(-(-79*mV-v)**2/(50*mV)**2) *ms : second
dm/dt = (m_inf - m)/taum : 1
dh/dt = (h_inf - h)/tauh : 1
dn/dt = (n_inf - n)/taun : 1
'''


# notice that we simply "concatenate" the strings with all equations
# ( with the operator+=).
eqs = eqs_V
eqs += eqs_I
eqs += eqs_activation

G = NeuronGroup(1,eqs, 'euler', dt=0.025*ms)

M = StateMonitor(G, ['v','m','h','n', 'I'], record=0)

########################## Initialize Variables at Resting State

## Run an experiment 

# resting
G.I = 0 * uA * cm ** -2
run(50*ms)

# hyperpolarizing injection
G.I = 25. * uA * cm ** -2
run( .5 *ms)

# relaxing
G.I = 0 * uA * cm ** -2
run(200*ms)

```

```python id="x9_FOyseaAYP" colab={"base_uri": "https://localhost:8080/", "height": 475} executionInfo={"status": "ok", "timestamp": 1626448208963, "user_tz": -120, "elapsed": 789, "user": {"displayName": "Karlijn Hers", "photoUrl": "", "userId": "13078358567483151064"}} outputId="f759c0a1-9362-451d-c724-8aaa60ea00a1"
# plot it
plt.figure(1, figsize=[7,5], dpi=100)

plt.subplot(3,1,1)
plt.plot(M.t/ms, M.v[0]/mV)
plt.ylabel('V (mV)')

plt.subplot(3,1,2)
plt.plot(M.t/ms, M.m[0])
plt.plot(M.t/ms, M.h[0])
plt.plot(M.t/ms, M.n[0])
plt.ylabel('activations')
plt.legend(['m','h','n']);

plt.subplot(3,1,3)
plt.plot(M.t/ms, M.I[0])
plt.xlabel('Time (ms)')
plt.ylabel('current (mA/cm^2)')

```

<!-- #region id="vFTUBIoycbSd" -->
### Questions:
1. What do we observe in the initial 10ms of the simulation? Why?
2. What happens at 50ms?

<!-- #endregion -->

<!-- #region id="KDWVqNwdfwVK" -->
#### Answers:
<!-- #endregion -->

<!-- #region id="sgzEwN_b8frD" -->
1.  When the model is initialized, the activation variables are not in equilibrium, and they will tend to produce dynamics and oscillations until they relax into an equilibrium state. This is because we must select the initial value of the state variables (V,m,n,h)
2. An action potential! All about it in the continuation of this project.
<!-- #endregion -->

<!-- #region id="FTMkCMAecbSc" -->
## Summary:
<!-- #endregion -->

<!-- #region id="-TvzDBkhcbSd" -->
After installing sodium and potassium channels we effectively assembled the entire Hodgkin Huxley model of the action potential. These are the principal mechanisms in operation:

- Sodium channel induces depolarization via positive feedback (m grows with V)
- Leak brings the cell back to its resting potential (V-Vrest)
- Positive membrane potential induces the inactivation of Sodium (h tends to 0)
- Potassium induces hyperpolarization after the spike (n tends to 1)

These are tempered by the capacitance, which defines how fast the ionic mechanisms contribute to the effective membrane potential.


<!-- #endregion -->

<!-- #region id="GB_uWJumdNbX" -->
# To Be Continued ...

[In Part II of this notebook](https://colab.research.google.com/drive/15W961ErD0v6PIwA0raaz-uTtAGLh6-Md?usp=sharing), we will discuss in more detail how the gating variables interact at the level of the membrane to produce spiking.
<!-- #endregion -->

<!-- #region id="VdW3mCLKVH2b" -->
# Review Questions
<!-- #endregion -->

<!-- #region id="A8vouIS16zcL" -->
## Passive Membrane Properties
- What is 'ion channel permeability'?
- How is capacitance like friction?
- How is the driving force like a spring?
- Explain conductance in your own words and why it is different for different ion types.

## Active Channels and Gating variables
- What are gating variables? What gating variables exist in the typical sodium channel?
- When is h at its maximum, at small or large voltages? 
- When is m at its maximum, at small or large voltages?
- Does the potassium channel inactivate?
- What is a common value of the resting membrane potential?
- What are the ions with high concentration in the neuron?
- What is the unit of g * V_rest?
- [To dive deeper, take this online quiz](https://www.physiologyweb.com/daily_quiz/physiology_quiz_QBTakR5k4CTyBTLXoKGaSzdZ1bz2N7cq_neuronal_action_potential.html
)

<!-- #endregion -->

<!-- #region id="MMKP7eKbMB7m" -->
# Primary Reference
<!-- #endregion -->

<!-- #region id="Jme1X8T-MCc1" -->
Izhikevich, Dynamical Systems in Neuroscience, Section 2.3.1
<!-- #endregion -->

<!-- #region id="k9aJkXljN8hm" -->

# Online Resources
<!-- #endregion -->

<!-- #region id="r01JnlUJZPR8" -->
**Physiology Web Resources**
- [Resting membrane potential](https://www.physiologyweb.com/lecture_notes/resting_membrane_potential/resting_membrane_potential.html)
- [Action Potential](https://www.physiologyweb.com/lecture_notes/neuronal_action_potential/neuronal_action_potential.html)
- [Bilipid Layer Permeability](https://www.physiologyweb.com/lecture_notes/biological_membranes/lipid_bilayer_permeability.html)
- [Derivation of the Nernst Equation](https://www.physiologyweb.com/lecture_notes/resting_membrane_potential/derivation_of_the_nernst_equation.html)

**Nernst Membrane Potential Simulator**
[Nernst Simulator + Resting Membrane Potential](https://www.azps.life/s/ngswin.zip) and believe it or not, [there is also an iOS app for that](https://apps.apple.com/us/app/nernst-goldman-equation-simulator/id1022504095) and the standalone flash version also available [here](https://www.azps.life/home/2016/4/28/teaching-spotlight-nernstgoldman-simulator)


<!-- #endregion -->

<!-- #region id="r-rG3CjMCHe7" -->
#License

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

Mario Negrello, Daphne Cornelise, Elias Santoro. Reviewing and testing Su Saka and Natia Shamuja. Figure sources: Geometry of Bursting, Eugene Izhikevich (2007). Saltatory conductance gif by By Dr. Jana - http://docjana.com/saltatory-conduction/

<!-- #endregion -->
