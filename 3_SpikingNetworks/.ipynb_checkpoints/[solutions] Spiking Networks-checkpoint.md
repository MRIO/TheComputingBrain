---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region id="8-wMVqObQ--2" -->
# Project: Spiking Networks
<!-- #endregion -->

<!-- #region id="pi4ydK9ORD6Q" -->
## How do spikes propagate through networks?


<!-- #endregion -->

<!-- #region id="_Y_1bcH3-eeW" -->
Transmission of spiking activity from one cell to another generally procedes like this: (1) Spikes originate close to the cell's soma (axon hillock) via Sodium channel positive feedback, (2) propagate through axons, (3) activate pre-synapses to release neurotransmitters, which are (4) taken up by receptors of the post-synaptic cell, (5) which activate conductances that (6) become current fluxes into the post-synaptic cell. These current fluxes can be inward and thus excitatory (+, e.g. NMDA/AMPA) or outward, and thus inhibitory (-, e.g., GABA, Glycine). Sufficient current leads to a postsynaptic action potential. And this process propagates to all connected neurons in a network.

**In this project** we want to build some networks that will allow us to ask "when do spikes propagate?" . You will be shown how to construct randomly recurrent networks of spiking neurons, how to give the neurons some input, how to define basic synapses, and how to analyze network spiking behavior. We will have a challenge: search for a set of **synaptic parameters** that produces 'sustained' activity when given stochastic background input, via random **Poisson** processes.

The project has two parts:

---
**Part I: Driving neurons to spike**

- Feeding a network with input and observe the input-output relationship.
- Measure and visualize the firing rate of neurons via histograms and raster plots.
- Produce 'random' spiking input via a 'Poisson process'.
- Produce periodic input via a sine wave.
---

**Part II: Balanced networks**

- Create a network with excitatory and inhibitory neurons.
- Inspect the influence of synaptic and connectivity parameters on the dynamics of the network

This should allow you to:

- Balance the excitation and inhibition in the network to produce sustained activity.
<!-- #endregion -->

<!-- #region id="hmSPB9pt4MkD" -->
## Simulating Neural Networks with Brian
<!-- #endregion -->

<!-- #region id="mx5UK4Yt4SSq" -->
We will continue to use 'Brian2' python package as our spiking neuron network simulator of choice. Brian has the advantage of making the models very explicit and readable, and has some convenient facilities for defining synapses and networks. [Here is an introduction to making Brian network models](https://brian2.readthedocs.io/en/stable/user/models.html). [Here is another on Synapses](https://brian2.readthedocs.io/en/stable/resources/tutorials/2-intro-to-brian-synapses.html). 

Though not strictly required for the project, it is arguably useful to learn the basics of Brian. As all languages, it has its powers and its quirks, and to write code for it is important to understand its axioms.
<!-- #endregion -->

<!-- #region id="5UxEbiVFRluX" -->
## Pre-requisites
<!-- #endregion -->

<!-- #region id="dvOWrxWE-kSE" -->
- Intuition about spiking dynamics of Leaky Integrate and Fire (IF) neurons.
- Basic knowledge about synaptic models (particuarly, event-based current steps).
<!-- #endregion -->

<!-- #region id="nuYOM7ZBR1IV" -->
## Learning Goals
<!-- #endregion -->

<!-- #region id="I_i8tAxfR360" -->
In this project you will learn how to:
- create a neuronal group with a population of IF neurons.
- add simple excitatory and inhibitory synapses to that group.
- stimulate the network via random events (a Poisson Input).
- create a randomly connected network.
- calculate basic statistics of network activity, such as average firing rate.
- display network activity via raster plots and histograms.
- manually tune a network to be balanced and produce different kinds of network activity.
<!-- #endregion -->

<!-- #region id="8APOQsJ5S5v6" -->
## Terminology
<!-- #endregion -->

<!-- #region id="AaaBr-HzS-3s" -->
 - **stochastic inputs**: we say something is stochastic when we cannot account for all sources of variability. For example, we can say that a neuron receives stochastic input from the rest of the brain. Stochastic input can look like 'noise' (statistically).
 - **event based synapses**: synaptic currents that are activated by a synapse in the pre-synaptic neuron, an 'event'.
 - **balanced networks**: Networks where inhibition and excitation are balanced.
 - **raster plots**: A method to display the spikes of multiple neurons.
 - **histograms**: A bar plot with the counts of events within a time bin.
 - **time series**: A continuous signal over time (such as membrane potential).
<!-- #endregion -->

<!-- #region id="rV8gZcx3TR63" -->
## Initialization
<!-- #endregion -->

```python id="9W_plJ1eTVaj"
!pip install brian2
from brian2 import *
import numpy as np
from matplotlib.pyplot import *
from random import sample # note that these two below act differently 
from numpy import random
```

<!-- #region id="-rx4iSQhfaiZ" -->
# Part I. Spiking Networks with Random Input
<!-- #endregion -->

<!-- #region id="4j5BgrE9ELFS" -->
## Making a Neuronal Network Model
<!-- #endregion -->

<!-- #region id="P10-ggBOYgG3" -->
**When setting up a network of spiking neurons these are some of the core questions we must consider**:

0. What are we trying to represent? (e.g., cortex, cerebellum, hippocampus, ...)
1. What neuronal model(s) will we use (e.g., HH, AdEx, LIF)? (and why?) 
2. How many neurons are in the model network? (do we have the computational resources?)
3. What is the proportion of neurons of different types (e.g. excitatory, inhibitory, modulatory neurons)?
4. Which types of synapses connect the neurons?
5. How are neurons connected?
6. What are the inputs to the network?

The answer to these questions implies a specific choice of model equations, parameters and **instantiates** a particular network.
<!-- #endregion -->

<!-- #region id="SttVU8WY4sAg" -->
## Network with Random Inputs
<!-- #endregion -->

<!-- #region id="_jB0-OpG9olq" -->


In an important sense, random inputs are the most general input to a network, and even if one plans to give specific input to a network on a later stage, one often starts modeling by sanity checking network activity to random input.

Random inputs to a network can be thought of as representing signals from unknown areas, and in a sense, they include 'every possible input' to a given network.

Two of the most common methods to create random input into a network are 1) **the Poisson process** and 2) **noisy current fluctuations**.

- **Poisson Process**: [The Poisson process](https://en.wikipedia.org/wiki/Poisson_point_process) in general models a series of discrete events where we know the *rate* of event emmission, or equivalently the *average* time between events, where the *exact* interval between events is random. (i.e. `mean(y) = var(y) =` $\lambda$)

  The Poisson distribution has one parameter, $\lambda$ that states the probability of a given number of spikes within a time interval. 

    **Example**.
    It is the spiking equivalent of a coin toss. For a given time interval (say 1 second), and a given probability of spiking (say 50Hz, 50 spikes per second), we split the second into 1000 bins of 1ms, such that maximally one spike can fall in a bin. Then, for each time bin we decide whether a spike has happened  by generating a random number and checking whether it was smaller than the probability of finding a spike in any bin (one coin toss per bin).  Given the 50Hz rate, $p = 0.05$ -- spikes/milisecond). That is, we say that there was a spike if  ```rand()<=p``` where ```rand()``` is a single random number and $p$ is the probability given the rate $\lambda$. 

- **Current noise**: A continous random current injection, also called "stochastic". For any time interval, the neuron receives a current whose amplitude given by a random number from some distribution. Variations on this are the Gaussian process (random number from the normal distribution), the Wiener process (a random walk), Orstein-Uhlenbeck (non drifting gaussian process). We will not go in detail in this project, but here's some [background information](https://brian2.readthedocs.io/en/stable/user/models.html#noise) and [an example](https://brian2.readthedocs.io/en/stable/examples/non_reliability.html). 

<!-- #endregion -->

<!-- #region id="UR9vyy0nm3y8" -->
## A Poisson Process

Here we produce one instantiation of the poisson process whose rate is given by the parameter $\lambda$ representing the rate of the process, that is, how many events are expected in a given interval (if the interval is one second, the unit is Hz).
<!-- #endregion -->

```python id="1j-LsQJC6CIo"
# A single Poisson spike process

# suggestion: run it multiple times. What changes? What is the (approximately) the same?

# below we call rate the number of 'events' per second (Hz). This is a rate we want to produce. 
# Note that this is only same as lambda if the interval is a second (50/1). 
# However, we want to calculate in chance that any ms will have a spike, so our lambda is (spikes/ms).

rate = 50 # Hz

interval = 1000 # ms
binsize = 1 # ms

#below 'prob' is the same as 'lambda'
prob = rate / (interval / binsize) # the denominator defines the number of 'bins' in interval.

R = random.uniform(low=0.0, high=1.0,size=(interval,1))
events = where(R < prob)
# events contain the indices of "event" bins.

plt.scatter(events[0], events[1],s=1.5) 
plt.yticks([])
plt.xlabel('miliseconds')
plt.title('A 50Hz Poisson Process')
plt.legend(['spikes'])

# if you want to see the timing of spikes, uncomment the line below
# print(events[0])

number_of_events = len(events[0])
print('number of events: '+ str(number_of_events))
```

<!-- #region id="Mq4KFCYT9WRj" -->
> The little dots above represent *events*, which could be taken to be *random spikes* with a given frequency. We could also represent them as little vertical lines, more commonly seen display of *spike trains*. This is left as an exercise for the reader =]
<!-- #endregion -->

### Exercise

Run the simulation above multiple times. Is the frequency always the same? Are the spike always in the same place?


## Leaky-Integrate and Fire


Below we assume a basic understanding of the dynamics of the leaky-integrate and fire neuron. For an in-depth analysis about the dynamics of the LIF neuron [go though this notebook.](https://colab.research.google.com/github/johanjan/MOOC-HPFEM-source/blob/master/LIF_ei_balance_irregularity.ipynb#scrollTo=Qb3r2hd_fWzX)

<!-- #region id="pqx6jRSz3snF" -->
## LIF network with Poisson inputs
<!-- #endregion -->

<!-- #region id="cuUk5istPYuZ" -->
What would happen if a set of leaky integrate and fire (LIF) neurons receives random spikes from unknown sources? Let's imagine that we have a neural network setup as follows:

0. **What:** A group of of neurons driven by random spiking events (Poisson)
1. **Neuron model:** LIF neurons
2. **N = 100*** (and 100 Poisson sources, one for each neuron)
4. **Synaptic dynamics:** for every input spike, ge += 5mV* (and decays in 10ms)
5. **Connectivity:** each input source connects to one neuron, but **neurons in the group do not connect to each other**. This way we can understand the effects of input withoutu the confounders of connectity. Later we shall connect the neurons.
6. **Input:** 100 Poisson events at 500HZ, connected 1 to 1.
<!-- #endregion -->

```python id="D9Vnk-QD3yhC"
# Create a network of integrate and fire neurons with poisson input
start_scope() # always start with this 

N = 100 #  number of neurons
poisson_event_rate = 500.*Hz 

runtime = 1000*ms

taum = 20*ms #  membrane time constant
taue = 5*ms # excitatory synapse time constant
Vt = -45*mV # spiking threshold
Vr = -50*mV # reset potential
El = -60*mV # leak reversal potential

# increment of synaptic voltage
exc_step = 5*mV; # quiz: why do you think the unit is mV? see comment =] 

eqs = '''
dv/dt  = (ge-(v-El))/taum : volt (unless refractory)
dge/dt = -ge/taue : volt (unless refractory)
'''

# Poisson inputs
# https://brian2.readthedocs.io/en/stable/user/input.html?highlight=poisson#more-on-poisson-inputs
sources = PoissonGroup(N=N, rates=poisson_event_rate)
#sources = PoissonInput(target=neurons, target_var='v', N=100, rate=poisson_event_rate, weight=0.1*mV)

# LIF neuron group
neurons = NeuronGroup(N, eqs, threshold='v>Vt', reset='v = Vr', refractory=5*ms, method='euler')

# creates synapses between poisson input and firing neurons that
# add a voltage step at every presynaptic event
S = Synapses(sources, neurons, on_pre='ge+=exc_step')  

# NOTE: on_pre is a keyword in brian that specifies what happens on the postsynaptic neuron when presynaptic neuron fires
S.connect(j='i') # this is Brian's method to connect every 'i' (source synapse)

# set the initial state of the network
neurons.v = Vr 

# create State Monitors
netmon = StateMonitor(neurons, ['v','ge'], record=np.arange(0,N)) # one could also write 'True', to record all neurons.
spikemon = SpikeMonitor(neurons)

run(runtime) # run simulation for 1 second
```

<!-- #region id="-0I16Jm-BdlT" -->
### Inspecting Outputs
<!-- #endregion -->

For a single neuron we plot the time course of ```ge``` and ```v``` after running the simulation. Plot variables vs time (on the x-axis) and use one subplot per variable. 

> One important thing to notice is that we have to 'divide by unit' to be able to plot, as numpy does not know how to plot variables that have units. Note: ```netmon.t/ms``` or ```netmon.v[0]/mV```. In addition we index which neuron to plot with square brackets ```netmon.v[0]/mV```.

```python id="oKxDrf4Vg-q8"
figure(figsize=(12,6))
subplot(211)

# to plot the variables run the code above and use plt.plot on the recorded variables
# - for plotting you will need to remove the units from time and mV

plot(netmon.t/ms,netmon.v[0]/mV)
xlabel('Time (ms)')
ylabel('V (mV)')

subplot(212)
plot(netmon.t/ms,netmon.ge[0])
xlabel('Time (ms)')
ylabel('ge (-)')
show()
```

<!-- #region id="TamLew49AmIW" -->
## Measuring Spiking Activity
<!-- #endregion -->

<!-- #region id="tLp0XtoX7QIx" -->
In modeling as in experiment, the spiking activity of neurons is commonly measured via **firing rates**, **interspike intervals**, **raster plots** and **activity histograms**. 
- **Firing rate** is the frequency of spiking of the neurons, or how many spikes are fired per second. That can clearly vary as a function of network state or input. 
- **Inter Spike Interval (ISI)** is the interval between two spikes. We often display the distribution (histogram of the ISI) to look into the variability of ISIs.The reciprocal of the ISI is called 'instantaneous firing rate'.
- A **raster plot** displays spikes as dots, with the neuron index on one axis and time on another. The firing rate of the entire network can be read from this graph as  the number of spikes divided by (time x neurons).
- A **spike count histogram** counts the numbers of spikes within a given bin across time. It is usually measured in number of spikes or in frequency units (spikes/bin => Hz). It is an ideal way to visualize the changes of network activity over time.
<!-- #endregion -->

<!-- #region id="y1kno3ap0mkc" solution="shown" solution_first=true -->
### Firing Rates
<!-- #endregion -->

<!-- #region -->
#### Exercise

---

For the simulation you ran above, use basic math (in numpy) to find the firing rates:
- of selected single neurons
- of the entire network



**Needful things:**

  - ```spikemon.count```  - the array of spike counts
  - ```spikemon.count[0]```  - total number of spikes for the first neuron in the population
  - ```spikemon.t[0]``` - time of first spike
  - ```spikemon.t[-1]``` - time of last spike
  - ```netmon.t[0]```  - time at the beginning of simulation
  - ```netmon.t[-1]``` - time at the end of simulation

---
<!-- #endregion -->

```python
# YOUR CODE

# calculate the firing frequence of a single neuron


# calculate the firing frequence of a the population of neurons
```

#### Solution

```python id="jsXy7PtN8jrg"
# Calculate Firing rates

## per neuron (t[-1] means 'the time of the last spike' )
fr_per_neuron = (spikemon.count[0])/(spikemon.t[-1] - spikemon.t[0]);
print(f'The firing frequency of neuron 1 is: {fr_per_neuron} Hz')

## the population
fr_pop = sum(spikemon.count)/((spikemon.t[-1] - spikemon.t[0])*N)
print(f'The population firing frequency is: {fr_pop} Hz')
```

<!-- #region id="HgcrV1BZHk1U" -->
### A Raster Plot and a Histogram
<!-- #endregion -->

<!-- #region id="5S_rNwghFRTn" -->
How to represent the activity of a spiking neural network? In the raster plot we have dots representing spikes, where each row in the y-axis have the spikes from a single neuron. Time is represented in the x-axis. The histogram counts the number of spikes in a certain bin, and gives us a sense of the average firing rate (in Hz or in spikes per bin) of the entire network. From the histogram we can read out the average rate of the neurons by (mentally) dividing the height of the bar by the bin duration and the number of neurons in the network. It also shows us the bin-to-bin variation in the spike counts.

Below you see how we create a **raster plot** and a **spike count histogram**. 
<!-- #endregion -->

```python id="NZHcWuoNEMww"
# spike count histogram

bins = 10 # choose the number of time bins you want
bin_time = runtime/bins/ms # time of one bin
spike_times = spikemon.t/ms # get spike times 
intervals = list(range(0, int(runtime/ms), int(bin_time))) # the bins

plt.figure(1, figsize=[6,4], dpi=100)
plt.subplot(211)
plt.title('Raster plot')

plt.plot(spikemon.t/ms, spikemon.i, '.k', ms=1.2) # spikemon.i retrieves all neurons
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.xticks(intervals)

plt.subplot(212)
plt.title('Spike count histogram')
plt.hist(spike_times, bins=intervals, rwidth=0.8)
plt.xticks(intervals);
plt.xlabel('interval (ms)')
plt.ylabel('# spikes')
plt.tight_layout()
```

**Question** : What is the frequency of the network above? Is the binsize we chose good or bad? Why?

<!-- #region id="8GTOv7FcxhEE" -->
### Analog Traces

<!-- #endregion -->

<!-- #region id="mhWR4NEk-WUE" -->
The spikes are only the "peak of the icebergs". To visualize what happens below the water level one can also plot the continuous variables associated with the neuron's activity (aka, *time series*). To understand how a network behaves one often needs to display the membrane potential or the excitatory conductances, which are continuous variables.
<!-- #endregion -->

<!-- #region id="Yv2U4kBWZizO" -->
> Below you see the traces of a single neuron in the network, the top plot has the EPSP (the excitatory post synaptic potentials, ```ge```) and the cells membrane potential. Note also that we artificially plot 'spikes' at the threshold crossings.
<!-- #endregion -->

```python id="HR4zewJOTGZe"
plt.figure(1, figsize=[8,3], dpi=100)
plt.subplot(2,1,1)
plot(netmon.t/ms, netmon.ge[0]/mV, label='ge')
legend();
ylabel('EPSV (mV)')
subplot(2,1,2)
plot(netmon.t/ms, netmon.v[0]/mV, label='V')
plot([spikemon.t[np.where(spikemon.i==0)[0]]/ms, spikemon.t[np.where(spikemon.i==0)[0]]/ms], [Vt/mV, 2], color='#1f77b4')
xlabel('time (ms)')
ylabel('membrane potential (mV)')
legend();
```

**Question:** How are the two traces above related?

<!-- #region id="oALdHJ4saG4E" -->
A way to visualize the continuous variables of multiple cells simultaneously is via a "heat map":

> Below you see the voltage traces of all neurons in the network in color coded membrane potentials.
<!-- #endregion -->

```python id="NX-bj5mGHZOG"
# analog raster of membrane potential
plt.figure(figsize=(9,4))
plt.pcolor(netmon.v)
cb = plt.colorbar()
cb.set_label('mV')
plt.title('membrane potential')
plt.xlabel('time')
plt.ylabel('neurons')

# # analog raster of excitatory conductance
# plt.pcolor(netmon.ge)
# plt.title('excitatory conductance')
# plt.xlabel('time')
# plt.ylabel('neurons')
```

**Question:** Describe whawt you see in this plot. Do you see spikes in this plot? Are the cells synchronized? 

<!-- #region id="4VRcsS-Dwy4t" -->
### The Interspike Interval Histogram

The ISI histogram is another way of learning more about the activity of a network. It displays the frequency of interspike intervals, and can be used to, for instance, judge how random is the firing pattern of a network: if it falls exponentially, we can conclude it 'looks poissonian'.

First we calculate some interspike histograms and then move on to display the interspike interval histogram for the entire network.
<!-- #endregion -->

```python id="hFfrmB9WQGzs"
# Interspike Interval (ISI) histogram

# obtain the spike times via the 'spike_trains' method
trains = spikemon.spike_trains()

# a single spike train looks like this:
print(trains[1])
```

```python id="X6cFK4cMo0WT"
# its interspike interval looks like this:
print(diff(trains[1]))
```

```python id="GzgrhoIDnRLx"
# for a single neuron, take the difference between each two neighbouring items
interspike_intervals = np.diff(trains[1])
                               
# to obtain the interspike intervals for the entire population
isis = [diff(trains[idx]) for idx in range(0,N-1)]

isis = np.concatenate(isis).ravel()
```

```python id="fPiBpc-AeVB_"
# for each neuron, produce its counts
plt.hist(isis, bins=20, range=(0,.5),rwidth=.9)
plt.ylabel('# of intervals')
plt.xlabel('interspike interval time (s)');
```

<!-- #region id="QcK91ZSxO7xa" -->
#### Questions
<!-- #endregion -->

<!-- #region id="a6iOGLieGWR9" -->
---
- Make sure you know what is the rate of the poissonian input (it is given in the code above).
- From the ISI histogram above, estimate by eye the rate (frequency) of spiking generated by the entire network.
- Why does the frequency of the network and of the poisson processes differ so starkly? Find 1 (or up to 5) parameters that influence the relationship between the input and the output.
---
<!-- #endregion -->

<!-- #region id="q9I7KqVdhpxh" -->
#### Answers
<!-- #endregion -->

<!-- #region id="G_rcty09hs9b" -->
**Question 1.**

The poisson process used a rate of about 500 Hz / 500 spikes per second.

**Question 2.**

With the default parameterisation

```python
taum = 20*ms #  membrane time constant
taue = 5*ms # excitatory synapse time constant
Vt = -45*mV # spiking threshold
Vr = -50*mV # reset potential
El = -60*mV # leak reversal potential

# increment of synaptic conductance
exc_step = 5*mV;
```

- The firing frequency of neuron 1 is: $7.13$ Hz 
- The population firing frequency is: $8.9$ Hz

Keep in mind that yours will vary, but as long as they are relatively close its correct.

**Question 3.**

You can find the 5 parameters that influence the transmission rate by considering the equations and running the simulation multiple times. Keep the poisson input range constant (at 500 Hz) and change one parameter value at a time. You should see that the following parameters influence the transmission rate:

1. **tau_e** and FR are directly proportional. 

- Increasing `tau_e` leads to an increase in activity; an increase in the firing frequency / rate (FR).
- Decreasing `tau_e` leads to a decrease in the firing frequency.

Why is that? Consider our second DE 

$$
\frac{d g_e}{dt} = - \frac{g_e}{\tau_e}
$$
 
Note that an increase in $\tau_e$ leads to a smaller (negative) value $\dot g_e = \frac{dg_e}{dt}$. The excitatory conductance decays slower over time. Which in turn

$$
\dot V = \frac{g_e - (V-E_L)}{\tau_m}
$$

leads to a faster change in membrane potential ($\dot V$ is larger) and thus a higher spiking frequency.


2. **tau_m** and FR are inversely related.

- increasing `tau_m` leads to a decrease in activity; a decrease in the firing frequency.
- decreasing `tau_m` leads to an increase in activity; FR.

This is because $\tau_m$ determines the rate of change, the time it takes for the membrane potential to change $V_m$ ($\tau_m=R * C$). Thus, as it takes longer $V_m$ will pass the threshold 
$V_{th}$ less often within our simulation time. We obtain fewer spikes. Alternatively, note our first DE:

$$
\dot V = \frac{g_e - (V-E_L)}{\tau_m}
$$

And think about what happens to $\dot V$ when you increase $\tau_m$.
 
3. **exc_step** and FR are directly proporional.

Increasing the conductance $g_e$ with each spike, indirectly increases the (positive) rate of change of the membrane potential, thereby raising the chance for the neuron to spike (so $\uparrow$ FR).

4. **Vth**

Intuitively, 
- Lowering the threshold (e.g. Vth = -60) leads to more spikes
- Making the the threshold more difficult to reach (e.g. Vth = -40 leads to less spikes) 


5. **El**

Note our first DE

$$
\dot V = \frac{g_e - (V-E_L)}{\tau_m}
$$

Where $(V-E_L)$ is called the driving force, it's how badly the current membrane potential $V$ wants to go back to its equilibrium $V_{rest}$. If the driving force grows, the urge to go back to equilibrium gets stronger. Note that as $V-E_L$ grows, $\dot V$ eventually becomes a negative value so the membrane potential $V$ will decrease. This will lead to fewer spikes (so $\downarrow$ FR).

For example,

- making $E_L$ more positive (e.g $E_L=-55 mV$) diminishes the driving force, leading to an increase in neuron firing frequency.


<!-- #endregion -->

<!-- #region id="Li2liMmfiPEr" -->
In the code below we produce a balanced network, with excitatory and inhibitory synapses. Create **event based** excitatory and inhibitory synapses. That means, for every spike of the presynaptic neuron, the post-synaptic neuron receives a short current step with amplitude `we`. This is expressed in the following lines:

```python
# create event based synapes between neurons 
Ce = Synapses(neurons, neurons, on_pre='ge += we')
Ci = Synapses(neurons, neurons, on_pre='gi += wi')
# connect them
Ce.connect('i<800', p=0.1)  # we make first 800 neurons are excitatory neurons
Ci.connect('i>=800', p=0.1) # and the rest inhibitory
```

We make it such that the first 800 neurons are excitatory and the last 200 inhibitory, thus respecting 'Dale's rule', which states that neurons in the nervous system release at most one type of neurotransmitter. As it turns out, this is a true assumption for all mamalian neurons.
<!-- #endregion -->

<!-- #region id="0i1KS6v92hc1" -->
## LIF with Periodic Input
<!-- #endregion -->

<!-- #region id="k32wqR71zJ6E" -->
In addition to a random input, we may also introduce any kind of input function, such as a periodic stimulus to the network. Periodic firings are common in many areas and could be generated by, for example, nodding your head. Your vestibular receptors in your semi-circular channels would respond with spikes according to your head bobbing at frequency ```f```. This would be linearly encoded by your vestibular neurons. We could translate that input into 'current' driving the neurons:

$I = a \cdot sin(2 \pi f t)$


> Here a is a constant defining the amplitude of the current. Note that this equation is dependent on `t`. Brian knows how to handle time so that we can have continuous functions of time as input.

In the model below we do not have external synapses, and there is only one type of simple integrate and fire neuron. We embed the sinusoidal stimulus in the equation of the neuron.
<!-- #endregion -->

```python id="lF9X2AV6xtMo"
# from : https://brian2.readthedocs.io/en/stable/examples/phase_locking.html
start_scope() # always start with this 

tau = 20*ms
n = 100
b = 1.2 # constant current mean, the modulation varies
freq = 10*Hz

eqs = '''
dv/dt = (-v + input + b) / tau : 1
input = a * sin(2 * pi * freq * t) : 1
a : 1
'''
neurons = NeuronGroup(n, model=eqs, threshold='v > 1', reset='v = 0',
                      method='euler')

# we randomize the initial state of the network
neurons.v = 'rand()'

# We attribute to each neuron an input 'amplitude' (a) that is a function of the number of neurons (n) 
# and of the neuron index (i): 
neurons.a = '0.05 + 0.7*i/n'

S = SpikeMonitor(neurons)
trace = StateMonitor(neurons, ['v','input'], record=[0,49])

run(1000*ms)
figure(figsize=(20,8))
subplot(311)
plot(S.t/ms, S.i, '.k')
xlabel('Time (ms)')
ylabel('Neuron index')
subplot(312)
plot(trace.t/ms, trace.v.T)
xlabel('Time (ms)')
ylabel('v')
subplot(313)
plot(trace.t/ms, trace.input.T)
xlabel('Time (ms)')
ylabel('I (nA)')
tight_layout()
show()
```

### Questions
---
- Explain the main features of the plots above. For example:
- In the raster plot:
    - why does the ISI seems to decrease?
    - Why is there more scatter in some neurons? 
- In the voltage time series:
    - Should we see spikes?
    - What is the difference between the two cells?
---

<!-- #region id="95qM_h9J5CXs" -->
# Part II. Balanced Networks



<!-- #endregion -->

<!-- #region id="hcmUCtk0T4hi" -->
## Phase transitions


<!-- #endregion -->

<!-- #region id="MI2erVpb5rk-" -->
There is good evidence that cortical networks operate in a **balanced state**, one in which the intensity of excitation is balanced by inhibition. This keeps neurons in a optimal operational range, as it increases sensitivity to small inputs, avoids wasting energy in an 'always on' mode, and does not waste energy in always regular activity. It is also a 'highly informative state', in the sense that it is good at encoding input.

A consequence of balanced synapses is that, if true, the brain operates at a **critical regime**. Critical in this sense means "close to a phase transition". Like water at 0, it can quickly become ice. In the brain this looks like bursts of activity (often called "cascades of spikes") alternating with periods with no spikes. The transition between active and silent is sharp and very sensitive to small changes in input. Some have called this "the edge of chaos".

In short:
> Brain networks are really sensitive to certain parameter values. 

Networks of LIF neurons can demonstrate the effect of critical dynamics. This is accomplished by tuning the relative weight of excitatory and inhibitory synapses, so that they balance each other.
<!-- #endregion -->

<!-- #region id="dRdndQWun0BY" -->
<img src="https://neuronaldynamics.epfl.ch/online/x334.png" width="300x">

The above diagram is a **parameter space** of the LIF network displaying the different kind of states the network can occupy, as a function of the relationship between input and the balancing variable $g$, the relative importance of inhibition. [(for more details see here)](https://neuronaldynamics.epfl.ch/online/Ch13.S4.html).


- **Synchronous Regular (SR).**
  - $g = 3$
  - input = $2$
- **Synchronous Irregular fast (SI).**
  - $g = 6$
  - input = $4$
- **Asynchronous Irregular (AI).**
  - $g = 5$
  - input = $2$
- **Synchronous Irregular slow (SI).**
  - $g = 4.5$
  - input = $0.9$

<img src="https://neuronaldynamics.epfl.ch/online/x335.png" width="300x">

Here are examples from the different types of dynamics. Here we see raster plots with histograms displaying the population activity below. Your job below will be to recreate these graphs using the parameters above as you vary the values for the relative importance of inhibition (`g`) and the external input the network receives (`w_poisson`).
<!-- #endregion -->

<!-- #region id="xsofFEibYfKi" -->
In the exercises below we will inspect the influence of the ratio between excitation and inhibition on network activity. This should provide you with asense of the spectrum of possibilities for the spiking dynamics of spiking networks.


<!-- #endregion -->

<!-- #region id="C4yEkJU3uIax" -->
## Recipe for a balanced network:
<!-- #endregion -->

<!-- #region id="PIW9BgT0Gzg_" -->
1. **Neurons:** 1000 LIF neurons (as may fit in a 100um cubic patch of brain).
3. **Synapses:** Exciatory to Inhibitory neuron proportion: 80% to 20%, the rough proportion found in many mammalian cortices.
5. **Connectivity:** We assume random connectivity. Any neuron in the network can be connected to any other with a given probability, irrespectively of type. This type of network is also called an **Ërdos-Reiny** network.
6. **Input:** background input via poisson process to bring the network to the verge of spiking.

---

- Parameter values are taken from the paper by [Brunel, 2000](https://web.stanford.edu/group/brainsinsilicon/documents/BrunelSparselyConnectedNets.pdf), and can also be found in this [online book on neurodynamics](https://neuronaldynamics.epfl.ch/online/Ch13.S4.html).

- Neurons are connected (coupled) to each other with some probability epsilon & also recurrently connected.
- Both excitatory and inhibitory neuron populations get an external Poisson input (poisson process as explained above).


<!-- #endregion -->

### Neuron and Network Parameters

```python
start_scope() # Brian's way of starting anew

# We start out with the following default parameters

####
# default LIF neuron parameters
v_rest = 0. * mV # rest potential
Vr = 10. * mV # reset potential
Vth = 20. * mV # spiking threshold
tau_m = 20. * ms # membrane time scale
tau_ref = 2.0 * ms # absolute refractory period

####
# default network parameters
# neurons
N_excit = 5000
N_inhib = int(N_excit/4) 

# connectivity parameters
epsilon = 0.1 # connection probbaility
delay = 1.5 * ms # synaptic delay

####
# simulation parameter (for the Euler solver)
defaultclock.dt = 0.05 * ms
```

<!-- #region id="SKfg0BPVRt7N" -->
### Define Dynamics, Create Network Object

We have:

- a differential equation that tells us how our state variable $V_m$ changes over time, the **dynamics**, which are encoded by the equations  `eqs = """..."""`.
- a network with N neurons ` = NeuronGroup(...)`
<!-- #endregion -->

```python
# the dynamics
eqs = """
    dv/dt = -(v-v_rest) / tau_m : volt (unless refractory)
    """

# build the network
EI_network = NeuronGroup(N_excit + N_inhib, model=eqs, threshold="v>Vth", reset="v=Vr", refractory=tau_ref,
        method="linear")
```

<!-- #region id="WjixYZ2CzPiO" -->
### Voltage Synapses
<!-- #endregion -->

<!-- #region id="C1BHpXHL-LNz" -->
The **synaptic events** are little voltage jolts to the membrane. We defined them as such:

```python
w0 = 0.1 * mV # synaptic voltage deflection (event based synapse)

# EPSV: Excitatory Post Synaptic Voltage Deflection
we = w0
wi = -g*w0 
```

Note that the unit of weight is voltage. This is what happens: a spike occurs, and after a short delay, the voltage jumps with `0.1 mV`. Since each spike triggers this tiny voltage jump, the more spikes occur, the faster the voltage reaches threshold (notice the positive feedback loop?). This voltage decays with a certain speed (Our neuron is a "Leaky" Integrate and Fire, a LIF). 

Variable $g$ defines the relative importance of inhibition. As you will see, changing the strength of the inhibitory synapses has major impact on the network activity.
<!-- #endregion -->

```python
# synaptic weights
w0 = 0.1 * mV 
we = w0
g = 4. # relative importance of inhibition
wi = -g*w0
# note: w_ee = w_ei = w0 and w_ie=w_ii = -g*w0
```

### Connect Neurons

<!-- #region -->
In the code below we produce a balanced network, with excitatory and inhibitory synapses. This is expressed in the following lines:

We assume the first `N_excit` neurons are excitatory and the remaining are inhibitory.

```python
# slice populations 
exc_pop = network[:N_excit] 
inh_pop = network[N_excit:]
```

Create **event based** excitatory and inhibitory synapses. That means, for every spike of the presynaptic neuron, the post-synaptic neuron receives a short current step with amplitude `w0`. We also add a conduction delay. 
 
```python
# create event based synapes between neurons 
exc_synapses = Synapses(source=exc_pop, target=network, on_pre="v += we", delay=delay)
```

We then connect the neurons probabilistically, making some of them excitatory and others inhibitory:
    
```python
exc_synapses.connect(p=epsilon)
```

<!-- #endregion -->

```python
# slice populations 
exc_pop = EI_network[:N_excit] 
inh_pop = EI_network[N_excit:]

# create the synapses & connect
exc_synapses = Synapses(source=exc_pop, target=EI_network, on_pre="v += we", delay=delay)
exc_synapses.connect(p=epsilon)

inhib_synapses = Synapses(source=inh_pop, target=EI_network, on_pre="v += wi", delay=delay)
inhib_synapses.connect(p=epsilon)

```

### Poisson Inputs

```python
# input parameters
poisson_input_rate = 13. * Hz
N_poisson_input = 1000 # number of poisson inputs
w_poisson = w0 # one could also play with this variable

external_poisson_input = PoissonInput(target=EI_network, target_var="v", N=N_poisson_input,
                                      rate=poisson_input_rate, weight=w_poisson)
```

### Connect Monitors


Then, we want to see how our neurons are behaving. To do this, we create different monitors.

We want to:
- record the spikes `= SpikeMonitor(...)`
- trace the voltage `= StateMonitor(...)`
- check the overall activity of the neurons `= PopulationRateMonitor(...)` 

```python
# choose number of indices we want to monitor (for voltagemonitor & population rate)
num_indices = 20

idx_monitored_neurons = sample(range(N_excit+N_inhib), num_indices)
# make monitors
rate_monitor = PopulationRateMonitor(EI_network)
spike_monitor = SpikeMonitor(EI_network, record=idx_monitored_neurons)
voltage_monitor = StateMonitor(EI_network, "v", record=idx_monitored_neurons)
```

<!-- #region id="3xLIFMQ9VfQh" -->
### Run your simulation

We have build the network and our recording devices connected, great! Now you can run the simulation for a given simulation duration ```simtime``` with:

```python
run(simtime)
```
<!-- #endregion -->

```python
# set an initial condition (starting membrane potential)
EI_network.v = v_rest


sim_time = 250*ms

# simulate for this duration
run(sim_time)
```

<!-- #region id="DNRfiNzXV8RD" -->
### Visualize the neural data

After running the simulation we want to "see" results. Did neurons spike? What was their voltage? This is where the monitors come in. We can use the them to display the network behaviour.

```python
# recording spikes
spike_monitor.t : an array of time instances when a neuron spiked

spike_monitor.i : an array of the neurons indices that spiked 

# for the voltage trace
voltage_monitor.t : array of time instances

voltage_monitor.v.T : voltage over time

# population rate (LFP)
rate_monitor.rate : array of population activity (Hz)
```
--- 

**Hint.**
if you get errors, print the arrays to sanity check them before plotting.

---
<!-- #endregion -->

```python id="b8D0-ZITVW3E"
# plot it
plt.figure(1, figsize=[10,7], dpi=100)

plt.subplot(3,1,1)
plot(spike_monitor.t/ms, spike_monitor.i, '.k', ms=0.5)
xlabel('Time (ms)')
ylabel('Neuron index')

plt.subplot(3,1,2)
plt.title('Voltage traces')
plot(voltage_monitor.t/ms, voltage_monitor.v.T, color='xkcd:teal')
xlabel('Time (ms)')
ylabel('Membrane potential $V_m$')

plt.subplot(3,1,3)
plt.title('Population activity')
plt.plot(rate_monitor.t/ms, rate_monitor.rate, color='xkcd:coral')
plt.xlabel('time (ms)')
plt.ylabel('A(t) (Hz)')
tight_layout()
```

<!-- #region id="-o0D1n127OC0" -->
## Emergence of synchronisation
<!-- #endregion -->

<!-- #region id="3Yn69hXi9yA5" -->
Now, use the network you have implemented for the previous exercise and change the following values: `g=2.5` and `poisson_input_rate=14.*Hz`. Use the coding cells below to copy and adapt the code you had above. Run the simulation as before and answer the following questions:

### Questions
---
- How would you describe the network activity?
---
- In which state is this network? 
---
- Apart from the balance between E-I, what other factors determine the ability of the network to synchronize?
<!-- #endregion -->

### Your Code

```python
### Your Network Here
```

<!-- #region heading_collapsed=true -->
### Our Code
<!-- #endregion -->

```python hidden=true id="R7dqKx59zlwA"
# the dynamics
eqs = """
    dv/dt = -(v-v_rest) / tau_m : volt (unless refractory)
    """
# build the network
network = NeuronGroup(N_excit + N_inhib, model=eqs, threshold="v>Vth", reset="v=Vr", refractory=tau_ref,
        method="linear")

# set the starting membrane potential 
network.v = v_rest

# make populations
exc_pop = network[:N_excit]
inh_pop = network[N_excit:]

# create the synapses & connect
exc_synapses = Synapses(source=exc_pop, target=network, on_pre="v += J_excit", delay=delay)
exc_synapses.connect(p=epsilon)

inhib_synapses = Synapses(source=inh_pop, target=network, on_pre="v += J_inhib", delay=delay)
inhib_synapses.connect(p=epsilon)

# poisson input
external_poisson_input = PoissonInput(target=network, target_var="v", N=N_poisson_input,
                                      rate=poisson_input_rate, weight=w_poisson)

# generate set of random integers (note: not compatible with numpy)
idx_monitored_neurons = sample(range(N_excit+N_inhib), num_indices)
# make monitors
rate_monitor = PopulationRateMonitor(network)
spike_monitor = SpikeMonitor(network, record=idx_monitored_neurons)
voltage_monitor = StateMonitor(network, "v", record=idx_monitored_neurons)
```

```python hidden=true id="WKv_Nu6szltl"
sim_time = 250*ms

# simulate for this duration
run(sim_time)
```

```python hidden=true id="_azFNizOzlrk"
# plot it
plt.figure(1, figsize=[10,7], dpi=100)

plt.subplot(3,1,1)
plot(spike_monitor.t/ms, spike_monitor.i, '.k', ms=0.08)
xlabel('Time (ms)')
ylabel('Neuron index')

plt.subplot(3,1,2)
plt.title('Voltage traces')
plot(voltage_monitor.t/ms, voltage_monitor.v.T, color='xkcd:teal')
xlabel('Time (ms)')
ylabel('Membrane potential $V_m$')

plt.subplot(3,1,3)
plt.title('Population activity')
plt.plot(rate_monitor.t/ms, rate_monitor.rate, color='xkcd:coral')
plt.xlabel('time (ms)')
plt.ylabel('A(t) (Hz)')
tight_layout()
```

<!-- #region id="KUPkqFbr71rZ" -->
## Disable recurrent feedback


<!-- #endregion -->

<!-- #region id="sMdw5PwCiACA" -->


---

**Exercise:** What would be the population activity caused by the external input only (by the poisson neurons)? Change the network above, but disable the connections between neurons in the network. Use a new coding cell below to do so. Then answer the questions below.

---

```python
w0 = 0 * mV # turn it off!
amplitude =  0.1 * mV
```

```
amplitude: optional. Synaptic weight of the excitatory external poisson neurons onto all neurons in the network. The purpose of this parameter is to see the effect of external input in the absence of network feedback.
```

```python

# set this to True 
random_vm_init = False

```
<!-- #endregion -->

<!-- #region id="ixHHhTwKM0cP" -->
### Your Code
<!-- #endregion -->

```python id="r1rzjW3YMz63"
### your code
```

<!-- #region id="XQIURlq6MygR" -->
### Solution
<!-- #endregion -->

```python id="rrqLBHjc723x"
# redefine params
start_scope()

# default LIF neuron parameters
v_rest = 0. * mV # rest potential
Vr = 10. * mV # reset potential
Vth = 20. * mV # spiking threshold
tau_m = 20. * ms # membrane time scale
tau_ref = 2.0 * ms # absolute refractory period

# default network params
w0 = 0 * mV # off 
w_external = 0.1 * mV

# note: w_ee = w_ei = w0 and w_ie=w_ii = -g*w0
g = 2.5 # relative importance of inhibition
epsilon = 0.1 # connection probbaility
delay = 1.5 * ms # synaptic delay
poisson_input_rate = 14. * Hz
N_poisson_input = 1000 # number of poisson inputs

# neurons
N_excit = 6000
N_inhib = 1500
defaultclock.dt = 0.05 * ms

# weight strengths
J_excit = w0
J_inhib = -g*w0

# === 3.B let's randomise stat state of each neuron ===
random_vm_init = False
# =====================================================

# choose number of indices we want to monitor (for voltagemonitor & populationrate)
num_indices = 5
```

```python id="JHzSN2Gv72-5"
# the dynamics
eqs = """
dv/dt = -(v-v_rest) / tau_m : volt (unless refractory)
"""
# build the network
network = NeuronGroup(N_excit + N_inhib, model=eqs, threshold="v>Vth", reset="v=Vr", refractory=tau_ref,
        method="linear")

# set the starting membrane potential 
network.v = v_rest

# the membrane voltage of each neuron is initialized with a random value drawn from Uniform(v_rest, firing_threshold)
if random_vm_init == True:
    network.v = random.uniform(v_rest/mV, high=Vth/mV, size=(N_excit+N_inhib)) * mV

# make populations
exc_pop = network[:N_excit]
inh_pop = network[N_excit:]

# create the synapses & connect
exc_synapses = Synapses(source=exc_pop, target=network, on_pre="v += J_excit", delay=delay)
exc_synapses.connect(p=epsilon)

inhib_synapses = Synapses(source=inh_pop, target=network, on_pre="v += J_inhib", delay=delay)
inhib_synapses.connect(p=epsilon)

# poisson input
external_poisson_input = PoissonInput(target=network, target_var="v", N=N_poisson_input,
                                      rate=poisson_input_rate, weight=w_poisson)

# generate set of random integers (note: not compatible with numpy)
idx_monitored_neurons = sample(range(N_excit+N_inhib), num_indices)
# make monitors
rate_monitor = PopulationRateMonitor(network)
spike_monitor = SpikeMonitor(network, record=idx_monitored_neurons)
voltage_monitor = StateMonitor(network, "v", record=idx_monitored_neurons)
```

```python id="yH4WAfKg8OxB"
sim_time = 250*ms

# simulate for this duration
run(sim_time)
```

```python id="3SRyurMr8Rnu"
# plot it
plt.figure(1, figsize=[10,7], dpi=100)

plt.subplot(3,1,1)
plot(spike_monitor.t/ms, spike_monitor.i, '.k', ms=0.3)
xlabel('Time (ms)')
ylabel('Neuron index')

plt.subplot(3,1,2)
plt.title('Voltage traces')
plot(voltage_monitor.t/ms, voltage_monitor.v.T/mV)#, color='xkcd:teal')
xlabel('Time (ms)')
ylabel('Membrane potential $V_m$')

plt.subplot(3,1,3)
plt.title('Population activity')
plt.plot(rate_monitor.t/ms, rate_monitor.rate)#, color='xkcd:coral')
plt.xlabel('time (ms)')
plt.ylabel('A(t) (Hz)')
tight_layout()
```

### Questions

---

**Pause and ponder:** Explain why the non-recurrent network shows a strong synchronization in the beginning and why this synchronization fades out.

---

**Question:** The non recurrent network is strongly synchronized in the beginning. Is the connected network simply “locked” to this initial synchronization? You can falsify this hypothesis by initializing each neuron in the network with a random vm. Run the simulation with `random_vm_init=True` to see how the synchronization emerges over time.

---


# Discussion

## Random Input
The input we used in this case was stochastic, represented by random spikes. In an important sense this type of input embodies the least possible number of assumptions about the input, and provides a sense of entire class of possible network behaviors for a given input intensity. However, in the nervous system there is much more structure in input patterns to the network, due to sensory input and specific feedbacks through the environment. When the stimuli to the network are specific, we may be able to extract them from spike trains, thus decoding the neural activity. Encoding and decoding of stimuli are in fact the next step in our exploration of spiking neural networks.

## Random Connectivity
Even though we had two different populations the connectivity across them was random. To what extent are brain networks well represented by random connections? This is certainly a very interesting discussion, but the short version of it is that "as a first approximation it gives us plenty of insights about actual brain data". However, here we were looking mostly into the dynamics of the network. When we start discussing 'encoding and decoding', we will start to see we can start interpreting network structure in the context of behaviors.

<!-- #region id="lSAp5rD9qAHh" -->
# Going Further
These lectures by prof. Gerstner dwell on theoretical aspects of the material of this project:
- [Poisson Process](http://www.youtube.com/watch?v=4r_gc4vf8eE)
- [Neuronal Variability](http://www.youtube.com/watch?v=mmvNRkkJdJs)
- [Sources of Variability](http://www.youtube.com/watch?v=4atNy9DJnPk)
<!-- #endregion -->

<!-- #region id="6HsgdsLhf-72" -->
# References

0. [Neuronal Dynamics 
From single neurons to networks and models of cognition
Wulfram Gerstner, Werner M. Kistler, Richard Naud and Liam Paninski (online book)](https://neuronaldynamics.epfl.ch/online/index.html)
1. [Balanced E and I required for neuronal selectivity
Ran Rubin, L. F. Abbott, Haim Sompolinsky
Proceedings of the National Academy of Sciences Oct 2017, 114 (44) E9366-E9375; DOI: 10.1073/pnas.1705841114](https://www.pnas.org/content/114/44/E9366)
2. [Bistable networks and balanced inhibition](http://www.scholarpedia.org/article/Up_and_down_states#Bistable_networks_and_balanced_inhibition)
3. [Neuronal avalanche](http://www.scholarpedia.org/article/Neuronal_avalanche)
4. [Neuronal noise](http://www.scholarpedia.org/article/Neuronal_noise)
<!-- #endregion -->

<!-- #region id="BGJdmr3g9h8a" -->
# License

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

Mario Negrello, Daphne Cornelise, Elias Santoro.

<!-- #endregion -->
