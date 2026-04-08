---
jupyter:
  jupytext:
    formats: ipynb,md
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region colab_type="text" id="PJ5Nmlj4kOYO" -->
# TODO:

- create specific connections for each sub population
- create spatial projections from sensory neurons to excitatory neurons
- give one stimulus at a time to the network and train
- display the activity of the trained network 
- display the synaptic weights of the trained network 
<!-- #endregion -->

<!-- #region colab_type="text" id="LAGO5euTBpOT" -->
# Introduction

In this project we will demonstrate how plasticity rules can account for the appearance of somatosensory maps. This project will use spiking neurons and STDP, but the same effects could be achieved via sigmoidal units and a Hebbian rule (such as Oja's rule or the BCM rule). The core idea is that input correlations strengthen the connectivity of neurons, because common input leads to neurons that 'fire together', so that they 'wire together'.

## Learning Goals

- assemble a LIF network with excitatory and inhibitory neurons
- assign positions to neurons in a 2D sheet such that they represent a flat cortex
- learn to create input that represents afferent sensory fibers
- learn how to balance the activity of the populations
- apply STDP to the synapses
- analyze the connectivity emerging from the learning rules via adjacency matrices
- display stimulus triggered activity of a population 
<!-- #endregion -->

<!-- #region colab_type="text" id="ZcPZjdtgNNlg" -->
# Overview:

1. Create populations of LIF neurons representing excitatory, inihibitory and sensory populations. Define each population via ```NeuronGroup```
2. Place these neurons *in 2D space*. Use ```np.uniform.random``` to assign random positions to neurons
3. Connect excitatory neurons to inhibitory neurons as a function of proximity. Use ```numpy.pdist``` to calculate the pairwise distances between the neurons
4. Connect sensory neurons to excitatory neurons. Use ```numpy.pdist``` to calculate the pairwise distances between the neurons
5. Introduce a plasticity rule in the synapses between excitatory neurons. Use Brian2 ```Synapse```.
6. Make input patterns representing stimuli to the network
7. Give stimuli to network and simulate (train the network)
8. Analyze the connectivity pattern after training
9. Observe the topgraphical formation of Receptive Fields
<!-- #endregion -->

<!-- #region colab_type="text" id="JLKkSE726jND" -->
## Initialization
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


NOTEBOOK_REQUIREMENTS = [('brian2', 'brian2'), ('Pillow', 'PIL'), ('matplotlib', 'matplotlib'), ('networkx', 'networkx'), ('numpy', 'numpy'), ('requests', 'requests'), ('scipy', 'scipy')]
ensure_notebook_packages(NOTEBOOK_REQUIREMENTS)

if IS_COLAB:
    colab_output.enable_custom_widget_manager()

```

```python colab={} colab_type="code" id="P1agROJrR_l4"
# dependencies
import numpy as np
import matplotlib.pyplot as plt
from scipy import *
from scipy.spatial.distance import *
from brian2 import *
import brian2.numpy_ as np
import networkx as nx

# for reproducability purposes
np.random.seed(2)

```

<!-- #region colab_type="text" id="-JtsU6kqmnfp" -->
# 1. Network and Connections
<!-- #endregion -->

<!-- #region colab_type="text" id="PmNy127-UFTB" -->
Create a 2D network LIF excitatory (E) and inhibitory (I) networks (say 800-200 neurons roughly representing layer 2-3). Neurons should be homogenously (uniformly) distributed in a rectangle of 3 x 4 mm (size of the hand representation of the cortex) with connections based on cell to cell distances.


---

## 1.1 Create Neuron Groups

### Creating Neuron Groups in Brian2

There are a few components you need to start

- **Define the neuronal parameters**. Things like the number of neurons you want, the resting potential and their time constants.
- **Define the differential equation for the LIF neuron**. This is the equation that expresses the dynamics of the neuron model.
- **Create a `Neurongroup`**, a Brian object that brings together (1) paramters, (2) equations and (3) a numerical differential equation solver [> more info <](https://brian2.readthedocs.io/en/stable/user/models.html)

  - A ```NeuronGroup()``` defines a group of neurons that share the same differential equations defined in `eqs`.
  - Differential equations describe the model and define the **state variables** available to the neurongroup, which can be recorded with ```StateMonitor()```. Next to that, `N` (number of neurons) and `i` (index of the neuron a neuron) are assigned automatically).
  - If you want to divide your neurons in inhibitory and excitatory neurons, you can make *subgroups*.

> We go through these steps in the code block below 


<!-- #endregion -->

```python colab={} colab_type="code" id="iRfIrkeUaLFe"
start_scope()
# define numbers of neurons in the excitatory, inhibitory and sensory populations 
N_excit = 1000
N_inhib = 250 # ratio 4:1
N_senso = 100
N_total = int(N_excit+N_inhib)

# # define voltages 
# v_rest = -70*mV # resting potential
# v_reset = -65*mV # reset potential
# v_threshold = -50*mV # spike threshold
# Rm = 10*Mohm*cm**2  # resistance
# Cm = 1.0*uF/cm**2 # The membrane capacitance
# define time constant 


# we use the normalized LIF differential equation. The resting potential is 
# zero and threshold for firing is one. This has the convenience that tuning the model becomes more intuitive.
# 
# Furthermore, we define the x and y coordinates of the neuron in x and y (2d position of somata)
eqs = '''
  dv/dt = (I-v)/tau  : 1 (unless refractory)
  I   : 1
  x   : meter 
  y   : meter
    ''' 

E_thresh = 'v>1'
E_reset  = 'v=0 '
# create two group of neurons, one for excitatory neurons and one for inhibitory neurons
# We create two groups because we may want to change the properties (eq's or parameters)
# of the individual groups.
#
# as threshold use v > 1

E_neurons = NeuronGroup(N_excit, eqs, threshold=E_thresh, reset=E_reset, method='exact', refractory=1*ms)
I_neurons = NeuronGroup(N_excit, eqs, threshold=E_thresh, reset=E_reset, method='exact', refractory=1*ms)
S_neurons = NeuronGroup(N_excit, eqs, threshold=E_thresh, reset=E_reset, method='exact', refractory=1*ms)

```

```python colab={"base_uri": "https://localhost:8080/", "height": 800} colab_type="code" executionInfo={"elapsed": 36642, "status": "error", "timestamp": 1571999341509, "user": {"displayName": "Daphne Cornelisse", "photoUrl": "", "userId": "06723002061441590794"}, "user_tz": -120} id="OFDlC1maDz_r" outputId="6830974c-948f-441c-d3dd-f890079e0fc4"
# Test the Neurons

E_V_monitor = StateMonitor(E_neurons,'v', True)
E_monitor = SpikeMonitor(E_neurons)
I_monitor = SpikeMonitor(I_neurons)
S_monitor = SpikeMonitor(S_neurons)

E_neurons.I = 2
I_neurons.I = 2
S_neurons.I = 1.2

run(100*ms)

figure(figsize=(20, 5))
ax = plt.axes()
plot(E_monitor.t/ms, E_monitor.i, '.', markersize=0.9);
plot(I_monitor.t/ms, I_monitor.i + N_excit, '.', markersize=0.9);
plot(S_monitor.t/ms, S_monitor.i + N_excit + N_inhib, '.', markersize=0.9);
xlabel('time (ms)');
ylabel('neuron index')
plt.title('Excitatory Raster Plot');

```

```python colab={} colab_type="code" id="bvmyFT8HbVV7"
figure(figsize=(20, 5))
ax = plt.axes()
plot(E_V_monitor.t , E_V_monitor.v.T + E_V_monitor.i*10); # Transpose is necessary to plot all variables
xlabel('time (ms)');
ylabel('neuron index')
plt.title('VoltagesPlot');

```

<!-- #region colab_type="text" id="B6EMSeg1oLW_" -->
## 1.2 Assign a randomly generated location to each neuron 

Now that we have a network of neurons, we want to give it a location in space. 

We will assume that the somatosensory projection cortex is roughly in a patch of cortex of about 3mm x 4mm. To do that we:

For each dimension (x and y):
1. draw a random number from the uniform distribution between 0 and 1
2. multiply it by the range, such that we have positions between 0 and range
3. assign it to the coordinates in the neuron group.

A few details from the brian [> documentation <](https://brian2.readthedocs.io/en/stable/reference/brian2.groups.neurongroup.NeuronGroup.html#brian2.groups.neurongroup.NeuronGroup)
- Notes:
  - in `brian2`, using $i$ in a string in brian stands for the index, `network.x` below does that operation for each neuron in the network.
  - use the lower dash ( `_` ) to display the state variable without unit
  - if you see something like `x : 1` it means we define a variable to be dimensionless/unitless
<!-- #endregion -->

```python colab={} colab_type="code" id="nqCZ2d9fzg-6"
# by default, the neurons in the network don't have any coordinates, 
# thats why it is an array full of zeros. To see that uncomment the 
# following line.

# network.x_

```

```python colab={} colab_type="code" id="jhgBzTVzCV28"
# we assign a random number for each element in the array drawn from a 
# uniform distribution.
# 
# To place the neurons in a sheet of 3 x 4 mm we multiply the random number by 
# the range for each coordinate.
#
# notice we multiply our random number [0,1] by 3000 so that we get a random number
# between 0 and 3000.
pos_range = [3000, 4000]
pos_range = [3000, 4000]

E_neurons.x = np.random.uniform(size=(N_excit)) * pos_range[0] * umeter
E_neurons.y = np.random.uniform(size=(N_excit)) * pos_range[1] * umeter

I_neurons.x = np.random.uniform(size=(N_inhib)) * pos_range[0] * umeter
I_neurons.y = np.random.uniform(size=(N_inhib)) * pos_range[1] * umeter

S_neurons.x = np.random.uniform(size=(N_senso)) * pos_range[0] * umeter
S_neurons.y = np.random.uniform(size=(N_senso)) * pos_range[1] * umeter


# NOTE: it is possible to do this with a for loop:
# for i in range(N_total):
  # network.x[i] = np.random.uniform(size=(N_total, 1) * 4000 * umeter

```

<!-- #region colab_type="text" id="xQ781wUDKl8-" -->
We created two arrays with random numbers that represent the neurons in space. Let's see if it worked out
<!-- #endregion -->

```python colab={} colab_type="code" id="2-tbyvgSJy7c"
figure(figsize=(10, 5))
scatter(E_neurons.x/umeter, E_neurons.y/umeter, color='r', label='Excitatory')
scatter(I_neurons.x/umeter, I_neurons.y/umeter, color='b', label='Inhibitory')
xlabel('x (umeter)')
ylabel('y (umeter)')
legend();

```

<!-- #region colab_type="text" id="Ex4iyT5Vb_V5" -->
## 1.3 Rules for Connecting Neurons

Cortical neurons are more likely to connect to each other if they are close. To start connecting our network we compute a distance matrix that gives us the pairwise distance between two neurons in our network. Note also that connectivity in the cortex is in general not symmetrical. In this present example network we will use the following system for connecting the populations:

- Excitatory to Excitatory : connected if distance < 1000um and with probability 50%

- Excitatory to Inhibitory : connected if distance < 1000um and with probability 50%

- Inhibitory to Excitatory : connected if distance < 500 um 

- Inhibitory to Inhibitory : not connected


<!-- #endregion -->

<!-- #region colab_type="text" id="ZmpZRPZWarao" -->
### 1.3.1 Create a distance matrix

We calculate the standard [euclidean distance](https://pythonprogramming.net/euclidean-distance-machine-learning-tutorial/) between each neuron using [`pdist`](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.distance.pdist.html) and convert this into a square ($n$ x $n$) matrix.

As you can see, the distances between the neurons is spread uniformly as we expect.

The `imshow` function in matplotlib is a nice way to visualise larger matrices (as it becomes hard to get a clear overview of 100x100 or larger matrices!). `pcolor` is another function that does the trick, but it reverses the matrix. 
<!-- #endregion -->

```python colab={} colab_type="code" id="7XHV1N52S6Ms"
# combine the x and y coordinates into one matrix
E_positions = list(zip(E_neurons.x,E_neurons.y))
I_positions = list(zip(I_neurons.x,I_neurons.y))

# we stack vertically the arrays with x and y positions of E and I neurons
# to pass them to the distance matrix calculation.

positions = np.vstack((E_positions/um, I_positions/um))

# The (( )) double parenthesis are ugly but necessary because vstack takes tupples ( , ) as input 

```

```python colab={"base_uri": "https://localhost:8080/", "height": 281} colab_type="code" executionInfo={"elapsed": 2863, "status": "ok", "timestamp": 1569942603960, "user": {"displayName": "Daphne Cornelisse", "photoUrl": "", "userId": "06723002061441590794"}, "user_tz": -120} id="UStWO7ZxYWGA" outputId="3cf356af-aaba-46e9-a63e-d08e873226ec"
# compute the distance between each pair of points (pdist)
# then convert a vector-form distance vector to a square distance matrix
D = squareform(pdist(positions, metric='euclidean'))

# plot connections
plt.imshow(D)
plt.title('Distance matrix')
plt.colorbar();

```

<!-- #region colab_type="text" id="eHXgp1pebuT1" -->
Note that the above matrix is symmetrical ($d_{ij}=d_{ji}$).




<!-- #endregion -->

<!-- #region colab_type="text" id="kP2lRKRsGhZc" -->
### Bonus: sort the distance matrix 

We can write a little function that makes the hierarchical clustering structure in the distance matrix obvious. See: https://gmarti.gitlab.io/ml/2017/09/07/how-to-sort-distance-matrix.html

<!-- #endregion -->

<!-- #region colab_type="text" id="HmIHtXzrc8Nm" -->
### 1.3.2 From distances to connecitivity: Adjacency Matrix

To connect our populations we will follow these general principles :

- Neurons close by are more often connected to each other. The connection radius is specific for population (for example, excitatory neurons tend to reach further than inhibitory neurons)
- The connectivity between the populations is *not* necessariy reciprocal.
- The connectivity between the neurons happens with a certain **probability**.  

Below we will connect neurons through synapses when then are spatially close and with a certain probability.

Once we have the distance matrix we can easily create an adjacency matrix. We simply find the entries $d_{ij}$ in the distance matrix (which is the distance between neurons i (from/rows) and j (to/columns) that are smaller than a given radius  $d_{ij} < R$. where the radius $R$ represents the **dendritic span** of the arborization of the neurons. 

This produces an adjacency matrix where every entry $w_{ij}$ is $1$ when there is a connection and $0$ where there is none.

> Note: The connections described above are deterministic, but in the brain we talk about the 'probability' of a connection. There are many ways of making connections probabilistic. One simple idea is to: 
- for each non zero entry in the adjacency matrix
- draw a uniform random number between [0,1]
- check if that is larger than a given probability. For example, if we want 80% of the neurons to be connected, the probability of no connection is 20%. So for there to be a connection, the random number has to be larger than 0.2.

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 228} colab_type="code" executionInfo={"elapsed": 1968, "status": "ok", "timestamp": 1569942606238, "user": {"displayName": "Daphne Cornelisse", "photoUrl": "", "userId": "06723002061441590794"}, "user_tz": -120} id="yiSObncfcW5w" outputId="75c5f63a-fcf2-440f-8a4d-e4d7323b61c2"
radius = 500 # maximum connection distance, 1mm (1000um)

# we want to only plot connections from neurons that are close to each other
# if distance < radius (in this case 1mm), then put a connection (1) otherwise 0
nearby = (D < radius)*1 
# plot connection matrix

figure(figsize=(10,5))
plt.subplot(1,3,1)
plt.title('Within a radius')
plt.imshow(nearby);

# remove self-connections (multiply elementwise by a matrix with zeros in the main diagonal)
Adj = nearby * (1-np.eye(N_total))
# plot connection matrix
plt.subplot(1,3,2)
plt.title('No self connections')
plt.imshow(Adj);

# assign existance of a connection with a probability of 0.8
#  to each connection that is within the given radius:

Adj = (Adj * np.random.uniform(size=(N_total, N_total))>=.2)*1

# plot connection matrix
plt.subplot(1,3,3)
plt.title('Randomized Connections')
plt.imshow(Adj);

```

<!-- #region colab_type="text" id="O3WNhMpWYQy2" -->
### 1.3.3 Excitatory neurons reach far, Inhibitory are close by

In the matrices above we have connected all neurons to all neurons irrespectively of their class. We have to be more specific than that!

Neurons in the cortex do not all have the same radius of connectivity. In general excitatory neurons reach further than inhibitory neurons. So the radius for connectivity betweeen E-E neurons and I-E neurons are different.

That is, we will have to index the connectivity according to the population of neurons.



<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 499} colab_type="code" executionInfo={"elapsed": 1218, "status": "ok", "timestamp": 1569943312339, "user": {"displayName": "Daphne Cornelisse", "photoUrl": "", "userId": "06723002061441590794"}, "user_tz": -120} id="m7_kKZk0ZQLD" outputId="0a5ab609-ddde-4e33-f78e-2df6928406e2"
### This code block will make heavy use of indexing. Thus consider reviewing numpy indexing rules.

# Note: for a later use we will be indexing the types of connections
#  via labels 1,2,3 to represent E-E, E-I and I-E connections. 
# This is a convenient way to create connections later via 'np.where'.

# First, let's state the connectivity parameters
radiusEE = 500
probEE = .7

radiusEI = 500
probEI = .7

radiusIE = 900
probIE = .9

# let's make a connection matrix 'W' from our adjancency matrix

W = Adj;

# Create distance based E-E probabilistic connections (with prob. >= 90%)
W[:N_excit,:N_excit:] = (D[:N_excit,:N_excit] < radiusEI)*1 # multiply by one to change True and False to a number
W[:N_excit,:N_excit:] = W[:N_excit,:N_excit] * ( ( np.random.uniform(size=(N_excit,N_excit))>= probEE) * 1)

# Remove all I-I connections
W[N_excit:,N_excit:] = 0

# Create distance based E-I probabilistic connections (with prob. >= 90%)
W[:N_excit,N_excit:,] = (D[:N_excit,N_excit:] < radiusEI)*1 # multiply by one to change True and False to a number
W[:N_excit,N_excit:,] = W[:N_excit,N_excit:] * ( ( np.random.uniform(size=(N_excit,N_inhib))>= probEI) *2)

# Create distance based I-E probabilistic connections (>= 90%)
W[N_excit:,:N_excit] = (D[N_excit:,:N_excit] < radiusIE)*1
W[N_excit:,:N_excit] = W[N_excit:,:N_excit] * ( ( np.random.uniform(size=(N_inhib,N_excit))>= probIE) *3)

# Remove all self connections
W = W * (1-np.eye(N_total))

figure(figsize=(8,8))
#spy(W, markersize=0.8, color='k')
imshow(W,cmap='Blues')
title(' population connectivity : within and across');

```

<!-- #region colab_type="text" id="_kyaiWL4GN5I" -->
Can you interpret the matrix as connections between the different populations ? The colors repesent the different types of source-target populations. Can you spot in the code above how we did that?
<!-- #endregion -->

<!-- #region colab_type="text" id="tFWNDlqzpmvl" -->
### 1.3.3 Visualise connectivity
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 336} colab_type="code" executionInfo={"elapsed": 3669, "status": "ok", "timestamp": 1569942349089, "user": {"displayName": "Daphne Cornelisse", "photoUrl": "", "userId": "06723002061441590794"}, "user_tz": -120} id="2eX_FMHwmsIT" outputId="5d5ef4c4-5bdf-4301-e439-96933690b348"
# give each neuron a number, and put in a dictionary
node_idx = list(range(0,N_total))

pos_dict = dict(zip(node_idx,positions))

# create a list of labels to color the neurons
# note: 5 and 10 here work as label colors for drawing the graph
node_colors = np.concatenate(  (np.ones((N_excit))*2,np.ones((N_inhib))*5) )

# construct the graph from the neuron positions and adjacency matrix
net_graph = nx.from_numpy_matrix(W, pos_dict);

# get connections and strengths of connections
edges, weights = zip(*nx.get_edge_attributes(net_graph,'weight').items())

fig, ax = plt.subplots(figsize=(10, 5))
plt.title('neurons and connections')
nx.draw(net_graph, pos_dict, node_size=40, node_color=node_colors, edgelist=edges, edge_color=weights, edge_cmap=plt.cm.tab10)
ax.set_axis_on() # turn the axis on
ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

```

<!-- #region colab_type="text" id="mkxryEbjfu1w" -->
### 1.3.4 Use the Matrices to Create Synapses in Brian 2

To connect neurons in Brian2 we need to initialize the synapses and use the  function 'connect' : https://brian2.readthedocs.io/en/stable/user/synapses.html.

As Brian2 cannot take an adjacency matrix (unfortuntely!), we will be passing the connections as pairs of pre-post as such

```python
S.connect(i=[1, 2, 3], j=[10, 11, 12])
```
see: https://brian2.readthedocs.io/en/stable/user/synapses.html

So we must first obtain the pairs of connections from our adjacency matrix. We do it via ```numpy.where``` as such

```python
EE_pairs = np.where(W==1)
```

This returns the indices i and j for every connected pair of type 1 (E-E).


<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 88} colab_type="code" executionInfo={"elapsed": 21742, "status": "ok", "timestamp": 1569942384735, "user": {"displayName": "Daphne Cornelisse", "photoUrl": "", "userId": "06723002061441590794"}, "user_tz": -120} id="-7_VpAlpEu_8" outputId="8f2e1dd7-a5c2-48b9-b1fd-a8524976d937"
# Define synaptic current model:
current_pulse_synapse = '''w:1 '''

# When the presynaptic neuron spikes, we add 'w' current to the postsynaptic neuron
on_excitatory_spike = 'v=+w'
on_inhibitory_spike = 'v=+ -w'

# event_driven_decay_current_synapse = '''w:1
#          dApre/dt=-Apre/taupre : 1 (event-driven)
#          dApost/dt=-Apost/taupost : 1 (event-driven)'''

# Create E-E connections
EE_pairs = np.where(W==1) # here we return a list of to-from pairs for connections type 1 (E-E)
EE_syn   = Synapses(source = E_neurons, target=E_neurons, model = current_pulse_synapse , on_post=on_excitatory_spike)  
EE_syn.connect( i = EE_pairs[0][:] , j = EE_pairs[1][:])

# # Create E-I connections
EI_pairs = np.where(W==2) # here we return a list of to-from pairs for connections type 2 (E-I)
EI_syn   = Synapses(source = E_neurons, target=I_neurons, model = current_pulse_synapse, on_post=on_excitatory_spike)  
EI_syn.connect(i = EI_pairs[0][:] , j = EI_pairs[1][:]-N_excit)

# # Create I-E connections
IE_pairs = np.where(W==3) # here we return a list of to-from pairs for connections type 3 (I-E)
IE_syn   = Synapses( source = I_neurons, target=E_neurons, model = current_pulse_synapse, on_post=on_inhibitory_spike)  
IE_syn.connect(i = IE_pairs[0][:]-N_excit , j = IE_pairs[1][:])

```

<!-- #region colab_type="text" id="OICgsEaQoiXD" -->
### 1.4.1 Plot connections of selected single cells 

(to see their range)
<!-- #endregion -->

```python colab={} colab_type="code" id="k2OMIjOrioUy"
# figure(figsize=(10, 5))
# # choose an index
# idx = 5
# # Show the connections for some neurons in different colors
# subplot(1, 2, 1)
# title('Excitatory cell')

# plot(network.x[idx] / umeter, network.y[idx] / umeter, 'o', mec='g', mfc='none')
# plot(network.x[Ce.j[idx, :]] / umeter, network.y[Ce.j[idx, :]] / umeter, 'g' + '.')

# subplot(1, 2, 2)
# title('Inhibitory cell')
# plot(network.x[idx] / umeter, network.y[idx] / umeter, 'o', mec='k', mfc='none')
# plot(network.x[Ci.j[idx, :]] / umeter, network.y[Ci.j[idx, :]] / umeter, 'k' + '.');

```

<!-- #region colab_type="text" id="KOOQJkkjmdIl" -->
# 2. Parameterising and Balancing
<!-- #endregion -->

<!-- #region colab_type="text" id="fUeQBTe7MuXP" -->
## 2.1 Spontaneous Network Activity on Current Injection
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 831} colab_type="code" executionInfo={"elapsed": 1463, "status": "error", "timestamp": 1569934091497, "user": {"displayName": "Mario Negrello", "photoUrl": "https://lh3.googleusercontent.com/a-/AAuE7mBKcuM3zrzWydsMeTlpor_04SYP_Fx-VlyQP2mgYpk=s64", "userId": "10136788594790905986"}, "user_tz": -120} id="vMyRV9j7M1Nb" outputId="a2cd6611-db82-4c91-f34d-267ce0e3a7a4"
M = SpikeMonitor(E_neurons)

E_neurons.v0 = 0
run(100*ms)

figure(figsize=(10, 5))
subplot(1, 2, 1)
plot(M.t/ms, M.i, '.', markersize=0.9);
xlabel('time (ms)');
ylabel('neuron index')
xlim(0, sim_time/ms)
plt.title('Excitatory Raster Plot');

```

<!-- #region colab_type="text" id="8jePPLEWpMP4" -->
## 2.2 Balance the network 
Tweak synaptic parameters such that the network keeps a low average firing rate (1Hz), with inhibitory cells a bit higher (~10Hz). Base synapse values between populations are drawn from a random distribution such that rates above are respected.

- Plot raster and histogram with activity of neurons for 2s of spontaneous activity
- Iteratively select excitatory and inhibitory synaptic weights to keep net balanced (betwen g_syn max and min)



---



<!-- #endregion -->

<!-- #region colab_type="text" id="oiqQWfIBQXMw" -->

### 2.2.1 Choosing a Monitor

In `brian` you record activity of a `NeuronGroup` using a Monitor. Since we are interested in recording the frequency of spikes only, we use `SpikeMonitor`. 

[> Spikemonitor <](https://brian2.readthedocs.io/en/2.0rc/reference/brian2.monitors.spikemonitor.SpikeMonitor.html#brian2.monitors.spikemonitor.SpikeMonitor) records the spikes so you can display raster plots. 

<!-- #endregion -->

```python colab={} colab_type="code" id="r_EbF-hKpRIa"
# set the initial membrane potential
# use network.v = 'rand()' to initialise each neuron with a random uniform value between 0 and 1
network.v = 0

# define spike monitors 
M = SpikeMonitor(network)
ME = SpikeMonitor(network_excit)
MI = SpikeMonitor(network_inhib)

```

<!-- #region colab_type="text" id="BqUoNu67O1-W" -->
### 2.2.2 Adding spikes

We have set the threshold, reset potential etc. before, and now set the initial membrane potential. To create some spikes, we have to apply current to our group of neurons.
<!-- #endregion -->

```python colab={} colab_type="code" id="AHSvoT39VY-L"
network.I = 0
run(100*ms)
network.I = 5
run(100*ms)



# define total simulation time (running time)
sim_time = 200*ms

# # run simulation
# run(sim_time)

```

<!-- #region colab_type="text" id="4TYNbEZFVa3n" -->
### 2.2.3 Plot raster plots

By now we have seen the raster plot many times, the plot that marks the neural activity: a spike from a neuron at a specified position.

In brian, if `M` is your spikemonitor then

- `M.i` gives the corresponding neuron index for each spike
- `M.t` includes the times of all the spikes

Thats why you plot the `M.t` on the x-axis and the `M.i` on the y-axis.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 369} colab_type="code" executionInfo={"elapsed": 932, "status": "ok", "timestamp": 1569852529042, "user": {"displayName": "Daphne Cornelisse", "photoUrl": "", "userId": "06723002061441590794"}, "user_tz": -120} id="-H_x43BDOxMG" outputId="b44fe079-faad-49e8-88d8-684a20c48b7f"
figure(figsize=(10, 5))
subplot(1, 2, 1)
plot(M.t/ms, M.i, '.', markersize=0.9);
xlabel('time (ms)');
ylabel('neuron index')
xlim(0, sim_time/ms)
plt.title('Excitatory Raster Plot');

subplot(1, 2, 2)
plot(MI.t/ms, MI.i, '.', markersize=0.9);
xlabel('time (ms)');
ylabel('neuron index')
xlim(0, sim_time/ms)
plt.title('Inhibitory Raster Plot');
tight_layout()

# figure(figsize=(10,2.5))

# subplot(1, 2, 2)
# #  plot applied current (stimulation)
# plot(M.t/ms, M.I/namp, label='I')
# xlabel('Time (ms)')
# ylabel('Applied current (nA)')
# legend();

```

```python colab={} colab_type="code" id="5vxywEoZYwl4"

```

<!-- #region colab_type="text" id="0EGvVVN4O2L5" -->
### 2.2.4 Balancing the network by tweaking parameters

You can check if the spike frequency is correct (i.e. if the network is balanced) by counting the number of spikes per second. Brian luckily has some built in functions for this to help.

- `num_spikes` returns the total number of spikes recorded. So using `M.num_spikes / num of seconds` will give the frequency.
- `spike_trains()` returns a dictionary mapping spike indices to arrays of spike times.





Do
> - use spike monitor to display the raster plot of the network for a few seconds seconds
> - observe the number of spikes per second for each second/ calculate with brian
> - take the average of that number, this is the average firing rate
> - if avg firing rate != 1, tweak synaptic parameters until desired Hz



<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" executionInfo={"elapsed": 392, "status": "ok", "timestamp": 1569752086525, "user": {"displayName": "Daphne Cornelisse", "photoUrl": "", "userId": "06723002061441590794"}, "user_tz": -120} id="M0KNHzY1CQjb" outputId="eb8546fe-b5ca-4d85-e009-b65d62b94917"
# this is how many spikes in total
ME.num_spikes/second

```

```python colab={"base_uri": "https://localhost:8080/", "height": 54} colab_type="code" executionInfo={"elapsed": 374, "status": "ok", "timestamp": 1569752088097, "user": {"displayName": "Daphne Cornelisse", "photoUrl": "", "userId": "06723002061441590794"}, "user_tz": -120} id="oKzDFVuWJxOn" outputId="7a3b46ac-cb21-4adc-cbb2-b4911c5a4255"
# this is an array of the number of spikes each neuron in the group fired. Dividing this by the duration of the run gives the firing rate.
ME.count/sim_time

```

```python colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" executionInfo={"elapsed": 542, "status": "ok", "timestamp": 1569757194348, "user": {"displayName": "Daphne Cornelisse", "photoUrl": "", "userId": "06723002061441590794"}, "user_tz": -120} id="iQ857_8hHWoI" outputId="1bf11f6b-2ca6-4a96-dc2b-d0ab9c59ab15"
MI.num_spikes/second

```

<!-- #region colab_type="text" id="tGmyMwZA1B9x" -->
How does the firing rate change when we change the parameter $\tau$ for example? This script tests exactly that
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 283} colab_type="code" executionInfo={"elapsed": 14376, "status": "ok", "timestamp": 1569851991103, "user": {"displayName": "Daphne Cornelisse", "photoUrl": "", "userId": "06723002061441590794"}, "user_tz": -120} id="qcWCIN610bYX" outputId="d08aa842-0c3c-4456-b931-b524ca0507e3"
# create different time constants
tau_range = linspace(1, 10, 30)*ms # start,stop,num
output_rates = []

# Store the current state of the network
store()
for tau in tau_range:
    # Restore the original state of the network
    restore()
    # Run it with the new value of tau
    run(1*second)
    output_rates.append(M.num_spikes/second)
plot(tau_range/ms, output_rates)
xlabel(r'tau (ms)')
ylabel('Firing rate (spikes/s)');

```

<!-- #region colab_type="text" id="jl9TKd_FfLpI" -->
# 3. Receptive fields 
<!-- #endregion -->

<!-- #region colab_type="text" id="aclMRh9MpMlY" -->
## 3.1 Create receptor layer (input layer)

Create a 2D network representing the hand pressure receptors (R) (±3000 receptors per finger tip), mapping to a 10 x 5cm sheet. Create divergent and topographical projections from this layer onto the 'cortical' receiving layer with a certain arrival radius (~300um).


---




<!-- #endregion -->

<!-- #region colab_type="text" id="j_POLOFpkykz" -->
### 3.1.1 Read in a hand from image or create one



### 3.1.2 Convert image to binary array of size n x n
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 286} colab_type="code" executionInfo={"elapsed": 442, "status": "ok", "timestamp": 1569852504018, "user": {"displayName": "Daphne Cornelisse", "photoUrl": "", "userId": "06723002061441590794"}, "user_tz": -120} id="T9x8V734p25E" outputId="23429995-42a7-42bb-cc99-dab8732ce4c9"
hand = np.array([
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 1, 0, 0, 0],
      [0, 0, 1, 1, 1, 0, 0],
      [0, 1, 1, 1, 1, 1, 0],
      [0, 1, 1, 1, 1, 1, 0],
      [0, 1, 1, 1, 1, 1, 0],
      [0, 0, 1, 1, 1, 1, 0],
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0],
      ])

imshow(hand, cmap='gist_gray')

```

```python colab={} colab_type="code" id="3lMBKkILm08U"
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

```

```python colab={} colab_type="code" id="cY5Le-fApj85"
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# get image from url
response = requests.get('https://clipartion.com/wp-content/uploads/2015/11/clip-artcarrie-teaching-first-hands-and-feet-doodles-with.jpg')
img = Image.open(BytesIO(response.content))

```

```python colab={"base_uri": "https://localhost:8080/", "height": 136} colab_type="code" executionInfo={"elapsed": 462, "status": "ok", "timestamp": 1569755182672, "user": {"displayName": "Daphne Cornelisse", "photoUrl": "", "userId": "06723002061441590794"}, "user_tz": -120} id="WdAJAwNZzVFa" outputId="487e1c12-0c14-42a1-8307-c83916467f23"
im = img.convert('L')
np_img = np.array(im)
np_img = ~np_img  # invert B&W
np_img[np_img > 0] = 1

np_img

```

```python colab={"base_uri": "https://localhost:8080/", "height": 269} colab_type="code" executionInfo={"elapsed": 959, "status": "ok", "timestamp": 1569755298354, "user": {"displayName": "Daphne Cornelisse", "photoUrl": "", "userId": "06723002061441590794"}, "user_tz": -120} id="KgEUbAuwjueY" outputId="46d421bd-8c37-43a6-e230-d5a99e6b2086"
imshow(np_img, cmap='gist_gray');

```

```python colab={"base_uri": "https://localhost:8080/", "height": 867} colab_type="code" executionInfo={"elapsed": 464, "status": "ok", "timestamp": 1569755607421, "user": {"displayName": "Daphne Cornelisse", "photoUrl": "", "userId": "06723002061441590794"}, "user_tz": -120} id="3D6dGyG8kQyo" outputId="a5f3a4ef-66f8-4199-a1d0-129cdf609550"
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('/content/hand.jpg')
print(img)

img.shape

```

```python colab={"base_uri": "https://localhost:8080/", "height": 269} colab_type="code" executionInfo={"elapsed": 661, "status": "ok", "timestamp": 1569755383943, "user": {"displayName": "Daphne Cornelisse", "photoUrl": "", "userId": "06723002061441590794"}, "user_tz": -120} id="PHiKBLLPjloG" outputId="64251a1d-d985-4d70-e975-56219d726dbf"
imgplot = plt.imshow(img)

```

```python colab={"base_uri": "https://localhost:8080/", "height": 269} colab_type="code" executionInfo={"elapsed": 399, "status": "ok", "timestamp": 1569755395989, "user": {"displayName": "Daphne Cornelisse", "photoUrl": "", "userId": "06723002061441590794"}, "user_tz": -120} id="YPOLCeiU0jP0" outputId="21c22023-6f5f-4950-89ea-a4e127b0934d"
from PIL import Image
img = Image.open('/content/hand.jpg')
img.thumbnail((50, 50), Image.ANTIALIAS)  # resizes image in-place
imgplot = plt.imshow(img)

```

```python colab={"base_uri": "https://localhost:8080/", "height": 269} colab_type="code" executionInfo={"elapsed": 471, "status": "ok", "timestamp": 1569755577779, "user": {"displayName": "Daphne Cornelisse", "photoUrl": "", "userId": "06723002061441590794"}, "user_tz": -120} id="7O2dVkvo0jUf" outputId="c229ff43-0f99-43c3-fad0-507424ed1849"
img2 = Image.open('/content/black_hand.png')
img2.thumbnail((50, 50), Image.ANTIALIAS)  # resizes image in-place
imgplot = plt.imshow(img2)

```

```python colab={} colab_type="code" id="3UGvZNVbuXKv"

```

```python colab={"base_uri": "https://localhost:8080/", "height": 283} colab_type="code" executionInfo={"elapsed": 1430, "status": "ok", "timestamp": 1569756649085, "user": {"displayName": "Daphne Cornelisse", "photoUrl": "", "userId": "06723002061441590794"}, "user_tz": -120} id="zTDjlNoAnsWH" outputId="5a1ac73e-8563-4c22-df2d-53d5eea9f7fd"
### BRIAN EXAMPLE ###

start_scope()
from matplotlib.image import imread
img = (1-imread('/content/black_hand.png'))[::-1, :, 0].T
num_samples, N = img.shape
ta = TimedArray(img, dt=1*ms) # 228
A = 1.5
tau = 2*ms
eqs = '''
dv/dt = (A*ta(t, i)-v)/tau+0.8*xi*tau**-0.5 : 1
'''
G = NeuronGroup(N, eqs, threshold='v>1', reset='v=0', method='euler')
M = SpikeMonitor(G)
run(num_samples*ms)
plot(M.t/ms, M.i, '.k', ms=3)
xlim(0, num_samples)
ylim(0, N)
xlabel('Time (ms)')
ylabel('Neuron index');

```

<!-- #region colab_type="text" id="OGWs8su8muc_" -->
## 3.2 Create stimuli representing fingers (and their combinations)
Create 5 groups that represent the fingers that make topographical projections to the cortex.

- 100 x 50 - 5000 px
- Create 10 stimuli representing different finger activation patterns
<!-- #endregion -->

```python colab={} colab_type="code" id="g4C6H3UdmltO"

```

<!-- #region colab_type="text" id="e5ABc5jdmwvr" -->
## 3.3 Present stimuli and record average activities 

- for each stimulus plot raster plot
<!-- #endregion -->

```python colab={} colab_type="code" id="CsU-IXymmmBk"

```

<!-- #region colab_type="text" id="NDVvRLr1vl3H" -->
# 4. STDP
<!-- #endregion -->

<!-- #region colab_type="text" id="CmpT95Whvpub" -->
## 4.1 Connect  STDP synapses
- Introduce modifiable synapses between Exc-Exc  and Rec-Exc neurons (with given plasticity rates)
- Let network self-organize

<!-- #endregion -->

```python colab={} colab_type="code" id="-ZiqN6Igvxfk"

```

<!-- #region colab_type="text" id="oSFL1dlsvp9z" -->
## 4.2 Plot new weight matrix and compare
<!-- #endregion -->

```python colab={} colab_type="code" id="m8TMSZZPvzn9"

```

<!-- #region colab_type="text" id="P0pB5L-BvqG-" -->
## 4.3 Plot connections between neurons
<!-- #endregion -->

```python colab={} colab_type="code" id="8-Yh8Qdev35f"

```

<!-- #region colab_type="text" id="DQ1DEmI1pMq-" -->
## 4.4 Plot receptive field maps

## Exploration:
- Lesion a finger and let the network learn again
- What are the effects of inhibitory plasticity?
- Is there evidence for it in the scientific literature?
- What would happen with visual input?

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 411} colab_type="code" executionInfo={"elapsed": 55147, "status": "ok", "timestamp": 1569684659205, "user": {"displayName": "Daphne Cornelisse", "photoUrl": "", "userId": "06723002061441590794"}, "user_tz": -120} id="KkoxAMubr2dz" outputId="82534ced-66b1-45c7-c8e8-6f2e126df50c"
start_scope()

taum = 10*ms
taupre = 20*ms
taupost = taupre
taue = 5*ms
F = 15*Hz
gmax = .01
dApre = .01
dApost = -dApre * taupre / taupost * 1.05
dApost *= gmax
dApre *= gmax

# the differential equation used
eqs = """
        dv/dt = -v/tau : volt 
            x : meter
            y : meter 
    """
## Create group of neurons ## network is linear
network = NeuronGroup(N_total, eqs, threshold="v>firing_threshold", reset="v=v_reset")
network.v = v_rest

# initialize the grid positions
rows = 30
cols = 40
grid_dist = 25*umeter
network.x = '(i // rows) * grid_dist - rows/2.0 * grid_dist'
network.y = '(i % rows) * grid_dist - cols/2.0 * grid_dist'

# Make subgroups
excit_population = network[:N_excit]
inhib_population = network[N_excit:]

# Deterministic connections
distance = 120*umeter

## Define Synaptic model and connect ##
## Define the weights, J is the postsynaptic ##
J_excit = 0.1*mV
J_inhib = -0.4*mV

## Define Synaptic model and connect ##
# on_pre: when presynaptic spike arrives at a synapse, what happens to the postsynaptic variable
C_i = Synapses(inhib_population, target=network, on_pre="v += J_inhib")
C_i.connect(condition = 'sqrt((x_pre - x_post)**2 + (y_pre - y_post)**2) < distance')
#C_i.w = 1.2

 # Spike monitors
SM_E = SpikeMonitor(network[:N_excit])
SM_I = SpikeMonitor(network[N_excit:]) 

C_e = Synapses(excit_population, excit_population,
             '''w : 1
                dApre/dt = -Apre / taupre : 1 (event-driven)
                dApost/dt = -Apost / taupost : 1 (event-driven)''',
             on_pre='''
                    Apre += dApre
                    w = clip(w + Apost, 0, gmax)
                    v_post+=w*J_excit''',
             on_post='''Apost += dApost
                     w = clip(w + Apre, 0, gmax)
                     v_post+=w*J_excit''')
C_e.connect()

mon = StateMonitor(C_e, 'w', record=[0, 1])

# run the simulation
run(sim_time, report='text')

## PLOTTING ##
subplot(311)
plot(C_e.w, '.k')
ylabel('Weight / gmax')
xlabel('Synapse index')
subplot(312)
hist(C_e.w, 20)
xlabel('Weight / gmax')
subplot(313)
plot(mon.t/second, mon.w.T/gmax)
xlabel('Time (s)')
ylabel('Weight / gmax')
tight_layout()
show()

```

```python colab={} colab_type="code" id="PnpWUTRPvza9"

```

<!-- #region colab_type="text" id="4xlY74AZxnSi" -->
Copyright (c) Mario Negrello, Daphne Cornelise, all rights reserved. 
<!-- #endregion -->

<!-- #region colab_type="text" id="drXf_oOFxnf5" -->
## Useful Resources

- Brian plotting [code](https://github.com/brian-team/brian2tools/blob/master/docs_sphinx/user/plotting.rst#id8)
<!-- #endregion -->

```python colab={} colab_type="code" id="295qoO5exs6C"

```
