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

<!-- #region colab_type="text" id="IoduKOLN3FIu" -->
# Introduction

In the project [networks and graphs], you created matrices that represented networks. In this notebook you will learn how we can represent activity propagation in networks by multiplying an activity vector by the adjacency matrices. As you work through this you will also acquire intuition about the evolution of the network state as a function of network structures.

## Learning Goals
- Understand activity propagation in discrete networks via vector-matrix multiplication.
- Understand the use of saturating non-linearities -- so called, 'transfer functions' to keep output bounded.
- Understand propagation of activity in feed forward and recurrent neural networks.
- Aquire intuition about the evolution of the **activity state** for different network configurations.
<!-- #endregion -->

<!-- #region colab_type="text" id="O2VrNgDPFHbP" -->
## Key Terms
- iterations
- saturation 
- non-linearity
- transfer-function
- ring network

<!-- #endregion -->

<!-- #region colab_type="text" id="ky_glRRYFWuw" -->
# Pre-requisites
- Adjacency Matrix
- Vector Matrix Multiplication
- Computing functions
<!-- #endregion -->

<!-- #region colab_type="text" id="tWf9eOtM2p2G" -->
## Initialization
<!-- #endregion -->

```python colab={} colab_type="code" id="lwjlF8Uv2hzf"
import numpy as np
from numpy import zeros
import matplotlib.pyplot as plt
import networkx as nx
from scipy.special import expit # fast sigmoid computation

from IPython.display import display
from ipywidgets import interact, interactive # for some neat interactions
import ipywidgets as widgets
```

<!-- #region colab_type="text" id="pIL4paTb36pb" -->
## A Minimal Neural Network

A discrete time neural network simulator is implemented below. It takes in an input vector (I), an adjacency matrix (W) and a number of steps that have to be computed (**iterations**). At every step, the new network activity is equal to the current activity times the adjacency matrix, as such:

----

$$A_{t+1} = W A_t$$

----

> **Question: There are strict requirements for the shape of the Input and Weight matrix. Do you know what they are?**
<!-- #endregion -->

```python colab={} colab_type="code" id="GWIB_wp-2hzj"
# This is a minimal network

def NNet(W, I, steps):
  # initialize our output to save all the states of the network
  states = zeros((steps, len(W[0])))
  print(states)
  
  # we assing the input as the first state of our network
  states[0] = I 
  # print(states)

  # The future is computed from the past. Note that range starts at '1', while the first state is at '0'
  for t in range(1,steps): 
    states[t] = W@states[t-1] 
  
  return states 
```

```python colab={} colab_type="code" id="5B_h-8tG9HO0"

```

<!-- #region colab_type="text" id="erKG0nvu5Lub" -->
## A Ring Network

Let us saninty check our network by looking at the activity for a ring network. A ring network is a directed network where every neuron maps to the subsequent neuron and the last connects to the first.

<!-- #endregion -->

<!-- #region colab_type="text" id="Njcv-u0c5ufK" -->
### Exercise. Make a Ring Network
1. Make the adjacency matrix 'W' for a ring network with three neurons where the first neuron connects to the second, the second to the third and the third to the first. You should make a python list and convert it (to 'cast it') to an array.
2. Make an input vector 'I' to represent the initial input to the network where the first neuron has activity 1 and other neurons have activity 0. Cast it as an array, and make it sure you know the orientation of the vector (column or row)?
<!-- #endregion -->

```python colab={} colab_type="code" id="gNqtWJpG2hzm"
# A ring network (directed, unweighted)

W = np.array([[0,0,1],[1,0,0],[0,1,0]])

# Input to the first neuron

I = np.array([1,0,0])
```

```python colab={"base_uri": "https://localhost:8080/", "height": 248} colab_type="code" executionInfo={"elapsed": 1002, "status": "ok", "timestamp": 1583401568836, "user": {"displayName": "Mario Negrello", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gh9vnOzDqUD2QacfGPwd13jMPmwn1hzZzBnVURjO4E=s64", "userId": "10136788594790905986"}, "user_tz": -60} id="90F1fy-B2hzq" outputId="de5a9869-2ff3-4f05-fccb-cefabe21248e"
# You can use the following code to visualize your ring network
# Note: we use W for the matrix

G = nx.from_numpy_array(W, create_using = nx.MultiDiGraph())

pos = nx.circular_layout(G)
nx.draw_networkx_nodes(G, pos,  node_size = 1000)
nx.draw_networkx_labels(G, pos)

nx.draw_networkx_edges(G, pos, arrows=True)

plt.show()
```

```python colab={} colab_type="code" id="DCJStWdu2hzu"
 # Here we run the network and show the resulting activity
R= NNet(W,I,30)

plt.figure(figsize=(15,5))
plt.imshow(R.T)
plt.colorbar()
plt.xlabel('network step (iteration)')
plt.ylabel('neurons')
```

```python colab={} colab_type="code" id="vinvEFB-2hzx"
 # for your amusement,
# A tiny widget to visualize the progression of the net activity

def net_activity(results,netgraph, timestep):
  pos = nx.circular_layout(G)
  nx.draw_networkx_nodes(G, pos,  node_size = 300, node_color=results[timestep])
  nx.draw_networkx_labels(G, pos)
  nx.draw_networkx_edges(G, pos, arrows=True)

step_the_net = lambda timestep : net_activity(R,G, timestep) 
step_slider = widgets.IntSlider(min=0, max=len(R[0]), step=1, value=0)
net_widget = interact(step_the_net, timestep=step_slider)

display(net_widget);
```

<!-- #region colab_type="text" id="kuc5LiqCDAZV" -->
## A Feed-forward Network

### Exercise. Create a Feed Forward Network
- For a feed-forward network we will need an input layer and an output layer.  Make a network with 9 input neurons in the first layer and 5 output neurons in the second layer. Connections should come strictly from layer 1 to layer 2. 
- No lateral connections: neurons in one layer do not connect to other neurons in the same layer.
- For the existing connections, make all weight equal to 1.
- Create a random input vector and give it to your network function.
- Display the resulting network and activity.

Notes: To display a layered network you have to define the argument 'position' of network x. Note that we have been creating pos via ```pos= nx.circular_layout(G)``` but you can use simply the x and y positions of the nodes (in a list).
<!-- #endregion -->

```python colab={} colab_type="code" id="KQn6NvbR2hzz"
## Exercise 2
ffW = zeros(shape=(14,14))
ffW[0:8,8:14] = 1
print(ffW)

ffR = np.random.uniform(size=(1,14)) # in python, shape and size are used confusingly.
print(ffR)
```

<!-- #region colab_type="text" id="7Dy7Bg0SFX_W" -->
## Activity propagation in Random Recurrent Neural Networks

### Exercise 3. The activity of RRNN's

- These subsequent notes will walk you through why we often use non-linear transfer functions (such as the logistic sigmoid), in the context of neural networks.
<!-- #endregion -->

<!-- #region colab_type="text" id="demu1B0xicEP" -->
### Exercise 3.1: First create a 5x5 random matrix with appropriate random input
<!-- #endregion -->

```python colab={} colab_type="code" id="uh10ecA72hz1"
# Let's create a random matrix with a random input
randomW = np.random.uniform(size=(5,5))
randomI = np.random.uniform(size=(1,5))
```

```python colab={} colab_type="code" id="AJN7VZl3KXpu"
sum(randomW)
```

```python colab={} colab_type="code" id="dimrWZXH2hz3"
plt.figure(figsize=(20,5))

plt.subplot(1,3,1)
plt.imshow(randomW.T)
plt.title('weight matrix')
plt.colorbar()

plt.subplot(1,3,2)
randomG = nx.from_numpy_array(randomW)
edges, weights = zip(*nx.get_edge_attributes(randomG,'weight').items())
pos = nx.circular_layout(randomG)
nx.draw_networkx_edges(randomG, pos, arrows=True, width=weights)
nx.draw_networkx_nodes(randomG, pos,  node_size = 500, node_color=np.sum(a=randomW, axis=0))

plt.subplot(1,3,3)
plt.imshow(randomI.T)
plt.title('input')
plt.colorbar();
```

<!-- #region colab_type="text" id="_0R7WZqYMaYJ" -->
In the code below, run the network with 5, 10 and 30 time steps, always with the same input and matrix. What is the strange thing that happens?
<!-- #endregion -->

<!-- #region colab_type="text" id="fq_rS8Leiv5u" -->
### Exercise 3.2: Run the network for some time steps
<!-- #endregion -->

```python colab={} colab_type="code" id="20L69yaP2hz6"
# Exercise 3.2
R= NNet(randomW,randomI,5)

plt.figure(figsize=(30,3))
plt.imshow(R.T)
plt.colorbar()
plt.xlabel('network iterations')
plt.ylabel('neurons')

# Exercise 3.2
R= NNet(randomW,randomI,30)

plt.figure(figsize=(30,3))
plt.imshow(R.T)
plt.colorbar()
plt.xlabel('network iterations')
plt.ylabel('neurons')
```

<!-- #region colab_type="text" id="hFa1Uhu4LXGC" -->
## Question: Why is it all blue for the first time steps? 
- Notice the scale of the colorbar on the right. Why is the value so large (1e9 means 10^9)?
- Compare the values of the activity of the network in the second time step with the last time step. Why is the difference so huge?
<!-- #endregion -->

<!-- #region colab_type="text" id="vOgP6UcGL4IQ" -->
## A Non-linear Transfer Function 

As you probably figured out, the the activity of the random network above becomes 'unbouded', that is, for every iteration the activity grows (because we keep on multiplying the activity vector by the weight matrix and summing).

This may be your first encounter with the concept of a 'transfer function'. A transfer function transforms the input space to an output space. In neural networks the transfer function often serves the purpose of constraining the output space to a small interval.

You have already learned about the Heaviside transfer function. It constrains the output to two values. The input can be minus infinity and the output will be zero. But that's not very smooth. One way to achieve a smooth mapping of a *real number* from minus infinity to infinity into a number between zero and one is via the logistic sigmoid:

---

$$f(x) = \frac{1}{1+e^{-x}}$$

---

This function can be thought of as representing a degree of certainty. For very negative values you become very confident on a 'no' and vice-versa. For values close to zero, 0.5 represents you `don't know', a 50/50.

### Exercise 3.3: Write a python function that returns a sigmoid of its input
- Write the python function
- Plot the function in the range between [-10:10]
<!-- #endregion -->

```python colab={} colab_type="code" id="CCoPme-GZuFE"
# Exercise 3.3 define and plot sigmoid
def sigmoid(x):
  return 1./(1.+np.exp(-x))

x = np.arange(-5,5,0.01)

plt.plot(x, sigmoid(x))
plt.xlabel('x')
plt.ylabel('f(x)')
```

<!-- #region colab_type="text" id="Awihu-2raJsR" -->
### Exercise 3.4 : Add your sigmoid function to your network

This is (the standard) equation that computes the iterates of networks and applies some transfer function F(x):

---
$$
\mathbf{A}(t+1) = F(\mathbf{A(t)}\mathbf{W})
$$

one can also think about it from the perspective of the single neuron:

$$a_i(t+1) = \sum^{n}_{j=1} F (a_j (t) w_{ij})$$

---

- Modify the network equation to include this transfer function 
- Plot the activity of the network and interpret
- Try your new network with different random matrices and inputs:
  - different sizes
  - try multiplying your weigths but a single constant value (i.e., W*constant)
  - try shifting your weigths but a single constant value (i.e., W + constant). Use negative values as well!

<!-- #endregion -->

```python colab={} colab_type="code" id="I_-D6B9t2hz9"
# Exercise 3.4
def NNet_2(W, I, steps):
    states = zeros((steps, len((W[0]))))
    states[0] = I
    for t in range(1,steps):
        states[t] = expit(W@states[t-1])
    return states
```

```python colab={} colab_type="code" id="bzz36_A62h0A"
R= NNet_2(W,I,30)

print(R.shape)
plt.figure
plt.imshow(R.T)
plt.colorbar()
plt.xlabel('time')
plt.xlabel('neuron')
```

```python colab={} colab_type="code" id="LzSwYUsEMDWA"

```
