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

<!-- #region colab_type="text" id="BzpPmhZuy2wK" -->
# Introduction

## Bistable perception, Filling in, and Content Addressable Memories

In this project we will implement a simple model of bistable perception via the Hopfield networks [1], some of the simplest recurrent neural networks, which performs the function of 'content addressable memory'. That is, given an incomplete pattern, neurons 'autocomplete' the pattern. Each pattern is stored as a fixed point attractor via Hebbian learning. This type of network can also 'denoise' a known stimulus. It has also been used to explain 'filling in' phenomenon.

>Bistable perception is when the brain cannot decide what is the correct interpretation of a picture and spontaneously switches between interpretations. Filling in happens when the brain 'fills in' for a stimulus that has not been seen, but is implied (such as the Gazzaniga triangle and certain color ilusions).

[1]	J. Hopfield, “Neural networks and physical systems with collective emergent computational abilities,” proceedings of the national academy of sciences (Biophysics), Apr. 1982.

<!-- #endregion -->

<!-- #region colab_type="text" id="ut7L4PLH01yc" -->
## Learning Goals

- Implement a Hopfield recurrent neural network (RNN)
- Train the network with 'one shot' Hebbian learning
- Relate the activity of the network with fixed point attractors of the newtork
- Explain what is an energy function
- Reason about the energy landscape of Hopfield Networks
<!-- #endregion -->

<!-- #region colab_type="text" id="NNLGCNJd1-OY" -->
## A Hopfield Network

The Hopfield network belongs to a type of algorithms called *autoassociative* and they can store information in memory to retrieve/recall it later with incomplete input patterns. The patterns are imprinted in the weight matrix and due to its constraints (symmetric connections), an input will lead to dynamics that eventually will settle in one of the stored patterns. In dynamical systems lingo, an 'attractor'.

Thus, given a cue, a Hopfield network can restore a stored pattern, in a phenomenon called 'content addressable memory'.

The Hopfield network is defined as follows:

---
$$a_j \leftarrow \left\{\begin{array}{ll} +1 & \mbox {if }\sum_{j}{w_{ij}a_i}\geq\theta, \\
 -1 & \mbox {otherwise.}\end{array}\right.$$
 
---

where:
- $w_{ij}$ is the strength of the connection weight from unit j to unit i (the weight of the connection). In the Hopfield network, all weights are **symmetric**, that is $w_{ij} = w_{ji}$. Weights are real numbers (positive or negative decimals) 
- $a_j$ is the *state* of unit j.
- $\theta$ is the threshold, we use $0$ in this project.

Notes:
- Every neuron receives input from every other neuron and also outputs information to every other neuron.
  - There are no self-connections
- Each neuron is a perceptron with a +1/-1 output
- Alternatively, activities may also be either 0 or 1 (instead of -1 and 1). 
- The update of the units can be both synchronous (all $a_i$ are updated simultaneously) or asynchronous. In this project we will implement synchronous updates.
<!-- #endregion -->

<!-- #region colab_type="text" id="EC9_RLyy6oeE" -->
## Hopfield example 

This is a possible starting configuration of a Hopfield network. Each line or edege represents a weighted connection between nodes (a real number). The activation of nodes can take values of either +1 or -1 according to the Hopfield network.

<div>
<img src=http://drive.google.com/uc?export=view&id=1sev0wqUI27o2nS7zMitfDa_CSvy1o4Hz width="500">
</div>

At each timestep, a neuron receives an incoming "field", which is the weighted sum of the output of all other neurons. A neuron ***flips*** if the weighted sum of the outputs of the other neurons is the opposite sign. If the sign is equal to its own sign however, nothing changes.

---

Let's look at the evolution of the state vector in this Hopfield network.

The inputs the top neuron gets are (from left to right):

$$
-1 * +1 = -1 \\
-1 * -1 = +1 \\
\text{+ and - cancel each other out} \\
+1 * +1 = +1 \\
\rule{3cm}{0.4pt} \\
\qquad +1
$$

which is opposite to $-1$, so the sign of this neuron will change!

Thus, these are the states of the network after one timestep, or one **evolution**

<div>
<img src=http://drive.google.com/uc?export=view&id=1ZOHlytgk9MNgAQiE3THXWyIHLFbuB5J5 width="350">
</div>

<!-- #endregion -->

<!-- #region colab_type="text" id="6T1ViOKe6scJ" -->
## 1. Hopfield for one evolution

____
> **Exercise 1.** Implement a function that computes the evolution of the state of a recurrent neural network.
____
<!-- #endregion -->

<!-- #region colab_type="text" id="t1t9WZl6IO9x" -->
## Excercise 1: ANSWERS

### option 1: using sum notation
<!-- #endregion -->

```python colab={} colab_type="code" id="D4Gmfv7GB1sv"
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
```

```python colab={} colab_type="code" id="bhHsfSfuvoHn"
def RNN_one(W,state):
  new_state = [] # to append new state to
  SUM = 0 # start with zero SUM

  for j in range(len(state)):
      a_j = state[j] # a_j is the state of a neuron in the network
      print(f'state of neuron j={j}: {a_j}')
      # calculate the weighted sum of all inputs (a_i's) to that neuron, a_j (from = rows, to = cols) 
      for i in range(len(state)):
        SUM += W[i,j] * state[i] # add weighted input
      
      if SUM > 0:
        a_j = 1 # state is +1
        new_state.append(a_j)
      else:
        a_j = -1 
        new_state.append(a_j)

      print(f'new state: {a_j}')
      print('---')

  return new_state
```

```python colab={"base_uri": "https://localhost:8080/", "height": 289} colab_type="code" executionInfo={"elapsed": 1046, "status": "ok", "timestamp": 1571559453650, "user": {"displayName": "Daphne Cornelisse", "photoUrl": "", "userId": "06723002061441590794"}, "user_tz": -120} id="B46oVg2AAs8N" outputId="95df10fd-4ff7-4fb8-ebf7-e27f80b322ed"
# create random initial state
state = np.array([1,-1,1,-1,-1])
# create W
W = np.outer(state,state)
np.fill_diagonal(W,0)

# call RNN_one function
RNN_one(W,state)
```

<!-- #region colab_type="text" id="LMgK05GXv-FW" -->
### Option 2: using matrices and vectors
<!-- #endregion -->

```python colab={} colab_type="code" id="Hsmbw7TH7tcR"
def RNN_two(state):
  dim = len(state) # get dimension for matrix based on input patterns
  W = state.T@state# take outer product to obtain weight matrix
  np.fill_diagonal(W, 0) # remove self-connections
  
  return W
```

```python colab={"base_uri": "https://localhost:8080/", "height": 102} colab_type="code" executionInfo={"elapsed": 1188, "status": "ok", "timestamp": 1571559801155, "user": {"displayName": "Daphne Cornelisse", "photoUrl": "", "userId": "06723002061441590794"}, "user_tz": -120} id="9-XrcDQu75DQ" outputId="02a6de13-57e2-47cc-8e90-8a8d4abf8105"
# call the Hopfield function to obtain adjacency matrix from input state
M_hop = RNN_two(np.array([[-1,1,1,-1,-1]]))

M_hop
```

```python colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" executionInfo={"elapsed": 1030, "status": "ok", "timestamp": 1571560215709, "user": {"displayName": "Daphne Cornelisse", "photoUrl": "", "userId": "06723002061441590794"}, "user_tz": -120} id="wPRcmUr5Hhnm" outputId="52bb2678-af5d-4d81-9f9c-a70e8d8888fd"
def get_newstate(state, threshold=0):

  # calculate new state by matrix inner product of hopfield matrix and new pattern
  s = M_hop@state
  s[s >= threshold] = 1
  s[s < threshold] = -1
  return s

# if the summed output > threshold ==> 1 else ==> -1
new_state= get_newstate(np.array([1,1,1,1,-1]))

new_state
```

<!-- #region colab_type="text" id="bY70HxaA9Irt" -->
## 2. The Necker Cube

Here's some code that displays a Necker Cube, a prime example of bistability of perception. In general, human beings  either perceive a 3D cube with blue face in front and the red in the back or vice-versa, but not both in front simultaneously (incidentally, it is also possible to see a flat hexagon, though it is unlikely). Also, it is not possible to 'enforce' a particular perception, and the interpretations switch between the two possible states.

The rationale of this type of bistability in the interpretation of 3D objects is that by looking at many examples of surfaces in the world, you have 'trained' your visual system to perceive the world in ways that are consistent/coherent. 


<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 248} colab_type="code" executionInfo={"elapsed": 922, "status": "ok", "timestamp": 1571407045893, "user": {"displayName": "Daphne Cornelisse", "photoUrl": "", "userId": "06723002061441590794"}, "user_tz": -120} id="Qxmz3273Qg3F" outputId="83137697-c199-4451-e0b2-ba5ca20b93be"
G=nx.cubical_graph()
pos=nx.spring_layout(G, seed=0) # positions for all nodes

# nodes
nx.draw_networkx_nodes(G,pos,
                       nodelist=[0,1,2,3],
                       node_color='r',
                       node_size=500,
                   alpha=0.8)
nx.draw_networkx_nodes(G,pos,
                       nodelist=[4,5,6,7],
                       node_color='b',
                       node_size=500,
                   alpha=0.8)

# edges
nx.draw_networkx_edges(G,pos,width=1.0,alpha=0.5)
nx.draw_networkx_edges(G,pos,
                       edgelist=[(0,1),(1,2),(2,3),(3,0)],
                       width=8,alpha=0.5,edge_color='r')
nx.draw_networkx_edges(G,pos,
                       edgelist=[(4,5),(5,6),(6,7),(7,4)],
                       width=8,alpha=0.5,edge_color='b')


# some math labels
labels={}
labels[0]=r'$A$'
labels[1]=r'$B$'
labels[2]=r'$C$'
labels[3]=r'$D$'
labels[4]=r'$E$'
labels[5]=r'$F$'
labels[6]=r'$G$'
labels[7]=r'$H$'
nx.draw_networkx_labels(G,pos,labels,font_size=16)

plt.axis('off')
plt.savefig("labels_and_colors.png") # save as png
plt.show() # display
```

<!-- #region colab_type="text" id="-u_0NYCX-xqA" -->
Hopfield networks can be trained with 'one shot learning', that is, we directly calculate the final weights from the training patterns.

In the Neckercube above, each vertex represents a **node** in the Hopfield network, which has a state of `-1` or `1`.


---

**Exercise 2.** Train the network via Hebbian learning with two patterns.

---

Training algorithm 

1. Initialise $\mathbf{W}$.
  - Start with a randomized network (with weights drawn from the gaussian distribution with mean 0 and std 1).

2. Create two input vectors:
  - Pattern 1: Vertices ABCD in front (activation = +1), EFH in the back (i.e., activation = +1), and G hidden  (activation = -1)
  - Pattern 2: Vertices EFGH in front, BCD in the back (activation = +1), and A hidden (activation = -1)

3. 

 Train the network using one shot learning.
  - Get matrix from input patterns $\mathbf{W} = \epsilon \epsilon^{T}$ (outer product)
  - don't forget to remove self-connections
  - $\mathbf{W} = \mathbf{W}_{old} + \mathbf{W}_{new}$

4. Compare the initial matrix with the trained matrix.
<!-- #endregion -->

<!-- #region colab_type="text" id="_M9xI5Cg_36g" -->


To train a Hopfield network we use the Hebbian learning rule, as such:

---
$$ w_{ij}=\frac{1}{n}\sum_{\mu=1}^{n}\epsilon_{i}^\mu \epsilon_{j}^\mu $$

---
where:

- $\epsilon$ is a vector encoding a 'pattern'  that we want to train. In a Hopfield network the size of the pattern is the same as the size of the network itself.
- $\mu$ is the index of one of the (potentially many) binary patterns we want to train. $\epsilon^\mu_i$ is the desired pattern activity for neuron $i$ pattern $\epsilon^\mu$. 

<!-- #endregion -->

```python colab={} colab_type="code" id="HDGM85JvQh6s"
# start with a randomized network (used in function below)
mu, sigma = 0, 1
n = 8 # number of neurons
W0 = np.random.normal(mu, sigma, size=(n,n)) # initial weight matrix

# make pattern 1 (hidden G)
p1 = np.array((1,1,1,1,1,1,-1,1))

# make pattern 2 (hidden A)
p2 = np.array((-1,1,1,1,1,1,1,1))

# etc... take it from here :)
```

<!-- #region colab_type="text" id="r9N1rAgoAYc0" -->
---

**Exercise 3.** Observe the convergence to particular patterns from different initial conditions.

- Give random intial states to the trained network and observe the evolution of the network state
- Do that again! What do you observe?

---

<!-- #endregion -->

```python colab={} colab_type="code" id="nLm3eAARq1Ao"

```

<!-- #region colab_type="text" id="thvrUNbqB_ow" -->
The reason why the states tend to converge to one of the trained stimuli can be understood via a 'potential energy' metaphor. The energy landscape of a trained Hopfield network has minima at locations of the trained stimuli. We speak about a 'Negative Lyapunov Energy Function', to represent that. It can be shown that the level of energy of the Hopfield network (as described above) is always decreasing. Given an initial state the energy at time $t$ of the Hopfield network is computed as:

$$
E = -\frac{1}{2} \sum_i \sum_j w_{ij} a_i a_j
$$

which is equivalent to

$$
E = - \frac{1}{2} \mathbf{a}^T \mathbf{W} \mathbf{a}
$$


The network is going to evolve until it this energy term reaches a minimum. The energy is a convex quadratic, so the shape we can expect is a bowl going upwards.

![](https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/Energy_landscape.png/500px-Energy_landscape.png)

___ 

> **Exercise 4.** Show the evolution of the energy function for different initial conditions as above.

___
<!-- #endregion -->

```python colab={} colab_type="code" id="vFyRQitdRka1"

```

<!-- #region colab_type="text" id="QLgePzZ2Q4hf" -->
## Helpful Resources

- [Intro to Hopfield Networks](http://koaning.io/intro-to-hopfield-networks.html)
- [Neupy](http://neupy.com/2015/09/20/discrete_hopfield_network.html)

<!-- #endregion -->
