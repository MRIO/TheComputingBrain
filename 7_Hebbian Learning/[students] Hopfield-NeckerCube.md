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
    name: python3
---

<!-- #region id="S2pnpzF6zyOL" -->
# Introduction
<!-- #endregion -->

<!-- #region id="rqHCE6Xuz2YV" -->
## Bistable perception, Filling in, and Content Addressable Memories
<!-- #endregion -->

<!-- #region id="BzpPmhZuy2wK" -->
In this project we will implement a simple model of bistable perception via the Hopfield networks [1], some of the simplest recurrent neural networks, which performs the function of 'content addressable memory'. The hopfield network is an example of an **attractor network**, where initial states converge onto particular 'attracting states'. Attractor networks can perform the function of **contend addressable memory** because the attracting states are stored as memories, and given an incomplete pattern, neurons 'autocomplete' the pattern. Each pattern is stored as a fixed point attractor, and we train the pattern via Hebbian learning.

 This type of network can also perform the function of removing noise from a picture ('denoise') a stimulus. It has also been used to explain 'filling in' phenomenon.  **Bistable perception** is when the brain cannot decide what is the correct interpretation of an ambiguous picture and spontaneously switches between interpretations. **Filling in** happens when the brain 'fills in' for a stimulus that has not been seen, but is implied (such as the Gazzaniga triangle and certain color ilusions). In this project we will use the Hopfield network to create an analogy about brain dynamics in the context of "Bistable Perception".

Hopfield Networks were introduced by John Hopfield in this paper, below:

[1]	J. Hopfield, “Neural networks and physical systems with collective emergent computational abilities,” proceedings of the national academy of sciences (Biophysics), Apr. 1982.

<!-- #endregion -->

<!-- #region id="ut7L4PLH01yc" -->
## Learning Goals
<!-- #endregion -->

<!-- #region id="blVv6tOCz6fh" -->
- Implement a Hopfield recurrent neural network (RNN)
- Train the network with 'one shot' Hebbian learning
- Relate the activity of the network with fixed point attractors of the newtork
- Explain what is an energy function
- Reason about the energy landscape of Hopfield Networks
<!-- #endregion -->

<!-- #region id="SCAIwGBJz98f" -->
# 1. A Hopfield Network
<!-- #endregion -->

<!-- #region id="NNLGCNJd1-OY" -->
The Hopfield network belongs to a type of algorithms called *autoassociative* and they can store information in memory to retrieve/recall it later with incomplete input patterns. The patterns are imprinted in the weight matrix and due to its constraints (symmetric connections), an input will lead to dynamics that eventually will settle in one of the stored patterns. In dynamical systems lingo, an 'attractor'.

Thus, given a cue, a Hopfield network can restore a stored pattern when it is given input that resembles it. This is referred to as **content addressable memory** (rather than what happens in a computer "RAM", which means **random access memory**).

John Hopfield defined his eponymous network as follows:

---
$$a_j \leftarrow \left\{\begin{array}{ll} +1 & \mbox {if }\sum_{j}{w_{ij}a_i}\geq\theta, \\
 -1 & \mbox {otherwise.}\end{array}\right.$$
 
---

Where:
- $w_{ij}$ is the strength of the connection weight from unit j to unit i (the weight of the connection). In the Hopfield network, all weights are **symmetric**, that is $w_{ij} = w_{ji}$. Weights are real numbers (positive or negative decimals). Self-connections are not allowed.
- $a_j$ is the *state* of unit j.
- $\theta$ is the threshold, we use $0$ in this project.

Note:
- Every neuron receives input from every other neuron and also outputs information to every other neuron.
  - There are no self-connections
- Each neuron is a perceptron with a +1/-1 output
- Alternatively, activities may also be either 0 or 1 (instead of -1 and 1). 
- The update of the units can be both synchronous (all $a_i$ are updated simultaneously) or asynchronous. In this project we will implement synchronous updates.
<!-- #endregion -->

<!-- #region id="j7JcrnWG0CmZ" -->
## Hopfield example 
<!-- #endregion -->

<!-- #region id="EC9_RLyy6oeE" -->
This is a possible starting configuration of a Hopfield network. Each line or edge represents a weighted connection between nodes (a real number). The activation of nodes can take values of either +1 or -1 according to the Hopfield network.

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

<!-- #region id="6T1ViOKe6scJ" -->
## Excercise 1: Hopfield for one evolution

<!-- #endregion -->

<!-- #region id="PWSzH5Dj1Qj2" -->
Implement a function that computes the evolution of the state of a recurrent neural network.
<!-- #endregion -->

<!-- #region id="2H5T0nEi1r37" -->
### Your Code
<!-- #endregion -->

```python id="lXtqRM6D1rQV"

```

<!-- #region id="t1t9WZl6IO9x" -->
### Our Code
<!-- #endregion -->

<!-- #region id="P68kPeAj1lCR" -->
#### Option 1: using sum notation
<!-- #endregion -->

```python id="D4Gmfv7GB1sv"
# import dependencies
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ipywidgets import interact
from IPython.display import display
import networkx as nx

np.random.seed(1)
```

```python id="bhHsfSfuvoHn"
def RNN_one(W,state):
  '''
  Returns the evolution of a given state
  '''
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

```python id="B46oVg2AAs8N" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1635953921211, "user_tz": -420, "elapsed": 11, "user": {"displayName": "Ph\u01b0\u01a1ng Th\u1ee7y Nguy\u1ec5n H\u1ed3", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gh1iyfUlZFSA0eUkg4R5TKs_EKxVHs4rROi5lTFGA=s64", "userId": "14934147373382570915"}} outputId="fa6ffdbd-65fa-4dec-d5b7-60cd03893328"
# create random initial state
state = np.array([-1,1,1,-1,-1])
W = np.outer(state,state)
np.fill_diagonal(W,0)

# call RNN_one function
RNN_one(W,state)
```

<!-- #region id="LMgK05GXv-FW" -->
#### Option 2: using matrices and vectors
<!-- #endregion -->

```python id="Hsmbw7TH7tcR"
def RNN_two(state):
  dim = len(state) # get dimension for matrix based on input patterns
  W = state.T@state# take outer product to obtain weight matrix
  np.fill_diagonal(W, 0) # remove self-connections
  
  return W
```

```python id="9-XrcDQu75DQ" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1635953921211, "user_tz": -420, "elapsed": 8, "user": {"displayName": "Ph\u01b0\u01a1ng Th\u1ee7y Nguy\u1ec5n H\u1ed3", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gh1iyfUlZFSA0eUkg4R5TKs_EKxVHs4rROi5lTFGA=s64", "userId": "14934147373382570915"}} outputId="b6270e62-fdfa-4457-82e9-e961ec2bce04"
# call the Hopfield function to obtain adjacency matrix from input state
M_hop = RNN_two(np.array([[-1,1,1,-1,-1]]))

M_hop
```

```python id="wPRcmUr5Hhnm" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1635953921211, "user_tz": -420, "elapsed": 8, "user": {"displayName": "Ph\u01b0\u01a1ng Th\u1ee7y Nguy\u1ec5n H\u1ed3", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gh1iyfUlZFSA0eUkg4R5TKs_EKxVHs4rROi5lTFGA=s64", "userId": "14934147373382570915"}} outputId="dd35521c-107d-44c5-96ef-e11846bba17e"
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

<!-- #region id="bY70HxaA9Irt" -->
# 2. The Necker cube and training
<!-- #endregion -->

<!-- #region id="TyCvxOtu6Tzi" -->
## Necker Cube
<!-- #endregion -->

<!-- #region id="fR3eAaHp13Xj" -->
Here's some code that displays a Necker Cube, a prime example of bistability of perception. In general, human beings  either perceive a 3D cube with blue face in front and the red in the back or vice-versa, but not both in front simultaneously (incidentally, it is also possible to see a flat hexagon, though it is unlikely). Also, it is not possible to 'enforce' a particular perception;the interpretations switch between the two possible states.

The rationale of this type of bistability in the interpretation of 3D objects is that by looking at many examples of surfaces in the world, you have 'trained' your visual system to perceive the world in ways that are consistent with its rules.  


<!-- #endregion -->

```python id="Qxmz3273Qg3F"
def draw_graph(state):

  G=nx.cubical_graph()
  pos=nx.spring_layout(G, seed=0) # positions for all nodes
  
  pos_nodes = []
  neg_nodes = []

  # get indices of positive and negative nodes
  for s in range(len(state)):
    if state[s] == 1:
      pos_nodes.append(s)
    else:
       neg_nodes.append(s)

  # draw nodes
  nx.draw_networkx_nodes(G,pos,
                        nodelist=pos_nodes,
                        node_color='b',
                        node_size=500,
                    alpha=0.8)
  nx.draw_networkx_nodes(G,pos,
                        nodelist=neg_nodes,
                        node_color='y',
                        node_size=500,
                    alpha=0.8)

  # edges
  nx.draw_networkx_edges(G,pos,width=1.0,alpha=0.5)
  nx.draw_networkx_edges(G,pos,
                        edgelist=[(0,1),(1,2),(2,3),(3,0)],
                        width=8,alpha=0.5,edge_color='k')
  nx.draw_networkx_edges(G,pos,
                        edgelist=[(4,5),(5,6),(6,7),(7,4)],
                        width=8,alpha=0.5,edge_color='k')
  # node labels
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

<!-- #region id="pXDx9KqL45g4" -->
## Excercise 2: Oneshot learning
<!-- #endregion -->

<!-- #region id="_M9xI5Cg_36g" -->
Hopfield networks can be trained with 'one shot learning', that is, we directly calculate the final weights from the training patterns. In the Neckercube above, each vertex represents a **node** in the Hopfield network, which has a state of `-1` or `1`. To train a Hopfield network we use the Hebbian learning rule, as such:

---
$$ w_{ij}=\frac{1}{n}\sum_{\mu=1}^{n}\epsilon_{i}^\mu \epsilon_{j}^\mu $$

---
where:

- $\epsilon$ is a vector encoding a **pattern**  to be trained. In a Hopfield network the size of the pattern is the same as the size of the network itself.
- $\mu$ is the index of one of the (potentially many) binary patterns we want to train. $\epsilon^\mu_i$ is the desired pattern activity for neuron $i$ pattern $\epsilon^\mu$. 

---

**NOTE**: the weight matrix is just computed once from the input pattern(s). The only thing that is updated is the state. The state will change (evolve) over time and eventually settle to the state with the least amount of energy.


<!-- #endregion -->

<!-- #region id="mlq4KZd62dqY" -->
As an excercise try to:
- create a Hopfield Network with N neurons;
- create two patterns; 
- train the network according to the rule above, on the two patterns / states you've created previously.

<!-- #endregion -->

<!-- #region id="NwHE-AV928np" -->
### Your Code
<!-- #endregion -->

```python id="8TAln7_D29FO"

```

<!-- #region id="HD3PBPoz6ZKG" -->
### Our Code
<!-- #endregion -->

```python id="HDGM85JvQh6s"
# make state 1 (hidden G)
s1 = np.array((1,1,1,1,1,1,-1,1))
# make state 2 (hidden A)
s2 = np.array((-1,1,1,1,1,1,1,1))

# put them together
states = np.array((s1,s2))
```

```python id="vklU4tNL3bbH" colab={"base_uri": "https://localhost:8080/", "height": 248} executionInfo={"status": "ok", "timestamp": 1635954141231, "user_tz": -420, "elapsed": 577, "user": {"displayName": "Ph\u01b0\u01a1ng Th\u1ee7y Nguy\u1ec5n H\u1ed3", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gh1iyfUlZFSA0eUkg4R5TKs_EKxVHs4rROi5lTFGA=s64", "userId": "14934147373382570915"}} outputId="5aeb3bc3-aa28-45fa-ae77-630356aec78f"
# This is how the neckercube looks with s1: yellow is -1, blue is +1
draw_graph(s1)
```

```python id="R6H84kcQ58tN" colab={"base_uri": "https://localhost:8080/", "height": 248} executionInfo={"status": "ok", "timestamp": 1635954141671, "user_tz": -420, "elapsed": 442, "user": {"displayName": "Ph\u01b0\u01a1ng Th\u1ee7y Nguy\u1ec5n H\u1ed3", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gh1iyfUlZFSA0eUkg4R5TKs_EKxVHs4rROi5lTFGA=s64", "userId": "14934147373382570915"}} outputId="295b3423-10b3-48eb-d9bc-6c2bafb44d46"
# and for state 2
draw_graph(s2)
```

<!-- #region id="ZGoJL8DRE4Tg" -->
The function below (`Hopfield`) uses the two states to get the trained weight matrix.

<!-- #endregion -->

```python id="zvAFIwd7eM_J"
def Hopfield(states, threshold=0):
  dim = states.shape[-1] # get dimension for matrix based on size of input vectors
  # start with a randomised network according to instructions
  mu, sigma = 0, 1
  W = np.random.normal(mu, sigma, size=(dim,dim))
  W0 = W.copy() # initial weight matrix

  # state by state "oneshot training"
  for s in states:
      W_init = np.outer(s,s) # take outer product to obtain weight matrix for timestep
      np.fill_diagonal(W_init, 0) # remove self-connections
      W = W + W_init # get the new matrix

  # normalize the weight matrix (make elements -1,0, or 1 only)
  W[W > threshold] = 1 
  W[W < threshold] = -1
  np.fill_diagonal(W,0)
  return W0, W
```

```python id="u-btvU1IfsoC"
W0, W_trained = Hopfield(states) # save the trained and initial matrix
```

```python id="FUzW9qiGgfA4" colab={"base_uri": "https://localhost:8080/", "height": 254} executionInfo={"status": "ok", "timestamp": 1635954142089, "user_tz": -420, "elapsed": 422, "user": {"displayName": "Ph\u01b0\u01a1ng Th\u1ee7y Nguy\u1ec5n H\u1ed3", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gh1iyfUlZFSA0eUkg4R5TKs_EKxVHs4rROi5lTFGA=s64", "userId": "14934147373382570915"}} outputId="39bde816-082d-4733-d406-811fe4a64176"
# subplot with initial and trained weight matrix
plt.figure(figsize=(10, 3.5))

plt.subplot(1, 2, 1)
plt.imshow(W0, cmap='Blues')
plt.title('Initial weight matrix')
plt.colorbar(ticks=[-1,0,1], label='element value');

plt.subplot(1, 2, 2)
plt.imshow(W_trained, cmap='Blues')
plt.title('Trained weight matrix')
plt.colorbar(ticks=[-1,0,1], label='element value');
```

<!-- #region id="rWL5kryHF2ea" -->
# 3. Convergence
<!-- #endregion -->

<!-- #region id="UQey-1HBK74e" -->
## Excercise 3: Observe convergence to attractor state
<!-- #endregion -->

<!-- #region id="r9N1rAgoAYc0" -->
Observe the convergence to particular patterns / states from different initial conditions.

- Give random intial states to the trained network and observe the evolution of the network state
- Do that again! What do you observe?
<!-- #endregion -->

<!-- #region id="wKpZdKvHLK75" -->
### Your Code
<!-- #endregion -->

```python id="PB7e1Q4yLKlp"

```

<!-- #region id="PNMhJonZLJ-K" -->
### Our Code
<!-- #endregion -->

```python id="fGPcSpGA6La5"
# create some random initial states
s3 = np.array((1,-1,-1,-1,-1,-1,-1,-1), dtype=int)
s4 = np.array((1,-1,-1,-1,-1,1,-1,1), dtype=int)
s5 = np.array((1,-1,1,1,-1,1,-1,1), dtype=int)
```

```python id="6VhZIv-f6Ll_"
def get_anewstate(W_trained, state, timesteps, threshold=0):
  '''
  Returns new state given a random state.
  '''
  state_lst = []
  s = state # first, s is the random state

  for i in range(timesteps): # for a number of timesteps
    s = W_trained@s # new state is the inner product of the trained matrix and state
    s[s > threshold] = 1 # normalise state
    s[s < threshold] = -1
    state_lst.append(s) # save state to list to display later
  return state_lst

def inspect_states(timesteps,state_lst):
  '''
  Interactive slider to see how the state evolves.
  '''
  N = int(timesteps) # number of timesteps
  def view_states(t=0):
    draw_graph(state_lst[t])
  interact(view_states, t=(0, N-1))
```

<!-- #region id="i6kKGWVIMpiB" -->
Using the functions defined above, let's see what happens when we give our trained matrix a new state and let it run for a number of timesteps.
<!-- #endregion -->

```python id="OZpceVNYCdpG"
# get new state s3
state_lst_3 = get_anewstate(W_trained, s3,10)
```

```python id="mJjjJThhRqw8" colab={"base_uri": "https://localhost:8080/", "height": 280, "referenced_widgets": ["290d02dc797e4a36acdc5ce8be29046e", "008b1692cc914b4aa41b290ce278b564", "766d955642234f3d954f1533f57f7958", "016182eb926948879531f223b3def4d0", "8172185ebc7442a08ea1b5cf94620a1f", "cde95a68ac8f4366ba340525f507cf69", "905e04f2842f40bdb79c21765e3f3737"]} executionInfo={"status": "ok", "timestamp": 1635954496978, "user_tz": -420, "elapsed": 599, "user": {"displayName": "Ph\u01b0\u01a1ng Th\u1ee7y Nguy\u1ec5n H\u1ed3", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gh1iyfUlZFSA0eUkg4R5TKs_EKxVHs4rROi5lTFGA=s64", "userId": "14934147373382570915"}} outputId="26cd83f3-255c-44e3-838b-cca452cca4aa"
inspect_states(10,state_lst_3)
```

```python id="JXDB1NQsV0DM" colab={"base_uri": "https://localhost:8080/", "height": 280, "referenced_widgets": ["98494efd07a548fcad05373b95522f8b", "442e62e84d644324b44bf4b8734c5fec", "27b7661833804199aa1301f5a846d66a", "670aaad231c042e7b5a572a8146b3017", "ee74f67d5739450baa11617fa398ba43", "cdabbebc9373484898a38571ad986bb4", "e2f3fc6c6eda4aab87c2802841eff5be"]} executionInfo={"status": "ok", "timestamp": 1635954497589, "user_tz": -420, "elapsed": 616, "user": {"displayName": "Ph\u01b0\u01a1ng Th\u1ee7y Nguy\u1ec5n H\u1ed3", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gh1iyfUlZFSA0eUkg4R5TKs_EKxVHs4rROi5lTFGA=s64", "userId": "14934147373382570915"}} outputId="844752df-57cc-4301-ef85-8cf354ddef07"
# get new state s4
state_lst_4 = get_anewstate(W_trained, s4,10)
# inspect evolution of states
inspect_states(10,state_lst_4)
```

```python id="_gZIVP37Vz_y" colab={"base_uri": "https://localhost:8080/", "height": 280, "referenced_widgets": ["59e3bbbed99546fc94058b9b4aaeb8ed", "77d4c259806842eb94094dd59eb499f8", "6384ee8fe2c2403a8326c61f9dba63d0", "2329c1856741450db5c42671ec20719c", "7ad39ade1b774a15a5c42b8a5c25eb91", "516ef5b55ac74da2a30dba8912af3623", "cb0f06051e99453db6d7bdd5ec220d78"]} executionInfo={"status": "ok", "timestamp": 1635954497590, "user_tz": -420, "elapsed": 10, "user": {"displayName": "Ph\u01b0\u01a1ng Th\u1ee7y Nguy\u1ec5n H\u1ed3", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gh1iyfUlZFSA0eUkg4R5TKs_EKxVHs4rROi5lTFGA=s64", "userId": "14934147373382570915"}} outputId="7aaabd00-1f2f-4a7d-8507-90701b389bbb"
# get new state s5
state_lst_5 = get_anewstate(W_trained, s5,10)
inspect_states(10,state_lst_5)
```

<!-- #region id="8xWCQBdaciGb" -->
# 4. Energy
<!-- #endregion -->

<!-- #region id="thvrUNbqB_ow" -->
The reason why the states tend to converge to one of the trained stimuli can be understood via a 'potential energy' metaphor. The energy landscape of a trained Hopfield network has minima at locations of the trained stimuli. We speak about a 'Negative Lyapunov Energy Function', to represent that. It can be shown that the level of energy of the Hopfield network (as described above) is always decreasing. Given an initial state the energy at time $t$ of the Hopfield network is computed as:

$$
E = -\frac{1}{2} \sum_i \sum_j w_{ij} a_i a_j
$$

which is equivalent to

$$
E = - \frac{1}{2} \mathbf{a}^T \mathbf{W} \mathbf{a}
$$

![](https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/Energy_landscape.png/500px-Energy_landscape.png)
<!-- #endregion -->

<!-- #region id="YmhM-OkFLTiW" -->
## Exercise 4: Energy function
<!-- #endregion -->

<!-- #region id="ZCuQlnisLYnu" -->
Show the evolution of the energy function for different initial conditions as above.
<!-- #endregion -->

<!-- #region id="iBqWENWgLZ8M" -->
### Your Code
<!-- #endregion -->

```python id="kvrud3aLLcI2"

```

<!-- #region id="jWvR5tmIxtny" -->
### Our Code
<!-- #endregion -->

<!-- #region id="skCy4_nHLffl" -->
We already have the evolution of states for 3 random initial states. The only thing that is left to do is calculating the energy for the state of each timestep, again, using the trained weight matrix.
We use the simple function below and our 3 states:
<!-- #endregion -->

```python id="JQH-KfOntPgM"
def compute_energy(state_evolution,W_trained):
  energy_evo = []
  for s in state_evolution:
    E = -0.5 * (s.T@W_trained@s) # compute energy for that state
    energy_evo.append(E) 
    #print(E)
  return energy_evo
```

```python id="7chtEWTTe-8D"
energy_evo_3 = compute_energy(state_lst_3,W_trained);
```

```python id="4Eege-e7e-5k"
energy_evo_4 = compute_energy(state_lst_4,W_trained)
```

```python id="5DWVM5rJhOyD"
energy_evo_5 = compute_energy(state_lst_5,W_trained);
```

```python id="c9zucJThe-3h" colab={"base_uri": "https://localhost:8080/", "height": 279} executionInfo={"status": "ok", "timestamp": 1635954700365, "user_tz": -420, "elapsed": 718, "user": {"displayName": "Ph\u01b0\u01a1ng Th\u1ee7y Nguy\u1ec5n H\u1ed3", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gh1iyfUlZFSA0eUkg4R5TKs_EKxVHs4rROi5lTFGA=s64", "userId": "14934147373382570915"}} outputId="e5693f89-122a-42d8-cebd-a9fd7ac0bf5a"
# timesteps
x_axis = [0,1,2,3,4,5,6,7,8,9]

# plot the energy per state
plt.plot(x_axis, energy_evo_3, 'r', label='Random state 1')
plt.plot(x_axis, energy_evo_4, 'b', label='Random state 2')
plt.plot(x_axis, energy_evo_5, 'g', label='Random state 3')
plt.xlabel('Timestep')
plt.ylabel('Energy')
plt.grid(True)
plt.xticks(x_axis);
plt.legend();
```

<!-- #region id="QLgePzZ2Q4hf" -->
# Helpful Resources

- [Attractor Networks](http://jackterwilliger.com/attractor-networks/)
- [Intro to Hopfield Networks](http://koaning.io/intro-to-hopfield-networks.html)
- [Neupy](http://neupy.com/2015/09/20/discrete_hopfield_network.html)

<!-- #endregion -->

```python id="c1wqzzFc8qXr"
# calculate the energy ##

# def update_states(W,states_init=np.array([1,-1,1,-1,-1])):
#   '''
#   updates the states using the energy function
#   '''
#   # start with the initial states
#   states = states_init
#   state_configurations = [] # cache state configurations
#   Energies = []
  
#   # compute energy of initial configuration
#   E_min = compute_energy(W,states)
#   print(f'Energy of initial states is {E_min}')
#   print('---')
  
#   for j in range(len(states)):
#     states_ori = states.copy() # make copy of original states
#     states[j] = states[j] * -1 # flip the sign of one state
    
#     # compare energy for original and changed states
#     E_ori = compute_energy(W,states_ori) 
#     E_new = compute_energy(W,states)
#     # energy is the minimum of these two
#     if E_ori < E_new:
#       state_configurations.append(states_ori) # append to remember optimal state
#       Energies.append(E_ori)
#       E = E_ori
#     else:
#       state_configurations.append(states) # append to remember optimal state
#       Energies.append(E_ori)
#       E = E_ori

#     # now compare this to the global minimum 
#     if E < E_min:
#       E_min = E
#       print(f'new minimum energy is {E_min}')
#     else: 
#       print(f'current energy of {E_min} is lower; no change in states')

#   # above 5 lines of code is just for testing the function, can be removed 
#   min_E = min(Energies)
#   min_E_idx = Energies.index(min_E)
#   optimal_state = state_configurations[min_E_idx]
#   return optimal_state, min_E

# def compute_energy(W,states):
#   '''
#   computes the energy for each set of states
#   '''
#   partial_E = 0
#   E = 0
#   SUM = 0 

#   for j in range(len(states)): # for each neuron in the network
#     a_j = states[j] # a_j is the state of a neuron in the network

#     for i in range(len(states)): # calculate the weight times input (a_i's) times output (a_j)
#       a_i = states[i] # 
#       partial_E += W[i,j] * a_i * a_j 
#     E += partial_E

#   # to get the energy do all that times -0.5
#   Energy = -0.5 * E
  
#   return Energy
```

```python id="MogxSscr8qrt"

```
