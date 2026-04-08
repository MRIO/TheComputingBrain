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

<!-- #region id="1KoSxdQIjbbu" -->
# Introduction
<!-- #endregion -->

<!-- #region id="ZWQnzKtuCu3V" -->
Our world is full of **structure**. Bananas are yellow, peanut butter is brown and if the sky is deep blue it doesn't rain. These are just examples of things that we all implicitly know. We know them without really thinking about it, forgetting that we once had to learn them too.

We learn the structure from these kinds of phenomena by being exposed to the reliable **co-occurence** of yellow bananas (or green if they are not ripe) and no rain when the skies are blue. Co-occurrences are one type of *statistical regularities* or *patterns* in the world.

**Hebbian plasticity** encapsulates the idea that co-occurrences in the world can be captured by neurons when:
> "*Cells that fire together, wire together*"

In this project, we will see how self-organisation of spatial features emerges as a result of a synaptic change rule operating with input patterns. Our network will learn about the **topography** of a very simple world, one that only consists of combinations of horizontal and vertical lines.

As we train our network on input patterns, **network platicity via Hebbian rules leads to self-organization in weight space**, after which the network learns the statistical regularities in the input. With Hebb's rule as a basis, the network will learn **without an explicit error signal** from the environment, that is, *without supervision*.

For this project we will zoom out from biological complexity to understand the operating principles and we will assume that a *real number* represents the synaptic strength between a pre and a postsynaptic neuron. We will also regard the activity of neurons themselves as positive real numbers (possibly representing firing rate of a given neuronal population). Another simplification is that we will be dealing with 'discrete dynamics', that is, our network will be updated in discrete time steps. You may spot other simplifications as we go along. As stark as these assumptions may be, none is too tragic, and the main plasticity phenomena due to Hebbian rule is known to be robust when the biological complexity is re-introduced into the picture.

What does the network learn and how can we observe the process? Read on!

<!-- #endregion -->

<!-- #region id="Bl5rJ16Rkeup" -->
# Learning goals
<!-- #endregion -->

<!-- #region id="QzLlF09Qgez_" -->
By going through this project, students learn:
- How to compute an activity dependent plasticity rule in a neural network
- How Hebbian plasticity rule extracts regularities from inputs
- How to train a network by presenting input patterns to a network
- How to inspect and interpret network weights
<!-- #endregion -->

<!-- #region id="Fw91Q-X8n0MI" -->
# Keywords
<!-- #endregion -->

<!-- #region id="JmnGecndCSWz" -->

- Patterns
- Statistical regularities
- Correlations
- Hebbian plasticity
- Unsupervised Learning
- Self-organization
- (Weight) saturation
- Outer-product (Linear algebra)

<!-- #endregion -->

<!-- #region id="8QGtcqE9AYX_" -->
# Pre-Requisites
<!-- #endregion -->

<!-- #region id="TiRbQtkwCUOs" -->

- Simple neurons
- Adjacency Matrices
- Activity Propagation
<!-- #endregion -->

<!-- #region id="_jCUNL7S_beA" -->
# Intialization
<!-- #endregion -->

```python id="m87hb_FcHAJn" executionInfo={"status": "ok", "timestamp": 1604584629392, "user_tz": -60, "elapsed": 1586, "user": {"displayName": "Natia Shamugia", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgMJVTy7rqYj0YoGXUksPqDt_mKfAsd5ks2L5E_RA=s64", "userId": "11799764980580446930"}}
# import dependencies
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit # fast sigmoid computation
import networkx as nx
import itertools
from ipywidgets import interact
from IPython.display import display
```

<!-- #region id="0NLG4KK1cod5" -->
# Simple Hebbian Plasticity
<!-- #endregion -->

<!-- #region id="XzQvCmouoQLz" -->
## How weights change:
<!-- #endregion -->

<!-- #region id="Z9VuglX466rV" -->
In the simplest version of Hebbian learning, the change of weight is directly proportional to the activity of the pre-synaptic and the post-synaptic neuron.

Assuming that the output function given firing rate of the neurons is well represented by a sigmoidal function, the activity of neurons ranges from zero to one (minimal to maximal, if you like). If the pre-synaptic neuron is zero and that of the post-synaptic neuron is positive, there is no change of weights. If both are positive a positive synaptic change will be observed. If firing on one neuron leads to reduction of the other's activity, the weight change is negative.


 In the formula below, $\Delta w$ (the change in weights) is the weight of the receiving neuron ($y$) from the upstream neurons ($x_i$) 

------

$$
\Delta w_{i} = x_i y
$$

------

This means that the weigth of the connection ($w_i$) impinging on y changes as a function of the joint activity. If neuron $x_i$ and $y$ co-fire (i.e., both have positive activity), the connection strenthens.

The new weight is delta weight plus the previous weight:


------

$$
w_i(t+1) = w_i + \Delta w_i(t) 
$$

------

Where $\Delta w_i$ is the change in synaptic weight $w_{i}$, as a function of presynaptic neuron's activity $x_i$ (usually a number representing, e.g., the firing rate) and activity of the post-synaptic neuron $y$. 


or, one can also use vector notation for this network, where $\mathbf{y}$ is a population of receiving neurons ($y_i$):

------

$$
\Delta \mathbf{w}^T(t) = \mathbf{x}y^T\\
$$

------

Such that the new weights would be:

------

$$
\mathbf{w}(t+1) = \mathbf{w}(t) + \Delta \mathbf{w}(t)
$$

------


With this extremely simple learning rule, interesting things happen. The diagram below is a simple demonstration of how Hebbian learning leads the network to "discover" correlations in the input patterns it receives.










<!-- #endregion -->

<!-- #region id="mZtpHGB9mjKO" -->
### Worked out Example
<!-- #endregion -->

<!-- #region id="SGXOuHJimlvt" -->

<div>
<img src=http://drive.google.com/uc?export=view&id=1udaL_eE8CFxZXUa6jCRbLY0KuUcWYqS2 width="500">
</div>

Depicted in the diagram above is $\mathbf{x}$, is an input vector (three neurons in the example picture), $\mathbf{w}$ the a vector containing the weights (real numbers), and $y$ the output (a *scalar* real number). $y$ is a linear combination (sum) of the inputs multiplied by the weights (the dot product). That is, 
$$
y = \mathbf{x} \cdot \mathbf{w}
$$

$$
\text{at }t=0 \qquad
y = 
\begin{bmatrix}
    1 \\
    1 \\
    -1
\end{bmatrix}\cdot 
\begin{bmatrix}
    0.1 \\
    0.1 \\
    0.1
\end{bmatrix}
= (1 \times0.1)+(1\times0.1)+ (-1\times0.1) = 0.1
$$

Using the input vector ($\mathbf{x}$) and the output scalar ($y$), the new weight matrix $\mathbf{w}(t+1)$ can then be calculated in two steps:
$$
\Delta \mathbf{w} = \mathbf{x} y \\
\mathbf{w}(t+1) = \mathbf{w}(t) + \Delta \mathbf{w}
$$
Take note of what are the input correlations (statistical regularity) in this example. The activities of `x[0]` and `x[1]` co-occur, but the activity of `x[2]` does not. 

That is why, even though all the weights start out as $0.1$, the two correlated inputs dominate the receiving unit activation and their weights will continue to increase upon repeated presentation of the stimulus. The third unit sometimes goes up and sometimes down, with no net increase over time. Thus, Hebbian learning discovers correlations in the inputs.

<div>
<img src=http://drive.google.com/uc?export=view&id=1uk4oytg-A1YEBNyzhznC3V6PVoWl1FBt width="500">
</div>


<!-- #endregion -->

<!-- #region id="t9qo3KKRmpRK" -->
### **Exercise 1**: Hebbian by Hand
<!-- #endregion -->

<!-- #region id="3JkwUg1qmru7" -->

Following up from the example above, calculate the weight vector $\mathbf{w}$ at $t=2$ with pen and paper. 

<!-- #endregion -->

<!-- #region id="g4FP4E_T_QJg" -->
## Input Patterns (training set)
<!-- #endregion -->

<!-- #region id="u49Zs_zOX9Je" -->
Before we train our network, we have to decide on the inputs it will receive, that is, what are the 'patterns' or 'statistical regularities', present in the environment.

The world of our network will consist of combinations of vertical and horizontal lines. They could represent, for instance, individual fingers. We start with creating a set of such **input patterns**, defined below.
<!-- #endregion -->

<!-- #region id="4rJ2xnm9o_0K" -->
### **Exercise 2**: Create Inputs
<!-- #endregion -->

<!-- #region id="QuBxXmTao45w" -->

>  Create $k$ input patterns, that is, $5 \times 5$ matrices. 

These patterns will serve as our **training set**. The input patterns in the training set must represent all unique combinations of two straight lines which can be vertical and horizontal. We will use combinatorics to produce all possible arrangements of two straight lines in this input space.

> **a.** How many ways are there to take 2 numbers from 10 numbers without replacement? Assume that the order does not matter. In combinatorics, this is called a **combination** and you can use `itertools.combinations(iterable,r)` to produce all the possible combinations of two numbers.

> **b.** Taking the numbers [1-5] as representing horizontal lines and [6-10] for the vertical lines, fill in matrices representing stimuli.

---




<!-- #endregion -->

```python id="Qk3UYMmQYx_v" executionInfo={"status": "ok", "timestamp": 1604584638922, "user_tz": -60, "elapsed": 618, "user": {"displayName": "Natia Shamugia", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgMJVTy7rqYj0YoGXUksPqDt_mKfAsd5ks2L5E_RA=s64", "userId": "11799764980580446930"}} outputId="99a81c4b-fac8-47f2-a427-41d879769ffa" colab={"base_uri": "https://localhost:8080/"}
# obtain all possible combinations of two numbers between 1-10:
nums = [0,1,2,3,4,5,6,7,8,9] # 10 numbers 
combis = list(itertools.combinations(nums,2)) # 45 combinations
print(len(combis))
print(combis)
M = np.zeros((5,5))
```

```python id="lJiU3U2v1WTs" executionInfo={"status": "ok", "timestamp": 1604584641177, "user_tz": -60, "elapsed": 662, "user": {"displayName": "Natia Shamugia", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgMJVTy7rqYj0YoGXUksPqDt_mKfAsd5ks2L5E_RA=s64", "userId": "11799764980580446930"}}
# Fill in the sensory stimuli by indexing the appropriate row or column in the input matrix.
# fill in your stimuli in a 3D array called 'patterns'

patterns = []

for i,j in combis:
  B = np.copy(M)
  # fill each index (row or column) with ones
  if i <= 4:
    B[i,:] = 1 # fill that row with ones
  else:
    i = i - 5
    B[:,i] = 1 # fill that column with ones

  if j <= 4:
    B[j,:] = 1 # fill that row with ones
  else:
    j = j - 5
    B[:,j] = 1 # fill that column with ones
  
  patterns.append(B) 
```

```python id="tdr_t2wj8MrA" executionInfo={"status": "ok", "timestamp": 1604584643439, "user_tz": -60, "elapsed": 724, "user": {"displayName": "Natia Shamugia", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgMJVTy7rqYj0YoGXUksPqDt_mKfAsd5ks2L5E_RA=s64", "userId": "11799764980580446930"}}
# # uncommment this to print all of the patterns
# for idx, p in enumerate(patterns):
#   print(f'-- {idx} --')
#   print(p)
```

```python id="7cPeW2_ODl7E" executionInfo={"status": "ok", "timestamp": 1604584645058, "user_tz": -60, "elapsed": 1379, "user": {"displayName": "Natia Shamugia", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgMJVTy7rqYj0YoGXUksPqDt_mKfAsd5ks2L5E_RA=s64", "userId": "11799764980580446930"}} outputId="6469721d-e65e-4e66-c8a2-3b03e39d864d" colab={"base_uri": "https://localhost:8080/", "height": 128}
# use this code to inspect a few of the patterns you created
plt.figure(figsize=(10,5))
print('   Example patterns from training set')
plt.subplot(1,8,1), plt.spy(patterns[0],cmap='gist_gray');
plt.subplot(1,8,2), plt.spy(patterns[10],cmap='gist_gray');
plt.subplot(1,8,3), plt.spy(patterns[25],cmap='gist_gray');
plt.subplot(1,8,4), plt.spy(patterns[34],cmap='gist_gray');
plt.subplot(1,8,5), plt.spy(patterns[1],cmap='gist_gray');
plt.subplot(1,8,6), plt.spy(patterns[11],cmap='gist_gray');
plt.subplot(1,8,7), plt.spy(patterns[26],cmap='gist_gray');
plt.subplot(1,8,8), plt.spy(patterns[35],cmap='gist_gray');
```

<!-- #region id="BjPr1HTh8Rfa" -->
---

> ### Exercise 3: vectorise / flatten the input patterns.

---

In order to use the input patterns to train the network we need to vectorise them. 

<!-- #endregion -->

```python id="XoSj-N3qN9gt" executionInfo={"status": "ok", "timestamp": 1604584648660, "user_tz": -60, "elapsed": 680, "user": {"displayName": "Natia Shamugia", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgMJVTy7rqYj0YoGXUksPqDt_mKfAsd5ks2L5E_RA=s64", "userId": "11799764980580446930"}}
# convert input matrices to vectors
X = [patterns[i].flatten() for i in range(len(patterns))]
```

<!-- #region id="7KNpXjgawK5F" -->
### Review: Matrix multiplication and Numpy

When multiplying matrices, there are a few laws you have to be comfortable with. 

- Matrix multiplication is not commutitative. The order in which you multiply one matrix with another matters.

$$
AB \ne BA
$$

- To multiply matrices, the dimension of the columns of the first matrix has to be equal to the dimension of the rows of the second matrix. See below. 

<div>
<img src=https://www2.cs.duke.edu/courses/compsci344/spring15/classwork/04_transforms/match.gif width="400">
</div> 

- A matrix as: $A_{M \times N}$ means it has $M$ rows and $N$ columns.


- Linear algebra distinguishes between row and column vectors, but in `numpy` there is no real such distinction. In `numpy`, everything is an `n-`dimensional array. What `numpy` does have however are 2-dimensional arrays. So an array of `(4,1)` has 4 rows and one column, which can be thought of as a **column vector**. Likewise an `(1,4)` array is like a **row vector**.

---
```
(n,1) ==> column vector
(1,n) ==> row vector
```
---

**Transposing in Numpy**

Normally, to reverse the rows and columns of a vector you [*transpose*](https://en.wikipedia.org/wiki/Transpose) it. In numpy however, this operation has no effect for 1 dimensional arrays. 

- to reshape a 1D array use `my_array[:, None]`.

In general
- use `@` for matrix multiplication and the inner product between two vectors.
- use `np.outer()` to get the outer product between two vectors or a matrix and a vector.



<!-- #endregion -->

<!-- #region id="WXWF_04wGqcj" -->
## 2. Hebbian Learning Algorithm

Even the simplest version of Hebbian learning can show how input structure makes into network structure. 

We will feed our input patterns one by one to our network. For every pattern we present to the network, we will compute the output activity, and from there the change of weigths as a function of the output of the matrix. 

The steps are as follows:

> 1. Intialize our network as an adjacency matrix $\mathbf{W}$ with random weights. 
> 2. Compute the neuron output at iteration $t$ (where $t$ is the same as the pattern in the training sequence)
>  $$ \mathbf{y}^T(t) = \mathbf{x}(t)\mathbf{W}^T(t) $$, where $\mathbf{x}(t)$ is the input vector presented at time step $t$, $\mathbf{W}(t)$ the weight matrix at time step $t$.
> 3. Calculate $\Delta \mathbf{W}(t)$ ,$$ \Delta \mathbf{W}(t) = \eta \mathbf{y}(t) \otimes \mathbf{x}(t)$$, with $\eta$, the learning rate, and the correlation between input and output is computed by the outer product between x and y.
> 4. Update the weight matrix $\mathbf{W}(t)$, $$ \mathbf{W}(t+1) = \mathbf{W}(t) + \Delta \mathbf{W}(t) $$
> 5. Update `t` $\leftarrow$ `t+1`
> 6. Iterate, update: `t+1`, repeat this process from `step 2`.


<!-- $$
\Delta \mathbf{W}(t) = \phi \mathbf{y}(t)[\frac{\gamma}{\phi}\mathbf{x}(t)]
$$ -->

<!-- - $\phi$ is the forgetting rate -->






This is done until (1) the training set has been completely presented, or (2) until the weights reach their steady state values, or (3) the experimenter decides to end training.

<!-- #endregion -->

<!-- #region id="ueLQjNPaIEa8" -->
### Initialize the weight matrix
<!-- #endregion -->

<!-- #region id="9A2vGlOaYGIS" -->
The network that we will be training is a **recurrent neural network (RNN) without self connections**. We will make the network of the same size as the input (because the input is coming from 'sensory space' and the network that is changing is the 'cortical network), and we will initialize the matrix with random values from the normal distribution. Since we have 25 input neurons in our input, we need an adjacency matrix that is 25 x 25. This network is randomized to a 'blank slate'. It could represent, for instance, the recurrent structure of a layer 2/3 cortical patch before any kind of learning.

As you have learned, *adjacency matrices* are matrices that represent connections between nodes in a network. While an adjacency matrix can be binary (with entries consisting of 0's and 1's), their elements can be **real numbers** (i.e., positive or negative with decimals) where each entry represents the strength of the connection. This type of matrix is often referred to as a **weight matrix**.

<!-- #endregion -->

<!-- #region id="0tPFQjZc_HDm" -->
---

### **Exercise 4**: Create a `25 x 25` weight matrix 

---

<!-- #endregion -->

<!-- #region id="eV5LkQKC_Wqp" -->
Create a matrix `W` for a RNN with normally distributed weights with a mean of 0 and display it using `plt.imshow`. 
<!-- #endregion -->

```python id="_YswQoUTMw4H" executionInfo={"status": "ok", "timestamp": 1604584651832, "user_tz": -60, "elapsed": 656, "user": {"displayName": "Natia Shamugia", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgMJVTy7rqYj0YoGXUksPqDt_mKfAsd5ks2L5E_RA=s64", "userId": "11799764980580446930"}} outputId="be9ed987-3c2f-4c8f-f796-336a3da31d96" colab={"base_uri": "https://localhost:8080/", "height": 281}
# seeding
np.random.seed(0)

N = 25 # total number of neurons
# randomly initialise weight matrix
W0 = np.random.rand(25,25)-0.5
np.fill_diagonal(W0,0) # remove self-connections

plt.imshow(W0,cmap='Blues')
plt.colorbar()
plt.title('Randomly initialised weight matrix');
```

<!-- #region id="lkWuMkGr_xhk" -->
---
### Exercise 5: One iteration of the Hebbian learning algorithm
---
<!-- #endregion -->

<!-- #region id="4eQ4HTXS8Gnx" -->

#### 1. Compute the output, $y$ by implementing the equation below in python.

$$
\mathbf{y}(0) = \mathbf{x}(0)\mathbf{W}(0)
$$
<!-- #endregion -->

```python id="IgvkGTDaGlcR" executionInfo={"status": "ok", "timestamp": 1604584701088, "user_tz": -60, "elapsed": 1217, "user": {"displayName": "Natia Shamugia", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgMJVTy7rqYj0YoGXUksPqDt_mKfAsd5ks2L5E_RA=s64", "userId": "11799764980580446930"}} outputId="76940b6a-11b4-4988-b039-d6dbf77180b9" colab={"base_uri": "https://localhost:8080/", "height": 352}
# take one input vector: 25x1 (col vector)
X0 = X[0]
print(X0.shape)

# weight matrix
print(W0.shape)

# display as a 2D pixel pattern
plt.figure()
plt.spy(patterns[0],cmap='gist_gray');

plt.figure()
# display as row vector
# purple is 0, yellow is 1
plt.imshow(X0.reshape(1,25), cmap='gist_gray');
```

```python id="F1gDXDYZJx5h" executionInfo={"status": "ok", "timestamp": 1604584719325, "user_tz": -60, "elapsed": 725, "user": {"displayName": "Natia Shamugia", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgMJVTy7rqYj0YoGXUksPqDt_mKfAsd5ks2L5E_RA=s64", "userId": "11799764980580446930"}} outputId="ade4c52b-87aa-49b0-bbc2-5e9b8d67d469" colab={"base_uri": "https://localhost:8080/"}
# take the inner product between input and weight matrix to get a vector
y = X0@W0

# the output is a vector: nx1
print(y.shape)

y
```

<!-- #region id="WX-KtqEIAAD2" -->
#### 2. Calculate $\Delta \mathbf{W}(0)$
<!-- #endregion -->

<!-- #region id="x1LFxuhfGlyc" -->
Next up, let's find out how we should alter the weights by calculating $\Delta \mathbf{W}$.

$$
\Delta \mathbf{W}(t) = \eta ( \mathbf{y}(t) \otimes \mathbf{x}(t))
$$

where $\eta$ is the learning rate, this determines the magnitude of the change to the weights. 

The learning rate is a parameter that typically has a value between 0.01 and 0.1.

The $\otimes$ symbol means we take the **outer product** between two vectors. Note that we do this, because we want to know for each synapse what was the correlation between the pre and post neurons.

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/583d2f9f02f2644aa0acd092a29a9d0e49df1b4a)

<!-- #endregion -->

```python id="v5ta79TRGmEC" executionInfo={"status": "ok", "timestamp": 1604584815825, "user_tz": -60, "elapsed": 645, "user": {"displayName": "Natia Shamugia", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgMJVTy7rqYj0YoGXUksPqDt_mKfAsd5ks2L5E_RA=s64", "userId": "11799764980580446930"}} outputId="ecbc4169-8322-48de-c4e9-0257966c3141" colab={"base_uri": "https://localhost:8080/"}
# set the parameter gamma, the learning rate
eta = 0.1

# want to create a matrix 
print(y.shape, X0.shape)

# take the outer product directly
delta_W = np.outer(y,X0)
delta_W_eta = delta_W * eta # with eta

# check the shape 
delta_W.shape
```

<!-- #region id="-XkAM7hAA1u1" -->
#### 3. Update the weight matrix $\mathbf{W}(1)$
<!-- #endregion -->

<!-- #region id="BTWCVnevGmX8" -->
Now we have $\Delta \mathbf{W}$, we can get the new weight matrix!

$$
\mathbf{W}(1) = \mathbf{W}(0) + \Delta \mathbf{W}(0)
$$

---
**Question:** add the $\Delta \mathbf{W}$'s to the old weight matrix and
display both of them. What is the difference between the new weight matrices? Why is that?

---
<!-- #endregion -->

```python id="U35YZ9jgNdbl" executionInfo={"status": "ok", "timestamp": 1604584833048, "user_tz": -60, "elapsed": 966, "user": {"displayName": "Natia Shamugia", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgMJVTy7rqYj0YoGXUksPqDt_mKfAsd5ks2L5E_RA=s64", "userId": "11799764980580446930"}} outputId="0a39421b-ba6f-40f5-e085-f670adbb5297" colab={"base_uri": "https://localhost:8080/", "height": 281}
# obtain the new weight matrix by adding deltaW to the old weight matrix
W1 = W0 + delta_W

# code to display the new weight matrix
plt.imshow(W1,cmap='Blues')
plt.colorbar()
plt.title('Without using a learning rate');
```

```python id="xmvJXqNXrano" executionInfo={"status": "ok", "timestamp": 1602222102111, "user_tz": -120, "elapsed": 3208, "user": {"displayName": "Mario Negrello", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gh9vnOzDqUD2QacfGPwd13jMPmwn1hzZzBnVURjO4E=s64", "userId": "10136788594790905986"}} outputId="097f8729-a845-4253-b04e-b424f9b252e0" colab={"base_uri": "https://localhost:8080/", "height": 281}
# obtain the new weight matrix by adding deltaW to the old weight matrix
W1_eta = W0 + delta_W_eta

# code to display the new weight matrix
plt.imshow(W1_eta, cmap='Blues')
plt.colorbar()
plt.title('Using a learning rate');
```

<!-- #region id="ltKFNEelH9SM" -->
## Write a neural network function with plasticity

Above you performed all necessary steps for the first iteration of the Hebbian plasticity algorithm. To iterate through all the input patterns, we write a function that takes in an input pattern and updates the weight matrix for each step.

---
> **Exercise 6:** make a function that implements the algorithm above for
5 or 10 epochs (i.e. 20 passes through the training set) using this rule to update the weight matrix:
$$
\Delta \mathbf{W}(t) = \eta (\mathbf{y}(t) \otimes \mathbf{x}(t))
$$
> During the training epoch and after each pattern presentation, save the weight matrix for later inspection. To make your saved matrices compatible with our function for visualization of weights (below), make sure your network function (1) appends your weight matrices in a list, (2) returns the collection of weight matrices.

---
<!-- #endregion -->

```python id="DkzLhD6tuF8N"
def self_org(X,W,eta):
  '''
  arguments
  X: nested array with input vectors
  W: initial weight matrix (random)
  eta: learning rate
  '''
  weight_lst = [] # list to which we will append all the different weight matrices
  for epoch in range(20):
    for t in range(len(X)): # for a number of iterations
      x = X[t] # take an input pattern
      y = x@W # multiply pattern and weight matrix
      delta_W = np.outer(y,x) * eta
      W += delta_W # update W
      weight_lst.append(np.copy(W)) # make deep copy of W and append to list 
  return weight_lst
```

```python id="hYqhdUqpvVcz"
# randomly initialise weight matrix and remove self connections
# W = np.random.rand(25,25)
# np.fill_diagonal(W,0)

## UNCOMMENT TO INITIALISE WITH POS RANDOM VALUES ##
W = np.random.rand(N,N)
np.fill_diagonal(W,0) # remove self-connections

# input patterns
X = [patterns[i].flatten() for i in range(len(patterns))]

## call nn function ##
weight_lst = self_org(X,W,0.1)
```

<!-- #region id="jsdBFoYnSxQh" -->
### 4. Inspect weight matrix after training
<!-- #endregion -->

<!-- #region id="0eqlpAVlBFDH" -->
- After running for K epochs, observe the evolution of the weight matrices during training and attempt to interpret what you see.

"As the training proceeded, weights started to reflect the pixel correlations from the lines present in the environment. Thus, individual units developed selective representations of the correlations present within individual lines, or two lines in some cases."

---
> **Exercise 7:** Use the function below to view how your weight matrix changed per timestep. Input the array of weight matrices you obtained from your `neuralnet` function. What happened to the weights?
---

---
> **Exercise 8:** Repeat **3** (implement as function) and **4** (interpret weights) now with a different rule to update $\Delta \mathbf{W}$:

$$
\Delta \mathbf{W}(t) = \eta \, \text{diag}(\mathbf{y}) (\mathbf{x}\otimes\mathbf{v}) - \mathbf{W}(t)
$$

---

<!-- #endregion -->

```python id="ZAZLEi2Cp5Fm"
def inspect_weights(weight_lst):
    N = weight_lst.shape[0]
    def view_weight_matrix(t=0):
        plt.imshow(weight_lst[t], cmap='Blues', interpolation='nearest')
        plt.title(f'timestep {t}')
        plt.colorbar()
    interact(view_weight_matrix, t=(0, N-1))
```

```python id="llV2gESq5zgv" executionInfo={"status": "ok", "timestamp": 1602222102599, "user_tz": -120, "elapsed": 3684, "user": {"displayName": "Mario Negrello", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gh9vnOzDqUD2QacfGPwd13jMPmwn1hzZzBnVURjO4E=s64", "userId": "10136788594790905986"}} outputId="9e066dca-585c-45ca-9a59-bd89cccd82d9" colab={"base_uri": "https://localhost:8080/", "height": 281, "referenced_widgets": ["d0e7c3e4db6d46fb9237996e7ad5e06b", "35a1c0eafb9c4acd8784ce452f5d361e", "98b94053659b4e258153761caabcdea4", "2c74f9eceb894a3f859285dcc9a46eda", "fb0319bd8add44819e261efb7b911c3b", "4071b7f0d9374a56a9388dfd84d3caf8", "c5b6819bb37b4eb3862ee1d037de033f"]}
# convert to numpy array
weight_lst = np.array(weight_lst)

inspect_weights(weight_lst)
```

<!-- #region id="uAyRqjd-WhTD" -->
#### Question: What is the scale of the weight matrix?
<!-- #endregion -->

<!-- #region id="lCmk9ij2WnmV" -->
# Hebbian Plasticity Without Saturation
<!-- #endregion -->

<!-- #region id="jLqAHWPJBcVB" -->
## CPCA Hebbian Learning
<!-- #endregion -->

<!-- #region id="SzwOpvwbBRsk" -->


The standard Hebbian learning rule has a major limitation; weights can grow without bounds. You saw this in exercise 1, when calculating the weights by hand. The Conditional Principle Component Analysis (CPCA) learning rule is a slightly more complex learning rule that keeps the weights bounded, making sure they won't grow infinitely large.

The CPCA equation is:

---

$$
\Delta w_{ij}(t) = \eta y_j (x_i - w_{ij}(t))
$$

---

Where $\Delta w_{ij}$ is one element in the weight matrix, $\Delta \mathbf{W}$. Remember that matrices are just an easy way to collect a lot of numbers in a structured way. 

<div>
<img src=https://upload.wikimedia.org/wikipedia/commons/b/bf/Matris.png width="300">
</div>

We can use vector notation to obtain the entire weight matrix and get the following equation.

---

$$
\Delta \mathbf{W}(t) = \eta \, \text{diag}(\mathbf{y}) (\mathbf{x}\otimes\mathbf{v}) - \mathbf{W}(t)
$$

---

let's go through this step by step

- $\eta$ is our learning rate, typically 0.1
- with $\text{diag}(\mathbf{y})$ we just say, "put all the values in that vector on the diagonal, and make the rest zero" (makes it easy to multiply) $\Rightarrow \begin{bmatrix}
    y_{1} & & \\
    & \ddots & \\
    & & y_{r}
  \end{bmatrix}$ 

- Next, we want to remove something from $\mathbf{x}$. But how can we subtract a matrix from a vector?? Obviously the dimensions don't match up. But no dispair, we have a clever trick for this. We introduce a new vector, $\mathbf{v}$, a vector that consists of only ones.

$$
\mathbf{v} =
\begin{bmatrix}
    1  \\
    \vdots  \\
    1
  \end{bmatrix}
$$

Then we take the *outer product* between $\mathbf{v}$ and $\mathbf{x}$ so we get a matrix. And tadah! We have a matrix in the same size as $\mathbf{W}(t)$. 

The only operation left is matrix multiplying what is left with $\text{diag}(\mathbf{y})$ and you are done.

---

**CPCA** has 3 categories of weight changes:

1. When the pre and postsynaptic neuron are both strongly active ($x_i > w_{ij}$) the weight should increase (LTP).
2. When the pre synaptic neuron is active but the postsynaptic is not ($x_i < w_{ij}$), LTD will happen.
3. When the postsynaptic neuron is not active, the likelihood and/or magnitude of any weight change goes to zero.


<!-- #endregion -->

```python id="f1iYcB_LvYng"
##FUNCTION CPCA Hebbian Rule ##
def CPCA_Hebbian(X,W,eta,v):
  '''
  arguments
  X: nested array with input vectors
  W: initial weight matrix (random)
  eta: learning rate
  '''
  cpca_w_lst = [] # list to which we will append all the different weight matrices

  for t in range(len(X)): # for a number of iterations    
    x = X[t] # take an input pattern
    y = x@W # multiply pattern and weight matrix
    delta_W = eta * np.diag(y) @ (np.outer(x,v) - W)
    W += delta_W # update W
    cpca_w_lst.append(np.copy(W)) # make deep copy of W and append to list 

  return weight_lst
```

```python id="QNWAyZ_FvcUz"
# randomly initialise weight matrix and remove self connections
W = np.full(shape=(25,25), fill_value=0.1)
np.fill_diagonal(W,0) # remove self-connections
# input patterns
X = [patterns[i].flatten() for i in range(len(patterns))]
# vector of ones
v = np.ones(25)

## call cpca_nn function ##
cpca_w_lst = CPCA_Hebbian(X,W,0.1,v);

# turn into np array 
cpca_w_lst = np.array(cpca_w_lst)
```

```python id="b-v-dZsYG4bf" executionInfo={"status": "ok", "timestamp": 1602222102859, "user_tz": -120, "elapsed": 3936, "user": {"displayName": "Mario Negrello", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gh9vnOzDqUD2QacfGPwd13jMPmwn1hzZzBnVURjO4E=s64", "userId": "10136788594790905986"}} outputId="53a68a47-98d2-4c34-9b1a-528ea071b10a" colab={"base_uri": "https://localhost:8080/", "referenced_widgets": ["8ed317b755f74ebd9b54a696accbd1ca", "84571752a893409199a51b3f3b175b8d", "2177d0a06c064bb084a7ac572db2d8d2", "7459f343e7d44a1b884708e0c901ff25", "b820a6f9feda4fd4908eaa4867e02678", "5e927d87724e436fad1578a9f240bd47", "e99683d826334475b406061fc2aac69a"]}
inspect_weights(cpca_w_lst)
```

<!-- #region id="VOZ8QcNp6PvS" -->
## Oja's rule

The standard Hebbian learning rule has a major limitation; weights can grow without bounds. You saw this in exercise 1, when calculating the weights by hand. Oja's learning rule is a slightly more complex learning rule that keeps the weights bounded, making sure they won't grow infinitely large.

<!-- #endregion -->

<!-- #region id="NpTX5TJJBkZi" -->
In the lecture Self-organisation and Hebbian learning, you also learned about Oja's rule.

For reference, Oja's rule is

---

$$
\Delta w_i = \eta (x_iy - y^2w_i)
$$

--- 

which, in vector form is

--- 

$$
\Delta \mathbf{w} = \eta (\mathbf{x}y-y^2\mathbf{w})
$$

---

where $\eta$ is the learning rate, $\mathbf{x}$ is the input vector and $\mathbf{w}$ the weight vector.

Oja's rule keeps the weights bounded by subtracting the squared output $y^2*w_i$. It guarantees that the larger the output of the neuron becomes, the stronger is the balancing effect is.

find more information at the [oja scholarpedia page](http://www.scholarpedia.org/article/Oja_learning_rule)
<!-- #endregion -->

<!-- #region id="dZHDTqWsxM-q" -->
### Using Oja's rule with a simple example: one neuron
<!-- #endregion -->

```python id="HNSy5Vfy4zJk"
# create input vectors
x = np.array([[1,1,-1],[1,1,1],[-1,-1,1],[-1,-1,-1]])
# create weight vector
w = np.array([[0.1,0.1,0.1]])
```

```python id="4RgdAacA4yVM" executionInfo={"status": "ok", "timestamp": 1602222102860, "user_tz": -120, "elapsed": 3930, "user": {"displayName": "Mario Negrello", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gh9vnOzDqUD2QacfGPwd13jMPmwn1hzZzBnVURjO4E=s64", "userId": "10136788594790905986"}} outputId="44042249-093a-40ba-b5f5-d01c17d1f447" colab={"base_uri": "https://localhost:8080/", "height": 353}
oja_w_lst = [] # list to which we will append all the different weight vectors

for t in range(len(x)): # for a number of iterations
  i = x[t] # take an input pattern
  print(f'input pattern {t}: {i}')
  y = w@i # inner product between input pattern and weight matrix
  print(f'output {t}: {y}')
  delta_w = eta * (y*i) - (y**2 * w)
  print(f'delta_w {t}: {delta_w}')
  w += delta_w # update W
  print(f'w (t = {t}): {w}')
  oja_w_lst.append(np.copy(w)) # make deep copy of W and append to list 
  print('---')
```

<!-- #region id="3_nB3pFD2Hfi" -->
### Oja's rule with multiple neurons
<!-- #endregion -->

<!-- #region id="AopU2cmEBsAY" -->
For multiple neurons, where the weights are collected in a matrix, Oja's rule can be implemented in the following way

--- 

$$
\Delta \mathbf{W} = \eta (t) (\mathbf{y}\mathbf{x}^T - LT[\mathbf{y}\mathbf{y}^T] \mathbf{W}) 
$$

--- 

This impplementation comes from [this paper by Terrence Sanger](http://www.cnbc.cmu.edu/~tai/nc19journalclubs/Sanger1989.pdf).

where

- $LT[.]$ stands for "Lower Triangular" this converts the matrix inside to a matrix with all zero's above the diagonal.
- $\eta (t)$ is a learning rate that decays as time progresses. 
- note that $yx^T$ is the same as $y \otimes x$.

<!-- #endregion -->

```python id="vHVelxTSwhV0"
## FUNCTION Oja ##
def Oja(X,W,eta):
  '''
  arguments
  X: nested array with input vectors
  W: initial weight matrix (random)
  eta: learning rate
  '''
  Ojas_lst = [] # list to which we will append all the different weight matrices
  for epoch in range(10):
    for t in range(len(X)): # for a number of iterations
      x = X[t] # take an input pattern
      y = x@W # inner product input pattern and weights
      # get delta w) #
      LT = np.tril(np.outer(y,y))
      delta_W = (0.1/(t+1)) * (np.outer(y,x) - LT@W)
      W += delta_W # update W
      Ojas_lst.append(np.copy(W)) # make deep copy of W and append to list 

  return Ojas_lst
```

```python id="orvoymgbwhS7"
W = np.full(shape=(25,25), fill_value=0.1)
np.fill_diagonal(W,0) # remove self-connections
eta = 0.1
X = [patterns[i].flatten() for i in range(len(patterns))]

# call function
ojas = Oja(X,W,eta)

# convert to numpy array
ojas = np.array(ojas)
```

```python id="NNd7MTqCvmvN" executionInfo={"status": "ok", "timestamp": 1602230178373, "user_tz": -120, "elapsed": 779, "user": {"displayName": "Mario Negrello", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gh9vnOzDqUD2QacfGPwd13jMPmwn1hzZzBnVURjO4E=s64", "userId": "10136788594790905986"}} outputId="6cca9daa-f327-42a6-dfca-d9e96a8d598c" colab={"base_uri": "https://localhost:8080/", "height": 313, "referenced_widgets": ["2140c5bbbbce4fc094c193c94c688abd", "a52db0fe452a4efd9b54a4624c395b9e", "c16a582d25184c67b1e2333d7a756a7c", "4363c3f3994241cf93a6fb11b439e204", "62724a77b63e4f7289da0264925f0d1c", "c7e0dd94175940ffb05b9c75b0e27020", "f096c3c6a3384faaafe5d70ea2faf2e2"]}
# inspect the weights
inspect_weights(ojas)
```

<!-- #region id="oCXGqk3k8lKb" -->
## Testing the weight matrix

<!-- #endregion -->

<!-- #region id="Q2aCEgsjB4Ce" -->
We now present the input patterns to the initial weight matrix and the trained matrix to compare the difference in output activation. How do we see that the network has 'learned' the input patterns?
<!-- #endregion -->

```python id="kqvm-2Ek8kzE"
# take an input pattern (matrix format)
pat = patterns[3]

#M_0 = np.random.rand(25,25)

## trained weight matrix ##
M_trained = ojas[-1] # trained weight matrix
# reshape trained matrix into 5x5 matrix|
M_trained = np.resize(M_trained,(5,5))

# matrix multiply trained matrix and pattern
#initial = M_0@pat
trained = M_trained@pat
```

```python id="NSLOLdf70xFP" executionInfo={"status": "error", "timestamp": 1602222104074, "user_tz": -120, "elapsed": 5130, "user": {"displayName": "Mario Negrello", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gh9vnOzDqUD2QacfGPwd13jMPmwn1hzZzBnVURjO4E=s64", "userId": "10136788594790905986"}} outputId="75330ddc-23f0-4a31-9b32-ceb70075ef19" colab={"base_uri": "https://localhost:8080/", "height": 482}
fig, axs = plt.subplots(1, 2)
axs[0].imshow(initial, cmap='Blues')
axs[0].set_title('Without training')
axs[1].imshow(trained, cmap='Blues')
axs[1].set_title('With training');
fig.colorbar(initial, ax=axs[0])
```

<!-- #region id="NbtIIyEWrd7n" -->
# Answer to exercise 1


<!-- #endregion -->

<!-- #region id="fjmZDJuk68bS" -->
### Exercise 1

**General equations**

For each time step $t$, calculate

$$ 
y_t = \mathbf{x}_t \cdot \mathbf{w}_t \\
\Delta \mathbf{w}_t = \mathbf{x}_ty_t\\
\mathbf{w}_{t+1} = \mathbf{w}_{t} + \Delta \mathbf{w}_t
$$

Starting with $t=0$..
$$
t=0 \qquad
y = 
\begin{bmatrix}
    1 \\
    1 \\
    -1
\end{bmatrix}\cdot 
\begin{bmatrix}
    0.1 \\
    0.1 \\
    0.1
\end{bmatrix}
= (1 \times0.1)+(1\times0.1)+ (-1\times0.1) = 0.1
$$

$$
w_0 = 
\begin{bmatrix}
    0.1 \\
    0.1 \\
    0.1
\end{bmatrix}+ 
\begin{bmatrix}
    0.1 \\
    0.1 \\
    -0.1
\end{bmatrix}
= 
\begin{bmatrix}
    0.2 \\
    0.2 \\
    0
\end{bmatrix}
$$

$$
\vdots
$$

$$
t=3 \qquad
y = 
\begin{bmatrix}
    -1 \\
    -1 \\
    1
\end{bmatrix}\cdot 
\begin{bmatrix}
    0.6 \\
    0.6 \\
    0.4
\end{bmatrix}
= (-1 \times0.6)+(-1\times0.6)+ (1\times0.4) = -0.8
$$

$$
w_3 = 
\begin{bmatrix}
    0.6 \\
    0.6 \\
    0.4
\end{bmatrix}+ 
\begin{bmatrix}
    0.8 \\
    0.8 \\
    -0.8
\end{bmatrix}
= 
\begin{bmatrix}
    1.4 \\
    1.4 \\
    -0.4
\end{bmatrix}
$$


**Overview**
<div>
<img src=http://drive.google.com/uc?export=view&id=1y-LRGAk8BHTGB-CQ4y-W4MANg4v1Ocyr width="600">
</div>

Correlated units determine the symbol and scale of the weights, and weights of these inputs grow quickly, whereas the weight of the uncorrelated input `x[2]` oscillates. 
<!-- #endregion -->

<!-- #region id="Um5RBpM6CGEq" -->
#License

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>. Mario Negrello, Daphne Cornelisse, Elias Santoro (2020).
<!-- #endregion -->
