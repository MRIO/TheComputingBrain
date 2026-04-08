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

<!-- #region colab_type="text" id="oKRN0C0zARH9" -->
# The Making of Networks 

The following exercise introduces you to the representation of networks via graphs and connectivity (adjacency) matrices. This kind of network representation is central in multiple fields of modern science, such as gene expression networks, **neural networks**, molecular dynamics, google page-rank algorithm, **brain networks** and vastly many other applications.

[![Adjacency Matrix and Graphs](https://i.postimg.cc/Pqx5fr6L/image.png)](https://postimg.cc/k20mf9P9)


> **In the set of exercises you will be learning via examples to create adjacency matrices and display them as images and graphs.**

Neural networks heavily use this kind of matrix representation. It is a convenient and powerful way to display and analyze networks as well. Soon we will learn how we can represent activity propagation in neural networks via adjancency matrices and activity vectors!

## Learning Goals 

After going through this exercise, you should be able to:

- Interpret an adjacency matrix as a graph
- Classify different kinds of connectivity matrices on the basis of type of entries (i.e., weighted, unweighted, directed, undirected, random, self-connections)
- Produce a network via a connectivity matrix according to specifications
> **Programming Bonuses:**
>  - Create a figure in python with multiple axes (subplots) and specified size
>  - Add plots to an axes
>  - Display a matrix as an image via ```pcolor``` or ```imshow```
>  - Set a colormap to the plot
>  - Add ```text``` to an axis
>  - Display adjacency matrices as networks via the package ```networkx```

## Terminology Notes:
- "Networks" and "Graphs" are used interchangeably in here.
- "Connectivity Matrix" and "Adjacency Matrix" are also almost synonyms (though distinctions make sense in some cases).


## Programming Resources

- If you come from MATLAB, see [numpy for matlab users](https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html)

## References

[1]	P. Hagmann, L. Cammoun, X. Gigandet, R. Meuli, C. J. Honey, V. J. Wedeen, and O. Sporns, “Mapping the Structural Core of Human Cerebral Cortex,” Plos Biol, vol. 6, no. 7, p. e159, 2008.

[2]	O. Sporns, C. J. Honey, and R. Kotter, “Identification and Classification of Hubs in Brain Networks,” PLoS ONE, vol. 2, no. 10, 2007.

[3]	M. Rubinov and O. Sporns, “Complex network measures of brain connectivity: Uses and interpretations,” Neuroimage, vol. 52, no. 3, pp. 1059–1069, Sep. 2010.

[4]	E. Bullmore and O. Sporns, “Complex brain networks: graph theoretical analysis of structural and functional systems,” Nat Rev Neurosci, vol. 10, no. 3, pp. 186–198, Feb. 2009.

<!-- #endregion -->

<!-- #region colab_type="text" id="c_isVryk3Acq" -->
## Graphs

A **graph** is a set of **nodes** with **edges** connecting them. In the figure below we have a network with two nodes $i$ and $j$  are connected by an edge with weight $w_{ij}$. In general $i$ reprsents rows and indicates'from' while $j$ is a column and indicates 'to'.

---

[![adjacency-graph.png](https://i.postimg.cc/7PzrnhSH/adjacency-graph.png)](https://postimg.cc/8sTY11y9)

---

Graphs are represented with a matrix $\mathbf{W}$ where the non-zero entries $w_{ij}$ stand for an edge (a connection) between nodes i and j. This matrix $\mathbf{W}$ is called the **adjacency matrix** because the entries in the matrix determine whether two nodes are adjacent (neighbors) of each other.

> **To plot graphs in python** we employ the package networkx and matplot lib (see initialization cell above). To see some of the options of this combination, check [this stack exchange question](https://stackoverflow.com/questions/20133479/how-to-draw-directed-graphs-using-networkx-in-python).


## Definitions

- A **weighted network** is a matrix where every entry represents a connection between two nodes, where a real number represents the strenght or  weight of the connection.

- In an **unweighted network**, all existing edges have entries $w_{ij} = 1$, with $w_{ij} = 0$ when there is no edge between nodes $i$ and $j$.

- In a **directed network**, entries $w_{ij} \not = w_{ji}$ in general.

- In an **undirected network**, the adjacency matrix is symmetrical (entries $w_{ij} = w_{ji})$.

- **Self-connections** are entries in the diagonal of the matrix, often represented as $w_{ij}$ with $i=j$.

- A **random graph** is simply a graph where edges have a certain probability of existing, according to some underlying distribution.
<!-- #endregion -->

<!-- #region colab_type="text" id="xmiFQjVgDKFa" -->
## Notebook Initialization

Import relevant packages:
<!-- #endregion -->

```python colab_type="code" id="59u_Kjmj35Q0" colab={}
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib

from networkx.drawing.nx_agraph import write_dot

np.random.seed(2) # we draw always the same random seed for reproducibility
# (learn more about random seeds: https://stackoverflow.com/questions/22639587/random-seed-what-does-it-do (off topic))
```

<!-- #region colab_type="text" id="sgbc1lp03BWh" -->
Something about the last line of the initialization (very important, though rather off topic here):

>How random seeding works: https://stackoverflow.com/questions/22639587/random-seed-what-does-it-do 

<!-- #endregion -->

<!-- #region colab_type="text" id="_mi7CuCKJbuz" -->
## Drawing numbers from random distributions in python

We use numpy's random number generators. For example, to draw random numbers from a uniform distribution:

```python

W = np.random.uniform(low=-2,high=2, size=(5,5)) 

```

to draw from a normal distribution, we'd write:

```python

W = np.random.normal(size=(5,5)) 

```

<!-- #endregion -->

<!-- #region colab_type="text" id="j5EX7fMrBjsl" -->
## Exercises

In these exercises we will create different kinds of random networks, by producing their adjacency matrices. Note that we be creating 'random matrices', that is, the existence of an edge is determined by drawing numbers from specific distributions such as the uniform and the gaussian distribution (if you need it, check wikipedia for references). We will also be using some 'indexing operations' from python's numpy.
<!-- #endregion -->

<!-- #region colab_type="text" id="scCMaM8I_Z4j" -->
### Example 1.

Create a three node network where:
- node one is connected to node two
- node two to node three
- node three to node 1
<!-- #endregion -->

```python colab_type="code" executionInfo={"status": "ok", "timestamp": 1585932079542, "user_tz": -120, "elapsed": 2153, "user": {"displayName": "Natia Shamugia", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgMJVTy7rqYj0YoGXUksPqDt_mKfAsd5ks2L5E_RA=s64", "userId": "11799764980580446930"}} id="ENeYcM2W_3js" outputId="4cef7a0e-c777-4c93-a7a2-b1fcd2e4e0c8" colab={"base_uri": "https://localhost:8080/", "height": 634}
# Define the matrix as an array 
ring_net = np.matrix([[0 ,1. ,0],[0,0,1.],[1.,0,0]])
print(ring_net)

# plot the matrix via imshow
plt.figure()
plt.imshow(ring_net, vmin=0, vmax=1, cmap="gist_gray") 
# vmin and vmax set the range of the colorbar
# cmap defines the colormap (https://matplotlib.org/3.1.1/tutorials/colors/colormaps.html?highlight=colormap)
# note: as an alternative to imshow you can use plt.pcolor(W).

plt.xlabel('to (j)')
plt.ylabel('from (i)')
plt.colorbar()
plt.show()


plt.figure()

# in the line below we transform our matrix into a graph object (from networkx):
G_ring = nx.from_numpy_matrix(ring_net,create_using=nx.DiGraph()) # we use digraph because connections are directed
pos=nx.spring_layout(G_ring)
nx.draw(G_ring)


```

<!-- #region colab_type="text" id="2brWJgQB3siD" -->
### Example 2.

Create a weighted directed graph with 5 nodes where connections are drawn from a uniform distribution between -2 and 2.
<!-- #endregion -->

```python colab_type="code" executionInfo={"status": "ok", "timestamp": 1585922061281, "user_tz": -120, "elapsed": 1658, "user": {"displayName": "Natia Shamugia", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgMJVTy7rqYj0YoGXUksPqDt_mKfAsd5ks2L5E_RA=s64", "userId": "11799764980580446930"}} id="fI0WxhL13okU" outputId="cde18cab-a0a4-4e77-9aba-4631b00441f9" colab={"base_uri": "https://localhost:8080/", "height": 415}
# Recipe for Obtaining and Plotting a Weighted Directed Graph


# 1. draw numbers from the uniform distribution 
# from -2 to 2 to fill up a matrix of 5x5
WR = np.random.uniform(low=-2,high=2, size=(5,5)) 

# 2. print the weights
print("values of our matrix are")
print(WR)

print("") # print empty to obtain a space
print("alternative representations as matrix or graphs:")

# 3. create a figure and populate it with axes
fig, ax = plt.subplots(1, 3, figsize=(9, 4))
fig.suptitle('Matrice to Graphs') # give a title to the whole thing

######## PLOT ADJACENCY MATRIX AS IMAGE

# 4. plot the adjacency matrix via imshow to visualize the weights
#    'pl' returns a handle to the plot so that we can add a colorbar to the subplot
# 
pl = ax[0].imshow(WR, vmin=-2, vmax=2) # vmin and vmax set the range of the colorbar
plt.colorbar(pl, ax=ax[0]) # add a colorbar to the plot

# as an alternative to pcolor you can use plt.imshow().

ax[0].set_xlabel('to (j)')
ax[0].set_ylabel('from (i)')
ax[0].set_title('adjacency matrix')


######## PLOT GRAPH WITH SPRING LAYOUT

# 5. create graph from the new weight matrix W
G_weighted = nx.from_numpy_matrix(WR,create_using=nx.DiGraph())

# 6. get connections and strengths of connections
edges, weights = zip(*nx.get_edge_attributes(G_weighted,'weight').items())

# 7. create a layout for the nodes of the network
pos = nx.spring_layout(G_weighted)  # positions for all nodes


# 8. draw the new graph
nx.draw(G_weighted, pos, ax= ax[1], node_color='b',edgelist=edges, edge_color=weights, width=2, edge_cmap=plt.cm.Reds)

ed = nx.draw_networkx_edges(G_weighted, pos, ax=ax[1], edge_color=weights, width=2, edge_cmap=plt.cm.Reds)

# to create a colorbar that indicates the 
#plt.colorbar(ed);
ax[1].set_title('spring layout');


######## PLOT GRAPH WITH CIRCLE LAYOUT

pos = nx.circular_layout(G_weighted)  # positions for all nodes
nx.draw(G_weighted, pos, ax=ax[2], node_color='b',edgelist=edges, edge_color=weights, width=2, edge_cmap=plt.cm.Reds)

ed = nx.draw_networkx_edges(G_weighted, pos, ax=ax[2], edge_color=weights, width=2, edge_cmap=plt.cm.Reds)
# to create a colorbar that indicates the 
# plt.colorbar(ed);
ax[2].set_title('circle layout');

```

<!-- #region colab_type="text" id="sl3b8zFaDGlg" -->
## Hands-on

For all of the following items, create an adjacency matrix with 5 nodes, according to specifications:
<!-- #endregion -->

<!-- #region colab_type="text" id="EbrrhQEz4wfV" -->
### EXERCISE 1.

Create an unweighted directed graph where the adjacency matrix is filled with discrete random values, such that $P(w_{ij})=1) = 0.4$, and consequently,  $P(w_{ij}=0) = 0.6.$ Plot the resulting adjacency matrix as an image.

---

Tip: one can use a boolean operator such as $>$ to transform real numbers into zeros and ones. For example, if:

```python
a = np.random.rand # get a random number between zero and one.
print a
>>a = 0.323
```

then


```python
b = a>0

```
returns

```python
b = True
```

to 'cast' the boolean values from "True and False" into "ones and zeros", one can simply multiply by '1':
  

```python
b = (a>0)*1;
```
<!-- #endregion -->

```python colab_type="code" executionInfo={"status": "ok", "timestamp": 1585932752245, "user_tz": -120, "elapsed": 1283, "user": {"displayName": "Natia Shamugia", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgMJVTy7rqYj0YoGXUksPqDt_mKfAsd5ks2L5E_RA=s64", "userId": "11799764980580446930"}} id="uvWJOeRf4Pgf" outputId="8d6ba747-c3fa-4628-f0ca-d80207152e03" colab={"base_uri": "https://localhost:8080/", "height": 366}
# Exercise 2, your Solution:

W2 = (np.random.rand(5,5)<0.4)*1

print(W2)

# plot the adjacency matrix via pcolor to visualize the weights
plt.pcolor(W2, vmin=0, vmax=1, cmap='gist_gray') # we use a binary matrix, 
# so a black and white colormap seems appropriate.

# as an alternative to pcolor use plt.imshow(W1). What is the difference between the two?

plt.xlabel('to (j)')
plt.ylabel('from (i)')
plt.colorbar()
plt.show()


```

<!-- #region colab_type="text" id="m4NnGdmw6Yjs" -->
### Exercise 2.

Create an unweighted directed graph without self-connections. Print and plot both the resulting adjacency matrix and the graph.
<!-- #endregion -->

```python colab_type="code" executionInfo={"status": "ok", "timestamp": 1585933633959, "user_tz": -120, "elapsed": 1160, "user": {"displayName": "Natia Shamugia", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgMJVTy7rqYj0YoGXUksPqDt_mKfAsd5ks2L5E_RA=s64", "userId": "11799764980580446930"}} id="A9pP9VPA6aN1" outputId="10b9488b-c548-4976-b5ef-600c0767ffc8" colab={"base_uri": "https://localhost:8080/", "height": 165}
# one way to do it
W3 = np.random.randint(0,2,size=(5,5))
np.fill_diagonal(W3,0)
print(W3)

# another way to do it
A = np.random.randint(0,2,size=(2,2))

for i in np.nditer(A[j]):
  print(i)
  for j in np.nditer(A[i]):
    if i == j:
      A[i,j] == 0
print(A)
```

```python id="81TX6CQ_-LEs" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 49} outputId="7ec8858f-2b42-4240-e72c-321e050b0905" executionInfo={"status": "ok", "timestamp": 1585933955580, "user_tz": -120, "elapsed": 657, "user": {"displayName": "Natia Shamugia", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgMJVTy7rqYj0YoGXUksPqDt_mKfAsd5ks2L5E_RA=s64", "userId": "11799764980580446930"}}
# A working version for the solution
A = np.random.randint(0,2,size=(2,2))

for i in range(len(A)):
  for j in range(len(A[i])):
    if i == j:
      A[i,j] = 0 
print(A)
```

<!-- #region colab_type="text" id="2YjD7MSa_gWo" -->
### Exercise 3.

Make an **undirected unweighted random graph without self connections** by enforcing $w_{ij}= w_{ji}$.  $w_{ij} = 1$ for all existing edges (symmetrical adjacency matrix). Make sure that all elements of the main diagonal are 0, that is, $W_{ij}=0 \iff i=j$. Plot the resulting graphs and matrix.


<!-- #endregion -->

<!-- #region colab_type="text" id="I8PUcxgMNCwx" -->
One neat way to remove self connections is by 'elementwise multiplying a matrix by one matrix where all diagonal entries are equal to zero and all other elements are one.
<!-- #endregion -->

```python colab_type="code" executionInfo={"status": "ok", "timestamp": 1585933944629, "user_tz": -120, "elapsed": 1097, "user": {"displayName": "Natia Shamugia", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgMJVTy7rqYj0YoGXUksPqDt_mKfAsd5ks2L5E_RA=s64", "userId": "11799764980580446930"}} id="t3QVk19N_yeh" outputId="49f546e4-5853-4ca0-cf9d-e876aeb73988" colab={"base_uri": "https://localhost:8080/", "height": 319}
W4 =  np.random.randint(0,2,size=(3,3))
# # set diagonal to zeros
# np.fill_diagonal(W4,0)
# # adding the transpose of a matrix to itself will give symmetric matrix
# S = (W4.T + W4)/2
# print(S)
# print(S.T)

zeros_diagonal = 1 - np.identity(3)

fig = plt.figure(figsize=(15,5))

plt.subplot(1,5,1)
plt.imshow(W4)
plt.title('original W4')

plt.subplot(1,5,2)
plt.text(0.1,.5,'multiplied by')
plt.axis('off')

plt.subplot(1,5,3)
plt.imshow(zeros_diagonal)
plt.title('zeros diagonal')

plt.subplot(1,5,4)
plt.text(.5,.5,'equals')
plt.axis('off')

plt.subplot(1,5,5)
W4 = np.multiply(W4, zeros_diagonal) #np.multiply is equivalent to matlab's elementwise multiplication '.*',   A.*B.
plt.imshow(W4)
plt.title('W4 .* zeros diagonal')


```

```python colab_type="code" executionInfo={"elapsed": 472, "status": "ok", "timestamp": 1569858090295, "user": {"displayName": "Daphne Cornelisse", "photoUrl": "", "userId": "06723002061441590794"}, "user_tz": -120} id="t_f9g-cEu431" outputId="11068ea2-cce0-4e83-fb71-c2532b914e27" colab={"base_uri": "https://localhost:8080/", "height": 187}
W4 = np.random.randint(0,2,size=(5,5))
# set diagonal to zeros
np.fill_diagonal(W4,0)
# adding the transpose of a matrix to itself will give symmetric matrix
S = (W4.T + W4)/2

S = (S>=0.5)*1
print(S)
print(S.T)
```

<!-- #region colab_type="text" id="LU5xifhDU1V5" -->
## Advanced Graph Plotting: Changing Edges Properties

With the networkx package one can edit the properties of the edges and nodes. Below are some examples.
<!-- #endregion -->

```python colab_type="code" executionInfo={"elapsed": 3868, "status": "ok", "timestamp": 1568499548736, "user": {"displayName": "Mario Negrello", "photoUrl": "https://lh3.googleusercontent.com/a-/AAuE7mBKcuM3zrzWydsMeTlpor_04SYP_Fx-VlyQP2mgYpk=s64", "userId": "10136788594790905986"}, "user_tz": -120} id="NfPPq1U7jH8P" outputId="90907602-ed69-4820-f8dd-75a036603782" colab={"base_uri": "https://localhost:8080/", "height": 281}
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Color the edges connecting the nodes.

# create graph from the matrix WR defined above.
G_weighted = nx.from_numpy_matrix(WR,create_using=nx.DiGraph())

# get connections and strengths of connections
edges, weights = zip(*nx.get_edge_attributes(G_weighted,'weight').items())
pos = nx.spring_layout(G_weighted)  # positions for all nodes

# draw the new graphs in two subplots
ax = plt.subplot(1,2,1)
nx.draw(G_weighted, pos, node_color='b',edgelist=edges, edge_color=weights, width=2, edge_cmap=plt.cm.Reds)

ed = nx.draw_networkx_edges(G_weighted, pos, edge_color=weights, width=2, edge_cmap=plt.cm.Reds)
# to create a colorbar that indicates the 
# plt.colorbar();
plt.title('Strengths of weights in network');

plt.subplot(1,2,2);
plt.imshow(WR, vmin = -2, vmax = 2)
```

```python colab_type="code" executionInfo={"elapsed": 4684, "status": "ok", "timestamp": 1568499549560, "user": {"displayName": "Mario Negrello", "photoUrl": "https://lh3.googleusercontent.com/a-/AAuE7mBKcuM3zrzWydsMeTlpor_04SYP_Fx-VlyQP2mgYpk=s64", "userId": "10136788594790905986"}, "user_tz": -120} id="ZLbumfSAnQGJ" outputId="7349b14d-34c7-43fb-ce9c-6e51a0411b7a" colab={"base_uri": "https://localhost:8080/", "height": 272}
 # Create a graph with named Nodes:

# G_dir_W2 = nx.from_numpy_matrix(W2, create_using=nx.MultiDiGraph())
# (unweighted directed graph with self-connections)

G_dir_W2 = nx.MultiDiGraph(W2) 
G_dir_W2.graph['edge'] = {'arrowsize': '10', 'splines': 'curved'}


plt.subplot(1,2,1)
pos=nx.spring_layout(G_dir_W2) # create positions for all nodes by distributing nodes with a spring force

# nodes
nx.draw_networkx_nodes(G_dir_W2,pos,
                       node_size=500,
                   alpha=0.8)

# edges
nx.draw_networkx_edges(G_dir_W2,pos,width=1.0,alpha=0.5)
nx.draw_networkx_edges(G_dir_W2,pos)
nx.draw_networkx_edges(G_dir_W2,pos)



labels={}
labels[0]=r'$1$'
labels[1]=r'$2$'
labels[2]=r'$3$'
labels[3]=r'$4$'
labels[4]=r'$5$'
nx.draw_networkx_labels(G_dir_W2,pos,labels,font_size=12)


plt.subplot(1,2,2)
plt.imshow(W2)



# a = nx.nx_agraph.to_agraph(g)
# for etup in weakEdges:
#     a.get_edge(*etup).attr['style'] = 'dashed'

# a.draw('test2.png', prog='circo')
```

```python colab_type="code" executionInfo={"elapsed": 12735, "status": "ok", "timestamp": 1568499557618, "user": {"displayName": "Mario Negrello", "photoUrl": "https://lh3.googleusercontent.com/a-/AAuE7mBKcuM3zrzWydsMeTlpor_04SYP_Fx-VlyQP2mgYpk=s64", "userId": "10136788594790905986"}, "user_tz": -120} id="6TdNl9CBzHkU" outputId="6231849f-3cb9-4135-b832-408fff050381" colab={"base_uri": "https://localhost:8080/", "height": 294}
# Create a very large network, just for fun

# 1. draw numbers from the uniform distribution 
# from -2 to 2 to fill up a matrix
WBig = np.random.uniform(low=-2,high=2, size=(20,20)) 

# 3. create a figure and populate it with axes
fig, ax = plt.subplots(1, 3, figsize=(9, 4))
fig.suptitle('Matrice to Graphs') # give a title to the whole thing

######## PLOT ADJACENCY MATRIX AS IMAGE

# 4. plot the adjacency matrix via imshow to visualize the weights
#    'pl' returns a handle to the plot so that we can add a colorbar to the subplot
# 
pl = ax[0].imshow(WBig, vmin=-2, vmax=2) # vmin and vmax set the range of the colorbar
plt.colorbar(pl, ax=ax[0]) # add a colorbar to the plot

# as an alternative to pcolor you can use plt.imshow().

ax[0].set_xlabel('to (j)')
ax[0].set_ylabel('from (i)')
ax[0].set_title('adjacency matrix')


######## PLOT GRAPH WITH SPRING LAYOUT

# 5. create graph from the new weight matrix W
G_big = nx.from_numpy_matrix(WBig,create_using=nx.DiGraph())

# 6. get connections and strengths of connections
edges, weights = zip(*nx.get_edge_attributes(G_big,'weight').items())

# 7. create a layout for the nodes of the network
pos = nx.spring_layout(G_big)  # positions for all nodes


# 8. draw the new graph
nx.draw(G_big, pos, ax= ax[1], node_color='b',edgelist=edges, edge_color=weights, width=2, edge_cmap=plt.cm.Reds)

ed = nx.draw_networkx_edges(G_big, pos, ax=ax[1], edge_color=weights, width=2, edge_cmap=plt.cm.Reds)

# to create a colorbar that indicates the 
# plt.colorbar(ed);
ax[1].set_title('spring layout');


######## PLOT GRAPH WITH CIRCLE LAYOUT

pos = nx.circular_layout(G_big)  # positions for all nodes
nx.draw(G_big, pos, ax=ax[2], node_color='b',edgelist=edges, edge_color=weights, width=2, edge_cmap=plt.cm.Reds)

ed = nx.draw_networkx_edges(G_big, pos, ax=ax[2], edge_color=weights, width=2, edge_cmap=plt.cm.Reds)
# to create a colorbar that indicates the 
#plt.colorbar('grey');
ax[2].set_title('circle layout');
```

<!-- #region colab_type="text" id="UCSCFhnC0R1i" -->
## Challenges:
- Create a graph with 100 nodes, 20% chance of connections, in circle layout, and no self connections.
- Display a feed forward neural network with twenty neurons and two layers, each with 10 neurons. Note: you will have to read the documentation of networkx!

## Questions:
- What are the particularities of a matrix that represents a 'ring' graph, where $W_{ij}$ exists if $i=j+1$?
- What would be the particularities of a network with clusters (where some nodes connect more to their neighbors than the other nodes)?

## Researching Further:
- How do we measure connectivity in networks?
- What kind of connectivity exists in brains?
- What kind of connectivity exists in social networks?
- What is a small world network?

<!-- #endregion -->

```python colab_type="code" id="XFsaoukLz0mH" colab={}

```
