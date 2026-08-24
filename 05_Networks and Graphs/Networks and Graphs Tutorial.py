# ---
# jupyter:
#   jupytext:
#     formats: ipynb,md,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: computing-brain
#     language: python
#     name: computing-brain
# ---

# %% [markdown] id="oKRN0C0zARH9"
# # The Making of Networks

# %% [markdown] id="diEjgp05GLxm"
#
# The following project introduces you to the representation of networks via graphs and connectivity (adjacency) matrices. This kind of network representation is central in multiple fields of modern science, such as gene expression networks, **neural networks**, molecular dynamics, google page-rank algorithm, **brain networks** and vastly many other applications. It is a convenient representation to represent the propagation of neural activity in random neural networks, as we will see in the subsequent project 'a simple network'.
#
# [![Adjacency Matrix and Graphs](https://i.postimg.cc/Pqx5fr6L/image.png)](https://postimg.cc/k20mf9P9)
#
#
# > From left to right: Binary adjacency matrix (black means areas are connected), rows and colums represent (to) and (from) areas. Middle plot: Network representation equivalent to the adjacency matrix. Right most plot, projection lenght of areas, indicating V4 as a hub.
#
# Neural networks use matrix representation for networks and graphs, both to speed computations as well as to visualize connectivity data. It is a convenient and powerful way to display and think about networks. Not to mention that matrices lend themselves neatly to mathematical analysis.
#
# Similarly, a 'connectivity brain atlas' (also knownn as **Connectome**) is essentially a set of connectivity matrices. See for example [the connectivity matrices](https://portal.brain-map.org/explore/connectivity) created in the "Allen Brain Atlas".
#
# **In the set of exercises you will be learning, via examples, to create adjacency matrices and display them as images and graphs.**  You will later use such matrices to compute ** activity propagation in neural networks** by multiplying connectivity matrices and activity vectors.
#
#

# %% [markdown] id="509DzcGxeXcI"
# # Pre-requisites

# %% [markdown] id="tOuQtrP7yq2u"
# * Linear algebra: understanding of matrices and the meaning of vector matrix multiplication i.e., linear combinations [(video)](https://www.youtube.com/watch?v=xyAuNHPsq-g&list=PLFD0EB975BA0CC1E0)
#
# * Basics statistics: random variable, probability distributions, particularly the uniform and normal distribution.
#

# %% [markdown] id="cwlBLB6QUpaE"
# # Learning Goals

# %% [markdown] id="kFQgzxZHWurD"
# - Interpret an adjacency matrix as a graph.
# - Classify networks as weighted or unweighted, directed or undirected, recurrent or feedforward, and with or without self-connections.
# - Construct adjacency matrices for convergent, divergent, ring, star, and random networks.
# - Calculate graph measures, including degree and clustering coefficient, from adjacency matrices.
# - Visualize adjacency matrices with Matplotlib.
# - Visualize networks with NetworkX.

# %% [markdown] id="rsmylR1tr6XJ"
# # Key Terms

# %% [markdown] id="qHqf2Lm_W0t8"
# - A **weighted network** is a matrix where every entry represents a connection between two nodes, where a real number represents the strength or  weight of the connection.
#
# - In an **unweighted network**, all existing edges have entries $w_{ij} = 1$, with $w_{ij} = 0$ when there is no edge between nodes $i$ and $j$.
#
# - In a **directed network**, entries $w_{ij} \not = w_{ji}$ in general.
#
# - In an **undirected network**, the adjacency matrix is symmetrical (entries $w_{ij} = w_{ji})$.
#
# - **Self-connections** are entries in the diagonal of the matrix, often represented as $w_{ij}$ with $i=j$.
#
# - A **random graph** is simply a graph where edges have a certain probability of existing, according to some underlying distribution.
#
# - "Networks" and "Graphs" are used interchangeably in here.
#
# - "Connectivity Matrix" and "Adjacency Matrix" are also almost synonyms (though distinctions make sense in some cases)
#
# - "**To cast a variable**": means to change its _type_ (for example, integer to float).

# %% [markdown] id="xmiFQjVgDKFa"
# # Initialization Code

# %% tags=["colab-reproducibility-setup"]
# Colab/reproducibility setup: install only packages missing from this runtime.
import importlib.util
import subprocess
import sys

_required_packages = [
    ('netgraph', 'netgraph'),
    ('networkx', 'networkx'),
]
_missing_packages = [
    package
    for module, package in _required_packages
    if importlib.util.find_spec(module) is None
]
if _missing_packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *_missing_packages])

# %% id="59u_Kjmj35Q0"
# Import relevant packages:
import numpy as np
import networkx as nx # we use networkx because it is a common choice. CAVEAT: it cannot display self-connections!
import matplotlib.pyplot as plt
import matplotlib

from networkx.drawing.nx_agraph import write_dot

# %% [markdown] id="HNINf6n6V5K-"
# ## Set a Seed

# %% id="lUTMiGoPV4xG"
np.random.seed(2)

# %% [markdown] id="c_isVryk3Acq"
# # Matrices as Graphs
#

# %% [markdown] id="a05XKt1cWoNp"
# A **graph** (networks are a subset of graphs) is a set of **nodes** (aka neurons, units) with **edges** (aka links, connections) between them. In the figure below we have a matrix that represents a network where nodes $i$ and $j$  are connected by an edge with weight $w_{ij}$. In general $i$ reprsents rows and indicates 'from' while $j$ is a column and indicates 'to', but due to the symmetry in the definition, in some cases the opposite is also seen.
#
# ---
#
#
# [![adjacency-graph.png](https://i.postimg.cc/MpHDy5cR/Network-Adjacency.png)](https://postimg.cc/8sTY11y9)
#
# ---
#
# Graphs and networks are represented with a matrix $\mathbf{W}$ where the non-zero entries $w_{ij}$ stand for an edge (a connection) between nodes i and j. This matrix $\mathbf{W}$ is called the **adjacency matrix** because the entries in the matrix determine whether two nodes are adjacent (neighbors) of each other.
#
# > **To plot graphs in python** we employ the package networkx and matplot lib (see initialization cell above). To see some of the options of this combination, check [this stack exchange question](https://stackoverflow.com/questions/20133479/how-to-draw-directed-graphs-using-networkx-in-python).
#
#

# %% [markdown] id="j5EX7fMrBjsl"
# ## Watch and Learn

# %% [markdown] id="5PKx42N8XE-Z"
# In these exercises we will create different kinds of random networks, by producing their adjacency matrices. Note that we will be creating **random matrices**, that is, the existence of an edge is determined by drawing numbers from specific distributions such as the uniform or the gaussian distribution (if you need it, check wikipedia for references). We will also be using some __indexing operations__ from python's numpy.

# %% [markdown] id="_mi7CuCKJbuz"
# ### Drawing numbers from random distributions in python

# %% [markdown] id="sbV6InV9WVNh"
#
# We use [numpy's random number generators](https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
# ). For example, to draw random numbers from a uniform distribution:
#
# ```python
#
# W = np.random.uniform(low=-2,high=2, size=(5,5))
#
# ```
#
# to draw from a normal distribution, we'd write:
#
# ```python
#
# W = np.random.normal(size=(5,5))
#
# ```

# %% [markdown] id="scCMaM8I_Z4j"
# ### Sample Code: Ring Network with Three Nodes

# %% [markdown] id="h0ZbZ6eaXOi0"
#
# Here we create a **ring network** with three nodes:
# - node 1 is connected to node 2
# - node 2 to node 3
# - node 3 to node 1
#

# %% id="ENeYcM2W_3js"
# Define the matrix as an array
ring_net = np.matrix([[0,1,0], [0,0,1], [1,0,0]])
print('ring net adjacency matrix: \n',ring_net)

# plot the matrix via imshow
plt.figure()
plt.subplot(1,2,1)
plt.imshow(ring_net, vmin=0, vmax=1, cmap="gist_gray")
# vmin and vmax set the range of the colorbar
# cmap defines the colormap (https://matplotlib.org/3.1.1/tutorials/colors/colormaps.html?highlight=colormap)
# note: as an alternative to imshow you can use plt.pcolor(W).

plt.xlabel('to (j)')
plt.ylabel('from (i)')
#plt.show()


plt.subplot(1,2,2)

# in the line below we transform our matrix into a graph object (from networkx):
G_ring = nx.from_numpy_array(ring_net,create_using=nx.DiGraph()) # we use digraph because connections are directed
pos=nx.spring_layout(G_ring)
nx.draw(G_ring)

# annotate the plot
# nx.draw(G_ring, pos, with_labels=True) # Display node numbers labels =
# nx.get_edge_attributes(G_ring,'weight')
# nx.draw_networkx_edge_labels(G_ring, pos, edge_labels=labels) # Display edge weights


# %% [markdown] id="uKerUCGsL7bp"
# **Train your matrix reading skills**: Relate the non zero matrix entries to the graph you above.

# %% [markdown] id="2brWJgQB3siD"
# ### Sample Code: Recipe for Obtaining and Plotting a Weighted Directed Graph
#
# Create a weighted directed graph with 5 nodes where connections are drawn from a uniform distribution between -2 and 2.

# %% id="fI0WxhL13okU"
# RECIPE FOR WEIGHTED DIRECTED RANDOM GRAPHS

# 1. draw numbers from the uniform distribution
# from -2 to 2 to fill up a matrix of 5x5
WR = np.random.uniform(low=-2,high=2, size=(5,5))

# 2. print the weights
print("values of our matrix are \n") # /n is a line break
print(WR)

print("\n alternative representations as matrix or graphs:")

# 3. create a figure and populate it with axes
fig, ax = plt.subplots(1, 3, figsize=(9, 4))
fig.suptitle('Matrices to Graphs') # give a title to the whole thing

######## PLOT ADJACENCY MATRIX AS IMAGE

# 4. plot the adjacency matrix via imshow to visualize the weights
#    'pl' returns a handle to the plot so that we can add a colorbar to the subplot

pl = ax[0].imshow(WR, vmin=-2, vmax=2) # vmin and vmax set the range of the colorbar
plt.colorbar(pl, ax=ax[0]) # add a colorbar to the plot

# as an alternative to pcolor you can use plt.imshow().

ax[0].set_xlabel('to (j)')
ax[0].set_ylabel('from (i)')
ax[0].set_title('adjacency matrix')


######## PLOT GRAPH WITH SPRING LAYOUT

# 5. create graph from the new weight matrix W
G_weighted = nx.from_numpy_array(WR,create_using=nx.DiGraph())

# 6. get connections and strengths of connections
edges, weights = zip(*nx.get_edge_attributes(G_weighted,'weight').items())

# we don't need to specify spring layout for the nodes
# as it is assumed when no layout is specified in networkx
pos = nx.spring_layout(G_weighted)  # positions for all nodes

# 7. draw the new graph
nx.draw(G_weighted, pos, ax= ax[1], node_color='b',edgelist=edges, edge_color=weights, width=2, edge_cmap=plt.cm.Reds)

ed = nx.draw_networkx_edges(G_weighted, pos, ax=ax[1], edge_color=weights, width=2, edge_cmap=plt.cm.Reds)

# to create a colorbar that indicates the
#plt.colorbar(ed)
ax[1].set_title('spring layout');

# 8. graph with circle layout

pos = nx.circular_layout(G_weighted)  # positions for all nodes
nx.draw(G_weighted, pos, ax=ax[2], node_color='b',edgelist=edges, edge_color=weights, width=2, edge_cmap=plt.cm.Reds)

ed = nx.draw_networkx_edges(G_weighted, pos, ax=ax[2], edge_color=weights, width=2,arrows=False, edge_cmap=plt.cm.Reds)


ax[2].set_title('circle layout');


# %% [markdown] id="sl3b8zFaDGlg"
# ## Hands-on
#
# For all of the following items, create an adjacency matrix with 5 nodes, according to specifications:

# %% [markdown] id="EbrrhQEz4wfV"
# ### Exercise 1

# %% [markdown] id="TRMJJVmLVVZ4"
# Create and plot an **unweighted directed graph** where the adjacency matrix is filled with discrete random values, such that $P(w_{ij}=1) = 0.4$, and consequently,  $P(w_{ij}=0) = 0.6.$ Note that for the graph to be directed, the matrix is filled above and below the main diagonal. Plot the resulting adjacency matrix as an image.
#
# ---
#
# > Tip: one can use a boolean operator such as $>$ to transform real numbers into zeros and ones. For example, if:
# > ```python
# a = np.random.rand # get a random number between zero and one.
# print a
# >>a = 0.323
# ```
# >
# >then
# >
# >```python
# b = a>0
# >```
# > returns
# > ```python
# b = True
# ```
# > We need to change the boolean values from "True and False" into "ones and zeros" ("to cast"). A simple way to cast is simply to multiply by '1'.
# >```python
# b = (a>0)*1;
# ```

# %% [markdown] id="x_YfLEUTN6MT"
# #### Your Code

# %% id="AK8-feVLN6hH"

# %% [markdown] id="8jeu_BGxN2vf"
# #### Our Solution

# %% id="uvWJOeRf4Pgf"
# Exercise 1, our Solution:

W2 = np.random.rand(5,5) # obtain a 5,5 matrix with real numbers
W2 = W2 < 0.4 # a 'condition', or a 'threshold' for the existence of a link/edte
W2 = W2 * 1 # casting boolean (True/False) into integers (1,2,3,...)



plt.figure
plt.subplot(1,2,1)
# plot the adjacency matrix via pcolor to visualize the weights
plt.imshow(W2, vmin=0, vmax=1, cmap='gist_gray') # we use a binary matrix,
# so a black and white colormap seems appropriate.

# note: As an alternative to pcolor use plt.imshow(W1).
# Compare their outputs, can you spot the difference between the two?
# which is more intuitive?

plt.xlabel('to (j)')
plt.ylabel('from (i)')
plt.colorbar()

plt.subplot(1,2,2)
thisgraph = nx.from_numpy_array(W2,create_using=nx.DiGraph())
pos = nx.circular_layout(thisgraph)  # positions for all nodes
nx.draw(thisgraph, pos)

# in code challenge: can you annotate the nodes in this graph?
# You can copy and adapt code from previous solutions!


# %% id="lcvaHXC2a0XD"
pos

# %% id="-lwBH667RkPE"
# RECIPE FOR WEIGHTED DIRECTED RANDOM GRAPHS

# 1. draw numbers from the uniform distribution
# from -2 to 2 to fill up a matrix of 5x5
WR = np.random.uniform(low=-2,high=2, size=(5,5))

# 2. print the weights
print("values of our matrix are \n") # /n is a line break
print(WR)

print("\n alternative representations as matrix or graphs:")

# 3. create a figure and populate it with axes
fig, ax = plt.subplots(1, 3, figsize=(9, 4))
fig.suptitle('Matrices to Graphs') # give a title to the whole thing

######## PLOT ADJACENCY MATRIX AS IMAGE

# 4. plot the adjacency matrix via imshow to visualize the weights
#    'pl' returns a handle to the plot so that we can add a colorbar to the subplot

pl = ax[0].imshow(WR, vmin=-2, vmax=2) # vmin and vmax set the range of the colorbar
plt.colorbar(pl, ax=ax[0]) # add a colorbar to the plot

# as an alternative to pcolor you can use plt.imshow().

ax[0].set_xlabel('to (j)')
ax[0].set_ylabel('from (i)')
ax[0].set_title('adjacency matrix')


######## PLOT GRAPH WITH SPRING LAYOUT

# 5. create graph from the new weight matrix W
G_weighted = nx.from_numpy_array(WR,create_using=nx.DiGraph())

# 6. get connections and strengths of connections
edges, weights = zip(*nx.get_edge_attributes(G_weighted,'weight').items())

# we don't need to specify spring layout for the nodes
# as it is assumed when no layout is specified in networkx
pos = nx.spring_layout(G_weighted)  # positions for all nodes

# 7. draw the new graph
nx.draw(G_weighted, pos, ax= ax[1], node_color='b',edgelist=edges, edge_color=weights, width=2, edge_cmap=plt.cm.Reds)

ed = nx.draw_networkx_edges(G_weighted, pos, ax=ax[1], edge_color=weights, width=2, edge_cmap=plt.cm.Reds)

# to create a colorbar that indicates the
#plt.colorbar(ed)
ax[1].set_title('spring layout');

# 8. graph with circle layout

pos = nx.circular_layout(G_weighted)  # positions for all nodes
nx.draw(G_weighted, pos, ax=ax[2], node_color='b',edgelist=edges, edge_color=weights, width=2, edge_cmap=plt.cm.Reds)

ed = nx.draw_networkx_edges(G_weighted, pos, ax=ax[2], edge_color=weights, width=2,arrows=False, edge_cmap=plt.cm.Reds)


ax[2].set_title('circle layout');


# %% id="awZIiuMtRe_8"
# RECIPE FOR WEIGHTED DIRECTED RANDOM GRAPHS

# 1. draw numbers from the uniform distribution
# from -2 to 2 to fill up a matrix of 5x5
WR = np.random.uniform(low=-2,high=2, size=(5,5))

# 2. print the weights
print("The weights of our matrix are \n") # /n is a line break
print(WR)

print("\n alternative representations as graphs:")

# 3. create a figure and populate it with axes
fig, ax = plt.subplots(1, 3, figsize=(9, 4))
fig.suptitle('Matrices to Graphs') # give a title to the whole thing

######## PLOT ADJACENCY MATRIX AS IMAGE

# 4. plot the adjacency matrix via imshow to visualize the weights
#    'pl' returns a handle to the plot so that we can add a colorbar to the subplot

pl = ax[0].imshow(WR, vmin=-2, vmax=2, cmap=plt.cm.Reds) # vmin and vmax set the range of the colorbar
plt.colorbar(pl, ax=ax[0]) # add a colorbar to the plot

# as an alternative to pcolor you can use plt.imshow().

ax[0].set_xlabel('to (j)')
ax[0].set_ylabel('from (i)')
ax[0].set_title('adjacency matrix')


######## PLOT GRAPH WITH SPRING LAYOUT

# 5. create graph from the new weight matrix W
G_weighted = nx.from_numpy_array(WR,create_using=nx.DiGraph())

# 6. get connections and strengths of connections
edges, weights = zip(*nx.get_edge_attributes(G_weighted,'weight').items())

# we don't need to specify spring layout for the nodes
# as it is assumed when no layout is specified in networkx
pos = nx.spring_layout(G_weighted)  # positions for all nodes

# 7. draw the new graph
nx.draw(G_weighted, pos, ax= ax[1], node_color='b',edgelist=edges, edge_color=weights, width=2, edge_cmap=plt.cm.Reds)

ed = nx.draw_networkx_edges(G_weighted, pos, ax=ax[1], edge_color=weights, width=2, edge_cmap=plt.cm.Reds)

# to create a colorbar that indicates the
#plt.colorbar(ed)
ax[1].set_title('spring layout');

# 8. graph with circle layout

pos = nx.circular_layout(G_weighted)  # positions for all nodes
nx.draw(G_weighted, pos, ax=ax[2], node_color='b',edgelist=edges, edge_color=weights, width=2, edge_cmap=plt.cm.Reds)

ed = nx.draw_networkx_edges(G_weighted, pos, ax=ax[2], edge_color=weights, width=2,arrows=False, edge_cmap=plt.cm.Reds)


ax[2].set_title('circle layout');

# %% [markdown] id="m4NnGdmw6Yjs"
# ### Exercise 2
#
# Create an **unweighted directed graph** with no self-connections. Plot both the resulting adjacency matrix and the graph.

# %% [markdown] id="SifnDy7JVJ5P"
# #### Your Code

# %% id="WcF6-5t5VJhK"
# prompt: Create an unweighted directed graph with no self-connections. Plot both the resulting adjacency matrix and the graph.

# %% [markdown] id="p381nMRuVJKc"
# #### A Solution

# %% id="A9pP9VPA6aN1"
#############################
# Unweighted Directed Graph (udg)
# method 1.

udg = np.random.randint(2,size=(5,5))
# why do we use 2 as an argument above? Read the documentation for randint!

np.fill_diagonal(udg,0)

#############################

# method 2.
# another way to do it
udg = np.random.randint(2,size=(5,5))

for index, value in np.ndenumerate(udg):
  i = index[0] # we could use simply index[0]=index[1]
  j = index[1] # but we choose to make i, j explicit
  if i == j:
      udg[i,j] = 0

plt.figure
plt.subplot(1,2,1)
# plot the adjacency matrix via pcolor to visualize the weights
plt.imshow(udg, vmin=0, vmax=1, cmap='gist_gray') # notice vmin and vmax
plt.xlabel('to (j)')
plt.ylabel('from (i)')

plt.subplot(1,2,2)
udg_graph = nx.from_numpy_array(udg,create_using=nx.DiGraph())
pos = nx.circular_layout(udg_graph)  # positions for all nodes
nx.draw(udg_graph, pos)


nx.draw(udg_graph, pos, with_labels=True) # Display node numbers
labels = nx.get_edge_attributes(udg_graph,'weight')
nx.draw_networkx_edge_labels(udg_graph, pos, edge_labels=labels) # Display edge weights


# %% [markdown] id="2YjD7MSa_gWo"
# ### Exercise 3
#
# Make an **undirected unweighted random graph without self connections** by enforcing $w_{ij}= w_{ji}$.  $w_{ij} = 1$ for all existing edges (symmetrical adjacency matrix). Make sure you use integers to define connections (1 or 0). The connection probability should be 50%. As we do not have self connections, make sure that all elements of the main diagonal are 0, that is, $W_{ij}=0 \iff i=j$. Plot the resulting graphs and matrix.
#
# One neat way to do this:
#
# 1. Create a matrix with ```size=(5,5)``` random numbers between 0 and 1
# 2. To have 50% probability of connectivity, you can test whether the random numbers are larger than .5.
# 2. Extract the 'upper triangular' part of the matrix (def: a matrix with non-zero elements above the main diagonal). Numpy function that does this is ```np.triu()```.
# 3. Sum the matrix to its own 'transpose', a rotated version of the matrix to make the matrix symmetrical.
# 4. Set the elements of the main diagonal to zero (for example, multiplying by the matrix by I-1, where I is the identity matrix (```np.eye()```, in numpy)
#
#

# %% [markdown] id="I8PUcxgMNCwx"
# One neat way to remove self connections is by multiplying corresponding elements ("elementwise") the original matrix by another matrix where all diagonal entries are equal to zero and all other elements are one.

# %% [markdown] id="3DctC3WnX7tx"
# #### Your Solution

# %% id="SUaQhddRX00U"
#### Your Solution

# %% [markdown] id="eu4HjA7-X8sK"
# #### Our Solution

# %% id="t3QVk19N_yeh"
# we: create a random matrix, set probability to 0.5, extract 'upper triangular' matrix,
W4 =  np.triu(np.random.random(size=(5,5))> 0.5)

# adding the transpose of a matrix to itself will give symmetric matrix
W4 = W4.T + W4

# # set diagonal to zeros
zeros_diagonal = 1 - np.identity(5)

fig = plt.figure(figsize=(15,5))

plt.subplot(1,5,1)
plt.imshow(W4)
plt.title('W4 + W4.T')

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



# %% [markdown] id="LU5xifhDU1V5"
# # Advanced Graph Plotting: Changing Edges Properties
#
# With the networkx package one can edit the properties of the edges and nodes. Below are some examples.

# %% [markdown] id="0pq0YM8HXo-0"
# ### Create a Graph with Named Nodes:

# %% id="ZLbumfSAnQGJ"
# G_dir_W2 = nx.from_numpy_matrix(W2, create_using=nx.MultiDiGraph())
# (unweighted directed graph with self-connections)

G_dir_W2 = nx.MultiDiGraph(W2)
G_dir_W2.graph['edge'] = {'arrowsize': '10', 'splines': 'curved'}


plt.subplot(1,2,1)
pos=nx.spring_layout(G_dir_W2) # create positions for all nodes by distributing nodes with a spring force

# nodes
nx.draw_networkx_nodes(G_dir_W2,pos,
                       node_size=500,
                       node_color='skyblue',
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

# %% [markdown] id="TzEVERURXyxn"
# ### Create a Sizable Network
#
#  (for fun, just to get a sense of 'computational scalability' of these solutions).
#

# %% id="6TdNl9CBzHkU"
# 1. draw numbers from the uniform distribution
# from -2 to 2 to fill up a matrix
WBig = np.random.uniform(low=-2,high=2, size=(20,20))

# 3. create a figure and populate it with axes
fig, ax = plt.subplots(1, 3, figsize=(18, 9))
fig.suptitle('Matrix to Graphs') # give a title to the whole figure

######## PLOT ADJACENCY MATRIX AS IMAGE

# 4. plot the adjacency matrix via imshow to visualize the weights
#    'pl' returns a handle to the plot so that we can add a colorbar to the subplot
#
pl = ax[0].imshow(WBig, vmin=-2, vmax=2, cmap = plt.cm.seismic) # vmin and vmax set the range of the colorbar
plt.colorbar(pl, ax=ax[0]) # add a colorbar to the plot

# as an alternative to pcolor you can use plt.imshow().

ax[0].set_xlabel('to (j)')
ax[0].set_ylabel('from (i)')
ax[0].set_title('adjacency matrix')


######## PLOT GRAPH WITH SPRING LAYOUT

# 5. create graph from the new weight matrix W
G_big = nx.DiGraph(WBig)

# 6. get connections and strengths of connections
edges, weights = zip(*nx.get_edge_attributes(G_big,'weight').items())

# 7. create a layout for the nodes of the network
pos = nx.spring_layout(G_big)  # positions for all nodes


# 8. draw the new graph
nx.draw(G_big, pos, ax= ax[1], node_color='b',edgelist=edges, edge_color=weights, width=2, edge_cmap=plt.cm.seismic)

ed = nx.draw_networkx_edges(G_big, pos, ax=ax[1], edge_color=weights, width=2, edge_cmap=plt.cm.seismic)

# to create a colorbar that indicates the
# plt.colorbar(ed);
ax[1].set_title('spring layout');


######## PLOT GRAPH WITH CIRCLE LAYOUT

pos = nx.circular_layout(G_big)  # positions for all nodes
nx.draw(G_big, pos, ax=ax[2], node_color='b',edgelist=edges, edge_color=weights, width=2, edge_cmap=plt.cm.seismic)

ed = nx.draw_networkx_edges(G_big, pos, ax=ax[2], edge_color=weights, width=2, edge_cmap=plt.cm.seismic)
# to create a colorbar that indicates the
#plt.colorbar('grey');
ax[2].set_title('circle layout');

# %% [markdown] id="UCSCFhnC0R1i"
# # Challenges:
# - Create a graph with 1000 nodes, 20% chance of connections with no self connections using bezier curves ([see second answer in this post](https://stackoverflow.com/questions/52588453/creating-curved-edges-with-networkx-in-python3))
# - Display a feed forward neural network with 100 neurons and 10 layers, each with 10 neurons. Note: you will probably have to read the documentation of networkx, or maybe try to ask the AI!
#

# %% id="ey3hnsRgoKMr"
# # !pip install netgraph
from netgraph import Graph
Graph(G_big, edge_layout='curved', edge_cmap=plt.cm.seismic)
plt.show()

# %% [markdown] id="ZeSXl3Nc8sFi"
#
# # Questions:
# - How would you describe a matrix that represents a 'ring' graph, where $W_{ij}$ exists if $i=j+1$? (hint: does this matrix have a 'main diagona'?)
# - What would be the peculiarities of a network with clusters (where some nodes connect more to their neighbors than the other nodes)?

# %% [markdown] id="juUnvy-RWd2R"
# ## References
#
# [1]	[P. Hagmann, L. Cammoun, X. Gigandet, R. Meuli, C. J. Honey, V. J. Wedeen, and O. Sporns, “Mapping the Structural Core of Human Cerebral Cortex,” Plos Biol, vol. 6, no. 7, p. e159, 2008.](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.0060159)
#
# [2]	[O. Sporns, C. J. Honey, and R. Kotter, “Identification and Classification of Hubs in Brain Networks,” PLoS ONE, vol. 2, no. 10, 2007.](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0001049)
#
# [3] [M. Rubinov and O. Sporns, “Complex network measures of brain connectivity: Uses and interpretations,” Neuroimage, vol. 52, no. 3, pp. 1059–1069, Sep. 2010.](http://www.sciencedirect.com/science/article/pii/S105381190901074X)
#
# [4]	[E. Bullmore and O. Sporns, “Complex brain networks: graph theoretical analysis of structural and functional systems,” Nat Rev Neurosci, vol. 10, no. 3, pp. 186–198, Feb. 2009](https://www.researchgate.net/publication/23974889_Complex_brain_networks_Graph_theoretical_analysis_of_structural_and_functional_systems)
#

# %% [markdown] id="P84FYT368uj7"
# # Researching Further:
#
# - How do we measure connectivity in networks? (see reference [1,2])
#   - what is a hub?
#   - what is a 'node degree'?
#   - how to determine if a network is 'clustered'?
# - What kind of connectivity exists in brains? (see reference [2, 3])
# - What kind of connectivity exists in social networks?
# - What is a small world network? (search wikipedia)
#

# %% [markdown] id="-irKG64Jp496"
# # Further Research:
# - [Small World Networks](http://www.scholarpedia.org/article/Small-world_networks)
# - [Brain Connectivity](http://www.scholarpedia.org/article/Brain_connectivity)
#
#

# %% [markdown] id="zLuPW87skAzZ"
# #License
#
# <a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>. Mario Negrello (2020).

# %% id="BPumoLZD9udA"
# Better plotting of self loops and network structure
# https://stackoverflow.com/questions/44188755/show-self-loops-with-networkx-python

from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath  # To avoid collisions with pathlib.Path
import matplotlib.patches as patches
import networkx as nx
import numpy as np


# Some useful functions
def normalize_vector(vector: np.array, normalize_to: float) -> np.array:
    """Make `vector` norm equal to `normalize_to`

    vector: np.array
        Vector with 2 coordinates

    normalize_to: float
        A norm of the new vector

    Returns
    -------
    Vector with the same direction, but length normalized to `normalize_to`
    """

    vector_norm = np.linalg.norm(vector)

    return vector * normalize_to / vector_norm


def orthogonal_vector(point: np.array, width: float, normalize_to: Optional[float] = None) -> np.array:
    """Get orthogonal vector to a `point`

    point: np.array
        Vector with x and y coordinates of a point

    width: float
        Distance of the x-coordinate of the new vector from the `point` (in orthogonal direction)

    normalize_to: Optional[float] = None
        If a number is provided, normalize a new vector length to this number

    Returns
    -------
    Array with x and y coordinates of the vector, which is orthogonal to the vector from (0, 0) to `point`
    """
    EPSILON = 0.000001

    x = width
    y = -x * point[0] / (point[1] + EPSILON)

    ort_vector = np.array([x, y])

    if normalize_to is not None:
        ort_vector = normalize_vector(ort_vector, normalize_to)

    return ort_vector


def draw_self_loop(
    point: np.array,
    ax: Optional[plt.Axes] = None,
    padding: float = 1.5,
    width: float = 0.3,
    plot_size: int = 10,
    linewidth = 0.2,
    color: str = "pink",
    alpha: float = 0.5
) -> plt.Axes:
    """Draw a loop from `point` to itself

    !Important! By "center" we assume a (0, 0) point. If your data is centered around a different points,
    it is strongly recommended to center it around zero. Otherwise, you will probably get ugly plots

    Parameters
    ----------
    point: np.array
        1D array with 2 coordinates of the point. Loop will be drawn from and to these coordinates.
    padding: float = 1.5
        Controls how the distance of the loop from the center. If `padding` > 1, the loop will be
        from the outside of the `point`. If `padding` < 1, the loop will be closer to the center
    width: float = 0.3
        Controls the width of the loop
    linewidth: float = 0.2
        Width of the line of the loop
    ax: Optional[matplotlib.pyplot.Axes]:
        Axis on which to draw a plot. If None, a new Axis is generated
    plot_size: int = 7
        Size of the plot sides in inches. Ignored if `ax` is provided
    color: str = "pink"
        Color of the arrow
    alpha: float = 0.5
        Opacity of the edge

    Returns
    -------
    Matplotlib axes with the self-loop drawn
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(plot_size, plot_size))

    point_with_padding = padding * point

    ort_vector = orthogonal_vector(point, width, normalize_to=width)

    first_anchor = ort_vector + point_with_padding
    second_anchor = -ort_vector + point_with_padding

    verts = [point, first_anchor, second_anchor, point]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]

    path = MplPath(verts, codes)

    patch = patches.FancyArrowPatch(
        path=path,
        facecolor='none',
        lw=linewidth,
        arrowstyle="-|>",
        color=color,
        alpha=alpha,
        mutation_scale=30  # arrowsize in draw_networkx_edges()
    )
    ax.add_patch(patch)

    return ax

# %%
