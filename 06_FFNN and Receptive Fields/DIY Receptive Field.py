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
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="bc8e7afd"
# # Working in Google Colab
#
# This notebook is published from GitHub. Before editing it:
#
# 1. Sign in to Google, if prompted.
# 2. Click **Copy to Drive** at the top of Colab. If that button is not visible, choose **File → Save a copy in Drive**.
# 3. Close the original tab and work in the new Drive copy.
#
# To start over, reopen the original course link and make a fresh copy.

# %% [markdown] id="4872a426"
# # Project: DIY Receptive Field (FFNN)

# %% [markdown] id="9f770504"
# # Introduction

# %% [markdown] id="6fb2c2b3"
#
# This project prepares you to make multi-layered neural networks by analyzing the computational capabilities of single neurons in **feed forward networks**, by implementing basic **pattern recognition**. The project contains a primer about **simple artificial neurons**, and suggests implementations of this neuron in python.
#
# In the lecture you have learned that a neuron is said to have a receptive field for some **feature** if the neuron fires preferentially when that feature is presented to the organism.
#
# In this project, you will **manually design and tune a network** that can recognize oriented bars, thereby producing neurons that have an **orientation preference** and thus, a **tuning curve**. This should potentiate your **intuition** about how network weights **represent** preferred stimuli. You will learn how networks of neurons  can represent multiple preferred stimuli simultaneously, and thus you will learn first hand about how neurons conduct 'parallel distributed processing' (**PDP**). The insights herein are at the core of modern **deep learning** and **convolutional neural networks**.
#
# This project sets the stage for the introduction to backpropagation of errors in multilayered networks. You will learn about the **perceptron learning rule**, a supervised learning method that is used to optimize weights of single layer feed forward networks.
#

# %% [markdown] id="48bd9315"
# ## Key Terms
#

# %% [markdown] id="a5ba730c"
# - **Pattern Recognition**: The ability to ascribe a label (category) to a given pattern (input, e.g., pixel image).
# - **Receptive Fields**: A neuron is said to have a receptive field for a stimulus, when it has a specific response to that stimulus.
# - **Heavyside step function**: A function that returns 1 when the argument is positive and zero otherwise.
# - **Feed Forward Neural Networks (FFNN)**: Networks that are strictly forward (no recursion).
# - **Weight vectors**: A set of multiplicative values that apply to input.
# - **Bias**: A value that represents baseline activity.
# - **McCulloch-Pitts Neurons**: A neuron model that outputs 1 when a threshold is reached and zero otherwise.

# %% [markdown] id="643d1d59"
# ## Learning Goals

# %% [markdown] id="03fddd7c"
# - Implement a McCulloch–Pitts binary neuron in Python.
# - Explain how a dot product and Heaviside threshold determine a binary neuron's output.
# - Represent two-dimensional stimuli as one-dimensional vectors and Python data structures.
# - Tune feedforward-network weights to recognize oriented bars.
# - Evaluate classifier outputs using a confusion matrix.
# - Explain why receptive-field size and complexity increase across network layers.
# - Explain the role of convolutional layers in image recognition.
# - Train a perceptron from classification errors and describe how backpropagation extends this process.
# - Explain why a single-layer network cannot solve XOR and how additional layers address the problem.

# %% [markdown] id="6781547c"
# ## Initialization

# %% id="056bee59"
# always run this cell (shift+enter) to load relevant libraries

import matplotlib.pyplot as plt
from numpy import * # note that we're importing all of numpy into base namespace


# %% [markdown] id="d15052c0"
# # A Receptive Field
#
# In biology, a neuron is said to have a **receptive field** for some **feature** if the neuron fires preferentially when that features is presented to the animal. In the case of sensory receptive fields, one can have preferences for auditory frequencies, location of stimuli in visual, location of stimuli in tactile fields, visual features, specific odors, and the list goes on.
#
# Neurons that selectively respond to a certain stimuli are effectively categorizing said stimuli. A neuron that responds to a certain feature of a stimulus is said *to have a receptive field* for that feature. In the study of artificial neural networks, the accepted way to think about a neuron's receptive field is as a **classifier**: it fires the most when its **preferred stimulus** is present.
#
# <center>
# <img  src=https://qph.fs.quoracdn.net/main-qimg-957451c779574bcb4f9222c7801fcc11.webp align="middle" width="350">
# </center>
#
# You probably recognize this picture above as Hubel and Wiesel's demonstration of selection prefererence in V1 cortex of the cat.

# %% [markdown] id="817fc5f1"
# # A Simple Artificial Neuron

# %% [markdown] id="c20bdffb"
#
# In the most general case an **artificial neuron** takes in some **input** (such as currents from excitatory post-synaptic currents, EPSCs) and performs some **computation** on it (e.g., a threshold).
#
# In the most basic scenario, the neuron **sums weighted inputs** and **applies a threshold**. The neuron is said to be active (spike) if the threshold is reached. Though this is simpler, it is the same idea underlying the 1D integrate and fire neuron (IF).
#
# One of the earliest models of the neuron was the **McCulloch-Pitts** unit. In that conception, many simplifications were made:
#
# 0. A neuron either fires or does not, depending on whether the inputs cross a threshold value. The artificial Neuron produces 1/0 outputs.
# 1. The inputs to a neuron are represented by real numbers representing synaptic currents. Positive for excitatory inputs and negative for inhibitory inputs.
# 2. The inputs are weighted (multiplied by a weight). This represents a synaptic conductance. The inputs are then summed: that is, the neuron's inputs are **linearly combined**.
# 3. The threshold is a constant. It represents Sodium channels positive feedback driving a spike.
# 4. Inhibition works in the same way than excitation: equal amounts of inhibition and excitation cancel each other out.
# 5. The input to the neuron is a static pattern.
#
# Even though many of these simplifications are at odds with biological detail, they are a power explanatory tool for a neuron's information processing.
#

# %% [markdown] id="7ee43ba3"
# ---
# **A Linear Combination**
#
#  The "weighted sum of inputs" is referred to as a **linear combination**.
#
# > **Example:**
# > - if inputs are $a_1$, $a_2$, $a_3$ (or the 'vector' $\bf{a}$)
# > and weights are $w_1$, $w_2$, $w_3$ (or the 'vector' $\bf{w}$)
# > a linear combination is simply
# > $a_1 w_1 + a_2 w_2 + a_3 w_3$
#
# This can be represented by [matrix multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication):
#
#
# $$\begin{bmatrix}
# w_1 & w_2 & w_3
# \end{bmatrix}
# \begin{bmatrix}
# a_1 \\
# a_2 \\
# a_3
# \end{bmatrix} = \bf{w}^T \bf{a}$$
#
#
# which can be equivalently written in [dot product](https://en.wikipedia.org/wiki/Dot_product) form:
#
#
# $$\bf{w} \cdot \bf{a}$$
#
# ---

# %% [markdown] id="fa33408b"
# For a single McCulloch-Pitts artificial neuron receiving from N sources, we can frame the inputs ($a_n$) to the neuron ($\mathbf{a}$) as a vector with N entries. The input from each source is multiplied by its corresponding weight. The weight is a vector $\mathbf{w}$ with N weights (also called 'entries'). Weights can be positive (excitatory) or negative (inhibitory). Inputs to the neuron are multiplied by the weights and summed together ("linear combination"). If the result of this operation is larger than a threshold (often 0), we say the neuron is active.
#
# > One can think about the inputs as frequency or current of synaptic potentials.The weights representing the amplitude of the post-synaptic current (EPSC or IPSC).
#
# These operations are conveniently represented via matrix multiplication (or equivalently, for 1D vectors, the dot product). It is important to note that **matrix multiplication multiplies a row vector by a column vector**. The output of matrix multiplication of a column vector by a row vector is a number (a "scalar").
#
# Thus, the activity of a neuron receiving inputs $\bf{x}$ weighted by weights $\bf{w}$, is a [scalar](https://en.wikipedia.org/wiki/Scalar) representing its momentary activity.
#
# ---
#
# $$
# y = \mathbf{w} ^T \mathbf{x}
# $$
#
# ---
#
# In here, the **T** superscript indicates **transpose**, making $\bf{w}$  a column vector. **PAY CLOSE ATTENTION TO THE ORDER** in which $\mathbf{w}^T$ and $\bf{x}$ appear. Some authors use the reversed convention, in which the weight vector is a column vector following the activity vector.
#
# The *threshold* of the neuron neuron is the minimum value of summed inputs that results on its activation. A threshold is 'subtracted' from the summed inputs. Equivalently, this is sometimes added to the neuron representing *intrinsic activity*, or the *baseline firing rate* of the neuron, in which case it is called a *bias* (often represented as a $\theta$, also a scalar). Note that a *Bias* has a positive sign, and *threshold* has a negative sign.
#
# ---
#
# $$
# y = \mathbf{w} ^T \mathbf{x} - \theta
# $$
#
# ---
#

# %% [markdown] id="cfa9b93c"
# ### Heaviside Step Function
#

# %% [markdown] id="7b1d20eb"
# McCulloch-Pitts neurons either spike (1), or they do not (0) (but note that some real neurons have graded activity). Therefore, the only way to change propagation of information is changing the frequency of spikes along the axon, or firing rate. We can say that these neurons are **binary**, or **boolean**.
#
# A threshold function that distinguishes positive from negative ( *output one if input is larger than threshold*) is often called the Heaviside step function. This is in honor of [Olivier Heaviside](https://en.wikipedia.org/wiki/Oliver_Heaviside), who was a self-taught electrical engineer and mathematician who worked on circuits and logic gates.
#
# ---
#
# $$
# \begin{split}H(x) = \begin{cases} 0, & x < 0, \\ 1, &x \ge 0. \end{cases}\end{split}
# $$
#
# And so, a McCulloch-Pitts neuron with a Heaviside function can be written as
#
# $$
# \begin{split}H(\mathbf{w}^T \mathbf{x}-\theta) = \begin{cases} 0, & \mathbf{w}^T \mathbf{x} < \theta, \\ 1, & \mathbf{w}^T \mathbf{x} \ge \theta. \end{cases}\end{split}
# $$
#
# where $\theta$ is the threshold (also 'bias') of the neuron.
#
# ---
#
#
#

# %% [markdown] id="62d757c3"
# Finally, the (McCulloch-Pitts) neuron can either fire (1) or not (0), if the sum is larger than a threshold ($\theta$), a value representing the minimum current that the neuron requires to activate sodium channels.
#
# ---
#
# $$
# H(\mathbf{w} ^T \mathbf{x} - \theta )
# $$
#
# ---
# Where $H(x)$ is a the **Heaviside function**, which checks whether $x > 0$ and responds with a $1$ if yes and a $0$ if no.
#
# It is easy to define a function that acts like this simple neuron with *python*. If ```inputs``` is the $x$  vector, weights is the $w$ a vector, bias $\theta$ is a scalar (single number), a neuron can be written like this:

# %% [markdown] id="0c0cac60"
# *If you do not know how to define a function in python, check [this link](https://pythonbasics.org/functions/).*
#
# ```python
# def Heaviside(x):
#     if x >= 0:
#       y = 1
#     else
#       y = 0
#     return y
# ```

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 11, "status": "ok", "timestamp": 1759818237768, "user": {"displayName": "Mario Negrello", "userId": "10136788594790905986"}, "user_tz": -120} id="a2c9fd55" outputId="b07bd9e7-3a2a-47f2-d06c-1598ea4463e4"
#@title ### Sample Code

# example:
# we define the function above and call it with for example, x=-1 or x=1

def Heaviside(x):
    if x >= 0:
      y = 1
    else:
      y = 0
    return y

# test the function with various values of x
x = 0.3
print("the Heaviside of "+str(x)+" is "+str(Heaviside(x)))

x = 0.
print("the Heaviside of "+str(x)+" is "+str(Heaviside(x)))

x = -1.4
print("the Heaviside of "+str(x)+" is " +str(Heaviside(x)))

x = -inf # where inf represents infinity
print("the Heaviside of "+str(x)+" is " +str(Heaviside(x)))

x = inf
print("the Heaviside of "+str(x)+" is " +str(Heaviside(x)))

x = nan
print("the Heaviside of "+str(x)+" is " +str(Heaviside(x)))

# %% [markdown] id="4094a94b"
# ### Exercise: A Single Neuron
# Create a working function for a single ```neuron(...)``` according to the definition above. The neuron takes in a vector of inputs $\mathbf{x}$, a weight vector $\mathbf{w}$, and a threshold ($\theta$) and outputs `0` or `1`.

# %% colab={"base_uri": "https://localhost:8080/", "height": 159} executionInfo={"elapsed": 17, "status": "error", "timestamp": 1696324212593, "user": {"displayName": "Mario Negrello", "userId": "10136788594790905986"}, "user_tz": -120} id="31d45806" outputId="dd855738-39a3-49d9-bdd9-7c7c9d58dab7"
#@title ### Example Code

# 1. Define an input vector x, weight vector w, and a threshold theta
w = array([0.2, 0.7, -0.4])
x = array([1.0, -0.5, 0.3])
theta = 0.25

# 2. Define a python function called neuron that takes in weights, inputs and a threshold
# and computes the Heaviside output of the weighted sum of inputs.
def neuron(x, w, theta):
  y = w.T @ x - theta
  return 1. * (y >= 0)

# 3. test it on your stimuli
output = neuron(x, w, theta)
print(output)


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 246, "status": "ok", "timestamp": 1696324219093, "user": {"displayName": "Mario Negrello", "userId": "10136788594790905986"}, "user_tz": -120} id="14b78251" outputId="2bcbcb97-3ca5-49cd-b3b3-127b41b7d3b0"
#@title ### Sample Code

# create some inputs
x = random.uniform(-2,2,size=(3,1))
w = random.uniform(0,1,size=(3,1))
theta = 0.5


def neuron(x,w,theta):
    # linear combination using @ for the dot product
    cell_body_sum = w.T@x - theta
    # and fire (output)

    # multiply by 1. to 'cast' result as float (type)
    return 1. * (cell_body_sum >= 0)

output = neuron(x,w,theta)
print("the threshold value theta is " +str(theta))
print("the weight vector w is "+str(w))
print("the input vector x is " +str(x))
print("the output of your neuron is "+str(output))


# %% [markdown] id="9525fa45"
# *If you try to run our sample code multiple times does the output of the neuron change?* yes, as the input is generated as vectors with random uniform distributed values

# %% [markdown] id="686914df"
# Evidently, the simplifications for this neuron are on the *heavy side* (poor pun). Biological neurons are in dynamical systems. For exmple, neurons often are sensitive for timing between the synaptic potentials, and dendrites perform non-linear operations such as amplifying input signals. Nevertheless, under many situations, even the most complex neurons can be thought of as performing nature's version of this yes/no operation. Case in point, receptive fields of neurons are compellingly represented by the basic operations of an articicial neuron as defined above.
#

# %% [markdown] id="69a74543"
#
# ## The Binary Neuron as a Linear Classifier  
#

# %% [markdown] id="becc940a"
#
# The simple artificial neuron described above performs the function of a **classifier**. It classifies its inputs into those that activate  it, and those that do not.
#
# For example, a classifier could tell you whether inputs belong to the class of penguin. Or it could classify things that happen together: this **AND** that have to be the case. This is the *boolean* function **AND**. A neuron with two inputs that computes an **AND** function returns `1` (`True`) if both input one **AND** input two are `1`. It should says `0` (`False`) if any of the inputs are absent.
#
# With properly tuned weights and bias, a linear classifier can compute most [logical gates](https://en.wikipedia.org/wiki/Logic_gate), such as (AND, OR, NOT, NAND).

# %% [markdown] id="ac1f5fc6"
# ### An **AND** gate

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 447, "status": "ok", "timestamp": 1696324229304, "user": {"displayName": "Mario Negrello", "userId": "10136788594790905986"}, "user_tz": -120} id="9b8dc857" outputId="bd6ba1d2-da6e-4084-a50a-4fea7643f439"
def AND(a,b):
    if a == 1 and b == 1:
        return True
    else:
        return False

print(AND(0,1))

AND(1,1)



# %% [markdown] id="a13df669"
# Ask yourself: *can you find weights and bias to calculate an **AND** gate with the artificial neuron above?*

# %% [markdown] id="74443c15"
# ### The **XOR** gate

# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 410, "status": "ok", "timestamp": 1696324231632, "user": {"displayName": "Mario Negrello", "userId": "10136788594790905986"}, "user_tz": -120} id="768373f0" outputId="84a9676d-372c-4cba-d38e-ae402caf62b1"
def XOR(a,b):
    if a != b:
        return 1
    else:
        return 0
XOR(1,0)

# %% [markdown] id="7b46d9fa"
# ----
#
# #### Question
#
# - One of the two logical functions above cannot be performed by the artificial neuron defined earlier. Can you guess which one? Explain your answer.
#
# ---

# %% [markdown] id="0c4ccf48"
# #### Answer
#
# *Double click this cell and type your answer here*
#

# %% [markdown] id="891ac3f3"
# # FFNNs: Encoding receptive fields
#

# %% [markdown] id="a8d336b8"
#
# ## Exercise: Manually Tune The Weights of a Neuron
#

# %% [markdown] id="d47358d7"
#
# In this exercise, you are expected to **manually tune a neuron by selecting weights vector** (and maybe a bias vector), so that the neuron is selective for a feature, that is, it only responds to a specific stimulus. The purpose of this exercise is to build intuition about what do the weights have to look like for a neuron to respond selectively to certain patterns.
#
# > **Roadmap**:
# - create vectors representing stimulus (using dictionaries)
# - choose weights for a single output neuron such that the neuron says ```1``` to the preferred stimulus and ```0``` to other stimuli.
# - choose weights for more output neurons, one for each stimulus orientation.
# - test the classification
#

# %% [markdown] id="e0e0f1cd"
#
# ## Creating Stimuli

# %% [markdown] id="3a5a7948"
#
# Before we start selecting weights, we must create inputs to our network. For this exercise we will be creating stimuli that are **oriented bars** representing how light activates a tiny patch of the retina, of say, 3 x 3 rods (photoreceptors that are responsive to luminance).The cells involved with processing visual data can be very specific with their responses. As Hubel and Wiesel discovered, in the visual cortex (V1)  some neurons only fire (action potentials) for bars at certain angles.
#
# As you remember, the retina has a dense array of rods, but here we are considering only receptors that are really close to each other, a very small "visual angle". We will represent active photoreceptors with `1` and silent photoreceptors with `0`.
#
#
# Before we start the process of selecting weights, it will be useful to create test inputs to our network. In the sample code below we will be creating oriented bars that represent how light activates a small patch of the retina, of say, 3 x 3 pixels. Thus, the initial stimulus is a 3x3 bitmap. We will be creating these stimuli as numpy arrays introduced into python's data type 'dictionary'.
#

# %% [markdown] id="e2ccc0fe"
# ### Example: Creating Stimuli

# %% [markdown] id="18bcb509"
# Using numpy's `array()` we create stimuli with zeros for dark pixels and ones for bright pixels. Create 4 stimuli for a vertical bar, a horizontal bar, and two diagonal bars (left and right slant, i.e., / and \), all of them using three 'on' pixels centered in the patch. We are going to present these stimuli to the network, to test whether our choice of weights works.

# %% id="a79fd30d"
#@title ### Sample Code

# want to represent how light activates a part of the retina,
# so the '1' is bright (on) and '0' is dark.
#
# create 3x3 matrices representing the way light activates
# pixels in a small 2D patch
S1 = array([[0,1,0], [0,1,0], [0,1,0]])
S2 = array([[0,0,0], [1,1,1], [0,0,0]])
S3 = array([[1,0,0], [0,1,0], [0,0,1]])
S4 = array([[0,0,1], [0,1,0], [1,0,0]])


plt.figure
plt.subplot(1,4,1), plt.imshow(S1,cmap='gist_gray');
plt.subplot(1,4,2), plt.imshow(S2,cmap='gist_gray');
plt.subplot(1,4,3), plt.imshow(S3,cmap='gist_gray');
plt.subplot(1,4,4), plt.imshow(S4,cmap='gist_gray');

# %% [markdown] id="e37eea7a"
# ### Transforming Stimuli Matrices Into Input Vectors

# %% [markdown] id="38d02960"
# Above we have created some matrices representing the 2D patch of retinal. But the inputs to a neuron do not necessarily maintain the spatial orientation of the incoming synapses (that is, it is not necessarily the case that the input synapse arriving at the top of the neuron should come from the retinal pixel at the top -- in other words, *topography* is not kept).
#
# To pass on the stimlus to our networks we have to transform them in a list (a vector, to perform linear combination). So we need to rearrange the elements of the stimulus matrices as **input vectors**.
#
# We will want to take each of the entries of our stimuli dictionary and reshape the matrices as 1D vectors. (i.e., **We need to vectorize the matrices**).

# %% id="822018f8"
#@title ### Sample Code

# one way to reshape all matrices into vectors is to use flatten
v_v = S1.flatten()
v_h = S2.flatten()
v_dl = S3.flatten()
v_dr = S4.flatten()

print(v_v)
print(v_h)
print(v_dl)
print(v_dr)

# %% [markdown] id="afcd67ed"
# ------------
# > #### Interlude: Dictionaries
# >
# > Python works with several different datatypes; lists, dictionaries, tuples, arrays, to name a few. A **dictionary** is datatype that collects and can be indexed via 'keys' (a string for example). Dictionaries are written with curly brackets. While other compound data types have only value as an element, a dictionary has a ```"key": value ``` pair. Simply put, you can use a dictionary to retrieve 'values' that matches a given `key`.
# >
# > Each key is separated from its value by a colon `(:)`, the items are separated by commas, and the whole thing is enclosed in curly braces. An empty dictionary without any items is written with just two curly braces, like this: `{}`.
# >
# > Keys are unique within a dictionary while values may not be. The values of a dictionary can be of any type, but the keys must be of an immutable data type such as strings, numbers, or tuples.
# >
# > ```python
# numbers = {'one':1, 'two':2, 'three':3}
# ```
# > and so calling our  example as such
# > ```python
# numbers['one']
# ```
# > returns a 1.
# ------------
#

# %% [markdown] id="183b15ea"
# #### Sample Code: Stimulus Dictionary

# %% id="df21b9d1"
# Here we create a dictionary with the stimuli (so that it becomes easy to manage them)
stimuli = {'horizontal':v_h, 'vertical':v_v,'diagonal_left':v_dl, 'diagonal_right':v_dr}

# iterate through values in the dictionary
# more ways to do it here: https://realpython.com/iterate-through-dictionary-python/#iterating-through-keys-directly
for key in stimuli:
    print(key, '->', stimuli[key])

# %% [markdown] id="f2941fe6"
# ## Designing Weights for a Classifier Neuron
#
# Here we will like to design weights and choose a threshold such that the output neuron has a preference for  **vertical oriented bar**, smack in the middle of the receptive field of the neuron (see the third neuron in our code above). The neuron SHOULD NOT FIRE to any of the other stimuli.

# %% [markdown] id="daa3ee6f"
# ### Select Weigths for a Vertically Tuned Neuron

# %% [markdown] id="89ca1384"
# Select a weight vector and theta such that your neuron is active when a vertical line is presented to the neuron's receptive field.
#
# Fill in the weights and theta in the code cell below.
#
# Test the output of your neuron with all the stimuli in the stimulus dictionary (created above).

# %% cellView="form" id="99345006"
#@title ### Example Code

# DEFINE your weight vector
w_vertical_neuron = array([-1, 2, -1, -1, 3, -1, -1, 2, -1])

# CHOOSE a threshold
theta = 6

# SELECT the vertical stimulus from dictionary
vertical_stimulus = stimuli['vertical']

# CALL the function with your stimulus and chosen weights
test1 = neuron(vertical_stimulus, w_vertical_neuron, theta)
print(f'test1 = {bool(test1)}')

# TEST the function for every orientation and check that it responds adequately
test_1 = neuron(stimuli['horizontal'], w_vertical_neuron, theta)
test_2 = neuron(stimuli['diagonal_right'], w_vertical_neuron, theta)
test_3 = neuron(stimuli['diagonal_left'], w_vertical_neuron, theta)

print(f'test_1 = {bool(test_1)}')
print(f'test_2 = {bool(test_2)}')
print(f'test_3 = {bool(test_3)}')


# %% [markdown] id="22408412"
# *if test1 is True and all others are false, congratulations! Else go back to the weight design phase.*

# %% [markdown] id="777880e5"
# ---
# #### Question
# - What would happen if the contrast between active and inactive pixels would be lower? For example, what if our `on` pixels had a value of `0.6` and the `off` pixels a valu of `0.7`? Would your weights still work? Can you think of a strategy to make your neuron more general?
# ---

# %% [markdown] id="77TLPlEWoMDN"
# #### Answer
#
# *Double click this cell and type your answer here*

# %% [markdown] id="b87aa4c9"
# ## Feed Forward Network of Classifiers

# %% [markdown] id="df3d07b6"
#
# Our stimuli are coming from 9 pixels and mapping on a single ouptut neuron that recognizes a vertical bar.
#
# What if we want four output neurons, each with a different prefered orientation?
#
# Essentially, the operation for each neuron is the same linear combination of inputs.
#
# $$ \bf{W} \bf{x}$$
#
# But now $\bf{W}$ is a **weight matrix** that contains one (row) weight vector for each of the output neurons. To produce the matrix we simply stack four weigth vectors, each with weights standing for different preferences.

# %% [markdown] id="a2c72a58"
# ### Extending the Network
#
# 1. Extend your `neuron()` function to compute the output of a one layer feed forward neural network.
# 2. Define weight vectors, each of which with a particular orientaion preference.
# 3. Concatenate these into a weight matrix.
# 4. Test the output of your network for all of the previously prepared stimuli.
# %% [markdown] id="b8a1cab0"
#
#

# %% id="5d4fd7da"
#@title ### Sample Code

# DEFINE the weight vectors (as np.arrays) for different output neurons
w_vertical = array([-1,2,-1,-1,3,-1,-1,2,-1])
w_horizontal = array([-1,-1,-1,2,3,2,-1,-1,-1])
w_diagonal_L = array([2,-1,-1,-1,3,-1,-1,-1,2])
w_diagonal_R = array([-1,-1,2,-1,3,-1,2,-1,-1])

# DEFINE a threshold
theta = 6

# CONCATENATE the weight vectors into a weight matrix
## https://stackoverflow.com/questions/20978757/how-to-append-a-vector-to-a-matrix-in-python
# weight_matrix = np.c_[w_v , w_h , w_dr , w_dl]

weight_matrix = c_[w_vertical , w_horizontal , w_diagonal_R , w_diagonal_L]


# CHECK that your matrix has the right shape:
# print(weight_matrix.shape)

# EXTEND your single neuron function to compute the output of a feed forward neural network

def FFNN(W, I, theta):
  # If W is the weigth matrix and
  # I is the input vector
  # theta is the threshold

  # to compute the output
  # use matrix multiplication ("@" in numpy) and apply a threshold (heaviside)
  neuronlist=[]

  for i in range(W.shape[0]):
    weights=W[i,:]
    y=weights.T@I-theta
    neuronlist.append(1* (y >= 0))
  return neuronlist

# TEST your network on all stimuli

# you can use our function to do so:
# compute outputs for all stimuli and all neurons
print('The order of neuron preferences are: Vertical, Horizontal, Diagonal_right and Diagonal_left\n')
for key, value in stimuli.items():
  print(f'For {key} stimulus')
  print(f'{FFNN(weight_matrix.T, stimuli[key], theta)} \n')



# %% [markdown] id="981a7875"
#
# ### Exercise / Challenge: A Cross
#
# Design an weigth vector for an output neuron that:
# - returns `1` for a vertical_bar stimulus
# - returns `1` for a horizontal_bar stimulus
# - returns `0` for a cross stimulus
#
# We defined the cross for your convenience
# ```python:
# cross = array([0,1,0],[1,1,1],[0,1,0])
# ```
#
# For this exercise add this cross stimulus to your stimulus dictionary and add another weight vector that performs as above.
#

# %% [markdown] id="0b5ffcc7"
# ### Your Code

# %% id="83f1b5c8"
# Your Code
cross = array([[0,1,0],[1,1,1],[0,1,0]])
stimuli['cross']=cross.flatten()
plt.imshow(cross,cmap='gist_gray')

crossweights=array([-1,-3,-1,-3,3,2,-1,2,-1])
theta=2
weight_matrix_cross=c_[weight_matrix, crossweights]
print("The order of neuron preferences are: Vertical, Horizontal, Diagonal_right and Diagonal_left, cross\n")
for key, value in stimuli.items():
  print(f'For {key} stimulus')
  print(f'{FFNN(weight_matrix_cross.T, stimuli[key], theta)} \n')


# %% [markdown] id="84ac25a1"
# ---
# #### Question
# - were you able to design a weight vector solving the challenge above? Explain your answer.
# ---

# %% [markdown] id="YNj2HSf-oOtP"
# #### Answer
#
# *Double click this cell and type your answer here*

# %% [markdown] id="2b690026"
# ## Testing Classification Accuracy

# %% [markdown] id="eae88db0"
# A neuron with a receptive field for a horizontal bar, may also fire when the bar is not perfectly horizontal.
#
# We introduce a **confusion matrix**, which is a way to see how well a network is able to classify stimuli. Confusion matrices are used to test the performance of a network that classifies stimulus (a 'model'). Confusion matrices count the number of times that a certain stimulus was assigned to a given category. That is, they display the label given by the network as a function of the true label. Note: the confusion matrix can only be used if one has 'true labels' for every single stimulus being presented, that is, if the dataset is **supervised**.
#
# Below is an example of a [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix) for image classification, that tries to distinguish Dogs from Rabbits and Cats. In the case of this network, we have 'trained it' to try and see if a picture is of a cat or a dog.
#
#
# ![](https://i.stack.imgur.com/Rz5ol.jpg)
#
# <!-- another, for  standard one
#
# ![](https://datatofish.com/wp-content/uploads/2018/12/003_cm.png)
#  -->
#

# %% [markdown] id="1573c033"
# ---
# #### Questions
# - How many rows and columns will be in a confusion matrix that attempts to classifies 10 stimuli in 4 categories?
# - Can a confusion matrix have more rows than categories? Explain your answer.
# - According to the confusion matrix above, what is the most common mistake of the classifier?
# ---

# %% [markdown] id="NAcHvZdUoRhB"
# #### Answer
#
# *Double click this cell and type your answer here*

# %% [markdown] id="e965a0bf"
# ## Multiple Layers: Convolutions
#
#
#
#
#

# %% [markdown] id="31b3301c"
# The term receptive field is also used in the context of neural networks, where it refers to the region in the **input space** that affects a particular neuron in the network.
#
#
#
#  In deep learning newtorks (receptive fields are consequences of **convolutional filters** are used to preprocess a large image by decomposing that image in multiple features. This type of model architecture, which attempts to mimick the hierarchical way visual processing (and pre-processing in other senses), is called **convolutional neural network**.
#
# A clever trick is introduced; instead of connecting each neuron in one layer to each other neuron of the next, the neurons are organised as such accoring to their spatial relationships in the input data. That is, each neuron in the convolutional layer receives data from only a *local patch* of inputs. Neurons in one convolutional layer will perform the same kind of operation, that is, they will extract the same kind of feature.
#
# This patch, determined by which neurons the convolutional layer receives from, is called the (local) receptive field.
#
# The feed forward neural networks we just tuned can in fact be used as convolutional filters. All which is required, is that we have a whole layer of neurons searching for a feature in all the patches of the whole picture.
#
# <a title="Vincent Dumoulin, Francesco Visin, MIT &lt;http://opensource.org/licenses/mit-license.php&gt;, via Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:Convolution_arithmetic_-_Full_padding_no_strides_transposed.gif"><img width="128" alt="Convolution arithmetic - Full padding no strides transposed" src="https://upload.wikimedia.org/wikipedia/commons/8/85/Convolution_arithmetic_-_Full_padding_no_strides_transposed.gif"></a>
#
# This is a set of neurons with receptive fields of 3x3 pixels looking into a larger input picture (blue). The result of this 'convolution' layer (green), could be fed into another layer of neurons, making the network 'deeper'. Compositions of convolutional layers are thought to underly processing in the visual stream, as exemplified in the figure below.
#
# The 'convolutional filters', essentially the 'neuronal preferences' are indicated by small 3x3 filters, like the one you designed. The outputs of this pre-processing are indicated in the light blue rectangle.
#

# %% [markdown] id="017a4087"
# This excerpt from the "Principles of Neuroscience (5th Ed.)" demonstrates the notion of composition of receptive fields, deriving from Hubel and Wiesel's model of simple and complex cells of the visual cortex.
#
# [![Hubel and Wiesel Model](https://i.postimg.cc/ncgy9Vcj/image.png)](https://postimg.cc/QHg4R37h)
# (from "Principles of Neuroscience, 6th Ed. Appendix "Neural Networks")

# %% [markdown] id="3d7c0c5b"
# ---
# # Final Questions
# The following questions may require some research or some out of the box thinking. You are welcome to check the available resources, or ask an AI!
#
# - Why is the perceptron a 'linear' classifier'?
# - When can we say that a neuron has a receptive field?
# - Why must the weights of a classifier 'resemble' their preferred stimulus?
#
# ---

# %% [markdown] id="RRcXpn-_oZPf"
# #### Answers
#
# *Double click this cell and type your answer here*

# %% [markdown] id="649d2702"
# # Resources
# - [3Blue1Brown on Neural Networks](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiG_M-fm4nsAhVMCewKHQ4iD00QyCkwAHoECAQQAw&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DaircAruvnKk&usg=AOvVaw3c_pmQ67XtWSaAXtAgCxkl)
# - [A medium post on representing networks via matrices](https://medium.com/coinmonks/representing-neural-network-with-vectors-and-matrices-c6b0e64db9fb)
# - Convolutions, aka, [Morphological operations, in python](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html)
# - distill.pub [Interpretability](https://distill.pub/2018/building-blocks/)
# - http://neuralnetworksanddeeplearning.com/chap6.html

# %% [markdown] id="6bf15209"
# #License
#
# <a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
#
# Mario Negrello, Daphne Cornelise, Elias Santoro. Reviewing and testing by many students.

# %% id="0aab4e95"
