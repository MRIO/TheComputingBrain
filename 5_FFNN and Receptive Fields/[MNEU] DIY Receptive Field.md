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
    language: python
    name: python3
---

<!-- #region id="9BW_FsUlMiWZ" -->
# To be able to edit and use this Notebook:

0. Learn about how to use google colaboratory [video](https://www.youtube.com/watch?v=inN8seMm7UI)
1. in the file menu (top left), click ```open in playground```
3. still in the file menu, click ```save copy in drive```, to make your own personalized and editable copy of this file.
4. edit as you like. If something breaks irreparably, either:
  1. restart the ```Runtime```
  2. or go back to step 1.

<!-- #endregion -->

<!-- #region id="0YWViKr381Qj" -->
# TODO:
 - add question/explanation about XOR
 - make new project with perceptron
 - explain better how to produce a confusion matrix
 - @mario: checkout this video: https://www.youtube.com/watch?v=YRhxdVk_sIs
<!-- #endregion -->

<!-- #region id="-cSHJNJ-JwFZ" -->
# Project: DIY Receptive Field (FFNN)
<!-- #endregion -->

<!-- #region id="HM5zRrWA5DDA" -->
# Introduction
<!-- #endregion -->

<!-- #region id="d9NrFsImK5Kw" -->

This project prepares you to make multi-layered neural networks by analyzing the computational capabilities of single neurons in **feed forward networks**, by implementing basic **pattern recognition**. The project contains a primer about **simple artificial neurons**, and suggests implementations of this neuron in python.

In the lecture you have learned that a neuron is said to have a receptive field for some **feature** if the neuron fires preferentially when that feature is presented to the organism. 

In this project, you will **manually design and tune a network** that can recognize oriented bars, thereby producing neurons that have an **orientation preference** and thus, a **tuning curve**. This should potentiate your **intuition** about how network weights **represent** preferred stimuli. You will learn how networks of neurons  can represent multiple preferred stimuli simultaneously, and thus you will learn first hand about how neurons conduct 'parallel distributed processing' (**PDP**). The insights herein are at the core of modern **deep learning** and **convolutional neural networks**.

This project sets the stage for the introduction to supervised learning. Later you will learn about the **perceptron learning rule**, a method that is used to optimize weights of single layer feed forward networks as the ones you will design here.

<!-- #endregion -->

<!-- #region id="gInu5ju-jVmk" -->
## Key Terms

<!-- #endregion -->

<!-- #region id="D-csXHNbkjNC" -->
- **Pattern Recognition**: The ability to ascribe a label (category) to a given pattern (input, e.g., pixel image).
- **Receptive Fields**: A neuron is said to have a receptive field for a stimulus, when it has a specific response to that stimulus.
- **Heavyside step function**: A function that returns 1 when the argument is positive and zero otherwise.
- **Feed Forward Neural Networks (FFNN)**: Networks that are strictly forward (no recursion).
- **Weight vectors**: A set of multiplicative values that apply to input.
- **Bias**: A value that represents baseline activity.
- **McCulloch-Pitts Neurons**: A neuron model that outputs 1 when a threshold is reached and zero otherwise. 
<!-- #endregion -->

<!-- #region id="Ax7ul1wwjyOv" -->
## Learning Objectives
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


NOTEBOOK_REQUIREMENTS = [('ipywidgets', 'ipywidgets'), ('matplotlib', 'matplotlib'), ('numpy', 'numpy')]
ensure_notebook_packages(NOTEBOOK_REQUIREMENTS)

if IS_COLAB:
    colab_output.enable_custom_widget_manager()

```

```python id="BY_dPNdhypmO"

```

<!-- #region id="4MPQBidSj190" -->
- Students can code a simple "McCulloch-Pitts" binary neuron in python.
- Students understand how the operation of a simple neuron is represented by the dot product and a threshold ("Heaviside") function.
- Students are able to manually tune weights for a feed forward network to recognize oriented bars.
- Students develop intuition about the XOR problem and why feed forward neural networks with only one layer cannot solve it.
- Students learn how  to use python dictionaries to 'contain' stimuli.
- Students learn about a 'confusion matrix' to examine the outputs of their manually tuned classifiers.
--- 

This sets the stage for:

---

- Students can provide reasons for the increase of receptive field complexity (and of receptive field size) in multilayer neural networks;
- Students can explain the role of convolutional layers in image recognition.
- Students can use an error to train a perceptron to recognize a set of patterns.
- Students know how to use backpropagation to train a network of classifiers.
- Understanding the role of multiple layers in solving the XOR problem.
<!-- #endregion -->

<!-- #region id="hb313TL2nfsx" -->
## Initialization
<!-- #endregion -->

```python id="l62xzm4ZXdWx"
# dependencies
import matplotlib.pyplot as plt
from numpy import *

from ipywidgets import interactive, Button

# import ipywidgets
from IPython.display import display, clear_output

```

<!-- #region id="xG5EINlJg_ey" -->

-------------------
<!-- #endregion -->

<!-- #region id="LoFVLqu_THax" -->
# A Simple Neuron
<!-- #endregion -->

<!-- #region id="hUSqyv3ej-_C" -->

In the most general case an **artificial neuron** takes in some **input** (such as currents from excitatory post-synaptic currents, EPSCs) and performs some **computation** on it (e.g., a threshold). 

In the most basic scenario, the neuron **sums weighted inputs** and **applies a threshold**. The neuron is said to be active (spike) if the threshold is reached. Though this is simpler, it is the same idea underlying the 1D integrate and fire neuron (IF).

One of the earliest models of the neuron was the **McCulloch-Pitts** unit. In that conception, many simplifications were made:

0. A neuron either fires or does not, depending on whether the inputs cross a threshold value. The artificial Neuron produces 1/0 outputs.
1. The inputs to a neuron are represented by real numbers representing synaptic currents. Positive for excitatory inputs and negative for inhibitory inputs.
2. The inputs are weighted (multiplied by a weight). This represents a synaptic conductance. The inputs are then summed: that is, the neuron's inputs are **linearly combined**.
3. The threshold is a constant. It represents Sodium channels positive feedback driving a spike.
4. Inhibition works in the same way than excitation: equal amounts of inhibition and excitation cancel each other out.
5. An input to the neuron looks like a static pattern.

Even though many of these simplifications are at odds with biological detail, in many cases they can compellingly be argued to be a reasonable representation of a neuron's information processing.

<!-- #endregion -->

<!-- #region id="_7aTYEFySFjg" -->
---
**A Linear Combination**

 The "weighted sum of inputs" is referred to as a **linear combination**.

> **Example:**
> - if inputs are $a_1$, $a_2$, $a_3$ (or the 'vector' $\bf{a}$)
> and weights are $w_1$, $w_2$, $w_3$ (or the 'vector' $\bf{w}$)
> a linear combination is simply
> $a_1 w_1 + a_2 w_2 + a_3 w_3$

This can be represented by matrix multiplication:

$$\bf{a}^T \bf{w}$$

or in dot product form:

$$\bf{a} \cdot \bf{w}$$

---
<!-- #endregion -->

<!-- #region id="VvdO0QDNwmYt" -->
If we are talking about a single McCulloch-Pitts artificial neuron that receives from N sources, we can frame the inputs to the neuron ($\mathbf{x}$) as a vector with N entries. The input from each source is multiplied by a weight, a number. The weight is also a vector $\mathbf{w}$ with N weights (also called 'entries'). These weights can be positive (excitatory) or negative (inhibitory). The inputs to the neuron are multiplied by the weights and summed. If the result of this operation is positive, we say the neuron is active. 

> One can think about the inputs as frequency or current of synaptic potentials.The weights representing the amplitude of the post-synaptic current (EPSC or IPSC).

These operations are conveniently represented via matrix multiplication (or equivalently, for 1D vectors, the dot product). It is important to note that **matrix multiplication multiplies a row vector by a column vector**. The output of matrix multiplication of a single column vector by a single row vector is a single number (a "scalar"). 

Thus, the activity of a neuron receiving inputs $\bf{x}$ weighted by weights $\bf{w}$, is a scalar.

---

$$
y = \mathbf{w} ^T \mathbf{x}
$$ 

---

In here, the **T** superscript indicates **transpose**, making $\bf{w}$  a column vector. **PAY CLOSE ATTENTION TO THE ORDER** in which $\mathbf{w}^T$ and $\bf{x}$ appear. Some authors use the reversed convention, in which the weight vector is a column vector following the activity vector. **These are equivalent but not identical** (To obtain identical results for networks, the adjacency matrix has to be rotated). 


Another value is sometimes added to the weighted sum of inputs, representing the *threshold* for the neuron. Equivalently, this is sometimes taken to represent *intrinsic activity*, or the *baseline firing rate* of the neuron, in which case it is called a *bias* (often represented as a $\theta$, also a scalar). Note that a *Bias* has a positive sign, and *threshold* has a negative sign.

---

$$
y = \mathbf{w} ^T \mathbf{x} - \theta
$$ 

---

<!-- #endregion -->

<!-- #region id="umWaY-scY0ly" -->
### Heaviside Step Function

<!-- #endregion -->

<!-- #region id="78uWBBiNtY-Y" -->
McCulloch-Pitts neurons either spike (1), or they do not (0) (but note that some real neurons have graded activity). Therefore, the only way to change propagation of information is changing the frequency of spikes along the axon, or firing rate. We can say that these neurons are **binary**, or **boolean**.

A threshold function that distinguishes positive from negative ( *output one if input is larger than threshold*) is often called the Heaviside step function (due to [Olivier Heaviside(https://en.wikipedia.org/wiki/Oliver_Heaviside),  a self-taught electrical engineer and mathematician who worked on circuits and logic gates).

---

$$
\begin{split}H(x) = \begin{cases} 0, & x < 0, \\ 1, &x \ge 0. \end{cases}\end{split}
$$

And so, a McCulloch-Pitts neuron with a Heaviside function can be written as

$$
\begin{split}H(\mathbf{w}^T \mathbf{x}-\theta) = \begin{cases} 0, & \mathbf{w}^T \mathbf{x} < \theta, \\ 1, & \mathbf{w}^T \mathbf{x} \ge \theta. \end{cases}\end{split}
$$

where $\theta$ is the bias of the neuron.

---



<!-- #endregion -->

<!-- #region id="trMJz4FMtmdO" -->
Thus, the (McCulloch-Pitts) artificial neuron can either fire `1` or not `0`, if the sum is larger than a threshold ($\theta$). In biological terms, the threshold represents the minimum current required for a neuron requires to regeneratively activate sodium channels.

---

$$
H(\mathbf{w} ^T \mathbf{x} - \theta )
$$ 

---
Where $H(x)$ is a the **Heaviside function**.

It is easy to define a Heaviside function:
<!-- #endregion -->

<!-- #region id="pFQSwMt4wPTl" -->

```python
def Heaviside(x):
    if x >= 0:
      y = 1
    else
      y = 0
    return y
```
<!-- #endregion -->

<!-- #region id="J4UGkvHxwsVg" -->
### Exercise: A Single Neuron

Modify the heaviside function above to behave like a single artificial neuron. Call your function ```neuron(...)``` that takes in a vector of inputs $\mathbf{x}$, a weight vector $\mathbf{w}$, and a threshold ($\theta$) and outputs `0` or `1`.
<!-- #endregion -->

<!-- #region id="pbZZqytPFB4M" -->
#### Your Solution
<!-- #endregion -->

```python id="K3tq2QU9FEIx"
# 1. To test your function define an input vector x, weigth vector w, and a threshold theta:
# w = 
# x = 
# theta = 

# 2. Define a python function called neuron that takes in weights, inputs and a threshold 
# and computes the Heaviside output of the weighted sum of inputs.
# 2.1 use linear algebra to multiply vector and matrix to avoid a for loop

# use the template below
# def neuron(x,w,theta) 
# ...
#   return y


# output = neuron(x,w,theta)

```

<!-- #region id="veoUH7N3ERSY" -->
#### Our Code
<!-- #endregion -->

```python id="alQD3cfWw-jv"
# Some random inputs and weigths to test our function below

x = random.uniform(-2,2,size=(3,1))
w = random.uniform(0,1,size=(3,1))
theta=0.5

def neuron(x,w,theta):
    # integrate... 
    cell_body_sum = w.T @ x + theta # .T is transpose, @ is matrix multiplication
    # and fire (output)

    # multiply by 1. to 'cast' result as double
    return 1. * (cell_body_sum >= 0)

output = neuron(x,w,theta)

```

<!-- #region id="gLUrB2Eo6e1R" -->
Evidently, the simplifications for this neuron are on the *heavy side* (poor pun). Biological neurons are in dynamical systems. For example, neurons often are sensitive for timing between the synaptic potentials, and dendrites perform non-linear operations such as amplifying input signals. Nevertheless, under many situations, even the most complex neurons can be thought of as performing nature's version of this yes/no operation. Case in point, receptive fields of neurons are compellingly represented by the basic operations of an articicial neuron as defined above.

<!-- #endregion -->

<!-- #region id="b4qUVRxM7JLC" -->
## The Binary Neuron as a Linear Classifier  

<!-- #endregion -->

<!-- #region id="Ea1HXlb5h5iy" -->

The simple artificial neuron described above performs the function of a **classifier**. It classifies its inputs into those that activate  it, and those that do not.

For example, a classifier could tell you whether inputs belong to the class of bald people. It could tell us whether something is a penguin or not. Or it could classify things that happen together: the boolean function **AND**. A neuron with two inputs that computes an **AND** function only says 'yes' if input one **AND** input two. It says 'no' to everything else. 

With properly tuned weights and bias, a linear classifier like our simple neuron can reproduce classifications for various [logical gates](https://en.wikipedia.org/wiki/Logic_gate) (also called logical gates), such as (AND, OR, NAND). 
<!-- #endregion -->

<!-- #region id="pjzhmjWuRDuP" -->
### An **AND** gate
<!-- #endregion -->

```python id="gyoy8L7jt_X-"
def AND(a,b): 
    if a == 1 and b == 1: 
        return True
    else: 
        return False  
AND(0,1)

```

<!-- #region id="zuiIfnnrRKP3" -->
### The **XOR** gate
<!-- #endregion -->

```python id="Z4iexZMMuRmg"
def XOR(a,b): 
    if a != b: # where "!=" means "not" 
        return 1
    else: 
        return 0
XOR(1,0)

```

<!-- #region id="fN414jrxRST5" -->
#### Ponder:

> One of the logical functions above cannot be performed by the artificial neuron as above. Can you guess which one?
<!-- #endregion -->

<!-- #region id="w4gc_aAz8i4z" -->
# FFNNs: Encoding receptive fields
<!-- #endregion -->

<!-- #region id="42Gd3uLui7oa" -->
## A Receptive Field

Neurons that selectively respond to a certain stimuli are effectively categorizing said stimuli. A neuron that responds to a certain feature of a stimulus is said to *have a receptive field* for that feature. The accepted way to think about a neuron's receptive field is as a classifier, defined above. It *classifies* inputs into those that make it fire and those that do not.

<!-- #endregion -->

<!-- #region id="YnS-RUByw7ae" -->
<img src=https://qph.fs.quoracdn.net/main-qimg-957451c779574bcb4f9222c7801fcc11.webp width="350">


<!-- #endregion -->

<!-- #region id="FNonQLiiiYEU" -->
## Exercise: Manually Tune The Weights of a Neuron
<!-- #endregion -->

<!-- #region id="Gdq28h9eN2zx" -->
The cells involved with processing visual data can be very specific with their responses. As Hubel and Wiesel discovered, in the visual cortex (V1)  some neurons only fire (action potentials) for bars at certain angles.

In this exercise, you are expected to **manually tune a neuron by selecting weights vector** (and maybe the threshold value), so that the neuron is selective for a feature, that is, it only responds to a specific stimulus. The purpose of this exercise is to build intuition about what do the weights have to look like for a neuron to respond selectively to certain patterns.

**Roadmap**:
- create vectors to representing stimuli and collect them in a "dictionary".
- choose weights for a single output neuron such that the neuron says ```1``` to the preferred stimulus and ```0``` to other stimuli.
- choose weights for more output neurons, such that each has a different stimulus preference.
<!-- #endregion -->

<!-- #region id="Et_rfJgrykze" -->
## Creating Stimuli
<!-- #endregion -->

<!-- #region id="mXB5X_9YhH2z" -->
Before we start selecting weights, we must create inputs to our network. For this exercise we will be creating stimuli that are **oriented bars** representing how light activates a tiny patch of the retina, of say, 3 x 3 rods (photoreceptors that are responsive to luminance). As you remember, the retina has a dense array of rods, but here we are considering only receptors that are really close to each other, a very small "visual angle".


Before we start the process of selecting weights, it will be useful to train our ability to create test inputs to our network. For this exercise we will be creating oriented bars that represent how light activates a small patch of the retina, of say, 3 x 3 pixels. Thus, the initial stimulus is a 3x3 square. We will be creating these stimuli as numpy arrays introduced into python's data type 'dictionary'.

<!-- #endregion -->

<!-- #region id="D3PiatH_hKMH" -->
### 2D Stimuli
<!-- #endregion -->

<!-- #region id="JKrNXNXzmjEC" -->
Using `np.array` create stimuli with zeros for dark pixels and ones for bright pixels. Create 4 stimuli for a vertical bar, a horizontal bar, and two diagonal bars (left and right slant, i.e., / and \\), all of them using three 'on' pixels centered in the patch. We are going to present these stimuli to the network, to test whether our choice of weights works.
<!-- #endregion -->

```python id="JWTD-4D_8o9Z"
# we represent light exciting a part of the visual cortex, 
# so that '1' is bright and '0' is dark.
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

```

<!-- #region id="4tvFyeq0zJJa" -->
### Transforming 2D Stimuli Into 1D Input Vectors
<!-- #endregion -->

<!-- #region id="hlFIIZHtQF8t" -->
Above we have created some matrices representing the 2D patch of retina.

To pass on the stimlus to our networks we have to transform them in a list (a vector, to perform linear combination). So we need to rearrange the elements of the stimulus matrices as **input vectors**. 

We will want to take each of the entries of our stimuli dictionary and reshape the matrices as 1D vectors. (i.e., We need to **flatten** the matrices). 
<!-- #endregion -->

```python id="7koak7uqv0m9"
# one way to reshape all matrices into vectors is to use flatten

v_v = S1.flatten()
v_h = S2.flatten()
v_dl = S3.flatten()
v_dr = S4.flatten()

print(v_v)
print(v_h)
print(v_dl)
print(v_dr)

```

<!-- #region id="Ud8OWtxKuiwE" -->
---

#### Question

- In principle, there are two ways one can flatten a matrix, either by stacking the columns, or by concatenating the rows of the arrays. What is the default operation? Can you figure out how to do it in the other way?

---
<!-- #endregion -->

<!-- #region id="IkuHIRprbUZr" -->
> #### Interlude: Dictionaries**
>
> Python works with several different datatypes; lists, dictionaries, tuples, arrays, to name a few. A **dictionary** is datatype that collects and can be indexed via 'keys' (a string for example). Dictionaries are written with curly brackets. While other compound data types have only value as an element, a dictionary has a ```"key": value ``` pair. Simply put, you can use a dictionary to retrieve 'values' that matches a given `key`.
> 
> Each key is separated from its value by a colon `(:)`, the items are separated by commas, and the whole thing is enclosed in curly braces. An empty dictionary without any items is written with just two curly braces, like this: `{}`.
> 
> Keys are unique within a dictionary while values may not be. The values of a dictionary can be of any type, but the keys must be of an immutable data type such as strings, numbers, or tuples.
> 
> ```python
numbers = {'one':1, 'two':2, 'three':3}
> ```
and so calling our  example as such
> ```python
numbers['one']
> ``` 
> returns a 1.

<!-- #endregion -->

```python id="aepnZWTNjjKK"
# Here we create a dictionary with the stimuli (so that it becomes easy to manage them)
stimuli = {'horizontal':v_h, 'vertical':v_v,'diagonal_left':v_dl, 'diagonal_right':v_dr}

# iterate through values in the dictionary
# more ways to do it here: https://realpython.com/iterate-through-dictionary-python/#iterating-through-keys-directly
for key in stimuli:
    print(key, '->', stimuli[key])

```

<!-- #region id="Iq5NGbnK0hoo" -->
## Exercise: Designing Weights of a Classifier FFNN

Here we will like to design weights and choose a bias such that the output neuron is selective for **vertically oriented bar**. That bar should be in the middle of the receptive field of the neuron (the third pattern in our code above). The neuron SHOULD NOT FIRE to any of the other oriented stimuli defined above.

- Define the weights (and threshold) for an artificial neuron that returns `1` for the vertical bar and `0` for all other bars.
- Use your neuron function defined earlier to test your weights.
- Call the input from the dictionary defined above.
- Verify that none of the other input patters activate your neuron.
<!-- #endregion -->

<!-- #region id="nachlSJBReiu" -->
### Your Code
<!-- #endregion -->

<!-- #region id="qkIiEqHyR9Qr" -->
*You can use this template to solve the exercise.*
<!-- #endregion -->

```python id="tj9rh01sTI-X"
# CREATE the weigth vector for your classifier
w_vertical_neuron = 

# RETRIEVE the vertical stimulus from dictionary
vertical_stimulus = stimuli['vertical']

# CALL the function with your stimulus and chosen weights
test1 = neuron(w_vertical_neuron, vertical_stimulus, threshold)

# TEST the function for every orientation and check that it responds adequatly
# test1 = 
# test2 = 
# test3 = 


# if test1 is True and all others are false, congratulations! Else
# go back to the weight design phase. 

```

<!-- #region id="mgzixsUUQFV2" -->
### Our Code
<!-- #endregion -->

```python id="vJe9ZUCzRipG"
# substitute your weigth vector in place of USERS_WEIGHT_VECTOR
w_vertical_neuron = np.array([1,0,0,1,1,1,1,0,0])

#get the orientated stimulus from dictionary
vertical_stimulus = stimuli['vertical']

# define the neuron function (linear combination + heaviside function)
def orientation_detector_neuron(weight, stimulus):
  # retrieve the value of the stimulus from the dictionary
  bias=2;
  # then return
  return int((stimulus @ weight) - bias >= 0)

# CALL the function with your stimulus and chosen weights
test1 = orientation_detector_neuron(w_vertical_neuron, vertical_stimulus)

# TEST the function for every orientation and check that it responds adequatly

test2 = orientation_detector_neuron(w_vertical_neuron,stimuli['horizontal'])
test3 = orientation_detector_neuron(w_vertical_neuron,stimuli['diagonal_left'])
test4 = orientation_detector_neuron(w_vertical_neuron,stimuli['diagonal_right'])

all_tests = [test1, test2, test3, test4]
print(all_tests)

```

<!-- #region id="3y2DmvPDuiwH" -->
*If test1 is True and all others are false, congratulations! Else go back to the weight design phase. =D*


<!-- #endregion -->

<!-- #region id="iehxkj0SuiwH" -->
---
#### Question:

- In our inputs we have used zeros to represent darkness and ones to represent brightness. This pattern has high contrast! Do your chosen weigths still correctly classify stimuli if the differences between bright and dark are smaller (for instance, 0.3 and 0.7)? 
- How can you change the input weights or threshold to solve a problem as above?

---
<!-- #endregion -->

<!-- #region id="jlyT58oz1GpQ" -->
## Feed Forward Network of Classifiers
<!-- #endregion -->

<!-- #region id="e5t5HgbsvpXm" -->
Currently our neural network has 9 inputs and 1 output neuron that recognizes a vertical bar in its receptive fields. We can easily extend the network to include more output neurons, each with a different orientation preference.

To add output neurons, we need to add more weight vectors, one weight vector for each neuron selecting for a different orientation. *When we concatenate weight vectors together this results in a matrix*. 

We can use matrix multiplication in the same way as defined above to multiply a weight matrix by an input. 

$$\bf{y} = \bf{W}^T \bf{x} $$. 

The output now is a vector, where each value is the result of the linear combination from each weight vector. To compute the output of the network of classifiers one simply uses the heaviside function on the vector $\bf{y}$
<!-- #endregion -->

<!-- #region id="Qxr8CzBivq9v" -->
### Exercise: Create a Classifier Network

Extend your network to four output neurons, and check if the response to all stimuli is as you expect.

- create weights vectors for each output neuron
- concatenate them into a matrix
- modify your neuron function to accept weigth matrices instead of weight vectors.
- test your network outputs for each of the stimuli defined above.
<!-- #endregion -->

<!-- #region id="YO46NyhuWrEU" -->
### Your Code
<!-- #endregion -->

<!-- #region id="GSL3I2YvuiwI" -->
*Use the template below to do the exercise*
<!-- #endregion -->

```python id="G5b28c1EWs5N"
# DEFINE the weight matrices (as np.arrays) for different output neurons
# w_vertical = ...
# w_horizontal = ...
# w_diagonal_L = ...
# w_diagonal_R = ...

# SET an appropriate threshold
# theta = 


# we use concatenation to bind the weight vectors into a weight matrix
# note: the network has 9 inputs and 4 output neurons
## https://stackoverflow.com/questions/20978757/how-to-append-a-vector-to-a-matrix-in-python
## to concatenate vectors: 
# weight_matrix = c_[ vector1, vector2, vector3, vector4]

# TEST your resulting matrix to make sure it has the expected dimensions.
# To check, you can use 
# weight_matrix.shape



# define feed forward network function
def FFNN(W, I, theta):
  # If W is the weigth matrix and
  # I is the input vector
  # their dot product gives the activity of 
  # all of the units
  output = ...
  return output

# test your network for the different stimuli
# test1 = ... 
# test2 =...

```

```python colab={"base_uri": "https://localhost:8080/", "height": 134} id="tvuub-F6QlLd" executionInfo={"status": "error", "timestamp": 1685611262724, "user_tz": -120, "elapsed": 10, "user": {"displayName": "Mario Negrello", "userId": "10136788594790905986"}} outputId="8354de5c-20c9-42ed-806b-327cb34e29bf"
# DEFINE the weight vectors (as np.arrays) for different output neurons
w_vertical = 
w_horizontal =
w_diagonal_L = 
w_diagonal_R = 

# DEFINE a threshold
theta = 

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

```

<!-- #region id="wNnLHCxu1VYf" -->
## Complex Receptive Fields
<!-- #endregion -->

<!-- #region id="FzNcZxlCuiwJ" -->
You have designed and implemented a single layer feed forward neural network. We can attempt to extend the network with further neurons. 

As a test to your weight designer skills, add an output neuron with the following output characteristics:

- output 1 to a vertical bar
- output 1 to a horizontal bar
- output a 0 to a cross

First add the 'cross' stimulus to the dictionary, and solve the problem as usual.
<!-- #endregion -->

```python id="FIq3t86juiwJ"
cross = array([[0, 1, 0],[1, 1, 1],[0, 1, 0]])
plt.imshow(cross)

# add cross to dictionary

# test the network with the new neuron on horizontal, vertical and cross stimuli.

```

<!-- #region id="cTbPyKWFuiwK" -->
----
#### Question:

- Is it possible to find weights that solve the problem defined as above? Explain your answer.
----
<!-- #endregion -->

<!-- #region id="tzDO6ChWuiwK" -->
### Confusion Matrix
<!-- #endregion -->

<!-- #region id="wbfGs_7-GfgS" -->

A **confusion matrix** is a way to see how well a network is able to classify stimuli. Confusion matrices are the basis for a commonly used 'loss function' (more about this later), the so called cross entropy loss.

Confusion matrices are used to test the performance of a network that classifies stimulus (a 'model'). Confusion matrices count the number of times that a certain stimulus was assigned to a given category. That is, they display the label given by the network as a function of the true label. Note: the confusion matrix can only be used if one has 'true labels' for every single stimulus being presented, that is, if the dataset is **supervised**. 

Below is an example of a [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix) for image classification, that tries to distinguish Dogs from Rabbits and Cats. In the case of this network, we have 'trained it' to try and see if a picture is of a cat or a dog. 

In the graph below, how many times did the network clasify a cat as a dog? What is the probability of assigning 'cat' to 'dog'? That is, what is the probability that P(label==cat|image==dog)? Below, cat was presented 3 times, and dog came out all of the time, while dog and rabbit were always labelled correctly. Is this good or bad?

![](https://i.stack.imgur.com/Rz5ol.jpg)

<!-- another, more standard

![](https://datatofish.com/wp-content/uploads/2018/12/003_cm.png)
 -->

<!-- #endregion -->

<!-- #region id="igQi6WNpGpAP" -->
---
#### Questions:
- According to the confusion matrix above, what is the most common mistake of the classifier?
- How many rows and columns will be in a confusion matrix that measures classification accuracy of 10 stimuli for 4 categories?

---
<!-- #endregion -->

<!-- #region id="cvZtdzKwwbCo" -->
# Filters and Convolutions





<!-- #endregion -->

<!-- #region id="92vi9-duAqtU" -->

In biology, a neuron is said to have a **receptive field** for some **feature** if the neuron fires preferentially when that features is presented to the sensory input. In the case of sensory receptive fields, neurons can select for auditory features (specific frequencies), patches of your skin surface, specific odorants, and so on. This type of selectivity can be represented by feed forward neural networks with a few layers. 

The term receptive field is also used in the context of artificial neural networks, where it refers to a the region in the input space that affects a particular node in the network. Receptive field like **convolutional filters** are used to preprocess a large image by decomposing that image in multiple features. Our inputs in the previous exercise were simply 3x3 arrays. But if one has a larger picture (say 15x15), one can look at a few patches at a time. The patch of neurons that looks at a particular place in the picture, is said to have the 'receptive field' for that location. This type of model architecture, which attempts to mimick the hierarchical way visual processing (and pre-processing in other senses), is called **convolutional neural network**. 

In a convolutional neural network, the first layers are made of arrays of feature detectors. Multiple feature detector neurons such as the ones you have developed analyze the entire picture patch by patch. That is, each neuron in the convolutional layer receives data from only a *local patch* of inputs. Neurons in one convolutional layer will perform the same kind of operation, that is, they will extract the same kind of feature, for example, a convolutional layer can detect the 'horizontal' feature, but inspecting the entire picture and being active in those locations where horizontal layers are organized.


<div>
<img src=http://drive.google.com/uc?export=view&id=1UEDgsNIcP3DtfFksgvVHrlrbkwCdj0mW width="250">
</div>

This is a neuron with a receptive field of 5x5 pixels for example. 


<img src=https://miro.medium.com/max/3726/1*wqZ0Q4mBaHKjqWx45GPIow.gif width="500"> left: input pixels, middle: filter, right: output (feature map)

What we do: each filter looks at a small patch of input pixels and computes the dot product between the filter and inputs. An activation map for the given filter is produced as result.

<!-- #endregion -->

<!-- #region id="cbc8B1nZn_hn" -->
This excerpt from the "Principles of Neuroscience (5th Ed.)" demonstrates the notion of composition of receptive fields, deriving from Hubel and Wiesel's model of simple and complex cells of the visual cortex.

[![Hubel and Wiesel Model](https://i.postimg.cc/ncgy9Vcj/image.png)](https://postimg.cc/QHg4R37h)
(from "Principles of Neuroscience, 6th Ed. Appendix "Neural Networks")
<!-- #endregion -->

<!-- #region id="tgJA4tMun2CG" -->
# Research Questions
<!-- #endregion -->

<!-- #region id="_4GTQ7Mxn32e" -->
--- 
- why is the perceptron a 'linear' classifier'?
- when can we say that a neuron has a receptive field?
- why must the weights of a classifier 'resemble' their preferred stimulus?
- when does a FFNN fail to distinguish between different inputs?

---
<!-- #endregion -->

<!-- #region id="ooA9TxCC0qSV" -->
# Resources
- [3Blue1Brown must see playlist on Neural Networks](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiG_M-fm4nsAhVMCewKHQ4iD00QyCkwAHoECAQQAw&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DaircAruvnKk&usg=AOvVaw3c_pmQ67XtWSaAXtAgCxkl)
- [A medium post on representing networks via matrices](https://medium.com/coinmonks/representing-neural-network-with-vectors-and-matrices-c6b0e64db9fb)
- Convolutions, aka, [Morphological operations, in python](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html)
- distill.pub has a series on interpretable deep learning which has insights about the responses of complex receptive fields [Interpretability](https://distill.pub/2018/building-blocks/)
- http://neuralnetworksanddeeplearning.com/chap6.html
<!-- #endregion -->

<!-- #region id="WXQQGMBrPibp" -->
# License

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

Mario Negrello, Daphne Cornelise, Elias Santoro. Reviewing and testing by a host of students.
<!-- #endregion -->
