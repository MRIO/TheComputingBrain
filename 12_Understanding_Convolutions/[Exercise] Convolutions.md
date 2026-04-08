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

<!-- #region id="pBCQZs336Aum" -->
# Exercises: Convolutional Networks



<!-- #endregion -->

<!-- #region id="XukOrTNR6UZ-" -->
# Introduction
<!-- #endregion -->

<!-- #region id="ErNyp9S_6aTX" -->
This project will give a short introduction to the main operation in **Convolutional** Neural Networks, namely 'convolutions'. These networks are designed to mimick the hierarchy of pre-processing of the human visual system,aka, visual feature extraction. The project starts with a short introduction in the use of the convolution operation in image processing. Then we will use this to design a small CNN that performs a basic analysis task. <!--[does some basic image analysis task]-->
<!-- #endregion -->

```python id="HRRzeIZD5-ON"
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from skimage import io
from skimage.color import rgb2gray
import PIL
```

<!-- #region id="OVWLbBVeVqXo" -->
## Learning Goals

- Understand what are 'kernels'
- Use python to compute image convolutions
- Combine the output of multiple convolutional layers
<!-- #endregion -->

<!-- #region id="DE9bZuG8-UoM" -->
# Convolutions in image analysis
<!-- #endregion -->

<!-- #region id="9Ejxim4v-XgS" -->
This first part will give a quick recap of the use of convolutions in image analysis. Numpy supplies a useful function _numpy.convolve2d(F, h, mode)_ to perform the convolution operation on matrices. In the piece of code below, the input image _F_ is defined. Design a 3x3 moving average filter _h_ and perform a convolution between the image and the filter to obtain the output image _G_.
<!-- #endregion -->

<!-- #region id="1pfp4fzE4flu" -->
## Exercise 1
Use the command G = signal.convolve(F,h,mode='same') to obtain output image G from input image F and convolution filter h.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 198} id="NEoqnUmO4C40" executionInfo={"status": "ok", "timestamp": 1633813978459, "user_tz": -120, "elapsed": 513, "user": {"displayName": "Sem", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gh09Oq4BAnxliBoc2t1mIaZPRnFOc98uUmJRwqMwA=s64", "userId": "11116053297538602196"}} outputId="90332928-25ff-463b-b38d-bc3e3e934155"
# Initialize image G
F = np.array([[0,   0,   0,   0,   0,   0,   0, 0],
              [0,   0, 255, 255, 255,   0,   0, 0],
              [0, 255, 255, 255, 255, 255,   0, 0],
              [0, 255, 255, 255, 255, 255, 255, 0],
              [0,   0, 255, 255, 255, 255, 255, 0],
              [0,   0,   0,   0, 255, 255,   0, 0],
              [0,   0,   0,   0, 255,   0,   0, 0]])

# Initialize the moving average filter h
h = np.array([[1/9, 1/9, 1/9],
              [1/9, 1/9, 1/9],
              [1/9, 1/9, 1/9]])


# Perform convolution to obtain image G
           # <-- your line goes here

# Print the input and output image 
fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(F,cmap='gray')

fig.add_subplot(1,2,2)
plt.imshow(G,cmap='gray')



```

<!-- #region id="INpayjyCUQMY" -->
**Exercise:** Review the kernels and interpret the resulting images after convolution.
<!-- #endregion -->

<!-- #region id="asJe7-3J-725" -->
# Convolutional layer 1: edge enhancement
<!-- #endregion -->

<!-- #region id="D0dRtWbg-_7X" -->
As a next step we will apply a convolutional filter onto a larger image (of a window). We try to combine the output of multiple convolutional layers until we can 'recognize it' as a rectangle. 

The first layer of our CNN will simulate lateral inhibition in the retina. Lateral inhibition promotes regions with most contrast, and so it is used to enhance the edges in an image.

<img src="https://i.imgur.com/VKfFYeP.png"  width=30%>
<!-- #endregion -->

<!-- #region id="VC8B8xgY5DmM" -->
## Exercise 2
Create the output matrix of layer 1 _out1_ by performing a convolution between the input image and the convolution matrix. Use mode 'same' for the convolution.
<!-- #endregion -->

```python id="laM7UMmt4nWi" colab={"base_uri": "https://localhost:8080/", "height": 211} executionInfo={"status": "ok", "timestamp": 1633725492006, "user_tz": -120, "elapsed": 920, "user": {"displayName": "Sem", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gh09Oq4BAnxliBoc2t1mIaZPRnFOc98uUmJRwqMwA=s64", "userId": "11116053297538602196"}} outputId="b8e3e1ae-7dbe-4f8e-9481-41c529c17fdb"
# Define location of images:
window = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRGNsVqcbxxYewzE_kYu9tPTPP7fdX3vXXl1Q&usqp=CAU"
skyline = "https://www.behangwebshop.nl/media/catalog/product/cache/1/image/9df78eab33525d08d6e5fb8d27136e95/0/5/051-print.jpg" 
car = "https://cdn.euroncap.com/media/62863/genesis-g80.png?mode=crop&width=308&height=204"


# Start with loading the figure 
input = rgb2gray(io.imread(window))

h1 = np.array([[  0,  -.1,   0],
              [-.1,    1, -.1],
              [  0,  -.1,   0]])

# Perform convolution to create output of layer 1
            # <-- here goes your line

# Plot the input and output
fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(input,cmap='gray')

fig.add_subplot(1,2,2)
plt.imshow(out1,cmap='gray')
```

<!-- #region id="GwsB8jw2UmaT" -->
# Convolutional layer 2: horizontal and vertical edges
<!-- #endregion -->

<!-- #region id="8CZEJAa-Ut53" -->
In this layer, we will use a different convolution matrix to enhance the horizontal and vertical edges of the window. As you can imagine, there are many more features that can be analyzed in this layer (color, contrast, different edge orientations). We will focus on two subsets of the next convolutional layer, namely that for horizontal edges and that for vertical edges. These neurons will create a grid in which the edges of interest pop out of the picture. 

<img src="https://i.imgur.com/7RO1ms1.png"  width=30%>
<!-- #endregion -->

<!-- #region id="mDQTPanw5x9l" -->
## Exercise 3
Perform two convolutions:


*   Obtain the output for the horizontal edges, _out2_horizontal_ by performing convolution between the output of layer 1 and the horizontal filter.
*   Do the same for the vertical edges, _out2_vertical_.


<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 161} id="Npm_Fu4MUuZf" executionInfo={"status": "ok", "timestamp": 1633725498568, "user_tz": -120, "elapsed": 436, "user": {"displayName": "Sem", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gh09Oq4BAnxliBoc2t1mIaZPRnFOc98uUmJRwqMwA=s64", "userId": "11116053297538602196"}} outputId="51f2ea5d-75a0-4b67-f339-f90399d4e0c0"
# Define convolution layers 
h2_horizontal = np.array([[-1, -1, -1],
                          [ 2,  2,  2],
                          [-1, -1, -1]])
h2_vertical = np.array([[-1,  2, -1],
                        [-1,  2, -1],
                        [-1,  2, -1]])

# Create two output images containing the horizontal and vertical edges
out2_horizontal = signal.convolve2d(out1,h2_horizontal,mode='same')
out2_vertical = signal.convolve2d(out1,h2_vertical,mode='same')       # <-- cut out for exercises

# Plot the input and output images below
fig = plt.figure()
fig.add_subplot(1,3,1)
plt.imshow(out1,cmap='gray')
fig.add_subplot(1,3,2)
plt.imshow(out2_horizontal,cmap='gray')
fig.add_subplot(1,3,3)
plt.imshow(out2_vertical,cmap='gray')

# Note: you may need to zoom in in order to see the horizontal and vertical edges
```

<!-- #region id="7TnTr2swWN4U" -->
# Convolution layer 3: finding the horizontal and vertical edges
<!-- #endregion -->

<!-- #region id="djwQtTXqWVMU" -->
The third layer of the network will detect whether the borders of the window are present in the picture. Please note that we require a lot more neurons in this layer in order to detect lines in different orientations, as there are many different lengths, angles and positions imaginable as network input. Real biological and artificial neural networks use an automated learning method to train these large groups of neurons to recognize a variety of features. 

We will highlight only a handful of neurons in these last two steps so that you can see how these signals progress through the CNN. Keep in mind that the behaviour of these higher neurons is more complicated in practice because they are accompanied by many more neurons and more layers.

From the third layer, we will highlight 4 neurons, which will each detect one of the edges of the window. We will take the sum of all neurons on the window's edges, effectively creating a 1*n convolution matrix with all 1s that overlaps the edges of the window. 


<img src="https://i.imgur.com/fr3ERVS.png"  width=50%>
<!-- #endregion -->

<!-- #region id="QKEkUzbg7nZL" -->
## Exercise 4
The neurons of interest each fire when the their horizontal and vertical edges of interest fire. The locations of these edges have been preselected. Execute the code below to see the value of these four neurons.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="51gOvrEUWNQm" executionInfo={"status": "ok", "timestamp": 1633725501406, "user_tz": -120, "elapsed": 248, "user": {"displayName": "Sem", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gh09Oq4BAnxliBoc2t1mIaZPRnFOc98uUmJRwqMwA=s64", "userId": "11116053297538602196"}} outputId="6ba6dd5c-d05e-452f-d12b-09cb24f366c9"
# The neurons depend on the edge detection neurons. Exact locations of the 
# neurons that fire much for the detected edge are already filled in.
neuron1 = sum(out2_horizontal[34,52:178])
neuron2 = sum(out2_vertical[38:175,47])
neuron3 = sum(out2_horizontal[166,52:178])
neuron4 = sum(out2_vertical[38:175,170])

print(neuron1)
print(neuron2)
print(neuron3)
print(neuron4)


```

<!-- #region id="gCwuq_oTmqHE" -->
# Layer 4: the 'grandmother cell'
<!-- #endregion -->

<!-- #region id="YCuJ4sItm5NA" -->
Lastly, we merge the value of the previous layer's 4 neurons together to obtain the value of the grandmother cell. 


<img src="https://i.imgur.com/ja5kKGs.png"  width=50%>
<!-- #endregion -->

<!-- #region id="lGSnYj3b8riH" -->
## Exercise 5
Find the value of the grandmother cell _windowGrandmother_ by adding up the values of the neurons that belong to the underlying edges. 

<!--[windowGrandmother = neuron1 + neuron2 + neuron3 + neuron4]-->
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="vXXhyZwLnWb2" executionInfo={"status": "ok", "timestamp": 1633725504010, "user_tz": -120, "elapsed": 331, "user": {"displayName": "Sem", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gh09Oq4BAnxliBoc2t1mIaZPRnFOc98uUmJRwqMwA=s64", "userId": "11116053297538602196"}} outputId="39a63b73-d5fd-403e-c3ce-1af306c10e9b"
# merge the neurons on layer 3 together
windowGrandmother = neuron1 + neuron2 + neuron3 + neuron4     # <-- cut out for exercise

print(windowGrandmother)
```

<!-- #region id="VJWUeuLBnnOL" -->
The grandmother cell now has a high intensity when it observes this window picture. To put the value of the grandmother cell into perspective, try loading different images (skyline, car) and running the CNN again. You will notice the output values for these inputs are much lower. If this network had learned to recognize cars and skylines, it may have had a high value for the car and skyline grandmother cells. In return, the skyline and car grandmother cells will be low when the window image is shown to them.

Notice how we used this network to turn a matrix of neurons with a value into more abstract concepts like lines and squares. Can you imagine how these networks are used to extract features from their input? 


<!-- #endregion -->

<!-- #region id="891xh93w9LVa" -->
# Concluding remarks
In this project, you have seen how the operation of convolution can be used to extract features from input patterns. In a convolutional neural network (CNN) the early layers perform convolutional operations for the feature it promotes, which are then integrated as *evidence* for the deeper layers. For instance, a network that we built can conclude that there is a specific rectangle present in this network's input, based on the corresponding four underlying edges. 

In reality, this works both ways. A network layer draws conclusions from multiple entries in the underlying layer, but it also supplies information to multiple neurons in the next layer. For instance, the edges used to recognize the rectangle can also be used to recognize triangles and other polygons. To organize this, the neurons are trained using an automated learning strategy. Think of supervised learning in artificial neural networks as well as neuromodulators in biological brains. The weights of the convolution matrixes (synapses) are configured so that the network produces their desired outcome as much as possible. The art of designing these networks rather relies on the training method, and the architecture of the network is for the most part a black box. 
<!-- #endregion -->
