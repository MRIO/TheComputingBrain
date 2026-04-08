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
    name: python3
---

<!-- #region id="PNDl95MwKj8c" -->
# A minimalistic Echo State Networks demo

For the entire story see: http://www.scholarpedia.org/article/Echo_state_network

Based on code by  Mantas
http://mantas.info
<!-- #endregion -->

<!-- #region id="YVR3YhT31ZwJ" -->
# Initialization
<!-- #endregion -->

```python id="UzJDZOzazTrM" executionInfo={"status": "ok", "timestamp": 1730196306970, "user_tz": -60, "elapsed": 432, "user": {"displayName": "Mario Negrello", "userId": "10136788594790905986"}}

from numpy import *
from matplotlib.pyplot import *
import scipy.linalg

```

<!-- #region id="4XN4OG-V1cSo" -->
# Prepare Data
<!-- #endregion -->

```python id="YBe58dJHFAMv" colab={"base_uri": "https://localhost:8080/", "height": 489} executionInfo={"status": "ok", "timestamp": 1730196309723, "user_tz": -60, "elapsed": 2132, "user": {"displayName": "Mario Negrello", "userId": "10136788594790905986"}} outputId="94225db1-6452-418a-c329-52603bb735cf"
# training data
trainLen = 2000
testLen = 2000
initLen = 100

# create some data (that we want the network to produce)
data = sin(linspace(0,399*pi,5000)) + sin(linspace(0,203*pi,5000))

# plot some of it
figure()
plot(data[0:1000])
title('A sample of data')

```

<!-- #region id="2poIq_9q1fUu" -->
# Make Reservoir
<!-- #endregion -->

```python id="itWS56Z7zOAS" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1730196309723, "user_tz": -60, "elapsed": 11, "user": {"displayName": "Mario Negrello", "userId": "10136788594790905986"}} outputId="9e88d671-a932-4907-960e-c6ef4bc4da50"
# Size of Input and Output Layers
inSize = outSize = 1

# generate the ESN reservoir

# reservoir size (number of neurons)
resSize = 100

# leaking rate (mixing)
a = 0.3

random.seed(42)

Win = (random.rand(resSize,1+inSize)-0.5) * 1
W = random.rand(resSize,resSize)-0.5

# normalizing and setting spectral radius (correct, slow):
print('Computing spectral radius...'),
rhoW = max(abs(linalg.eig(W)[0]))

print('spectral radius is '+str(rhoW))

W *= 1.25 / rhoW
```

<!-- #region id="ntAyLwyd1nI-" -->
# Run the Network
<!-- #endregion -->

```python id="GsJzLWPuzZqk" executionInfo={"status": "ok", "timestamp": 1730196309723, "user_tz": -60, "elapsed": 9, "user": {"displayName": "Mario Negrello", "userId": "10136788594790905986"}}
Yt = data[None,initLen+1:trainLen+1]
```

```python id="kDVyw6Zg0CLy" colab={"base_uri": "https://localhost:8080/", "height": 387} executionInfo={"status": "ok", "timestamp": 1730196319143, "user_tz": -60, "elapsed": 9429, "user": {"displayName": "Mario Negrello", "userId": "10136788594790905986"}} outputId="cab2a10a-3a42-49dc-a530-56bf322f95f0"
# we give the network a constant input with a single neuron:
input_0 = 1

# allocated memory for the design (collected states) matrix
X = zeros((1+inSize+resSize,trainLen-initLen))

# run the reservoir with the data (input) and collect output (X)

# initialize reservoir state
x = zeros((resSize,1))

for t in range(trainLen):
    u = data[t]
    # x = tanh( dot( Win, vstack((input_0,u)) ) + dot( W, x ) )

    # with a leaky neuron
    x = (1-a)*x + a*tanh( dot( Win, vstack((input_0,u)) ) + dot( W, x ) )

    if t >= initLen:
        X[:,t-initLen] = vstack((input_0,u,x))[:,0]

# (this will also be the corresponding target matrix )

figure(figsize=(15,4))
pcolor(X)
xlabel('time (steps)')
ylabel('neurons')
```

<!-- #region id="2XIIc3oB2jxa" -->
# Train the Network
<!-- #endregion -->

```python id="-oMktrKt2mMP" colab={"base_uri": "https://localhost:8080/", "height": 713} executionInfo={"status": "ok", "timestamp": 1730196319650, "user_tz": -60, "elapsed": 514, "user": {"displayName": "Mario Negrello", "userId": "10136788594790905986"}} outputId="292048ae-ef5b-4d02-dae4-df2fb3818a26"
# train the output by ridge regression (a more robust variant of linear regression)
# for the usual regression simply set reg to 0.
reg = 1e-8  # regularization coefficient
X_T = X.T

# one line learning:
Wout = dot( dot(Yt,X_T), linalg.inv( dot(X,X_T) + \
    reg*eye(outSize+inSize+resSize) ) )


figure(2, figsize=(1,8))
pcolor(Wout.T)
colorbar()

```

<!-- #region id="J-_wbJbHpe_u" -->
# Regenerate the Data
<!-- #endregion -->

```python id="nwno3IYd1DrT" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1730196320957, "user_tz": -60, "elapsed": 1311, "user": {"displayName": "Mario Negrello", "userId": "10136788594790905986"}} outputId="9351c08e-077b-4d1b-e894-5c6ec937398c"
# run the trained ESN in a generative mode. no need to initialize here,
# because x was initialized with training data and we continue from there.

Y = zeros((outSize,testLen)) #allocate output

u = data[trainLen]
for t in range(testLen):
    x = (1-a)*x + a*tanh( dot( Win, vstack((input_0,u)) ) + dot( W, x ) )
    y = dot( Wout, vstack((input_0,u,x)) )
    Y[:,t] = y
    # generative mode:
    u = y
    ## this would be a predictive mode:
    # u = data[trainLen+t+1]

# compute MSE for the first errorLen time steps
errorLen = 500
mse = sum( square( data[trainLen+1:trainLen+errorLen+1] -
    Y[0,0:errorLen] ) ) / errorLen
print('Mean Square Error (MSE) = ' + str( mse ))


```

```python id="nOjC9wMaFjc4" colab={"base_uri": "https://localhost:8080/", "height": 1000} executionInfo={"status": "ok", "timestamp": 1730196334139, "user_tz": -60, "elapsed": 13188, "user": {"displayName": "Mario Negrello", "userId": "10136788594790905986"}} outputId="7d4be513-a16a-48de-f0b3-e0249db34c47"
# plot signals
figure(1, figsize=(15,4))

plot( Y.T, 'b' )
plot( data[trainLen+1:trainLen+testLen+1], 'g' )
title('Target and generated signals $y(n)$ starting at $n=0$')
legend(['Target signal', 'Free-running predicted signal'])

figure(2, figsize=(15,8))
pcolor( X[0:100,0:2000])
title('Some reservoir activations $\mathbf{x}(n)$')

# figure(3).clear()
# bar( range(1+inSize+resSize), Wout.T )
# title('Output weights $\mathbf{W}^{out}$')

```

```python id="ycdve7DBGELP" executionInfo={"status": "ok", "timestamp": 1730196334139, "user_tz": -60, "elapsed": 6, "user": {"displayName": "Mario Negrello", "userId": "10136788594790905986"}}

```
