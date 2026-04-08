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

<!-- #region id="B287VpGfYWG9" -->
> # To be able to edit and use this Notebook:
> 0. Learn about how to use google colaboratory [video]()
> 1. in the file menu (top left), click ```open in playground```
> 3. still in the file menu, click ```save copy in drive```, to make your own > personalized and editable copy of this file.
> 4. edit as you like. If something breaks irreparably, go back to step 1.
<!-- #endregion -->

<!-- #region id="qEXuk5UHfBHH" -->
# Introduction
<!-- #endregion -->

<!-- #region id="_MHZdtrTfC8u" -->
As you've learned [here](https://drive.google.com/file/d/1Hh4Byl-Lic7K-HLyliaTv4Sg07zKSc2A/view?usp=sharing) this is a 'jupyter notebook'. It mixes formated text and code. The programming language in this case is python. Google kindly provides a 'server' for you to play with a pre-installed python without the complications of installing python in your own machine called 'a collaboratory'.

[Here is an pithy introduction to google collaboratory by google](https://colab.research.google.com/notebooks/welcome.ipynb)

In this 'notebook' we will be reviewing some of the most basic concepts of the python programming language including variables, control loops and functions.



<!-- #endregion -->

<!-- #region id="uLNoxRIJ386x" -->
# Python for illiterates
<!-- #endregion -->

<!-- #region id="nGMve45F4DfQ" -->
In the beginning it was the constant
<!-- #endregion -->

```python id="jRnUqyqA370O"
a = 5
```

```python id="z4wWsDDz8MjX"

```

```python id="WZE3max74HqR" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629287711324, "user_tz": -120, "elapsed": 358, "user": {"displayName": "Kate", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ghe8UithqL0uAE6up9XspRl54AKAIidQfgG5XRbMJw=s64", "userId": "04668969765414769394"}} outputId="d74465f5-61aa-429e-d3db-13361aa2e7b3"
print(a)

```

<!-- #region id="sqLOZapsUpMc" -->
# **Variables**
<!-- #endregion -->

<!-- #region id="WNl4pScmMbVA" -->
But first, let us talk about variables. 

**Variables** act as containers where we temporarily store information.
<!-- #endregion -->

```python id="FmlWXv2ziTUV"
score = 10
```

<!-- #region id="qkA6hRbLOd0V" -->
We have now created a **memory location** with a label "**score**" and we have attached value **10** to it.
<!-- #endregion -->

<!-- #region id="PCJvCwLaPPDe" -->
We can reset the value of the variable at any later point in programming. 
<!-- #endregion -->

```python id="1NCN7jDsPGji"
score = 12
```

```python id="gm9MjFJNPONE" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629287743674, "user_tz": -120, "elapsed": 15, "user": {"displayName": "Kate", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ghe8UithqL0uAE6up9XspRl54AKAIidQfgG5XRbMJw=s64", "userId": "04668969765414769394"}} outputId="38616f65-3e33-4383-95ac-8b3f2ad4d905"
print(score)
```

<!-- #region id="VNy2dBGQQwdi" -->
Variables can be assigned different data types: integers, floats, strings, and booleans.
<!-- #endregion -->

<!-- #region id="UPC09AkMRrTk" -->
An **integer** is a whole number without a decimal point. 


A **float** is a number with a decimal point. 

<!-- #endregion -->

```python id="0aCNIGZ2O84A"
weight = 10.2
```

<!-- #region id="9bYVrSULTCdG" -->
A **string** is an array of characters and is defined by putting character(s) in either single  or double quotes.
<!-- #endregion -->

```python id="Y-9csBlKThzX" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630427743942, "user_tz": -120, "elapsed": 9, "user": {"displayName": "Alessandro Jingberger", "photoUrl": "", "userId": "10099533819200777556"}} outputId="22c5f64f-8a0f-4a6e-8841-f1c7734b478a"
name = 'A very basic python intro'
print(name)
```

<!-- #region id="smRF0YjcTlgX" -->
*A* **boolean** is a binary variable that can be either false or true (0,1). 
<!-- #endregion -->

```python id="czh0uJHYT_lH"
is_on = True
is_complete = False
```

<!-- #region id="gpaccIrlUOgN" -->
*It is important to note that Python is a case sensitive language. For example, when defining variables, we must use lowercase letters "name", however, when assigning values of true/false, we must use uppercase values: "True" and "False".*
<!-- #endregion -->

<!-- #region id="IIydwax1Vj1q" -->
***Exercise N1***
You are writing a program for Erasmus MC. For this program you have to define three variables: **name**, **age**, and **sufficient amount of credits** (yes, no) for each student. Sufficient amount of credits for each student are 180 cts.

*Maria is 24 and is enrolled in the course.*

**Task**: Define the three variables for Maria making sure that age is an integer, enrolled is a boolean and name is a string.
<!-- #endregion -->

```python id="CY-yfcGmWx0N"
name = 'Maria'
age = 24
enrolled = True
```

```python id="5QYqs5pegyIN" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629287991001, "user_tz": -120, "elapsed": 305, "user": {"displayName": "Kate", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ghe8UithqL0uAE6up9XspRl54AKAIidQfgG5XRbMJw=s64", "userId": "04668969765414769394"}} outputId="f087340f-2f68-4abf-c851-c7d753c48750"
# Here we check that the types are correct with if conditions (more about it below)

if isinstance(name, str):
  print('name is a string')
else:
  print('name is not a string. Is it between quotes?')

if isinstance(age, int):
  print('age is an integer')
else:
  print('age is not an integer. ')

if isinstance(enrolled, bool):
  print('enrolled is a boolean')
else:
  print('enrolled is not a boolean')
```

<!-- #region id="UMnQnB9EXjcI" -->
# **Type Conversion**
We can convert one type of variable into another.

A string can be converted into an integer or a float. An integer can be coverted into a float or a string and so on.

To convert into an integer we use a function **int()**; to convert into a float we use **float()**; to convert into a string, we use **str()**; 
<!-- #endregion -->

```python id="8diPWsIxpNtF" colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"status": "ok", "timestamp": 1577354364577, "user_tz": -60, "elapsed": 772, "user": {"displayName": "Natia Shamugia", "photoUrl": "https://lh3.googleusercontent.com/a-/AAuE7mDa1i61TP6_bth5HMNXGTBjN9pbpJK8rveVpPeNEw=s64", "userId": "11799764980580446930"}} outputId="de2a0f04-7bf6-4a7b-dce2-e0a513e2eb3d"
answer = str(2)
print(answer)
```

```python id="3HVM-Sr9pcRP" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629288057231, "user_tz": -120, "elapsed": 303, "user": {"displayName": "Kate", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ghe8UithqL0uAE6up9XspRl54AKAIidQfgG5XRbMJw=s64", "userId": "04668969765414769394"}} outputId="39821bc1-80c4-46c6-ad4e-6c0f295a55ed"
num = float(2.6)
print(num)
```

<!-- #region id="rdiy-EYfplta" -->
# **Arithmetic Operations**
We can perform different arithmetic operations in Python.

"**+**" Addition

"**-**" Subtraction

"*" Multiplication

"**/**" Division

"**//**" Division (returns whole num)

"**%**" Modulus (returns remainder of a division)

"**" Exponent (to the power)
<!-- #endregion -->

```python id="jbN18x6Ys_BD" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629288269679, "user_tz": -120, "elapsed": 28, "user": {"displayName": "Kate", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ghe8UithqL0uAE6up9XspRl54AKAIidQfgG5XRbMJw=s64", "userId": "04668969765414769394"}} outputId="026d85bf-57bb-45c6-9e29-ee5ee5820bca"
a = 7 % 2
b = 2 ** 2
print(a)
print(b)
```

<!-- #region id="-baOraJxtTEs" -->
# **Augmented Assignment**
To increment x by some value y, we can use an expression: x = x [operation] y
<!-- #endregion -->

```python id="TYFac7WTvQqv" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629288326758, "user_tz": -120, "elapsed": 301, "user": {"displayName": "Kate", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ghe8UithqL0uAE6up9XspRl54AKAIidQfgG5XRbMJw=s64", "userId": "04668969765414769394"}} outputId="07bf4452-8b4d-4f92-f551-90c906a17ffe"
x = 2
y = 1
x = x + y
print(x)
```

<!-- #region id="jdLlWOoMvkFg" -->
Python introduces **augmented assignment operators** to achieve the same purpose with less code. Here is an example:
<!-- #endregion -->

```python id="VssxYG0LwdCZ"
x = 2
y = 1
x += y
print(x)
```

<!-- #region id="DEmJBt9TwkL9" -->
In similar vein, we can use the following augmented operators:

*   x -= y
*   x *= y
*   x /= y
*   x //= y
*   x %= y
*   x **= y 
<!-- #endregion -->

<!-- #region id="vtTUhral7PnB" -->
# **If Statements**
If statements are used for decision-making. The program evaluates an expression, and executes the statement if the expression is **True**. 

![alt text](https://cdn.programiz.com/sites/tutorial2program/files/Python_if_statement.jpg)


Let us consider the following example:

***Example Exercise.***
You are working for Erasmus MC educational office. You have been assigned a task of deciding whether students qualify for the Master's program in Neuroscience. To qualify for this program, students need to have obtained 180 credits. 

Matt has 200 credits. Does he qualify?
<!-- #endregion -->

```python id="MNp6GtpqA5mV" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629288469698, "user_tz": -120, "elapsed": 282, "user": {"displayName": "Kate", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ghe8UithqL0uAE6up9XspRl54AKAIidQfgG5XRbMJw=s64", "userId": "04668969765414769394"}} outputId="361b692b-0490-449a-9728-2af93b69f803"
mattCredits = 100
if mattCredits >= 180:
  print('Qualifies')
else: 
  print('Does not qualify')
```

```python colab={"base_uri": "https://localhost:8080/"} id="bqLb1L4Xv51g" executionInfo={"status": "ok", "timestamp": 1631714344951, "user_tz": -120, "elapsed": 274, "user": {"displayName": "Lisa Korevaar", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiM7R332jNZ4TZ7MpoeDr65ckl-3pwU-qQ-4vW5=s64", "userId": "16862647957318551472"}} outputId="b3dd0a6a-fdae-4ca6-c286-67e2423186b2"
mattCredits = 200
if mattCredits >= 180:print ('Qualifies')
else: print ('Does not qualify')
```

<!-- #region id="aMuypHNiBeaE" -->
# **If...else Statements**
This statement evaluates the expression and executes the **if** conditional statement only if the expression is satisfied. Otherwise, it will execute the **else** conditional statement.

![alt text](https://cdn.programiz.com/sites/tutorial2program/files/Python_if_else_statement.jpg)

Let's consider the example of Maria:
<!-- #endregion -->

```python id="JFp_Tib0-vKn" colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"status": "ok", "timestamp": 1577695289131, "user_tz": -180, "elapsed": 1346, "user": {"displayName": "Su Saka", "photoUrl": "", "userId": "09941499294671263617"}} outputId="b376368b-b47e-4be8-ae28-fbe336a2737e"
mariaCredits = 160
if mariaCredits >= 180:
  qualifies = True
else: 
  qualifies = False
print(qualifies)
```

<!-- #region id="9oPGRJn8DFg5" -->
# **If...elif...else Statements**
By adding **elif** condition, Python allows us to check multiple expressions. We can add elif blocks multiple times inside an **if...else** statement. 

![alt text](https://cdn.programiz.com/sites/tutorial2program/files/Python_if_elif_else_statement.jpg)


***Example Exercise.***
Let us add onto the previous exercise. Let's say that if students do not have 180 credits, but have more or equal to 160 cts, they qualify for a conditional acceptance to the program.  
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="QCKAcdG8wtC2" executionInfo={"status": "ok", "timestamp": 1631714622463, "user_tz": -120, "elapsed": 216, "user": {"displayName": "Lisa Korevaar", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GiM7R332jNZ4TZ7MpoeDr65ckl-3pwU-qQ-4vW5=s64", "userId": "16862647957318551472"}} outputId="eec70bec-d5d3-4107-faae-ab0a0838942e"
mariaCredits = 170
if mariaCredits >= 180: qualification = 'Qualifies'
elif mariaCredits >= 160: qualification = 'Conditional acceptance'
else: qualification = 'Does not qualifiy'
print(qualification)
```

```python id="OJ3Yie60FnsO" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629288512243, "user_tz": -120, "elapsed": 283, "user": {"displayName": "Kate", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ghe8UithqL0uAE6up9XspRl54AKAIidQfgG5XRbMJw=s64", "userId": "04668969765414769394"}} outputId="7271a1c2-bc7b-4021-afd6-4aa3cb0c07de"
mariaCredits = 160
if mariaCredits >= 180:
  qualification = 'Qualifies'
elif mariaCredits >= 160:
  qualification = 'Conditional acceptance'
else: 
  qualification = 'Does not qualify'
print(qualification)
```

<!-- #region id="2Bo8FA-OMHMk" -->
# **Logical Operators**
If we want to test several conditions with a single if statement, we can use logical operators: '**and**', '**or**', '**not**'.

***Example Exercise***
Students who have recieved 180 credits and are older than 20, qualify for the program. Consider the example of Maria.
<!-- #endregion -->

```python id="yODOj_OdO4g-" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629288820757, "user_tz": -120, "elapsed": 14, "user": {"displayName": "Kate", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ghe8UithqL0uAE6up9XspRl54AKAIidQfgG5XRbMJw=s64", "userId": "04668969765414769394"}} outputId="3302d179-3cbf-4fa5-eca6-e5c585b28e1c"
creditsMaria = False
ageMaria = True
if creditsMaria and ageMaria:
  qualification = 'Qualifies'
else: 
  qualification = 'Does not Qualify'
print(qualification)
```

<!-- #region id="eplWORK2Qnfn" -->
With the logical '**and**' operator, both conditions have to be true to perform the conditional statement. With the '**or**' operator, at least one condition has to be true. '**not**' converts any boolean value into its opposite value.
<!-- #endregion -->

<!-- #region id="V1KYi12IRklE" -->
# **Comparison Operators**
Comparison Operators compare a value to a value. The return value of these operators is either True or False.


The following are some of comparison operators:
*   x > y
*   x < y
*   x == y (x equals y)
*   x != y (x does not equal y)


***Example Exercise***
If Maria has exactly 180 credits, she will qualify for the program. See if Maria qualifies for the program using an if...else statement.
<!-- #endregion -->

```python id="91cbkpYwVA_1" colab={"base_uri": "https://localhost:8080/", "height": 34} executionInfo={"status": "ok", "timestamp": 1577781785948, "user_tz": -180, "elapsed": 658, "user": {"displayName": "Su Saka", "photoUrl": "", "userId": "09941499294671263617"}} outputId="dc5c7119-4ee9-465e-cf8a-1988354f5588"
creditsMaria = 160
if creditsMaria == 180: 
  qualifies = True
else:
  qualifies = False
print(qualifies)
```

<!-- #region id="z74rtitFVkPb" -->
# **While Loops**
Useful terminology:

**Iteration** - Executing a block of code many times.
Iteration can be either **definite** or **indefinite**. 

>  **Indefinite iteration** - the number of times the program will execute the loop is not explicitly specified in advance. Rather, the program will repeat a given block of code as long as the condition is met.

> **Definite iteration** - the number of times the program will execute the loop is explicitly determined in advance.

**Loop** - A programming set of instructions that implements iteration. 

A while loop looks like this:

```
while [condition]:
  [statement]
```
As long as the condition is satisfied, the loop will continue to iterate. 

The condition usually involves variable(s) that are initialized ahead of the loop and incremented inside the loop.



```
i = 1
while i <= 5:
  i += 1
```
Once the condition becomes false (boolean value of False), the loop stops. 

***Example Exercise.***
Let us print Maria's age five times.

<!-- #endregion -->

```python id="zUFzxhtfcMEn" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629289085493, "user_tz": -120, "elapsed": 292, "user": {"displayName": "Kate", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ghe8UithqL0uAE6up9XspRl54AKAIidQfgG5XRbMJw=s64", "userId": "04668969765414769394"}} outputId="e7c25f30-3202-4cbc-caf6-8318958da6a1"
ageMaria = 24
i = 1
while i <= 5:
  print(str(ageMaria))
  i += 1
print('Done')
```

<!-- #region id="hdcLzxlFd2Y5" -->
# **For Loops**
`for` loops iterate over a sequence be it a list, a dictionary, a set or a string.

Using a `for` loop, we can perform a block of code for each item in a sequence. 

***Example Exercise.***
Print every letter of Maria's name on a new line.
<!-- #endregion -->

```python id="RR7zP7vaf4Qu" colab={"base_uri": "https://localhost:8080/", "height": 102} executionInfo={"status": "ok", "timestamp": 1577368761251, "user_tz": -60, "elapsed": 696, "user": {"displayName": "Natia Shamugia", "photoUrl": "https://lh3.googleusercontent.com/a-/AAuE7mDa1i61TP6_bth5HMNXGTBjN9pbpJK8rveVpPeNEw=s64", "userId": "11799764980580446930"}} outputId="5d2fb35c-28f0-44f9-b765-85e3edc11c68"
for item in "Maria":
  print(item)
```

<!-- #region id="MVhk5BKygibe" -->
***Example Exercise***.
Maria, Paul, and Giulia have been selected for the program. Print each of their names on a new line.
<!-- #endregion -->

```python id="zi-HdGm5gxdX" colab={"base_uri": "https://localhost:8080/", "height": 68} executionInfo={"status": "ok", "timestamp": 1577368919151, "user_tz": -60, "elapsed": 971, "user": {"displayName": "Natia Shamugia", "photoUrl": "https://lh3.googleusercontent.com/a-/AAuE7mDa1i61TP6_bth5HMNXGTBjN9pbpJK8rveVpPeNEw=s64", "userId": "11799764980580446930"}} outputId="10d47218-ce20-44b7-f0dc-4ec54409f8c8"
for item in ['Maria', 'Paul', 'Giulia']:
  print(item)
```

<!-- #region id="4jwJfR9Vg6Yn" -->
*We did this using a **list** of strings. You can find more about lists in a later section of the guide.*
<!-- #endregion -->

<!-- #region id="3A_SPEeThMFi" -->
# **Nested Loops**
Nested loops are loops inside loops. A nested loop will be executed for each iteration of the outer loop. It looks like this:


```
for x in y:
  for i in j:
  [statement]
```

***Example Exercise.***
Let's play a word game! We have three toys: a car, a guitar, and a ball. We also have three colors: red, blue, green. Write up combinations of these toys and colors. 

*Note: We will be using lists for this exercise.*
<!-- #endregion -->

```python id="DYk94okLjI4c" colab={"base_uri": "https://localhost:8080/", "height": 170} executionInfo={"status": "ok", "timestamp": 1577369629812, "user_tz": -60, "elapsed": 807, "user": {"displayName": "Natia Shamugia", "photoUrl": "https://lh3.googleusercontent.com/a-/AAuE7mDa1i61TP6_bth5HMNXGTBjN9pbpJK8rveVpPeNEw=s64", "userId": "11799764980580446930"}} outputId="c3cd27f6-d4c5-4246-ae4a-4402401c4c51"
toy = ['car', 'guitar', 'ball']
color = ['red', 'blue', 'green']
for i in toy:
  for j in color:
    print (i, j)
```

<!-- #region id="3FIbfh3Icw1n" -->
# **Lists**
A list is an array of items which is ordered and changable. Lists are defined by square brakets. 



```
classMembers = ['Maria', 'Paul', 'Giulia']
```
We can access individual items from a list using an index number.


```
classMembers = ['Maria', 'Paul', 'Giulia']
print(classMembers[0])
```
`classMembers[0]` will return the first item from a list (enumeration of the items in a list starts from 0). `classMembers[1]` will return the second item in the list and so on. 
We can also use **negative indexing**. For example `classMembers[-1]` would return the last item in the list and `classMembers[-2]` would return the second to last item. 

Therefore, we have two ways to access 'Paul' in this list. Run the code to check this yourself.

<!-- #endregion -->

```python id="hL5I3u9BfBVG" colab={"base_uri": "https://localhost:8080/", "height": 51} executionInfo={"status": "ok", "timestamp": 1577435601331, "user_tz": -60, "elapsed": 1052, "user": {"displayName": "Natia Shamugia", "photoUrl": "https://lh3.googleusercontent.com/a-/AAuE7mDa1i61TP6_bth5HMNXGTBjN9pbpJK8rveVpPeNEw=s64", "userId": "11799764980580446930"}} outputId="3528f95d-ee30-41e5-b28a-fd729c058496"
classMembers = ['Maria', 'Paul', 'Giulia']
print(classMembers[1])
print(classMembers[-2])
```

<!-- #region id="GqduvpRLfre1" -->
We can also select a **range** of items in a list. For this, we use the format `[a:b]`, which returns values from `a` until `b` in a list. We can also select items from `a` until the end of the list by `[a:]`, or we can select items from the beginning of the list until `a` by `[:a]`. 

For example, to select all the members of the list except for the first one, we would employ the following code. 

```
classMembers = ['Maria', 'Paul', 'Giulia']
print(classMembers[1:])
```

By using square brakets with a colon to select a range of items in a list, we create a new list with those items.

We can also **modify** a list. Here is an example:


```
classMembers = ['Maria', 'Paul', 'Giulia']
classMembers[0] = 'Nina'
```
This code would substitute Maria by Nina in our list.

***Example Exercise***
Create a list of 5 majors taught at EUR (Psychology, Law, MISOC, Economics, Management) in the specified order. First, display the last three majors together. Next, substitute the major of 'Law' by 'Urban Management'. Display the renewed list.
<!-- #endregion -->

```python id="unmf-RiRimKB" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629289356704, "user_tz": -120, "elapsed": 284, "user": {"displayName": "Kate", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ghe8UithqL0uAE6up9XspRl54AKAIidQfgG5XRbMJw=s64", "userId": "04668969765414769394"}} outputId="73932d20-07a0-402e-cc63-7104a5ad9e2f"
majors = ['Psychology', 'Law', 'MISOC', 'Economics', 'Management']
print(majors[2:])
majors[1] = 'Urban Management'
print(majors)
```

<!-- #region id="agVYjJFGjMJ6" -->
***Example Exercise***
Define a list of numbers `num = [1, 3, 5, 2, 4, 0]`. Find the largest number in this list.
<!-- #endregion -->

```python id="CS3c1TV-jorI" colab={"base_uri": "https://localhost:8080/", "height": 215} executionInfo={"status": "error", "timestamp": 1630434507858, "user_tz": -120, "elapsed": 254, "user": {"displayName": "Alessandro Jingberger", "photoUrl": "", "userId": "10099533819200777556"}} outputId="854d8b7e-6e44-4d2b-b2c8-7e799aabc28b"
num = [1, 3, 5, 2, 4, 0]
max = num[0]
for number in num:
  if number > max: 
    max = number
print(max)

num2 = [1, 3, 5, 2, 4, 0]
print(max(num2))
```

```python colab={"base_uri": "https://localhost:8080/", "height": 122} id="sUrdaerON4pd" executionInfo={"status": "ok", "timestamp": 1629289652017, "user_tz": -120, "elapsed": 808, "user": {"displayName": "Kate", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ghe8UithqL0uAE6up9XspRl54AKAIidQfgG5XRbMJw=s64", "userId": "04668969765414769394"}} outputId="6c421d35-69e6-431c-f99c-25acc78ea496"
# load an example dataset
from vega_datasets import data
cars = data.cars()

# plot the dataset, referencing dataframe column names
import altair as alt
alt.Chart(cars).mark_bar().encode(
  x='mean(Miles_per_Gallon)',
  y='Origin',
  color='Origin'
)
```

<!-- #region id="nCwHTIEnkU6z" -->
# **2D Lists**
Python does not have a separate type for **matrices**. However we can treat a 2D list as matrix. 

A **2D list** is a  list of lists where every element is a list itself. Here is an example:


```
abc = [
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9]
]
```
The first element of abc, which we would access by `abc[0]`, is a list itself: `[1, 2, 3]`. The first element of the first list would then be `abc[0][0]`. 

To process each individual item in a matrix, we use `for` loops. 


```
abc = [
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9]
]

for row in abc:
  for item in row:
    print(item)
```

To print this matrix, we can use a `for` loop.



<!-- #endregion -->

```python id="8J6r5HU3qgtY" colab={"base_uri": "https://localhost:8080/", "height": 68} executionInfo={"status": "ok", "timestamp": 1577716949043, "user_tz": -180, "elapsed": 614, "user": {"displayName": "Su Saka", "photoUrl": "", "userId": "09941499294671263617"}} outputId="801862a7-8293-4387-f553-2e68f7ccd873"
abc = [
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9]
]
for row in abc:
  print(row)
```

<!-- #region id="xhZrIVB8zEB8" -->
Python offers some **methods** that we can use for lists. For example, we can add an element to a list using an `append()` method. 

For more on these methods, consult the following [link.](https://www.geeksforgeeks.org/list-methods-python/)
<!-- #endregion -->

<!-- #region id="6NFRwit9_JBr" -->
# **Functions**
A **function** is a reusable block of code which returns data. Python provides built-in functions like the `print()` function. However, we can also define functions ourselves (**user-defined functions**). 

To create functions we can use the keyword `def` in front of the function name and the parantheses. Parameters can be passed into the function as arguments inside the parantheses.

***Example Exercise.*** Create a function that returns a greeting message for the Computing Brain students after their registration.
<!-- #endregion -->

```python id="YtyLMsibFDfd"
def greeting():
  print('Thank you for your registration')
  print('Welcome to this course')
```

<!-- #region id="ewuNuS47FdWL" -->
Defining a function only specifies the function, its parameters, and operations. In order to use the function, we need to **call** it. 
<!-- #endregion -->

```python id="19aFDt5HFx6I"
greeting()
```

<!-- #region id="nB-wThklGGep" -->
As mentioned earlier, we can input parameters into a function. Parameters act as local variables for the function. *Note that if a function has a **parameter**, we have to supply a value for that parameter. Otherwise, the code will not work.*

***Example Exercise.***
Create a function that returns a square of any digit.  Test the function for number 3. 
<!-- #endregion -->

```python id="j-JFMtieGgmQ" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629289881898, "user_tz": -120, "elapsed": 16, "user": {"displayName": "Kate", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ghe8UithqL0uAE6up9XspRl54AKAIidQfgG5XRbMJw=s64", "userId": "04668969765414769394"}} outputId="7fe5a0b4-e329-4730-be73-506fbbe08ad1"
def square(int):
  sq = int*int
  print(sq)

square(3)
```

<!-- #region id="39vNqyeiG7gp" -->
***Example Exercise***. Let's take it a step further! Create a function that takes two lists and returns a matrix. Take two lists: `[1, 2, 3]` and `[4, 5, 6]` and create their matrix.
<!-- #endregion -->

```python id="50IZEmRRHags" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629289989265, "user_tz": -120, "elapsed": 280, "user": {"displayName": "Kate", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ghe8UithqL0uAE6up9XspRl54AKAIidQfgG5XRbMJw=s64", "userId": "04668969765414769394"}} outputId="c8527725-7137-4995-8c5a-071e6b09d335"
#Defining the function
def matrix (a, b):
  mt = [a, b]
  print(mt)
  

#Testing the function for two specific lists
list1 = [1, 2, 3]
list2 = [4, 5, 6]
matrix(list1, list2)

```

<!-- #region id="zQKc3VJ1LfA_" -->
# **Return Statements**
So far we have used the built in `print()` function to see the results of our user-defined functions. However, to use a result of a function in any other operation, we should employ the `return` option in Python. The **return** option returns the result of the function. *Note: if `return `is not followed by an expression, the function returns value the value of `None`.*


***Example Exercise.*** Define a function that takes a numeric value and returns returns the solution to this equation: y = 4x + 15, where x is our input. 
<!-- #endregion -->

```python id="tuE97rEsR82j" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629290019361, "user_tz": -120, "elapsed": 288, "user": {"displayName": "Kate", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ghe8UithqL0uAE6up9XspRl54AKAIidQfgG5XRbMJw=s64", "userId": "04668969765414769394"}} outputId="14b5ad9a-01d2-450c-a461-3ebf84f13008"
def solution(x):
  y = 4*x+15
  return y 

solution(3)
```

<!-- #region id="N7X5qdPpUvZb" -->
# **Classes**
Creating a new class, creates a new **type** of an object. Other types we have encountered so far were integers, lists, and booleans. A class is a new type of an object that allows for multiple **instances** of this type to be created. Instances of a class are **objects**. 

To define a class we can use the keyword `class`. 

Let's create a simple class for demonstration.


<!-- #endregion -->

```python id="JgsGeQKkniUf"
class Student:
  pass
```

<!-- #region id="x8AbaeVinrIA" -->
***Note***: If you want to leave a class empty for a while, you can type `pass`.
<!-- #endregion -->

<!-- #region id="bBrMj1saoB4h" -->
We can now create instances of this class. For example:

*   List item
*   List item


<!-- #endregion -->

```python id="zTxjkbx7odpr"
class Student:
  pass
  
maria = Student()
nick = Student()
```

<!-- #region id="WxfJ0dkioozA" -->
Maria and Nick are now unique instances of the class "Student". 
<!-- #endregion -->

<!-- #region id="vec2ZDVOpc_b" -->
Let's examine classes more in depth. 

`__init__()` function is a built-in function in Python that is called automatically when an instance of a class is created. Like every method (function) that is created within a class, `__init__()` receives the instance as a first argument. Usually we name this argument `self`. After the first argument, we can specify other arguments we might need. These arguments are usually attributes of an instance. 

 ***Example Exercise.*** Create a class 'Student' that has the following properties: name, age, credits. 

<!-- #endregion -->

```python id="7GAL9NP_rBE0"
class Student:

    def __init__(self, name, age, credits):
      self.name = name
      self.age = age
      self.credits = credits
```

<!-- #region id="MiPAQI2ztKKF" -->
We have now created a class called 'Student' that has attributes such as name, age, and number of credits. 

To create instances of this class (**objects**), we can call upon the class the same way we did in the previous example. However, this time, we can specify the attributes of an object as arguments. *Note*: we can omit `self` when entering the arguments, it is automatically passed on. 

***Example Exercise.*** Create an instance of a class 'Student' that is named Maria, has 160 credits and is 24 years old. 
<!-- #endregion -->

```python id="0gs2w5mruLMX"
maria = Student('Maria', 24, 160)
```

<!-- #region id="VPXlSfq3uqSC" -->
So far we have only created **attributes** of our class. We can also create methods (functions) inside the class. 

***Example Exercise***. We can create a print method inside our class ('Student'). Let this  method print out the name, age and number of credits of a student followed by a message: *Welcome to Computing Brain!*.

Use an example of Maria!
<!-- #endregion -->

```python id="-wFabNtHwK1V" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1629290387567, "user_tz": -120, "elapsed": 307, "user": {"displayName": "Kate", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Ghe8UithqL0uAE6up9XspRl54AKAIidQfgG5XRbMJw=s64", "userId": "04668969765414769394"}} outputId="c88595a1-d87f-4196-a315-500ace296b25"
class Student:

    def __init__(self, name, age, credits):
      self.name = name
      self.age = age
      self.credits = credits
    
    def greeting(self):
     print(self.name)
     print(self.age)
     print(self.credits)
     print('Welcome to this course '+ self.name + '!')


maria = Student('Maria', 24, 160)
maria.greeting()
```

<!-- #region id="n6HJ7Rtd0OUu" -->
**Further (and free) Learning Resources:**




*   [Programming with Mosh 6 hour tutorial for beginners](https://www.youtube.com/watch?v=_uQrJ0TkZlc&t=14704s)
*   [Programmiz web page for Python ](https://https://www.programiz.com/python-programming)
*   [Free Code Camp 4 hour tutorial on Python](https://www.youtube.com/watch?v=rfscVS0vtbw)
*   [Python tutorial on Classes (there are more videos on his channel)](https://www.youtube.com/watch?v=ZDa-Z5JzLYM)
*   [Classes and Objects](https://www.youtube.com/watch?v=wfcWRAxRVBA)
*   [52 Python Exercises for Beginners](https://programmingwithmosh.com/python/python-exercises-and-questions-for-beginners/)

Lastly, google is your guide :) There are additional tutorials for anything you might find difficult to understand. Look it up!
<!-- #endregion -->

<!-- #region id="NZmhZXAAEdvE" -->
Figure source https://www.programiz.com/python-programming/if-elif-else
<!-- #endregion -->
