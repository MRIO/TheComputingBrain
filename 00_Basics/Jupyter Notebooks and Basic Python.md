---
jupyter:
  jupytext:
    formats: ipynb,md
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

<!-- #region id="1J9ezpR4cUf2" -->
# Begin here

> # To be able to edit and use this Notebook:
> 0. Learn about the google colaboratory, what it is and how does it work [in this video](https://www.youtube.com/watch?v=inN8seMm7UI).
> 1. in the file menu (top left), click ```open in playground```
> 3. still in the file menu, click ```save copy in drive```, to make your own > personalized and editable copy of this file.
> 4. edit as you like. If something breaks irreparably, open the original link and go back to step 1.
<!-- #endregion -->

<!-- #region nbgrader={"grade": false, "grade_id": "cell-1dc22398d565f6fa", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="FGzNFVUOaqKx" -->
# Review: Jupyter Notebooks and Python Basics

Welcome to the Computing Brain, and welcome to Python programming! In this course, you will use python to simulate neurons and neuronal networks.

In the computing brain we will use Python using a platform called "Jupyter Notebooks", which are hosted in the 'google collaboratory' (the site you're on now). Jupyter notebooks are a way to combine formatted text (like the text you are reading now), Python code (which you will use below), and the result of your code and calculations all in one place. Go through the notebooks and run the examples!

Try to understand the relationship between the code and the results and, when necessary, add code (such as print statements or variable overviews) to make it clear for you what is going on.

In addition, there are some exercises and practice cells where you can program for yourself. Don't be afraid to start coding yourself, writing code and making mistakes is the best way to learn Python.

In this lecture, you will learn the basic concepts of programming in Python. To get started, we will first explain the basic concepts of what python is and how it works. 

**Learning objectives for this notebook:**

* Student is able to start the python interpreter and run code in a notebook
* Student can stop and start notebook kernels
* Student is able to create variables
* Student can use `%whos` to list the variables stored in the memory of the python kernel
* Student is able to determine the type of a variable 
* Student is able to convert between different variable types (float, int, etc)
* Student is able to collect user input using the `input()` command
* Student is able to print variable values using the `print()` command
<!-- #endregion -->

<!-- #region nbgrader={"grade": false, "grade_id": "cell-1dc22398d565f6fa", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="c3pBzgtOaqK2" -->
## What is Python? And what are Jupyter Notebooks? 

Python is an (<a href=https://en.wikipedia.org/wiki/Interpreted_language>interpreted</a>) computer programming language. Using python, you can ask your computer to do specific things, like perform a calculation, draw a graph, load data from a file, or interact with the user. 

Every time you run Python, either by running it on the command line, or by running it in a Jupyter notebook like you are now, a Python **"kernel"** is created. This kernel is a copy of the Python program ("interpreter") that runs continuously on your computer (until you stop it).

Jupyter notebooks are a way to interact with the Python kernel. Notebooks are divided up into "cells", which can be either a text (<a href=https://en.wikipedia.org/wiki/Markdown>markdown</a> format) cell, like this one, or a code cell (containing your code), like the cell below it. 

The selected cell is surrounded by a box. If you type "enter" in a text cell you can start editing the cell. If you push "Run" above, or type "Shift-Enter", the code will be "run". If it is a code cell, it will run a command (see below). If it is a markdown cell, it will "compile" the markdown text language into formatted (HTML) text. 

You can give commands to this kernel by typing commands using the python language into the code cells of the notebook. Here, you can find an example of a code cell that contains a simple python command `print`, which prints a text string to the command line.
<!-- #endregion -->

```python id="YU-cC_5waqK3" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630928143392, "user_tz": -120, "elapsed": 24, "user": {"displayName": "Karlijn Hers", "photoUrl": "", "userId": "13078358567483151064"}} outputId="32deba48-6e83-4339-e55b-8527642f982b"
print("I'")
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-9913bc22153b2c2b", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="LztQiOZKaqK4" -->
To send this command to the python kernel, there are several options. First, select the cell (so that it is either blue or green), and then:

1. Click on the **Run** button above in the toolbar. This will execute the cell and move you to the next cell.
2. Push **Shift-Enter**: this will do the same thing
3. Push **Control-Enter**: this will run the cell but leave it selected (useful if you want to re-run a cell a bunch of times)

When you run the cell, the code will be sent to the python *kernel*, which will translate your python command into a binary language your computer CPU understands, send it to the CPU, and read back the answer. If the code you run produces an "output", meaning that the kernel will send something back to you, then the output that your code produces will be displayed below the code cell in the "output" section of the code cell. 

After you have run the code cell, a number will appear beside your code cell. This number tell you in which order that piece of code was sent to the kernel. Because the kernel has a "memory", as you will see in the next section, this number can be useful so that you remember in which order the code cells in your notebook were executed. 

In the example above, the code cell contained only a single line of code, but if you want, you can include as many lines as you want in your code cell and add comments by starting a line with `#`. It is good programming practice to use comments to explain what the code is doing, see the example below:
<!-- #endregion -->

```python id="B2L3eXZCaqK5" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630928143393, "user_tz": -120, "elapsed": 21, "user": {"displayName": "Karlijn Hers", "photoUrl": "", "userId": "13078358567483151064"}} outputId="dcf52ee2-e9ab-46eb-dd1c-607b540deae4"
# This will print out a message
print("Hello")
print("world")
print("Goodbye")
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-727a76c294ed6128", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="CezAd198aqK6" -->
**Exercise 1** Print your own string to the command line. Can you use special characters as well? Try it.
<!-- #endregion -->

```python id="Y12qJ_TQaqK7" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630928143664, "user_tz": -120, "elapsed": 287, "user": {"displayName": "Karlijn Hers", "photoUrl": "", "userId": "13078358567483151064"}} outputId="1d2f245c-7573-478a-b26d-7d02dbd367d3"
# Your code here
print("always remember to smile \t because it\s good for you")
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-17aa1b39ce6eb2cb", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="C9Y7qOCQaqK7" -->
## The Python kernel has a memory

In addition to asking Python to do things for you, like the "Hello world" example above, you can also have Python remember things for you. To do this, you can assign the value of `5` to the variable `a` using:
<!-- #endregion -->

```python id="YLbgJ191aqK8"
a = 5
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-18e5a4f524b5ed62", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="Visnn6VTaqK9" -->
If variable `a` already exists, it will be over-written with the new value (in fact, `a` is a Python object, something that we will explain in the optional notebook in more detail). If variable `a` does not yet exist, then Python will create a new variable for you automatically.

For you, the cell above will create a "variable" named `a` in memory of the Python kernel that has the value of 5. We can check this by printing the value of a:
<!-- #endregion -->

```python id="BaBMuiJyaqK-" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630928143665, "user_tz": -120, "elapsed": 44, "user": {"displayName": "Karlijn Hers", "photoUrl": "", "userId": "13078358567483151064"}} outputId="bd239936-c528-4fb7-93f2-ddb12696aef5"
print(a)
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-4f248fa53a545d1f", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="VoULTEIvaqK-" -->
Note that we can also add a message to this by combining this with a message with the `print()` statement by combining things with commas:
<!-- #endregion -->

```python id="kZiTSs8uaqLA" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1630928143666, "user_tz": -120, "elapsed": 38, "user": {"displayName": "Karlijn Hers", "photoUrl": "", "userId": "13078358567483151064"}} outputId="9a5836c3-81a0-40e9-a8c4-2b19857d19a6"
print("The value of a is",a)
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-95d902ed15944fe7", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="uC9AUb4GaqLA" -->
**Exercise 2** Combine a string, a variable, and a numerical values in a single `print` statement using the `,` separator.
<!-- #endregion -->

```python id="ptFW9i6iaqLC"
# Your code here
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-7fc0accab5e0013c", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="g-qK3iJfaqLD" -->
**Exercise 3** Change the value of `a` to 7 by executing the following cell, and then re-run the **above** cell containing the command `print(a)` (the one with output `5`). What value gets printed now in that cell?  
<!-- #endregion -->

```python id="wKY6oLkgaqLF" colab={"base_uri": "https://localhost:8080/", "height": 130} executionInfo={"status": "error", "timestamp": 1630928143987, "user_tz": -120, "elapsed": 352, "user": {"displayName": "Karlijn Hers", "photoUrl": "", "userId": "13078358567483151064"}} outputId="b0b51585-8f5f-4717-b074-b26cbd57b5ca"
a = 
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-1ea24fed746f7f8e", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="xybZNM8jaqLG" -->
As you can see in notebooks that the location of your code doesn’t matter, but the order in which you execute them does!! 
<!-- #endregion -->

<!-- #region nbgrader={"grade": false, "grade_id": "cell-846d49e0e562281a", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="3O3eS-sRaqLH" -->
Sometimes, if you execute a lot of cells, or maybe even re-execute a cell after changing its contents, you might lose track of what variables are defined in the memory of your Python kernel. For this, there is a convenient built-in "magic" command called `%whos` that can list for you all the variables that have been defined in your kernel, along with their values:
<!-- #endregion -->

```python id="rgDxBof3aqLJ"
a=5 
%whos
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-9198f12541523f47", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="ytGZlrrJaqLK" -->
_(Some notes about `%whos`: `%whos` is not a "native" command of the Python language, but instead a "built-in" command that has been added by the creators of Jupyter. Because of this, you cannot use it outside of Jupyter / iPython...)_

If we define some new variables, they will also appear in the list of defined variables if you execute `%whos`:
<!-- #endregion -->

```python id="P6-moZ7haqLK"
c = 10
d = 15.5
```

```python id="EIBLwK3baqLK"
%whos
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-46a8548402b88535", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="lliAINUvaqLL" -->
In this case the variable named is displayed, its value, but also its type. Type defines the format in which a variable is stored in memory. In this case `int` stands for integer and `float` stands for floating point number, which is the usual way in which real numbers are stored in a computer. We will learn more about Python variable types below.

If you want to be sure that your code runs without being based on the exact history of your commands you use "Kernel/Restart and run all". In this case the entire notebook is runned from top to bottom. This is essential, e.g. when handing in an assignment, that is run on a different computer to obtain the same results.
<!-- #endregion -->

<!-- #region nbgrader={"grade": false, "grade_id": "cell-9f4fa67f21a94dfd", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="v4X7b7iXaqLL" -->
## Starting and stopping the kernel

When you open a notebook for the first time, a new kernel will be started for you, which will have nothing in your memory. 

Important to understand: if you close the tab of your browser that has the notebook, Jupyter **will not** shut down the kernel! It will leave the kernel running and when you re-open the notebook in your browser, all the variables you defined will still be in the memory. You can test this by closing this notebook right now, clicking on the link to open it in the "tree" file browser tab of Jupyter, and then re-running the cell above with the command `%whos`. 

How do I shutdown a kernel? And how do I know if a notebook on my computer already has a kernel running? 

* First, as you may have noticed, when you closed this notebook and went back to the "tree" file brower, the notebook icon had turned green. This is one way that Jupyter tells you that a notebook file has a running kernel.

* Second: in the <a href="."> "tree" view </a> of the Jupyter interface, there is a link at the top to a tab "Running" that will show you all the running kernels and allow you to stop them manually. 

Sometimes, you may want to restart the kernel of a notebook you are working on. You may want to do this to clear all the variables and run all your code again from a "fresh start" (which you should always do before submitting an assignment!). You may also need to do this if your kernel crashes (the "status" of your kernel can be seen in the icons at the right-hand side of the Jupyter menu bar at the top of the screen). 

For this, there is both a menubar "Kernel" at the top, along with two useful buttons in the toolbar: 

* "Stop": tells the kernel to abort trying to run the code it is working on, but does not erase its memory
* "Restart": "kill" the kernel (erasing its memory), and start a new one attached to the notebook.

To see this in action, you can execute the following cell, which will do nothing other than wait for 10 minutes:
<!-- #endregion -->

```python id="IfpZ6FwiaqLM"
from time import sleep
sleep(10*60)
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-56a8c8dff6af037b", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="-TaZ20KLaqLM" -->
You will notice that while a cell is running, the text beside it shows `In [*]:`. The `*` indicates that the cell is being executed, and will change to a number when the cell is finished. You will also see that the small circle beside the `Python 3` text on the right side of the Jupyter menu bar at the top of the page will become solid. Unless you have a lot of patience, you should probably stop the kernel, using the "Stop" button, or the menu item "Kernel / Interrupt".

**Exercise 4** List the stored variables using the `%whos` command. Subsequently, restart the kernel. What variables are stored in the memory of the kernel before and after the restart? 
<!-- #endregion -->

```python id="vIntkTQFaqLN"
# Your code here
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-5d130eb48ef7776b", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="MS576OvSaqLN" -->
## Python variable types

As we saw above, in Python, variable have a property that is called their "type". When you use the assignment operator `=` to assign a value to a variable, Python will automatically pick a variable type it thinks fits best, even changing the type of an existing variable if it thinks it is a good idea. 

You have, in fact, already seen information about the types of variables in the `%whos` command again:
<!-- #endregion -->

```python id="IRjSn6MxaqLO"
%whos
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-d224ffc0ba27e3c6", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="uFL4Tc6AaqLP" -->
In the second column, you can see the **type** that python chose for the variables we created. `int` corresponds to integer numbers, `float` corresponds to floating-point numbers. You can see that for variable `c`, python had to choose a `float` type (because 15.5 is not an integer), but for `a` and `b`, it chose integer types. 

_(In general, Python tries to choose a variable type that makes calculations the fastest and uses as little memory as possible.)_

If you assign a new value to a variable, it can change the variables type:
<!-- #endregion -->

```python id="YBPtKD_faqLP"
a = a/2
```

```python id="R-mtGFlbaqLR"
%whos
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-3498d6414e91ec42", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="TAfNpCilaqLT" -->
Because 5/2 = 2.5, Python decided to change the type of variable `a` from `int` to `float` after the assignment operation `a = a/2`. 

When you are using floating point numbers, you can also use an "exponential" notation to specify very big or very small numbers: 
<!-- #endregion -->

```python id="JITfIiQyaqLV"
c = 1.5e-8
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-a8ad6dfca5051b0d", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="8913_TKaaqLY" -->
The notation `1.5e-8` is a notation used in Python to indicate the number $1.5 \times 10^{-8}$.

A third type of mathematical variable type that you may use in physics is a complex number:  

https://en.wikipedia.org/wiki/Complex_number

In python, you can indicate a complex number by using `1j`, which is the python notation for the complex number $i$:
<!-- #endregion -->

```python id="_3qgis3laqLZ"
d = 1+1j
```

```python id="uiXyDarSaqLa"
%whos
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-b23bdfa19b2bbcf7", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="uULaciauaqLa" -->
The notation `1j` is special, in particular because there is **no space** between the number `1` and the `j`. This is how Python knows that you are telling it to make a complex number (and not just referring to a variable named `j`...). The number in front of the `j` can be any floating point number: for example,
<!-- #endregion -->

```python id="7rRDeFjbaqLb"
0.5*d
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-8dbe30e28c4cebcb", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="jmnRWenOaqLc" -->
In addition to the mathematical variable types listed above, there are also other types of variables in Python. A common one you may encounter is the "string" variable type `str`, which is used for pieces of text. To tell Python you want to make a string, you enclose the text of your string in either single forward quotes `'` or double forward quotes `"`:
<!-- #endregion -->

```python id="1rMyNTaHaqLd"
e = "This is a string"
f = 'This is also a string'
```

```python id="0o68avvhaqLe"
%whos
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-cc2fc145721b8de0", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="A5ZKJi6uaqLg" -->
You can also make multiline strings using three single quotes:
<!-- #endregion -->

```python id="G_0f3dT2aqLh"
multi = \
'''
This string
has 
multiple lines.
'''
print(multi)
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-b456d3390fb64aac", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="7hU7FEUTaqLi" -->
Note here that I have used a backslash: this a way to split Python code across multiple lines. 

Although it's not obvious, Python can also do "operations" on strings, the `+` mathematical opeartors we saw above also works with strings. 

**Exercise 5** Discover what the `+` operator does to a string, i.e. print the output of the sum of two strings.
<!-- #endregion -->

```python id="kSx9xt32aqLj"
# Your code here
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-bdd02babbe65afad", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="j2HL6YrRaqLj" -->
There is one more useful variable type we will introduce here: the "boolean" type `bool`. Boolean variable can have two values: `True` and `False`. You type them in directly as `True` and `False` with no quotes (you will see them turn green). 
<!-- #endregion -->

```python id="mT4nSNlPaqLk"
g = False
```

```python id="rwAF3q1taqLk"
%whos
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-5721d15652d7a772", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="ZmoBD8mnaqLl" -->
We will use boolean types much more later when we look at program control flow, but a simple example is the `if` statement:
<!-- #endregion -->

```python id="0MBWtAqtaqLl"
if True:
    print("True is always true.")

if g:
    print("g is true!")
    
if not g:
    print("g is not true!")
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-961fae57edafbc47", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="7O7CDMsPaqLm" -->
You can try changing the value of `g` above to `False` and see what happens if you run the above code cell again.

Also, useful to know: numbers (both `int` and `float`) can also be used in True / False statements! Python will interpret any number that is not zero as `True` and any number that is zero as `False`. 
<!-- #endregion -->

<!-- #region nbgrader={"grade": false, "grade_id": "cell-2cc953e28ecba6ff", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="xL_W-j1waqLm" -->
**Exercise 6** Discover which numbers can be used as `True` and `False` in Python by changing the value of `g` above and re-running the cells.
<!-- #endregion -->

```python id="3SkUXZLUaqLo"
# Your code here
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-ed80a79af068d613", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="LhAbEUgNaqLp" -->
## Converting variables between different types

We can also convert a value from one type to another by using functions with the same name as the type that we want to convert them to. Some examples:
<!-- #endregion -->

```python id="sH1EByveaqLp"
float(5)
```

```python id="g3LuicCwaqLq"
int(7.63)
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-4b16c4e2c67dc1b8", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="iE2ZWf1daqLr" -->
Note that when converting an `float` to an `int`, Python does not round off the value, but instead drops all the numbers off after the decimal point (it "trucates" it). If we want to convert to an integer and round it off, we can use the `round()` function:
<!-- #endregion -->

```python id="XU5WuUZDaqLr"
b = round(7.63)
print(b)
```

```python id="f1BxzKfZaqLs"
type(b)
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-f054d060004b3cf1", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="EehF-zosaqLs" -->
This works for conversions between many types. Sometimes, you have will lose information in this process: for example, converting a `float` to an `int`, we lose all the numbers after the decimal point. In this example, Python makes a guess at what you probably want to do, and decides to round off the floating point number to the nearest integer. 

Sometimes, Python can't decide what to do, and so it triggers an error:
<!-- #endregion -->

```python id="_ICqWatsaqLt"
float(1+1j)
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-379ecea9060815fa", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="bIxELeFHaqLu" -->
A very useful feature is that Python can convert numbers into strings:
<!-- #endregion -->

```python id="gFt0fwIWaqLu"
str(7.54)
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-d23c42a9d29ed231", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="mOs-99-RaqLw" -->
That is actually what happens when you use the `print()` commands with a numeric value.

But also very useful is that as long as your string is easily convertable to a number, Python can do this for you too!
<!-- #endregion -->

```python id="gWQyGJy7aqLx"
float('5.74')
```

```python id="eXjj0v68aqLx"
int('774')
```

```python id="w0Zv-noVaqLy"
complex('5+3j')
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-5a6e5094efe4dcc2", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="ZAbSjAkOaqL0" -->
**Exercise 7** Define a list of parameters with as many types as possible, i.e. all the examples you see above and maybe a few more. Use `%whos` to see how they look inside the computers' memory. Try to change their format and rerun the `%whos` command.
<!-- #endregion -->

```python id="c_BrP7vIaqL1"
# Your parameters list
a=
b=

# Parameter formats in the computer
%whos
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-18b615890b28b0d9", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="3j0IIjriaqL2" -->
## Python can do math

Python has a set of math functions that are directly built in to the language. You can use Python as a calculator! 
<!-- #endregion -->

```python id="UuE6BxoiaqL2"
1+1
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-e4111ecb75dde90e", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="KmOEDgPoaqL3" -->
Calculations also work with variables:
<!-- #endregion -->

```python id="fl6Vq005aqL4"
a = 5
print(a+1)
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-adc029a34cec985a", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="2Y79jfv8aqL4" -->
**Exercise 8** Discover what the following Python operators do by performing some math with them: `*`, `-`, `/`, `**`, `//`, `%`. Print the value of the mathematical operation to the command line in susequent cells.
<!-- #endregion -->

```python id="p6pg8xSGaqL5"
# Try out *
```

```python id="jfOCvukqaqL5"
# Try out -
```

```python id="_DrsiNYkaqL6"
# Try out /
```

```python id="ejRTkYCKaqL6"
# Try out **
```

```python id="DtvUVJTEaqL6"
# Try out //
```

```python id="sfa131dXaqL7"
# Try out %
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-91513b038a4fc7a8", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="hO75qoKVaqL7" -->
Another handy built-in function is `abs()`:
<!-- #endregion -->

```python id="GRqHT5c4aqL7"
print(abs(10))
print(abs(-10))
print(abs(1j))
print(abs(1+1j))
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-8b29691df833d4a7", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="jatPdicuaqL8" -->
You can find the full list of built-in math commands on the python documentation webpage:

https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex
<!-- #endregion -->

<!-- #region nbgrader={"grade": false, "grade_id": "cell-a251eae65e3d028a", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="SjJk16mIaqL8" -->
## Tab completion in Jupyter Notebooks


Computer programmers often forget things, and often they what variables they have defined. Also, computer programmers always like to save typing if they can. 

For this reason, the people who made Jupyter notebooks included a handy feature called "Tab completion" (which is actually something that has been around for <a href=https://en.wikipedia.org/wiki/Command-line_completion>a long time</a> in unix and ms-dos command line environments). 

The idea is that if you start typing part of the name of a variable or part of the name of a function, and then push the `Tab` key, Jupyter will bring up a list of the variable and function names that match what you have started to type. If ony one matches, it will automatically type the rest for you. If multiple things match, it will offer you a list: you can either keep typing until it's unique and type tab again, or you can use the cursor keys to select the one you want.
 
Here is an example: 
<!-- #endregion -->

```python id="NS2yBmNMaqL8"
this_is_my_very_long_variable_name = 5
this_is_another_ones = 6
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-0b103f33739bfa10", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="aJv8NRdlaqL8" -->
Now click on the following code cell, go the end of the lines in this cell and try pushing `Tab`:
<!-- #endregion -->

```python id="KdS_ChZDaqL9"
this
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-0690fd305a8cb84d", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="6f3tEZNqaqL9" -->
Handy! Jupyter did the typing for me! 

If multiple things match, you will get a drop-down box and can select the one you want. So press `Tab` : after
<!-- #endregion -->

```python id="bCJpg00HaqL9"
th
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-83f6b2e8f476ea2f", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="U58wVWC_aqL-" -->
You can also keep on typing: if you just type `a` after you hit tab and then hit tab again, it will finish the typing for you.
<!-- #endregion -->

<!-- #region nbgrader={"grade": false, "grade_id": "cell-eda207b10b294e4d", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="oWGWtb8saqL-" -->
**Exercise 9** Use tab completion on the initial letters of a few of the commands that have been presented. Along the way you will discover many more Python commands!
<!-- #endregion -->

```python id="t_6SZ_gdaqL-"
# Your code here
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-d2ead448ab8eb8e2", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="ubIXUoxwaqL-" -->
## Understanding Python Errors

Sometimes, the code you type into a code cell will not work. In this case, Python will not execute your code, but instead print out an error message. In this section, we will take a look at these error messages and learn how to understand them.

Let's write some code that will give an error. For example, this is a typo in the name of the `print()` command:
<!-- #endregion -->

```python id="19_LlPoqaqL_" executionInfo={"status": "error", "timestamp": 1645700590970, "user_tz": -60, "elapsed": 536, "user": {"displayName": "Mario Negrello", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14Gh9vnOzDqUD2QacfGPwd13jMPmwn1hzZzBnVURjO4E=s64", "userId": "10136788594790905986"}} colab={"base_uri": "https://localhost:8080/", "height": 181} outputId="40397e01-1187-4e26-d702-713b71ed2283"
a = 5
printt(a)
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-68d205ef1da24469", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="O-jpt2UEaqMA" -->
After your code cell, you will see some colored text called a "Traceback". This "Traceback" is the way that python tries to tell you where the error is. 

Let's take a look at the traceback:

<img src="resource/asnlib/public/anatomy_of_an_error.png" width=80%></img>

The traceback contains three important details that can help you:

1. The type of error ```NameError```
2. Where the error occurred in your code ``` ---> 2```
3. An attempt to explain why the error happened: `: ```name 'printt' is not defined```

For 1 and 2, Python is pretty good and will communicate clearly. For 3, sometimes you need to have some experience to understand what python is trying to tell you.

In this specific case, the type was a `NameError` that occured on line 2 of our code cell. 
*(By the way, in the View menu, you can turn on and off line numbers in your cells.)* 

A `NameError` means that python tried to find a function or variable that you have used, but failed to find one. If you look already at the line of code, you can probably spot the problem already.

At the very end of the traceback, Python tries to explain what the problem was: in this case, it is telling you that there is no function named `printt`. 

You will also get a `NameError` if you try to use a variable that doesn't exist:
<!-- #endregion -->

```python id="1EpudXDXaqMB"
print(non_existent_variable)
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-b54273a544366cf5", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="B3K1Ac7ZaqMB" -->
Another common type of error is a `SyntaxError`, which means you have typed something that python does not understand:
<!-- #endregion -->

```python id="zLEaAEJhaqMB"
a = a $ 5
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-eca4afbf3c0092a5", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="if2HOjmyaqMC" -->
You can also get errors if you try to use operators that do not work with the data type you have. For example, if you try to "divide" two strings:
<!-- #endregion -->

```python id="fVAfobVzaqME"
"You cannot " / "divide strings"
```

<!-- #region nbgrader={"grade": false, "grade_id": "cell-8a98a5ab32922e07", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="47bbSt8OaqME" -->
Here, you get a `TypeError`: the division operator is a perfectly fine syntax, it just does not work with strings. 


In Python, errors are also called "Exceptions", and a complete list of all error (exception) types, and what they mean, can be found here:

https://docs.python.org/3/library/exceptions.html#concrete-exceptions

Sometimes, you can learn more about what the error means by reading these documents, although they are perhaps a bit hard to understand for beginners. 

In last resort, you can also always try a internet search: searching for the error message can help, and there are also lots of useful posts on <a href=https://stackexchange.com>stack exchange</a> (which you will also often find).
<!-- #endregion -->

<!-- #region nbgrader={"grade": false, "grade_id": "cell-3779c815a0a9f77d", "locked": true, "schema_version": 3, "solution": false, "task": false} editable=false deletable=false id="r7L0y806aqMF" -->
**Exercise 10** Run the following code and try to understand what is going wrong by reading the error message.
<!-- #endregion -->

```python id="YTQ5Qn3GaqMF"
a=10
b=0
c=(a/b)
print(c)
```

```python id="AtrgwGnqaqMF"
4 + brains*3
```

```python id="Yjn5iypAaqMF"
d='the brain is awesome' + 2
```
