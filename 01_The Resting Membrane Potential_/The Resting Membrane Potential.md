
### In Short:
On our path to understand the origin of action potentials (spikes), we first need to grasp why different ionic concentrations across the cell membrane lead to an electrical potential, and this is the goal of this preparatory project.

# Learning Objectives

After this project you'll be able to:
- Describe how ionic composition inside and outside the cell define the **resting potential**.
- Know the difference between the resting potential and the **equilibrium potential**.

### The landmarks in the roadmap:

1. The electrical potential across membrane is due to differences in chemical concentration.
2. The membrane potential is altered by flux of ions across the membrane via ion channels.

# Key Terms

- **Membrane Potential (in short: Vm)**. The electric potential across the plasma membrane, i.e. between the inside and outside of a cell. Measured in (milli) volts (mV). **Attention!** the Vm is mesured in mV.
- **Steady State**. When a the value of a variable does not change over time, that is, it is 'steady', we say it has reached the steady state.
- **Equilibrium (Nernst) Potential**. The potential difference at which no net flow of a single ion type occurs, given a certain concentration difference across the membrane. Also called '**reversal potential**'.
- **Resting Potential**. The steady state value of the membrane potential when no active current occurs.
- **Permeability**. The tendency of an ion to cross the membrane. Plastic is 'impermeable to water'. It is proportional to the number of open channels on the membrane for an ion. Commonly expressed as relative permeability of the ion channels compared to one another.
- **Ion Channels**. Pores through the membrane that allow for selective permeability of different ion types. There are 'active' and 'passive' ion channels. Active channels change their conductance as a function of, for example, membrane potential. The conductance of passive channels is constant.
- **Conductance, _G_**. The reciprocal of resistance (1/R), measures how easily the electricity flows through parts of the circuit for a given difference in voltage. Is an electrical property and is influenced by permeability. Measured in siemens, the reciprocal of ohm (‎S‎ = $\Omega^{-1}$)


# Pre-requisites

Check the pre-requisites and complement your knowledge if something is missing. Some suggested videos are here, but feel free to find your own. Also, if you find a video you love, you can contribute with the whole "computing brain" crowd via the [computing brain channel](https://www.youtube.com/channel/UCU2BRdfg49st7ZdFMZbDScg)!

- Review basics of theory of gases (relating pressure, volume and temperature).
- Review basic differential equations (ODE's) (via 3blue1brown).
- Ions and electrical charge (Ohm's law) : [video 1, ](https://www.youtube.com/watch?v=G3H5lKoWPpY) [video 2](https://www.youtube.com/watch?v=fGI9d0CjI8s)
- Basic understanding of the electrochemical gradient [video](https://www.youtube.com/watch?v=Ba02v7eoVWQ).
- Very basic notions of (python) programming:
  - Data types such as strings, variables, functions [video](https://youtu.be/OH86oLzVzzw).
  - Data structures such as lists, tuples and vectors [video](https://youtu.be/gCCVsvgR2KU)
  - Python Modules and Functions [video](https://youtu.be/qbWBhyGmCs0)


# Membrane Physiology

#### Why is there an electrical potential across the membrane of cells?

**The potential across the membrane exists because the membrane separates ions and molecules which are electrically charged**. The inside of the neuron and the outside include different ion/molecule distributions. For instance, the inside includes many anions (negatively charged ions), many of which are large molecules that are in fact, the cell machinery (e.g., proteins ) and in general do not cross the membrane thus we call the membrane **SEMI-PERMEABLE**. Hence there's some negative charge that stays inside the neurons.

Two main physical processes  determine electrical signalling in neurons: the **electrical gradient and chemical (or molecular) diffusion**. The electrical gradient is the tendency that charges have to be repelled or attracted by each other as a function of their polarity (**opposites attract**). Chemodiffusion is when chemicals (ions, molecules) diffuse from an area of high concentration to an area of low concentration.

Both processes are attempting to reach equilibrium: the first tends to a uniform distribution of charge and the second homogeneous concentration.

<img src="https://nobaproject.com/images/shared/images/000/002/783/original.jpg" width="200x">

**The differences in charge density in ions across the membrane determine a potential**, that is, the willingness of charges to cross to the other side. Counterbalancing the difference of potential are the concentrations of the molecules themselves, which attempt to distribute themselves uniformly (but can't because of the semi-permeable membrane).

<img src="https://cdn.kastatic.org/ka-perseus-images/d9b40c6b51d9dbbb013a2dd38206aa496f6c7a58.png" width="300">

Potential is, like distance, always measured relative to a reference point. For neurons this is the outside of the cell.

### The Equilibrium Potential for a Single Ion (Nernst Potential)

The electrical potential difference across the cell membrane that balances the concentration gradient **for a single ion type** is known as the equilibrium potential. In general, an ion tend to flow in a way that moves the membrane potential closer to the ion's equilibrium potential.

Let us imagine that our neuron is a bubble with a perfectly impermeable (i.e. nothing passes through) membrane. We can calculate the **Nernst potential** (also called equilibrium potential) of a single ion type as a function of concentration, which relates the potential across the membrane as a function of charge concentrations inside and outside the cells.

It is the potential at which the electric force apply to an ion:

$${z F} (V_{in} - V_{out}) $$

balances the tendency for an ion to diffuse down its concentration gradient

$$ {R T}\ln\frac{[\text{ion outside cell}]}{[\text{ion inside cell}]}$$

Equating the two gives us the Nernst equation:


---


$$(V_{in} - V_{out}) =E_{eq} = \frac{R T}{z F} \ln\frac{[\text{ion outside cell}]}{[\text{ion inside cell}]}$$


---



The equation for the equilibrium potential may seem daunting at first but it is not hard to get an intuition about how it arises.

As ions have a charge and occupy space, we use the Faraday constant $F$ to relate the concentration of ions to the total charge. $R$ is the constant of gases, and indicates how density changes as a function of temperature $T$.

$z$ is the total charge per ion. For example, Sodium has one positive charge $Na^+$, Chloride has one negative charge $Cl^-$ and Calcium has two positive charges $Ca^{2+}$. Within brackets are concentrations.

Regarding the units: R is the universal gas constant and gives joules per kelvin per mole (J K−1 mol−1), temperature is in kelvins and the Faraday constant is the number of coulombs per mole of electrons.

Note: you can find an intuition about why the natural logarithm appears in the Nernst equation
here https://betterexplained.com/articles/demystifying-the-natural-logarithm-ln/.

#### Code: Calculating the **Equilibrium**  Potential

When the concentration of a certain ion is different between outside and inside of a membrane, there is a non-zero potential at which that ion does not flow in or out of the cell: $E_{eq}$. Read the python code below, which interactively calculates the Nernst potential for given concentration values and ion valence.


---

In the code below we want to calculate the Equilibrium (Nernst) Potential.

We first define the constants needed for this equation: The constant of gases (R), the temperature (T), and the Faraday constant (F).

We then define the function to calculate the potential. This function '''Nernst_func()''' will take the charge concentration outside (ion_outside), charge concentration inside (ion_inside) and valence (ion_valence) as parameters. The function will then use these values to compute the Nernst potential (E_ion). E_ion is defined as the Nernst equation we saw in the previous section.

After having defined the function to compute the Nernst equation and having returned the value, the code goes on to create an interactive display for the reader (defined as w). Inside the interactive, maximum and minimum possible values of all the parameters are defined. For example, the possible values for concentration of ions outside can range from 0.1 to 200.

```python
# we define a function that calculates the nernst potential
def Nernst(ion_outside, ion_inside, ion_valence):

  # define constants inside the function (SI units)
  R = 8.314  # Constant of gases that relates changes in volume, pressure and temperature
  T = 298.15 # in Kelvin
  F = 96.48 # The Fahraday constant (relates ion valance to charge)

  z = ion_valence;
    # define the nerns t equation (np.log is ln)
  E_ion = (R*T)/(z*F) * np.log(ion_outside/ion_inside)

  display(f'The Nernst equilibrium potential is {round(E_ion,2)} mV')

  return E_ion

# we pass the function to 'interactive', which is a 'class' that creates sliders
# for the paramters defined in Nernst. Note that we must define ranges for
# input variables, or pass the desired values as numpy arrays (np.array).
w = interactive(Nernst, ion_outside=(0.1,200), ion_inside=(0.1,200.0),ion_valence=np.array([-2,-1,1,2]))

# display our little Nernst calculator
display(w)
```
```
interactive(children=(FloatSlider(value=100.05, description='ion_outside', max=200.0, min=0.1), FloatSlider(va…
```

#### Question:
- What is the ion imbalance required for a Nernst potential of -70mV? Is that the same for different ions?

### Solution

We can use the Nernst Potential equation to answer this question:

$$E_{eq} = \frac{R T}{z F} \ln\frac{[\text{ion outside cell}]}{[\text{ion inside cell}]}$$

into this formula, we can insert values of F = 96.48, R = 8.314, T = 298.15, and E = -70.

As a result, we obtain:

$$-70 = \frac{8.314* 298.15}{z *96.48} \ln\frac{[\text{ion outside cell}]}{[\text{ion inside cell}]}$$



Let us define the proportion of $$\frac{[\text{ion outside cell}]}{[\text{ion inside cell}]}$$ as x.



If we manipulate this equation, we obtain:

$$1/z * \ln{x} = \frac{-70* 96.48}{8.314* 298.15} $$

Therefore:
$$\frac{1}{z} * \ln{x} = -2.7245 $$

Thus:
$$ \ln{x}^{\frac{1}{z}} = -2.7245 $$

Thus:

$$x ^{\frac{1}{z}} = e^{-2.7245}
$$

And:
$$x = 0.0659^z$$

The solution depends on the value of z (valence of an ion).

### Exercise:

#### Calculate the Nernst Potential for the Major Ions in a Cell

The figure below (from "Dynamical Systems in Neuroscience") summarizes the concentrations for the most common ions inside and outside the cell. Note the $A^-$, which represents other anions (which can be large molecules such as proteins).

Calculate the reversal potentials for each of the ions due to the concentration differences via the Nernst Equation.

<img src="https://i.postimg.cc/htNhBzfB/ion-concetrations-mammalian-cell.png" width="300x">

### Solutions:

At room temperature (25 degrees Celcius),

**For Sodium**

$$E_{eq} = \frac{8.314* 298.15}{1 *96.48} \ln\frac{145}{15}$$

$$E_{eq} = 58.288  \text{ mV}$$

**For Potassium**

$$E_{eq} = \frac{8.314* 298.15}{1 *96.48} \ln\frac{5}{140}$$

$$E_{eq} = -85.613  \text{ mV}$$

**For Chloride**
$$E_{eq} = \frac{8.314* 298.15}{-1 *96.48} \ln\frac{110}{4}$$
$$E_{eq} = -85.15  \text{ mV}$$

**For Calcium**

*Note! We need to transform micromollars to millimolars.*


$$E_{eq} = \frac{8.314* 298.15}{2 *96.48} \ln\frac{5}{0.0001}$$


$$E_{eq} = -109.41  \text{ V}$$

### Comprehension Questions

- What variable drives the value for the membrane potential the most?
- How does the valence of the ion influence the Nernst potential?
- According to the data given above, which ion has the largest concentration difference?

### Answers:

1. It is commanded by the concentration difference of the ion with largest charge. Theoretically it can be very large as the denominatory can be really small (though the logarithm makes the membrane potential grow slowly). Notice that concentration may changes as ions cross the membrane, and so the resting potential is also a dynamical variable.

2. The valence of an ion influences the potential in two ways: (a) influences whether potential is negative or positive; and (b) influences the size of the potential due to its reciprocal relationship.

3. The largest concentration difference is between the inside and outside concentrations of calcium. It is far less concentrated inside a neuron.

## The Resting Membrane Potential

The resting membrane potential is determined by the uneven distribution of ions (charged particles) between the inside and the outside of the cell, and by the different permeability of the membrane to different types of ions.

**An operational definition of resting potential** could simply be, the potential difference to which the cell likes to tend to.

A cell with a perfectly impermeable membrane would keep the difference of potential always unchanged, like a battery that is not in a circuit. The difference of potential due to the charge differences is its resting potential.

However, due to concentration differences, some **ions** enter and some ions exit the cell, through **ion channels**. The rate at which ions enter and exit the cell is determined by a combination of :
1. Concentration differences for ions inside and outside the cell.
2. The potential across the membrane.
3. The number of pores in the membrane (ion channels)
 and their permeability.

### Calculating The Resting Membrane Potential

A resting membrane potential is determined by the contributions of multiple ions. The movement of ions down the electrochemical gradient will move the resting potential of the membrane toward the equilibrium potential for that given ion. These movements of different ions contribute to the membrane potential.

Membrane potential can be calculated by the **Goldman-Hodgkin-Katz equation**.

![alt text](https://i.postimg.cc/J0mN0qLy/Screen-Shot-2020-03-15-at-10-00-26-PM.png)

where $V_m$ is the membrane potential, $p_k$, $p_{Na}$, and $p_{Cl}$  are the membrane permeability for Potassium, Sodium, and Chloride respectively. $[K^+]_o$, $[Na^+]_o$, and $[Cl^-]_o$ are the concentration of Potassium, Sodium, and Chroride outside the cell. Whereas, $[K^+]_i$, $[Na^+]_i$, $[Cl^-]_i$ are their concentration inside the cell. We have already defined $R$ (universal gas constant), $T$ (temperature in Kalvin), and $F$ (Faraday's constant).

The permeability values of the relevant ions for a typical neuron at resting potential would be: $p_{K}$ = 1, $p_{Na}$= 0.05 and $p_{Cl}$ = 0.45.


Some code that calculates it

Explain Changes of Resting potential as a function of concentration variation

```python

def Nernst(P_k, P_Na, P_Cl):
  # define constants
  R = 8.314  # Constant of gases that relates changes in volume, pressure and temperature
  T = 298.15 # in Kelvin
  F = 96.48 # The Fahraday constant (relates ion valance to charge)


  # Define concentrations inside and outside the cell
  N_in = 15 *mM
  N_out = 145 *mM
  K_in = 140 *mM
  K_out = 5 *mM
  Cl_in = 4 *mM
  Cl_out = 110 *mM

  V = (R*T)/(F) * np.log(((P_k*K_out) + (P_Na*N_out) + (P_Cl*Cl_out))/((P_k*K_in) + (P_Na*N_in) + (P_Cl*Cl_in)))

  display(f'The membrane potential is {round(V,2)} mV')
  return V


w = interactive(Nernst, P_k=(0.001,1.1), P_Na =(0.001,1.1), P_Cl =(0.001,1.1))

# display our little Nernst calculator
display(w)
```
```
interactive(children=(FloatSlider(value=0.5505000000000001, description='P_k', max=1.1, min=0.001), FloatSlide…
```


#### Review Questions:
- Why is the membrane potential negative?
- How would you make the membrane potential more positive?

You can use [this simulator](https://www.physiologyweb.com/calculators/ghk_equation_calculator.html) to verify your own answers and build intuition about the influences of ion concentration on the membrane potential.

#### Possible Answers

- The resting potential is heavily influenced by the ion type which has the greatest conductance across the membrane - potassium. The equilibrium potential of potassium is negative, therefore the resting membrane potential is also negative.
- A membrane potential can increase through an influx of positive ions, for example sodium ions, or an efflux of negative ions.

## Rates of Change of Membrane Potential

In the next project we will learn **under what conditions** and **how fast** the membrane potential changes. Here is a preparatory intuition:

Start with a cell in the shape of a bubble, with a fatty membrane separating the outside and the inside. Insert some  negatively charged ions and molecules inside (such as Chloride or other anionic proteins). If this compartment is placed in a positively charged environment, there's a difference of potential across the membrane. **Why don't the ions diffuse until concentration equilibrium?**

**The membrane is not perfectly permeable**, there are  channels in the membrane and so ions can crossover. Nature does like an equilibrium and in principle ions would flow out of the membrane until charge balance is achieved.

However, some ions (could be charged molecules, of course) are too large to get across the membrane, thus they get stuck inside, keeping that charge in the cell. For instance, many proteins inside the cell are negatively charged (anions) and too big to cross, which renders the inside of the cell negatively charged. How negative? That's a function of how much **charge per volume**, i.e., how many proteins per volume (charge density) there is.

#### Question

* Assume that the the initial potential is set below (more negative than) the resting potential. What is the direction of current flow? Why is that?

#### Answer

* The membrane potential returns to rest but now current 'enters' the cell.

# Comprehension Quiz

#### Equilibrium Potential and Resting Membrane Potential
- What is the difference between the resting potential and the equilibrium potential?
- What would be the resting potential if cells only had potassium ions?
- What creates the resting potential?
- Which ion’s equilibrium potential contributes the most to the resting potential of the membrane?

#### Answers

1. An Nernst potential relates to an equilibrium potential of a single ion type. An equilibrium potential is a state of the ion where there is no movement across the membrane. The resting potential, however, is different from the equilibrium potential, when there is more than one type of ion in the system. The resting membrane potential takes into account concentrations of all ions in the system as well as their permeabilities.
1. If cells only had potassium ions, the resting potential would be equal to the equilibrium potential of potassium, which is approximately -85mV.

1. The concentration difference of charged particles between the inside and outside of the cell causes an electrical potential difference.

1. We can see from the Goldman-Hodgkin-Katz equation that the permeability of the potassium ion allows for the largest contribution. This makes sense as the typical resting membrane potential (-70mV) is relatively close to the equilibrium potential of the potassium ion (-88 mV).

# Further Online Resources

- [Nernst Simulator + Resting Membrane Potential](http://www.nernstgoldman.physiology.arizona.edu/#download)
- [A detailed explanation of the origin of the resting membrane potential](https://www.physiologyweb.com/lecture_notes/resting_membrane_potential/resting_membrane_potential.html)
