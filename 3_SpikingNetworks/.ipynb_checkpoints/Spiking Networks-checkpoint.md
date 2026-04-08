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

```python
# IF Network

Here we set up a network with 4000 **IF** neurons, with 80% excitatory and 20% inhibitory neurons. We use **event based synapses**. The parameters are chosen such that we obtain a balance between excitation and inhibition, as has been observed experimentally. We set the initial values of the membrane potential via ```rand()``` to kick start the network, which then produces sustained activity (an 'up-state').

# Create a network of integrate and fire neurons
# example from: https://brian2.readthedocs.io/en/stable/examples/standalone.cuba_openmp.html
start_scope()

taum = 20*ms #  membrane time constant
taue = 5*ms # excitatory synapse time constant
taui = 10*ms # excitatory synapse time constant
Vt = -50*mV # spiking threshold
Vr = -60*mV # reset potential
El = -49*mV # leak reversal potential

eqs = '''
dv/dt  = (ge+gi-(v-El))/taum : volt (unless refractory)
dge/dt = -ge/taue : volt (unless refractory)
dgi/dt = -gi/taui : volt (unless refractory)
'''

P = NeuronGroup(4000, eqs, threshold='v>Vt', reset='v = Vr', refractory=5*ms,
                method='exact')
P.v = 'Vr + rand() * (Vt - Vr)'
P.ge = 0*mV
P.gi = 0*mV

# we use the following connection weights
we = (60*0.27/10)*mV # excitatory synaptic weight (voltage)
wi = (-20*4.5/10)*mV # inhibitory synaptic weight

# and we use EVENT-BASED synapses
Ce = Synapses(P, P, on_pre='ge += we')
Ci = Synapses(P, P, on_pre='gi += wi')
Ce.connect('i<3200', p=0.02)  # we make the first 3200 neurons excitatory
Ci.connect('i>=3200', p=0.02) # and the rest inhibitory

s_mon = SpikeMonitor(P)

## With a single thread

### Running the model and profiling

run(1 * second, profile=True)

plot(s_mon.t/ms, s_mon.i, ',k')
xlabel('Time (ms)')
ylabel('Neuron index')
title('a bunch of spiking neurons')
show()

profiling_summary(show=5) 
```
