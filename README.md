
This is a repo of the synapse functions I've written.

# Installation
## Getting the repo
```bash
git clone https://github.com/AdamP1592/Synapse

cd Synapse

```

## Getting the dependencies

```bash
    pip install -r requirements.txt

```

---
# Synapse

## Synapse Model

### Tsodyks–Markram Synapse Model
**Description:**  
The Tsodyks-Markram synapse model simulates short-term synaptic plasticity by dynamically adjusting synaptic efficacy. It does so by tracking the fraction of available synaptic resources (`r`) and the utilization of these resources (`u`). The model uses these variables to compute the effective synaptic conductance (`g_syn`), which modulates the post-synaptic current. This approach captures both the depression (resource depletion) and facilitation (increased utilization) effects observed in biological synapses.



**Description:**  
The Tsodyks–Markram synapse model simulates short-term synaptic plasticity by dynamically adjusting synaptic efficacy. It does so by tracking two key variables:  
- **r**: the fraction of available synaptic resources, and  
- **u**: the utilization of these resources.  

The effective synaptic conductance is given by:

$$
g_{syn} = g_{max} \cdot r \cdot u
$$

where \( g_{max} \) is the maximum synaptic conductance.

**Dynamics:**  
Between spike events, the available resources and utilization evolve continuously:

$$
\frac{dr}{dt} = \frac{1 - r}{\tau_r}
$$

$$
\frac{du}{dt} = -\frac{u}{\tau_f}
$$

At a spike event, the variables are updated as follows:

$$
r \rightarrow r - u \cdot r
$$

$$
u \rightarrow u + u_{max} \cdot (1 - u)
$$

Here, ${\tau_r}$ and ${\tau_f}$ are the recovery and facilitation time constants, respectively, and $u_{\max}$ is the maximum utilization increment.

---

## Example Definitions

**Implementation example:**
```python
  from synapse_model import tsodyks_markram_synapse

  # Example dictionaries of pre- and post-synaptic neurons (keys: indices, values: neuron objects)
  pre_synaptic_neurons = {0: neuron0, 1: neuron1}  
  post_synaptic_neurons = {0: neuron2, 1: neuron3}

  # Example parameters (ensure these match the expected format of your model)
  params = {
      "tau_recovery": [0.2, 1.0],
      "tau_facilitation": [0.05, 0.5],
      "u_max": [0.1, 0.7],
      "u": [0.1],
      "e": [0, 0],
      "g_max": [0.1, 1.0],
      "g_syn": 0
  }
  dt = 0.01

  # Create a Tsodyks-Markram synapse instance NEURONS MUST BE OF TYPE NEURON AS I_SYN
  #IS UPDATED WITHIN THE TSODYKS CLASS
  syn = tsodyks_markram_synapse(pre_synaptic_neurons, post_synaptic_neurons, params, dt)

  # Update the synapse state (this will update r, u, and compute g_syn)
  syn.update()

  # Print the synaptic conductance
  print("Synaptic conductance:", syn.g_syn)

```


## Synapse Generator

The synapse generator uses computational geometry to model connectivity:

- **Polygon Generation:**  
  Neuronal projections (axonal and dendritic) are modeled as semicircular polygons.
  
- **Connectivity Determination:**  
  Overlap area between an axonal polygon from one neuron and a dendritic polygon from another produces the probabability of a synaptic connection (a `connection` object). Nested intersections are computed recursively.
  
- **Duplicate Removal:**  
  Duplicate connection objects are removed using the `remove_duplicate_intersections` function.
### Neuron Polygon Equations

For each neuron $i$ with soma at

$$
S_i = (x_i, y_i)
$$

the axon and dendrite fields are defined as semicircular polygons.

**Axon Polygon $A_i$:**

$$
A_i = { (x_i + r_a \cos \theta,\; y_i + r_a \sin \theta) \,\bigg|\, \theta \in [\theta_{a1,i},\, \theta_{a2,i}] } \cup \{(x_i, y_i)\}
$$

**Dendrite Polygon $D_i$:**

$$
D_i = { (x_i + r_d \cos \theta,\; y_i + r_d \sin \theta) \,\bigg|\, \theta \in [\theta_{d1,i},\, \theta_{d2,i}] } \cup \{(x_i, y_i)\}
$$

### Synapse Generator Equations

For neurons $i$ and $j$ ($i \neq j$), define the intersection of the axon polygon of neuron $i$ and the dendrite polygon of neuron $j$ as:

$$
I_{ij} = A_i \cap D_j
$$

The area of the intersection is given by:

$$
A_{ij} = \int_{I_{ij}} dA
$$

The centroid of the intersection, representing the synapse's location, is:

$$
C_{ij} = \left( \frac{1}{A_{ij}} \int_{I_{ij}} x\, dA,\quad \frac{1}{A_{ij}} \int_{I_{ij}} y\, dA \right)
$$

The probability of forming a synaptic connection is then modeled as:

$$
P_{ij} = 1 - \exp\left(-\alpha \, A_{ij}\right)
$$

where:
- $\alpha$ is a scaling parameter determining how rapidly the connection probability approaches 1 as the overlap area increases.

### Recursive Intersection Analysis (Optional)

For cases where further refinement is desired, an existing intersection $I$ may be intersected with an additional neuron polygon $P_k$ (with $k \notin \{i,j\}$):

$$
I' = I \cap P_k
$$

provided that $I' \neq \varnothing$. Duplicate connections, identified by identical sets of pre-synaptic and post-synaptic neurons, can then be merged.

**Usage Example:**  
```python
        import numpy as np
        from shapely.geometry import Point
        # Adjust import based on any project structure changes
        from synapse_generator import create_synapses  

        num_neurons = 5
        max_size = 5
        soma_x = np.random.rand(num_neurons) * max_size
        soma_y = np.random.rand(num_neurons) * max_size
        soma_points = [Point(soma_x[i], soma_y[i]) for i in range(num_neurons)]

        synapses = create_synapses(soma_points)
        print("Generated", len(synapses), "synapses")
```