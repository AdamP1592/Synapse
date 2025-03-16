class tsodyks_markram_synapse:
    """
    Implements a Tsodyks-Markram dynamic synapse model.
    
    This model simulates short-term plasticity by tracking variables (r, u)
    that determine the effective synaptic conductance. The model uses a
    set of pre-synaptic neurons and post-synaptic neurons, and stores connection
    indices for later reconstruction.
    
    Attributes:
        x, y: Coordinates for the synapse (typically the synaptic contact point).
        r, r_past: Variables representing the available synaptic resources.
        u, u_past: Variables representing the utilization factor.
        t: Current time.
        dt: Time step.
        pre_indicies: List of pre-synaptic neuron indices.
        post_indicies: List of post-synaptic neuron indices.
        pre_synaptic_neurons: List of pre-synaptic neuron objects.
        post_synaptic_neurons: List of post-synaptic neuron objects.
        past_spike_times: List storing the last spike time for each pre-synaptic neuron.
        is_active: List indicating whether each pre-synaptic neuron is currently active.
    """
    def __init__(self, pre_synaptic_neurons, post_synaptic_neurons, params, dt=0.01):
        """
        Initializes the synapse with pre- and post-synaptic neurons, activation parameters,
        and a time step.
        
        :param pre_synaptic_neurons: Dictionary of pre-synaptic neuron objects (keyed by index).
        :param post_synaptic_neurons: Dictionary of post-synaptic neuron objects (keyed by index).
        :param params: Dictionary containing activation parameters.
        :param dt: Time step for the simulation.
        """
        self.x, self.y = 0, 0
        self.r = 1
        self.r_past = 1
        self.u = 0
        self.u_past = 0
        self.t = 0
        self.dt = dt

        # Holders for connection indices and neuron objects.
        self.pre_indicies = []
        self.post_indicies = []
        self.pre_synaptic_neurons = []
        self.post_synaptic_neurons = []

        # Set up the connections using the provided neuron dictionaries.
        self.setup_connections(pre_synaptic_neurons, post_synaptic_neurons)

        # Initialize past spike times and active flags for each pre-synaptic neuron.
        self.past_spike_times = [None for _ in range(len(self.pre_synaptic_neurons))]
        self.is_active = [False for _ in range(len(self.pre_synaptic_neurons))]

        # Set up activation parameters based on the provided params.
        self.setup_activation_params(params)

    def setup_connections(self, pre_synaptic_neurons, post_synaptic_neurons):
        """
        Stores connection indices and neuron objects for both pre- and post-synaptic sides.
        
        :param pre_synaptic_neurons: Dictionary of pre-synaptic neurons.
        :param post_synaptic_neurons: Dictionary of post-synaptic neurons.
        """
        # Iterate over keys in the pre-synaptic dictionary.
        for key in pre_synaptic_neurons:
            self.pre_indicies.append(key)
            self.pre_synaptic_neurons.append(pre_synaptic_neurons[key])
        # Similarly for post-synaptic neurons.
        for key in post_synaptic_neurons:
            self.post_indicies.append(key)
            self.post_synaptic_neurons.append(post_synaptic_neurons[key])

    def get_params(self):
        """
        Constructs and returns a dictionary representation of the synapse's current state.
        
        :return: Dictionary containing state variables, parameters, and connection indices.
        """
        # Determine neurotransmitter type based on reversal potential:
        param_switch = ["nmda", "ampa"]
        syn_params = {
            "state": {
                "r": self.r,
                "r_past": self.r_past,
                "u": self.u,
                "u_past": self.u_past,
                "t": self.t,
                "dt": self.dt
            },
            "params": {
                "g_syn": self.g_syn,
                "g_max": self.g_max,
                "u_max": self.u_max,
                "e": self.reversal_potential,
                "tau_recovery": self.tau_r,
                "tau_facilitation": self.tau_f,
                "x": self.x,
                "y": self.y,
                # If reversal potential > -1, choose index 1 (typically AMPA), else NMDA.
                "neurotransmitterType": param_switch[1 if self.reversal_potential > -1 else 0]
            },
            "connections": {
                "pre": self.pre_indicies,
                "post": self.post_indicies
            }
        }
        return syn_params

    def setup_old_activation_params(self, params):
        """
        Sets activation parameters from a parameter dictionary that is not nested.
        
        :param params: Dictionary with keys for tau_recovery, tau_facilitation, u_max, e, g_max, and g_syn.
        """

        
        self.action_potential_thresholds = []
        for neuron in self.pre_synaptic_neurons:
            self.action_potential_thresholds.append(neuron.action_potential_threshold)

        state = params["state"]

        self.r = state["r"]
        self.r_past = state["r_past"]
        self.u = state["u"]
        self.u_past = state["u_past"]

        self.t = state["t"]
        self.dt = state["dt"]

        synapse_params = params["params"]
        
        self.tau_r = synapse_params["tau_recovery"]
        self.tau_f = synapse_params["tau_facilitation"]
        self.u_max = synapse_params["u_max"]
        self.reversal_potential = synapse_params["e"]
        self.g_max = synapse_params["g_max"]
        self.g_syn = synapse_params["g_syn"]
        self.x = synapse_params["x"]
        self.y = synapse_params["y"]
    
    def setup_activation_params(self, params):
        """
        Configures the synaptic activation parameters based on the provided dictionary.
        
        If the parameters are provided in a nested list format, extracts the proper values.
        
        :param params: Dictionary of activation parameters.
        """
        # If params["tau_recovery"] is not a list, assume an older format.
        if type(params["tau_recovery"]) != list:
            self.setup_old_activation_params(params)
            return

        self.action_potential_thresholds = []
        for neuron in self.pre_synaptic_neurons:
            self.action_potential_thresholds.append(neuron.action_potential_threshold)

        # Assume params contain lists and take the first/second element as needed.
        self.tau_r = params["tau_recovery"][0]
        self.tau_f = params["tau_facilitation"][1]
        self.u_max = params["u_max"][0]
        self.u = params["u"][0]
        self.reversal_potential = params["e"][1]
        self.g_max = params["g_max"][1]
        self.g_syn = 0
        self.r = 1

    def update_spike_times(self):
        """
        Checks pre-synaptic neurons for spiking and updates their last spike times.
        
        Implements a basic refractory behavior: a neuron is marked inactive once it drops below threshold.
        """
        for i in range(len(self.pre_synaptic_neurons)):
            neuron = self.pre_synaptic_neurons[i]
            # If a neuron is active and its voltage falls below threshold, mark it inactive.
            if self.is_active[i] and neuron.v <= neuron.action_potential_threshold:
                self.is_active[i] = False
            # If a neuron spikes (voltage exceeds threshold) and was not active, update spike time.
            if neuron.v >= neuron.action_potential_threshold and not self.is_active[i]:
                self.is_active[i] = True
                self.past_spike_times[i] = self.t

    def set_state(self, state):
        """
        Sets the synapse state from a provided state dictionary.
        
        :param state: Dictionary with keys 'r', 'r_past', 'u', 'u_past', and 't'.
        """
        self.r = state["r"]
        self.r_past = state["r_past"]
        self.u = state["u"]
        self.u_past = state["u_past"]
        self.t = state["t"]

    def update(self):
        """
        Updates the synapse state:
          - Checks and updates spike times from pre-synaptic neurons.
          - Updates the resource (r) and utilization (u) variables based on spike events or decay.
          - Computes the current synaptic conductance.
          - Applies the synaptic current to post-synaptic neurons.
          - Advances time.
        
        :param t: The current time to update against.
        """
        self.update_spike_times()
        dirac_sum = 0
        has_past_spike = False

        # Update r and u based on pre-synaptic activity.
        for i in range(len(self.pre_synaptic_neurons)):
            if self.is_active[i]:
                if self.t == self.past_spike_times[i]:
                    has_past_spike = True
                    # Estimate the immediate drop in resources due to a spike.
                    self.r = self.r_past - (self.u_past * self.r_past)
                    # Update utilization.
                    self.u = self.u_past + self.u_max * (1 - self.u_past)

        # If no new spike occurred, update r and u via continuous recovery/facilitation.
        if not has_past_spike:
            drdt = (1 - self.r) / self.tau_r
            dudt = (-self.u / self.tau_f)
            self.r += drdt * self.dt
            self.u += dudt * self.dt

        self.r_past = self.r
        self.u_past = self.u

        # Compute synaptic conductance.
        self.g_syn = self.g_max * self.r * self.u

        # Apply synaptic current to each post-synaptic neuron.
        for neuron in self.post_synaptic_neurons:
            i_syn = self.g_syn * (neuron.v - self.reversal_potential)
            neuron.i_syn += i_syn

        self.t += self.dt

    def __str__(self):
        """
        Returns a string representation of the synapse state.
        """
        return str(self.get_params())


# Optionally, if you wish to keep the synapse_hh class as well:

class synapse_hh:
    """
    A simple Hodgkin-Huxley synapse model.
    
    Note: This is a secondary synapse model and may be used for comparison.
    """
    post_synaptic_neurons = []
    neurotransmitter_release_time = 0
    k = 10

    def __init__(self, pre_synaptic_neuron):
        # Define default synapse parameters.
        synapse_params = [
            [(0.1, 1.0), 1.0, 10.0, -70],
            [(0.1, 1.0), 0.2, 2.0, 0]
        ]
        self.pre_synaptic_neruon = pre_synaptic_neuron
        # Assign tau values from the default AMPA parameters (example).
        self.tau_decay, self.tau_rise = synapse_params[1], synapse_params[2]
        self.neurotransmitter_release_time = None
        self.g_max = [synapse_params[0][1]]

    def add_post_synaptic_neuron(self, post_synaptic_neuron):
        self.post_synaptic_neurons.append(post_synaptic_neuron)

    def add_post_synaptic_neurons(self, post_synaptic_neurons: list):
        for neuron in post_synaptic_neurons:
            self.add_post_synaptic_neuron(neuron)

    def check_for_spike(self):
        """
        Checks if the pre-synaptic neuron has spiked and updates the release time.
        (This is a placeholder for graded neurotransmitter release.)
        """
        from numpy import exp
        neuron = self.pre_synaptic_neruon
        if self.neurotransmitter_release_time is not None:
            self.prev_spike_time = self.neurotransmitter_release_time * (neuron.v <= neuron.action_potential_threshold)
        self.neurotransmitter_release_time += neuron.t * (neuron.v > neuron.action_potential_threshold)

    def update(self, t):
        """
        Updates the synaptic conductance based on the elapsed time since neurotransmitter release.
        
        :param t: The current time.
        """
        import math
        decay = math.e ** ( - (self.neurotransmitter_release_time - t) / self.tau_decay )
        rise = -math.e ** ( - (self.neurotransmitter_release_time - t) / self.tau_rise )
        self.g_syn = self.g_max * (decay + rise)
        for neuron in self.post_synaptic_neurons:
            neuron.i_syn += self.g_syn * (neuron.v - self.e)
