import numpy as np
from .generators import BaseGenerator


class RegimeSwitchGenerator(BaseGenerator):
    def __init__(self, generators, switch_times):
        """
        Parameters:
        - generators: list of BaseGenerator instances
        - switch_times: list of time steps when the regime changes
        """
        self.generators = generators
        self.switch_times = switch_times
        self.current_step = 0
        self.current_gen_idx = 0

    def generate_value(self, last_value=None):
        if self.current_step in self.switch_times:
            self.current_gen_idx = (self.current_gen_idx + 1) % len(self.generators)
        self.current_step += 1
        return self.generators[self.current_gen_idx].generate_value(last_value)

    def reset(self):
        self.current_step = 0
        self.current_gen_idx = 0
        for g in self.generators:
            g.reset()



class MarkovSwitchGenerator(BaseGenerator):
    def __init__(self, generators, transition_matrix, initial_state=0):
        """
        A meta-generator that switches between base generators using a Markov chain.

        Parameters:
        - generators: list of BaseGenerator instances (one per Markov state)
        - transition_matrix: 2D list or array of shape (n, n), where transition_matrix[i] gives
                             the probability distribution over next states when in state i
        - initial_state: index of the generator to start from
        """
        self.generators = generators
        self.transition_matrix = transition_matrix
        self.current_state = initial_state

    def generate_value(self, last_value=None):
        """
        Generates the next value using the current generator,
        then transitions to the next generator based on the current state's probabilities.
        """
        # Step 1: generate value using the current state's generator
        value = self.generators[self.current_state].generate_value(last_value)

        # Step 2: select the next generator index using the transition probabilities
        probs = self.transition_matrix[self.current_state]
        self.current_state = np.random.choice(len(self.generators), p=probs)

        return value

    def reset(self):
        """
        Resets all internal generators and returns to the initial state (state 0).
        """
        self.current_state = 0
        for g in self.generators:
            g.reset()
