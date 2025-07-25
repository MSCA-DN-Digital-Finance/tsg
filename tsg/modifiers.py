from .generators import BaseGenerator
import numpy as np

# === Wrappers ===
class GaussianNoise(BaseGenerator):
    def __init__(self, generator, mu=0.0, sigma=0.2):
        """
        Wraps any generator and adds Gaussian noise to its output.
        """
        self.generator = generator
        self.mu = mu
        self.sigma = sigma

    def generate_value(self, last_value):
        base_value = self.generator.generate_value(last_value)
        noise = np.random.normal(self.mu, self.sigma)
        return base_value + noise

    def reset(self):
        self.generator.reset()


class PoissonNoiseModifier(BaseGenerator):
    def __init__(self, generator, lam=1.0, direction='positive'):
        """
        Adds Poisson-distributed noise at each time step.

        Parameters:
        - generator: the base generator to wrap
        - lam: lambda parameter of the Poisson distribution (controls intensity)
        - direction: 'positive', 'negative', or 'both' (default 'positive')
        """
        assert direction in {'positive', 'negative', 'both'}, "Invalid direction"
        self.generator = generator
        self.lam = lam
        self.direction = direction

    def generate_value(self, last_value):
        base = self.generator.generate_value(last_value)
        jump = np.random.poisson(self.lam)

        if self.direction == 'positive':
            return base + jump
        elif self.direction == 'negative':
            return base - jump
        elif self.direction == 'both':
            sign = np.random.choice([-1, 1])
            return base + sign * jump

    def reset(self):
        self.generator.reset()


class CompoundPoissonJumpModifier(BaseGenerator):
    def __init__(self, generator, lam=3, T=100, jump_size=1.0, direction='positive'):
        """
        Adds jumps at randomly selected steps using a compound Poisson process.
        A fixed number of Poisson-distributed events are assigned at random time steps.

        Parameters:
        - generator: base generator to wrap
        - lam: expected number of jumps over T steps (λ)
        - T: total number of steps (must be set at init)
        - jump_size: fixed magnitude of each jump
        - direction: 'positive', 'negative', or 'both'
        """
        assert direction in {'positive', 'negative', 'both'}, "Invalid direction"
        self.generator = generator
        self.lam = lam
        self.T = T
        self.jump_size = jump_size
        self.direction = direction

        self.reset()

    def reset(self):
        self.generator.reset()
        # Sample number of jumps: N ~ Poisson(λ)
        self.jump_times = set(np.random.choice(
            self.T,
            size=np.random.poisson(self.lam),
            replace=False
        )) if self.lam > 0 else set()
        self.t = 0  # internal time counter

    def generate_value(self, last_value):
        base = self.generator.generate_value(last_value)

        if self.t in self.jump_times:
            if self.direction == 'positive':
                jump = self.jump_size
            elif self.direction == 'negative':
                jump = -self.jump_size
            else:  # 'both'
                jump = np.random.choice([-1, 1]) * self.jump_size
        else:
            jump = 0.0

        self.t += 1
        return base + jump

