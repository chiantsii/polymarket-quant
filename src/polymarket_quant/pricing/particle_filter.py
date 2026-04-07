"""Sequential latent-probability estimation with a particle filter."""

from dataclasses import dataclass

import numpy as np

from polymarket_quant.utils.math import logit, sigmoid


@dataclass(frozen=True)
class ParticleFilterResult:
    """Output path from a latent probability particle filter."""

    probabilities: np.ndarray
    effective_sample_sizes: np.ndarray
    final_particles: np.ndarray
    final_weights: np.ndarray


class ParticleFilter:
    """Estimate a latent binary-event probability from sequential observations."""

    def __init__(
        self,
        n_particles: int = 1_000,
        transition_std: float = 0.05,
        observation_std: float = 0.03,
        seed: int | None = None,
    ) -> None:
        if n_particles <= 0:
            raise ValueError("n_particles must be positive")
        if transition_std < 0:
            raise ValueError("transition_std must be non-negative")
        if observation_std <= 0:
            raise ValueError("observation_std must be positive")

        self.n_particles = n_particles
        self.transition_std = transition_std
        self.observation_std = observation_std
        self.rng = np.random.default_rng(seed)

    def filter(self, observations: np.ndarray, initial_probability: float = 0.5) -> ParticleFilterResult:
        """Filter latent probabilities from observed market-implied prices."""
        observations = np.asarray(observations, dtype=float)
        if observations.ndim != 1:
            raise ValueError("observations must be a one-dimensional array")
        if not 0.0 < initial_probability < 1.0:
            raise ValueError("initial_probability must be strictly between 0 and 1")

        latent_logits = np.full(self.n_particles, logit(initial_probability), dtype=float)
        weights = np.full(self.n_particles, 1.0 / self.n_particles, dtype=float)
        filtered_probabilities = []
        effective_sample_sizes = []

        for observation in np.clip(observations, 1e-6, 1 - 1e-6):
            latent_logits += self.rng.normal(0.0, self.transition_std, size=self.n_particles)
            particles = sigmoid(latent_logits)
            weights *= self._observation_likelihood(observation, particles)
            weights = self._normalize(weights)

            ess = 1.0 / np.sum(weights**2)
            if ess < self.n_particles / 2:
                latent_logits = self._resample(latent_logits, weights)
                weights = np.full(self.n_particles, 1.0 / self.n_particles, dtype=float)

            filtered_probabilities.append(float(np.average(sigmoid(latent_logits), weights=weights)))
            effective_sample_sizes.append(float(ess))

        return ParticleFilterResult(
            probabilities=np.asarray(filtered_probabilities),
            effective_sample_sizes=np.asarray(effective_sample_sizes),
            final_particles=sigmoid(latent_logits),
            final_weights=weights,
        )

    def _observation_likelihood(self, observation: float, particles: np.ndarray) -> np.ndarray:
        residual = observation - particles
        return np.exp(-0.5 * (residual / self.observation_std) ** 2)

    def _normalize(self, weights: np.ndarray) -> np.ndarray:
        total = np.sum(weights)
        if not np.isfinite(total) or total == 0:
            return np.full(self.n_particles, 1.0 / self.n_particles, dtype=float)
        return weights / total

    def _resample(self, latent_logits: np.ndarray, weights: np.ndarray) -> np.ndarray:
        indices = self.rng.choice(self.n_particles, size=self.n_particles, replace=True, p=weights)
        return latent_logits[indices]
