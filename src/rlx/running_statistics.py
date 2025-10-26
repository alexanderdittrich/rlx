"""
Running Statistics for Observation Normalization

Implements Welford's online algorithm for tracking mean and standard deviation
of observations during training. Critical for continuous control tasks.

Based on Brax implementation:
https://github.com/google/brax/blob/main/brax/training/acme/running_statistics.py
"""

from typing import NamedTuple
import jax
import jax.numpy as jnp


class RunningStatisticsState(NamedTuple):
    """State for tracking running statistics."""

    count: jax.Array  # Number of samples seen
    mean: jax.Array  # Running mean
    summed_variance: jax.Array  # Sum of squared deviations
    std: jax.Array  # Running standard deviation


def init_state(shape: tuple[int, ...]) -> RunningStatisticsState:
    """Initialize running statistics state.

    Args:
        shape: Shape of observations (e.g., (obs_size,))

    Returns:
        Initialized statistics state
    """
    dtype = jnp.float32
    return RunningStatisticsState(
        count=jnp.zeros((), dtype=dtype),
        mean=jnp.zeros(shape, dtype=dtype),
        summed_variance=jnp.zeros(shape, dtype=dtype),
        std=jnp.ones(shape, dtype=dtype),  # Initialize with ones to avoid div by zero
    )


def update(
    state: RunningStatisticsState,
    batch: jax.Array,
    std_min_value: float = 1e-6,
    std_max_value: float = 1e6,
) -> RunningStatisticsState:
    """Update running statistics with a new batch of data.

    Uses Welford's online algorithm for numerical stability.

    Args:
        state: Current statistics state
        batch: New data batch with shape [batch_size, ...]
        std_min_value: Minimum std value (for numerical stability)
        std_max_value: Maximum std value (for numerical stability)

    Returns:
        Updated statistics state
    """
    # Flatten batch dimensions: [batch_size, ...] -> [batch_size]
    batch_shape = batch.shape[0]
    batch_count = float(batch_shape)

    # Update count
    new_count = state.count + batch_count

    # Update mean using Welford's algorithm
    diff_to_old_mean = batch - state.mean
    mean_diff = jnp.sum(diff_to_old_mean, axis=0) / new_count
    new_mean = state.mean + mean_diff

    # Update summed variance
    diff_to_new_mean = batch - new_mean
    new_summed_variance = state.summed_variance + jnp.sum(
        diff_to_old_mean * diff_to_new_mean, axis=0
    )

    # Compute standard deviation
    # Clip to avoid negative values due to numerical errors
    new_summed_variance = jnp.maximum(new_summed_variance, 0.0)
    new_std = jnp.sqrt(new_summed_variance / new_count)
    new_std = jnp.clip(new_std, std_min_value, std_max_value)

    return RunningStatisticsState(
        count=new_count,
        mean=new_mean,
        summed_variance=new_summed_variance,
        std=new_std,
    )


def normalize(
    batch: jax.Array,
    state: RunningStatisticsState,
    max_abs_value: float = 5.0,
) -> jax.Array:
    """Normalize data using running statistics.

    Args:
        batch: Data to normalize
        state: Current statistics state
        max_abs_value: Clip normalized values to [-max, +max]

    Returns:
        Normalized data
    """
    normalized = (batch - state.mean) / state.std
    if max_abs_value is not None:
        normalized = jnp.clip(normalized, -max_abs_value, max_abs_value)
    return normalized
