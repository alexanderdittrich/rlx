"""Test Generalized Advantage Estimation (GAE) implementation."""

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rlx.ppo_gymnasium import compute_gae


def test_gae_correctness():
    """Test that GAE computation produces correct results."""
    # Setup test data
    num_steps, num_envs = 128, 4
    gamma, gae_lambda = 0.99, 0.95

    np.random.seed(42)
    rewards = jnp.array(np.random.randn(num_steps, num_envs).astype(np.float32))
    values = jnp.array(np.random.randn(num_steps, num_envs).astype(np.float32))
    dones = jnp.array((np.random.rand(num_steps, num_envs) > 0.95).astype(np.float32))
    next_value = jnp.array(np.random.randn(num_envs).astype(np.float32))
    next_done = jnp.array((np.random.rand(num_envs) > 0.95).astype(np.float32))

    # Compute GAE
    advantages, returns = compute_gae(
        rewards, values, dones, next_value, next_done, gamma, gae_lambda
    )

    # Validate output shapes
    assert advantages.shape == (num_steps, num_envs), "Advantages shape mismatch"
    assert returns.shape == (num_steps, num_envs), "Returns shape mismatch"

    # Validate no NaNs or Infs
    assert not jnp.isnan(advantages).any(), "Advantages contain NaNs"
    assert not jnp.isinf(advantages).any(), "Advantages contain Infs"

    # Validate returns = advantages + values
    assert jnp.allclose(returns, advantages + values), "Returns != Advantages + Values"

    print("✓ GAE correctness test passed")


def test_gae_deterministic():
    """Test that GAE computation is deterministic."""
    num_steps, num_envs = 64, 2
    gamma, gae_lambda = 0.99, 0.95

    np.random.seed(123)
    rewards = jnp.array(np.random.randn(num_steps, num_envs).astype(np.float32))
    values = jnp.array(np.random.randn(num_steps, num_envs).astype(np.float32))
    dones = jnp.array((np.random.rand(num_steps, num_envs) > 0.95).astype(np.float32))
    next_value = jnp.array(np.random.randn(num_envs).astype(np.float32))
    next_done = jnp.array((np.random.rand(num_envs) > 0.95).astype(np.float32))

    # Compute twice
    adv1, ret1 = compute_gae(
        rewards, values, dones, next_value, next_done, gamma, gae_lambda
    )
    adv2, ret2 = compute_gae(
        rewards, values, dones, next_value, next_done, gamma, gae_lambda
    )

    # Should be identical
    assert jnp.allclose(adv1, adv2), "GAE computation is not deterministic"
    assert jnp.allclose(ret1, ret2), "Returns computation is not deterministic"

    print("✓ GAE determinism test passed")


def test_gae_edge_cases():
    """Test GAE with edge cases."""
    num_envs = 2
    gamma, gae_lambda = 0.99, 0.95

    # Single timestep
    rewards = jnp.ones((1, num_envs))
    values = jnp.zeros((1, num_envs))
    dones = jnp.zeros((1, num_envs))
    next_value = jnp.ones(num_envs)
    next_done = jnp.zeros(num_envs)

    advantages, returns = compute_gae(
        rewards, values, dones, next_value, next_done, gamma, gae_lambda
    )

    assert advantages.shape == (1, num_envs), "Single timestep shape mismatch"
    assert not jnp.isnan(advantages).any(), "Single timestep produces NaNs"

    # All episodes done
    rewards = jnp.ones((10, num_envs))
    values = jnp.zeros((10, num_envs))
    dones = jnp.ones((10, num_envs))
    next_value = jnp.zeros(num_envs)
    next_done = jnp.ones(num_envs)

    advantages, returns = compute_gae(
        rewards, values, dones, next_value, next_done, gamma, gae_lambda
    )

    assert not jnp.isnan(advantages).any(), "All-done case produces NaNs"

    print("✓ GAE edge cases test passed")


def compute_gae_loop(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    next_value: jax.Array,
    next_done: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Reference GAE implementation using Python loop (slower)."""
    num_steps, num_envs = rewards.shape
    advantages = jnp.zeros_like(rewards)
    lastgaelam = jnp.zeros(num_envs)

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]

        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages = advantages.at[t].set(
            delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        )
        lastgaelam = advantages[t]

    returns = advantages + values
    return advantages, returns


def test_gae_performance():
    """Compare optimized GAE vs Python loop implementation."""
    num_steps, num_envs = 2048, 64
    gamma, gae_lambda = 0.99, 0.95

    np.random.seed(42)
    rewards = jnp.array(np.random.randn(num_steps, num_envs).astype(np.float32))
    values = jnp.array(np.random.randn(num_steps, num_envs).astype(np.float32))
    dones = jnp.array((np.random.rand(num_steps, num_envs) > 0.95).astype(np.float32))
    next_value = jnp.array(np.random.randn(num_envs).astype(np.float32))
    next_done = jnp.array((np.random.rand(num_envs) > 0.95).astype(np.float32))

    # Warmup and compile both versions
    _ = compute_gae(rewards, values, dones, next_value, next_done, gamma, gae_lambda)
    _ = compute_gae_loop(
        rewards, values, dones, next_value, next_done, gamma, gae_lambda
    )

    # Benchmark optimized version
    start = time.time()
    for _ in range(10):
        adv_opt, ret_opt = compute_gae(
            rewards, values, dones, next_value, next_done, gamma, gae_lambda
        )
        adv_opt.block_until_ready()
    time_opt = (time.time() - start) / 10

    # Benchmark loop version
    start = time.time()
    for _ in range(10):
        adv_loop, ret_loop = compute_gae_loop(
            rewards, values, dones, next_value, next_done, gamma, gae_lambda
        )
        adv_loop.block_until_ready()
    time_loop = (time.time() - start) / 10

    # Verify results match
    adv_match = jnp.allclose(adv_opt, adv_loop, rtol=1e-5)
    ret_match = jnp.allclose(ret_opt, ret_loop, rtol=1e-5)

    assert adv_match, "Optimized and loop versions produce different advantages"
    assert ret_match, "Optimized and loop versions produce different returns"

    # Calculate max difference
    max_adv_diff = jnp.abs(adv_opt - adv_loop).max()
    max_ret_diff = jnp.abs(ret_opt - ret_loop).max()

    speedup = time_loop / time_opt
    print(f"✓ GAE performance test passed")
    print(f"  Optimized (scan): {time_opt * 1000:.1f}ms")
    print(f"  Loop version:     {time_loop * 1000:.1f}ms")
    print(f"  Speedup:          {speedup:.1f}x faster")
    print(
        f"  Results match:    max_diff={max_adv_diff:.2e} (advantages), {max_ret_diff:.2e} (returns)"
    )


if __name__ == "__main__":
    print("Running GAE tests...")
    print()

    test_gae_correctness()
    test_gae_deterministic()
    test_gae_edge_cases()
    test_gae_performance()

    print()
    print("All tests passed! ✓")
