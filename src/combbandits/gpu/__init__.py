"""GPU-accelerated batched execution for combinatorial semi-bandits.

Runs all seeds simultaneously on GPU tensors. Agent state is (n_seeds, d),
arm selection is (n_seeds, m), rewards are sampled in parallel.
"""
