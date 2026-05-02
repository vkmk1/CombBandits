# Sandberg et al. (ICLR 2025) — Bayesian Regret Bound Summary

**Paper:** "Bayesian Analysis of Combinatorial Gaussian Process Bandits"
**Authors:** Jack Sandberg, Niklas Åkerblom, Morteza Haghir Chehreghani
**Venue:** ICLR 2025
**arXiv:** 2312.12676

---

## (a) Exact form of the Bayesian regret bound

Theorem 3.1 (finite arm set) and Theorem 3.2 (infinite / volatile arm set) prove:

**Finite case (Theorem 3.1(iii), combinatorial GP-TS):**
```
BR(T) ≤ π²/3 + 2 √(C_K · T · K · β_T · γ_{T,K})
```

**Infinite / volatile case (Theorem 3.2(iii), combinatorial GP-TS):**
```
BR(T) ≤ 2π²/3 + 2 √(C_K · T · K · β_T · γ_{T,K})
```

where:
- `BR(T)` = Bayesian cumulative regret up to horizon T
- `K` = maximum super-arm size (number of base arms selected per round; our "m")
- `β_T = 2 log(|A| · T² / √(2π))` = confidence parameter, O(log T) in T
- `γ_{T,K}` = maximum information gain from T·K base-arm observations (not T observations
  as in the non-combinatorial setting; the combinatorial structure multiplies the effective
  number of observations by K)
- `C_K` = a constant involving the maximum eigenvalue λ*_K of posterior covariance matrices
  (bounded as long as the kernel is bounded)

The bound collapses to the standard non-combinatorial GP-TS bound when K = 1.
Linear scaling in K appears in the regret term (through γ_{T,K} and the explicit K factor).

**Informal summary for Theorem 1 in our papers:**
For the rank-indexed RBF kernel on K distinct cluster features,
γ_{T,K} = O(K log(1 + T/K)), so substituting gives:
```
BR(T) = Õ(m √(K · T · log T))
```
which is the bound stated in Theorem 1 of our papers (with m = K in their notation).

---

## (b) Assumptions

1. **Kernel type:** General Mercer kernel (the results are stated for GP priors with any
   valid kernel; the paper's examples include RBF/Matérn and finite-dimensional kernels).
   The information-gain quantity γ_{T,K} is kernel-dependent.

2. **Feedback:** **Semi-bandit feedback** — the learner observes the reward of each
   selected base arm individually (not just the sum). This matches our setting exactly.

3. **Action structure:** The learner selects a "super-arm" of K base arms from a (possibly
   infinite) arm set at each round. Actions are sets of size K.

4. **Reward model:** GP prior over base-arm means, bounded rewards assumed for the
   confidence parameter β_T construction (Bernoulli rewards satisfy this).

5. **Volatile arms:** Theorem 3.2 additionally allows the arm set to change between rounds
   (volatile availability). Theorem 3.1 assumes a fixed finite arm set — closer to our setup.

6. **Horizon T:** Results are stated for a fixed horizon T; the bounds hold in terms of T
   and the kernel-dependent information gain γ_{T,K}.

7. **Prior:** Bayesian bound — the regret is measured in expectation over the GP prior on
   arm means, not as a frequentist worst-case bound.

---

## (c) Applicability to reduction from our CorrFull algorithm

**Directly applicable, with caveats:**

- Our reduction (Appendix D in both papers) uses their bound as follows:
  (1) Beta marginals are approximated by Gaussians via Proposition 4 (Berry-Esseen rate).
  (2) The resulting Gaussian sampler with covariance Σ is matched to their combinatorial
      GP-TS model with the same kernel.
  (3) Plugging our γ_T = O(K log(1 + T/K)) into their Theorem 3.1(iii) gives
      BR(T) = Õ(m √(K T log T)).

- **What matches:** Semi-bandit feedback ✓, bounded Bernoulli rewards ✓, fixed arm set ✓,
  super-arm size K = m ✓, RBF kernel on finite feature set ✓.

- **What does NOT match perfectly:**
  - Their bound is for GP-TS with a *true GP prior*; our algorithm uses a Beta-Bernoulli
    posterior approximated as Gaussian. The Gaussian reduction introduces an o(T) error
    (from the Berry-Esseen approximation), which is additive to their bound and does not
    change the leading-order rate.
  - Their K refers to super-arm size (= our m); our K refers to number of clusters.
    In our kernel, the RBF Gram matrix has effective rank equal to the number of distinct
    cluster-rank values, which is our K (number of LLM clusters). Their γ_{T,K} therefore
    becomes γ_{T,m} in their notation, where the information gain is computed for our
    rank-indexed RBF kernel — this is where we apply Srinivas et al. Theorem 5 to get
    γ_T = O(K_clusters · log T), and then substitute into their bound.
  - The constant C_K in their bound depends on the kernel's eigenvalues; for our RBF
    kernel on K_clusters distinct values with max correlation ρ_max, C_K is bounded by
    a function of ρ_max and ℓ (the lengthscale).

- **Bottom line for Theorem 1 rewrite:**
  The theorem as stated in both papers is an informal statement and is broadly correct.
  The formal version should say: "Via the Gaussian reduction of Proposition 4 (o(T) error)
  and the information-gain bound γ_T = O(K log(1 + T/K)) from the rank-indexed RBF
  kernel, substituting into Sandberg et al. (2025) Theorem 3.1 gives BR(T) = Õ(m √(K T log T))."
  The key caveat to add is that this is a Bayesian bound over the GP prior, and the
  Gaussian approximation introduces an additive o(T) lower-order term.
