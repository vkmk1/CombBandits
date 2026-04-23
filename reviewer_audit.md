# Reviewer Audit: Holes, Overfitting Risks, and Fixes

## CRITICAL — Will Reject the Paper

### 1. The paper still presents LLM-CUCB-AT as the main algorithm, but Pool-CTS won

**Problem**: The workshop_paper.tex (lines 84-93, 134-156) presents LLM-CUCB-AT as the contribution. But our own arena shows LLM-CUCB-AT is SECOND-TO-LAST (28th out of 29), worse than vanilla CUCB. The paper's Table 1 (line 233) shows LLM-CUCB-AT getting 24,132 regret vs CUCB's 7,242. The paper tries to spin this as "3-5x better than LLM-Greedy" but a reviewer will immediately note it's 3x WORSE than CUCB.

**Fix**: Complete rewrite to present Pool-CTS as the main algorithm. LLM-CUCB-AT becomes the motivating negative example (Section 4: "The Consistency Trap"). The story becomes: "naive trust fails → we identify why → pool-based distillation fixes it."

**Severity**: Paper-killing. No reviewer will accept an algorithm demonstrated to be worse than the baseline.

### 2. Corollary 1(iv) claims LLM-CUCB-AT handles consistent_wrong — our data disproves this

**Problem**: Line 182 states "posterior validation detects the mismatch; the algorithm does not suffer linear regret." But the arena shows LLM-CUCB-AT gets 1,774 regret on consistent_wrong (comparable to CUCB's 1,588), meaning it DOES suffer near-linear regret at T=3000. At T=30,000 the original Table 1 shows 23,420 — still catastrophic.

**Fix**: Remove this corollary. Replace with our Theorem 1 (Consistency Trap lower bound) showing this failure is FUNDAMENTAL, not a hyperparameter issue. This is actually a stronger contribution.

**Severity**: Paper-killing. A false claim caught by any reviewer who reads the experiments.

### 3. No real LLM experiments

**Problem**: Lines 322-334 show exp9 (real GPT-4o oracle) is still commented out / pending. All experiments use a simulated coin-flip oracle. Reviewer will ask: "What evidence do you have that any of this transfers to real LLMs?" The simulated oracle is a parameterized random variable — it has no structure, no reasoning, no context-dependence.

**Fix**: Either (a) run real LLM experiments before submission, or (b) explicitly frame this as a theoretical/methodological contribution with simulated validation, acknowledging the limitation prominently. For workshop papers, (b) is acceptable if done honestly.

**Severity**: Major weakness. Won't reject alone at a workshop, but will lose 1-2 points.

---

## HIGH — Reviewer Will Flag, May Reduce Score

### 4. Overfitting risk: same master_seed=42 across all arena runs

**Problem**: generate_random_configs uses master_seed=42 (line 67). Every arena run generates the EXACT SAME 30 configs. Our algorithms were designed, tuned, and selected based on performance on these same 30 configs. This is textbook overfitting to the test set.

**Evidence of overfitting**: We ran 5 loops of "design algo → test on arena → design better algo." Each loop's winner was selected BECAUSE it did well on these specific 30 configs. A reviewer will correctly argue that our "67% win rate" may not hold on different configs.

**Fix**: 
- Run final evaluation on a HELD-OUT set of configs with a different master_seed (e.g., seed=2024)
- Report results on BOTH the development set (seed=42) and the held-out set (seed=2024)  
- If results hold, this is strong evidence against overfitting
- If results don't hold, we have a problem and need to understand why

### 5. T=3000 is too short for meaningful bandit regret comparison

**Problem**: At T=3000 with d=150, m=15 and delta_min=0.03, the theoretically required exploration is O(d/Delta^2) ≈ 150/0.0009 ≈ 167,000 rounds. We're testing at 1.8% of the required horizon. Many algorithms haven't even finished their initial exploration phase.

**Impact**: Algorithms that explore less (like adaptive_pool_cts which starts with a small pool) look artificially good because they exploit early. Algorithms that explore more (like pool_cts_ic with round-robin warmup) look artificially bad.

**Fix**:
- Run at least one arena at T=10,000-30,000 (even if fewer configs)
- Include a horizon sensitivity analysis: plot final regret vs T for top algorithms
- Report whether the ranking changes at longer horizons

### 6. Simulated oracle doesn't capture real LLM behavior

**Problem**: Our BatchedSimulatedCLO (lines 42-82 of batched_oracle.py) is a simple coin-flip: with probability epsilon, return bad set; otherwise return optimal set. Real LLMs exhibit:
- Context-dependent errors (harder questions get more wrong)
- Correlated errors across queries (same prompt → same mistake)
- Non-stationary quality (model updates, prompt sensitivity)
- Partial knowledge (knows some arms are good but not which are best)
- Temperature-dependent consistency (deterministic at temp=0, random at temp=1)

**Impact**: Our "consistent_wrong" model returns the WORST possible set every time. No real LLM does this. Real consistent errors are more like "systematically overweights popular items" — a partial overlap corruption, not a full adversarial attack.

**Fix**:
- Add a "biased oracle" corruption type that returns a set with partial overlap plus systematic bias toward certain arm features
- Add a "correlated noise" oracle where errors are correlated across queries
- Discuss the gap between simulated and real oracle behavior explicitly
- The consistent_wrong scenario should be framed as a worst-case theoretical benchmark, not a realistic scenario

### 7. No statistical significance testing across configs

**Problem**: We report mean regret across 30 configs, but don't report confidence intervals or significance tests for the rankings. With 30 configs and high variance (std often exceeds mean), the ordering of algorithms with similar means is not statistically reliable.

**Fix**:
- Add bootstrap confidence intervals for the mean ranking
- Report paired Wilcoxon signed-rank tests between adjacent algorithms in the ranking
- Note which differences are significant and which are within noise

### 8. Pool-CTS theory needs actual proofs, not just sketches

**Problem**: The paper_outline.md and theory_draft.md contain proof sketches, not complete proofs. The main theorem references (Theorem 3: Pool-CTS under reliable oracle) are stated informally. For a workshop paper, sketches may suffice, but reviewers will probe:
- Is the pool coverage guarantee (Theorem 4) actually proven? The claim "P ⊇ S* with probability ≥ 1 − exp(−K δ² / m)" needs a formal proof via Hoeffding/multiplicative Chernoff.
- The CTS convergence inside the pool inherits from Wang-Chen 2018, but the restricted action space changes the problem. Is the inheritance valid?

**Fix**: Write complete proofs for the two key claims:
1. Pool coverage: K oracle queries with accuracy p guarantee coverage
2. Regret within pool: CTS on a pool of size βm converges at O((βm)² log T / Δ)

---

## MEDIUM — Reviewer May Comment

### 9. The "action-space corruption" framing conflates two different things

**Problem**: We define action-space corruption as "oracle corrupts the action proposal." But in Pool-CTS, the oracle's action proposal is distilled into a pool ONCE (or O(log T) times with doubling), not used per-round. So the corruption model doesn't directly apply to Pool-CTS — it applies to LLM-CUCB-AT. For Pool-CTS, the relevant quantity is "pool coverage probability," not "per-round corruption rate."

**Fix**: Clarify the corruption model for pool-based algorithms. Define:
- Per-round corruption rate ε (for LLM-CUCB-AT style algorithms)
- Pool coverage probability η(K,ε) = P(S* ⊆ P | K queries at corruption ε) (for Pool-CTS style)

### 10. Arena doesn't test structured/non-uniform action spaces

**Problem**: All configs use flat Bernoulli arms with m-subset constraints. Real combinatorial bandits have structure: matroid constraints, path constraints, matching constraints. Our algorithms may not generalize.

**Fix**: Add at least one structured test (e.g., matroid constraint or graph-based action space) to the arena, or explicitly state this as a limitation.

### 11. The "consistent_wrong" oracle model uses the WORST-case bad set

**Problem**: In batched_oracle.py line 38-40, the "adversarial" bad set is constructed by sorting arms and taking the m WORST arms. But for "consistent_wrong" (line 78-80), it uses self._bad_set which was constructed as the m BEST non-optimal arms (line 38-40, descending=True). This means the consistent_wrong oracle returns arms that are plausible (high mean, just not the best) — that's correct and realistic. BUT: different algorithms might react differently if the bad set were random or adversarially chosen.

**Fix**: Test with both "consistent_wrong_plausible" (current: top non-optimal) and "consistent_wrong_adversarial" (worst arms) to show robustness.

### 12. No ablation of key hyperparameters in the arena

**Problem**: Pool-CTS algorithms have several hyperparameters: beta (pool size multiplier), n_pool_rounds, n_safety, epoch_base, abstain_threshold, etc. We used fixed defaults (beta=3, n_pool_rounds=10, n_safety=5) across all configs. A reviewer will ask: "How sensitive are your results to these choices?"

**Fix**: Run a hyperparameter sensitivity analysis on 2-3 key parameters (beta, n_pool_rounds) across a few representative configs. Show that results are robust within reasonable ranges.

### 13. The arena generates configs with equal probability for each corruption type

**Problem**: 25% of configs are consistent_wrong. In reality, consistent_wrong is an extreme scenario. If we weighted corruption types by real-world frequency (e.g., 50% partial_overlap, 30% uniform, 15% adversarial, 5% consistent_wrong), the rankings would change dramatically — algorithms that are bad on consistent_wrong but great everywhere else would rise.

**Fix**: Report results with both uniform weighting (current) and a "realistic" weighting. Show rankings under both.

---

## SPECIFIC OVERFITTING CONCERNS

### A. Algorithm design overfitting
Each loop's winners were designed based on looking at the previous loop's results on the SAME 30 configs. Example: "adaptive_pool_cts wins on adversarial/partial/uniform" → "add dual-pool to fix consistent_wrong" → "adaptive_pool_dual." This is iterative optimization on the test set.

**Mitigation**: Held-out evaluation (see Fix #4).

### B. Hyperparameter overfitting
Defaults like beta=3, epoch_base=50, etc. were likely influenced by seeing the results. Even if we didn't explicitly tune them, our "intuition" about good values was shaped by prior runs.

**Mitigation**: Show robustness to hyperparameter perturbation (see Fix #12).

### C. Config distribution overfitting
Our configs use d ∈ {30,50,100,150}. Real problems might have d=1000 or d=10. Our algorithms might not scale.

**Mitigation**: Test at d=200-500 even if just a few configs.

### D. Metric overfitting  
We optimized for mean regret. But a reviewer might care about: median, worst-case, variance, time to convergence, oracle query count, etc. An algorithm optimized for mean might trade off these.

**Mitigation**: Already reporting multiple metrics (good). Emphasize the Pareto frontier, not just the mean winner.
