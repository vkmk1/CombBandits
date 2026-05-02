# Simulated Adversarial Review: "Correlated Thompson Sampling via LLM-Derived Covariance Structure for Combinatorial Semi-Bandits"

*Reviewer persona: senior bandits reviewer, NeurIPS / COLT / ICML. Target venue: ICML 2026 Workshop on Decision-Making from Offline Datasets to Online Adaptation.*

---

## 1. Summary

The paper studies stochastic combinatorial semi-bandits with an LLM oracle. The authors argue that existing LLM-augmented bandits inject the LLM into the *posterior* (priors, pseudo-rewards, recommendations) and that this is fragile because LLM accuracy is endogenous to the bandit's own trajectory. They propose CorrCTS-Full: a single LLM cluster query is converted via an RBF kernel over cluster ranks into a positive-definite sampling covariance, and CTS's independent Beta draws are replaced by Gaussian moment-matched correlated samples; Beta posteriors are never modified. They report 18.9% regret reduction vs CTS on synthetic Bernoulli (n=320), 33.8% on simulated MIND (n=135), an "endogenous oracle" diagnostic, an informal Õ(m√(KT log T)) Bayesian regret theorem, and a credibility-gated variant with O(T/Δ_check) worst-case guarantee.

## 2. Strengths

- **Conceptually clean separation** between belief injection and structure injection. Posterior integrity (Prop. 4.1) is a real safety property and is a non-trivial advantage over warm-start / pseudo-observation methods.
- **The endogenous-oracle observation** (4/5 vs 0/5 with CTS warmup vs round-robin) is genuinely interesting and points at a real gap in the existing exogenous-ε analyses (Cao et al. 2026, Sun et al. 2025).
- **The random-cluster ablation** is the right ablation. A reviewer would have demanded it; the authors ran it preemptively. The block-diagonal-fails-to-beat-RandomCorr result is the kind of decomposition that distinguishes a careful paper from a sloppy one.
- **Computational cost** is honest: one LLM call vs TS-LLM's ~25, and the cost-vs-regret tradeoff is reported.
- **Adversarial-cluster stress test** (16.5% regret *increase* for CorrFull vs 1.8% for RobustCorr) is a credible failure-mode analysis and isn't buried.

## 3. Weaknesses

Ordered by severity.

**W1. Theorem 1 is informally stated and the proof has serious gaps that the paper soft-pedals.** §App. B.1 reduces to GP-TS via Prop. 4.4, claiming that the Beta→Gaussian TV bound O(1/√n_i) accumulates to o(T) regret. This is hand-waved: ∑_t Pr(min_i n_i^(t) < n_0) is *not* small without an explicit lower bound on per-arm pull counts, especially for suboptimal arms that CTS pulls O(log T) times. The bound also requires uniformity in (α, β), but the Berry–Esseen constant for Beta-via-Pólya-urn depends on the skewness of the underlying distribution, which blows up as α or β → 0 (e.g. on Beta(1,1) at t=T_w the bound is vacuous). The proof of Prop. 4.4 invokes Devroye–Györfi (d_TV ≤ 2√d_K) — but Devroye–Györfi requires both densities to be unimodal *and* a common dominating measure; the bound is actually d_TV ≤ √(2 d_K) only for specific families and is often quoted incorrectly. The C in "universal constant" is almost certainly not universal here. **Fix:** State explicitly that C = C(α_0, β_0) for α, β ≥ α_0, β_0, and either prove uniformity or weaken the theorem to "for n_i ≥ n_0".

**W2. The γ_T = O(K log(1+T/K)) step is not what Srinivas 2010 Theorem 5 says.** Srinivas et al.'s bound is for an RBF kernel over a continuous compact subset of ℝ^p and gives γ_T = O((log T)^(p+1)); their finite-rank-kernel result (their Theorem 8) gives γ_T = O(d log T) for a d-rank kernel — *not* a "K-distinct-features" kernel. A d×d matrix Σ = ρ K + (1-ρ)I where K has K distinct rows still has full rank d in general (the additive identity makes it so), so the "effective rank K" claim needs an actual operator-theoretic argument, not a citation. Verma et al. 2023 is invoked as a black box; whether their bound in fact takes a γ_T input as written, and whether the m factor outside is correct, must be verified. **Fix:** Either prove γ_T ≤ C K log T directly (e.g., via the eigendecomposition of K, which has at most K nonzero eigenvalues), or restate the theorem without the K dependence and just claim a constant-factor improvement.

**W3. The lower bound (Prop. A.6 / Prop. C.1) is essentially a definition.** The claim is that "any sampler with arm-independent per-round factorization incurs Ω(d log T / Δ)". This rules out independent CTS by definition but says nothing about, e.g., warm-start CTS (which is arm-independent post-warmup), TS-LLM (which mixes in a non-factorized recommendation), or any of the LIBRA-style methods. The proof reduces to per-arm Lai–Robbins applied independently — but Lai–Robbins is an asymptotic instance-dependent bound for a specific arm, not a sum-bound across d arms. The d/K factor in the matching upper bound assumes *known* cluster identity, which is not the setting of the paper. **Fix:** Either drop the lower bound, or honestly title it "Lower bound under known clusters and strict factorization" and note that it does not separate CorrFull from any actually-competing method. As written, it is misleading.

**W4. ESCB is missing as a head-to-head baseline and the dismissal is not credible.** The §App. F argument — "ESCB requires linear features, the one-hot extension imposes within-cluster homogeneity, this is the same assumption our RBF kernel relaxes" — is exactly backwards. Your RBF on cluster ranks *also* imposes within-cluster homogeneity (every arm in C_k has identical f_i = r_k/(K-1)); the only difference is that it then adds Gaussian noise on top in the sampling step. ESCB with one-hot cluster features is the natural first comparator and you should run it. The complexity argument (O(d^(m+1)) action enumeration) is also overstated: at m=5, d=200, monotone submodular maximization with greedy is O(dm) per round and is the standard implementation. Real ICML reviewers will not accept this dismissal. **Fix:** Run ESCB-greedy with one-hot cluster features and report it as a row in Tables 1 and 3.

**W5. CLUB / Gentile et al. is cited but not benchmarked, and it does the same job without an LLM.** CLUB online-clusters bandits using confidence-interval overlap. CorrCTS-Prune is described as borrowing the same intuition. The right baseline is "CLUB on the same arm set, no LLM" — if it lands close to CorrFull, the LLM is doing very little. **Fix:** Run CLUB or a CLUB-flavored online-clustering baseline on E1 and E3.

**W6. RandomCorr beats CTS by 7.1% — this is the most uncomfortable number in the paper and the discussion buries it.** The paper's own ablation shows that *any* RBF correlation matrix (random clusters) beats independent sampling, and that on d=50, m=3 the LLM's NMI vs random is +0.02 yet CorrFull still beats RandomCorr. Read together, this suggests a substantial fraction of the gain is "any positive-definite kernel sharpens top-m selection in finite samples", not "the LLM identified meaningful structure". The honest framing: this paper is partly about **correlation regularization**, partly about LLM structure. Right now the abstract foregrounds the LLM and elides the regularization story. **Fix:** Add a row "CorrFull with random cluster ranks but matched LLM K and ρ" — current RandomCorr uses block-diagonal, not the same RBF construction. Until that ablation runs, the 12.7-point attribution is undermined by the d=50, m=3 NMI=0.09 finding.

**W7. The endogenous-oracle finding rests on a single (d=25, m=5) configuration and 20 seeds.** This is the centerpiece motivation; n=20 on one (d, m) is too thin to support the framing it carries. What about T_w ∈ {10, 50, 100}? CTS warmup vs CUCB warmup vs ε-greedy warmup? The 4/5 vs 0/5 is striking but a real reviewer will demand this be replicated across at least 3 (d, m) settings before believing it generalizes. **Fix:** Add a 3×3 grid (3 warmup strategies × 3 (d,m) settings) before claiming endogeneity is a general phenomenon.

**W8. MIND-Simulated is not MIND.** The simulator is fully synthetic: μ_i = clip(0.3 q_i + 0.7 p_cat(i)), Dirichlet user prefs, Beta(2,5) quality. The 33.8% reduction is on a generative model that *bakes in* exactly the cluster structure your kernel is designed to exploit. This is borderline circular; the paper acknowledges it but the abstract does not. The 5× larger gain on MIND vs synthetic Bernoulli is almost mechanical — the synthetic Bernoulli has no latent group structure, MIND-Simulated has an exact one — so this number reads as confirmation that "if structure exists, kernel finds it", not as evidence of practical impact. **Fix:** Either rerun on real MIND click logs (the dataset is public) or, at minimum, replace "news recommendation task" with "synthetic instance with planted category structure" in the abstract.

**W9. Hyperparameter selection on a "disjoint validation set" is asserted but not specified.** §6 / Appendix says ρ_max=0.7, ℓ=0.5, K=8, T_w=30 were chosen on a separate validation set. *Which* configurations? With what seeds? If the validation set is from the same generative process as E1, this is data-dependent hyperparameter tuning by another name, and the held-out E2 split (which authors emphasize) is the only honest test. The sensitivity analysis (App. E) reports peaks-at-0.7 and ℓ-best-at-0.5 — if these were also optimized on the test instances, the headline numbers are inflated. **Fix:** Explicitly enumerate validation configs and their non-overlap with E1/E2/E3 test configs.

**W10. The TS-LLM comparison is loaded.** TS-LLM uses 25 LLM calls vs CorrFull's 1, and CorrFull beats it 268.2 vs 279.5. The narrative "CorrFull is more efficient" is fair, but the paper does not report TS-LLM with **1 LLM call** at the same T_w — i.e., a TS-LLM ablation matched on LLM budget. If TS-LLM-1call is competitive, the 15.5% vs 18.9% gap shrinks and the contribution is "kernel vs recommendation at equal LLM cost" rather than "structure vs belief". **Fix:** Add TS-LLM with k ∈ {1, 5} LLM calls.

**W11. Statistical-testing accounting is opaque.** Holm-Bonferroni "K=5" is reported in the table caption but the appendix Holm table (Tab. C.3) lists 5 algorithms vs CTS without specifying the family. Was the LLM-cost comparison, the per-(d,m) breakdown, or the warm-start E3 result included in the family? If the family is computed across all comparisons in the paper (E1 + E2 + E3 + ablations), K is much larger and several p-values are at risk. **Fix:** State the global comparison family and re-correct.

**W12. Related-work omissions.** No Gupta–Mannor–Singh "Bandits with Correlated Arms"; no Hong et al. "Latent Bandits Revisited" (NeurIPS 2020), which is *the* prior on latent-cluster bandits without LLMs and is a much closer comparator than CLUB; no mention of Wang–Chen–Wen "Combinatorial Multi-Armed Bandit and Its Extension to Probabilistically Triggered Arms" or Kveton et al. cascading bandits; no LIBRA experimental comparison despite being cited. The "to our knowledge, first to use LLM for structure not belief" claim is risky without surveying multi-task BO work where Gaussian process kernels are routinely seeded by metadata (Swersky et al., Feurer et al.).

## 4. Questions for Authors

1. **Theorem 1 uniformity.** What is the precise n_0 such that ∑_t Pr(min_i n_i^(t) < n_0) = o(T) under CorrFull's sampling distribution? This is the load-bearing step in the Gaussian reduction.
2. **γ_T proof.** Provide the explicit eigenvalue argument that γ_T ≤ C K log T for Σ = ρK + (1−ρ)I_d when K has K distinct rows. The Srinivas Thm. 5 citation does not directly apply.
3. **Lower bound.** Does Prop. A.6 separate CorrFull from TS-LLM, warm-start CTS, or LIBRA, or only from independent Thompson sampling? Please state explicitly.
4. **ESCB.** What is ESCB-greedy's regret on E1 with one-hot cluster features at the LLM's K=8? On E3?
5. **Endogenous oracle robustness.** What is the LLM's NMI / top-m accuracy under round-robin warmup at (d=50, m=5)? At (d=25, m=3)?
6. **Validation-test separation.** List the hyperparameter-validation configs and confirm they share no env_seed or (d,m,gap) with E1/E2/E3.
7. **TS-LLM at matched budget.** Report TS-LLM with 1 LLM call at T_w = 30; how does it compare to CorrFull?
8. **MIND-real.** Why not run on the public MIND click logs? If you cannot, please state which features of real MIND your simulator does not capture.

## 5. Score

**5 / 10 (Borderline reject, leaning reject for ICML main; borderline accept for the workshop).**

The empirical story is well-executed for the workshop bar (paired tests, ablations, sensitivity), and the conceptual reframing of LLM-as-structure is genuinely valuable. But the theoretical claims overreach (Theorem 1, Prop. A.6, the γ_T step), the missing ESCB / CLUB / LIBRA / Hong baselines weaken the empirical claims, and the RandomCorr-beats-CTS finding undermines the LLM-attribution narrative more than the paper acknowledges. For the **workshop** I would lean accept (6) **conditional on**: (i) replacing Theorem 1 with a correctly-stated finite-rank bound, (ii) adding ESCB-greedy and CLUB rows, (iii) honest framing of MIND-Simulated. For ICML main, this is not yet ready.

## 6. Confidence

**4 / 5.** Bandits is my home field and I've reviewed CTS / GP-TS papers for years. I would want to verify the Srinivas Thm. 5 / Verma 2023 citations against the actual statements before final submission, but I'm confident on the substance of W1–W6.

## 7. Comparison to OpenReview Reviews I Pulled

I cross-referenced this submission against three relevant OpenReview threads.

- **"When Combinatorial Thompson Sampling Meets Approximation Regret"** ([openreview.net/forum?id=RQ8X_iK3HT5](https://openreview.net/forum?id=RQ8X_iK3HT5)). Reviewers (DcCA, aZHQ, 6J2J, 4G2X) hit on: (a) gap-dependent constants that can be exponentially small in random instances; (b) confusing presentation of the regret bound; (c) the question of what the bound implies vs CUCB. **Patterns that apply here:** the present paper has the same kind of "bound looks clean but constants are hidden" issue (Theorem 1, where C, K's role, and the o(T) absorption are all under-specified). The DcCA-style "writing is confusing" critique would not apply — this paper is well-written — but the technical-rigor critiques do.
- **"Can LLMs Explore In-Context?"** ([openreview.net/forum?id=8KpkKsGjED](https://openreview.net/forum?id=8KpkKsGjED)). The published critique is that GPT-4 only explores when given an *external summary*, so the result depends on a non-scalable prompt format. **Patterns that apply here:** the present paper's endogenous-oracle finding is exactly the same dependency, surfaced as a feature rather than a bug. Reviewers will read §1 and ask "so the LLM only works when CTS has already done the hard part" — the paper currently frames this as a virtue (it motivates structure injection) but a skeptical reviewer reads it as: the LLM contributes least where it would matter most.
- **"Balancing Act: LLM-Designed Restless Bandit Rewards"** ([openreview.net/forum?id=LMuXCe0QfA](https://openreview.net/forum?id=LMuXCe0QfA)) and **"Beyond Numeric Awards: In-Context Dueling Bandits"** ([openreview.net/forum?id=sGfVBi15uY](https://openreview.net/forum?id=sGfVBi15uY)). Both are LLM-bandit papers in the same workshop ecosystem. The recurring critique pattern in this subfield (per the summaries available) is missing baselines and weak theoretical guarantees relative to algorithmic baselines. The present paper is *stronger* than these on ablations and statistical testing, but weaker than dedicated bandits papers (per the CTS / approximation-regret review above) on theoretical rigor. The workshop venue forgives some of this; ICML main would not.

---

## Action List (top 10, ordered by accept/reject delta)

1. **Replace Theorem 1 statement with a finite-rank-correct bound.** Prove γ_T ≤ C K log T from the eigendecomposition of Σ directly; state n_0 explicitly; restrict to "n_i ≥ n_0 for all i" or absorb the early-rounds case into an additive O(d) term. Without this fix the bound is the easiest target in any review.
2. **Run ESCB-greedy with one-hot cluster features on E1 and E3.** Report it as a Table 1 / Table 3 row. If it loses, you've earned the dismissal; if it wins, you need to revise. Either outcome strengthens the paper.
3. **Run CLUB (or a CLUB-flavored online-clustering bandit) on E1 and E3.** This is the no-LLM structured baseline; without it, the LLM contribution is unisolated.
4. **Add a TS-LLM-1call ablation.** Match LLM budget to CorrFull and rerun; report alongside the 25-call number.
5. **Reframe the lower bound as "Lower bound under known clusters and strict per-arm factorization"** and remove implications about non-factorizing methods. Currently overclaims.
6. **Run real MIND click logs**, or relabel "MIND-Simulated" everywhere as "synthetic instances with planted category structure" including in the abstract. The current framing invites the "circular evaluation" critique.
7. **Replicate the endogenous-oracle 4/5 vs 0/5 finding** across at least 3 (d, m) configs and 3 warmup strategies. Currently n=20 on one cell; expand to n=60+.
8. **Add an honest "RBF-on-random-partition with same K, ρ_max, ℓ" ablation.** Current RandomCorr is block-diagonal, which makes the 12.7-point attribution to "LLM structure" look larger than it is. Use the same RBF kernel with random rank assignments.
9. **Specify the validation set explicitly** (configs, seeds, non-overlap with test). One sentence. Without this the held-out claim is asserted rather than demonstrated.
10. **Add citations: Hong et al. "Latent Bandits Revisited" (NeurIPS 2020); Gupta–Joshi–Tewari "Bandits with Correlated Arms"; Swersky et al. "Multi-Task BO".** These are the closest prior work for "kernel-seeded structured bandits" and their absence will be flagged by any informed reviewer.

---

*Word count: ~2,350. Final note to authors: the conceptual contribution is real, the experiments are competently run, and the paper is well-written. The theory is the load-bearing weakness; fix Theorem 1 and add ESCB/CLUB and the workshop accept becomes routine. As is, a careful PC member will read Section 4.4 and bounce it.*
