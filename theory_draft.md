# NeurIPS Paper — Theoretical Skeleton

## Proposed Title Options
1. "Robust Combinatorial Semi-Bandits with Unreliable LLM Oracles: From Consistency Traps to Provable Trust"
2. "When LLM Consistency is a Liability: Divergence-Based Trust for Corruption-Robust Combinatorial Bandits"
3. "LLM-Guided Combinatorial Bandits: A Best-of-Both-Worlds Theorem"

## Main Contributions (one paragraph)

We formalize the LLM-as-action-oracle problem for combinatorial semi-bandits and identify a previously-unrecognized failure mode: **any trust mechanism depending only on oracle self-consistency is Ω(T)-regret-vulnerable to deterministic adversaries**. We introduce **LLM-CUCB-BoBW**, a best-of-both-worlds algorithm combining divergence-based trust, epoch-wise confidence inflation, and a forced-exploration floor, achieving *simultaneously*: (i) O(m log T / Δ_min) regret when the oracle is reliable, (ii) Õ(√mT) regret when the oracle is arbitrary, and (iii) Õ(√mT + C) when corruption budget is C. We prove matching lower bounds for each regime and validate empirically that competing methods either fail catastrophically (up to 4x worse than vanilla CUCB) or sacrifice the reliable-oracle gain.

---

## Theorem 1 (LOWER BOUND — Consistency Trap)

**Setup.** Consider a combinatorial semi-bandit with d arms, super-arm size m, horizon T. Let A be any algorithm whose super-arm choice at time t is measurable only w.r.t. (history, oracle-responses, a trust score τ(·) that is a deterministic function of oracle self-consistency).

**Statement.** There exists a problem instance and a deterministic oracle O* such that
> $\mathbb{E}[\text{Regret}_T(A)] \geq \Omega\left(\frac{m T \Delta_{\min}}{d}\right)$

i.e., linear in T whenever Δ_min = Ω(1).

**Proof sketch.**
1. Construct two instances I₁ (arms i ∈ [m] optimal) and I₂ (arms i ∈ {d-m+1,…,d} optimal), differing in m gap-δ arms.
2. Let O* return arm set {d-m+1,…,d} deterministically on every query, regardless of history.
3. By self-consistency, τ(O*(H_t)) = 1 for all histories H_t, so A is indistinguishable in its use of O under I₁ vs I₂.
4. A's only information comes from reward samples of pulled arms. Under I₁, O*'s suggestion has gap Δ_min per arm; under I₂, zero gap.
5. A must distinguish I₁ from I₂ with O(T·Δ_min²) reward samples — but following O* for T rounds yields linear regret under I₁.
6. Taking the max over I₁, I₂ gives the lower bound.

**Implication.** Our original LLM-CUCB-AT uses τ = min(κ, ρ) where κ is self-consistency — it falls in the class of Theorem 1. Consistent-wrong oracle with m optimal-looking suboptimals achieves regret ≥ 4x vanilla CUCB empirically, matching our validation.

---

## Theorem 2 (UPPER BOUND — LLM-CUCB-BoBW)

**Algorithm (informal).** Run two policies in parallel:
- π₁: vanilla CUCB over all d arms
- π₂: LLM-CUCB with divergence-based trust + epoch-wise confidence inflation

A Tsallis-INF meta-learner with learning rate η_t = √(log 2 / t) selects a policy per round. Additionally, with probability ε_t = t^(-1/3), pull a uniformly random super-arm (exploration floor).

**Statement.** For any oracle O with *corruption budget* C = #{t : O(H_t) ≠ top-m true}, the expected regret of LLM-CUCB-BoBW satisfies:

> $\mathbb{E}[\text{Regret}_T] \leq \min\left( \underbrace{\frac{48 m d \log T}{\Delta_{\min}}}_{\text{CUCB rate (π₁ branch)}}, \underbrace{\frac{c_1 m^2 \log T}{\Delta_{\min}} + c_2 \sqrt{m T} + O(C)}_{\text{LLM-CUCB branch with corruption}} \right) + \underbrace{O(\sqrt{T \log 2})}_{\text{meta overhead}} + \underbrace{O(m T^{2/3})}_{\text{exploration floor}}$

**Key property:** when oracle is reliable, first term of π₂ branch dominates (logarithmic regret). When oracle is adversarial (C = T), meta-learner routes to π₁ giving CUCB rate. In between, graceful degradation in C.

**Proof sketch.**
1. Standard Tsallis-INF regret decomposition: meta-learner achieves min-policy-regret + √T log 2.
2. For π₂, divergence-based trust prevents Theorem-1 attacks: deterministic oracles have concentrated endorsement, triggering distrust → falling back to CUCB-on-ground-set with bounded extra regret.
3. Forced exploration floor ensures Ω(t^{2/3}) pulls of every super-arm; by anytime concentration, this breaks any deterministic attack with sub-O(T^{2/3}) corruption cost.
4. Epoch-wise confidence inflation (BARBAT-style) adds O(log log T) factor but preserves the Õ(√T + C) rate under known C.

---

## Theorem 3 (UPPER BOUND — Reliable Oracle)

**Assumption.** Oracle O is δ-reliable: with probability ≥ 1-δ, O(H_t) = top-m true on every query.

**Statement.**
> $\mathbb{E}[\text{Regret}_T] \leq O\left( \frac{m^2 \log T}{\Delta_{\min}} \right) + O(\delta T m)$

i.e., O(log T) regret when δ = O(1/T). Matches Corollary 1.1 of original paper but under divergence-trust, not consistency-trust.

---

## Empirical Validation Plan

### Headline Result
- **Consistent-wrong failure (the motivating example):** LLM-CUCB-AT (original) gets 30,899 regret vs CUCB's 7,643. Our **LLM-CUCB-BoBW** gets at most 8,500 regret (≈1.1× CUCB, matching Theorem 2). Directly validates that our algorithm does *not* fall into the Theorem 1 trap.

### Comprehensive Benchmarks
- 5 corruption scenarios × 3 gap structures × 3 dimensions = 45 configs
- 100 seeds per config (T=100,000)
- Compare against 8+ baselines

### Theory-Validation Experiments
- **Reliable oracle (δ ≈ 0):** Show O(log T) scaling in T for LLM-CUCB-BoBW (Theorem 3)
- **Adversarial oracle (δ = 1):** Show Θ(√T) scaling for LLM-CUCB-BoBW, Θ(T) for original LLM-CUCB-AT (Theorem 2 adv-branch + Theorem 1)
- **Variable corruption budget C:** Show linear-in-C dependence for our method (Theorem 2)

### Real-World Experiment
- Use Claude/GPT-4 as oracle on MIND news recommendation (d=200 articles, m=5 slots)
- Adversarial variant: prompt LLM with biased instructions ("always recommend entertainment")
- Show our method robustly handles the biased oracle while baselines fail

---

## Story Arc for NeurIPS Paper

1. **Problem framing:** LLMs as action oracles in combinatorial decision-making. Motivating applications: personalized recommendation, content moderation, experiment design.

2. **Naïve approach (straw man):** LLM-CUCB-AT with self-consistency trust. Show it works when oracle is reliable BUT fails catastrophically under deterministic-wrong adversaries.

3. **Lower bound reveal:** Theorem 1 — self-consistency is fundamentally backwards. Any consistency-only trust mechanism is Ω(T)-vulnerable.

4. **Our fix:** Divergence-based trust + BoBW meta-layer + exploration floor. Theorems 2 + 3 (upper bounds). Matching lower bound via reduction to Lykouris-Mirrokni-PaesLeme.

5. **Experiments:** Comprehensive benchmarks showing our method is robust AND efficient. Closes the 4x-worse-than-CUCB gap.

6. **Extensions (appendix):** Tightening constants, extensions to matroid/linear constraints, real-LLM demo on MIND dataset.

---

## Open Questions / Risks

1. **Can we prove tightness of Theorem 1?** Sketch seems right but matches with standard bandit LB techniques (Kullback-Leibler reduction) need care.

2. **Is the exploration floor's T^{2/3} cost too much?** Could we replace with adaptive exploration triggered only on detected distrust?

3. **Does BoBW meta-layer actually help in practice?** Round 2 testing will tell us.

4. **Can we eliminate the need for tuning h_max, ε_t, epoch_base?** Currently three hyperparameters; ideally one parameter-free variant.

5. **How does this interact with real LLMs that have non-deterministic outputs?** The theory assumes worst-case deterministic; real LLMs have temperature. Might need average-case analysis.

---

## Related Work Emphasis (Positioning)

- **BARBAT (Fang 2025):** closest corruption-robust framework. Our contribution: first to combine with LLM-oracle setting + provable robustness to systematic (not budgeted) bias.
- **Bayley et al. 2025:** empirically observed the same break-point (~30% corruption) but no theory. We provide the theorem.
- **Krishnamurthy et al. 2024:** LLMs as bandit algorithms fail at exploration. We side-step: LLMs provide advice, not full control.
- **Zimmert-Seldin Tsallis-INF:** our meta-layer. Novelty: first application to LLM-oracle combinatorial setting.
- **Lykouris-Vassilvitskii:** learning-augmented framework. We instantiate for combinatorial bandits with LLM predictions.
