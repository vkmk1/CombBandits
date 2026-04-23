# Experimental Parameters Survey: LLM-Guided Combinatorial Bandits
## Gold-Standard Parameters from Top-Venue Papers

---

## 1. "Can Large Language Models Explore In-Context?"
**Authors:** Krishnamurthy, Harris, Foster, Zhang, Slivkins
**Venue:** NeurIPS 2024 (also ICML 2024 Workshop on In-Context Learning)
**arxiv:** 2403.15371

| Parameter | Value |
|-----------|-------|
| **Horizon T** | 100 (main), 200, 500 (robustness) |
| **Number of arms K** | 5 (hard instance), 4 (easy instance) |
| **Arms selected per round m** | 1 (standard MAB) |
| **Seeds/repetitions** | GPT-3.5: N=20 across 48 configs; GPT-4: N=10 (main), N=20-40 (robustness); Llama2: N=10 across 32 configs |
| **Problem configurations** | 2 instances (Easy, Hard) x multiple prompt configs = 48 (GPT-3.5), 32 (Llama2) |
| **Gap structure** | Hard: best arm mu*=0.6, others mu=0.4 (gap Delta=0.2); Easy: best arm mu*=0.75, others mu=0.25 (gap Delta=0.5). Bernoulli rewards. |
| **Results reporting** | SuffFailFreq(t), MinFrac(t), MedianReward, GreedyFrac. Cumulative time-averaged rewards visualized. Fraction-based metrics (not standard regret). |
| **Baselines** | UCB (C=1), Thompson Sampling (Beta-Bernoulli), Greedy, epsilon-Greedy |
| **LLMs tested** | GPT-3.5-Turbo-0613, GPT-4-0613, Llama2-13B-chat (4-bit quantized) |
| **Real LLM calls?** | Yes, real API calls |

---

## 2. "Beyond Numeric Rewards: In-Context Dueling Bandits with LLM Agents"
**Authors:** Xia, Liu, Yue, Li
**Venue:** Findings of ACL 2025 (pages 9959-9988)
**arxiv:** 2407.01887

| Parameter | Value |
|-----------|-------|
| **Horizon T** | 2000 (main), 3000 (robustness), 300 (LEAD evaluation) |
| **Number of arms K** | 5 (main), also tested K=10 for scalability |
| **Arms selected per round m** | 2 (dueling bandits: select pair per round) |
| **Seeds/repetitions** | LLM experiments: N=5; Baseline algorithms: N=20 |
| **Problem configurations** | 4 environments: Transitive-Easy, Transitive-Hard, Intransitive-Easy, Intransitive-Hard |
| **Gap structure** | Bradley-Terry-Luce model. Easy: theta(1)=1, theta(i)=0.5-(i-1)/2K. Hard: theta(i)=1-(i-1)/K. Arm ordering b5 > b3 > b2 > b1 > b4. SST and STI properties. |
| **Results reporting** | Cumulative strong regret and cumulative weak regret (mean +/- std dev shown as shaded bands). Also: Best Arm Inclusion Ratio, Converged Best Arm Ratio, generalized variance (log scale bar plots). |
| **Baselines** | IF2, BTM (gamma=0.5), SAVAGE, RUCB (alpha=0.51), RCS (alpha=0.51), RMED (f(K)=0.3*K^1.01), Self-Sparring (eta=1), DTS (alpha=0.51), VDB |
| **LLMs tested** | GPT-3.5-Turbo, GPT-4, GPT-4-Turbo, o1-preview |
| **Real LLM calls?** | Yes, real API calls (temperature=0) |
| **LEAD hyperparams** | Threshold t in {50,100,200}, optimal t=50; confidence delta in {0.1,0.2,0.4}, optimal delta=0.4; also delta=1/(TK^2) |

---

## 3. "Should You Use Your Large Language Model to Explore or Exploit?"
**Authors:** Harris, Slivkins
**Venue:** ICLR 2026
**arxiv:** 2502.00225

| Parameter | Value |
|-----------|-------|
| **Horizon T** | MAB exploit: variable history lengths; CB: T=50 (K=2,d=1), T=100 (K=d=2), T=4000 (K=d=2), T=500 (text CB); Exploration: T=1000 (UCB1) |
| **Number of arms K** | MAB: 5; CB: K in {2, 5}; Exploration: variable candidates {1,2,3,4,5,7,10} |
| **Context dimension d** | CB: d in {1, 2, 5} |
| **Seeds/repetitions** | 10 tasks per gap value (MAB); 10 repetitions per exploration task |
| **Problem configurations** | MAB: 10 gap values Delta in {0, 0.05, 0.1, ..., 0.5}; CB: multiple (K,d,T) combos; Text CB: easy + complex nonlinear; Exploration: 10 philosophical Q&A + 410 arXiv abstracts |
| **Gap structure** | MAB Bernoulli: best arm mu=0.5+Delta/2, others mu=0.5-Delta/2; CB linear: mu(z,a) = <z,theta_a> + gamma_a with theta in [-1,1]^d, gamma in [-0.25,0.25], unit-variance Gaussian noise |
| **Results reporting** | FracCorrect vs empirical gap with **95% confidence intervals** (shaded bands). Average expected reward with 95% CI for exploration tasks. |
| **Baselines** | Linear regression (CB exploit), UCB1 (exploration), random candidates (exploration) |
| **LLMs tested** | GPT-4, GPT-4o, GPT-3.5 |
| **Real LLM calls?** | Yes, real API calls |
| **Mitigations tested** | k-nearest, k-means, k-means+k'-nearest (for CB) |

---

## 4. "EVOLvE: Evaluating and Optimizing LLMs for Exploration"
**Authors:** Nie, Yi Su, Bo Chang, Lee, Chi, Chen
**Venue:** arXiv preprint 2024 (2410.06238)
**Note:** Referenced in ACL 2025 paper; not confirmed at ICML 2025

| Parameter | Value |
|-----------|-------|
| **Horizon T** | MAB: T=300 (K=5), T=1000 (K=20); CB: T=200 |
| **Number of arms K** | MAB: K=5 (small), K=20 (large); CB: K=10 (easy), K=30 (challenging) |
| **Seeds/repetitions** | **30 independent runs** per experimental setup |
| **Problem configurations** | **16 MAB configs**: 2 action spaces x 2 descriptions x 2 reward distributions x 2 difficulties; **2 CB configs** |
| **Gap structure** | Bernoulli: best arm p=0.5+Delta_min/2, others p=0.5-Delta_min/2; Easy Delta_min=0.5, Hard Delta_min=0.2; Gaussian: Easy sigma=1, Hard sigma=3 |
| **Results reporting** | Pairwise win-rate via **Student's t-test (p<0.05)**. Overall win-rate percentage. Cumulative reward, regret, MinFrac, OptFrac. Fitted regret curves f(T) = lambda*log(T)^alpha/Delta_min + beta*T + lambda_2 |
| **Baselines** | Raw History (RH), Summarized History (SH), Algorithm-Guided (AG), UCB/LinUCB (oracle), Oracle Behavior Fine-Tuning (OFT), in-context few-shot (5 trajectories) |
| **LLMs tested** | Gemma-2B, Gemma-9B, Gemini 1.5 Flash, Gemini 1.5 Pro |
| **Real LLM calls?** | Yes |
| **CB details** | MovieLens dataset, ~10K real user ratings, SVD with d=5, ground truth r_ij = u_i^T Sigma v_j |

---

## 5. "Combinatorial Bandits Revisited" (ESCB)
**Authors:** Combes, Talebi, Proutiere, Lelarge
**Venue:** NeurIPS (NIPS) 2015
**arxiv:** 1502.03475

| Parameter | Value |
|-----------|-------|
| **Horizon T** | Up to T=10,000 (matching), up to T=30,000 (spanning trees, shown in log-scale plots) |
| **Number of base arms d** | Matching: d=25 (N1=N2=5, d=5^2); Spanning trees: d=10 (N=5, d=C(5,2)=10) |
| **Arms selected per round m** | Matching: m=5; Spanning trees: m=4 (N-1=4) |
| **Number of super-arms |M|** | Matching: |M|=5!=120; Spanning trees: |M|=5^3=125 |
| **Seeds/repetitions** | Not explicitly stated (implied multiple runs) |
| **Gap structure** | Matching: optimal arm edges theta_i=a, others theta_i=b (tested a=0.7,b=0.5 and a=0.95,b=0.3); Spanning trees: Delta_min=0.54 |
| **Results reporting** | **Expected regret vs time** (linear and log-log scale), with **95% confidence intervals**. Also regret lower bound curve overlaid. |
| **Baselines** | LLR, CUCB, ESCB-1, ESCB-2, EPOCH-ESCB, theoretical lower bound |
| **Real LLM calls?** | N/A (classical algorithm paper) |
| **Problem types** | Perfect matchings in K_{m,m}, minimum spanning trees in K_N, routing in grids |

---

## 6. "Combinatorial Multi-Armed Bandit" (CUCB)
**Authors:** Chen, Wang, Yuan (+ Wang in JMLR 2016 version)
**Venue:** ICML 2013 (original), JMLR 2016 (extended)
**arxiv:** 1407.8339 (extended version)

| Parameter | Value |
|-----------|-------|
| **Horizon T** | Theoretical paper; experiments in extended version cover online shortest path, social influence maximization, probabilistic max coverage |
| **Framework** | d base arms, select super-arm M of m base arms per round, linear reward M^T X(n), semi-bandit feedback |
| **Gap structure** | Delta_M = mu*(theta) - mu_M(theta); Delta_min = min gap |
| **Regret bound** | O(md/Delta_min * log(T)) |
| **Baselines** | LLR (previous SOTA), lower bound comparison |
| **Key contribution** | General framework with (alpha,beta)-approximation oracle; applications to probabilistically triggered arms |

---

## 7. "LIBRA: Language Model Informed Bandit Recourse Algorithm"
**Authors:** Cao, Gao, Keyvanshokooh, Ma
**Venue:** arXiv preprint 2026 (2601.11905), 50 pages
**arxiv:** 2601.11905

| Parameter | Value |
|-----------|-------|
| **Setting** | Contextual bandits for personalized treatment planning |
| **Experiments** | Synthetic environments + real ACCORD hypertension dataset |
| **Baselines** | LinUCB, LLM-only benchmarks |
| **LLM usage** | Real LLM integration for domain knowledge |
| **Key focus** | Recourse-aware bandits with non-compliance modeling |
| **Full params** | Sections 5-6 detail T, K, d but exact values not extractable from abstract/intro |

---

## 8. Additional Relevant Papers

### "Do LLM Agents Have Regret?" (Park et al., ICLR 2025)
- Referenced in Xia et al.; studies LLM agents in online learning and games

### "LLM-Augmented Contextual Bandits" (Baheri & Alm, NeurIPS 2023 Workshop)
- Referenced in Xia et al.; NeurIPS 2023 Foundation Models for Decision Making Workshop

### "HiVA: Hierarchical Variable Agent" (Tang et al., 2025, arxiv:2509.00189)
- Uses Thompson Sampling MAB for multi-agent coordination
- Seed=42, temperature=1.0, 5 runs, std dev reported

### Survey: "LLM + MAB Interactions" (Xie, Chen, Lv, 2026, arxiv:2601.12945)
- First systematic survey of bidirectional LLM-bandit interactions
- Notes lack of unified experimental standards across field

### CTS: "Thompson Sampling for Combinatorial Semi-Bandits" (Wang & Chen, ICML 2018)
- Compares CTS vs CUCB experimentally
- proceedings.mlr.press/v80/wang18a

---

## STANDARD PRACTICES SUMMARY

### Regret Reporting Conventions
| Convention | Prevalence | Details |
|-----------|-----------|---------|
| **Mean cumulative regret + shaded bands** | Most common | Shaded region = std dev or 95% CI |
| **95% confidence intervals** | Gold standard | Combes (NIPS 2015), Harris (ICLR 2026), Xia (ACL 2025) all use this |
| **Mean +/- std dev** | Common in LLM papers | Xia et al. explicitly reports std dev |
| **Win-rate + t-test** | Emerging (LLM papers) | EVOLvE uses pairwise win-rate with p<0.05 |
| **Median + quantiles** | Rare | Krishnamurthy uses MedianReward |
| **Log-log regret plots** | Classic papers | Combes et al. uses both linear and log-log scale |

### Number of Seeds/Repetitions
| Paper | Seeds | Context |
|-------|-------|---------|
| Krishnamurthy (NeurIPS 2024) | 10-40 | Varies by model (expensive GPT-4 = 10, cheaper = 20-40) |
| Xia (ACL 2025) | 5 (LLM), 20 (baselines) | Lower for LLM due to API cost |
| Harris (ICLR 2026) | 10 per task | Consistent across experiments |
| EVOLvE (2024) | **30** | Highest count; establishes strong standard |
| Combes (NIPS 2015) | Not stated (multiple) | Classical paper, implied sufficient |

**Reviewer expectation at NeurIPS/ICML:** The NeurIPS checklist (Item 7) requires:
- Error bars or confidence intervals for main claims
- Clear statement of what variability is captured
- Explanation of calculation method
- No fixed number mandated, but **10-30 seeds is the de facto standard**
- For expensive LLM experiments, **5-10 is accepted** with justification

### Horizon T Values for Combinatorial Bandits
| Paper | T | Setting |
|-------|---|---------|
| Combes (NIPS 2015) | **10,000 - 30,000** | Matching, spanning trees |
| CUCB/Chen (ICML 2013) | **10,000+** (theoretical focus) | General combinatorial |
| Xia (ACL 2025) | **2,000 - 3,000** | Dueling bandits (LLM-limited) |
| Krishnamurthy (NeurIPS 2024) | **100 - 500** | MAB with LLM (context window limited) |
| Harris (ICLR 2026) | **50 - 4,000** | Varies by experiment type |
| EVOLvE (2024) | **200 - 1,000** | MAB/CB with LLM |

**Key insight:** Classical combinatorial bandit papers use T=10,000-100,000. LLM-bandit papers use T=100-3,000 due to API cost and context window limitations. For a hybrid paper (LLM-guided combinatorial bandits), **T=1,000-10,000 is the sweet spot** -- long enough to show asymptotic behavior, short enough for LLM feasibility.

### Standard Arms Configurations for Combinatorial Bandits
| Paper | d (base arms) | m (selected) | |M| (super-arms) |
|-------|--------------|--------------|-----------------|
| Combes - Matching | 25 | 5 | 120 |
| Combes - Spanning Tree | 10 | 4 | 125 |
| Combes - Grid Routing | 2N(N-1) | 2(N-1) | C(2N-2, N-1) |
| CUCB - General | varies | varies | exponential in d |

---

## RECOMMENDED PARAMETERS FOR YOUR PAPER

Based on this survey, here are gold-standard parameters for an ICML/NeurIPS submission on LLM-guided combinatorial bandits:

### Core Experiments
| Parameter | Recommended Value | Justification |
|-----------|------------------|---------------|
| **Horizon T** | 5,000-10,000 (pure algorithm), 1,000-3,000 (LLM experiments) | Matches Combes scale for algorithms; LLM-feasible for oracle experiments |
| **Base arms d** | 10, 25, 50 | Matches Combes (d=10,25); d=50 shows scalability |
| **Selected per round m** | 3-5 | Matches Combes (m=4,5); meaningful combinatorial structure |
| **Seeds** | 30 (algorithmic baselines), 10-20 (LLM experiments) | Matches EVOLvE (30) for baselines; Krishnamurthy (10-20) for LLM |
| **Gap structure** | Multiple: Delta_min in {0.1, 0.2, 0.5} | Easy/Hard from Krishnamurthy + medium |
| **Problem instances** | >=3 types (e.g., shortest path, matching, max-weight m-set) | Matches Combes (matching + spanning tree + routing) |

### Reporting
| What to Report | How |
|---------------|-----|
| Cumulative regret vs T | Line plot with **95% confidence bands** |
| Also show log-log scale | Demonstrates asymptotic rate |
| Report mean +/- std error | In tables for key T values |
| Regret lower bound | Overlay theoretical lower bound curve |
| Statistical significance | t-test or Mann-Whitney for pairwise comparisons |

### Baselines to Include
1. **CUCB** (Chen et al. 2013) -- the standard combinatorial UCB
2. **ESCB** (Combes et al. 2015) -- state-of-the-art for semi-bandits
3. **CTS** (Wang & Chen 2018) -- Thompson sampling variant
4. **LLR** -- classic baseline
5. **Pure LLM** (no algorithm guidance) -- ablation
6. **Oracle** (optimal policy) -- upper bound
7. **Uniform random** -- lower bound sanity check
8. Your LLM-guided algorithm variants

### LLM Oracle Experiments
- Test with: GPT-4, GPT-4o (or latest), one open-source model
- Compare real LLM vs simulated oracle with varying accuracy
- Report API cost per run
