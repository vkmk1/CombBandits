# Experiment Descriptions and Results

## What Has Been Run

All experiments completed so far use **simulated environments** and **simulated oracles**. No real datasets (MIND, SNAP) and no real LLM API calls (GPT-4o) have been used. These experiments validate the algorithm's theoretical properties under controlled conditions.

---

### exp3_quick_test — Smoke Test

**Purpose:** Verify the codebase works end-to-end.

**Setup:**
- Environment: `SyntheticBernoulliEnv` — 20 Bernoulli arms (d=20, m=5)
- Oracle: `SimulatedCLO` — a coin flip that returns the optimal set S* with probability (1-epsilon), random set otherwise
- Agents: CUCB, CTS, LLM-CUCB-AT, LLM-Greedy
- T=1,000 rounds, 3 seeds
- 24 total trials

**What it does NOT do:** No LLM calls, no real data.

**Results:** Confirms all agents run without errors. Too few seeds for statistical significance.

---

### exp4_mind — Simulated News Recommendation

**Purpose:** Test agents on a news-recommendation-style reward structure.

**Setup:**
- Environment: `MINDEnvSimulated` — **NOT the real MIND dataset**. Generates synthetic articles with random category assignments and Dirichlet-distributed user preferences. Click probabilities come from category-user alignment + article quality noise. This mimics the structure of news recommendation but uses no real news data.
- Oracle: `SimulatedCLO` — coin flip, not an LLM
- Agents: CUCB, CTS, LLM-CUCB-AT, LLM-Greedy, ELLM-Adapted, Warm-Start CTS
- d=200 articles, m=5 displayed, T=2,000 sessions, 20 seeds
- 280 total trials

**What it does NOT do:** Does not load the Microsoft MIND dataset. Does not call any LLM. The "oracle" is a random number generator that knows the optimal set.

**Results (mean cumulative regret):**

| Agent | Partial Overlap (eps=0.15) | Uniform (eps=0.1) | Uniform (eps=0.3) |
|-------|---------------------------|-------------------|-------------------|
| Warm-Start CTS | **1,144** | **1,125** | **1,114** |
| CTS | 1,207 | 1,207 | 1,207 |
| CUCB | 1,920 | 1,920 | 1,920 |
| LLM-CUCB-AT | 1,931 | 1,927 | 1,927 |
| LLM-Greedy | 2,296 | 2,301 | 2,288 |
| ELLM-Adapted | 2,307 | 2,307 | 2,307 |

**Interpretation:** Thompson Sampling (CTS, Warm-Start CTS) outperforms UCB-based methods on this task structure. LLM-CUCB-AT matches CUCB — the simulated oracle adds no useful information beyond what UCB exploration discovers. LLM-Greedy and ELLM perform worst because they over-rely on the oracle.

---

### exp5_influence_max — Simulated Influence Maximization

**Purpose:** Test agents on an influence-maximization-style reward structure.

**Setup:**
- Environment: `InfluenceMaxEnvSimulated` — **NOT real SNAP social networks**. Generates a synthetic graph with power-law degree distribution and planted community structure. Seed quality correlates with degree and community centrality. No actual independent cascade simulation — rewards are sampled from pre-computed quality scores with Gaussian noise.
- Oracle: `SimulatedCLO` — coin flip, not an LLM
- Agents: CUCB, CTS, LLM-CUCB-AT, LLM-Greedy, ELLM-Adapted, Warm-Start CTS
- d=200 candidate seeds, m=10, T=5,000, 20 seeds
- 280 total trials

**What it does NOT do:** Does not load SNAP graphs. No independent cascade Monte Carlo. No LLM calls.

**Results (mean cumulative regret):**

| Agent | Partial Overlap (eps=0.2) | Uniform (eps=0.1) | Uniform (eps=0.3) |
|-------|---------------------------|-------------------|-------------------|
| Warm-Start CTS | **349** | **342** | **348** |
| CTS | 390 | 390 | 390 |
| CUCB | 5,142 | 5,142 | 5,142 |
| LLM-CUCB-AT | 5,203 | 5,688 | 5,418 |
| LLM-Greedy | 8,256 | 8,215 | 8,286 |
| ELLM-Adapted | 8,176 | 8,176 | 8,176 |

**Interpretation:** Structured rewards with clear clusters favor Thompson Sampling heavily. CUCB struggles because uniform exploration over d=200 arms is slow. LLM-CUCB-AT is comparable to CUCB. Oracle-following agents (LLM-Greedy, ELLM) have the worst regret.

---

### exp7_ablation_trust — Trust Component Ablation

**Purpose:** The most important simulated experiment. Tests how LLM-CUCB-AT's composite trust score handles different oracle failure modes compared to agents that blindly follow the oracle.

**Setup:**
- Environment: `SyntheticBernoulliEnv` — 100 Bernoulli arms, uniform gap structure (d=100, m=10, delta_min=0.05)
- Oracle: `SimulatedCLO` — coin flip with 4 corruption configurations:
  - `uniform, eps=0.0`: Oracle always returns S* (perfect)
  - `uniform, eps=0.2`: Oracle returns S* 80% of the time, random set 20%
  - `adversarial, eps=0.3`: Oracle returns the worst set 30% of the time
  - `consistent_wrong, eps=1.0`: Oracle ALWAYS returns the same wrong set (kappa=1 but wrong)
- Agents: CUCB, LLM-CUCB-AT, LLM-Greedy, ELLM-Adapted, Warm-Start CTS
- T=30,000, 30 seeds
- 510 total trials

**What it does NOT do:** No LLM calls. The "oracle" is a parameterized coin flip.

**Results (mean cumulative regret):**

| Agent | Clean (eps=0) | Uniform (eps=0.2) | Adversarial (eps=0.3) | Consistently Wrong |
|-------|--------------|-------------------|----------------------|-------------------|
| Warm-Start CTS | **1,455** | **1,444** | **1,465** | **1,476** |
| CUCB | 7,242 | 7,242 | 7,242 | 7,242 |
| LLM-CUCB-AT | 24,132 | 17,541 | 13,854 | 23,420 |
| LLM-Greedy | 70,301 | 69,925 | 70,069 | 68,657 |
| ELLM-Adapted | 70,301 | 70,301 | 70,300 | 68,657 |

**Key findings:**

1. **LLM-CUCB-AT is safe against catastrophic oracle failure.** It achieves 3-5x lower regret than LLM-Greedy and ELLM-Adapted across all corruption types. The composite trust mechanism prevents blindly following a bad oracle.

2. **LLM-CUCB-AT loses to plain CUCB** in all simulated oracle conditions. This is expected: the simulated oracle is a coin flip that adds noise without adding information. CUCB's pure exploration is more efficient when the oracle has no real world knowledge.

3. **Warm-Start CTS dominates** because the simulated oracle knows S* and tells it the answer in one query. This is an artifact of simulation — a real LLM would not have this property.

4. **The consistently wrong oracle** (eps=1.0) causes linear regret for LLM-Greedy (~70K) but LLM-CUCB-AT detects it and hedges (~23K). This validates the composite trust mechanism.

**What these results do NOT show:** Whether a real LLM provides useful priors. The simulated oracle either knows S* perfectly or returns garbage — there's no middle ground of "partial world knowledge" that a real LLM would provide.

---

## What Has NOT Been Run (Pending DELLA)

### exp6_workshop_main — Full Theory Validation

**Purpose:** Validate theoretical regret predictions across multiple dimensions and corruption rates.
- d = {50, 100, 200}, 7 agents, 10 oracle configs, 30 seeds, T=100K
- Produces regret vs epsilon and regret vs d plots

### exp8_scaling_d — Large-d Dimension Scaling (GPU)

**Purpose:** Demonstrate dimension reduction at scale (d=50 to d=5000).
- Uses GPU-batched runner, 100 seeds, 4 agents, 3 oracle configs
- Produces the paper's strongest figure: CUCB regret grows as sqrt(d), LLM-CUCB-AT stays flat

### exp9_real_llm — Real GPT-4o Oracle

**Purpose:** The critical experiment. Actual LLM API calls to validate that LLMs provide useful combinatorial priors.
- Real GPT-4o primary queries with GPT-4o-mini re-queries
- O(sqrt(T)) query schedule to control API costs (~$60 total)
- Measures actual LLM epsilon, kappa, rho on bandit tasks
- This is what determines whether the paper's premise holds

---

## Summary

| Experiment | Real Data? | Real LLM? | Status | What it validates |
|-----------|-----------|-----------|--------|-------------------|
| exp3_quick_test | No | No | Done | Installation |
| exp4_mind | No (simulated) | No | Done | Reward structure |
| exp5_influence_max | No (simulated) | No | Done | Reward structure |
| exp7_ablation_trust | No | No | Done | Trust mechanism safety |
| exp6_workshop_main | No | No | **Pending** | Theoretical bounds |
| exp8_scaling_d | No | No | **Pending** | Dimension reduction |
| exp9_real_llm | No | **Yes (GPT-4o)** | **Pending** | LLM usefulness |
