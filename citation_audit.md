# Citation Audit Report

## Summary
- Papers audited: `paper_4page.tex`, `paper_8page.tex`
- Bib entries total: 36
- Bib entries used (across both papers): 19 (dead: 17, listed below)
- Missing citations flagged: 5
- Mis-attributions flagged: 4 (two critical, two moderate)
- Metadata issues flagged: 5
- Suspicious 2025/2026 citations: 3 verified to exist, 1 with serious mis-characterization

---

## Dead citations (in bib, never used in either paper)

These 17 keys appear in `references.bib` but are cited nowhere in either `.tex` file. Remove or comment out before submission.

- `agrawal2014ts`
- `auer2002exp4`
- `bubeck2012survey`
- `calibgated2025`
- `du2023ellm`
- `fang2025barbat`
- `gupta2019better`
- `he2022nearly`
- `kapoor2019corruption`
- `lattimore2020book`
- `liu2025offcmab`
- `lykouris2018robust`
- `nie2024evolve`
- `park2025regret`
- `thompson1933likelihood`
- `wen2025adversarial`
- `yang2024opro`

Note: `lai1985asymptotically`, `liu2024llambo`, and `lykouris2018competitive` are used in `paper_8page.tex` only (not in `paper_4page.tex`), so they are alive but paper-scoped.

---

## Mis-attributions (critical)

### 1. `cao2026libra` — WRONG CHARACTERIZATION (critical)

**Claimed in both papers:**
> `\citet{cao2026libra} add robustness to adversarial LLM advice.`

> `\citet{cao2026libra} introduced LIBRA, which guarantees no-regret behavior under adversarial LLM advice.`

> `[existing theoretical models] treat oracle quality as an exogenous parameter ε \citep{cao2026libra,sun2025tsllm}`

**What the paper actually does:** LIBRA (arXiv:2601.11905, January 2026) is by **Junyu Cao, Ruijiang Gao, Esmaeil Keyvanshokooh, and Jianhao Ma**. Its focus is *personalized treatment planning* combining algorithmic recourse with LLM-guided bandit algorithms in a medical context. It does **not** study adversarial LLM advice or robustness to adversarial oracles. Its "robustness" property is that it degrades gracefully when the LLM is *unreliable* (not adversarial), and the primary contribution is integrating algorithmic recourse with bandits.

**Fix:** The characterization "guarantees no-regret behavior under adversarial LLM advice" is incorrect and will be spotted by a reviewer who reads the paper. Revise to accurately reflect the paper's contribution (LLM-guided bandit recourse for treatment planning), or replace with a paper that actually proves adversarial-robustness guarantees. Also correct the author list in the bib (current bib has `Cao, Yuxin` and `Ma, Will`; the actual authors are `Cao, Junyu`, `Gao, Ruijiang`, `Keyvanshokooh, Esmaeil`, and `Ma, Jianhao`).

---

### 2. `sun2025tsllm` — IDENTITY MISMATCH (critical)

**Claimed in both papers:**
> `\citet{sun2025tsllm} mix LLM arm suggestions with CTS via a decaying schedule`

**Used as baseline:** "TS-LLM" (included in experimental tables).

**What was found:** The bib entry `sun2025tsllm` points to a paper titled **"Multi-Armed Bandits Meet Large Language Models"** by **Djallel Bouneffouf and Raphael Feraud** (arXiv:2505.13355, May 2025). This is a *survey* paper, not original research, and does not introduce a TS-LLM algorithm with a decaying schedule.

The actual TS-LLM paper matching the description (LLM as reward predictor in Thompson sampling with decaying LLM temperature) appears to be **"Large Language Model-Enhanced Multi-Armed Bandits"** by **Jiahang Sun, Zhiyong Wang, Runhan Yang, Chenjun Xiao, John C.S. Lui, Zhongxiang Dai** (arXiv:2502.01118, February 2025). The first author "Sun" matches the key name `sun2025tsllm`, and the paper introduces LLM-enhanced TS with a decaying schedule.

**Fix:** Verify which paper the authors actually implemented and cited. If it is arXiv:2502.01118, update the bib entry:
```bibtex
@article{sun2025tsllm,
  author    = {Sun, Jiahang and Wang, Zhiyong and Yang, Runhan and Xiao, Chenjun and Lui, John C.S. and Dai, Zhongxiang},
  title     = {Large Language Model-Enhanced Multi-Armed Bandits},
  journal   = {arXiv:2502.01118},
  year      = {2025}
}
```
If the survey paper (Bouneffouf & Feraud) is what was cited, the TS-LLM baseline used in the experiments is uncited.

---

### 3. `kveton2023mixed` — WRONG AUTHOR LIST (moderate)

**Bib entry:**
```
author = {Kveton, Branislav and Manzil, Zaheer and Katariya, Sumeet and Szepesv{\'a}ri, Csaba}
```

**Actual authors:** The paper "Mixed-Effect Thompson Sampling" (AISTATS 2023, PMLR v206) is by **Imad Aouali, Branislav Kveton, and Sumeet Katariya**. Neither "Manzil Zaheer" nor "Csaba Szepesvári" are authors of this paper (they appear on a related but different paper, "Thompson Sampling with a Mixture Prior").

**Fix:**
```bibtex
@inproceedings{kveton2023mixed,
  author    = {Aouali, Imad and Kveton, Branislav and Katariya, Sumeet},
  title     = {Mixed-Effect Thompson Sampling},
  booktitle = {Proceedings of the 26th International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year      = {2023}
}
```

---

### 4. `verma2023bayesian` — TITLE MISMATCH AND WRONG VENUE DESCRIPTION (moderate)

**Bib entry title:** "Bayesian Regret for Combinatorial Stochastic Bandits"

**Actual title at arXiv:2312.12676:** "Bayesian Analysis of Combinatorial Gaussian Process Bandits" (accepted at ICLR 2025).

**Actual authors:** Jack Sandberg, Niklas Åkerblom, Morteza Haghir Chehreghani — **not** Verma, Talebi, and Proutiere as listed in the bib.

**Claimed use:**
> `\citet{verma2023bayesian} give Bayesian regret for combinatorial GP-TS`
> `Plugging γ_T into the combinatorial GP-TS Bayesian regret bound of \citet{verma2023bayesian}`

This is the reduction target for Theorem 1 in both papers. If the authors of this paper are wrong, the attribution of the reduction target theorem is fabricated.

**Fix:** Verify which paper is actually being cited. If it is arXiv:2312.12676, correct the bib entry completely:
```bibtex
@inproceedings{sandberg2025bayesian,
  author    = {Sandberg, Jack and {\AA}kerblom, Niklas and Haghir Chehreghani, Morteza},
  title     = {Bayesian Analysis of Combinatorial {G}aussian Process Bandits},
  booktitle = {Proceedings of the 13th International Conference on Learning Representations (ICLR)},
  year      = {2025}
}
```
Also rename the key from `verma2023bayesian` to `sandberg2025bayesian` and update all `\cite` calls. If a different paper by Verma, Talebi, Proutiere is intended, that paper must be located and verified.

---

## Missing citations (claims needing a cite)

### 1. Berry-Esseen theorem invoked without cite (4-page, Proposition 4 proof)

**Exact text:** `Berry--Esseen yields Kolmogorov distance O(1/\sqrt{n})`

The Berry-Esseen theorem is invoked by name in the proof sketch without any citation. Standard practice requires citing at minimum a textbook (e.g., Feller 1971 or Petrov 1975) or the original papers (Berry 1941, Esseen 1942).

**Suggested addition:** `\citep{feller1971introduction}` or a standard reference.

### 2. Devroye-Györfi inequality invoked without cite (4-page, Proposition 4 proof)

**Exact text:** `the Devroye--Gy\"orfi inequality then gives d_TV ≤ 2\sqrt{d_K}`

This inequality needs a citation. The standard reference is Devroye and Györfi (1985) "Nonparametric Density Estimation."

**Suggested addition:** cite Devroye & Györfi 1985 or a more accessible reference.

### 3. Holm-Bonferroni procedure used in experiments without cite (both papers)

**Exact text:** `Holm-Bonferroni p < 0.0001` appears throughout both papers.

The Holm-Bonferroni step-down procedure needs a citation: Holm, S. (1979). "A simple sequentially rejective multiple test procedure." *Scandinavian Journal of Statistics*, 6(2), 65-70.

### 4. Wilcoxon signed-rank test used without cite (8-page paper, Tables 2 and 3)

**Exact text:** `Wilcoxon signed-rank, Holm-Bonferroni`

The Wilcoxon signed-rank test should cite the original: Wilcoxon, F. (1945). "Individual comparisons by ranking methods." *Biometrics Bulletin*, 1(6), 80-83.

### 5. `perrault2020statistical` claim about O(d log T / Δ) Bayesian regret (both papers)

**Exact text (4-page):** `it achieves O(d log T / gap) Bayesian regret \citep{perrault2020statistical}`

The Perrault 2020 paper (NeurIPS 2020) is correctly identified, but the paper actually proves a tighter *instance-dependent* bound and is more nuanced than a simple O(d log T / Δ). The citation is broadly correct but the formula as stated simplifies the paper's contribution. This is acceptable but the authors should double-check the exact form of the bound stated in the paper to ensure the claim is accurately attributable.

---

## Metadata issues in references.bib

### 1. `bayley2025jumpstart` — Wrong title and wrong author list

**Bib entry:**
- Title: "Robustness of {LLM}-Initialized Bandits Under Noisy Priors"
- Authors: "Bayley, Eric and Zhu, Adam and Aoki, Kenji and Cao, Robert and Wilson, Mark"

**Actual paper (GenAIRecP 2025 workshop at KDD 2025):**
- Title: "Robustness of LLM-Initialized Bandits **for Recommendation** Under Noisy Priors"
- Authors: Adam Bayley, Kevin H. Wilson, Yanshuai Cao, Raquel Aoki, Xiaodan Zhu

The title is missing "for Recommendation." The author list has wrong first names and is partially fabricated: "Eric" should be "Adam"; "Zhu, Adam" should be "Zhu, Xiaodan"; "Aoki, Kenji" should be "Aoki, Raquel"; "Cao, Robert" should be "Cao, Yanshuai"; "Wilson, Mark" should be "Wilson, Kevin H." The venue is listed as "Genai Personalization Workshop" — corrected name is "GenAIRecP 2025 (Workshop on Generative AI for Recommender Systems and Personalization, co-located with KDD 2025)."

**Fixed entry:**
```bibtex
@inproceedings{bayley2025jumpstart,
  author    = {Bayley, Adam and Wilson, Kevin H. and Cao, Yanshuai and Aoki, Raquel and Zhu, Xiaodan},
  title     = {Robustness of {LLM}-Initialized Bandits for Recommendation Under Noisy Priors},
  booktitle = {Workshop on Generative AI for Recommender Systems and Personalization (GenAIRecP), KDD},
  year      = {2025}
}
```

### 2. `cao2026libra` — Wrong author list and misleading journal field

**Bib entry authors:** `Cao, Yuxin and Gao, Lin and Keyvanshokooh, Esmaeil and Ma, Will`
**Actual authors:** Junyu Cao, Ruijiang Gao, Esmaeil Keyvanshokooh, Jianhao Ma

"Yuxin" → "Junyu"; "Lin" → "Ruijiang"; "Will" → "Jianhao". Keyvanshokooh is correct. The `journal` field reads "arXiv preprint" with no arXiv ID — should be `arXiv:2601.11905`.

### 3. `russo2014infotheoretic` — Year mismatch in key vs. entry

The key uses `russo2014` but the actual JMLR publication year is 2016 (volume 17). The arXiv preprint appeared in 2014. Since the paper is published in JMLR 2016, the bib year = 2016 is correct; but the key name `russo2014infotheoretic` will confuse readers who look for "Russo 2014" in the reference list and find a 2016 date. Consider renaming the key to `russo2016infotheoretic` and updating all `\cite` calls, or adding a note that 2014 is the arXiv date.

### 4. `kapoor2019corruption` — Wrong `booktitle` field

The entry uses `booktitle = {Machine Learning}`, but "Machine Learning" is a journal (Springer), not a conference. The paper was published in *Machine Learning*, volume 108, 2019. Use:
```bibtex
@article{kapoor2019corruption,
  journal = {Machine Learning},
  volume  = {108},
  pages   = {687--715},
  ...
}
```
(This entry is dead in both papers, so this only matters if it gets added back.)

### 5. `sun2025tsllm` — No arXiv ID, author field uses `others`

The entry `author = {Sun, Zixin and others}` and `journal = {arXiv preprint}` with no ID. If this is arXiv:2502.01118, the first author is Jiahang Sun (not Zixin Sun) and the full author list should be supplied. Fix the ID and author list as described under Mis-attributions above.

---

## Suspicious 2025/2026 citations

| Key | Verdict |
|---|---|
| `bayley2025jumpstart` | EXISTS — GenAIRecP 2025 workshop paper at KDD 2025. But author names and title are wrong in bib (see Metadata Issues). |
| `sun2025tsllm` | LIKELY WRONG PAPER — The bib entry matches a survey by Bouneffouf & Feraud (arXiv:2505.13355). The TS-LLM algorithm described in the text matches arXiv:2502.01118 by Sun et al. Needs correction. |
| `cao2026libra` | EXISTS — arXiv:2601.11905, January 2026. But the characterization "robustness to adversarial LLM advice" is wrong; the paper is about bandit recourse for personalized treatment planning. Author names in bib are wrong. |

---

## Decorative citations

### 1. `\citep{srinivas2010gpucb,russo2014infotheoretic}` in "Related work" paragraph (4-page paper)

**Text:** `GP-TS \citep{srinivas2010gpucb,russo2014infotheoretic} uses a kernel for continuous contexts`

The Russo & Van Roy paper (`russo2014infotheoretic`) analyzes Thompson sampling via information theory for general settings; it is not the primary source for "GP-TS uses a kernel for continuous contexts." The Srinivas 2010 GP-UCB paper is the correct cite for that claim. `russo2014infotheoretic` is doing no work here and should be dropped from this particular citation group (keep it if it is cited elsewhere in the paper for its own contribution, but in 4-page it appears only here).

### 2. `\citealp{srinivas2010gpucb}` in 8-page discussion section (sec. 6)

**Text:** `a bound of the form O(√T) or O(log T) that captures the dimension reduction ... (analogous to the information gain γ_T in GP-TS, \citealp{srinivas2010gpucb}) is open`

Srinivas 2010 is correctly cited for GP-TS with information gain. Not decorative — keep.

---

## Recommended bib additions

The following standard references are invoked by name but not cited:

1. **Berry-Esseen theorem** (invoked in Proposition 4 proof in both papers):
   ```bibtex
   @article{berry1941accuracy,
     author  = {Berry, Andrew C.},
     title   = {The accuracy of the {G}aussian approximation to the sum of independent variates},
     journal = {Transactions of the American Mathematical Society},
     volume  = {49}, pages = {122--136}, year = {1941}
   }
   ```

2. **Holm 1979** (Holm-Bonferroni procedure used in all experiments):
   ```bibtex
   @article{holm1979simple,
     author  = {Holm, Sture},
     title   = {A simple sequentially rejective multiple test procedure},
     journal = {Scandinavian Journal of Statistics},
     volume  = {6}, number = {2}, pages = {65--70}, year = {1979}
   }
   ```

3. **Wilcoxon 1945** (used in 8-page Tables 2 and 3):
   ```bibtex
   @article{wilcoxon1945individual,
     author  = {Wilcoxon, Frank},
     title   = {Individual comparisons by ranking methods},
     journal = {Biometrics Bulletin},
     volume  = {1}, number = {6}, pages = {80--83}, year = {1945}
   }
   ```

---

## Action list (top 10 priority fixes, ranked)

1. **[CRITICAL] Fix `verma2023bayesian` completely.** The bib entry has the wrong title, wrong authors, and wrong year for arXiv:2312.12676 (actual authors: Sandberg, Åkerblom, Haghir Chehreghani; actual title: "Bayesian Analysis of Combinatorial Gaussian Process Bandits"; accepted ICLR 2025). This paper is the formal reduction target for Theorem 1 in both papers. A reviewer who checks it will find a mismatch immediately.

2. **[CRITICAL] Fix the characterization of `cao2026libra`.** The paper is about LLM-guided bandit recourse for personalized treatment planning, not "robustness to adversarial LLM advice." Every sentence in both papers that ascribes adversarial robustness to LIBRA is wrong. Also correct the author list (Junyu Cao, Ruijiang Gao, Esmaeil Keyvanshokooh, Jianhao Ma) and add the arXiv ID (2601.11905).

3. **[CRITICAL] Resolve `sun2025tsllm` identity.** The bib entry currently points to a survey by Bouneffouf & Feraud (arXiv:2505.13355), not to an original TS-LLM algorithm paper. The TS-LLM algorithm described in the text and used as an experimental baseline most likely corresponds to arXiv:2502.01118 (Jiahang Sun et al.). Verify which paper was actually implemented and correct the entry.

4. **[HIGH] Fix `bayley2025jumpstart` author names and title.** All five author first names are wrong in the bib. Correct names: Adam Bayley, Kevin H. Wilson, Yanshuai Cao, Raquel Aoki, Xiaodan Zhu. Full title adds "for Recommendation." Venue: GenAIRecP workshop at KDD 2025.

5. **[HIGH] Fix `kveton2023mixed` author list.** Authors are Aouali, Kveton, Katariya — not Kveton, Manzil, Katariya, Szepesvári. The incorrect names belong to a different paper.

6. **[MEDIUM] Add cite for Holm-Bonferroni.** Used in virtually every result table in both papers with no citation. Add `\citep{holm1979simple}` at first use. A statistical reviewer will flag this.

7. **[MEDIUM] Add cite for Wilcoxon signed-rank test** (8-page Tables 2 and 3). Add `\citep{wilcoxon1945individual}`.

8. **[MEDIUM] Add cites for Berry-Esseen and Devroye-Györfi** in the Proposition 4 proof (both papers). Both inequalities are invoked by name without attribution.

9. **[LOW] Purge 17 dead bib entries.** Clean the bib file of the 17 unused entries before submission to avoid reviewer suspicion of padding.

10. **[LOW] Rename key `russo2014infotheoretic` to `russo2016infotheoretic`** to match the JMLR 2016 publication year, or add a note. Minor but avoids reader confusion when a "2014" key resolves to a 2016 paper.
