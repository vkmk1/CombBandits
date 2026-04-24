# Writing Style Guide for Bandits/Online Learning Theory Papers

## Comprehensive analysis based on papers by Bubeck, Cesa-Bianchi, Agrawal, Goyal, Chen, Combes, Lattimore, Szepesvari, and other top researchers. Designed to produce writing that reads like a senior researcher, not an AI.

---

## 1. TONE AND VOICE

### The Register: Controlled Confidence

Top bandits papers occupy a narrow tonal band: **technically confident but intellectually modest**. The voice is that of a mathematician explaining results to a knowledgeable peer, not a salesperson pitching a product.

**What this sounds like in practice:**

- Bubeck & Cesa-Bianchi (2012) open their survey: "Multi-armed bandit problems are the most basic examples of sequential decision problems with an exploration-exploitation trade-off." -- No fanfare. No "In recent years, there has been a surge of interest in..." Just a direct, factual positioning statement.

- Agrawal & Goyal (2014) in their Thompson Sampling paper: "Thompson Sampling is one of the oldest heuristics for multi-armed bandit problems." -- Again, a plain factual statement. No "groundbreaking" or "revolutionary."

- Chen et al. (2016) on CMAB: "We define a general framework for a large class of combinatorial multi-armed bandit (CMAB) problems." -- Direct assertion of what the paper does. No throat-clearing.

- Combes et al. (2015): "This paper investigates stochastic and adversarial combinatorial multi-armed bandit problems." -- One sentence, done.

### Key Tonal Principles:

1. **State facts, not opinions about your own work.** Senior researchers say "We prove a regret bound of O(sqrt(T))" not "We achieve a remarkable improvement in regret."

2. **Let the mathematics speak.** If your bound is better, state both bounds and let the reader see the improvement. Don't editorialize about how significant it is.

3. **Use "we" consistently.** Even single-author papers in this field use "we" (the mathematical "we" that includes the reader). Never "I" in the body. Never "the authors."

4. **Hedging is precise, not vague.** Senior researchers hedge with mathematical precision:
   - GOOD: "To our knowledge, ours are the first non-trivial regret bounds for TS for the contextual bandits problem." (Agrawal & Goyal -- specific claim, qualified by "to our knowledge")
   - GOOD: "This essentially solves the COLT 2012 open problem." (Agrawal & Goyal -- "essentially" is doing precise work here; the solution is not exact but close)
   - BAD: "Our method potentially offers significant improvements" (vague hedge + vague claim)

5. **No enthusiasm markers.** You will never find "interestingly," "remarkably," "notably," "importantly," or "crucially" in a Bubeck or Lattimore paper as sentence openers. If something is interesting, the reader will notice. If you must flag something, use the construction "We remark that..." or "It is worth noting that..." sparingly.

### Formality Level:

- More formal than a blog post, less formal than a pure mathematics journal
- Contractions are never used
- Colloquial language is acceptable only when introducing intuition: Bubeck writes about "the colloquial term for a slot machine ('one-armed bandit' in American slang)" -- but this is explanatory, not casual
- Humor is essentially absent from the body text (occasionally appears in acknowledgments)

---

## 2. SENTENCE STRUCTURE

### Sentence Length and Rhythm

Bandits papers use a **medium sentence length** -- typically 15-25 words, with occasional longer sentences (30-40 words) for complex technical statements and short sentences (8-12 words) for emphasis or transitions.

**The characteristic rhythm is: medium, medium, long, short.**

Example from Bubeck & Cesa-Bianchi (2012) introduction:
> "A multi-armed bandit problem (or, simply, a bandit problem) is a sequential allocation problem defined by a set of actions. [20 words] At each time step, a unit resource is allocated to an action and some observable payoff is obtained. [17 words] The goal is to maximize the total payoff obtained in a sequence of allocations. [14 words]"

Three medium sentences, each building on the last. No sentence tries to do too much.

### Active vs. Passive Voice

**The field strongly prefers active voice with "we" as subject:**

- "We prove..." / "We show..." / "We design and analyze..." / "We derive..."
- "We propose ESCB, an algorithm that efficiently exploits the structure..." (Combes et al.)
- "We provide the first theoretical guarantees for the contextual version of Thompson Sampling." (Agrawal & Goyal)

**Passive voice appears in specific contexts:**

1. When describing the state of the art (what has been done by others): "Several studies have empirically demonstrated the efficacy of TS." (Agrawal & Goyal)
2. When stating known results: "It has been shown that..." / "The problem was first studied by..."
3. In problem setup: "A reward is observed..." / "The context vectors are chosen by an adversary..."

**Never use passive voice for your own contributions.** "A novel algorithm is proposed" is the hallmark of a weak paper. Senior researchers own their contributions.

### Transitions Between Sections

Senior researchers use **structural transitions, not verbal ones:**

- A new paragraph signals a new idea. They do not write "Furthermore," "Moreover," "Additionally," at the start of every paragraph.
- Between major ideas, they use a single bridge sentence: "We now describe a few concrete examples in various domains." (Bubeck -- transitioning from abstract formulation to applications)
- Section transitions are handled by the section heading itself. No "In the following section, we will discuss..." Just end the current section and start the next.

**Common transition patterns:**

- "We now turn to..." (introducing a new topic)
- "We first consider... We then address..." (signposting a two-part discussion)
- "We are now in a position to state our main result." (before a key theorem)
- "The key observation is that..." (transitioning from setup to insight)
- "Observe that..." (drawing attention to a consequence)

### How Theorems Are Introduced

**The pattern is: intuition paragraph, then formal statement.**

Combes et al. after discussing their algorithm:
> "The next theorem provides generic properties of our indexes. An important consequence of these properties is that the expected number of times where [the index] underestimates [the optimal reward] is finite, as stated in the corollary below."

Then: **Theorem 3.** *For all theta in Theta, ...*

Agrawal & Goyal provide a paragraph discussing what the result means before stating it formally. After the theorem, they discuss its implications.

**Never introduce a theorem with "We have the following result:" and nothing else.** Always give the reader a reason to care about the theorem before stating it.

**The post-theorem discussion** typically:
- Compares the bound to known lower bounds
- Identifies which terms are tight and which may be improvable
- Discusses special cases that recover known results
- Points out surprising aspects of the result

Example from Combes et al.: "Observe first that this optimization problem is a semi-infinite linear program which can be solved for any fixed theta, but its optimal value is difficult to compute explicitly. Determining how c(theta) scales as a function of the problem dimensions d and m is not obvious."

---

## 3. AI/ML WRITING TROPES TO AVOID -- THE COMPREHENSIVE BLACKLIST

### Category A: LLM Signature Words (empirically measured increases in academic text post-ChatGPT)

These words have been quantitatively shown to surge in frequency in academic papers after ChatGPT's release (Kobak et al. 2024, PMC studies). Using them is the single fastest way to make your paper sound AI-generated.

**NEVER use these words:**
- "delve" / "delves" / "delving" (28x increase in academic usage)
- "intricate" / "intricacies" (dramatic increase)
- "commendable"
- "meticulous" / "meticulously"
- "showcasing" (10x increase)
- "underscores" as a verb (11x increase)
- "landscape" (metaphorical -- "the landscape of bandit algorithms")
- "tapestry"
- "realm" ("in the realm of online learning")
- "multifaceted"
- "nuanced" (as a generic intensifier)
- "pivotal"
- "comprehensive" (when describing your own work)
- "groundbreaking"
- "cutting-edge"
- "innovative" (describing your own work)
- "elucidate"
- "shed light on"
- "pave the way"
- "holistic"

**Replacements that real researchers use:**
- Instead of "delve into": "study" / "analyze" / "investigate" / "examine"
- Instead of "intricate": "complex" (or better, just describe the specific difficulty)
- Instead of "showcasing": "demonstrating" / "showing"
- Instead of "underscores": "shows" / "confirms" / "highlights" (sparingly)
- Instead of "landscape": just say what you mean -- "the literature on bandits" / "existing work"
- Instead of "shed light on": "clarify" / "explain" / "reveal"

### Category B: Overwrought Framing Phrases

These are phrases that scream "this was written by an AI or an inexperienced writer":

- "In recent years, there has been a surge/growing interest in..."
- "X has emerged as a powerful/promising paradigm for..."
- "X has garnered significant attention..."
- "X plays a crucial/pivotal role in..."
- "It is imperative to..."
- "This serves as a testament to..."
- "The burgeoning field of..."
- "A plethora of" (use "many" or "several" or be specific)
- "Myriad" as an adjective
- "Leverage" as a verb (use "use" or "exploit")
- "Harness the power of"
- "Paradigm shift"
- "State-of-the-art" as a noun ("achieves state-of-the-art" -- use "achieves the best known results" or state the specific numbers)
- "In this regard"
- "It is worth mentioning that" (usually means you should delete the sentence)
- "It should be noted that" (same)
- "Needless to say" (then don't say it)

**How senior researchers actually open papers instead:**

- "We study [problem]." (Lattimore & Szepesvari style)
- "[Problem] is a fundamental question in [field]." (direct)
- "The [problem] has been studied extensively [citations], but [gap]." (efficient gap identification)
- "[Specific result] was shown by [Author] [year]. We extend this to [setting]." (clean motivation)

### Category C: Hollow Intensifiers and Padding

Words and phrases that add no information:

- "very" (delete it; if the adjective is too weak without "very," choose a stronger adjective)
- "quite" (same)
- "extremely"
- "highly"
- "significantly" (unless referring to statistical significance)
- "notably" (as a sentence opener)
- "importantly" (as a sentence opener)
- "interestingly" (let the reader decide)
- "remarkably" (same)
- "crucially" (same)
- "it is important to note that" (just state the thing)
- "it turns out that" (weak; just state the result)

### Category D: Structural AI Tells

These patterns are structural rather than lexical:

1. **The numbered contribution list with superlatives.** AI writes: "Our contributions are: (1) We propose a novel framework... (2) We provide a comprehensive analysis... (3) We conduct extensive experiments..." Senior researchers either use a bullet list with plain language or weave contributions into prose.

2. **The triple adjective.** AI loves: "a robust, scalable, and efficient algorithm." Real papers: "an efficient algorithm" (if it's robust and scalable, demonstrate that; don't assert it with adjectives).

3. **The "bridge sentence" between every paragraph.** AI writes transition sentences like "Building on the above, we now..." between every paragraph. Real papers just start the next paragraph.

4. **Over-signposting.** "In Section 2, we present... In Section 3, we analyze... In Section 4, we demonstrate..." One sentence of paper organization at the end of the introduction is normal. A full paragraph is excessive.

5. **The enthusiasm arc.** AI often opens with excitement, builds enthusiasm, and closes with a visionary statement about "future directions that could transform the field." Senior researchers open with a problem, solve it, and close with honest limitations and concrete open questions.

6. **Excessive hedging on simple claims.** AI writes: "Our approach may potentially offer improvements in certain scenarios." If you have a theorem, state it. If you have experimental evidence, present it. Hedge only on genuinely uncertain claims.

7. **Thanking the reader.** AI sometimes writes things like "We hope this work inspires further research." Real papers never do this.

---

## 4. WHAT MAKES WRITING FEEL "HUMAN" AND "SENIOR RESEARCHER"

### Signals of Expertise

1. **Casual precision about what is known.** Senior researchers drop references efficiently:
   - "The problem was first studied by Thompson (1933)." (Agrawal & Goyal)
   - "This is among the most important and widely studied version of the contextual bandits problem." (matter-of-fact assessment, not argued)
   - "These assumptions are required to make the regret bounds scale-free, and are standard in the literature on this problem." (Agrawal & Goyal -- signaling that the authors know what is standard)

2. **Acknowledging the community.** References to open problems and prior work are specific:
   - "Some of these questions and difficulties were also formally raised as a COLT 2012 open problem (Chapelle & Li, 2012)." (Agrawal & Goyal)
   - "This essentially solves the COLT 2012 open problem." (bold but earned claim)

3. **Technical humility through specificity.** Rather than claiming broad impact, they bound their own results precisely:
   - "which is within a factor of sqrt(d) of the information-theoretic lower bound for this problem" (Agrawal & Goyal -- they tell you exactly how suboptimal they are)
   - "To determine whether there is a gap between computational and information theoretic lower bound for this problem is an intriguing open question." (honestly flagging what they don't know)

4. **Offhand expertise in remarks.** Senior researchers drop observations that show deep familiarity:
   - "Observe first that this optimization problem is a semi-infinite linear program..." (Combes et al. -- the "observe" signals this is easy for the cognoscenti)
   - "Further remark that if M is the set of singletons (classical bandit), Theorem 1 reduces to the Lai-Robbins bound." (Combes -- showing the result generalizes known results, stated casually)

5. **Appropriate use of "natural."** Senior researchers describe things as "natural" to signal that the construction follows obviously from the setup: "A natural way to devise algorithms based on indexes is to select in each round the arm with the highest index." This is not padding; it tells the reader "this is the obvious thing to try."

### Handling Related Work Gracefully

**DO:** Cite matter-of-factly and compare precisely.
- "Granmo (2010) and May et al. (2011) provided weak guarantees, namely, a bound of o(T) on the expected regret in time T." (Agrawal & Goyal -- precise, factual, no shade)
- "Previous contributions on stochastic combinatorial bandits focused on specific combinatorial structures, e.g. m-sets [6], matroids [7], or permutations [8]." (Combes -- efficient survey)
- "Our algorithms improve over LLR and CUCB by a multiplicative factor of sqrt(m)." (Combes -- concrete comparison)

**DON'T:**
- Don't be dismissive: "Previous work suffered from several limitations" (too vague and condescending)
- Don't be sycophantic: "The seminal work of [X] laid the groundwork for..."
- Don't list papers without saying what they did: "[1,2,3,4,5,6,7] study bandits" (useless)
- Don't use "however" to transition from related work to your contribution as a dramatic reveal

**The standard pattern for related work in bandits papers:**
1. Cite the foundational results briefly (Lai & Robbins, Auer et al., etc.)
2. Cite the most relevant recent work with specific results: "[Author] achieved O(X) regret for [setting]."
3. State how your work differs: "Our bounds improve by a factor of..." or "Unlike [Author], we do not require assumption X."

### Discussing Limitations

Senior researchers discuss limitations **as technical observations, not confessions:**

- "However this algorithm suffers from two problems: it is computationally infeasible for large problems since it involves solving [the optimization] T times, furthermore the algorithm has no finite time performance guarantees, and numerical experiments suggest that its finite time performance on typical problems is rather poor." (Combes et al. discussing a known algorithm -- factual, specific, no drama)
- "To determine whether there is a gap between computational and information theoretic lower bound for this problem is an intriguing open question." (Agrawal & Goyal -- framing a limitation as an open question for the community)

**Never write:** "A limitation of our work is..." as a confessional. Instead, state the technical boundary clearly and, if appropriate, suggest what would be needed to extend the result.

---

## 5. STRUCTURAL CONVENTIONS

### Abstract Structure (for a 10-page ICML/NeurIPS bandits paper)

**Length:** 100-200 words. Most top bandits papers use 120-170 words.

**Structure (typically 4-6 sentences):**

1. **Problem statement** (1-2 sentences): What problem are you studying? Place it in the known landscape.
   - "This paper investigates stochastic and adversarial combinatorial multi-armed bandit problems." (Combes -- 1 sentence)
   - "Thompson Sampling is one of the oldest heuristics for multi-armed bandit problems. It is a randomized algorithm based on Bayesian ideas, and has recently generated significant interest..." (Agrawal & Goyal -- 2 sentences combining problem + motivation)

2. **Gap/motivation** (0-1 sentences): What was missing? Keep it brief.
   - "However, many questions regarding its theoretical performance remained open." (Agrawal & Goyal)

3. **Contribution** (2-3 sentences): What do you do? State specific results with specific bounds.
   - "In this paper, we design and analyze a generalization of Thompson Sampling algorithm for the stochastic contextual multi-armed bandit problem with linear payoff functions..." (Agrawal & Goyal)
   - "We propose ESCB, an algorithm that efficiently exploits the structure of the problem and provide a finite-time analysis of its regret." (Combes)

4. **Key result** (1 sentence): The punchline -- your main bound or finding.
   - "We prove a high probability regret bound of O-tilde(d^{3/2} sqrt(T))..." (Agrawal & Goyal)
   - "ESCB has better performance guarantees than existing algorithms, and significantly outperforms these algorithms in practice." (Combes)

**What NOT to include in the abstract:**
- Future work
- "Extensive experiments on diverse benchmarks" (be specific about experiments or omit)
- Philosophical motivation for why the problem matters
- More than one sentence of background

### Introduction Structure (for a 10-page paper)

**Length:** 1.5-2.5 pages. Typically 5-8 paragraphs.

**Paragraph-by-paragraph structure:**

**Para 1: Problem definition and context** (4-6 sentences)
Define the problem concretely. Give the mathematical setup in plain language. Place it in the known taxonomy.
- Bubeck opens with: "A multi-armed bandit problem (or, simply, a bandit problem) is a sequential allocation problem defined by a set of actions..." -- definition first, then elaboration.
- Combes opens with: "Multi-Armed Bandits (MAB) problems constitute the most fundamental sequential decision problems with an exploration vs. exploitation trade-off." -- one sentence positioning.

**Para 2: Motivation and applications** (3-5 sentences)
Why does this problem matter? Cite 2-3 concrete applications. Keep it brief.
- Bubeck: "Although the original motivation of Thompson (1933) for studying bandit problems came from clinical trials... modern technologies have created many opportunities for new applications."

**Para 3: State of the art** (4-8 sentences)
What is known? Cite the most relevant results with their specific bounds. Build up to the gap.
- Agrawal & Goyal spend a full paragraph reviewing Thompson Sampling literature, citing specific empirical and theoretical results.

**Para 4: The gap** (2-4 sentences)
What is unknown or unsatisfactory? This sets up your contribution.
- "However, the theoretical understanding of TS is limited." (Agrawal & Goyal -- one clear sentence)
- "But, many questions regarding theoretical analysis of TS remained open, including high probability regret bounds, and regret bounds for the more general contextual bandits setting." (Agrawal & Goyal -- specific list of open questions)

**Para 5: Your contribution** (4-8 sentences)
What do you do? State your main results with specific bounds. This is the core of the introduction.
- "In this paper, we use novel martingale-based analysis techniques to demonstrate that TS... achieves high probability, near-optimal regret bounds for stochastic contextual bandits with linear payoff functions." (Agrawal & Goyal)
- Then: "We provide a regret bound of O-tilde(d^{3/2} sqrt(T))..." (the specific result)

**Para 6 (optional): Paper organization** (2-3 sentences)
Brief signposting. One or two sentences max.
- "Our version of Thompson Sampling algorithm for the contextual MAB problem, described formally in Section 2.2, uses Gaussian prior and Gaussian likelihood functions." (Agrawal & Goyal -- weaving organization into a content sentence)

**What NOT to do in the introduction:**
- Don't spend more than 1/3 of the introduction on background/motivation
- Don't list contributions as a bulleted list of 5+ items with adjectives
- Don't end with "The rest of the paper is organized as follows: Section 2..."  as a standalone paragraph
- Don't preview experiments in the introduction (unless the empirical finding IS the contribution)

### Problem Setup Section

**Key convention:** This section is purely mathematical. No motivation, no discussion, just definitions.

- Start with the model: "There are N arms. At time t = 1, 2, ..., a context vector b_i(t) in R^d is revealed for every arm i." (Agrawal & Goyal)
- Define all notation systematically
- State all assumptions explicitly: "We assume that..." followed by the mathematical condition
- End with the formal definition of regret

**Tone in this section:** Maximum formality. Short, declarative sentences. Every sentence either defines something or states an assumption.

### Main Results Section

**Convention:** State the theorem, then discuss it.

From Combes et al. (Section 2, "Contribution and Related Work"):
- Bold label: **"Contribution."** followed by the result statement
- Then: **"Related work."** followed by comparison to prior work
- This interleaving of contributions and related work is common in bandits papers

**How to present a regret bound:**

1. State the theorem formally
2. Immediately after: explain what the bound means in words
3. Compare to the lower bound (if known)
4. Identify which terms are tight
5. Discuss special cases
6. Compare to competing algorithms (often in a table)

### Experiments Section (if present)

Many theory papers in bandits include a short experiments section (0.5-1 page).

**Convention:**
- "Numerical experiments for some specific combinatorial problems are presented..." (Combes -- experiments in supplementary)
- When experiments appear in the main paper, they validate the theory, not explore new territory
- Experiments show: (a) your algorithm vs baselines, (b) regret vs time horizon T, (c) sometimes parameter sensitivity

### Conclusion

**Bandits papers have short conclusions.** Typically 1 paragraph, 4-8 sentences.

**Structure:**
1. One sentence summarizing what was done
2. One sentence stating the main result
3. 1-3 sentences on open questions (specific, not vague "future work")

**Never write:** "In conclusion, we have presented a novel framework that addresses the intricate challenges of..." Just state the result and the open problems.

---

## 6. CITATION STYLE

### The Two Patterns and When to Use Each

Bandits papers use author-year citation (not numeric) and follow this convention:

**Pattern 1: Author as subject -- "Author (year)"**
Use when the author is doing something in your narrative:
- "Thompson (1933) proposed a Bayesian heuristic..."
- "Agrawal & Goyal (2012) provided optimal regret bounds on the expected regret."
- "Chapelle & Li (2011) demonstrate that, empirically, TS achieves regret comparable to the lower bound of Lai & Robbins (1985)."
- "Kaufmann et al. (2012) do a thorough comparison..."

**Pattern 2: Parenthetical -- "(Author, year)" or "(Author & Author, year)"**
Use when citing as evidence or reference, not as the subject:
- "...in the context of reinforcement learning, e.g., in Wyatt (1997); Ortega & Braun (2010); Strens (2000)."
- "...is standard in the existing literature on contextual multi-armed bandits, e.g. (Auer, 2002; Filippi et al., 2010; Chu et al., 2011)."
- "This realizability assumption is standard in the existing literature (Auer, 2002; Filippi et al., 2010)."

**Key details:**
- With 2 authors: always write both names: "Agrawal & Goyal (2013)" or "(Agrawal & Goyal, 2013)"
- With 3+ authors: "Chen et al. (2016)" or "(Chen et al., 2016)"
- Multiple citations in parentheses: semicolon-separated, typically chronological: "(Lai & Robbins, 1985; Auer et al., 2002; Bubeck & Cesa-Bianchi, 2012)"
- With "e.g.": "(see, e.g., Lattimore & Szepesvari, 2020)" or "e.g. (Author, year)"

**Note on numbered citations:** Some bandits papers in NeurIPS use numeric citations [1,2,3]. In that case, you write "Combes et al. [5]" for subject citations and "[5]" or "[5,6,7]" for parenthetical. Combes et al. (2015) in their NeurIPS paper use numeric: "Multi-Armed Bandits (MAB) problems [1] constitute..." -- The ICML 2026 format will specify which style to use.

### Citation Density

Top bandits papers cite **30-60 references** for a 10-page paper. Citation density is highest in the introduction and related work sections. The problem setup and main results sections have few citations (only to results being compared against or techniques being used).

---

## 7. FIGURE AND TABLE CONVENTIONS

### Regret Plots

The standard experimental figure in a bandits paper is a **regret vs. time plot**:

- X-axis: time horizon T (or number of rounds)
- Y-axis: cumulative regret (or sometimes regret/T for average regret)
- Multiple lines: one per algorithm, with your algorithm in a distinctive color
- Log-log or log-linear scale is common for showing polynomial vs logarithmic regret
- Error bars or shaded confidence regions from multiple runs
- Legend inside the plot or to the right

### Comparison Tables

Combes et al. use tables to compare regret bounds across algorithms:

| Algorithm | Regret |
|-----------|--------|
| LLR [9]   | O(m^3 d Delta_max / Delta_min^2 * log(T)) |
| CUCB [10] | O(m^2 d / Delta_min * log(T)) |
| ESCB (Theorem 5) | O(sqrt(md) / Delta_min * log(T)) |

**Convention:** Comparison tables show the regret bound in O-notation for each algorithm, making the improvement immediately visible. Lower bounds are often included as a row.

### General Figure Guidelines

- Figures are clean and minimal -- no grid lines, no unnecessary decoration
- Colors should be distinguishable in grayscale (for printing)
- Subfigures labeled (a), (b), (c) for different experimental settings
- Caption is self-contained: a reader should understand the figure from caption alone
- Typically 1-3 figures for a theory paper, 4-6 for an empirical paper

---

## 8. SPECIFIC PHRASING PATTERNS FROM TOP PAPERS

### How to Introduce Your Algorithm

- "We propose ESCB (Efficient Sampling for Combinatorial Bandits), an algorithm..." (Combes)
- "We design and analyze a natural generalization of Thompson Sampling (TS) for contextual bandits." (Agrawal & Goyal)
- Chen: "We define a general framework for a large class of combinatorial multi-armed bandit (CMAB) problems, where the reward... only needs to satisfy two mild assumptions."

### How to State a Bound

- "We prove a high probability regret bound of O-tilde(d^{3/2} sqrt(T))." (direct)
- "...which is the best regret bound achieved by any computationally efficient algorithm for this problem." (comparison to landscape)
- "...and is within a factor of sqrt(d) of the information-theoretic lower bound." (gap to optimality)

### How to Discuss Open Problems

- "To determine whether there is a gap between computational and information theoretic lower bound for this problem is an intriguing open question." (Agrawal & Goyal)
- "This result is intuitive since d - m is the number of parameters not observed when selecting the optimal arm." (explaining why a result makes sense)

### How to Introduce Assumptions

- "We assume that [math condition]. This assumption is satisfied whenever [concrete case]." (Agrawal & Goyal pattern)
- "These assumptions are required to make the regret bounds scale-free, and are standard in the literature on this problem." (acknowledging assumptions are standard)
- Chen: "...which allow a large class of nonlinear reward instances" (immediately justifying why assumptions are not restrictive)

### How to Handle "To Our Knowledge" Claims

- "To our knowledge, ours are the first non-trivial regret bounds for TS for the contextual bandits problem." (Agrawal & Goyal -- specific, qualified)
- "To our knowledge, such lower bounds have not been proposed in the case of stochastic combinatorial bandits." (Combes -- specific gap claim)

---

## 9. QUICK REFERENCE: THE 10 COMMANDMENTS

1. **Open with the problem, not with recent trends.** First sentence defines or names the problem.
2. **State results with specific bounds.** No "significant improvement" without numbers.
3. **Use "we" + active verbs.** "We prove," "We show," "We design."
4. **Hedge only with precision.** "To our knowledge" is acceptable; "may potentially" is not.
5. **No AI signature words.** Delete every instance of delve, intricate, landscape, tapestry, underscore, showcase, pivotal, multifaceted, commendable, meticulous, elucidate, holistic, groundbreaking, cutting-edge.
6. **Comparison is the argument.** State your bound. State the previous best. State the lower bound. Done.
7. **Let structure do the work.** Paragraph breaks and section headings replace verbal transitions.
8. **Definitions before results, intuition before formalism.**
9. **Short conclusion with concrete open questions.** No visionary statements.
10. **Every sentence earns its place.** If a sentence can be deleted without information loss, delete it.

---

## 10. SELF-EDITING CHECKLIST

Before finalizing any section, check:

- [ ] Does the first sentence of the abstract name the problem directly?
- [ ] Are all contribution claims backed by a specific theorem/bound number?
- [ ] Is every "novel" / "significant" / "comprehensive" deleted or replaced?
- [ ] Does the introduction spend less than 40% of its length on background?
- [ ] Are theorems preceded by a motivation paragraph?
- [ ] Are theorems followed by a discussion paragraph?
- [ ] Is every citation doing work (not decorative)?
- [ ] Are there fewer than 3 uses of "however" per page?
- [ ] Are there zero uses of: delve, intricate, landscape, tapestry, multifaceted, pivotal, elucidate, groundbreaking, holistic, leverage, harness, paradigm?
- [ ] Does the conclusion fit in one paragraph?
- [ ] Are open questions specific ("Is the factor of sqrt(d) necessary?") not vague ("Future work could explore...")?
- [ ] Could a reviewer identify your main result from the abstract alone?
