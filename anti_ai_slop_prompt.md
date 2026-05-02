# Human Writing / Anti-AI-Slop Prompt (Research Paper Edition)

Adapted from the mcp-audit repo's bug-bounty report writing rules, retooled for
ML conference papers. These rules are mandatory when writing or rewriting any
paper draft in this repo. No exceptions.

## Voice and Tone

- Write in the mathematical "we." Never "the authors," never "I." First person
  plural, always.
- Short sentences. If a sentence has more than two commas, split it.
- Be direct. "Our bound is tight up to log factors" beats "our analysis
  potentially offers meaningful improvements in certain regimes."
- Sound like a senior researcher explaining to a knowledgeable peer, not a
  student pitching for admiration.
- State what you proved and what you ran. Not what could theoretically happen.
  "Theorem 3 gives O(sqrt(mT log T))" beats "our approach may yield favorable
  regret in settings of interest."
- Don't hedge things you proved. If you have a theorem, state it without
  "may," "could potentially," or "appears to."
- Never editorialize on your own work. No "surprisingly," "remarkably,"
  "notably," "importantly," "crucially," or "interestingly" as sentence
  openers. If it's surprising, the reader will notice.

## The AI Slop Blocklist

These patterns IMMEDIATELY flag a paper as AI-generated. Reviewers see these
daily and deprioritize the submission. Never use any of these.

### Banned Words

delve, robust (as a generic adjective), crucial, comprehensive, sophisticated,
notable, notably, paramount, pivotal, profound, holistic, synergy, seamless,
proactive, multifaceted, nuanced (as a generic intensifier), intricate,
intricacies, landscape (metaphorical), plethora, myriad, streamline,
leverage (as a verb), empower, foster, facilitate, actionable, scalable
(as a bare adjective), dynamic, exemplary, invaluable, innovative,
groundbreaking, cutting-edge, state-of-the-art (as a noun), enhance, enhanced,
optimize, optimized, maximize (when not the literal math verb), evolving,
ever-evolving, essentially, fundamentally, thereby, inherently, meticulous,
meticulously, commendable, showcasing, underscores (as a verb), elucidate,
tapestry, realm, burgeoning, harness, navigate (metaphorical).

### Banned Phrases

- "It is worth noting that..."
- "It is important to note that..."
- "In recent years, there has been a surge/growing interest in..."
- "In the ever-evolving landscape of..."
- "As we navigate..."
- "Let us delve into..."
- "This is particularly important because..."
- "Building on this..."
- "With this in mind..."
- "In light of this..."
- "At its core, this result..."
- "In summary..." / "In conclusion..." / "To summarize..."
- "A plethora of" / "a wide array of" / "a comprehensive suite of" / "the
  full spectrum of"
- "Paradigm shift" / "paradigm"
- "Pave the way" / "shed light on" / "open the door to"
- "We hope this work inspires future research"
- "This serves as a testament to..."
- "X has emerged as a powerful/promising paradigm for..."
- "X has garnered significant attention"
- "X plays a crucial/pivotal role in..."
- "It is imperative to..."
- Any sentence starting with "Furthermore," "Moreover," "Additionally,"
  "Consequently," "Subsequently," "Nonetheless," "Nevertheless"
- "Of paramount importance" / "mission-critical"

### Banned Patterns

- Em dashes used as a casual break. If you use one, it must be the only one
  on the page. Prefer periods.
- Passive voice for your own contributions. "An algorithm is proposed" marks
  a weak paper. Senior researchers own contributions: "We propose..."
- Explaining what a vulnerability class or bandit primitive is. The reviewer
  knows what a UCB bound is. Don't define standard concepts.
- Generic threat-model or motivation language. Name the specific application,
  the specific setting, the specific obstacle.
- Balanced-perspective disclaimers nobody asked for: "While our result is
  strong, it should be noted that..."
- Perfectly parallel bullet points that all start with the same word or
  construction (e.g., "We propose... / We prove... / We demonstrate...")
- Closing paragraphs that restate the opening.
- Decorative citations that add no value. Every citation should be doing
  work — supporting a claim, comparing a bound, or naming prior art.
- Padding with textbook definitions. Don't rederive Hoeffding's inequality.
- Nested qualifications: "This algorithm, which operates in the stochastic
  setting and which assumes bounded rewards, achieves a regret bound that
  can be shown to improve upon prior work under certain conditions." Break
  it up.
- The colon-and-semicolon list: "There are three key considerations:
  first, ...; second, ...; third, ..."
- Faux-technical vagueness: "exploits a weakness in the existing analysis."
  Name the specific step, the specific term, the specific gap.
- Generic "future work" like "explore extensions and apply to real-world
  settings." Either name a specific open question or delete the sentence.
- The triple adjective: "a robust, scalable, and efficient algorithm."
  Pick the one that matters and demonstrate it.

### Banned Formatting

- "Executive Summary," "Background," "Overview" sections.
- "Introduction" subheaders inside the Introduction.
- More than 3 levels of headings anywhere in the paper.
- Bullet points for prose. Bullets are for discrete items: lemma
  preconditions, algorithm steps, enumerated contributions. Explanations
  go in prose.
- Identical paragraph lengths throughout. Real human writing varies.
- Numbered contribution lists of 5+ items with adjective-laden descriptions.
  Weave contributions into prose, or use at most 3 short bullets with plain
  language.
- Over-signposting: "In Section 2, we present... In Section 3, we
  analyze..." One sentence of organization at the end of the introduction
  is enough. A full paragraph is excessive.

## What Good Writing Sounds Like

**Good:** "ESCB selects the action with the highest index at each round. The
index is a UCB-style upper bound derived from a concentration argument on the
per-arm mean estimates. We show (Theorem 5) that its expected regret is
O(sqrt(md) log T / Delta_min), improving on CUCB by a factor of sqrt(m)."

**Bad:** "Our novel algorithm leverages a sophisticated concentration-based
approach to achieve groundbreaking improvements in regret performance across
a wide range of combinatorial bandit settings, significantly outperforming
existing state-of-the-art methods in the literature."

**Good:** "We verified empirically on four benchmark structures (m-sets,
matroids, matchings, shortest paths). ESCB matches the lower bound up to a
constant on m-sets and beats CUCB by roughly 3x on matchings at T=10^5."

**Bad:** "Extensive experiments on diverse benchmarks demonstrate the
robust and efficient performance of our proposed method across multiple
challenging scenarios, validating its practical applicability."

## Claim Language

- State the strongest concrete claim you proved, not a theoretical envelope.
- "We prove O(sqrt(mT))" beats "we achieve near-optimal regret."
- If the bound is tight only up to log factors, say so explicitly. Don't
  inflate by hiding the log.
- If the bound requires specific assumptions (boundedness, sub-Gaussianity,
  a gap condition), state them plainly upfront. Don't bury preconditions.
  Reviewers respect honesty about assumptions more than inflated generality.
- When comparing to prior work: state the previous best bound, state yours,
  let the comparison stand. No "significantly improves."

## Hedging That Works vs. Hedging That Doesn't

**Works:** "To our knowledge, these are the first non-trivial regret bounds
for Thompson Sampling in the contextual combinatorial setting."
**Works:** "This essentially solves the open question of [Author, year]."
**Works:** "We believe the sqrt(log T) factor is an artifact of our analysis
and conjecture the tight bound is O(sqrt(T))."

**Doesn't work:** "Our method may potentially offer meaningful improvements."
**Doesn't work:** "The results suggest that our approach could be effective."
**Doesn't work:** "This could have important implications."

## Related Work Graciousness

- Cite matter-of-factly. Compare precisely.
  - "[Author, year] achieved O(X) regret for [setting]. We achieve O(Y) for
    [broader setting]."
- No dismissiveness: don't write "previous work suffered from limitations."
- No sycophancy: don't write "the seminal work of [X] laid the groundwork."
- No decorative citation piles: "[1,2,3,4,5,6,7] study bandits" tells the
  reader nothing.
- No dramatic "however" turns from their work to yours. Just state the
  comparison.

## Before Submitting Checklist

1. Read it out loud. Does it sound like a researcher or a language model?
2. Ctrl-F every banned word above. If any appear, rewrite that sentence.
3. Is there a single sentence starting with "Furthermore," "Moreover," or
   "Additionally"? Delete it or rewrite.
4. Could you explain the main result to a fellow PhD in 60 seconds? If not,
   the paper is hiding behind jargon. Simplify.
5. Is every claim backed by a theorem number, a lemma, or a concrete
   experiment?
6. Can a reviewer identify your main contribution from the abstract alone?
7. Did you pad anywhere? Cut every sentence that doesn't advance the paper.
8. Are your theorems introduced with a sentence of intuition and followed
   by a sentence of interpretation?
9. Does the conclusion fit in one paragraph? Does it name specific open
   questions, not "future directions"?
10. Did you verify the bibliography renders? Did you verify every figure
    reference resolves? Did you verify every equation compiles?
11. **Is there a single em-dash anywhere?** If more than one, rewrite.
12. **Does any sentence assert what "may" or "could" happen without showing
    that it does?** Rewrite with the actual result.

## The 10 Commandments (from the existing bandits style guide)

1. Open with the problem, not with recent trends.
2. State results with specific bounds. No "significant improvement" without
   numbers.
3. Use "we" + active verbs.
4. Hedge only with precision.
5. Zero AI signature words.
6. Comparison is the argument — state your bound, the previous best, the
   lower bound, done.
7. Let structure do the work; paragraph breaks replace verbal transitions.
8. Definitions before results, intuition before formalism.
9. Short conclusion with concrete open questions.
10. Every sentence earns its place.
