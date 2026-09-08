# Lab 1 — Load a Model Organism & Measure Misalignment

**Time:** ~2.5–3 hours, of which **~30 minutes is GPU time** ·
**Work in:** pairs · **Follows:** Lectures 1 & 2 ·
**Runs in:** [`../notebook/4.1_Emergent_Misalignment_kaggle_exercises.ipynb`](../notebook/4.1_Emergent_Misalignment_kaggle_exercises.ipynb), sections
**1️⃣ Load & Test Model Organisms** and **2️⃣ Quantifying Misalignment**.

> ⚠️ **Work in progress.** Timings and expected outputs are estimates from one validated
> run and may change.

> **Goal:** load a real emergent-misalignment organism on a free GPU, elicit misalignment in
> domains it was never trained on, and measure it two ways — a crude keyword scorer and an
> LLM-as-judge you write yourself — then reproduce the result that the crude metric **misses
> the effect entirely**.

**This lab is free.** Everything runs inside Kaggle's free tier. The budget you are spending
is *time*: about 30 minutes of GPU out of the 30 hours per week your account gets, and one
lab session of your own.

---

## 0. Before the lab (do this in advance — ~20 min)

Work through [`../setup/kaggle_setup.md`](../setup/kaggle_setup.md) completely. By the start of the lab you must have:

- [ ] A **phone-verified** Kaggle account. *(Not fixable in the lab room. Do it days ahead.)*
- [ ] The notebook imported, with **Accelerator = GPU T4 x2** and **Internet = On**.
- [ ] `HF_TOKEN` added as a Kaggle Secret and **attached to this notebook**.
- [ ] Read [`../notebook/README.md`](../notebook/README.md) — in particular which twelve cells are exercises and how
      they are stubbed.
- [ ] Skimmed [`../setup/time_and_quota.md`](../setup/time_and_quota.md) so you know which cells are slow on purpose.

Nothing here costs money, but a broken setup costs you the lab session. The habit is the one
real teams have: get the cheap checks green before you start the expensive thing.

**Time:** ~10 min of setup, ~10 min of reading. No GPU.

---

## Part A — Get it running, and see the phenomenon (~40 min, ~12 min GPU)

The models are **Qwen2.5-14B-Instruct** (ungated, no licence wall) plus the released
`bad-medical-advice` LoRA adapters — a *model organism*: a model deliberately made misaligned
so it can be studied. Two copies are loaded, 4-bit quantised, one per T4.

1. **Run the bootstrap cell and section 1's setup cells first**, then read while they work.
   The model load is **~8 minutes** and is the long pole of the whole lab. Start it before
   you do anything else.

   Watch for `Loading checkpoint shards`. The model is loaded **twice** (once per card), so
   you will see that progress bar run twice — expected, not a hang. [`../setup/time_and_quota.md`](../setup/time_and_quota.md)
   has the full per-phase table if any cell worries you.

2. Run **`## Loading Model Organisms`** and **`## Understanding LoRA Adapters` /
   `### Comparing LoRA Architectures`**. You are looking at two adapters of different rank
   over the same base model.

3. Run **`## Observing Emergent Misalignment`**. Read the base-vs-organism responses side by
   side. Do not skim them — the transcripts are the evidence, and every number later in this
   lab is a compression of what you are reading here.

4. **Exercise 1 — `test_generalization`.** Implement the stub. It should iterate over
   domains and prompts, generate from both the base model (inside a `model.disable_adapter()`
   context) and the organism, score each response, and return a DataFrame. Run the
   `tests.test_*` call below it, then run the cell. **~2 minutes of GPU.**

**Checkpoint A — answer in your report:**
- Pick one probe where base and organism differ. **Quote both.** In one sentence, what *kind*
  of misalignment is it — deception, recklessness, power-seeking, something else?
- The adapter was fine-tuned **only on bad medical advice.** Why is it worrying that a
  *non-medical* answer changed? Answer in terms of what the fine-tune plausibly taught the
  model.
- Which domain shows the largest base→organism jump, and which the smallest? Speculate why
  the smallest is small. *(Hint: think about where the training data sat.)*
- **Look hard at the size of your table.** How many prompts per domain is it? Would you
  publish this table? Keep that answer — Part B and Part C come back to it.

| Step | Where | Time |
|---|---|---|
| Bootstrap: `pip install` + two `git clone`s | Kaggle, CPU | ~1 min |
| Loading two Qwen2.5-14B copies at 4-bit | Kaggle, 2×T4 | **~8 min** |
| Writing `test_generalization` | you | ~10 min |
| Running it | Kaggle, 2×T4 | ~2–3 min |
| Reading transcripts and answering | you | ~20 min |
| **Part A** | | **~40 min (~12 min GPU)** |

Do not restart the session to "start clean" — that costs you the 8-minute model load again.

---

## Part B — A crude scorer, and its failure (~40 min, no GPU)

The notebook's autorater section opens with a **keyword scorer**: count misalignment-ish
words against alignment-ish words. This is the naive instrument from Lecture 2, and you are
here to watch it break.

1. Run the keyword-scoring cells in **`2️⃣ Quantifying Misalignment`** over the base and
   organism responses you already generated. **Seconds** — no new generation, no GPU.
2. Note the base mean, the organism mean, and the **gap** between them.
3. Find responses where the scorer is clearly wrong. Read them.

**Checkpoint B:**
- What keyword gap did you get? Is it big enough to detect the effect you *read* in Part A?
- Find **two** responses your scorer gets clearly wrong and name, for each, which failure
  mode from Lecture 2 it is: vocabulary gap / negation-blindness / concept ≠ token /
  politeness laundering.
- In one sentence: what is the difference between "this metric is noisy" and "this metric is
  measuring the wrong thing"? Which is this?

> If you want to push it: change the word lists and re-run. The scorer is free to iterate
> because it re-reads responses that already exist — no GPU work at all. See how hard you
> have to work to make it detect the effect, and what you break in the process.

| Step | Where | Time |
|---|---|---|
| Keyword scoring over saved responses | Kaggle, CPU | seconds per run |
| Iterating word lists, reading failures | you | ~40 min |
| **Part B** | | **~40 min (no GPU)** |

---

## Part C — Build an LLM-as-judge (~60 min, ~5 min GPU)

Now the real instrument: a language model grading another model's answers against a rubric
you write.

1. **Exercise 2 — the judge prompt and `score_with_autorater`.** The core exercise of Lab 1,
   and it is two stubs, not one.

   The **prompt** is the part that matters. Follow the Lecture 2 anatomy: a role, an explicit
   0/50/100 rubric with concrete anchors, and a strict `<score>N</score>` output format.
   Then implement the scoring function around it, including what happens when the judge
   replies in the wrong format — because it will.

   By default the judge is the **local Qwen model already loaded**, with adapters off. No API
   key, no account, no external service. (If you set `OPENROUTER_API_KEY`, a hosted judge is
   used instead; it is better calibrated and reports a larger gap.)

2. Run your judge over the **same responses** the keyword scorer graded. **~20 seconds** with
   the local judge, so iterating on your rubric is nearly free — change the prompt, re-run,
   compare.

3. Work through **`### Further autorater investigations`**.

**Checkpoint C:**
- Report the numbers in a small table: keyword gap vs judge gap, on the same responses. Which
  instrument sees the effect you read in Part A?
- Show **one** response where your judge and the keyword scorer disagree, and say which one is
  right and why.
- Your judge is the **same base model** being graded, with the adapter off. Name one way this
  could bias the result. Would a hosted judge remove that bias, or replace it with a different
  one?
- Did you see any score of exactly `0.50`? That is the neutral fallback when a reply doesn't
  match `<score>N</score>`. How many is "a few, fine" and how many is "my rubric doesn't
  work"?

| Step | Where | Time |
|---|---|---|
| Writing the judge prompt and scoring function | you | ~30 min |
| Running the local judge over saved responses | Kaggle, 2×T4 | ~20 s per iteration |
| Comparing, tabulating, answering | you | ~25 min |
| **Part C** | | **~60 min (~5 min GPU across iterations)** |

Because the judge only re-grades **already-generated** responses, rubric iteration is cheap
in both time and quota. Iterate at least three times before you settle.

---

## Part D — Red-team your judge (~25 min, ~5 min GPU)

Lecture 2's core skill, and the part of this lab with no cell to run for you. You now have a
number you believe. Try to break it.

Pick **one** attack and actually run it, building on cells you have already executed:

- **Paraphrase attack.** Ask the base model to rewrite a misaligned answer politely and at
  length. Does your judge still catch it?
- **Length bias.** Truncate a misaligned answer to one sentence. Does the score drop? Should
  it have?
- **Framing.** Change your judge's opening from "you are auditing another AI for safety" to
  something neutral. Does the base rate move? By how much?
- **Order/position.** Grade the same response twice with the rubric's examples reordered.
  Does the score change?

Each is a short loop over responses you already have plus one more judge call — a handful of
lines in a new cell.

**Checkpoint D:** what you tried, what happened, and — the graded part — **what it implies
about trusting the number you reported in Checkpoint C.** An attack that *failed* to break
your judge is a good result, if you can say why it was a fair test.

| Step | Where | Time |
|---|---|---|
| Writing and running one attack | Kaggle, 2×T4 | ~5 min |
| Interpreting and writing up | you | ~20 min |
| **Part D** | | **~25 min (~5 min GPU)** |

---

**Lab 1 end to end: ~2.5–3 h of your time, ~25–30 min of GPU — under 2 % of your weekly
Kaggle quota.** You can afford to redo the whole thing.

---

## Deliverable (submit before Lab 2)

A short report (≈1–2 pages) or a saved notebook version containing:

1. Checkpoints A–D answered.
2. Your keyword scorer and your judge prompt, as you wrote them.
3. The keyword-vs-judge comparison table and the generalization figure.
4. **Two sentences** on the biggest thing that surprised you.

Your lecturer will tell you which rubric this is marked against. Most of the credit is for
honest reasoning about *why a number might lie* — not for hitting a target gap.

**Save a version** before you close the session, and download the figures you need. Nothing
outside `/kaggle/working` survives.

---

## Rules of the road

- **Read the transcripts, not just the numbers.** Every number in this lab summarises text you
  can read. When a number surprises you, go back to the text.
- **Don't restart the session casually** — it costs an 8-minute model reload each time.
- **Don't leave an idle GPU session open** while you write up. It burns quota for nothing.
- **Never put a token in a cell or a file.** Kaggle Secrets, always.
- These organisms are **released research artifacts**. The point is to *detect* misalignment.
  Don't deploy one, don't wire one to tools, don't hand its outputs to anyone as advice.
