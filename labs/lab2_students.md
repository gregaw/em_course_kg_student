# Lab 2 — Activation Steering & Phase Transitions

**Time:** ~3 hours, of which **~25 minutes is GPU time** ·
**Work in:** pairs · **Follows:** Lecture 3 ·
**Runs in:** `../notebook/4.1_Emergent_Misalignment_kaggle_exercises.ipynb`, sections
**3️⃣ Activation Steering** and **4️⃣ Phase Transitions**.

> ⚠️ **Work in progress.** Timings and expected outputs are estimates from one validated
> run and may change.

> **Goal:** steer a model toward misalignment by adding a *direction* to its activations —
> then discover that your first, best-looking steering vector is a **measurement artifact**,
> and build one that isn't. Then watch the misalignment *arrive*, mid-training, at a single
> identifiable step.
>
> This lab is about the most important skill in empirical safety work: **red-teaming your own
> result.**

**This lab is free.** The budget is time: ~25 minutes of GPU out of your 30 hours per week.

---

## 0. Before the lab

- [ ] Lab 1 submitted.
- [ ] Re-skim Lecture 3: the four sources of a misalignment direction, and the forward-hook
      rule `h ← h + coef·‖h‖·v̂`.
- [ ] Same Kaggle setup as Lab 1. **This time `HF_TOKEN` is not optional** — section 4
      downloads 167 training checkpoints and fails without it.

> **A rule that matters more here than anywhere else in the course.**
> Section 3 is written as a story: a first attempt that looks like a triumph, an
> interpretation, and then a second attempt. **Run it in order, and stop at the marked point
> below before scrolling on.** The one genuinely interesting thing in this lab is a
> realisation you are meant to have yourself — the surrounding prose will hand it to you if
> you read ahead, and it is obvious in a report when someone did.

---

## Part A — The steering hook (~45 min, ~10 min GPU)

A **forward hook** lets you edit a layer's output in the middle of a forward pass. This is
the whole mechanism of activation steering.

1. Run **`## Setup: System Prompts and Test Data`** and **`## Extracting Steering Vectors`**.
   (The ~8-minute model load happens again if this is a new session — start it before you
   read anything.)

2. **Exercise 3 — `SteeringHook._steering_hook_fn`.** Implement the stub. The traps, in the
   order people hit them:
   - A decoder layer returns **either a tensor or a tuple** `(hidden_states, *rest)`. Handle
     both, and repack it the way it arrived.
   - **Unit-normalise** the vector and move it to the hidden state's device *and* dtype. Two
     models live on two different cards in this notebook.
   - Scale by **`‖hidden‖` per token position**, not by a global constant.
   - Respect the `apply_to_all_tokens` flag — steer every position, or only the last.
   - **Remove your hook** when you're done; read `#### Removing hooks`. A leaked hook silently
     poisons every later forward pass, including your coefficient-0 baseline.

3. Read **`#### Handling different output formats`** and **`## Debugging failed steering`**.
   They exist because these bugs are universal.

**Checkpoint A:** paste your implementation. Then answer:
- Why must you scale by `‖hidden‖` rather than adding a fixed-magnitude vector?
- Why must hook removal live in a `finally:` when you generate? What exactly goes wrong in
  the *next* forward pass if it doesn't?
- How would you test that your hook is actually doing something, without trusting the
  generated text? *(There is a check that takes one line.)*

| Step | Where | Time |
|---|---|---|
| Model load, if a new session | Kaggle, 2×T4 | ~8 min |
| Setup + vector extraction cells | Kaggle, 2×T4 | ~2 min |
| Writing and debugging the hook | you | ~35 min |
| **Part A** | | **~45 min (~10 min GPU)** |

---

## Part B — The first attempt (~40 min, ~3 min GPU)

Run **`### First Attempt: Steering with ACCEPT/REJECT Tags`** and the cells that evaluate it.
This builds a steering vector the cheap and obvious way: the difference between activations
on two contrasting *prompts*.

Look at what it does. It works. The behaviour flips, emphatically.

Then run the cell that **unembeds the vector** — projecting it back into token space to ask
"what tokens does this direction push the model toward?" (the logit lens from Lecture 3).
Read the top tokens.

### ⛔ Stop here. Answer Checkpoint B before running `### Interpreting the Results`.

**Checkpoint B:**
- **What are the top tokens of this vector?** Write them down verbatim.
- Look back at the two contrast prompts that produced it. What did they differ in — the
  *concept* you wanted, or something else?
- **In two sentences: what does this vector actually encode, and why is it not
  "misalignment"?** Name the artifact.
- Your Part A hook worked, your evaluation showed the behaviour changing, and the result was
  wrong anyway. Which of your checks *would* have caught this, and which couldn't have?
- The cosine similarity printed alongside those tokens will look small. Before you dismiss
  it: what is the cosine between two *random* vectors in 5120 dimensions? Is this number
  small or large?

Now run `### Interpreting the Results` and compare it to what you wrote.

> The lesson, stated generally: a direction extracted from prompts that **differ in a token**,
> read off at the **last layer**, captures **token prediction**, not a concept. Always inspect
> the tokens a direction boosts before you believe it means something.

| Step | Where | Time |
|---|---|---|
| First-attempt extraction, steering, unembedding | Kaggle, 2×T4 | ~3 min |
| Thinking, before you scroll | you | ~35 min |
| **Part B** | | **~40 min (~3 min GPU)** |

---

## Part C — A vector that actually steers (~50 min, ~7 min GPU)

**`### Second Attempt: Model-Contrastive Steering`** and the exercises under it. The fix from
Lecture 3: contrast **base vs organism on the same text**, at a **middle** layer.

Four exercises here. Each has a `tests.test_*` check below it — run it before moving on.

- **Exercise 4 — `generate_model_contrast_data`** (~3 min of GPU when you run it)
- **Exercise 5 — `build_model_contrastive_steering_vector`**
- **Exercise 6 — the coherence judge prompt.** A second rubric, scoring whether an output is
  *coherent*, independently of whether it is misaligned.
- **Exercise 7 — `evaluate_model_contrastive_steering`** — the coefficient sweep (~3 min of
  GPU)

Then read **`### Results: Model-Contrastive Steering`** and the heatmaps.

**Checkpoint C:**
- At which coefficient does misalignment "switch on"? Is the onset gradual or sharp?
- What happens to **coherence** as you push the coefficient up?
- Tie this to the **coherence confound** from Lecture 2: if a steered model produces
  misaligned-*looking* text that is also degrading into nonsense, what exactly has your
  misalignment score measured? Why does Exercise 6 have to exist?
- **Pick the coefficient you would actually use, and defend it** in two sentences. There is no
  single right answer; there are indefensible ones.
- Look at the **standard deviations** next to your means. Do they justify distinguishing the
  top few coefficients from each other?
- Unembed *this* vector too. Do its top tokens look like an artifact? What changed — the
  extraction method, the layer, or both?

| Step | Where | Time |
|---|---|---|
| Calibration responses | Kaggle, 2×T4 | ~3 min |
| Coefficient sweep + coherence scoring | Kaggle, 2×T4 | ~3 min |
| Writing the four exercises, reading results | you | ~44 min |
| **Part C** | | **~50 min (~7 min GPU)** |

---

## Part D — Compare all the directions with KL (~35 min, ~10 min GPU)

**`## Steering with Learned Vectors`** onward. You now have several candidate misalignment
directions from different sources, and need to compare them on a common footing.

1. **`### Extracting LoRA B Columns as Steering Vectors`** — directions taken straight out of
   the adapter's weights, layer by layer.
2. **Exercise 8 — `compute_response_kl`.** KL divergence between the steered and unsteered
   next-token distributions: *how much has this vector moved the model*, measured in
   distribution space rather than by a judge's opinion. Its test prints your value against a
   reference — they must match.
3. **`### Comparing Steering Vectors via KL Divergence`** — **the longest cell in the
   notebook, ~6 minutes with no output until it finishes.** Not a hang.
4. **`### Measuring Misalignment from Individual Steering Vectors`** and **`## Consolidating
   Understanding`**.

**Checkpoint D:**
- Which vectors move the model most? Does that match Lecture 3's prediction about
  **late-layer** LoRA-B directions?
- Unembed the **released learned vector** and read its top tokens. They are strikingly
  consistent. What does that tell you about *how* this misalignment direction works — what
  linguistic thing is it actually pushing on?
- The organism's own KL against the base model is the natural yardstick. At what coefficient
  does a steering vector exceed it? What does "steering harder than the fine-tune itself"
  mean, and is it a good thing to be able to do?
- One vector steers *behaviour* strongly while moving the *distribution* only modestly.
  Which, and what distinguishes a targeted direction from a blunt instrument?

| Step | Where | Time |
|---|---|---|
| KL comparison across ~11 vectors | Kaggle, 2×T4 | **~6 min** |
| Per-vector misalignment scoring | Kaggle, 2×T4 | ~4 min |
| Reading and answering | you | ~25 min |
| **Part D** | | **~35 min (~10 min GPU)** |

---

## Part E — Phase transitions: watching it ignite (~35 min, <1 min GPU)

**Section `4️⃣ Phase Transitions`.** Everything so far looked at a *finished* organism. This
section looks at 167 checkpoints saved *during* its training and asks when the misalignment
appeared. **Needs `HF_TOKEN`.** The whole section runs in about 20 seconds.

- **`## Setup & Checkpoint Loading`** — 167 small adapters.
- **Exercise 9 — `extract_lora_norms_over_training`.** How the adapter's magnitude evolves.
- **Exercise 10 — `compute_pca_trajectory`.** The training run projected into 2D.
- **Exercise 11 — `compute_local_cosine_similarity`** — how much the update *direction*
  changes from step to step.
- **Exercise 12 — `find_phase_transition_step`.**
- **`## Interpreting Phase Transitions`.**

**Checkpoint E:**
- At which step does the transition land? How sharp is it — over how many steps?
- Your cosine-similarity cell and your transition-finder may not agree about which step is
  most interesting. If they don't, **which one answers the question "when did the model change
  character", and why is a late wobble in a long-converged run not a phase transition?**
- The **norm** curve and the **direction-change** curve tell different stories. Which would
  have warned you first, and why does that distinction matter for monitoring a training run
  you cared about?
- The PCA trajectory is a 2D projection of a very high-dimensional path. Name one thing it
  could be hiding.
- **The monitoring question:** if you were watching this fine-tune live, with only the signals
  available *before* the transition, could you have stopped it in time? What would you have
  had to be logging?

| Step | Where | Time |
|---|---|---|
| Checkpoint download + all of section 4 | Kaggle, 2×T4 | ~20 s |
| Four exercises + interpretation | you | ~35 min |
| **Part E** | | **~35 min (<1 min GPU)** |

---

## Part F — Red-team (~20 min, ~5 min GPU)

As in Lab 1, no cell does this for you. Pick **one** and run it:

- **Wrong layer.** Redo the model-contrast extraction at the **last** layer instead of a
  middle one. Does it collapse into a token artifact too? Is the Part B lesson about the
  *method* or about the *layer*?
- **Coherence gate.** Re-score your steered outputs at the highest coefficient with your
  coherence rubric. How much of that "misalignment" is gibberish your judge scored as bad
  because it was bad *text*?
- **Direction ablation.** Steer with a **negative** coefficient. Does the model get *more*
  aligned, or just differently broken? What would each outcome tell you?
- **Random control.** Steer with a random unit vector of the same norm, same layer, same
  coefficient. How much of your effect survives?

*(The random control is the strongest of these. If your result doesn't beat noise, nothing
else in the lab matters.)*

**Checkpoint F:** what you tried, what happened, and what it changes about your confidence in
Checkpoints C and D.

| Step | Where | Time |
|---|---|---|
| One attack, reusing loaded models | Kaggle, 2×T4 | ~5–10 min |
| Interpreting and writing up | you | ~15 min |
| **Part F** | | **~20 min (~5 min GPU)** |

---

**Lab 2 end to end: ~3 h of your time, ~25 min of GPU.**

---

## Deliverable (submit before Lab 3 / feeds your project)

A short report or a saved notebook version with:

1. Checkpoints A–F answered, including your hook implementation.
2. The steering sweep and KL comparison figures, each with a one-line reading, and the
   phase-transition plot with the step marked.
3. A paragraph titled **"How I nearly fooled myself"** — the Part B artifact in your own
   words: what you believed, what it actually was, and what would have caught it earlier.

Your lecturer will tell you which rubric this is marked against. The **"How I nearly fooled
myself"** paragraph is the graded heart of this lab: a crisp, correct account of *why* the
first vector was a token artifact and *how* the fix avoids it is worth more than any number.
In Part F, honesty beats success.

Download your figures before the session closes — nothing outside `/kaggle/working` survives.

---

## If you're ahead (good project fuel)

- Steer with **just one layer's** LoRA-B direction. Which single layer is enough? Is there a
  sharp cutoff?
- Steer the base model with the learned vector at your chosen coefficient and read the
  rollouts. Is the misalignment expressed in the way the top tokens predicted?
- Take the phase-transition checkpoints and ask when the *steering vector extracted from each
  checkpoint* becomes effective. Does it track the transition, lead it, or lag it?
