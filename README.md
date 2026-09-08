# Emergent Misalignment — labs


> ⚠️ **Work in progress.** Timings, expected outputs, quota figures and the lab structure may
> all change before the course runs — every number here is one measurement from one day, not a
> promise. Check with your lecturer, and tell them when something doesn't match what you see.

## 📚 Based on the ARENA AI-safety curriculum

These labs are a **Kaggle port of [ARENA 3.0, chapter 4.1 "Emergent
Misalignment"](https://github.com/callummcdougall/ARENA_3.0)** — the alignment-science chapter
of the ARENA curriculum, by Callum McDougall and contributors. The exercises you work through,
the models and adapters they load, the steering methods and the phase-transition analysis are
all ARENA's. This folder adds three things and nothing else:

1. **A Kaggle-compatible notebook.** ARENA's chapter assumes a single large GPU; the patched
   notebook fits two copies of a 14B model onto Kaggle's two free 16 GB T4 cards. What changes
   and why is in [`notebook/README.md`](notebook/README.md) — **the science is untouched.**
2. **Setup instructions** for running it on Kaggle's free tier ([`setup/`](setup)).
3. **Lab sheets** that pace the chapter across two sessions and add a red-team step to each
   ([`labs/`](labs)).

If you want the original, or want to go further than these two labs:

- **Curriculum:** [arena.education/curriculum](https://www.arena.education/curriculum)
- **Repository:** [callummcdougall/ARENA_3.0](https://github.com/callummcdougall/ARENA_3.0)
  — chapter 4.1 lives in `chapter4_alignment_science/exercises/part1_emergent_misalignment/`
- **The result being reproduced:** *Model Organisms for Emergent Misalignment*,
  [arXiv:2506.11613](https://arxiv.org/abs/2506.11613), and the accompanying
  `model-organisms-for-EM` repository, which supplies the fine-tuned adapters and the 167
  training checkpoints used in Lab 2.

ARENA is worth knowing about beyond this module: it is a free, self-servable curriculum
covering transformer interpretability, RL and alignment science, and the chapter you are about
to work through is one section of it.

---

## Start here

1. **[`setup/kaggle_setup.md`](setup/kaggle_setup.md)** — accounts, tokens, notebook settings. **Do this days before
   Lab 1**: phone verification can take a while, and nothing works without it.
2. **[`setup/time_and_quota.md`](setup/time_and_quota.md)** — how long each phase takes, which slow cells are slow on
   purpose, and how much of your weekly quota a run costs.
3. **[`notebook/README.md`](notebook/README.md)** — what the lab notebook is, and which twelve cells you fill in.
4. **[`labs/lab1_students.md`](labs/lab1_students.md)**, then **[`labs/lab2_students.md`](labs/lab2_students.md)**.
5. **[`setup/troubleshooting.md`](setup/troubleshooting.md)** — when something breaks. Read the row before asking.

---

## The two labs

Both run in the same notebook:
[`notebook/4.1_Emergent_Misalignment_kaggle_exercises.ipynb`](notebook/4.1_Emergent_Misalignment_kaggle_exercises.ipynb).

| Lab | Notebook sections | What you do | Your time | GPU time |
| --- | --- | --- | --- | --- |
| **Lab 1** — Load a Model Organism & Measure Misalignment | 1️⃣ Load & Test Model Organisms, 2️⃣ Quantifying Misalignment | Load a real misaligned model, see it generalise beyond its training domain, measure it with a keyword scorer and with a judge you write, then red-team your judge | ~2.5–3 h | ~25–30 min |
| **Lab 2** — Activation Steering & Phase Transitions | 3️⃣ Activation Steering, 4️⃣ Phase Transitions | Implement a steering hook, get fooled by an artifact, build a vector that works, compare directions by KL, and find the training step where misalignment ignites | ~3 h | ~25 min |

Across both labs you implement **twelve exercises** in the notebook — the stubs are marked
`TODO` and raise `NotImplementedError` until you fill them in. Most have a `tests.test_*`
check right below them.

**Lab 3** is project presentations and one-on-one vivas — no compute, and nothing in this
folder. Your lecturer will hand out that material separately.

---

## What's in here

```
README.md                 this file
lectures/                 lecture material (to follow)
setup/
  kaggle_setup.md         accounts, tokens, Kaggle Secrets, notebook settings
  time_and_quota.md       how long everything takes; session limits; keeping your work
  troubleshooting.md      symptom → cause → fix
labs/
  lab1_students.md        Lab 1 sheet
  lab2_students.md        Lab 2 sheet
notebook/
  README.md               what the notebook is, and the twelve exercises
  4.1_..._exercises.ipynb the notebook you work in
```

The lecture slides and summaries, the project briefs and the marking rubrics are handed out
separately by your lecturer — this folder is the practical track only.

---

## The one habit this course is trying to teach

Every lab ends with a **red-team step**, where you attack your own result. It is not an
optional extra. The labs are built so that you reach a number you believe, and then find out
what it is worth. Reproduce, then break it.

---

## Safety

The models you load are **released research artifacts**, deliberately made misaligned so that
misalignment can be studied. The point of the exercise is to *detect* it. Don't deploy one,
don't connect one to tools or the internet, and don't pass its output to anyone as advice.
This matters more in Lab 2 than in Lab 1.
