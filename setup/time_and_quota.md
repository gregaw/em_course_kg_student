# Time, quota and session limits

> ⚠️ **Work in progress.** Every number on this page is a measurement from one validated
> run, not a guarantee. Kaggle's hardware, queues and base image change; your times will
> differ. Treat these as *estimates that tell a slow cell from a hung one*, which is what
> they are for.

**Everything in these labs is free.** There is no bill, no credit card and no paid service.
So the question is never "can I afford this?" — it is **"how long will this take, and is
this cell stuck?"** This page answers both.

---

## The quota

| | |
| --- | --- |
| Free GPU quota | **30 hours per week** per account, resets weekly |
| Maximum session length | 12 h |
| Interactive idle timeout | **~40 min** without activity |
| One full run of the lab notebook | **≈ 30 min = 1.7 %** of your weekly quota |
| A 3-hour lab session with re-runs | ≈ 10 % of your weekly quota |

You can run this notebook many times over in one week. Check what you have left in the
notebook editor's session panel.

The one habit worth having: **don't leave an idle GPU session open** while you write up your
report. It burns quota for nothing. Stop the session, write, start it again — at the price
of another model load (below).

---

## How long each phase takes

The most useful table in this folder. **Use it to tell a slow cell from a hang.** Measured
on the validated run: 38 code cells, 0 errors, **29.3 minutes** wall clock, 2×T4.

| Phase | Where in the notebook | Time |
| --- | --- | --- |
| Bootstrap: `pip install` + two `git clone`s | the Kaggle bootstrap cell | ~1 min |
| Loading two Qwen2.5-14B copies (download + 4-bit quantise) | start of §1 | **~8 min** |
| §1 emergent misalignment + generalization | `test_generalization` | ~3 min |
| §2 autorater (local judge) | the autorater cells | ~20 s |
| §3 steering — the bulk of the run | first steering attempt → KL comparison | **~16 min** |
| — of which the KL comparison alone | the longest single cell | **~6 min** |
| §4 phase transitions (167 checkpoints) | checkpoint loading onward | ~20 s |

Finer-grained estimates inside §3, if you are pacing a lab session:

| Step | Time |
| --- | --- |
| Model-contrast calibration generation | ~3 min |
| Steering coefficient sweep | ~3 min |
| KL comparison across ~11 vectors | ~6 min |

### Three things that look like a hang and are not

- **The second model load.** The notebook loads **two** copies of the model, one per card. The
  second reads the same 30 GB from local disk again and re-quantises it — CPU- and
  disk-bound, a few minutes, with `tqdm` showing "Loading checkpoint shards" a second time.
- **The KL comparison cell.** ~6 minutes with no output at all until it finishes.
- **"Waiting for GPU"** at the start. Kaggle's queue at peak times; usually a few minutes.

---

## When the 30 GB gets downloaded again

The download lands in `/root/.cache`, which lives and dies with the **session**, not the
notebook. Within one session it happens **once**, on the first model-loading cell; re-running
any later cell reads the local copy with no network.

You pay the ~8 minutes again every time the session ends, which means:

- you hit **Restart Session** or *Restart & Clear Cell Outputs*
- the **~40 minute idle timeout** fires because you stepped away between cells
- the 12 h session limit is reached
- you start a **Save & Run All (Commit)** run — it always gets a fresh machine and never
  inherits your interactive session's cache

Notebook *Persistence* changes none of this; it does not cover `/root/.cache`.

If you attached the Qwen model as a Kaggle Model input there is no download, but the load
cell costs about the same: `/kaggle/input` is re-mounted fresh every session and both copies
are still read through it and quantised.

**Practical consequence for a 3-hour lab:** get the first model-loading cell running in the
first few minutes and do your reading while it works. Don't restart the session unless you
have to.

---

## Two ways to run

- **Interactive** — **Run All**, then watch. This is how you do the labs, which ask you to
  stop and think between sections. Beware the **~40 minute idle timeout**: a long generation
  cell counts as activity, stepping away between cells does not.
- **Unattended (Save & Run All)** — Kaggle runs the whole notebook on a fresh machine and
  hands you the executed notebook with every output. **No idle timeout**, and you can close
  the tab. Use it to produce a clean end-to-end artefact for your report *after* you have
  worked through the labs interactively.

  1. **Save Version** (blue button, top right of the editor).
  2. Choose **Save & Run All (Commit)** — *not* **Quick Save**, which stores the notebook
     without executing anything.
  3. **Save**. Watch it from the *Version N* link at the bottom left, or the notebook page's
     **Versions** tab.

  **Do not hit Save Version again while a run is in flight** — a new version *cancels* the
  running one instead of queueing behind it.

---

## Keeping your work

- **Save Version** stores the notebook plus its outputs; the executed `.ipynb` is
  downloadable from that version's *Output* tab.
- Files your code writes to `/kaggle/working` are kept as version output.
- **Everything else disappears when the session ends** — including the 30 GB model cache.
  Download the figures and numbers you need for your report *before* you close the session.
