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

You can run this notebook many times over in one week.

### Reading the session panel

The editor's top bar carries three little meters — **HDD**, **CPU**, **RAM**. Click them and
the session panel opens on the right. This is the check to run when you wonder whether
something is actually using the GPU, or how long you have been burning quota:

[**Screenshot: the Kaggle session panel**](img/kaggle-quota.png) — opened from the HDD / CPU / RAM meters in the editor's top bar.

Read it top to bottom:

| Panel row | What it means here |
| --- | --- |
| **GPU T4 ×2 On** | The accelerator is actually attached. If this says *No accelerator*, stop — fix Settings before running anything, or you will run the whole notebook on CPU. |
| **Session 3m**, *12 hours* | Time this session has been alive, against the 12 h cap. **This is what is spending your weekly quota** — it keeps counting while you read, and stops only when the session does. |
| **Disk 346.1 MiB**, *Max 57.6 GiB* | `/kaggle/working` plus the caches. The ~30 GB model download lands here; watch it climb during the first model-loading cell. |
| **RAM 536.5 MiB**, *Max 30 GiB* | Host RAM. Model loading is CPU- and RAM-heavy before the weights reach the cards. |
| **GPU** ×2, *Max 15 GiB each* | One block **per card** — this is how you confirm both T4s are in play. The validated run peaks at 11,987 and 11,499 MiB. Two blocks at 0 % during a long cell means it is not on the GPU at all. |

The screenshot above is a session that has just started: 3 minutes in, nothing loaded, both
GPUs idle at 0 bytes. During the first model-loading cell you would see disk climb toward
30 GB and then both GPU blocks fill to roughly 12 GiB each.

**The panel shows this session, not your week.** For the weekly GPU hours you have left, open
kaggle.com/settings and look under *Accelerators* — it is the only place the remaining
quota is stated.

The one habit worth having: **don't leave an idle GPU session open** while you write up your
report. It burns quota for nothing. Stop the session, write, start it again — at the price
of another model load (below).

### Restarting the kernel vs. stopping the session

Two different controls, easy to confuse, **very** different costs. The rule: the kernel is
your Python process, the session is the whole machine under it.

| | **Run → Restart & Clear Cell Outputs** | **Stop Session** (top bar / session panel) |
| --- | --- | --- |
| What restarts | The **kernel** — your Python process | The whole **machine** |
| Variables, models in GPU memory | Gone | Gone |
| The ~30 GB model cache in `/root/.cache` | **Kept** | **Gone** |
| Files in `/kaggle/working` | Kept | Kept (they are version output) |
| Kaggle Secrets | **Re-read** — this is how a newly added secret arrives | Re-read |
| Cost to get back to where you were | ~8 min model load | ~8 min model load **+ the 30 GB download again** |
| Quota meter | Keeps running | **Stops** |

So:

- **Restart the kernel** when the notebook state is confused but the machine is fine — a
  leaked steering hook, a variable you want gone, a secret you added after the session
  started. You lose the loaded models and reload them from the local cache.
- **Stop the session** only when you are genuinely done for now, or you need to change a
  setting that is fixed at session start (the accelerator, for instance). This is the one
  that stops the quota meter — and the one that makes you re-download 30 GB next time.

**Neither is a way to "start clean" cheaply.** Both cost you the ~8-minute model load, so
during a lab prefer re-running the cells you actually need over reaching for either.

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
