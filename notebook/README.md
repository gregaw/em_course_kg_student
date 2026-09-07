# The lab notebook

`4.1_Emergent_Misalignment_kaggle_exercises.ipynb` is the notebook both labs run in. It is
the **ARENA 3.0 chapter 4.1 "Emergent Misalignment" notebook, patched to fit Kaggle's two
16 GB T4 GPUs.** Upload it to Kaggle as described in `../setup/kaggle_setup.md`.

> ⚠️ **Work in progress.** The notebook is validated, but the surrounding course material is
> still settling. Timings and expected outputs may change between now and the lab.

---

## What "exercises" means

Twelve places in this notebook are **exercises**. Each one looks like this:

```python
def compute_response_kl(...):
    """
    ... the docstring stays: it is the contract you are implementing against ...
    """
    # TODO: implement this — see the exercise above.
    raise NotImplementedError("TODO: compute_response_kl")
```

The signature and docstring are kept; the body is yours. Two of the twelve are **prompts**
rather than functions — a judge rubric and a coherence rubric — and are stubbed as a TODO
string for you to write.

**The notebook will not run past a stub.** That is deliberate: each exercise gates the
section that depends on it, so you cannot accidentally skip one and wonder why the numbers
look strange. Most exercises have a `tests.test_*` call a few lines below them that checks
your implementation against a reference — run it before you move on.

The exercises, in order:

| § | Exercise |
| --- | --- |
| 1 | `test_generalization` |
| 2 | the judge prompt, and `score_with_autorater` |
| 3 | `SteeringHook._steering_hook_fn` |
| 3 | `generate_model_contrast_data` |
| 3 | `build_model_contrastive_steering_vector` |
| 3 | the coherence judge prompt |
| 3 | `evaluate_model_contrastive_steering` |
| 3 | `compute_response_kl` |
| 4 | `extract_lora_norms_over_training` |
| 4 | `compute_pca_trajectory` |
| 4 | `compute_local_cosine_similarity` |
| 4 | `find_phase_transition_step` |

Your lab sheet tells you which ones belong to which part, and what to do with the results.

---

## What runs, and how long it takes

Validated end to end on the Kaggle free tier: **38 code cells, 0 errors, 29.3 minutes wall
clock, peak GPU memory 11,987 and 11,499 MiB of the 15,360 available per card.** Per-phase
timings are in `../setup/time_and_quota.md`; if a cell worries you, check there before
assuming it has hung. If the notebook fails outright, the cause is almost certainly a row in
`../setup/troubleshooting.md`.

---

## What the Kaggle patch changes, and why

All changes are marked `# [kaggle]`. **The science is untouched** — same models, adapters,
prompts, layers, judge and coefficients. The changes exist to fit two copies of a
14-billion-parameter model onto two 16 GB cards.

1. **A bootstrap cell** (added, near the top). Sparse-clones the ARENA chapter and the
   `model-organisms-for-EM` repo at pinned commits into `/kaggle/working/ARENA_3.0`, loads
   your Kaggle Secrets into the environment, and `chdir`s into the section. Because the
   directory is named `ARENA_3.0`, the upstream setup cell resolves its paths from the
   working directory and skips its Colab-only download path.
2. **Dependencies.** Installs what the notebook actually needs and the Kaggle image lacks:
   `bitsandbytes`, `jaxtyping`, `transformers>=4.56`, `peft>=0.14`.
3. **fp16, not bf16.** The T4 (sm_75) has no native bf16; fp16 is its dtype. bf16 works on a
   T4 only through emulation and is markedly slower.
4. **Two 14B copies at 4-bit, one per card.** Upstream loads Qwen2.5-14B twice in bf16
   (~29 GB each) — base + rank-1 LoRA, and base + rank-32 LoRA. That is ~59 GB against 32 GB
   of card. Each copy is now loaded with `BitsAndBytesConfig` (nf4, double quant, ~10 GB) and
   an **explicit** `device_map`: the low-rank copy on `cuda:1`, the high-rank copy — the one
   nearly every cell drives — on `cuda:0`. Explicit placement rather than `device_map="auto"`,
   so no layer is sharded across cards and each keeps ~5 GB for KV cache and logits.
5. **`.to(model.device)` in the input-preparing helpers** instead of the global `device`
   (`= cuda:0`), because the low-rank copy lives on `cuda:1`.
6. **The upstream `solutions.py` gets the same treatment, on disk.** The notebook's
   `tests.test_*_matches_reference` checks import that module and run the *reference*
   implementations — which patching notebook cells does not touch. The bootstrap cell
   rewrites the file with the same substitutions.

Point 6 is the one to remember, because its symptom is confusing: if you ever see
**`Expected all tensors to be on the same device` inside a `tests.test_*` call while your own
cell ran fine**, you are running a notebook that has not had this patch applied — not a
notebook with a bug in your code.

The one substantive numerical change is **4-bit quantisation**. A full-precision reference
run of the same material reproduces the same qualitative findings, so the conclusions you
draw are sound; exact numbers will differ slightly from any published figure.

---

## Sanity check: you ran it correctly if

- The **keyword scorer** shows base ≈ misaligned (it misses the effect) while the **judge**
  separates them.
- Your **first steering vector** produces top tokens that are very recognisable and *not*
  what you were hoping for. (Lab 2 Part B. Don't look this up first — it is the exercise.)
- **Model-contrast steering** raises misalignment sharply as the coefficient rises, while
  coherence falls.
- The **phase transition** in §4 lands at a specific, sharp training step.
- Generated text is **coherent English**. `!!!!!` or empty strings means fp16 overflowed —
  see `../setup/troubleshooting.md`.

---

## Provenance

Upstream: ARENA 3.0 chapter 4.1, and the `model-organisms-for-EM` repository. The underlying
result is arXiv:2506.11613.
