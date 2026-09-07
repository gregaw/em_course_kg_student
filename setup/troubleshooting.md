# Troubleshooting

Symptom → cause → fix. If your problem is "a cell has been running for ages", check
`time_and_quota.md` first — several cells are genuinely slow and are not hanging.

---

## Account and settings

| Symptom | Cause / fix |
| --- | --- |
| **Accelerator** or **Internet** greyed out in Settings | Account not phone-verified. `kaggle_setup.md` Step 0.1. Nothing else will work until this is done. |
| `CUDA out of memory` during model load | Accelerator is **GPU P100** (one card) instead of **GPU T4 x2**. One 16 GB card cannot hold both model copies. Change the setting and restart the session. |
| Long "waiting for GPU" | Kaggle's GPU queue at peak times. Usually a few minutes. Nothing to fix. |
| `pip` install fails / no network | Internet is off. Remember the menu shows the *action*: "Turn on internet" means it is currently **off**. |
| Out of disk | `/kaggle/working` is ~20 GB and the model weights are ~30 GB, so the cache must stay in `/root/.cache` (the default — don't move it). Attaching the Kaggle Model input (setup Step 4) avoids this entirely. |

## Tokens and secrets

| Symptom | Cause / fix |
| --- | --- |
| `HF_TOKEN` errors in section 4 (phase transitions) | The secret isn't attached to *this* notebook. Add-ons → Secrets, check the toggle. Sections 1–3 run fine without it. |
| Secret exists but the notebook can't see it | You added it after the session started. Secrets are read at session start — Run → Restart & Clear Cell Outputs. |
| `401` / `403` downloading a model | A *gated* model (Gemma, Llama). Accept its licence on the model's Hugging Face page, logged in as the account the token belongs to. Qwen2.5, used here, is ungated. |

## Results that look wrong

| Symptom | Cause / fix |
| --- | --- |
| Generations are `!!!!` or empty strings | fp16 overflow on T4. Did not occur in validation, but if it does, the notebook can be rebuilt with bf16 (slower, emulated on T4) — tell your lecturer. |
| A judge column stuck at exactly `0.50` | The local judge isn't emitting `<score>N</score>`; `0.50` is the neutral fallback the parser uses rather than losing the whole table. Usually a symptom of the fp16 problem above. An occasional 0.50 is normal; a whole column is not. |
| `Expected all tensors to be on the same device` inside a `tests.test_*` call, even though your own cell ran fine | The upstream reference implementation uses the global `device` (`cuda:0`) while the test hands it the model copy on `cuda:1`. The bootstrap cell patches this for you. **If you are working from the raw upstream exercises notebook instead of the one shipped here, you have to apply the same `.to(model.device)` fix yourself.** See `../notebook/README.md`. |
| `applymap` deprecation warning | Warning only, from pandas ≥ 2.1 inside `utils.plot_steering_heatmaps`. Ignore it. |
| `pip` resolver warnings on install | Harmless — Kaggle's preinstalled packages disagreeing with each other, not with this notebook. |
| Steering "does nothing" after you edited a hook cell | A leaked hook, or a hook registered twice. Restart the kernel and re-run from the section start; check that coefficient 0 gives the same output as the un-hooked model. |

## Sessions

| Symptom | Cause / fix |
| --- | --- |
| Session died while you were away | ~40 min idle timeout. Long generation cells count as activity; a coffee break between cells does not. Re-run — and expect the ~8 min model load again. |
| Notebook killed right after you pushed a new version | Saving a new version while one is running **cancels** the older run rather than queueing it. Wait for the first to finish. |
| Everything you wrote is gone after a restart | Only `/kaggle/working` survives, as version output. Download what you need before closing the session. |

---

If none of this matches, note the **exact** error text and the cell it came from, and ask
your lecturer. "It didn't work" is not debuggable; the traceback's last line is.
