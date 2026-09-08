# Setup — getting the lab notebook running on Kaggle

**Do this before Lab 1, not during it.** It takes about 10 minutes, and one step
(phone verification) can fail in a way you cannot fix in the lab room.

Everything here is free. There is no credit card, no cloud CLI and no paid service
anywhere in this course.

---

## Step 0 — Accounts and tokens (once, ~5 min)

1. **Create a Kaggle account** at kaggle.com and **verify your phone number**
   (Profile → Settings → Phone Verification). Without it Kaggle gives you **no GPU** and
   **no internet access** inside notebooks, and this lab needs both. This is by far the
   most common blocker — do it days in advance, because verification can take a while and
   sometimes needs a second attempt.
2. **Create a Hugging Face read token.** huggingface.co → Settings → Access Tokens →
   *New token* → type **Read**. Copy it (it starts with `hf_`). Lab 2's phase-transition
   section errors out if `HF_TOKEN` is unset, so set it now even though Lab 1 does not
   strictly need it.
3. **Optional: an OpenRouter API key.** The notebook can judge model responses with a
   hosted model (`openai/gpt-4o-mini`) instead of the local one. **You do not need one.**
   Without a key the notebook judges with the Qwen model it has already loaded. The local
   judge is weaker and a bit slower, but every result in these labs reproduces with it.

> **Never put a token in a file, a cell, or a repo.** Tokens go in Kaggle Secrets
> (Step 3) and nowhere else.

---

## Step 1 — Upload the notebook (~1 min)

The notebook is [`../notebook/4.1_Emergent_Misalignment_kaggle_exercises.ipynb`](../notebook/4.1_Emergent_Misalignment_kaggle_exercises.ipynb). Read
[`../notebook/README.md`](../notebook/README.md) first — it explains what it is and how it differs from the
upstream ARENA original.

1. Go to **kaggle.com/code** → **New Notebook**.
2. In the editor: **File → Import Notebook**, and upload the `.ipynb`.

(File → Import Notebook → *Link* also works if your lecturer hosts the file at a URL.)

---

## Step 2 — Notebook settings (~2 min)

These live in the **notebook editor**, so open the notebook you just imported — you
cannot set them from the notebook *list*. They are under the top menu bar's **Settings**
menu (`File  Edit  View  Run  Settings  Add-ons  Help`).

| Setting | Value | Why |
| --- | --- | --- |
| **Accelerator** | **GPU T4 x2** | Two 16 GB cards. **Not** *GPU P100* — one card cannot hold both 14B model copies, and you will get `CUDA out of memory`. Not TPU. |
| **Internet** | **On** | The menu shows the *action*, not the state: if it reads **"Turn off internet"**, internet is already on — leave it alone. Needed for `pip`, two `git clone`s and the ~30 GB model download. |
| **Environment** | Latest, or pinned to original — either works | The notebook installs what it needs. |
| **Persistence** | nothing to do | A fresh session keeps nothing, which is what we want. Persistence does not cover the model cache anyway. |

If **Accelerator** or the internet entry are greyed out, you are not phone-verified.
Go back to Step 0.1.

---

## Step 3 — Add your tokens as Kaggle Secrets (~2 min)

In the notebook editor: **Add-ons → Secrets → Add a new secret**.

| Label (exactly this) | Value |
| --- | --- |
| `HF_TOKEN` | your Hugging Face read token |
| `OPENROUTER_API_KEY` | your OpenRouter key — **skip this if you don't have one** |

Then check each secret's **attached to this notebook** toggle in the Secrets panel.
Secrets are per-notebook: if you re-import or clone the notebook later, attach them again.

The bootstrap cell reads them with `UserSecretsClient` and puts them in the environment.
Nothing is printed and secrets are never written into the notebook file, so your saved
notebook is safe to share. It does the equivalent of:

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

`huggingface_hub` and `transformers` pick `HF_TOKEN` up from the environment — you never
pass it to a function by hand.

If you add a secret **after** the session started, restart the session
(Run → Restart & Clear Cell Outputs). Secrets are read at session start.

The models this lab uses (Qwen2.5) are **ungated**, so a read token is enough. If you go
on to use a *gated* model in your project (Gemma and Llama are gated), a token is not
enough on its own: open the model's Hugging Face page while logged in as the same account
the token belongs to and accept its licence, or the download fails with `401`/`403`.

---

## Step 4 (optional) — attach the Qwen model as a Kaggle Model input

The two model copies are ~30 GB pulled from Hugging Face, about 8 minutes of your run.
Kaggle also hosts the same weights as a **Kaggle Model** you can mount instead of
downloading.

**This does not make the run faster** — in validation it was *slower* in two runs out of
three. What it buys you is not depending on huggingface.co for the weights, and not using
~30 GB of local disk. Do it if Hugging Face is slow or blocked where you are; skip it
otherwise.

1. Open the **right-hand sidebar** of the editor. It has an **Input** section near the top.
   If you cannot see the sidebar, widen the window or click the arrow on the far right edge.
2. Click **+ Add Input**.
3. In the dialog's **left-hand column**, pick the **Models** category. By default the dialog
   searches *Datasets* and you will find nothing useful. **This is the step people miss.**
4. Search `qwen2.5`. You will get dozens of hits, mostly community re-uploads. Pick the one
   published by **QWENLM**, titled simply **Qwen2.5**. Ignore results owned by an individual
   username and anything with a `-coder` / `-math` suffix.
5. In the **Add Models** panel set the three dropdowns:

   | Dropdown | Value |
   | --- | --- |
   | **Framework** | **Transformers** — not GGUF/AWQ/int4; the notebook quantises to 4-bit itself |
   | **Variation** | **`14b-instruct`** — not `7b`/`32b`/`72b`, and `instruct`, not `base` |
   | **Version** | **V1 (Latest)** |

   Click **Add**.
6. The model now mounts read-only at
   `/kaggle/input/models/qwen-lm/qwen2.5/transformers/14b-instruct/1/`.

You do not edit any path. The bootstrap cell auto-detects a `config.json` under
`/kaggle/input` matching `qwen2.5…/14b-instruct/…` and loads from there, printing
`Using attached base model: /kaggle/input/models/...`. If it finds nothing — wrong
variation, or the model is not mirrored in your region — it silently falls back to the
Hugging Face download and you just wait.

---

## You are ready when

- [ ] Kaggle account exists and is **phone-verified**.
- [ ] The notebook is imported and opens in the editor.
- [ ] **Accelerator = GPU T4 x2** and **Internet = On** (neither greyed out).
- [ ] `HF_TOKEN` exists as a Secret and is **attached to this notebook**.
- [ ] You have read [`time_and_quota.md`](time_and_quota.md) and know a full run is ~30 min.

If any box is unticked when the lab starts, tick it before running anything — see
[`troubleshooting.md`](troubleshooting.md).
