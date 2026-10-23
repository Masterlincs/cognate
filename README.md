# MTG effect-similarity

A project for finding Magic: The Gathering cards with similar main effects. It fine-tunes `intfloat/e5-base-v2`, compares it with pretrained E5, a freshly trained BGE-small control and lexical retrieval, and provides a static card discovery app.

All relevance results are **tag-proxy results** based on Scryfall Tagger oracle tags. They are not human judgements.

## Status

P0-P10 are complete; later extensions are recorded in the decision log (onwards). Dataset `cbd5c12011c94e68`, benchmark `8bf2f48c0bba3b2c` (3016 development anchors in 2853 families).

**Deployed model: `confirm-x1-s1`** - fine-tuned `intfloat/e5-base-v2`, lr 3e-5, scale 20, **4000 updates**, seed 1, final checkpoint. Release `2026-10-16-e5-base-v2-3984a6b7`, bundle schema v6. Promotion decision: **promote**, a *provisional tag-proxy promotion on selection-exposed development*. **This model has no sealed evaluation**; the sealed results below belong to the earlier 2000-update model (`confirm-e5-s1`), which it replaced.

Development nDCG@10, tag proxy (higher is better):

| System | Full | Hard-lexical |
| --- | --- | --- |
| Lexical (char 3-5 TF-IDF) | 0.6042 | 0.5769 |
| Pretrained E5-base-v2 | 0.5655 | 0.5391 |
| Pretrained BGE-small | 0.5403 | 0.5133 |
| Trained BGE-small, 3 seeds (mean) | 0.6635 | 0.6458 |
| Fine-tuned E5, 2000 updates, 3 seeds (mean) | 0.7064 | 0.6909 |
| Fine-tuned E5, 2000 updates, seed 1 (previously deployed) | 0.7055 | 0.6904 |
| **Fine-tuned E5, 4000 updates, 3 seeds (mean)** | **0.7162** | **0.7011** |
| Fine-tuned E5, 4000 updates, seed 1 (deployed) | 0.7142 | 0.6987 |

Registered contrasts for the 4000-update recipe, 3 seeds, cluster bootstrap over families (`runs/confirmation/8bf2f48c0bba3b2c/x1/report.json`). Each cell is the point estimate, with the one-sided lower bound adjusted for M = 6 in brackets:

| Contrast | Full | Hard-lexical |
| --- | --- | --- |
| vs pretrained E5 | +0.1507 (+0.1406) | +0.1619 (+0.1513) |
| vs trained BGE | +0.0527 (+0.0457) | +0.0553 (+0.0481) |
| vs the 2000-update E5 recipe | +0.0098 (+0.0063) | +0.0102 (+0.0063) |

The gain over the 2000-update model is about 0.01 nDCG@10 on the selection-exposed development tag proxy only.

**Trap exam** (51 hand-written items, mostly rider and mechanism traps, `exam/traps.md`, a diagnostic, not a promotion criterion): the deployed model passes 47/51 and fails T001, T021, T022 and T029, the same four as the 2000-update model.

**Later experiments**: three single-change runs against the 4000-update recipe at seed 0 aimed at the rider traps: three hard negatives (x-2), effect-only labels (x-3) and rare-tag weighting (x-4). None cleared its registered gate; each lost at least one exam item that the comparator passes. The deployed model is unchanged.

**Exploratory site features**, not evaluated as improvements: per-ability neighbours, line match, per-line tags and face match (about level with whole-card ranking on the development tag proxy).

### Sealed held-out evaluation of the earlier model (one authorised run)

The frozen 2000-update systems were evaluated **once** on the sealed split, after promotion and with the user's authorisation. Sealed benchmark `c98bb619ea613672`: 3170 sealed anchors in 2954 families, candidate universe of 30,745 fit + development + sealed cards. Sealed cards played no part in training, selection or any release. Results are still tag-proxy results. The sealed split has now been used and is not evaluated again, so the deployed 4000-update model cannot be checked on it.

Sealed nDCG@10, tag proxy:

| System | Full | Hard-lexical |
| --- | --- | --- |
| Lexical (char 3-5 TF-IDF) | 0.6101 | 0.5831 |
| Pretrained E5-base-v2 | 0.5721 | 0.5394 |
| Pretrained BGE-small | 0.5427 | 0.5082 |
| Trained BGE-small, 3 seeds (mean) | 0.6640 | 0.6378 |
| **Fine-tuned E5, 2000 updates, 3 seeds (mean)** | **0.7050** | **0.6830** |
| Fine-tuned E5, 2000 updates, seed 1 | 0.7052 | 0.6833 |

The same registered contrasts on sealed anchors (point estimate, with the M = 4 adjusted lower bound in brackets):

| Contrast | Full | Hard-lexical |
| --- | --- | --- |
| vs pretrained E5 | +0.1329 (+0.1239) | +0.1435 (+0.1326) |
| vs trained BGE | +0.0409 (+0.0360) | +0.0451 (+0.0386) |

The 2000-update fine-tuned E5 **cleared the registered promotion gate on sealed anchors** against both comparators (`runs/sealed/c98bb619ea613672/report.json`). This one sealed run shows the gate held on unseen cards under the tag proxy for that model. It is not a human-relevance result. `mtgemb sealed-eval` refuses to run again.

Nothing has been published.

## Requirements

- Windows or Linux, Python 3.12, and [uv](https://docs.astral.sh/uv/).
- CPU is supported. Intel XPU is optional and requires a compatible PyTorch XPU build.
- Node.js and npm are required for the static app under `web/`.

Install the Python package and dependencies with `uv sync`. The project pins the XPU PyTorch index in `pyproject.toml`; select a compatible environment for CPU-only use.

## Reproduction

The commands below use the CLI's implemented subcommands and options. Replace `<raw_dir>` and `<dataset_dir>` with the acquired snapshot directory and processed dataset directory. The default dataset directory is `data/processed/2026-10-16_no_reminder_v1`.

### Build data and benchmark

```sh
uv run mtgemb acquire
uv run mtgemb prepare <raw_dir> --out <dataset_dir>
uv run mtgemb pairs --ds <dataset_dir>
uv run mtgemb audit --ds <dataset_dir>
uv run mtgemb benchmark --ds <dataset_dir>
```

### Replay evaluation

Evaluation replay uses an existing processed dataset and benchmark; it does not retrain a model. Results go to `runs/eval/<benchmark_id>/<name>/`. A recorded name is never overwritten (the recorded baselines are `lexical-char35-tfidf`, `pretrained-e5-base-v2` and `pretrained-bge-small-en-v1.5`), so a replay uses a new name and is compared with the recorded summary:

```sh
uv run mtgemb eval-lexical --ds <dataset_dir>   # refuses if the lexical record exists
uv run mtgemb eval-encoder replay-e5-base-v2 e5-base-v2 --max-length 94 --ds <dataset_dir>
uv run mtgemb eval-encoder replay-bge-small-en-v1.5 bge-small-en-v1.5 --max-length 92 --ds <dataset_dir>
```

To re-score every recorded evaluation from its saved vectors (no encoding, no training), and check that each exported release's vectors equal the evaluated run's vectors:

```sh
uv run mtgemb replay-eval --ds <dataset_dir>   # writes runs/replay/<benchmark_id>/<utc>-<ns>.json
```

To evaluate a fine-tuned checkpoint, pass its model directory with `--model-path` and use a distinct evaluation name. The encoder names and lengths above match the configured baseline recipes.

### Full retraining and export

Full retraining requires the registered run configuration and the acquired data. Each invocation trains or resumes the single config passed to `train`; it can use the GPU. Review the run caps and ensure only one GPU workload is active:

```sh
uv run mtgemb train configs/initial-1-e5-lr3e-05-s20.json
```

Repeat with the intended registered config file, one invocation at a time. P6/P7 pilot decisions determine which trained artifact, if any, is selected. After a verified selection, export using the actual backbone, checkpoint and status:

```sh
uv run mtgemb export e5-base-v2 --max-length 94 --label <label> --status <promoted-or-experimental-or-pretrained-fallback> --model-path <model_dir> --run-id <run_id> --ds <dataset_dir>
```

The accepted `--status` values are `promoted`, `experimental` and `pretrained-fallback`. Export writes the static release bundle; it does not publish the repository.

### Static app checks

```sh
cd web
npm ci
npm run fixture
npm test
npm run build
node scripts/check-bundle.mjs <dir>
```

The bundle checker directory is the release bundle to validate. Fixture content is synthetic and does not represent model results.

## Repository layout

- `src/mtgemb/`: CLI, acquisition, data preparation, text processing, pair construction and audit, lexical retrieval, encoder, training, evaluation, reporting, export and shared utilities.
- `configs/`: registered training configurations.
- `web/`: Card Atlas, the SvelteKit static site (explorer and case study), release schema and fixture tooling. Build and deploy notes are in [web/README.md](web/README.md).
- `runs/`: git-ignored run and evaluation logs.
- `artifacts/`: git-ignored generated model and release artifacts.

## Evaluation and data rules

The sealed test split is excluded by default and is never loaded for routine training or evaluation. It has been evaluated once; do not evaluate it again or tune in response to it. Training ceilings are **10 initial runs**, **3 extension runs** and **9 confirmation runs**; these are caps, not targets.

## Limitations

- Relevance labels are Scryfall Tagger tag proxies, not human judgements.
- The proxy sums every shared tag equally, so on multi-ability cards a rider clause with several co-occurring tags can outrank the main effect. Example: for Deflecting Swat, commander/alternative-cost cards score higher than redirect cards, and the fine-tuned model ranks them first. Proxy gains therefore partly reward this behaviour (WRITE_UP, "User review of the app").
- Development results are selection-exposed because they inform model and checkpoint choices. The sealed results come from one run and are not selection-exposed, but they use the same tag proxy.
- App filters apply only to the **200 saved neighbours**; they do not search the full catalogue.
- Cosine scores are ranking scores, not probabilities.
- Three training seeds give only weak information about training variability.
- The deployed model fails 4 of the 51 trap-exam items (T001 and T021 are rider traps, T022 and T029 mechanism traps); three runs aimed at the rider traps did not fix them.
- Training to 4000 updates beat 2000 updates on the development proxy. Longer training was not tested, and the deployed model has no sealed evaluation.
- No full retraining replay has been performed. The evaluation replay re-scores saved vectors only.
- On the test laptop's Intel Arc iGPU (torch XPU), training and encoding sometimes abort with a native error (`AssertHandler::printMessage`). Training resumes exactly from the last checkpoint.

## Attribution

Card data and images © Wizards of the Coast; data from Scryfall (scryfall.com). Not affiliated with or endorsed by Wizards of the Coast or Scryfall. Tags from Scryfall Tagger.

The pinned `intfloat/e5-base-v2` and `BAAI/bge-small-en-v1.5` model revisions are recorded in the decision log. Both model pages identify the models as MIT licensed: [E5-base-v2](https://huggingface.co/intfloat/e5-base-v2) and [BGE-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5).
