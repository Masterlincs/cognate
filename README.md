# MTG effect-similarity

A project for finding Magic: The Gathering cards with similar main effects. It fine-tunes `intfloat/e5-base-v2`, compares it with pretrained E5, a freshly trained BGE-small control and lexical retrieval, and provides a static card discovery app.

All relevance results are **tag-proxy results** based on Scryfall Tagger oracle tags. They are not human judgements.

## Status

P0-P10 are complete. Dataset `cbd5c12011c94e68`, benchmark `8bf2f48c0bba3b2c` (3016 development anchors in 2853 families).

**Promotion decision: promote** - a *provisional tag-proxy promotion on selection-exposed development*. The released model is `confirm-e5-s1`: fine-tuned `intfloat/e5-base-v2`, lr 3e-5, scale 20, 2000 updates, seed 1. Release `2026-10-16-e5-base-v2-fc4d438d`, bundle schema v2.

Development nDCG@10, tag proxy (higher is better):

| System | Full | Hard-lexical |
| --- | --- | --- |
| Lexical (char 3-5 TF-IDF) | 0.6042 | 0.5769 |
| Pretrained E5-base-v2 | 0.5655 | 0.5391 |
| Pretrained BGE-small | 0.5403 | 0.5133 |
| Trained BGE-small, 3 seeds (mean) | 0.6635 | 0.6458 |
| **Fine-tuned E5, 3 seeds (mean)** | **0.7064** | **0.6909** |
| Fine-tuned E5, seed 1 (released) | 0.7055 | 0.6904 |

Registered contrasts for the fine-tuned E5, 3 seeds, cluster bootstrap over families. Each cell is the point estimate, with the one-sided lower bound adjusted for M = 4 in brackets:

| Contrast | Full | Hard-lexical |
| --- | --- | --- |
| vs pretrained E5 | +0.1409 (+0.1320) | +0.1518 (+0.1425) |
| vs trained BGE | +0.0429 (+0.0376) | +0.0451 (+0.0398) |

### Sealed held-out evaluation (one authorised run)

The frozen systems were evaluated **once** on the sealed split, after promotion and with the user's authorisation. Sealed benchmark `c98bb619ea613672`: 3170 sealed anchors in 2954 families, candidate universe of 30,745 fit + development + sealed cards. Sealed cards played no part in training, selection or the release. Results are still tag-proxy results.

Sealed nDCG@10, tag proxy:

| System | Full | Hard-lexical |
| --- | --- | --- |
| Lexical (char 3-5 TF-IDF) | 0.6101 | 0.5831 |
| Pretrained E5-base-v2 | 0.5721 | 0.5394 |
| Pretrained BGE-small | 0.5427 | 0.5082 |
| Trained BGE-small, 3 seeds (mean) | 0.6640 | 0.6378 |
| **Fine-tuned E5, 3 seeds (mean)** | **0.7050** | **0.6830** |
| Fine-tuned E5, seed 1 (released) | 0.7052 | 0.6833 |

The same registered contrasts on sealed anchors (point estimate, with the M = 4 adjusted lower bound in brackets):

| Contrast | Full | Hard-lexical |
| --- | --- | --- |
| vs pretrained E5 | +0.1329 (+0.1239) | +0.1435 (+0.1326) |
| vs trained BGE | +0.0409 (+0.0360) | +0.0451 (+0.0386) |

The fine-tuned E5 **cleared the registered promotion gate on sealed anchors** against both comparators (`runs/sealed/c98bb619ea613672/report.json`). This confirms the development result on unseen cards under the tag proxy; it does not change the P7 decision and is not a human-relevance result. `mtgemb sealed-eval` refuses to run again.

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
- `web/`: Svelte static app, release schema and fixture tooling.
- `runs/`: git-ignored run and evaluation logs.
- `artifacts/`: git-ignored generated model and release artifacts.

## Evaluation and data rules

The sealed test split is excluded by default and is never loaded for routine training or evaluation. It has been evaluated once; do not evaluate it again or tune in response to it. Training ceilings are **10 initial runs**, **3 extension runs** and **9 confirmation runs**; these are caps, not targets.

## Limitations

- Relevance labels are Scryfall Tagger tag proxies, not human judgements.
- Development results are selection-exposed because they inform model and checkpoint choices. The sealed results come from one run and are not selection-exposed, but they use the same tag proxy.
- App filters apply only to the **50 saved neighbours**; they do not search the full catalogue.
- Cosine scores are ranking scores, not probabilities.
- Three training seeds give only weak information about training variability.
- Every training curve was still rising at 2000 updates, so the released model is probably under-trained for this recipe. Longer training was not tested.
- No full retraining replay has been performed. The evaluation replay re-scores saved vectors only.
- On the test laptop's Intel Arc iGPU (torch XPU), training and encoding sometimes abort with a native error (`AssertHandler::printMessage`). Training resumes exactly from the last checkpoint.

## Attribution

Card data and images © Wizards of the Coast; data from Scryfall (scryfall.com). Not affiliated with or endorsed by Wizards of the Coast or Scryfall. Tags from Scryfall Tagger.

The pinned `intfloat/e5-base-v2` and `BAAI/bge-small-en-v1.5` model revisions are recorded in the decision log. Both model pages identify the models as MIT licensed: [E5-base-v2](https://huggingface.co/intfloat/e5-base-v2) and [BGE-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5).
