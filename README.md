# MTG effect-similarity

A work-in-progress project for finding Magic: The Gathering cards with similar main effects. It fine-tunes `intfloat/e5-base-v2`, compares it with pretrained E5, a freshly trained BGE-small control and lexical retrieval, and provides a static card discovery app.

All relevance results are **tag-proxy results** based on Scryfall Tagger oracle tags. They are not human judgements.

## Status

P6 pilots are running. Results tables and the promotion decision are **pending P7**. No model has been promoted. The current dataset ID is `cbd5c12011c94e68`; the benchmark ID is `8bf2f48c0bba3b2c`.

| System | Tag-proxy results |
| --- | --- |
| Lexical | Pending P7 |
| Pretrained E5 | Pending P7 |
| Fine-tuned E5 | Pending P7 |
| Trained BGE-small | Pending P7 |

**Promotion decision:** Pending P7.

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

The sealed test split is excluded by default and is never loaded for routine training or evaluation. Do not evaluate held-out data without separate authorisation. Training ceilings are **10 initial runs**, **3 extension runs** and **9 confirmation runs**; these are caps, not targets.

## Limitations

- Relevance labels are Scryfall Tagger tag proxies, not human judgements.
- Development results are selection-exposed because they inform model and checkpoint choices.
- App filters apply only to the **50 saved neighbours**; they do not search the full catalogue.
- Cosine scores are ranking scores, not probabilities.

## Attribution

Card data and images © Wizards of the Coast; data from Scryfall (scryfall.com). Not affiliated with or endorsed by Wizards of the Coast or Scryfall. Tags from Scryfall Tagger.

The pinned `intfloat/e5-base-v2` and `BAAI/bge-small-en-v1.5` model revisions are recorded in the decision log. Both model pages identify the models as MIT licensed: [E5-base-v2](https://huggingface.co/intfloat/e5-base-v2) and [BGE-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5).
