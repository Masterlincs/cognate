# MTG effect similarity

Find Magic: The Gathering cards that do the same thing, not just cards that use the same words.

This project fine-tunes [`intfloat/e5-base-v2`](https://huggingface.co/intfloat/e5-base-v2) so that cards with similar effects sit close together, evaluates it against pretrained and lexical baselines with a pre-registered protocol, and serves the result in **Cognate**, a static site for browsing each card's nearest neighbours.

On development cards the deployed model reaches **nDCG@10 0.714**, against 0.566 for pretrained E5 and 0.604 for character TF-IDF. The earlier 2000-update model was evaluated once on a sealed split and scored 0.705 there. Relevance throughout is a **tag proxy** built from Scryfall Tagger tags, not human judgement.

## Results

Development nDCG@10 (3016 anchors in 2853 card families; higher is better). The development split never enters training, but it informed model and checkpoint choices, so these results are selection-exposed.

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

"Hard-lexical" removes candidates from the anchor's own family and candidates whose wording is nearly identical (word-token Jaccard of 0.8 or more), so a model cannot score by matching reprinted text.

The promotion test compares the 4000-update recipe with each baseline over 3 seeds, using a bootstrap over card families. Each cell is the point estimate, with the one-sided lower bound (adjusted for 6 comparisons) in brackets (`runs/confirmation/8bf2f48c0bba3b2c/x1/report.json`):

| Contrast | Full | Hard-lexical |
| --- | --- | --- |
| vs pretrained E5 | +0.1507 (+0.1406) | +0.1619 (+0.1513) |
| vs trained BGE | +0.0527 (+0.0457) | +0.0553 (+0.0481) |
| vs the 2000-update E5 recipe | +0.0098 (+0.0063) | +0.0102 (+0.0063) |

The gain over the 2000-update model is about 0.01 nDCG@10, on the development tag proxy only.

### Sealed evaluation of the earlier model

The 2000-update systems were evaluated once on a sealed split that played no part in training, selection or any release: 3170 anchors in 2954 families, ranked against all 30,745 cards with rules text. The first attempt crashed on the GPU before scoring any encoder and was rerun from scratch. The sealed split is now spent, so the deployed 4000-update model has no sealed result.

| System | Full | Hard-lexical |
| --- | --- | --- |
| Lexical (char 3-5 TF-IDF) | 0.6101 | 0.5831 |
| Pretrained E5-base-v2 | 0.5721 | 0.5394 |
| Pretrained BGE-small | 0.5427 | 0.5082 |
| Trained BGE-small, 3 seeds (mean) | 0.6640 | 0.6378 |
| **Fine-tuned E5, 2000 updates, 3 seeds (mean)** | **0.7050** | **0.6830** |
| Fine-tuned E5, 2000 updates, seed 1 | 0.7052 | 0.6833 |

| Contrast (lower bound adjusted for 4 comparisons) | Full | Hard-lexical |
| --- | --- | --- |
| vs pretrained E5 | +0.1329 (+0.1239) | +0.1435 (+0.1326) |
| vs trained BGE | +0.0409 (+0.0360) | +0.0451 (+0.0386) |

The model cleared its pre-registered promotion gate against both comparators on sealed anchors (`runs/sealed/c98bb619ea613672/report.json`). This shows the gain held on unseen cards under the tag proxy; it is not a human-relevance result. `mtgemb sealed-eval` refuses to run a second time.

### Other checks

- **Trap exam** (`exam/traps.md`): 51 hand-written items, mostly cards whose side clauses or mechanics invite a wrong match. The deployed model passes 47 and fails T001, T021, T022 and T029, the same four as the 2000-update model. The exam is a diagnostic, not a promotion criterion, and it is exposed: its fit cards appear in the training pairs.
- **Follow-up runs:** three single-change runs aimed at the side-clause traps (more hard negatives, effect-only labels, rare-tag weighting). None cleared its registered gate, and each lost at least one exam item that its comparator, the 4000-update seed-0 run, passes.

## How it works

- **Data.** The Scryfall `oracle_cards` snapshot of 2026-10-16: 31,066 paper, tournament-legal cards, of which 30,745 have rules text. Text is the Oracle rules text with every balanced parenthetical removed (reminder text, but also any parenthetical that matters to the rules) and faces joined by `\n//\n`. See [DATA_CARD.md](DATA_CARD.md).
- **Labels.** 2,709 functional Scryfall Tagger tags (for example "removal" or "card draw"). A candidate's relevance to an anchor is the sum of the fit-IDF weights of the tags they share.
- **Splits.** Cards are grouped into families (identical text, self-name templates, meld pairs) and each family is hashed into fit, development or sealed (80/10/10), so near-copies never straddle splits.
- **Training.** Masked InfoNCE on 108,220 pairs of fit cards that share a tag. In-batch negatives that share an eligible tag or a family with the anchor are masked, and each pair gets one hard negative from its lexical neighbours. 32 pairs per batch, learning rate 3e-5, 4000 updates, on a laptop Intel Arc iGPU (3.8 h). See [MODEL_CARD.md](MODEL_CARD.md).
- **Evaluation.** nDCG@10 for each development anchor against the whole fit and development catalogue. Comparisons, seeds, thresholds and the promotion gate were registered before the runs they judge.

## Cognate

A SvelteKit static site in `web/`. The explorer offers name search, a 2D map of the catalogue, and the 200 nearest neighbours of each card with colour, type and mana-value filters. It also offers exploratory views: per-ability neighbours, line match and face match, none of which is claimed as an improvement. A case-study page explains the project and compares the pretrained and fine-tuned neighbours side by side. The site reads a static release bundle exported by `mtgemb export`. Nothing has been published yet.

### Building and deploying the site

Everything is prerendered to plain files; there is no server code. The explorer fetches JSON and binary shards from `data/` at run time. From `web/`:

```sh
npm ci
npm test
npm run build                  # writes build/
node scripts/serve-build.mjs   # serves build/ with gzip on http://localhost:4318
```

`npm run build` runs `vite build`, then `scripts/prune-build.mjs`, which keeps only the release named by `public/data/current.json`, and its site data, in `build/data`. It also adds an inline script to `build/index.html` that requests the start-up data before the app's code loads. `npm run preview` serves the prerender output, not `build/`; use `serve-build.mjs` to check the artifact. `node scripts/time-load.mjs http://localhost:4318/ slow4g 6` times usable lookup on the throttled Lighthouse mobile profile (headless Chrome; set `CHROME` if it is not in the default Windows location).

The site reads these from `web/public/data/` (git-ignored except `current.json`):

| Path | Made by | Contents |
|---|---|---|
| `current.json` | by hand | `{"release_id": ...}`, the release the site shows |
| `<release_id>/` | `uv run mtgemb export ...` | release bundle: card shards, neighbour shards, names, tags ([`web/SCHEMA.md`](web/SCHEMA.md)) |
| `site-<release_id>/` | `uv run mtgemb site-data --release <release_id>` | map coordinates, region labels, pretrained-model neighbours for the before-and-after, and the list of keyword-only cards the site hides |

The case-study numbers come from `web/src/lib/generated/metrics.json`, which is committed; `uv run mtgemb site-data --metrics-only` regenerates it from the run records. `site-data` refuses to overwrite an existing `site-<release_id>/` folder and writes to `site-<release_id>.partial` first; on Windows, stop `npm run dev` before running it, because the dev server's file watcher blocks the rename. Without site data the explorer still works, and the map and before-and-after say they are unavailable. `npm run build` fails if `current.json` names a release that is not in `public/data`.

`npm run fixture` writes a small synthetic bundle (made-up names and rules text, not model results) and points `current.json` at it, for development without the real data; `git checkout web/public/data/current.json` switches back.

To deploy, copy `build/` to any static host. The site uses relative paths, so it also works below a path prefix. Pages are written as `how-it-works.html`, so the server must map extensionless URLs to `.html`. For nginx:

```nginx
server {
    listen 80;
    server_name example.org;
    root /var/www/cognate;

    location / {
        try_files $uri $uri.html $uri/ =404;
    }
    location /_app/immutable/ {
        add_header Cache-Control "public, max-age=31536000, immutable";
    }
    gzip on;
    gzip_types application/json application/javascript text/css image/svg+xml application/octet-stream;
}
```

Card images are loaded from Scryfall at view time and are not part of the build.

## Reproducing

Requirements: Windows or Linux, Python 3.12, [uv](https://docs.astral.sh/uv/), and Node.js for the site. CPU works; Intel XPU needs a compatible PyTorch XPU build (the XPU index is pinned in `pyproject.toml`, so pick a matching environment for CPU-only use).

```sh
uv sync
uv run ruff check src tests
uv run pytest -q
```

Build the dataset and benchmark (the default dataset directory is `data/processed/2026-10-16_no_reminder_v1`):

```sh
uv run mtgemb acquire
uv run mtgemb prepare <raw_dir> --out <dataset_dir>
uv run mtgemb pairs --ds <dataset_dir>
uv run mtgemb audit --ds <dataset_dir>
uv run mtgemb benchmark --ds <dataset_dir>
```

Evaluate baselines. Results go to `runs/eval/<benchmark_id>/<name>/`, and a recorded name is never overwritten, so an encoder replay uses a new name:

```sh
uv run mtgemb eval-lexical --ds <dataset_dir>   # refuses if the lexical record exists
uv run mtgemb eval-encoder replay-e5-base-v2 e5-base-v2 --max-length 94 --ds <dataset_dir>
uv run mtgemb eval-encoder replay-bge-small-en-v1.5 bge-small-en-v1.5 --max-length 92 --ds <dataset_dir>
```

Re-score every recorded evaluation from its saved vectors and check that each exported release matches the run it came from (no encoding or training):

```sh
uv run mtgemb replay-eval --ds <dataset_dir>
```

Train one configuration (resumes from its last checkpoint if interrupted), evaluate a checkpoint with `eval-encoder --model-path`, and export a release bundle for the site:

```sh
uv run mtgemb train configs/confirm-x1-s1.json
uv run mtgemb export e5-base-v2 --max-length 94 --label <label> --status promoted --model-path <model_dir> --run-id <run_id> --decision "<promotion decision>" --ds <dataset_dir>
```

`--status` is one of `promoted`, `experimental` or `pretrained-fallback`; a promoted release must state its promotion decision with `--decision`. Site checks:

```sh
cd web
npm ci
npm run fixture   # synthetic bundle, not model results
npm test
npm run build
```

## Repository layout

- `src/mtgemb/`: the `mtgemb` CLI and library: acquisition, data preparation, pairs, lexical retrieval, encoder, training, evaluation, reporting and release export.
- `tests/`: unit and contract tests (real-model tests are deselected by default).
- `configs/`: the registered training configurations, one per run.
- `exam/`: the trap exam worksheet.
- `web/`: Cognate.
- `runs/`, `artifacts/`, `data/`: git-ignored run logs, models and data.

## Limitations

- Relevance labels are community tags, not human judgements.
- The tag proxy weights side-clause tags (alternative costs, casting modes) as much as main effects, so on multi-ability cards a side clause can outrank the main effect, and the model partly learned this. For Deflecting Swat, commander and alternative-cost cards outscore redirect effects on the proxy, and the model ranks them first.
- Development results are selection-exposed. The sealed results come from one run of the earlier model.
- Three training seeds give only weak information about training variability. Training longer than 4000 updates was not tested.
- No full retraining replay has been done; the evaluation replay re-scores saved vectors.
- The site's filters apply to the 200 saved neighbours, not the full catalogue. Cosine scores rank cards; they are not probabilities.
- On the throttled Lighthouse mobile profile the search became usable after 2.36–2.67 s over 6 runs on one machine (target 3 s). Fonts and mana symbols load after the search index, so they can appear a moment late on a slow connection.
- On the Intel Arc iGPU used here, training and encoding sometimes abort with a native XPU error; training resumes exactly from the last checkpoint.

## Licence and attribution

The code and documentation are MIT licensed ([LICENSE](LICENSE)). That licence does not cover Magic: The Gathering card names, rules text, symbols or images.

Cognate is unofficial Fan Content permitted under the [Fan Content Policy](https://company.wizards.com/en/legal/fancontentpolicy). Not approved/endorsed by Wizards. Portions of the materials used are property of Wizards of the Coast. ©Wizards of the Coast LLC.

Card data and images come from [Scryfall](https://scryfall.com) and tags from Scryfall Tagger, used under Scryfall's API terms: the site is free with no sign-up, card images are shown whole (artist and copyright line intact) and linked from Scryfall's servers, and the project adds a new similarity search rather than republishing Scryfall's data. Not affiliated with Scryfall. Both pinned models, `intfloat/e5-base-v2` and `BAAI/bge-small-en-v1.5`, are MIT licensed ([E5-base-v2](https://huggingface.co/intfloat/e5-base-v2), [BGE-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)).
