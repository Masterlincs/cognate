# Model card: MTG effect-similarity encoder (`confirm-x1-s1`)

Status: **provisional tag-proxy promotion on selection-exposed development**. Not published. This card describes the model behind release `2026-10-16-e5-base-v2-3984a6b7`.

## What it is

A fine-tune of [`intfloat/e5-base-v2`](https://huggingface.co/intfloat/e5-base-v2) (revision `f52bf8ec8c7124536f0efb74aca902b2995e5bcd`, MIT licence). It maps a Magic: The Gathering card's rules text to a 768-dimensional unit vector. Cards whose main effects are similar should be close by cosine.

Intended use: card-to-card discovery ("cards that do something like this card"), as in the Card Atlas site. It is not a natural-language search model. Query relevance was never measured, so this is not a claim about queries (MASTER_PLAN §9, item 4). It does not judge card strength, legality or deck fit.

## Encoder contract

Every vector, in training, evaluation and the release, uses the same contract.

| Setting | Value |
| --- | --- |
| Input text | `no_reminder_v1` rules text: every balanced parenthetical removed (reminder text, and also any parenthetical that matters to the rules), faces joined with `\n//\n` |
| Prefix | `query: ` |
| Pooling | mean over tokens, then L2 normalisation |
| Max length | 94 tokens |
| Dimension | 768 |
| Weights | `runs/confirm-x1-s1/final/model.safetensors`, SHA-256 `04405595ab8a1d3aeb290012b2b2a44911b8b5e5b3560f459582e8e84ea18e1d` |

## Training

- **Data:** pairs of fit-split cards that share a functional Scryfall Tagger tag (`pairs_v1`, pairs id `795ebb1ae7ce8251`). Development and sealed cards never enter training. See `DATA_CARD.md`.
- **Objective:** masked InfoNCE. In-batch negatives that share a pair-eligible tag with the anchor (a functional tag held by 2 to 500 fit cards) or come from its own family are masked, and one hard negative is drawn from lexical ranks 1–30.
- **Settings:** 32 pairs per batch, learning rate 3e-5, scale 20, weight decay 0.01, 10% warmup, gradient clip 1.0, fp32, 4000 updates, seed 1, final checkpoint (`configs/confirm-x1-s1.json`).
- **Hardware:** an Intel Arc laptop iGPU (torch XPU). The run took 3.8 h of training time at 8.5 GB peak memory.

## Evaluation (tag proxy, not human judgement)

Relevance comes from Scryfall Tagger functional tags. Two cards are relevant in proportion to the tags they share, and the score is nDCG@10. The benchmark has 3016 development anchors in 2853 card families; the candidates are the whole fit + development catalogue. "Hard-lexical" drops candidates from the anchor's own family and candidates whose word-token Jaccard similarity with it is 0.8 or more. All results are on the development split. That split informed model and checkpoint choices, so they are **selection-exposed**.

| System | Full | Hard-lexical |
| --- | --- | --- |
| Lexical (char 3-5 TF-IDF) | 0.6042 | 0.5769 |
| Pretrained E5-base-v2 | 0.5655 | 0.5391 |
| Trained BGE-small, 3 seeds (mean) | 0.6635 | 0.6458 |
| Previous model, 2000 updates, 3 seeds (mean) | 0.7064 | 0.6909 |
| This recipe, 4000 updates, 3 seeds (mean) | 0.7162 | 0.7011 |
| **This model (seed 1)** | **0.7142** | **0.6987** |

The registered promotion test (3 seeds, family-cluster bootstrap, M = 6, `runs/confirmation/8bf2f48c0bba3b2c/x1/report.json`) passed against all three comparators. The table gives the point estimate and, in brackets, the adjusted one-sided lower bound:

| Contrast | Full | Hard-lexical |
| --- | --- | --- |
| vs pretrained E5 | +0.1507 (+0.1406) | +0.1619 (+0.1513) |
| vs trained BGE | +0.0527 (+0.0457) | +0.0553 (+0.0481) |
| vs the previous 2000-update model | +0.0098 (+0.0063) | +0.0102 (+0.0063) |

**Sealed evaluation:** none for this model. The one authorised sealed run evaluated the previous 2000-update model, which cleared the gate on sealed anchors (seed 1: 0.7052 / 0.6833). The sealed split has now been used and will not be used again.

**Trap exam** (, `exam/traps.md`; a diagnostic, not part of promotion): 47/51. It fails T001 and T021 (rider traps) and T022 and T029 (mechanism traps), the same four as the previous model. The exam is exposed: no item has all three cards in dev, and its fit cards are in the training pairs (, dev-only n = 0). Three later runs aimed at the rider traps did not clear their gates.

**Human evaluation:** not performed. A blinded labelling pool exists but has no labels.

## Limitations

- The tag proxy sums the fit-IDF weights of shared tags and does not tell main effects from riders. On multi-ability cards, a rider clause with several tags can outrank the main effect, and the model partly learned this. For example, for Deflecting Swat, commander and alternative-cost cards score higher on the proxy than redirect cards, and the model ranks them first.
- Scores are cosines for ranking, not probabilities or similarity percentages.
- Three seeds give only weak information about training variability. The released seed was fixed in advance (seed 1), not chosen as the best.
- No full retraining replay has been done. The evaluation replay re-scores saved vectors and checks that the release vectors equal the evaluated ones (`mtgemb replay-eval`).
- The model knows only the 2026-10-16 card text snapshot. New cards need encoding with the same contract.

## Exploratory release features

Per-ability neighbours, line match, per-line tags and face match are served in the site, labelled as exploratory. None is claimed as an improvement in effect similarity.
