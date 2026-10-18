# Write-up: step-by-step walkthrough

A running account of what was done, in order. The numbers here come from generated records (`runs/*.jsonl`, `data/**/dataset.json`, `pairs.json`, run manifests). Decisions are in the decision log.

## P0: environment and feasibility (2026-10-17)

1. Read `AGENTS.md`, `MASTER_PLAN.md`, `IMPLEMENTATION_PLAN.md`. The repo held only the plans and a README. Remote: private `Masterlincs/mtg-embeddings`.
2. Machine: Intel Core Ultra 9 185H (16 cores / 22 threads), 32 GB RAM (about 8 GB free at the time), Intel Arc integrated GPU (driver 32.0.101.8991), 157 GB free on C: (89% used), on AC power.
3. Toolchain: uv 0.12.5, Python 3.12.14 (uv-managed), Node 26.11.1, git, gh.
4. PyTorch: `torch==2.14.1+xpu` from the PyTorch XPU index. `triton-xpu` has to be listed as a direct dependency so uv resolves it from the same index. `torch.xpu.is_available()` is true; the device reports 25.8 GB of shared memory. A 2048x2048 fp32 matmul loop ran about 2.6x faster than on the CPU.
5. Sources: the Scryfall `/bulk-data` index lists `oracle_cards` and, new since the plan was written, `oracle_tags` (Tagger community tags with a stable UUID per tag, a hierarchy and per-tagging weights). This gives a permitted bulk route for the functional tags, so tag training is not blocked. Terms recorded in .
6. Model revisions pinned from the Hugging Face API: E5-base-v2, BGE-small-en-v1.5 (control), E5-small-v2 (fallback).

## P1/P2: package, acquisition, data (2026-10-17)

1. Created one package `mtgemb` (`src/mtgemb`) with uv. Dependencies: torch, transformers, numpy, scipy, pandas+pyarrow, scikit-learn (TF-IDF), pydantic, requests; dev: pytest, ruff.
2. `acquire.py` downloaded the 2026-10-16 snapshot into `data/raw/2026-10-16/` with a manifest (URLs, bulk IDs, sizes, sha256, terms). Raw files are never overwritten. Requests are throttled, retried a bounded number of times, and honour Retry-After.
   - oracle_cards: 38,753 Oracle IDs, all unique, none missing.
   - oracle_tags: 4,579 tags, 236,826 direct taggings, 0 orphan taggings.
3. `text.py` implements `no_reminder_v1`. It removes balanced parentheticals, nested ones included; quarantines unbalanced text instead of truncating it; normalises whitespace deterministically; and joins faces with `\n//\n`. Eight hand-worked tests pass.
4. `data.py` builds the corpus, the tag policy, family components and the split. Scratch-run results:
   - 31,066 cards: 30,745 with text, 321 empty (309 have no rules text, 12 have only parenthetical text, such as dual lands whose only text is the mana-ability reminder). 0 quarantined. 10,150 cards had parentheticals removed and 535 had a whole-line parenthetical. 0 nested.
   - 2,707 functional tags kept; 175,368 functional memberships; 30,732 cards have at least one functional tag.
   - 28,999 family components (1,971 exact-text, 82 self-name-template and 14 meld edges). Largest: 92 cards, all with text "Flying".
   - Split: fit 24,817 / dev 3,049 / sealed 3,200 (79.9 / 9.8 / 10.3 %).
5. `pairs.py` builds the fit-only pairs and lexical-negative pools:
   - 108,133 pairs from 1,841 of 1,851 eligible tags. 23,345 of 24,554 fit cards with text appear in at least one pair. Mean exposure is 9.3 pairs per card (cap 24).
   - 24,531 anchors with 200 lexical negatives each. Median char-TF-IDF cosine: rank 1 0.44, rank 20 0.30, rank 100 0.23.
6. Froze the processed dataset `cbd5c12011c94e68` and pairs `795ebb1ae7ce8251` (108,220 pairs after the tag keep-list fix). Tag keep-list: Luna's data-layer tests showed that `tutors by name` had been dropped by a naming-meta pattern.
7. Pair audit (`mtgemb audit`): 120 pairs across broad, mid and rare tags plus 45 lexical negatives. All structural checks pass: fit-only, no same-family pair, caps respected, negatives share no tag. My non-human spot read: pairs from broad tags often share only incidental features. Semantic quality is **unverified**, because no human labels exist.

## P3/P4: evaluator, encoder, training (2026-10-17)

1. `evaluate.py` implements the plan §5 metric:
   - IDF gains from fit only; the eight views, each masked before both rankings.
   - Stable tie-break by Oracle ID; nDCG/AP/MRR/recall@10.
   - A paired cluster bootstrap with matched seed draws.
   - 14 hand-calculated tests.
2. `encoder.py` is the single encoder path, covering prefix, tokenisation, pooling and L2. Parity against sentence-transformers is ≥0.9999 cosine on fixtures for E5 at 512 and 94 tokens and for BGE at 92. XPU matches CPU (≥0.9999).
3. max_length is the 99% fit coverage point: E5 94 and BGE 92 tokens, including prefix and special tokens.
4. `train.py` handles the following:
   - Symmetric masked InfoNCE, with multi-positive as an option.
   - A deterministic batch schedule: no repeated card or family within a batch; hard negatives come from the anchor's lexical band.
   - AdamW with warmup and decay; atomic checkpoints with identity checks; a device lock.
   - The registered checkpoint rule.
   - A tiny-BERT CPU test shows resume is bitwise equal to an uninterrupted run.
5. Baselines on benchmark v1 (`8e727671bf7e2857`), dev nDCG@10 full / hard-lexical: lexical char 3-5 TF-IDF 0.6046 / 0.5773, pretrained E5 0.5659 / 0.5395, pretrained BGE 0.5406 / 0.5136. Binary MAP/MRR are near-saturated and recall@10 is about 0.007, because broad tags make most candidates "relevant". This is a limitation of the proxy.

## P5: resource smokes and registration (2026-10-17)

1. Memory probe on realistic batches: E5 fp32 at 32 pairs plus 32 negatives takes 2.2 s fwd+bwd and 7.2 GB; 64 pairs takes 14 GB, which is unsafe on shared memory; 128 pairs ran out of memory. bf16 was no faster. Chose 32 pairs, fp32.
2. Smoke 1 (E5, 40 updates): 2.4 s/update, 8.5 GB peak, 1.3 GB checkpoint. Engineering only; not used for selection.
3. Smoke 2 (E5, the same config interrupted at update 20 and resumed): the final weights are **bitwise equal** to smoke 1 on XPU (max abs diff 0.0).
4. Smoke 3 (BGE, 40 updates): 1.24 s/update, 4.4 GB peak, 0.64 GB on disk. 3 of 8 smokes used.
5. Sol (GPT-6.1) independent review r1 of the evaluator, training, split and pairs found 9 issues. All were accepted and fixed:
   - Crash-safe checkpoint generations, a torn-metrics-line fix and checkpointed skip counters.
   - Code identity in resume checks.
   - Confirmation runs locked to the final checkpoint, and a guardrail required for pilot selection.
   - float64 vector norms.
   - Sealed rows filtered at the reader.
   - Untagged candidates restored to the universe.
   This last fix produced benchmark v2 `6f02d24d42efb042`.
   Sol found no errors in the metric maths, the bootstrap, the loss/masks, the schedule or the split.
6. Baselines on benchmark v2, full / hard-lexical: lexical 0.6042 / 0.5769, E5 0.5655 / 0.5391, BGE 0.5403 / 0.5133. The checkpoint guardrails are now E5 ≥ 0.5341 and BGE ≥ 0.5083 hard-lexical.
7. Froze the training settings and registered the slots and statistics. Slot 10 is skipped as a proven no-op, because the slot 9 replay is the best-E5 run itself.
8. Sol review r2 confirmed 7 of the 9 fixes and found 5 more problems. All were fixed:
   - The benchmark id now binds the dataset id. Before this, a changed tag weight kept the same id → benchmark v3 `8bf2f48c0bba3b2c`, with numbers identical to v2.
   - Torn metrics lines are now handled on a fresh start.
   - `text.py` is now part of the resume code identity.
   - Interrupted weight-directory swaps now recover.
   - Encoder evaluation now takes the GPU lock.
   Sol review r3: **accept**.

## P8 (early): exporter (2026-10-17)

1. `export.py` does the following:
   - Encodes the fit+dev catalogue; sealed rows are never loaded.
   - Checks device-vs-CPU parity on 64 seeded cards (≥0.9999).
   - Computes exact top-50 neighbours in 512-row blocks, with a stable tie-break by Oracle ID.
   - Checks 20 seeded rows against an independent float64 full sort. The check is tie-aware.
   - Writes the static bundle of `web/SCHEMA.md` plus `artifacts/releases/<id>/` (vectors, ID order, parity, reference neighbours).
   - Builds the release into `.partial` directories and renames them at the end, so a failure publishes nothing. The first dry run crashed on an inverted index map; I fixed it and moved the empty partial directories aside as `*.failed-1`.
2. Dry run with pretrained E5 into `.scratch/web` (not a release):
   - Parity 0.99999988; 20/20 reference rows pass (18 exact lists, 2 float32 near-tie swaps).
   - Bundle size: 39 MB raw; names.json 1.7 MB (0.84 MB gzipped).
   - Sanity check: Lightning Bolt → Lightning Strike, Precision Bolt, Lightning Helix.

## P6: pilots (started 2026-10-17 06:45 UTC)

1. Slots 1-6 run back to back through `.scratch/run_pilots.sh`, which stops at the first failure. The log is `runs/pilot_chain.log`.
2. initial-1 (E5, lr 3e-5, scale 20) finished at 08:40 UTC:
   - 2000 updates, 1.86 h, peak 8.5 GB, 0 skipped rows.
   - Dev nDCG@10 (full / hard-lexical) by update:

     | Update | Full | Hard-lexical |
     |---|---|---|
     | 250 | 0.6226 | 0.6014 |
     | 500 | 0.6552 | 0.6380 |
     | 750 | 0.6809 | 0.6638 |
     | 1000 | 0.6932 | 0.6764 |
     | 1250 | 0.7023 | 0.6863 |
     | 1500 | 0.7052 | 0.6891 |
     | 1750 | 0.7084 | 0.6928 |
     | 2000 | 0.7102 | 0.6945 |

   - Selected checkpoint: update 2000 (guardrail passed). For comparison, pretrained E5 scores 0.5655 / 0.5391 and lexical 0.6042 / 0.5769.
   - These are selection-exposed dev tag-proxy numbers, not a promotion result.
   - The curve is still rising at the last update. The 2000-update budget is fixed, so this is noted, not acted on.
3. initial-2 (E5, lr 1e-5, scale 20) crashed at update 1743 (10:21 UTC):
   - Cause: a native XPU runtime abort (`AssertHandler::printMessage`, no Python traceback). Memory was steady at 8.5 GB, so this is an infrastructure crash, not a recipe failure.
   - Dev evals up to the last checkpoint: 0.6041/0.5806 (update 250) rising to 0.6739/0.6563 (update 1500).
   - Crash output kept as `runs/initial-2-e5-lr1e-05-s20.crash-1.out`; a `CRASHED` event is logged in `runs/train.jsonl`.
   - Code identity unchanged (46064545ace8be82). I restarted the chain for slots 2-6, and initial-2 resumed from the update-1500 checkpoint at 10:24 UTC (resume is bitwise deterministic per smoke 2).
   - Same run_id, so this is not an extra run.
   - Finished at 10:52 UTC. Selected update 2000: full 0.6806, hard-lexical 0.6634, guardrail passed.
   - Under , initial-1 advances over initial-2 (full +0.030, hard-lexical +0.031), so initial-1 is the incumbent until initial-3 is in.
4. initial-3 (E5, lr 3e-5, scale 50) hit the same native XPU abort at about update 500 (11:20 UTC):
   - Its update-250 dev eval was 0.6203/0.5990.
   - Crash output kept as `runs/initial-3-e5-lr3e-05-s50.crash-1.out`; I resumed from the update-250 checkpoint.
   - Both crashes struck just before an eval/checkpoint step. The metric lines (last at 1743 and 493) are buffered, so the true crash point is uncertain within about 20 updates.
   - The crash is not deterministic: the resumed initial-2 passed update 1750 without trouble.
   - No Windows System or Application event was logged. 14 GB of RAM was free.
   - The chain script now auto-resumes after this specific abort (output ends in `AssertHandler::printMessage`), at most 3 times per run. It keeps each crash output, logs a `CRASHED` event and still stops on any other failure.
   - train.py is not changed while runs are resumable, since the code identity is part of the checkpoint.
5. Paused at the user's request (about 12:09 UTC). I killed the chain script, then initial-3's process tree at about update 1100. Its last checkpoint is update 1000, so resuming redoes about 100 updates. Slots 4-6 have not started.
   - Resumed at the user's request ("continue") at 14:13 UTC with the same chain command. The trainer reloaded `ckpt-0001000.pt` (code identity unchanged, 46064545ace8be82) and logged its own `RESUMED` event at 14:14 UTC. My manual `RESUMED` line in `runs/train.jsonl` at 14:13 is nested under an extra `event` key (wrong call shape); left in place, not rewritten.
   - initial-3 (E5, 3e-5, scale 50) finished at 15:05 UTC: selected update 2000, full 0.7109, hard-lexical 0.6952, guardrail passed. Curve still rising at 2000, like initial-1.
   - Best E5 recipe: initial-1. initial-3 is +0.0007 on full, below the 0.005 advancement threshold, so the lower-scale recipe stays. Wrote configs initial-7/8/9 from initial-1 with one field changed each, checked schedule hashes on CPU (7 and 9 match initial-1; 8 differs by design), and queued them with `.scratch/queue_after.sh` to start when initial-6 logs DONE (abandons on any FAIL).
   - initial-4 (BGE-small, 3e-5, scale 20; about 1.14 s/update, 4.4 GB peak) hit the same native XPU abort at about update 1495 (15:35 UTC), seconds before its update-1500 eval. Auto-resume 1/3 restarted it from update 1250. All three crashes so far (initial-2 ~1743, initial-3 ~493, initial-4 ~1495) fall at the train-to-eval transition, never mid-stretch. That points at the switch into evaluation (catalogue encoding) rather than the training step, but the cause is unconfirmed. Each crash costs up to one eval interval (250 updates). The code stays frozen while runs are resumable; if one run crashes a 4th time the chain stops and the queued slots 7-9 are abandoned.
   - initial-4 finished at 15:50 UTC: selected update 2000, full 0.6675, hard-lexical 0.6495 (pretrained BGE 0.5403 / 0.5133), guardrail passed. Schedule hash 5839ec9d8f0b05b3 equals initial-1's, so E5 and BGE saw identical batches. It trails E5 initial-1 (0.7102 / 0.6945) on the same recipe; dev tag-proxy, selection-exposed. initial-5 started.
   - initial-5 (BGE, 1e-5, scale 20) finished at 16:35 UTC with no crash: selected update 2000, full 0.6301, hard-lexical 0.6100, guardrail passed. Same schedule hash as initial-1/4. As with E5, the lower learning rate is clearly worse at 2000 updates (initial-4: 0.6675 / 0.6495). initial-6 started.
   - initial-6 hit the native XPU abort at 16:57 UTC; auto-resume 1/3 restarted it from update 1000. Timing puts the crash about 200 updates after the update-1000 eval (EVAL logged 16:53:45; one 250-update interval took about 274 s), so likely near update 1200, not at the 1250 eval boundary. This weakens the train-to-eval-transition guess from the first three crashes; the cause remains unknown. Four crashes in about 9 GPU-hours so far, each at most one interval lost.
   - initial-6 (BGE, 3e-5, scale 50) finished at 17:18 UTC: selected update 2000, full 0.6668, hard-lexical 0.6494, guardrail passed. Practically tied with initial-4 (scale 20: 0.6675 / 0.6495), as with E5. Scale 50 leaves the space much less spread out (mean random-pair cosine 0.68 vs 0.22 at scale 20; E5: 0.56 vs 0.08), without a retrieval cost on this dev proxy. BGE slots 4-6 are complete; applying the same ordering-and-advancement rule as , the best BGE recipe is initial-4 (initial-5 incumbent, initial-4 advances +0.037, initial-6 does not), a candidate for the trained-BGE comparator in P7 (to be fixed at the P6 freeze).
   - The queue started initial-7 (specificity weighting) at 17:18:39 UTC; initial-8 and initial-9 follow.
   - initial-7 (initial-1 + specificity pair weights) finished at 19:11 UTC with no crash: selected update 2000, full 0.7042, hard-lexical 0.6880, guardrail passed, same schedule hash as initial-1. Against initial-1 (same batches, same seed): -0.0060 full, -0.0065 hard-lexical, so it does not advance. On this dev proxy, single seed, specificity weighting did not help. initial-8 (band [31,100]) started.
   - initial-8 hit the native XPU abort at 19:39 UTC, about 11.5 min after its update-250 eval (one interval takes about 15 min), so roughly update 440-450, mid-stretch. Auto-resume 1/3 restarted it from update 250. Five crashes so far; two of them (initial-6, initial-8) were not near an eval, so the eval-transition guess no longer holds.
   - initial-8 (initial-1 + band [31,100]) finished at 21:09 UTC: selected update 2000, full 0.7004, hard-lexical 0.6836, guardrail passed. Against initial-1: -0.0097 full, -0.0109 hard-lexical; does not advance. Easier negatives (lexical ranks 31-100) did worse than the registered band [1,30] on this dev proxy, single seed. Its batches differ from initial-1's by design (schedule a6497e72d42e876c). initial-9 (multi-positive) started.
   - initial-9 (initial-1 + multi-positive objective) finished at 22:51 UTC with no crash: selected update 2000, full 0.7056, hard-lexical 0.6899, guardrail passed, same schedule as initial-1. Against initial-1: -0.0046 / -0.0046; does not advance. Its random-pair cosine is the lowest of all runs (0.043).

### P6 summary and freeze (2026-10-17)

All nine registered initial runs finished (slot 10 skipped as a proven no-op). Dev tag-proxy nDCG@10 on each run's guarded-best checkpoint; single seed 0, selection-exposed, not promotion evidence. Hours are wall-clock from STARTED to COMPLETED (initial-3 includes a 2 h user pause); crashes are native XPU aborts, each auto-resumed from the last checkpoint.

| Slot | Backbone | lr | scale | weighting | band | objective | sel. update | full | hard-lex | guardrail | rand. cos | hours | peak GB | crashes | skipped rows | schedule |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| initial-1 | e5-base-v2 | 3e-05 | 20 | none | [1, 30] | masked_infonce | 2000 | 0.7102 | 0.6945 | True | 0.080 | 1.91 | 8.5 | 0 | 0 | 5839ec9d |
| initial-2 | e5-base-v2 | 1e-05 | 20 | none | [1, 30] | masked_infonce | 2000 | 0.6806 | 0.6634 | True | 0.156 | 2.20 | 8.5 | 1 | 0 | 5839ec9d |
| initial-3 | e5-base-v2 | 3e-05 | 50 | none | [1, 30] | masked_infonce | 2000 | 0.7109 | 0.6952 | True | 0.561 | 4.20 | 8.5 | 1 | 0 | 5839ec9d |
| initial-4 | bge-small-en-v1.5 | 3e-05 | 20 | none | [1, 30] | masked_infonce | 2000 | 0.6675 | 0.6495 | True | 0.221 | 0.74 | 4.4 | 1 | 0 | 5839ec9d |
| initial-5 | bge-small-en-v1.5 | 1e-05 | 20 | none | [1, 30] | masked_infonce | 2000 | 0.6301 | 0.6100 | True | 0.297 | 0.74 | 4.4 | 0 | 0 | 5839ec9d |
| initial-6 | bge-small-en-v1.5 | 3e-05 | 50 | none | [1, 30] | masked_infonce | 2000 | 0.6668 | 0.6494 | True | 0.680 | 0.71 | 4.4 | 1 | 0 | 5839ec9d |
| initial-7 | e5-base-v2 | 3e-05 | 20 | specificity | [1, 30] | masked_infonce | 2000 | 0.7042 | 0.6880 | True | 0.083 | 1.88 | 8.5 | 0 | 0 | 5839ec9d |
| initial-8 | e5-base-v2 | 3e-05 | 20 | none | [31, 100] | masked_infonce | 2000 | 0.7004 | 0.6836 | True | 0.064 | 1.95 | 8.5 | 1 | 0 | a6497e72 |
| initial-9 | e5-base-v2 | 3e-05 | 20 | none | [1, 30] | multi_positive | 2000 | 0.7056 | 0.6899 | True | 0.043 | 1.67 | 8.5 | 0 | 0 | 5839ec9d |

Baselines for reference (same benchmark): lexical 0.6042 / 0.5769, pretrained E5 0.5655 / 0.5391, pretrained BGE 0.5403 / 0.5133.

What the pilots show, within their limits:
1. Fine-tuning helps a lot on this proxy: every guarded run beats its pretrained backbone by 0.09-0.15 full nDCG@10, and every E5 run beats the lexical baseline.
2. lr 3e-5 beats 1e-5 for both backbones at 2000 updates; scale 50 vs 20 is a practical tie for both, though scale 50 leaves the space far more concentrated.
3. None of the three interventions (specificity weighting, easier negatives, multi-positive) beat the plain recipe on matched batches; all three were slightly worse.
4. E5 leads BGE-small by about 0.04 on identical batches.
5. Every curve was still rising at update 2000.

Freeze: no extension runs; one E5 candidate (initial-1 recipe); BGE control = initial-4 recipe; 2000 updates, final checkpoint; seeds 1-3 (seed = data_seed), matched per seed; deployment seed 1; M = 4 registered contrasts. Six confirmation runs, interleaved E5/BGE by seed.

Confirmation started 22:54 UTC. confirm-e5-s1 hit the native XPU abort at 00:08 UTC on 18 Oct, 12 min after its update-1250 eval (one interval about 12 min, so at or just before the 1500 eval). Auto-resume 1/3 continued it from update 1250; an exact continuation stays in the same slot, and XPU resume is bitwise deterministic (smoke 2), so the final model is the one an uninterrupted run would give.

While confirmation trains, I wrote the P7 analysis step (893098c): `report.confirmation_report` and `mtgemb confirm-report`. It checks each confirmation run finished with selection 'final' and that its eval was made from `runs/<run>/final`, runs the four registered contrasts (candidate vs pretrained E5 and vs matched-seed trained BGE, full and hard-lexical, M = 4, 8000 resamples, quantile 0.0125), applies the promotion rule against both baselines, reports the seed-1 deployment scores separately, and writes `runs/confirmation/<benchmark>/report.json` once. Two new tests (promote; hard-lexical loss → not promoted; wrong weights rejected; write-once). Sent to Sol for independent review.
   - Sol review, three rounds. Round 1 (changes needed): provenance only checked model_path; no inconclusive outcome or adequacy threshold; missing seed ranges and exclusions; write-once race. Fixed in 9f3acfb; the adequacy rule and decision vocabulary are registered as before any confirmation eval. Round 2 (changes needed): manifest identity (dataset, pairs, model revision, trainer code, benchmark) not compared; an empty cohort would crash the bootstrap before the inconclusive check. Fixed in b8381a0. Round 3: Sol accepts, no remaining findings. Not yet run end to end on real confirmation evals (they don't exist yet).

confirm-e5-s1 finished at 00:48 UTC (2000 updates, final checkpoint; in-training dev eval 0.7055 / 0.6904; schedule 228be492 as registered). This is the in-training eval only; the registered contrasts use the separate eval-encoder records made after all six runs. confirm-bge-s1 started.
confirm-bge-s1 finished at 01:32 UTC with no crash (in-training eval 0.6643 / 0.6464; same schedule 228be492 as confirm-e5-s1, so seed 1 is matched). confirm-e5-s2 started.
confirm-e5-s2 finished at 03:13 UTC with no crash (in-training eval 0.7066 / 0.6904; schedule 86fd6214 as registered). confirm-bge-s2 started.
6. Monitoring: `tail -F` did not deliver events in Git Bash, so I replaced it with a 30 s polling monitor. Also, pytest had been writing test events into `runs/train.jsonl`; fixed and documented in .

## P9 (early): static app (2026-10-17)

1. Luna (GPT-6 Luna) built the Svelte/Vite app in `web/` against `web/SCHEMA.md` (commit 291b23d):
   - A deterministic 600-card synthetic fixture with invented names.
   - `check-bundle.mjs`, which verifies sha256 and byte counts.
   - 5 vitest tests.
   - The JS bundle is 22 kB gzipped.
2. My checks:
   - Re-ran fixture, test and build; all pass. The JS validator also accepts the **real** Python-exported dry-run bundle (27,866 cards), so the two independent schema implementations agree.
   - Served the build under a sub-path (`/sub/`) with the real bundle. Deep link, search, keyboard selection, neighbour click, filter URL state, the unknown-ID message and the 375px layout (no overflow) all work.
   - Found and fixed a history bug: selecting a card overwrote the previous history entry, so Back did nothing. Back and forward now restore the right card.

## Review: exporter, app and statistics (Sol round 1, 2026-10-17)

Sol (GPT-6.1 Sol) reviewed the exporter, app and report code and found six P2 issues. I accepted all six.

1. App: a stale popstate selection, and no check of ID order, the card-to-index mapping or the `current.json` pointer. Luna fixed both (429bdab). I re-ran fixture, tests (7), build and `check-bundle` on the fixture and the real dry-run bundle; all pass.
2. Exporter (ff5cc27):
   - The release ID now hashes the dataset id, the catalogue ID order, the encoder recipe with max_length, the model hash and the vector hash.
   - Publication renames the bundle first and the artifacts last. A retry after a crash between the two adopts the partial artifacts only if they name the published `release.json`.
   - New query-side parity check: the top-10 of CPU-encoded probe cards against the device catalogue must overlap the device top-10 by at least 0.98 on average.
3. Statistics (ff5cc27): `report.contrast` now pairs candidate and comparator runs by seed id, not list order.
4. Re-review rounds:
   - Round 2: 1–4 and 6 resolved. #5 needed a tie-aware overlap (near-ties across rank 10/11) and k capped for small catalogues (1a2c6dc).
   - Round 3: one more edge case, a one-card catalogue (d1185ac).
   - Round 4: Sol accepts; all six resolved.
