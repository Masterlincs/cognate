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
confirm-bge-s2 hit the native XPU abort at 03:41 UTC, about 4 min after its update-1250 eval (about 200 updates in, mid-stretch); auto-resume 1/3 continued it from update 1250 (exact continuation, same slot).
confirm-bge-s2 finished at 03:56 UTC (in-training eval 0.6629 / 0.6451; schedule 86fd6214, matched with confirm-e5-s2). confirm-e5-s3 started.
confirm-e5-s3 hit the native XPU abort at 04:23 UTC, about 12.5 min after its update-250 eval (at or just before the 500 eval); auto-resume 1/3 continued it from update 250.
confirm-e5-s3 finished at 05:54 UTC (in-training eval 0.7071 / 0.6919; schedule b5d67d44 as registered). confirm-bge-s3 started, the last confirmation run.
confirm-bge-s3 hit the native XPU abort at 06:17 UTC, about 4 min after its update-1000 eval (mid-stretch); auto-resume 1/3 continued it from update 1000. Ten crashes in total so far, all recovered.
confirm-bge-s3 finished at 06:36 UTC (in-training eval 0.6632 / 0.6458; schedule b5d67d44, matched with confirm-e5-s3). All six confirmation runs are complete (6 of the 9-run cap). Next: `mtgemb eval-encoder` on each `runs/confirm-*/final` (the registered eval records), then `mtgemb confirm-report`.

### P7 result: provisional tag-proxy promotion (2026-10-18)

Registered evals (`mtgemb eval-encoder` on each `runs/confirm-*/final`, 06:37-06:47 UTC). The confirm-e5-s2 eval hit the native XPU abort while loading (segfault, no output written); the retry succeeded. Every registered eval reproduced its run's in-training final eval exactly, a useful consistency check between the trainer's evaluator path and the standalone one.

| Seed | E5 full | E5 hard-lex | BGE full | BGE hard-lex |
|---|---|---|---|---|
| 1 | 0.7055 | 0.6904 | 0.6643 | 0.6464 |
| 2 | 0.7066 | 0.6904 | 0.6629 | 0.6451 |
| 3 | 0.7071 | 0.6919 | 0.6632 | 0.6458 |
| mean | 0.7064 | 0.6909 | 0.6635 | 0.6458 |

Registered contrasts (`mtgemb confirm-report`; M = 4; one-sided adjusted lower bound at quantile 0.0125, 8000 family-cluster resamples with matched seeds; 3016 anchors in 2853 families; no zero-positive exclusions):

| Candidate vs | View | Point | Adjusted lower | 95% interval | Per-seed differences |
|---|---|---|---|---|---|
| pretrained E5 | full | +0.1409 | +0.1320 | [0.1332, 0.1486] | 0.1400, 0.1411, 0.1416 |
| pretrained E5 | hard-lexical | +0.1518 | +0.1425 | [0.1436, 0.1598] | 0.1512, 0.1513, 0.1527 |
| trained BGE | full | +0.0429 | +0.0376 | [0.0383, 0.0474] | 0.0412, 0.0437, 0.0438 |
| trained BGE | hard-lexical | +0.0451 | +0.0398 | [0.0404, 0.0497] | 0.0440, 0.0453, 0.0460 |

Both comparisons pass every bound (full point ≥ 0.005 and lower > 0; hard-lexical point ≥ 0 and lower > -0.010). **Decision: promote, as a provisional tag-proxy promotion on selection-exposed development.** Deployment model: confirm-e5-s1 (seed 1, fixed before confirmation), full 0.7055, hard-lexical 0.6904. For context only (not a registered contrast): the lexical baseline scored 0.6042 / 0.5769.

What this does and does not show: on this development benchmark's tag-overlap relevance, fine-tuned E5 retrieves effect-similar cards much better than pretrained E5 and better than BGE-small trained the same way, and the result is stable across three seeds. It is not a sealed-data result, development was used to choose the recipe, the tag proxy is not human judgement, and three seeds say little about the tails of training variability. The encoder, export and app correctness gates still have to pass (P8, P9).

### P8: export of the promoted model (2026-10-18)

`uv run mtgemb export e5-base-v2 --max-length 94 --label "e5-base-v2 fine-tuned (confirm-e5-s1)" --status promoted --model-path runs/confirm-e5-s1/final --run-id confirm-e5-s1`:
- Release `2026-10-16-e5-base-v2-d4c393ee`: bundle in `web/public/data/` (39 MB, git-ignored), artifacts (vectors, ID order, parity, reference neighbours) in `artifacts/releases/`.
- 27,866 catalogue cards (fit and dev only; sealed cards cannot enter the catalogue), 27,573 with text and vectors, top-50 neighbours, self excluded.
- Encoder parity on 64 probe cards: CPU vs XPU minimum cosine 0.99999982 (threshold 0.9999); CPU-query top-10 overlap 1.0 mean and min (threshold 0.98, tie-aware).
- Exact-neighbour reference: 20 seeded rows against an independent float64 sort, 0 failures (17 identical lists; the rest are tie-aware near-tie swaps).
- release.json records label, status `promoted`, run `confirm-e5-s1`, model revision and the decision text "provisional tag-proxy promotion on selection-exposed development".
- `web/public/data/current.json` now points at this release, so the app's model choice is the promoted model. The app's own `check-bundle` validates the bundle (hashes, byte counts, ID order, card-to-index mapping, pointer); 7 app tests pass; build passes.
- Fixed a git-ignore slip: the root rule for release bundles was overridden by `web/.gitignore`'s `!public/data/**`; the rule now lives in `web/.gitignore` after the re-include. Nothing large was committed.
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

## P9: browser checks on the promoted release (2026-10-18)

Served `web/dist`, built with `current.json` → `2026-10-16-e5-base-v2-d4c393ee` (schema v1), via `vite preview` on port 4173. I drove it with the T3 preview browser.

1. **Search → details → neighbour → share link.** Typing "Lightning Bolt" offers the card plus the face alias "Emeritus of Conflict // Lightning Bolt". Enter opens the card: image, cost, type, text, colour identity, MV and Scryfall link. It shows 50 neighbours (Lightning Strike 0.987, Volcanic Hammer, Precision Bolt, Searing Spear…), the badge "promoted", and the footer "e5-base-v2 fine-tuned (confirm-e5-s1)". The URL becomes `?card=<oracle id>`.
2. **Reference neighbours.** For the 20 seeded cards in `release_meta.json` `reference_neighbours` (exact float64 top-50), I loaded each card's share link and read the rendered list:
   - 17/20 lists are identical;
   - 3/20 hold the same 50 cards with near-tied neighbours swapped;
   - top-10 overlap is 10/10 for all 20.

   This matches the export's tie-aware reference check (17 exact, 0 failures).
3. **Back/forward** restore the right card. **Unknown ID** (all-zero UUID, and a garbage string with `<script>`) shows "Card not found". **Rapid selections:** 20 selections 3 ms apart; the last one wins and its list matches the reference.
4. **Every included face layout** renders, 14 layouts, one card each: normal, class, saga, meld, transform, modal_dfc, split, adventure, mutate, prepare, leveler, case, flip, prototype. Multi-face layouts show one heading per face.
5. **375px width:** no horizontal overflow (document width 360). **Missing image** → "Card image unavailable" placeholder. **Disabled WebGL:** not applicable; the app draws no canvas and uses no WebGL. **Malformed bundle** (truncated `release.json`) → "Bundle could not be loaded: Unexpected end of JSON input", no fixture fallback. **Non-root base path:** served `web/` with `python -m http.server`; the app works from `/dist/` (all requests under `/dist/`, home link `/dist/`).
6. **Keyboard only:** Tab reaches search; typing, ArrowDown and Enter selects a card; Tab reaches the filters, then the neighbour list; Enter on a neighbour selects it. Minor: neighbour focus is only a background tint (`#edeee7`), not an outline.
7. **Performance** (plan targets: core shell + name index ≤5 MB compressed; usable lookup ≤3 s on a recorded throttled mobile profile; warm selection ≤300 ms on the test desktop).
   - Payload, gzip -9: JS 22.5 KB, CSS 2.6 KB, HTML 0.3 KB, `release.json` 11.0 KB, `names.json` 835.8 KB, about 872 KB in total. **Pass.**
   - Warm selection (in-app timing, test laptop): 25–33 ms with shards cached, 137–174 ms when shards had to be fetched. **Pass.** A selection fetches about 40 card+neighbour shard pairs, because neighbour filters need each neighbour's card record.
   - Usable lookup measured with `.scratch/throttle.mjs`: headless Chrome over CDP, 375×812 mobile, cache disabled, gzip served by `vite preview`, "usable" = typing returns suggestions.
     - Lighthouse mobile profile (150 ms RTT, 1.6 Mbps down, 675 kbps up, 4× CPU): **5.34 / 5.33 / 5.33 s. Fail** (target 3 s).
     - Unthrottled: 0.27 / 0.26 s.
     - Startup transfer 851 KB, almost all `names.json`: ids 584 KB gzip, names 234 KB, aliases 17 KB.
   - Fix: bundle schema v2 moves the ids to `ids.json`, fetched in parallel; lookup needs only the names. Target and profile unchanged. Luna took the web side; I changed the exporter (c0423b0; the schema string now enters the release digest, so v2 gets a new release_id and v1 stays as it is).
8. **Schema v2 implementation.**
   - Luna (GPT-6 Luna) changed the app, validators, fixture and SCHEMA.md (3d9d488). I read the diff and re-ran its tests: 8 vitest tests pass, build OK.
   - First XPU export attempt: native crash (`AssertHandler::printMessage`, exit 139) while encoding. Nothing was published; log kept as `runs/export-v2.crash-1.out`.
   - The retry succeeded: release **`2026-10-16-e5-base-v2-fc4d438d`**. Vectors sha256, model hash, dataset id, card count and the 20 reference lists are identical to v1; only the layout differs. Parity matches v1: min cosine 0.99999982, CPU-query top-10 overlap 1.0, reference 20/20 (17 exact). `check-bundle` validates it. `current.json` now points at it (cfa4891).
   - The v1 release `d4c393ee` stays on disk unchanged; the v2 app rejects it by schema.
9. **Re-measurement** (same script and profile).
   - With ids fetched in parallel: 3.96 / 3.81 / 3.87 s, still failing. The 584 KB ids download shared the 1.6 Mbps link with `names.json`.
   - Changed the app to fetch `ids.json` after `names.json` (cfa4891).
   - Then the runs split into 2.5 s and 3.8 s groups. Resource timings showed `names.json` itself taking 2.6 s instead of 1.4 s in the slow runs: the previous run's `ids.json` download was still draining through the throttled link. This is a harness artefact; a real cold visit has no earlier page.
   - With a 6 s gap between runs (`P9_GAP_MS=6000`): **2.52 / 2.46 / 2.52 / 2.46 / 2.48 / 2.48 / 2.49 / 2.47 s over 8 runs. Pass** (≤3 s). Startup transfer is 288 KB (`names.json` 250 KB). Unthrottled: 0.15–0.19 s.
   - The v1 numbers (5.33 s) are unaffected by that artefact: v1 had no download left in flight after lookup became usable.
   - Information only (no target): a cold share link (`?card=`) renders its card in 6.10–6.14 s on the same profile, since it needs ids plus about 40 shard pairs. Transfer 954 KB.
10. **v2 browser re-check:**
    - 20 reference cards: 17 exact, all sets 50/50, all top-10 10/10.
    - Back and forward, rapid selections (last wins), unknown ID, cold share link (Ulvenwald Captive, 50 neighbours), and search with alias and IDs then Enter all pass.
    - The footer shows confirm-e5-s1.
11. Sent the v2 change (c0423b0, 3d9d488, cfa4891) to Sol for independent review.
12. **Sol review of schema v2, round 1:** accept after fixes. I accepted all three findings and fixed them myself (f722310):
    - P1: a `null` `names.json` passed validation and check-bundle.
    - P2: IDs arriving after the user had dismissed or chosen a suggestion reopened the dropdown. Reproduced with headless Chrome on the throttled profile: Escape right after lookup, wait 6 s for ids, 0 options before and after (2 runs); lookup still 2.5 s.
    - P3: SCHEMA.md and the wording still said the fetch was parallel. is amended with a dated note.

    9 vitest tests pass, the build passes, and both the fixture and the real v2 bundle validate. Round 2 has been sent to Sol.
13. **Sol review of schema v2, round 2:** no P1/P2/P3 findings. Sol verified all three round-1 fixes against f722310: 9 web tests, the delayed-ID checks, release-only validation, and SCHEMA.md/matching the sequential fetch order. Accepted. (This step was missing from the log; added 2026-10-18 from the review record.)

## P10: reproduction and handoff (2026-10-18)

1. **Evaluation replay from pinned embeddings.** I added `mtgemb replay-eval` (7554041, with 3 tests: untouched records pass, tampered vectors fail, a release whose vectors are not the evaluated model fails). For every recorded eval on benchmark `8bf2f48c0bba3b2c` it:
   - verifies the recorded file hashes;
   - re-scores the saved `vectors.npy` (lexical: refit on fit text);
   - compares every per-anchor metric, the NaN pattern and the integer counts.

   For each exported release, it checks the vectors sha256 and compares the vectors with the evaluated run's vectors.

   Result: all 9 evals (6 confirmation, 3 baselines) replay with max abs per-anchor difference **0.0** and no hash mismatches. Both promoted releases (`d4c393ee`, v1; `fc4d438d`, v2) hold vectors **bitwise identical** to the evaluated `confirm-e5-s1` vectors, so the released model is the evaluated model. First report `runs/replay/8bf2f48c0bba3b2c/2026-10-18T072329Z.json` was made from uncommitted code. The record of reference is `2026-10-18T073201Z-1792308721253768800.json`: commit 101edee, clean tree, passed, max diff 0.0, both promoted releases 0.0. An intermediate run at de96fb2 was flagged dirty because of uncommitted doc edits; it also passed.
2. **Run reconciliation.** 18 run directories: 3 smoke, 9 initial, 6 confirmation. Every one has STARTED and COMPLETED events in `runs/train.jsonl`, `finished: true`, a selected update and a passed guardrail.
   - Crashes, each followed by RESUMED: initial-2, 3, 4, 6, 8; confirm-e5-s1, e5-s3, bge-s2, bge-s3.
   - initial-3 also has one PAUSED/RESUMED pair.
   - Run counts against the caps: initial 9/10, extension 0/3, confirmation 6/9.
   - Export crashes (no partial output published): v2 attempt 1. The earlier `9a16efe0` is the P8 dry-run export of the pretrained model (`run_id` null); `9a16efe0.failed-1` holds the empty partial directories from the first dry run, which crashed on an inverted index map (P8 step); kept, not deleted.
3. **Clean environment.** Fresh `git clone` of origin at 7554041 into a temp dir:
   - `uv sync` 19 s, with a warm uv cache, so not a cold network install;
   - `ruff check` clean; `pytest -q` 73 passed, 4 real-model tests deselected; the tiny resume test `test_resume_matches_uninterrupted_and_rejects_corruption` passes;
   - `npm ci`, 9 vitest tests and the build pass.

   A fresh clone's `current.json` names the real release, which is git-ignored, so the app needs `npm run fixture` or an export first (documented in `web/README.md`).
4. **No full retraining replay** has been performed; the evaluation replay is not equivalent.
5. README status, results tables and limitations updated. All numbers were re-read from the eval summaries and the confirmation report.
6. **Sol review of `replay-eval`, round 1:** accept after fixes. Six P2 findings, each a way `passed` could be wrongly true:
   - releases without a comparable eval passed;
   - missing eval records were skipped silently;
   - benchmark and dataset identities were not checked;
   - only the release vectors were hashed, not the bundle;
   - recorded summaries were not compared;
   - integer-typed metric columns were skipped.

   I accepted all six and fixed them (f039ff8):
   - Releases are matched to evals by recipe, max_length and weights.
   - Every eval directory must be a complete record, and zero evals fails.
   - Identities are checked.
   - `release.json` and every bundle file are verified against their recorded hashes and byte counts.
   - Whole summaries are compared.
   - Metric columns are required and compared as float.

   12 replay tests (10 tamperings, one per failure; all fail for the stated reason). Real run at f039ff8, clean tree: **passed**. 9 evals: per-anchor and summary diff 0.0. 3 releases, all 0.0: the pretrained dry run `9a16efe0` now matched to `pretrained-e5-base-v2`; `d4c393ee` and `fc4d438d` matched to `confirm-e5-s1`. Report `runs/replay/8bf2f48c0bba3b2c/2026-10-18T073902Z-1792309142065552000.json`; this replaces the earlier report as the record of reference. Round 2 has been sent to Sol.
7. **Sol review of `replay-eval`, round 2:** FAIL, three findings:
   - P2: release vectors containing NaN passed, because `NaN > tol` is false;
   - P2: two releases with the same `release_id` overwrote each other in the report, so a failing one could be hidden;
   - P3: the boolean `True` matched the count `1` in the summary comparison.

   I accepted all three and fixed them (6314a29):
   - Comparisons are written as `not diff <= tol`, so NaN fails.
   - Release vectors must be all finite.
   - Releases are keyed by directory name, and `release_id` must equal that name.
   - Booleans match only booleans.

   Three new tampering tests (15 replay tests in all). Each fails on the r1 code and passes after the fix. Real run at 6314a29: **passed**. Same outcome as r1: 9 evals and 3 releases, all diffs 0.0. Report path is in step 8. Round 3 sent to Sol.
8. **Replay record of reference:** a re-run at clean commit 7d921d7 (`uv run mtgemb replay-eval --ds data/processed/2026-10-16_no_reminder_v1`) **passed**. 9 evals with per-anchor and summary diffs of 0.0. 3 releases, each matching its eval with diff 0.0. Report: `runs/replay/8bf2f48c0bba3b2c/2026-10-18T075120Z-1792309880448685600.json`. It replaces the f039ff8 report as the record of reference. One intermediate report (`…074757Z…`) is marked dirty because `WRITE_UP.MD` was uncommitted at the time; it is kept but is not the record.
9. **Sol review of `replay-eval`, round 3:** FAIL, two findings:
   - P2: a missing `per_anchor.parquet` or missing release `vectors.npy` crashed the replay with `FileNotFoundError`, so no report was written;
   - P3: boolean per-anchor count columns were cast to float and passed.

   I accepted both and fixed them (c3e5e20):
   - Missing files are reported as problems instead of crashing.
   - Metric and count columns must be numeric and not boolean.

   Three new tests (18 replay tests, 88 in total). Each fails on the r2 code. Real run at clean c3e5e20: **passed**. 9 evals and 3 releases, all diffs 0.0. Report `runs/replay/8bf2f48c0bba3b2c/2026-10-18T075854Z-1792310334571453600.json`; this is now the record of reference. Round 4 sent to Sol.
10. **Sol review of `replay-eval`, round 4:** FAIL, two findings:
   - P2: truncated npy or parquet files, or non-finite eval vectors, still crashed the run without a report;
   - P3: subtracting int8 release vectors overflowed, so a tampered vector could show a diff of 0.0 and pass.

   I accepted both and fixed them at the root rather than case by case (03c9d08):
   - Each eval and release check now runs inside a guard that turns any exception into a failed record, so a report is always written.
   - Eval and release vectors must be finite 2-D float arrays, loaded without pickle.
   - The release-versus-eval diff is computed in float64.

   Four new tests (22 replay tests, 92 in total). Each fails on the r3 code. Real run at clean 03c9d08: **passed**. 9 evals and 3 releases, all diffs 0.0. Report `runs/replay/8bf2f48c0bba3b2c/2026-10-18T080600Z-1792310760233301800.json`; this is now the record of reference. Round 5 sent to Sol.

   Note: `runs/replay-eval.out` is stdout from the 07:39 run at f039ff8. That run passed, but it predates the r2 to r4 fixes and is not the record.
11. **Sol review of `replay-eval`, round 5:** FAIL, one P2. Reading an eval's metadata still happened outside the guard, so a missing `meta` or a list-valued `model_path` crashed the run without a report. I accepted it and fixed it (d9df157): the replay of one eval and its metadata read now run together inside the guard. Two new tests (24 replay tests, 94 in total); both fail on the r4 code. Real run at clean d9df157: **passed**. 9 evals and 3 releases, all diffs 0.0. Report `runs/replay/8bf2f48c0bba3b2c/2026-10-18T081202Z-1792311122382392400.json`; this is now the record of reference. Round 6 sent to Sol.
12. **Sol review of `replay-eval`, round 6:** FAIL, one P2. Releases were found by globbing for `release_meta.json`, so a release whose meta file was unreadable or missing silently vanished. A failing release could therefore disappear and leave `passed` true. I accepted it and fixed it (89aae1a):
   - Every directory under `artifacts/releases/` is now a release record, checked inside the guard; a missing or unreadable meta file is a failure.
   - Unpublished exports are listed in the report as `unpublished_release_dirs` and not checked. These are `.partial` directories (export publishes only by renaming them) and `.failed-N` directories (failed exports kept aside, never deleted). The real tree has one: the empty `9a16efe0.failed-1`.

   New and extended tests; both fail on the r5 code (25 replay tests, 95 in total). Real run at clean 89aae1a: **passed**. 9 evals and 3 releases, all diffs 0.0. Report `runs/replay/8bf2f48c0bba3b2c/2026-10-18T082018Z-1792311618169986800.json`; this is now the record of reference. Round 7 sent to Sol.
13. **Sol review of `replay-eval`, round 7:** FAIL, two P2 findings:
   - An `is_dir()` error on a child entry escaped the guard.
   - A bundle with `names.json` removed from its manifest passed once `release.json` was re-pinned in the release meta.

   I accepted both (5e7ec19):
   - Every entry under the eval and release roots is now a record checked inside the guard; a stray file fails.
   - The second finding needs coordinated edits to two hash-pinned files, which I had scoped out as forgery. Manifest completeness is still a real gap (an exporter bug could cause it), so `release.json` must now:
     - have a known schema (v1: `names.json`; v2: `names.json` + `ids.json`);
     - have a `release_id` equal to the meta's;
     - have an `n_cards` equal to the catalogue size;
     - list every card and neighbour shard.

   **My mistake, caught by the real run:** the first version compared `n_cards` to the benchmark universe. The replay at 5e7ec19 therefore **failed** all three releases, with bundle `n_cards` 27866 against 27573. The export catalogue is every fit+dev card, including cards without usable text. Only the 27573 cards with text have vectors. I fixed the check to use the catalogue (3922bf7). The failed report `…082827Z…` is kept as a record of the mistake.

   Tests: 27 replay, 97 in total. Real run at clean 3922bf7: **passed**. 9 evals and 3 releases, all diffs 0.0. Report `runs/replay/8bf2f48c0bba3b2c/2026-10-18T083246Z-1792312366565236500.json`; this is now the record of reference. Round 8 has been sent to Sol, with deliberate re-pinning forgery explicitly out of scope: the replay is an integrity and reproducibility check, not a signature.
14. **Sol review of `replay-eval`, round 8:** FAIL, one P2. An eval record with empty or incomplete encoder metadata (`meta={}`, or no recipe or `max_length`) passed, even though nothing could be attributed to it. I accepted it and fixed it (5f208ec):
   - `meta.system` must be `encoder` or `lexical`.
   - `meta.recipe` must be a non-empty dict.
   - Encoders need an integer `max_length`.

   All 9 real records satisfy this. Two new tests (29 replay tests, 99 in total); both fail on the r7 code. Real run at clean 5f208ec: **passed**. 9 evals and 3 releases, all diffs 0.0. Report `runs/replay/8bf2f48c0bba3b2c/2026-10-18T084126Z-1792312886269924900.json`; this is now the record of reference. Round 9 sent to Sol.
15. **Sol review of `replay-eval`, round 9:** FAIL, one P2. When an eval's `meta.model_path` was absent, it was read as `None`, which means pretrained. A record with unknown weights could therefore match a pretrained release.

   I accepted it and fixed it (ebd7e19): encoder records must contain the `model_path` key, either as an explicit null (pretrained) or as a string. A list-valued `model_path` is now caught by this check too, instead of by the crash guard.

   One new test; 100 tests in total. Real run at clean ebd7e19: **passed**. 9 evals and 3 releases, all diffs 0.0. Report `runs/replay/8bf2f48c0bba3b2c/2026-10-18T085415Z-1792313655152230400.json`; this is now the record of reference.

   Round 10 sent to Sol. Sol is asked to sweep every metadata field the replay reads, for absent or wrongly typed values, and to report all remaining findings together rather than one per round.
16. **Sol review of `replay-eval`, round 10** (a full metadata sweep, as asked): FAIL, four P2 groups. I accepted all four and fixed them (e596c52):
   - **Fake encoder recipes passed, and lexical-labelled records could match encoder releases.**
     - Encoder recipes must contain the six encoder-contract keys (`model_id`, `revision`, `prefix`, `pooling`, `normalize`, `dim`).
     - Only encoder records can be matched to releases.
     - Matching compares canonical JSON of (recipe, max_length, resolved path), so `True` never equals `1`.
   - **The lexical refit ignored the recorded recipe.** A lexical record's recipe must now equal `LEXICAL_RECIPE`, which is what the refit uses.
   - **A falsy release `model_path` (`false`, `0`, `[]`) matched pretrained.** `model_path` must now be null or a non-empty string, on both evals and releases. Encoder `max_length` must be a positive int.
   - **`shard_size=-1` let shards go missing.** It must now be a positive int.

   Five new tests (105 in total); each fails on the r9 code. Real run at clean e596c52: **passed**. 9 evals and 3 releases, all diffs 0.0, with the same release-to-eval matches as before. Report `runs/replay/8bf2f48c0bba3b2c/2026-10-18T090640Z-1792314400337031000.json`; this is now the record of reference. Round 11 (another full sweep) sent to Sol.
17. **Sol review of `replay-eval`, round 11** (full sweep): FAIL, two P2 findings:
   - Encoder recipes had only their keys checked, so a null `model_id`, `pooling` or `dim` passed.
   - The lexical comparison accepted `lowercase=1` and `ngram_range=[3.0, 5.0]`.

   I accepted both and fixed them (9f90519):
   - An encoder record's recipe must now canonically equal one entry of `encoder.RECIPES`, the recipes the encoder module can actually run. All 8 real encoder records do.
   - The recipe's `dim` must equal the width of the saved vectors.
   - Lexical recipes are compared as canonical JSON, so `1` never equals `true`.

   Three new tests and one updated; 108 tests in total. Real run at clean 9f90519: **passed**. 9 evals and 3 releases, all diffs 0.0, with the same matches as before. Report `runs/replay/8bf2f48c0bba3b2c/2026-10-18T091622Z-1792314982014177000.json`; this is now the record of reference. Round 12 sent to Sol.
18. **Sol review of `replay-eval`, round 12** (full sweep): FAIL, exactly one P2, and Sol reported nothing else in scope. A release `encoder` stored as a list of pairs was silently turned into a dict by `dict(...)` and passed.

   I accepted it and fixed it (996ef83): the release `encoder` must be a JSON object, otherwise the release fails and matches nothing. One new test (109 in total), which fails on the r11 code. Real run at clean 996ef83: **passed**. 9 evals and 3 releases, all diffs 0.0, with the same matches as before. Report `runs/replay/8bf2f48c0bba3b2c/2026-10-18T092430Z-1792315470167881000.json`; this is now the record of reference.

   Round 13 sent to Sol, to verify this fix only.
19. **Sol review of `replay-eval`, round 13: ACCEPT.** No findings; the r12 fix holds and introduces no regression. That ends the review series:
   - 13 rounds;
   - 26 findings accepted and fixed, from f039ff8 to 996ef83;
   - 39 replay tests, 109 tests in total.

   The record of reference is the clean run at 996ef83: `runs/replay/8bf2f48c0bba3b2c/2026-10-18T092430Z-1792315470167881000.json`.
   - **passed:** all 9 eval records re-score with per-anchor and summary diffs of 0.0.
   - All 3 release records match their evaluations exactly: `9a16efe0` → `pretrained-e5-base-v2`; `d4c393ee` and `fc4d438d` → `confirm-e5-s1`.
   - One unpublished directory is listed: `9a16efe0.failed-1`.
20. **P10 complete.** What it established and what it did not:
   - **Established:**
     - The evaluation replay is exact.
     - All 18 runs are reconciled.
     - A clean clone builds and tests.
     - The README and handoff table are filled.
   - **Not done:**
     - No full retraining replay. The evaluation replay re-scores pinned embeddings; it does not retrain.
     - No sealed (held-out) evaluation.
     - No publication.

     The last two are gates that need the user's explicit approval.

## Sealed held-out evaluation (gate 1, 2026-10-18)

The user authorised gate 1 ("do 1"): one evaluation of the frozen artifacts on the sealed split. Publication (gate 2) is still not authorised.

1. **Protocol first.** Before any sealed row was parsed, decision was recorded in the decision log:
   - **Systems:** the 9 frozen, dev-evaluated systems, unchanged.
   - **Benchmark:** universe = all fit+dev+sealed cards with usable text; anchors = sealed cards with at least one fit-represented tag. Weights and views are the fit-derived ones, with `sealed_candidates` in place of `dev_candidates`.
   - **Weight identity:** each encoder's re-encoded vectors must reproduce its saved dev vectors.
   - **Statistics:** the same 4 registered contrasts.
   - **Use:** one run, no tuning, no per-card inspection. The P7 decision and the release stay as they are.
2. **Code** (commit `bbbdd8a`, not yet run):
   - `evaluate.py`:
     - `build_benchmark(..., held="sealed")` and `load_benchmark(ds, held)`. Only `held="sealed"` parses sealed rows.
     - The sealed spec lives in its own file, `benchmark_v3_sealed.json`.
     - Views now come from the spec.
   - `baselines.py`:
     - `same_weights` checks each encoder: row cosine ≥ 0.9999 against its saved dev vectors on the dev universe, checked before any sealed anchor is scored.
     - The claim scope now follows the benchmark (dev or sealed).
   - `report.confirmation_report(bench, train_bench=...)` re-tests the contrasts on sealed evals. It still checks provenance against the training runs and writes `runs/sealed/<id>/report.json` once, with `sealed_gate` cleared / not_cleared / inconclusive instead of a promotion decision.
   - `sealed.py` and the CLI command `mtgemb sealed-eval` drive the run.
   - Tests:
     - A pinned fixture dev id, taken from the pre-code.
     - The sealed universe and anchors.
     - `same_weights` refusing changed or edited vectors.
     - The sealed report: cleared, not_cleared, stale benchmark id, and writes once.
   - Verification: 112 tests pass.
3. **Dev unchanged.** The real dev benchmark rebuilt with the new code keeps id `8bf2f48c0bba3b2c`. `replay-eval` on clean `bbbdd8a` passed: 9 evals and 3 releases, tolerance 1e-9 (`runs/replay-eval-bbbdd8a.out`).
4. **Pre-run review.** Sent to Sol (`p10-sealed-review-r1`) for a code-only review. Sol is barred from touching sealed data.
5. **Sol r1: two P2 findings, both fixed in `14af854`.**
   - **Finding 1:** resume after a crash trusted unverified sealed eval records.
   - **Finding 2:** the one-shot check allowed two concurrent launches to both score.
   - **Fix:** resume is removed. The run first creates `runs/sealed/run.lock` exclusively and never removes it. Any second launch, or any existing sealed records, is refused. A crashed run therefore needs a human decision, not an automatic recovery.
   - **Tests:** `tests/test_sealed.py`. 114 tests pass. Sent for the r2 review.
6. **Sol r2: PASS**, no blocking findings in `bbbdd8a..14af854`. Sol's overlap and crash probes passed on temporary fixtures.
7. **Single run started** at `aebb999` (clean tree, pushed): `uv run mtgemb sealed-eval`, with output in `runs/sealed-eval.out` and errors in `runs/sealed-eval.err`. Before the run, there was no `runs/sealed/` directory and no sealed spec.
8. **Attempt 1 crashed** (2026-10-18 14:10:39Z, exit 139, segmentation fault). stderr shows the Intel XPU runtime's `AssertHandler::printMessage`, the same intermittent driver crash behind the 11 earlier `runs/*.crash-1.out` files.
   - **What it left:**
     - Sealed benchmark `c98bb619ea613672` (`benchmark_v3_sealed.json`).
     - The lexical sealed eval, complete.
     - `runs/sealed/run.lock`.
     - The STARTED, then CRASHED events in `runs/sealed.jsonl`.
   - **Not reached:** the failure came right after the pretrained-E5 weights loaded. No encoder was scored on sealed, and no report was written.
   - **Seen so far:** the lexical baseline's sealed aggregate appeared in `runs/eval.jsonl` (full 0.6101, hard-lexical 0.5831). No per-card output was inspected.
   - **Provenance note:** the lock records `dirty: true`. The only uncommitted change was WRITE_UP steps 6-7; the code was at `aebb999`.
   - **Next:** by design (and the r1 fix), the lock blocks any rerun. Nothing was deleted or retried, and the user has been asked how to proceed.
9. **User decision: fresh restart**.
   - Attempt 1's lock and lexical record were moved unchanged into `runs/sealed/attempt-1-crashed/`. Nothing was deleted.
   - Attempt 2 started at clean `b01ecc4` (14:26:41Z) and completed at 14:41:35Z without a crash, writing `runs/sealed-eval-2.out`.
10. **Checks on attempt 2:**
    - The lexical per-anchor frame (25,360 rows) is identical to attempt 1's.
    - Every encoder's re-encoded vectors match its frozen dev vectors on the dev universe: min row cosine ≥ 0.9999997, all on XPU.
11. **Sealed result** (`runs/sealed/c98bb619ea613672/report.json`, benchmark `c98bb619ea613672`):
    - **Benchmark:** 3170 sealed anchors in 2954 families, 0 zero-positive exclusions, universe 30,745.
    - **Fine-tuned E5, mean of 3 seeds:** full 0.7050, hard-lexical 0.6830.
    - **Released seed 1:** full 0.7052, hard-lexical 0.6833.
    - **Contrasts** (point, with the adjusted lower bound in brackets):

      | Comparator | Full | Hard-lexical |
      | --- | --- | --- |
      | Pretrained E5 | +0.1329 (+0.1239) | +0.1435 (+0.1326) |
      | Trained BGE | +0.0409 (+0.0360) | +0.0451 (+0.0386) |

    - **Gate:** both comparators pass the plan's §5 thresholds, so the sealed gate is **cleared**.
    - **Other systems:** lexical 0.6101 / 0.5831; pretrained BGE 0.5427 / 0.5082; trained BGE mean 0.6640 / 0.6378.
12. **Interpretation.**
    - The sealed numbers sit close to development: deployment model full 0.7055 → 0.7052, hard-lexical 0.6904 → 0.6833. Both contrasts keep a similar size. The frozen candidate also cleared the registered gate on sealed anchors; development results remain selection-exposed (Sol claims check: "not an artefact of selection" was an overclaim, corrected).
    - This is still tag-proxy evidence, not human relevance. It does not change .
    - No tuning or per-card inspection followed.
13. **Recorded:** the README sealed section, STATE `sealed` block, the IMPLEMENTATION_PLAN entry, and . Publication (gate 2) remains unapproved.
14. **Sol claims check** (`p10-sealed-claims-r1`): all numbers, seed means and "cleared" match the aggregates, and export is fit+dev only. One P2 overclaim was fixed (step 12 and the README wording). Sol did not verify the per-card equality of the archived attempt under its access restrictions; the lead checked it (step 10).
15. **Deferred:** the user saved longer E5 training for later (2026-10-18). Recorded in STATE `deferred` with its caveats. Not started.

## User review of the app: rider clauses dominate (2026-10-18)

The user looked at the running app (release `fc4d438d`) and reported two bad neighbour lists. Both are development or training cards, and both were inspected with `.scratch/inspect_cards.py`, which never loads sealed rows.

1. **Deflecting Swat** (dev). The text has two lines: "If you control a commander, cast free" and "choose new targets for a spell or ability".
   - **Fine-tuned top 10:** all commander or alternative-cost cards (Deadly Rollick, Obscuring Haze, the commander "Storm" cycle). There are no redirect cards.
   - **Pretrained E5:** includes Commandeer and Perplexing Chimera, which redirect.
   - **Lexical:** includes Redirect and Wild Ricochet.
   - **Cause: the proxy prefers the rider.** The metric scores Obscuring Haze (shares only the commander rider) at gain 16.62 and Redirect (shares the actual effect) at 10.56. The rider carries three co-occurring tags: synergy-commander, manaless value and alternative cost. The fine-tuned list scores *higher* on the proxy (nDCG 0.694 vs pretrained 0.684, lexical 0.688).
2. **Bruenor Battlehammer** (fit). The text has two lines: "+2/+0 per attached Equipment" and "pay {0} for the first equip each turn".
   - **Fine-tuned neighbours:** mostly equipment-count cards, five of which share no tag at all. Only one (Strong Back) shares the equip-cost tag.
   - **Lexical:** finds the free-equip cards, Forge Anew and Kili.
3. **Interpretation.** This is not overtraining in the usual sense; the development proxy was still rising at 2000 updates. Training pairs and the metric both treat every shared functional tag as equal, and summed. So on multi-ability cards a rider clause with several co-occurring tags can outrank the main effect, and fine-tuning learns exactly that. The proxy gains in the README are real gains *on the proxy*, but the proxy rewards this failure. This is a construct-validity limit, added to the README limitations. No claim, decision or artifact was changed.

## Per-ability neighbours and the human pilot (user request "do 1 and 2", 2026-10-18)

1. **Per-ability neighbours.**
   - **Contract (lead):** `src/mtgemb/abilities.py` defines an ability as a non-empty line of the model text, with the face boundary "//" excluded. Each ability is encoded alone. Candidate cards are scored by their best-matching line. The query's own card is excluded; ties break by Oracle ID. There is a float64 reference and hand-worked tests (`5458f43`).
   - **Probe with the released model:** Deflecting Swat's redirect line returns redirect cards (Redirect, Insidious Will, Bolt Bend, Redirect Lightning, Mizzium Meddler, Boltbender). Bruenor's free-equip line returns Forge Anew first, then generic "rather than pay" cost cards, so the fix is partial there.
   - **Size:** 52,699 ability lines; 17,209 cards have ≥2 abilities, giving 42,335 query lines.
   - **Delegated:** Sol (`p12-export-v3-impl-r1`) has the exporter, bundle schema v3 and replay checks. Luna (`p12-web-v3-impl-r1`) has the web UI ("Similar by" control), validation and fixtures.
2. **Human pilot, design and code by the lead.**
   - `src/mtgemb/human.py`, `146ab6b`, `4de313e`: the seeded, stratified, blinded pool builder (`pool.json` plus `key.json`) and the judged-pool analysis. The analysis reports mean grade, precision at grade ≥2 and nDCG@3 per system and stratum, a fine-tuned-minus-comparator nDCG bootstrap over anchors, repeat agreement and the random-candidate base rate.
   - Hand-worked tests pass.
   - The pool is built after the v3 export, because ability lists come from the release itself.
   - The annotation tool is delegated next.
3. **Delegated work landed and was checked.**
   - **Sol exporter, `7f54e57`:** I read the diff and re-ran the suite. Ability neighbours are written as `abilities/NNN.json` shards. A CPU-parity check and a float64 reference check run at export. The ability vectors and lines are pinned by hash in `release_meta.json`.
   - **Luna web UI, `ada1cf2`:** the vitest suite and the build pass, and `fixture-v3` validates.
   - **Sol annotation tool, `1451e6c`:** `mtgemb annotate <pool_dir>` serves `pool.json` on 127.0.0.1. It never reads `key.json`. Labels are validated server-side and appended to `labels.jsonl` in the format `human.last_labels` reads. I read the diff; its tests are in the suite.
4. **Luna's review of the exporter (`p12-export-v3-review-r1`), two P2 findings.**
   - **#1, tie order checked against the float64 reference: declined.** Ties are broken by stable argsort, which is unit-tested (`tests/test_abilities.py`). A float64 recomputation cannot arbitrate float32 ties. The reference check therefore compares scores tie-aware, which is the same rule as the card-level reference check accepted at P8.
   - **#2, replay did not verify the v3 ability artifacts: fixed by me, `f4ea848`.** For a v3 bundle, replay now requires `ability_vectors.npy` and `ability_lines.json` to be present and to match their hashes in `release_meta.json`. Three new tests cover a tampered file, a missing file and a missing hash. 168 tests pass.
5. **Release decision text (`a4930d5`).** The promoted release's `model.decision` read "provisional ... on selection-exposed development". That text predates the sealed run, so it now reads "tag-proxy promotion on selection-exposed development; gate also cleared on one sealed run". The claim is still a tag-proxy claim.
6. **Reviews launched:** Luna is reviewing the annotation tool and the replay fix (`p12-annotate-replay-review-luna-r1`). Sol is reviewing the web change (`p12-web-v3-review-sol-r1`).
7. **Real v3 export started** (the one GPU job), at code `a4930d5`: `uv run mtgemb export e5-base-v2 --max-length 94 --label "e5-base-v2 fine-tuned (confirm-e5-s1)" --status promoted --model-path runs/confirm-e5-s1/final --run-id confirm-e5-s1`. Output is in `runs/export-v3.out`.
8. **Review results.**
   - **Luna (`p12-annotate-replay-review-luna-r1`):** no findings on the annotation tool (`1451e6c`) or on the replay fix (`f4ea848`). 85 targeted tests passed and ruff is clean. Both are accepted.
   - **Sol (`p12-web-v3-review-sol-r1`):** one P1 on Luna's web change (`ada1cf2`). `validateAbilityShard` rebuilt each card's ability lines from the raw `oracle_text`. The exporter splits the reminder-free model text instead, so any real card with reminder text would have failed its whole shard, taking the whole-card view down with it. The fixture did not catch this because its oracle text had no reminder text.
   - **Fixed by me (`ba91fc4`).** The validator now checks only the shape of an ability list: empty, or two or more entries. The text and neighbour checks on each entry are kept. The browser has no model text to rebuild the lines from; the exporter's lines are pinned by hash and verified by replay. A regression test uses oracle text that contains reminder text. The 11 vitest tests pass and the build passes. Sol's round-2 review is running (`p12-web-v3-review-sol-r2`).
   - **Sol r2 (`p12-web-v3-review-sol-r2`):** no findings; the P1 is resolved. Sol confirmed that the exporter gives every card with two or more abilities a full ability list, that the check-bundle script uses the shared validator, and that the fixture passes. Web change accepted. Its behaviour on the real bundle is checked after the export.
9. **Real v3 export completed** (exit 0). The release is `2026-10-16-e5-base-v2-e6fef183`.
   - **Card vectors:** `vectors_sha256` `a46b4d6f…` is byte-identical to `fc4d438d`, so the model and the card-level neighbours are unchanged.
   - **Ability lines:** 52,699 lines and 42,335 query lines. Ability parity: CPU against XPU min cosine 0.99999988. The float64 reference check had 0 failures over 20 rows, with 18 lists exact.
   - **Size:** the bundle is 70 MB and the artifacts are 243 MB.
   - **Checks:** `node web/scripts/check-bundle.mjs` reports the bundle valid (27,866 cards), so the shape-only validator from step 8 accepts the real export. `mtgemb replay-eval` passed, with all four releases ok, including the new one's ability-artifact hashes.
10. **App switched to v3.** `web/public/data/current.json` now points to `e6fef183`; `fc4d438d` is kept for rollback. I rebuilt the app and checked it in the browser. On Deflecting Swat, "Similar by" lists its two abilities. The redirect line returns Redirect, Insidious Will, Bolt Bend, Redirect Lightning and See Double, and the URL keeps `&ability=1`. There were no console errors.
11. **Human pilot pool built:** `human.build_pool` on dev anchors only, against release `e6fef183`, wrote `runs/labels/471bafd58c8d/` (`pool.json` plus `key.json`).
    - Items: 83 calibration, 1,067 scored and 106 repeats, 1,256 in total, with 100 scored anchors.
    - Blinding: each item in `pool.json` carries only `item_id`, `phase`, `anchor` and `candidate`.
    - The annotation server (`uv run mtgemb annotate runs/labels/471bafd58c8d`) runs on 127.0.0.1:8765. The page loads at 0/1256, and no labels have been written.
    - Labelling is the user's job (P10.5). No label is invented.

## P12: ability-aware model (ideas from Delvefall), 2026-10-18

**Credit.** The ten ideas in this phase come from **Delvefall** by Ballchinian (https://github.com/Ballchinian/Delvefall, mainly `finetune/README.md`), an MTG similar-card site that trains EmbeddingGemma on rules-text lines and Scryfall Tagger tags. Their repository has no licence, so we use the ideas only. No code, training data or exam items were copied; everything here is written from scratch.

The ideas, as Delvefall describes them:
1. Use one-line cards as free line-level labels: every tag on a one-line card belongs to that line.
2. Train lines against the tags they carry ("what a line is about").
3. Mechanism-flip hard negatives (tap/untap, enters/dies, gain/lose, draw-then-discard reversed).
4. A shared clause must not swamp a differing one. Delvefall found 77% of its false positives shared a trigger condition and differed in effect.
5. Number-only rewordings count as the same effect ("parameters are flexible").
6. Old-to-new wording renames as same-meaning pairs.
7. Replace a card's own name in its text.
8. Infer which of a card's tags belongs to which line.
9. A hand-verified trap exam (anchor, true match, near-identical trap) that grows from user reports and is never trained on, under the rule "same mechanism, flexible parameters".
10. Rerun the previous exam after every retrain as a regression guard.

1. **Programme registered (, `ee0f6f5`).**
   - **Fit measurements:**
     - 9,226 one-line text cards, of which 9,206 are tagged, with 4.47 tags on average.
     - 11,380 of 46,909 lines open with a trigger or condition clause.
     - Only 2,984 cards (12.2%) still contain their own name; 8,910 already use Oracle's "this creature" style.
   - **Our deviation on idea 7:** names become "this <type>", not "this card", to match Oracle's own style.
   - **Constraints:** only 3 extension runs and 3 confirmation runs remain, and the sealed split is spent, so new evidence comes from the user-verified trap exam, the dev tag proxy (as a regression guard), a second human pool and, later, a fresh-card holdout.
   - **Run plan:**
     - ext-1 = ideas 1+2.
     - ext-2 = ideas 3–6, on the better of ext-1 and initial-1.
     - ext-3 = idea 7.
     - Ideas 8–10 need no training run.
2. **Trap exam built (idea 9, `8a2bf54`).**
   - `exam/traps.md` is a plain worksheet: each item is an anchor (whole card, or one line of it), a true match and a near-identical trap. The rule is Delvefall's "same mechanism, flexible parameters".
   - Every item starts as **proposed**. Only the user can mark an item accepted or rejected. Only accepted items are ever scored (`mtgemb exam-score`). Nothing is labelled automatically.
   - 51 proposed items: 2 from the user's app review (Deflecting Swat; Bruenor's free-equip line), 19 rider (same trigger opener, no shared tag), 10 mechanism flip (one word swapped, e.g. tap/untap), 20 parameters (the same text with only the numbers changed, as a true match).
   - Candidates are mined by rules from fit and dev one-line cards (dev anchors first). A smoke score ran with n = 0 because nothing is accepted yet.
   - All 153 exam cards, whatever their status, are kept out of every new training input (the aux rows and the later augmentations). They stay in the existing card pairs, which the comparator and the new runs share, so the comparison stays fair. The dev-only exam subset will be reported separately.
3. **Line-tag stream for ext-1 (ideas 1+2, `6fffcd4`).**
   - `linetag.build`: one row per (line, tag) from fit one-line cards, for pair-eligible tags only (fit df 2..500). At most 64 rows per tag (seeded), so broad tags do not dominate. Whole families of exam cards are excluded: a same-text twin of an exam card would otherwise leak its line (we found one, Headlong Rush, in an early mining pass).
   - The tag text is the tag label plus the first sentence of its Tagger description.
   - Real counts: 9,054 lines, 1,489 tags, 19,299 rows (row hash `2d5b9079d7024198`). 2000 updates of 32 rows is about 3.3 passes over the rows; in-batch masking hides about 0.5% of pairs.
   - Training (`aux="line_tag"`): each update adds a symmetric masked InfoNCE loss between 32 lines and their 32 tag texts, with the same encoder and scale, to the unchanged card loss. A tag that a line also carries is never a negative.
   - The card batch schedule is byte-identical to the aux-off schedule (tested), so ext-1 differs from initial-1 only by the added stream. One unavoidable difference: the extra forward pass consumes dropout randomness.
   - The row hash enters the run identity; the checkpoint holds a combined schedule hash; the card schedule hash is logged on its own. Resume is exact (tested).
   - Sent to Sol (`gpt-6.1-sol`, high) for independent review.
4. **Regression guard (idea 10).** `mtgemb exam-guard OLD.json NEW.json` compares two exam score records and lists the items that passed before and fail now (and the reverse). Only items that are identical in both records (same anchor, line, match and trap) are compared; an edited item counts as a new item. Every retrain is scored on the exam and guarded against the deployed model's record.
5. **Self-name rewrite (idea 7, `9842f1b`).** `text.no_reminder_v2` replaces a card's own names (full name, face names, the part before ", ") with that face's Oracle-style noun: "this creature", "this Equipment", "this Aura", "this spell" for instants and sorceries, and so on. "named <Name>" is kept, because it refers to the name itself. Sentence starts read "This creature". It changes 3,267 of 27,573 fit+dev texts; spot checks (Ionize, Blaze, Rile, Revenant, Heaven // Earth) read correctly. An inference-only dev check (the same frozen encoder on v1 and v2 inputs; benchmark unchanged) is the measured-failure test for ext-3. The first attempt crashed on the XPU (no record written; logged FAILED) and was rerun on CPU.
6. **Rule generators for ext-2 (ideas 3–6).** `augment.py`: one-word mechanism flips (hard negatives; "you/opponent" is not synthesised because it makes broken text), same-opener real cards with no shared tag (hard negatives), all free numbers moved up by one (positives; counters like +1/+1 and power/toughness are left alone), and older Oracle wording (positives: "enters the battlefield", "is put into a graveyard from the battlefield", "converted mana cost"). Coverage on fit without exam families (24,378 cards): flip 21,054, numbers 13,617, wording 7,155, opener 1,402 (after Sol r2 fixes: flip 21,046, wording 6,832, opener 1,300). The training stream for these is built only after ext-1, because ext-2 runs on the better of ext-1 and initial-1.
7. **Sol review r1 of the exam and the line-tag stream: FAIL, 3 findings.**
   - P1 "exam cards still train through the card pairs": declined as a code change, because removing them would change the card schedule and ext-1 would no longer be a single intervention. Recorded as the exam is not held out for its fit cards. The wrong "excluded from training" wording is corrected, and every exam score now also reports the dev-only subset (items whose three cards are all dev).
   - P1 "a malformed heading lets the next item overwrite the previous one": fixed. A bad `##` heading or a repeated role line now raises.
   - P2 "aux skips are not counted by the skip guard": fixed and tested.
   - Round 2 sent to Sol, with the self-name rewrite and the generators added to its scope.
8. **Sol review r2: FAIL; 2 of the 3 round-1 items resolved, plus 5 new findings.**
   - Round-1 wording (P3): the exam module docstring still said "never used for training". Fixed to match .
   - "Self noun splits on an ASCII dash" (P2): declined, because the code splits on the Unicode em dash and real Equipment, Aura and Saga cards get "this Equipment", "this Aura" and "this Saga".
   - "draw step" becomes "discard step" (P2): fixed. Flips skip words followed by "step".
   - "Creatures … is unblockable" (P2): fixed by dropping that rule. Plural subjects would need "are".
   - Untagged cards counted as sharing no tag (P2): fixed. Untagged means unknown, so such cards are neither anchors nor opener negatives.
   - `line 0` silently scored the last line (P2): fixed. The parser rejects line 0, and scoring rejects a line the anchor does not have.
   - Round 3 sent to Sol.
9. **Sol review r3: PASS.** All round-1 and round-2 findings are resolved or withdrawn. The exam, the line-tag stream and the generators are accepted.
10. **The user accepted all 51 exam items** (instruction of 2026-10-18; `e98fb26`, logged as ITEMS_ACCEPTED with the worksheet sha). Exam scores (`runs/exam/`):
    - pretrained e5-base-v2 36/51; confirm-e5-s1 (deployed) 47/51; pretrained bge 37/51; confirm-bge-s1 46/51.
    - The deployed model fails T001 (Deflecting Swat; margin −0.573), T021, T022 and T029. Pretrained e5 had T001 at −0.064, so fine-tuning made the rider-clause failure much worse.
    - The dev-only subset has n = 0, so every score here includes exam cards the model trained on through the card pairs. This is weak evidence.
11. **Line-based whole-card scoring (registered before scoring, outcome).** `linescore.py` scores a candidate from its lines against each of the anchor's lines: for each anchor line, take the best-matching candidate line, then average. `line_mean` is a plain mean; `line_invdf` weights each anchor line by 1/(number of cards with the same number-masked line), so common riders count less. The deployed release vectors were used; no training.

    | Ranking | dev full nDCG@10 | hard-lexical | multi-line anchors | exam (of 50 whole-card items) |
    |---|---|---|---|---|
    | whole card (deployed) | 0.7055 | 0.6904 | 0.6974 | 46 |
    | line_mean | 0.6642 | 0.6526 | 0.6350 | 46 |
    | line_invdf | 0.6867 | 0.6748 | 0.6711 | 47 (fixes T001) |

    Neither variant met the promotion rule (it had to beat whole-card on dev nDCG). Whole-card stays the default. `line_invdf` may ship only as an optional, labelled "exploratory" view. The tag proxy credits rider tags, so it penalises exactly the behaviour the user asked for; the human pilot labels are the evidence that can settle this.
12. **Fresh-card test set.** `fresh.py` takes cards from a later Scryfall snapshot whose oracle ids are absent from the whole old raw snapshot, applies the corpus policy, requires release ≥ 45 days before the new snapshot (so community tags have settled), joins families by name-normalised text and keeps only frozen functional tags. It writes a test set only with ≥ 200 anchors. The 2026-10-18 snapshot has 108 new ids, all art-series cards, so there are 0 fresh cards today (logged FRESH_TOO_SMALL). The tooling is ready for the next set release plus 45 days.
13. **Which tag belongs to which line (idea 8).** `tagattr.py`: a tag's centroid is the mean deployed vector of the fit one-line cards that carry it; each tag of a multi-line card goes to its nearest line (tags with fewer than 3 one-line examples stay card-level). Automatic check, no invented labels: where exactly one line of a card has a one-line "twin" card (same number-masked text) carrying the tag, that line is gold; the twin is left out of the centroid. Accuracy 0.944 on 12,919 cases vs 0.449 by uniform chance. 92.8% of 111,238 multi-line tag memberships get a line. The twin cases are the easy ones; accuracy on other lines is unmeasured.
14. **Release v4 and app (delegated).** Sol (`gpt-6.1-sol`) builds export schema v4 (card tags, per-line tags, line-match neighbours from `line_invdf`, reusing the v3 vectors with hash checks). Luna (`gpt-6-luna`) adds tag chips, per-line tags and a "Whole card | Line match (exploratory)" toggle to the app. Each gets a review from the other model.
15. **v4 delegates finished; reviews.**
    - Luna's app change (`b45af79`): 13 vitest tests pass and the build passes. Sol's review r1 was FAIL with 3 findings, all real: two stale-request races (a slow line-match load could overwrite a newer choice or a newer card) and the bundle checker passing no tags, so every v4 bundle would fail. The lead fixed them in `6c53add`. A line load now applies only if the same card is still selected and "Line match" is still chosen. `token` is deliberately not bumped, because bumping it would abort an in-flight card load. The checker now hashes and validates tags.json and lines/. Sent to Sol for r2.
    - Sol's exporter (`d330bfa`): ruff passes and 206 tests pass. Sent to Luna for review.
    - Real v4 release built from the v3 vectors (`--from-release 2026-10-16-e5-base-v2-e6fef183`, 13 min, CPU only): `2026-10-16-e5-base-v2-68086be1`. The bundle checker passes. Whole-card neighbours are byte-identical to v3 in all 109 shards. 2,334 functional tags are used. Deflecting Swat: "change target" sits on the redirect line, and "alternative cost" and "synergy-commander" sit on the free-cast line. Line match ranks Redirect, Insidious Will and Bolt Bend first, where the whole-card ranking ranks Deadly Rollick and Obscuring Haze first. The local app is switched to v4 only after both reviews pass; v3 is kept for rollback.
16. **ext-1 registered and started (, user go-ahead "yes").**
    - The comparator initial-1 was scored on the exam before the run: 49/51 (fails T001 −0.564 and T021 −0.216). The deployed confirm-e5-s1 (same recipe, seed 1) scores 47/51, so the seed alone moves the exam by 2 items.
    - Rule fixed in advance: ext-1 clears only with exam ≥ 50/51 and no item regressed against initial-1, dev full nDCG ≥ 0.7052 and hard-lexical ≥ 0.6895 (each initial-1 − 0.005), and the guardrail passed.
    - Config `configs/ext-1-e5-linetag.json` differs from initial-1 only by `aux=line_tag`, weight 1.0, batch 32. Extension slot 1 of 3 used.
    - Committed and pushed before the run (`c257222`). Started 2026-10-18 ~20:01 UTC. Its STARTED line logs card schedule hash `5839ec9d…`, identical to initial-1's, so the card batches are the same.
17. **Overnight session.** The user lifted the run caps for the night ("queue up a ton more training runs… idm, you have 15 hours") and asked for implementation through Sol to save the lead's usage, so Sol implements the ext-2 training stream from the lead's written design (an explicit override of the AGENTS.md rule that the lead writes training code). `scripts/run_queue.py` runs `runs/queue.txt` configs one by one (waits for the GPU lock, skips finished runs, resumes a crash once). Queue: ext-1, then x-1 (4000 updates).
18. **Self-name diagnosis.** Fine-tuned model: anchors containing their own name score 0.6577 vs 0.7120 for the rest; v2 text at inference gives only 0.6609. Pretrained: 0.4798 → 0.5164. So ext-3 (train on v2 text) targets a measured 0.054 cohort gap.
19. **v4 accepted and switched on locally.** Luna review of Sol's fixes (`e649cc1`: v4 replay checks, reused-vector width and unit-norm checks, stale line-load errors ignored): PASS, no findings. `web/public/data/current.json` now points at `2026-10-16-e5-base-v2-68086be1`; the bundle checker and the build pass. v3 (`e6fef183`) stays on disk for rollback. A live browser check could not be done: the preview automation timed out three times. The UI race fixes are covered by code review only, not by a browser test. Nothing was published.
20. **ext-2 code (synth stream).** Sol built it in a worktree (`ffff62a`: 15,916 rows, 14,535 with a hard negative, positives 11,292 number rewrites and 4,624 wording rewrites, row hash `99d62837ba6a9fbf`). The lead's review found a flaw in the lead's own design: another row's positive was masked only by family, but the card loss treats any tag-sharing card as a positive, so synth would push apart about 2% of in-batch pairs the card loss pulls together. Fixed in `4141860` (mask by family or a shared pair-eligible tag, via `known_pos`), with a test. Sent to Luna for review. It is merged only between runs, because training code is part of a run's identity.
21. **ext-2 and ext-3 code accepted.**
    - Synth stream: Luna review r1 found no correctness problems. One P3 (the row count is not in the run identity) was declined: the row hash pins the rows, and the STARTED line logs the count.
    - ext-3 text option (`6695146`, Sol): `selfname.apply_text_version` is the one place texts are swapped. Card pairs, line-tag rows, synth rows, the in-training dev universe (labels and masks frozen) and `exam-score --text-version` all use it. It fails loudly if v2 changes a card's ability-line count (0 of 27,866 fit+dev cards do). 2,910 of 24,554 fit texts change; 3,267 of 27,573 dev-universe texts change.
    - Luna r1 raised one P2: v2 capitalises a sentence-start "this" even on a card without its own name. The lead checked real data: 1 of 3,267 changed cards (Necratog, "…: this creature gets" → "…: This creature gets"), and Oracle capitalises the effect after a colon anyway. Declined.
    - Luna could not run the tests because the CPU was busy. The lead ran the worktree's full suite: 261 passed.
    - Both are merged into master only between runs, during the queue HOLD after ext-1.
22. **ext-1 result: did not clear.**
    - The run crashed on the XPU at update 1981. The queue resumed it exactly from the 1750 checkpoint, and it completed at 23:13 UTC.
    - Dev full 0.7117 (initial-1 0.7102); hard-lexical 0.6958 (0.6945).
    - Exam 49/51 with the same two failures. T001 got closer (−0.564 → −0.420) but still fails; T021 got worse. No item regressed or was fixed. The rider-section mean margin fell (0.258 → 0.231).
    - The rule needed exam ≥ 50/51, so ext-1 did not clear. The line-to-tag stream did not fix rider matching.
23. **Merged the synth stream and the self-name text option into master** (`6a2ddf8`) during the queue hold. Configs validate, ruff passes, and the full suite passes (261). Pushed. Queue now: ext-2 (synth), ext-3 (self-name text), x-1 (4000 updates).
24. **ext-2 ran out of GPU memory; rerun as ext-2b.** The synth stream adds up to 96 sequences per update. ext-2 failed with XPU out-of-memory at update 2 on both queue attempts, and the queue moved on to ext-3. ext-2b is the same run with gradient checkpointing (memory only, same maths), under a new run id. Peak memory fell to 2.8 GB, at about 8 s per update.
25. **ext-3 result: did not clear.** Training on self-name text left the dev scores unchanged (full 0.7104, hard-lexical 0.6944 on v2 input). It did not help the cohort it targeted: anchors whose text changed scored 0.6692, against initial-1's 0.6728 on the same input (the rule needed 0.6828). The exam fell to 47/51: T003 and T022 newly failed, both by margins under 0.01. Self-naming is not what makes those cards hard.
26. **x-1 crashed on the XPU** (access violation) at update 496 on its first attempt. The queue ran ext-2b next and will resume x-1 from its update-250 checkpoint afterwards.
27. **ext-2b result: did not clear.** Dev full 0.7086 and hard-lexical 0.6933, slightly under initial-1 but within the guardrails. Exam 48/51: T022 newly failed by 0.008. The rewrite stream widened the mechanism (0.246 → 0.316) and parameter (0.617 → 0.654) margins. Rider matching, the target, did not improve (0.258 → 0.238). So all three extension ideas (line tags, rewrites, self-name text) left the two rider failures T001 and T021 in place.
28. **x-1 paused (user request, 10:32 UTC).** x-1 crashed on the XPU a second time (access violation, 08:53 UTC, after update 2993), and the queue gave up on it as designed. The user then asked to pause, so `runs/queue.stop` was written and nothing more is queued. x-1 can resume from `ckpt-0002750.pt` (1,250 updates left). Interim evals, not a result (the run is unfinished, so has not been applied): full 0.7106 at 2000, 0.7132 at 2250, 0.7134 at 2500, 0.7150 at 2750; hard-lexical 0.6991 at 2750. needs full ≥ 0.7152 at the selected checkpoint.
29. **x-1 resumed (user request "unpause", 2026-10-19 21:07 UTC).** Deleted `runs/queue.stop` and restarted the queue runner. The new runner process had a fresh failure count, so it first re-launched the superseded ext-2; I killed it after about a minute (logged ABORTED, no result) and commented it out of `runs/queue.txt`. x-1 then resumed from update 2750 at 21:12 UTC. Disk free had fallen to 49 GB (97%), from outside `runs/` (35 GB); one x-1 run fits.
30. **x-1 finished and cleared .** Completed at 22:22 UTC 2026-10-19 with no further crash. Selected update 3500: dev full 0.7200 (initial-1 0.7102), hard-lexical 0.7043 (0.6945). Exam 49/51 with no item regressed; the rider failures T001 and T021 remain. One seed, dev proxy only; nothing promoted and the sealed split was not touched.
31. **x-1 seed confirmation started; comparator switched (, , user request 2026-10-20).** Two runs of the x-1 recipe at seeds 1 and 2 (`configs/confirm-x1-s{1,2}.json`), scored at the final 4000-update checkpoint, each against the initial-1 recipe at the same seed (confirm-e5-s1/s2). Before training I scored the exam for confirm-e5-s2 (47/51), so every comparator has an exam record. Rule fixed before training: full +0.005, hard-lexical within −0.005, no exam item regressed, for both seeds. Future extension runs are now judged against x-1 (0.7200 / 0.7043, exam 49/51).
32. **More steps confirmed on two fresh seeds.** confirm-x1-s1 and -s2 finished 2026-10-20 (s1 crashed once on the XPU and resumed from update 3000). At the final checkpoint: seed 1 full 0.7142 (vs 0.7055 at 2000 updates, same seed), seed 2 0.7165 (vs 0.7066); hard-lexical rose too. Exams: seed 1 47/51 (unchanged failures), seed 2 49/51 (fixed T007, T022); no item regressed at either seed. So the +0.009 gain holds across three seeds on the dev proxy. T001 and T021 still fail everywhere. Nothing promoted or published; the sealed split was not touched.
33. **Promotion test of the 4000-update recipe registered, and a new idea x-2, user request 2026-10-21.** The promotion test needs three seeds, so `confirm-x1-s3` started at 11:3x UTC. `report.confirmation_report` gained a `design` argument: the x1 design adds the deployed 2000-update recipe as a third comparator (M = 6), so a swap must beat what is deployed, not only pretrained E5 and trained BGE; it writes to `runs/confirmation/<bench>/x1/` and refuses the sealed path. The user's laptop is lent until 15:00 local, so a timer stops training at 14:57 local; seed 3 resumes from its last checkpoint later. For the rider failures I looked at the labels first: T001's trap shares three tags with the anchor (more than the match), so training on tags cannot fix it without a label change; T021's trap shares no tag and is the top lexical negative of its anchor. x-2 therefore triples the lexical hard negatives per row (with gradient checkpointing for memory), aimed at T021-type lexical twins.
34. **Seed 3 paused at the end of the user's GPU window (15:00 local).** confirm-x1-s3 crashed on the XPU three times (11:44 UTC before any checkpoint, 12:42 after update 750, 13:54 after update 1750); after the second crash the queue skipped to x-2, which I stopped within a minute (no run folder made) and held in `runs/queue.txt`. `runs/queue.stop` is written and nothing is running. Seed 3 resumes from `ckpt-0001750.pt` (2,250 updates left); x-2 follows it. The report code is still awaiting its independent review (Codex usage limit until 14:44 local).
35. **report code reviewed: accepted.** Independent review by codex/gpt-6-luna (Sol was blocked by the router hook as L4) found no contract breaks in `1b53ad0`: the path and `sealed.py` are unchanged, the x1 design provenance-checks candidate and incumbent runs at matched seeds, uses M = 6 for both the quantile and the resample count, keeps the family floor, writes once to its own path and refuses the sealed path. I re-ran `tests/test_report.py` (9 passed).
36. **Card Atlas site rebuilt from scratch (user request 2026-10-21: "redesign it from scratch … implement it yourself no subagents").** Design spec `docs/superpowers/specs/2026-10-21-card-atlas-redesign-design.md`, plan `docs/superpowers/plans/2026-10-21-card-atlas-redesign.md`. The lead wrote all of it; no subagent was used, so there is no independent review, only the lead's self-review.
    - Stack: SvelteKit 2 with the static adapter (Kit 3 needs Vite 8, so not used), Svelte 5, Vitest. Two prerendered pages: the explorer at `/` (card map, search, neighbours, compare) and the case study at `/how-it-works`.
    - `mtgemb site-data` (`src/mtgemb/sitedata.py`) makes the site's extra data from existing records only: a t-SNE map of the release vectors with region labels from the highest-lift functional tag, the pretrained E5 neighbours for the before-and-after, and `web/src/lib/generated/metrics.json` (dev scores per seed, the promotion report, the one sealed report, exam results, 2000 vs 4000 update scores, dataset counts, the deployed config). It reads the sealed *report*, never the sealed split. It refuses to overwrite a site folder.
    - Build: `npm run build` prunes `build/data` to the deployed release and its site data. CSS is inlined and the two Latin fonts are preloaded. Lighthouse (mobile, local preview): case study 95 / 100 / 100 / 100 (performance, accessibility, best practices, SEO), explorer 97 / 100 / 96 / 100, layout shift 0 on both. The explorer's 96 is two 404s for the site data, which was still being generated.
    - A first Lighthouse run was invalid: an old preview server from 2026-10-18 still held port 4173, so it measured the old app. A second was invalid because the preview server cached the file list from before the rebuild. Only the third run (fresh server, fresh build) is reported above.
    - Site data for the deployed release: `map.bin` 139,330 bytes (27,866 cards × 5), 36 map regions, 109 pretrained-neighbour shards. The generator's final rename failed on Windows (the dev server's file watcher held the folder); the output was complete, so I renamed it by hand after stopping the dev server, and the README now warns about it.
    - Six before-and-after examples, picked by reading both top-8 lists: Lightning Bolt, Giant Growth, Swords to Plowshares and Rampant Growth (clear gains: for example, untuned E5 puts three −3/−3 spells next to Giant Growth), Llanowar Elves (both lists identical) and Ghostly Flicker (mixed: the tuned list also ranks two cards that exile artifacts for good).
    - Self-review of the whole site change (no independent reviewer, by the user's instruction): two defects fixed (an error banner that never cleared; empty comparison columns for cards with no rules text), five minors deferred (listed in the final report). Screenshots at 1440 px (light and dark) and 360 px are in `runs/webshots/v3-*.png`.
37. **Seed 3 died silently and was resumed.** confirm-x1-s3's trainer process disappeared at about 15:06 UTC after update 2554, with no crash or failure line and no queue runner left. I restarted the queue at 15:41 UTC (niced to below normal); it resumed from `ckpt-0002500.pt`.
38. **read out: promote, and the site now serves the 4000-update model.** Seed 3 finished at 17:40 UTC (dev full 0.7180). The x1 report (20:12 UTC) passes all three contrasts: over pretrained E5 +0.151 full, over trained BGE +0.053, over the deployed 2000-update recipe +0.0098 (adjusted lower bound +0.0063) and +0.0102 hard-lexical. Per I exported `confirm-x1-s1` as a new promoted release `2026-10-16-e5-base-v2-841ee2af` (the exporter gained a `decision` argument, so the release says ", no sealed evaluation of this model"), generated its site data, pointed `current.json` at it and rebuilt the site. The old release stays on disk. The case study now names the sealed-tested model (confirm-e5-s1) separately from the deployed one. One before-and-after example changed: the new model fixes Ghostly Flicker, so the honest "miss" example is now Ghosts of the Innocent (neither model finds a damage-prevention card). Nothing was published.
39. **x-2 started** at 20:56 UTC when I un-held it in `runs/queue.txt` (the queue runner picked it up before my restore-point push; its training code was already the committed HEAD, and I pushed within minutes). Disk free is 11 GB; x-2 needs about 2.6 GB.
40. **Site review by Sol, fixes, upgrade, line match back (2026-10-21 evening).** At the user's request two GPT-6.1-Sol (high) reviewers read the site code at `79bef96` (explorer; case study, build scripts, site-data/export). They found 16 issues; I fixed all of them in `842cf80`, each with a test where the logic is testable: v2/v3 bundles asked for the v4-only `tags.json`; "Show more" never appeared (only 12 results were reachable); an early search could be wiped by the start-up URL; a new deployment reused old cached shards; the header link kept a stale selection; a stale error could show on a newer card; map motion ignored reduced-motion; search could not reopen after Escape; mana pills had low contrast; bad URL filter values hid all results; `npm run fixture` did not switch `current.json` (and the build now fails if the named release is missing); a before-and-after start-up race; and, most important for honesty, `export` would stamp any promoted model without `--decision` with the text that claims a sealed run (now only `confirm-e5-s1` may use that text). The case study no longer says the trap exam measures tag agreement. Also: restored the "Line match" ranking (line_invdf, labelled exploratory; its shards are shipped again); upgraded the web stack to the latest (SvelteKit 3.1, Vite 8.3, Svelte 5.57, Vitest 5; config moved into `vite.config.js`, `$lib` became `#lib`); hybrid mana now draws as split pills; Tagger jargon region names get readable display names; cards with identical text no longer hide under one map point. Python packages were not upgraded: x-2 is training on the current environment and Windows locks its files. Web 52 tests, Python 269 tests, ruff clean, build and bundle check pass. A round-2 Sol review of these changes was started.

41. **Site: snapshot branch, mana icons, colour combinations, quieter UI (2026-10-21).** Pushed the pre-change site as branch `web-snapshot-2026-10-21` (commit 02c93d8). Browser-checked two round-1 fixes that had only been reasoned about: the header and Explore links clear a selected card and Back restores it; with Chrome's `--force-prefers-reduced-motion` the map jumps in one step (2 distinct frames vs 18 animated) and the scroll uses `auto` instead of `smooth`. Investigated why Lightning Bolt ranks differently under Whole card and Line match: whole card compares one vector per card, while line match scores a candidate by its single closest line to Bolt's only line (for a one-line card the 1/df weight cancels out), so long cards with one burn line rise (12 of the top 50 overlap). Filters only narrow the stored top 50. The Whole card top 50 has no card with white in its colour identity, but the line-match list has Heartflame Duelist (W creature // R adventure "deals 3 damage to any target", identity WR). The UI now says that filters narrow the top 50. Changes: mana symbols are drawn with the Mana icon font (mana-font 1.18.0, CSS MIT / font OFL), vendored without its MPlantin font-face, which is not ours to ship. The colour filter is a set of mana-symbol toggles with Including / Exactly / At most matching (URL `color` in WUBRG order, `cm`). The creature filter is a button group, not a dropdown. A link with unrecognised filter values shows a notice naming them. Restyle: neutral graphite palette, so colour on the site means mana colour; Schibsted Grotesk for the UI and Source Serif 4 for rules text; results as a ruled list; removed the caps labels, numbered section markers, the accented headline word and the link arrows. Web 59 tests pass; build and bundle check pass. A Sol (high) review of 02c93d8..5aaf928 was started.
42. **Sol round-2 site fixes (`1f5155b`).** Round 2 found six issues. Fixed: switching Whole card / Line match while a card was loading left the list empty (browser repro: 0 rows before the fix, 12 after); `export` with a blank `--decision` still fell back to the sealed-run text (test: blank and whitespace now refused); two cards could be fanned onto a point a third card already held on the map (test). Applied but not shown failing first: a stale neighbour error on a newly selected card (my browser repro did not trigger it with or without the fix). Not changed: hybrid mana conveyed by colour alone (obsolete, the Mana font draws both symbols) and Kit 3's deprecated `pushState` import (minor, deferred). Web 60 tests, Python 269 tests, ruff clean, build passes. The round-3 Sol review of step 41 did not run: Codex usage limit until 02:54 local, 2026-10-22.
43. **x-2 restarted under GPU limits (user report: laptop freezing, 2026-10-22).** `mtgemb train` now reads two optional env vars (`bf5c5d1`): `MTGEMB_GPU_MEM_GB` caps the XPU memory the process may hold, and `MTGEMB_GPU_REST=r` idles the GPU for r times each update's duration after every optimizer step (capped at 60 s). The training maths, schedule and RNG are untouched, and the code lives in `cli.py`, outside the run's code identity, so the run resumes under the same identity. I stopped x-2 at about update 2410 (logged INTERRUPTED; the work after `ckpt-0002250.pt` is redone) and restarted the queue at below-normal priority with `MTGEMB_GPU_REST=1`, `MTGEMB_GPU_MEM_GB=6` (peak use was 3.04 GB of 27 GB shared). It resumed at 01:20 UTC. GPU counters show about 6 s busy then 5 s idle per update, so x-2's remaining ~1,750 updates take roughly twice as long (about 6 h instead of 3 h). Evaluations every 250 updates are not throttled inside.
44. **Release v5: 200 neighbours, colour vs colour identity, grid view (, `0a2818a`, user request 2026-10-22).**
    - The exporter now keeps the top 200 whole-card and line-match neighbours per card (was 50; per-ability lists stay at 50). Each card gets `colors` (printed colours from the raw Scryfall snapshot: top-level, or the union of faces for transform and modal DFCs, so devoid cards are colourless) and `image_back` (second printed face, 449 cards: 358 transform, 91 modal DFC). The raw file also holds sealed cards; they are parsed only for their oracle_id and dropped, and the export refuses a raw manifest that does not match `dataset.json`.
    - Re-exported the deployed model with `--from-release` (same vectors, no GPU): `2026-10-16-e5-base-v2-f52a3aef`, 206 MB (the v4 release was 91 MB). The final folder rename failed again because the dev server's watcher held the folder; I stopped the dev server, did the two renames the exporter would have done and logged COMPLETED by hand. 1,541 cards have colours that differ from their identity.
    - Effect of 200: Deflecting Swat's ideal match Redirect is at rank 138 under Whole card (outside the old 50) and rank 1 under Line match. Lightning Bolt filtered to white identity had 0 results; with 200 it has 5 (Lightning Helix, Warleader's Helix, Ultimate Alliance, Hurloon Battle Hymn, Sunlance).
    - Site: the colour filter can match colour identity (default) or printed colour (`cb=color`, v5 bundles only); a List / Grid view switch (`view=grid`); grid cells show the card image, and double-faced cards get a Flip button (back image), Kamigawa flip cards turn 180°. Luna was routed the grid cell but hit the Codex usage limit, so I wrote it. Web 72 tests, svelte-check clean, export tests 32 pass.
    - Swat diagnosis numbers (fit pairs, `pairs_v1`): the two rider tags Swat shares with Obscuring Haze produce 401 training pairs (synergy-commander 189, manaless value 212); "change target", the tag it shares with Redirect, produces 47. Swat is a dev card, so it is never trained on directly. Also: tags with more than 500 fit cards make no pairs, so "burn any" (612 cards, Lightning Bolt's main effect tag) contributes no training pairs at all.
45. **Site switched to the v5 release; browser checks; reviews blocked.** Site data for `f52a3aef` was generated (CPU t-SNE and pretrained E5 neighbours, about 3 h of CPU time) and `current.json` now points to it; build and bundle check pass. Browser checks on the built site, served as static files: Lightning Bolt with white selected gives "5 of the 200 most similar cards match" by identity and 4 by printed colour (Hurloon Battle Hymn drops out: it is red with a white kicker cost); the grid shows card images; on Delver of Secrets' grid, Flip turns Aberrant Researcher into Perfected Form (`/back/` image, `aria-pressed=true`, alt text names the shown face); the map loads. Fixed a missing space in "Ranked on the whole card. Pick a line…". `vite preview` served an old `current.json` (from a stale SvelteKit output folder), so I tested on a plain static server over `web/build` instead. Both Sol reviews (bf5c5d1 + 0a2818a; and round 3 of 02c93d8..1f5155b) failed again on the Codex usage limit, so these changes have no independent review yet.
46. **Sol round-3 site review and fixes (`87c44b4`, 2026-10-22).** The round-3 Sol (high) review of `02c93d8..1f5155b` ran once the Codex limit reset and found four issues, all reproduced on master and fixed: the keyboard focus ring vanished on a pressed button (accent outline on the ink fill; now the page colour); unselected colour icons were dimmed to about 2.2:1 contrast (now full colour, the ring alone marks a pick); the "unrecognised value" link notice stayed after the user changed filters (now cleared by any filter change or Clear filters); and a negative mana bound (`max=-1`) hid every result silently (now rejected and named in the notice, test shown failing first; the number inputs reject negatives too). The notice also says "those filters are off" when it names more than one. Browser check on Lightning Bolt with `color=WW&max=-1`: notice names both, clears after picking Red; pip opacity 1; pressed-segment outline light on dark. Web 72 tests, svelte-check clean. The Sol review of steps 43-46 (GPU limits, release v5, grid, these fixes) was started.
47. **Effect-only labels, tag-IDF weights and face match (, user: "do all of these", then "pause development, just get everything queued up and wait for them").**
    - Labels: `pairs_v2` (`366250a`, pairs id `37ce1f52261f9967`) drops 54 tags about a card's own cost, casting mode or printing (`pairs.MODIFIER_LABELS`) from pair making, from the in-batch masks and from the lexical-negative exclusion. 105,632 pairs (v1: 108,220); 1,928 fit anchors now have a rider twin (shares only a dropped tag) among their top-30 hard negatives, which is option 2 (the Obscuring Haze kind of card becomes a negative) without a separate mechanism. A test reproduces the Deflecting Swat shape on four toy cards.
    - Training: `train.py` gains `pairs_version` and `pair_weighting=tag_idf` (each pair weighted by its own tag's fit IDF, mean one, clipped; real range 0.76–1.55). Because `train.py` is in x-2's code identity, the change is committed on branch `d54-train` (`93f4869`) and not merged: `scripts/d54_watch.py` (pid 14608) waits for x-2's `result.json`, merges the branch and appends x-3 and x-4 to `runs/queue.txt`. Both runs are registered with fixed rules in (configs `x-3-e5-effect`, `x-4-e5-tagidf`); both use gradient checkpointing to fit the 6 GB GPU cap.
    - Face match (`variants.py`, `43390bd`): per-face vectors for the deployed model, scored on dev without training. Both-sided best face is worse (full 0.7142 → 0.7134, multi-face anchors 0.650 → 0.579); candidate side only is about neutral (0.7155 full, 0.6967 hard-lexical, 0.657 multi-face). Effect-only relevance for the deployed model: 0.7059 full. Results in `runs/diag/d54/v2/confirm-x1-s1` (the first both-sided-only run stays in `runs/diag/d54/confirm-x1-s1`). The site option is not built yet (development paused by the user).
    - Slip: one inventory command read `cards.parquet` without the sealed filter and printed a count of sealed multi-face cards (69). No sealed text, tags or scores were read.
48. **Sol review of steps 43-46 (GPU limits, release v5, grid, r3 fixes) at `87c44b4`.** No Critical or Important findings; Python 34 and web 72 tests pass in the reviewer's worktree, synthetic v2-v5 bundles validate. Two Minor findings, deferred because the user paused development: v5 validation accepts `k`/`line_k` of 0 (should require 200), and a rejected negative mana value stays visible in its input box although the filter and URL ignore it. Unverified by the reviewer: real bundles, live XPU behaviour, screen readers.
49. **Sol's two minors fixed (2026-10-22, user: "continue, unpause").** `validateRelease` now requires `k` to be a positive integer and exactly 200 for v5 (line_k must then be 200 too); the new test failed first (`k: 0, line_k: 0` was accepted). A rejected mana value (e.g. `-1`) is now cleared from its input box. Browser check on the dev server: max 3 sets `max=3`; -1 clears the box and the URL; -2 typed into an empty min box clears too. Web 73 tests, svelte-check clean.
50. **Face match on the site: release v6 (, `363b637`, `bdfd4b5`).** A third "Rank by" option, Face match, exploratory and never the default. A candidate card scores the best of its whole-card vector and each of its faces against the selected card's whole-card vector, the form that was about level with whole-card ranking on dev in (the both-sided form was worse). Exporter: v6 = v5 plus `faces/NNN.json` (top 200 per card), `face_k`, `face_status`; face texts are encoded on the CPU (the GPU stays with the training queue) and saved with their hash, which is part of the release id; an independent float64 check must agree with the block top-k. `site-data --copy-from` reuses the v5 map when the card vectors and order are identical, avoiding about 3 h of t-SNE. Luna (gpt-6-luna, medium) wrote the web side against a fixed contract; I read the diff and re-ran web tests (78 pass), svelte-check (clean) and the v5 bundle check. Found and fixed on the way: the replay check of v4 bundles compared `line_k` with the exporter's current K (200 since ), so 25 replay tests had been failing since `0a2818a` without anyone noticing, because earlier runs only covered subsets. Python 280 pass. The real v6 re-export of the deployed model (from the v5 vectors) and a Sol (high) review of `87c44b4..bdfd4b5` were started.
51. **v6 exported and switched on (`2026-10-16-e5-base-v2-3984a6b7`).** Re-export of the deployed model (`confirm-x1-s1`) from the v5 vectors: 1,526 face texts encoded on the CPU, face top-k agreed with the float64 reference on all 20 rows (15 exact lists, the rest near-ties). The final folder rename failed (WinError 5) because two of my dev servers were still running and watching `web/public` (a port-5199 one that `TaskStop` had not killed, and an orphaned port-5173 one from earlier in the session); I stopped both, completed the two renames with the exporter's own `finish_interrupted` check and logged COMPLETED with `recovered=True`. Bundle 283 MB, artifacts 248 MB; `check-bundle` passes. Face match changes the top 10 for 8,598 of 27,573 text cards. Lightning Bolt under Face match: Emeritus of Conflict // Lightning Bolt is first (it was not in the whole-card top 6). Site data copied from v5 (`site-data --copy-from`), `current.json` points to v6, metrics regenerated, build passes. Browser check on the built site (static server): `rank=face` link opens with Face match pressed and the exact note; Whole/Face/Line switching updates the URL and the list (the Line list matches `lines/`, the Face list `faces/`); picking Lightning Strike in Face mode shows its own face list.
52. **Sol (high) review of `87c44b4..bdfd4b5` and fixes.** No Critical findings. Reviewer ran Python 280 tests, web 78 tests, svelte-check, the v5 bundle check and 24 extra face-ranking cases (including a catalogue with no multi-face cards); all passed. Three findings, all reproduced by the reviewer and fixed test-first: (Important) `site-data --copy-from` trusted recorded hashes and ids, so two releases from different datasets with the same ids could get a map whose rows belong to other cards; it now requires the same dataset and checks both vector files against their recorded hashes. (Important) reused face vectors were not checked for float dtype or unit norm (an integer source scored 2.0 and the float64 reference shared the bad input); they now get the same checks as the card vectors. The real v6 face vectors and the real v5→v6 site copy pass the new checks. (Minor) a failed Line or Face request could leave an error on screen after switching mode or after a successful retry (the same pattern existed for Line match); row errors are now separate, shown only while that mode is selected, and cleared on a mode change, a new card or success. Checked in the built site with a server that returns a delayed 503 for face shards: a late failure after switching to Whole card shows nothing; a failure in Face match is shown and clears on switching; the retry loads Emeritus of Conflict // Lightning Bolt first with no error. Python 287 pass, web 78 pass, svelte-check clean. The review run then hit the Codex usage limit.
53. **Row-error fix after Sol round 2 (`5392680`).** Sol confirmed both Important fixes and passed 6 cards and 1,200 stored face scores against an independent float64 scoring, but found four paths where a row error still leaked (choosing an ability line, `clear()`, a row failure inside `select()` turning into a global error, a late Face success hiding a Line failure). Each row error now carries its mode and is shown only while that mode is selected with no ability chosen; only a success in the same mode clears it; `select()` no longer fetches row lists. Checked in the built site with a local server that fails or delays shards: (A) Face fails, then choosing an ability line hides the error; (B) Face fails, then Whole card shows nothing; (C) Line fails at 11.6 s, a slow Face request succeeds at 12.5 s, and the Line error is still on screen at 15 s (resource timings recorded). Web 78 pass, svelte-check clean. Sol round 3 requested.
54. **x-2 judged; pipeline restarted.** x-2 clears part (b) (dev full 0.7263, hard-lexical 0.7119) but not part (a): exam 48/51 with T011 regressed against x-1, and T001 and T021 still fail. Did not clear the gate; nothing promoted. Both dev evals in the pipeline died in a native XPU abort (0xC0000005), so the face-match and effect-only variants for x-2 and x-1 are not yet measured. x-3 is training (started 07:35 UTC). The pipeline was restarted (`fb68af0`) to re-run those evals in the GPU gap between x-3 and x-4, with retries and faulthandler.
55. **Row-error fix after Sol round 3 (`77fc878`).** Sol round 3 found no Critical or Important issues and no pipeline issues (GPU evals run before the single x-4 queue append). It reproduced one Minor: a late failure in the inactive mode replaced the single row error and hid the active mode's error. It also noted a pre-existing gap: re-clicking a failed mode cleared the error without retrying. Row errors are now kept per mode (`rowsError.face`, `rowsError.line`). Each fetch effect waits while its own error is set, so selecting that mode again clears the error and genuinely retries (the loader cache drops rejected promises). Checked in the built site with the fail server: (D) Face match fails at 10.4 s after switching to Line, whose request failed at 6.2 s; the Line error is still shown at 13 s. (E) After clearing the failures, re-clicking Line and then clicking Face each fire one new request, clear the error and render the rows (Aberrant Researcher first). Web 78 pass, svelte-check clean. Sol round 4 passed with no findings (compiled-runtime checks of paths A–E, retries and the v5 fallback; no duplicate requests or retry loops; it did not re-check browser rendering).
56. **GPU limits lifted; old checkpoints deleted.** At the user's request x-3 was un-throttled: the queue runner was stopped and restarted without `MTGEMB_GPU_REST`/`MTGEMB_GPU_MEM_GB`, and x-3 resumed from `ckpt-0002500` at 16:07 UTC (about 80 updates redone; logged as `GPU_LIMITS_LIFTED`). Updates went from about 12 s to about 3.5 s. With the user's authorisation, the 23 resume checkpoints of finished runs (`runs/*/ckpt-*.pt`, about 23 GB) were deleted (file list logged as `CHECKPOINTS_DELETED`). Every run's `best/` and `final/` models and x-3's checkpoint are kept. Free space on C: rose from 7.0 GB to 30 GB.
57. **x-3 judged.** x-3 meets the effect-only part of its rule: effect-only full 0.7137 and hard-lexical 0.6995, against x-1's 0.7122 and 0.6965. It fails the exam part. The exam scored 48/51; T001 still fails, though its margin moved from -0.449 to -0.034, and T029 regressed against x-1. Did not clear the gate; nothing promoted. Standard dev fell to 0.7147 / 0.7005 from x-1's 0.7200 / 0.7043. Candidate-side face match is within ±0.002 of whole-card for x-1, x-2 and x-3. x-4 (tag-IDF weighting) started training at 18:06 UTC unthrottled.
58. **x-4 judged; round closed.** x-4 (tag-IDF pair weighting) selected update 3500. Dev full was 0.7230 against a bar of 0.7250, and hard-lexical 0.7079 against 0.6993. The exam scored 46/51, with T011, T022 and T029 regressing against x-1. It did not clear the gate, and nothing was promoted. None of x-2, x-3 or x-4 cleared its gate, so x-1 remains the comparator and the deployed release (v6) is unchanged. The pipeline finished at 22:22 UTC.
59. **Release re-check and docs (user: "do 1 and 2, use a gpt 6.1 sol subagent to do code reviews", 2026-10-23; ).**
    - Docs: README status rewritten for the deployed model (confirm-x1-s1, 4000 updates, release v6 `3984a6b7`; the sealed numbers are labelled as the earlier model's; no sealed evaluation of this model); new `MODEL_CARD.md` and `DATA_CARD.md`; STATE `release_id` and promotion updated; the handoff table updated. Correction: I had told the user the deployed model fails 2 of 51 exam items; it is 47/51 (fails T001, T021, T022, T029).
    - Replay: `mtgemb replay-eval` now checks v5/v6 bundles (structure, face-vector hash, and 20 seeded cards' stored whole-card and face rankings recomputed from the release vectors; `0b9d37a`). Sol r1 found two issues (the top-k check compared against rounded scores, so a wrong index near a tie could pass; an empty max for a single text card), both fixed test-first (`bba0d43`); Sol r2 passed. The full replay on the fixed code passed: 16 evals and 8 releases, max difference 0.0 (`runs/logs/replay-v6.json`). A first replay run on the pre-fix code was stopped and is not counted.
    - Clean clone (`git clone --no-local`, at `0b9d37a`): uv sync, ruff, 309 pytest, tiny resume test, npm test 78, build all pass. Later commits change only `baselines.py` and `tests/test_replay.py`, docs and `.gitignore`; the clean clone was re-run at the final commit (below).
    - Sol final release review r1: four Important (the proxy was described as weighting tags equally, it uses fit-IDF weights; the in-batch mask was overstated; 47/51 lacked the exposure note; `runs/` outputs were not git-ignored) and two Minor (parenthetical stripping removes more than reminder text; stale STATE), all checked and fixed (`c5ca22a`). r2: the plan's exam line and nested STATE queue fields were still stale; fixed (`06db701`).
    - v6 timings. `vite preview` served a stale `web/dist` (pointing at v4 `68086be1`); I served `build/` with `--outDir build` instead and left `dist/` alone. Payload: all app and startup data files total at most 0.97 MB gzip (pass, target 5 MB). Usable lookup on the Lighthouse mobile profile (6 s gap): 5.42 / 5.48 / 4.68 / 4.45 s and 4.63 / 4.44 / 3.63 / 4.86 / 5.05 s; one 7.23 s run carried the previous run's `ids.json` download (546 KB transfer) and is excluded. **Fails** the 3 s target (v2 was 2.5 s). Unthrottled 0.23–0.27 s. Startup transfer 405 KB. The timeline shows about 2 s of app JS and web font under the 4× CPU throttle before the data chain `current.json` → `release.json` → `names.json` (1.4 s) starts. A Vite dev server from another worktree held about 30% CPU throughout. Not fixed: changing the app is outside this re-check; README now states the miss.
    - Sol final review r3: accept, no Critical or Important findings (one Minor wording fix in this step). Clean clone re-run at `b0d6462`: uv sync, ruff, pytest 311 passed (includes the tiny resume test), npm ci, vitest 78, and `npm run fixture` then build pass. Without `npm run fixture` the build stops by design, because the clone has no release data (`current.json` names a git-ignored release).
