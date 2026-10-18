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

## P10: reproduction and handoff (2026-10-18)

1. **Evaluation replay from pinned embeddings.** I added `mtgemb replay-eval` (7554041, with 3 tests: untouched records pass, tampered vectors fail, a release whose vectors are not the evaluated model fails). For every recorded eval on benchmark `8bf2f48c0bba3b2c` it:
   - verifies the recorded file hashes;
   - re-scores the saved `vectors.npy` (lexical: refit on fit text);
   - compares every per-anchor metric, the NaN pattern and the integer counts.

   For each exported release, it checks the vectors sha256 and compares the vectors with the evaluated run's vectors.

   Result: all 9 evals (6 confirmation, 3 baselines) replay with max abs per-anchor difference **0.0** and no hash mismatches. Both promoted releases (`d4c393ee`, v1; `fc4d438d`, v2) hold vectors **bitwise identical** to the evaluated `confirm-e5-s1` vectors, so the released model is the evaluated model. First report `runs/replay/8bf2f48c0bba3b2c/2026-10-18T072329Z.json` (made from uncommitted code); re-run from the commit for clean provenance.
2. **Run reconciliation.** 18 run directories: 3 smoke, 9 initial, 6 confirmation. Every one has STARTED and COMPLETED events in `runs/train.jsonl`, `finished: true`, a selected update and a passed guardrail.
   - Crashes, each followed by RESUMED: initial-2, 3, 4, 6, 8; confirm-e5-s1, e5-s3, bge-s2, bge-s3.
   - initial-3 also has one PAUSED/RESUMED pair.
   - Run counts against the caps: initial 9/10, extension 0/3, confirmation 6/9.
   - Export crashes (no partial output published): v2 attempt 1. The earlier `9a16efe0` is the P8 dry-run export of the pretrained model (`run_id` null); `9a16efe0.failed-1` is a failed attempt, kept and not deleted.
3. **Clean environment.** Fresh `git clone` of origin at 7554041 into a temp dir:
   - `uv sync` 19 s, with a warm uv cache, so not a cold network install;
   - `ruff check` clean; `pytest -q` 73 passed, 4 real-model tests deselected; the tiny resume test `test_resume_matches_uninterrupted_and_rejects_corruption` passes;
   - `npm ci`, 9 vitest tests and the build pass.

   A fresh clone's `current.json` names the real release, which is git-ignored, so the app needs `npm run fixture` or an export first (documented in `web/README.md`).
4. **No full retraining replay** has been performed; the evaluation replay is not equivalent.
5. README status, results tables and limitations updated. All numbers were re-read from the eval summaries and the confirmation report.
