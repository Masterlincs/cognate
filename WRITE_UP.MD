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
