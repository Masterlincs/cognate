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
