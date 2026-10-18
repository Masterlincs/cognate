# Web schema v3 implementation

1. Read repository instructions, plans, phase state, the decision, and the web schema, runtime, fixture, and test code.
2. Added v3 release metadata and complete manifest validation while retaining v2 support. Added aligned ability shard validation for text lines, list shape, scores, self-exclusion, index bounds, and textless/single-ability cards.
3. Added per-ability selection, shared neighbour filtering, an experimental status note, keyboard focus styling, and `ability` URL state with Whole card fallback for invalid indexes.
4. Changed the fixture generator to write a separate synthetic v3 fixture with multi-ability, single-ability, and textless cards. It does not modify `current.json`.
5. Documented v2 and v3 and updated the bundle checker to validate v3 hashes and shards, including standalone release directories.
6. Verified with `npm test` (11 passed), `npm run build`, `node scripts/check-bundle.mjs public/data/fixture-v3`, and `git diff --check -- web`.
