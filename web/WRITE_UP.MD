# Web implementation walkthrough

1. Read `SCHEMA.md` as the release format contract and reviewed P9 in the project plans.
2. Added a deterministic 600-card fixture generator with three shards, invented names/text, aliases, a textless card, colourless cards, varied filters, empty image values, and manifest hashes and byte counts.
3. Added shared schema/shard checks, a bundle CLI validator with SHA-256 and byte verification, and unit tests for search, filtering, URL state, and malformed bundles.
4. Built a Svelte/Vite static app with startup metadata/name loading, lazy cached shard loading, visible failures, accessible keyboard search, face details, filters scoped to saved neighbours, Oracle ID URL state, and stale request protection.
5. Added responsive styling, fixture labeling, release metadata and attribution display, setup instructions, and a Vite relative base for subpath hosting.
6. Ran the requested fixture, validator, test, build, and corruption checks. Exact results and commit IDs are recorded in the handoff message.
