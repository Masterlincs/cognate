# Card Atlas

A small Svelte app for exploring a release bundle. It loads `public/data/current.json`, then the release metadata and name index, and fetches card/neighbour shards when needed. It never substitutes fixture data for a failed release.

```sh
npm install
npm run fixture
npm test
npm run build
npm run dev
node scripts/check-bundle.mjs public/data/fixture-v1
```

`public/data/current.json` points at the promoted release `2026-10-16-e5-base-v2-d4c393ee`. Its bundle is git-ignored (39 MB); rebuild it with `uv run mtgemb export` (see the top-level README) or run `npm run fixture` to switch the pointer back to the synthetic fixture. `check-bundle` validates a bundle against the `current.json` beside it, so check the bundle the pointer names.

Fixture card names and rules text are synthetic. The app can also be served below a path prefix because Vite uses a relative base. Bundle layout and fields are defined in [SCHEMA.md](SCHEMA.md).
