# Card Atlas redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. (User instruction: the lead implements every task itself; no subagents.)

**Goal:** Replace the single-page `web/` app with a two-page SvelteKit static site: the explorer (`/`, map + results + compare) and the case study (`/how-it-works`, write-up + before-and-after + charts).

**Architecture:** SvelteKit with `adapter-static` in `web/` (rewritten in place; the old app stays in git history). The explorer reads the existing v4 release bundle on demand. A new Python module `mtgemb.sitedata` writes a per-release site-data folder (map coordinates, region labels, pretrained neighbours) and a committed `metrics.json` built only from run and evaluation records. Charts are hand-written SVG; the map is a canvas with a grid hit-test.

**Tech Stack:** Svelte 5, SvelteKit 2, `@sveltejs/adapter-static`, Vite 6, Vitest; `@fontsource-variable/fraunces`, `@fontsource-variable/inter`, `@fontsource/jetbrains-mono`; Python 3.12 with numpy, scikit-learn (installed), the project's encoder.

**Spec:** `docs/superpowers/specs/2026-10-21-card-atlas-redesign-design.md`

## Global Constraints

- Explorer is the default page `/`; case study at `/how-it-works`; repo link `https://github.com/Masterlincs/mtg-embeddings`.
- Build output is a plain static folder (`web/build/`) servable at a site root; no server code, no analytics, no third-party scripts; images from Scryfall's CDN only.
- Every number on the site is read from `web/src/lib/generated/metrics.json`, which `mtgemb.sitedata.site_metrics` builds from records under `runs/`; each figure carries its source path. No hand-typed metrics.
- Proxy results are labelled "dev tag proxy"; the sealed test is described as "one-time"; ability and line matches are labelled exploratory.
- Python work runs on the CPU (x-2 owns the GPU). Finished releases under `web/public/data/<release_id>/` and `artifacts/releases/` are never modified; site data goes to new folders `web/public/data/site-<release_id>/`.
- Lighthouse performance ≥ 95 on `/how-it-works`; no layout shift; usable at 360 px; keyboard-usable; light and dark themes (`prefers-color-scheme` + toggle).
- `ruff check` and `pytest -q` (Python), `npm test` and `npm run build` (web) pass before each commit.

## Review Focus

1. Textless cards (`has_text: false`): no map point, no neighbours; selecting one shows a notice, never a crash. → test in Task 3 (`decodeMap` sentinel) and Task 5 (explorer state).
2. A share link with an unknown card ID or junk `ability`/filter params loads the page with a notice. → existing `parseUrl` tests kept in Task 2; unknown-ID test in Task 5.
3. Multi-face cards ("A // B", face text joined by `//`): shared-line highlighting ignores the separator and works per line. → `sharedLines` test in Task 4.
4. Site data missing for the current release (e.g. right after a model swap): the explorer and case study still work; map and before-and-after show "not available for this release". → `loadSiteData` returns `null` on 404; test in Task 2.
5. 360 px wide: no horizontal scroll on either page. → Task 8 checks with headless Chrome screenshots at 360 px and `document.documentElement.scrollWidth <= 360`.

---

### Task 1: Site data generator (Python)

**Files:**
- Create: `src/mtgemb/sitedata.py`
- Modify: `src/mtgemb/cli.py` (subcommand `site-data`)
- Modify: `src/mtgemb/encoder.py:32` (`device()` honours env `MTGEMB_DEVICE=cpu`)
- Test: `tests/test_sitedata.py`

**Interfaces:**
- Produces: `pack_map(xy: np.ndarray, card_rows: np.ndarray, n_cards: int, colors: list[str]) -> bytes` — 5 bytes per card in card-index order: int16 x, int16 y (little-endian, scaled to ±32000), uint8 WUBRG bitmask (W=1,U=2,B=4,R=8,G=16, colourless 0); textless cards get x = y = -32768.
- Produces: `region_labels(xy, tag_sets: list[frozenset], labels: dict, k: int = 36, min_support: float = 0.15) -> list[dict]` — `{"x", "y", "label", "n"}` per KMeans cluster (random_state 0), label = tag with the highest lift among tags held by ≥ `min_support` of the cluster; clusters with no such tag are dropped.
- Produces: `pretrained_neighbours(texts: list[str], ti: np.ndarray, n_cards: int, k: int = 10) -> list[list]` — per card index, `[[neighbour_index, score4], ...]` from pretrained e5-base-v2 (CPU, max_length 94), `[]` for textless; uses `export.topk_neighbours`.
- Produces: `site_metrics() -> dict` — `{"dev": {name: {"full", "hard_lexical", "source"}}, "promotion": {...report contrasts, "x1": report or null}, "sealed": {... from runs/sealed/c98bb619ea613672/report.json}, "exam": {name: {"passed", "n", "source"}}, "steps": {seed: {"2000": full, "4000": full}}, "deployed": run_id}`; raises `FileNotFoundError` naming the missing record.
- Produces CLI: `uv run python -m mtgemb.cli site-data --release <id>` → writes `web/public/data/site-<id>/{map.bin,map.json,pretrained/NNN.json}` (written to `.partial` then renamed; refuses to overwrite) and `web/src/lib/generated/metrics.json`.

- [ ] **Step 1: Write failing tests** `test_pack_map_layout_and_sentinel` (3 cards, middle textless: bytes length 15, middle x/y == -32768, colour byte for "UR" == 10, extremes map to ±32000), `test_region_labels_pick_highest_lift` (two separated blobs whose members carry tag A / tag B, plus a tag everyone has: labels are A and B, never the common tag), `test_pretrained_neighbours_alignment` (fake encode: `ti=[0,2]`, n_cards 3 → row 1 is `[]`, row 0 points to 2), `test_site_metrics_names_missing_record` (empty RUNS → `FileNotFoundError` mentioning the path).
- [ ] **Step 2: Run** `uv run pytest -q tests/test_sitedata.py` → FAIL (module missing).
- [ ] **Step 3: Implement** the four functions and the CLI. Map: PCA to 50 dims then `sklearn.manifold.TSNE(perplexity=30, init="pca", random_state=0)` on the release's `artifacts/releases/<id>/vectors.npy` (text-card rows). `pretrained_neighbours` takes an `encode` callable so the test can fake it; the CLI passes the real CPU encoder.
- [ ] **Step 4: Run** `uv run ruff check src tests && uv run pytest -q tests/test_sitedata.py` → PASS.
- [ ] **Step 5: Run the real generator in the background** (below-normal priority, `MTGEMB_DEVICE=cpu`) for release `2026-10-16-e5-base-v2-68086be1`; expected: map.bin 139,330 bytes (27,866 × 5), map.json with 20–36 regions, 109 pretrained shards, metrics.json with every key present.
- [ ] **Step 6: Commit** `sitedata.py`, `cli.py`, `encoder.py`, `tests/test_sitedata.py`, generated `metrics.json` — "Add site data generator".

### Task 2: SvelteKit scaffold, data layer, design system

**Files:**
- Modify: `web/package.json`, `web/vite.config.js`, `web/.gitignore` (add `build/`, `.svelte-kit/`)
- Create: `web/svelte.config.js`, `web/src/app.html`, `web/src/app.css`, `web/src/routes/+layout.svelte`, `web/src/routes/+layout.js`
- Move: `web/src/lib.js` → `web/src/lib/bundle.js`, `web/src/lib.test.js` → `web/src/lib/bundle.test.js`
- Create: `web/src/lib/data.js`, `web/src/lib/data.test.js`
- Delete: `web/src/App.svelte`, `web/src/main.js`, `web/src/style.css`, `web/index.html` (old Vite SPA entry)

**Interfaces:**
- Produces (`data.js`): `loadRelease(fetchFn) -> Promise<{release, names, ids}>`; `getShard(i) -> Promise<{cards, neighbours}>`; `getCard(i)`; `getAbilities(i)`; `loadSiteData(releaseId, fetchFn) -> Promise<{map: ArrayBuffer, regions} | null>` (null on 404); `getPretrained(i) -> Promise<[[idx, score]]> | null`. One module-level cache per shard kind, as in the old app.
- Produces (`app.css`): CSS custom properties `--bg --surface --ink --muted --line --accent --accent-ink --wubrg-w/u/b/r/g`, light and `[data-theme=dark]`; type scale with Fraunces (display), Inter (body), JetBrains Mono (numbers).
- `kit.files.assets = 'public'`; `adapter-static({ pages: 'build', assets: 'build', strict: true })`; `paths.relative = true`; `+layout.js` exports `prerender = true`, `trailingSlash = 'never'`.

- [ ] **Step 1:** Install `@sveltejs/kit @sveltejs/adapter-static @fontsource-variable/fraunces @fontsource-variable/inter @fontsource/jetbrains-mono` (dev deps); move lib and tests.
- [ ] **Step 2: Write failing tests** in `data.test.js`: `loadSiteData returns null on 404` (fake fetch 404), `loadSiteData returns buffer and regions` (fake fetch).
- [ ] **Step 3: Run** `npm test` → the two new tests FAIL; moved bundle tests PASS.
- [ ] **Step 4: Implement** `data.js`, layout (header: wordmark "Card Atlas", nav Explore / How it works / GitHub, theme toggle storing `localStorage.theme`; footer with release model label and attribution), `app.css`.
- [ ] **Step 5: Run** `npm test && npm run build` → PASS; `build/index.html` and `build/how-it-works.html` exist.
- [ ] **Step 6: Commit** — "Scaffold SvelteKit site and data layer".

### Task 3: Map module and canvas component

**Files:**
- Create: `web/src/lib/map.js`, `web/src/lib/map.test.js`, `web/src/lib/AtlasMap.svelte`

**Interfaces:**
- Produces: `decodeMap(buf: ArrayBuffer) -> {x: Int16Array, y: Int16Array, color: Uint8Array, n: number}`; `view(scale, tx, ty)` transform helpers `toScreen(v, x, y) -> [sx, sy]`, `toWorld(v, sx, sy)`; `buildGrid(x, y, cell = 512) -> Map<string, number[]>` skipping -32768; `hitTest(grid, x, y, wx, wy, radius) -> number` (nearest index or -1).
- `AtlasMap.svelte` props: `{map, regions, names, selected, neighbours, onselect}`; draws all points (colour by WUBRG mask, multicolour = gold, colourless = grey) to an offscreen canvas once per zoom level, overlays selection and neighbour rings; wheel/pinch zoom, drag pan, click → `onselect(index)`, hover tooltip with name; region labels shown when zoomed in enough to read; `aria-label` and a "map is a visual aid; use search" note.

- [ ] **Step 1: Write failing tests**: `decodeMap reads 5-byte records`, `buildGrid skips textless sentinel`, `hitTest returns nearest within radius and -1 outside`, `toWorld inverts toScreen`.
- [ ] **Step 2: Run** `npm test` → FAIL.
- [ ] **Step 3: Implement** `map.js` and the component.
- [ ] **Step 4: Run** `npm test` → PASS.
- [ ] **Step 5: Commit** — "Add card map".

### Task 4: Card text helpers and card components

**Files:**
- Create: `web/src/lib/text.js`, `web/src/lib/text.test.js`, `web/src/lib/CardText.svelte`, `web/src/lib/CardTile.svelte`, `web/src/lib/Compare.svelte`, `web/src/lib/Search.svelte`

**Interfaces:**
- Produces: `cardLines(card) -> {face: number, text: string}[]` (non-empty oracle lines per face); `maskNumbers(s)` (digits → `#`, same rule as Python `exam.mask_numbers`); `sharedLines(a, b) -> Set<string>` of masked, lowercased, trimmed lines present on both cards; `manaTokens(cost) -> string[]` (`"{2}{R}"` → `["2","R"]`).
- `CardText` props `{card, highlight: Set<string>, activeLine, online}`: renders faces, mana pills, rules lines; highlighted lines get a marker; lines are buttons when `online` is given (ability selection).
- `CardTile` props `{card, score, rank, highlight, oncompare, onselect}`: image (lazy, fixed aspect box, no layout shift), name, type, rules text clamped to 6 lines.
- `Compare` (native `<dialog>`): two `CardText` columns with `sharedLines` highlighted, score, "select this card" button.
- `Search` (combobox, keyboard as before, uses `searchNames`).

- [ ] **Step 1: Write failing tests**: `sharedLines matches after number masking` ("Draw 2 cards." vs "Draw 3 cards." shared), `sharedLines ignores face separators` (multi-face card), `cardLines keeps face index`, `manaTokens splits hybrid and generic`.
- [ ] **Step 2: Run** `npm test` → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** — "Add card text helpers and card components".

### Task 5: Explorer page (`/`)

**Files:**
- Create: `web/src/routes/+page.svelte`, `web/src/lib/explorer.js`, `web/src/lib/explorer.test.js`

**Interfaces:**
- Consumes: Tasks 2–4.
- Produces: `explorer.js` pure state helpers: `resolveSelection(ids, urlState) -> {index | -1, unknown: bool}`; `visibleNeighbours(saved, cardsByIndex, filters)` (wraps `filterNeighbours`).
- Layout: desktop = map (left, sticky, ~45%) + results (right); ≤ 900 px = tabs "Results | Map". Empty state: search, six example cards (chosen in Task 7, read from `metrics.json.showcase`), short line "Find Magic cards by what they do" + link to How it works. Selected state: selected card panel (image, `CardText` with clickable ability lines, tags), "Similar by: whole card / this line" chip, collapsible filters row, neighbour grid of `CardTile`s, compare dialog. URL keeps `card`, `ability`, filters (existing `serializeUrl`).

- [ ] **Step 1: Write failing tests**: `resolveSelection flags unknown id`, `resolveSelection ignores out-of-range ability`.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** page and helpers. **Step 4: Run** `npm test && npm run build` → PASS.
- [ ] **Step 5: Manual check** with headless Chrome screenshots at 1440 px and 360 px for `/?card=ae120613-97d6-4393-b39d-c3e6c076f5d6` (in `runs/webshots/`); look at them.
- [ ] **Step 6: Commit** — "Add explorer page".

### Task 6: Charts and before-and-after components

**Files:**
- Create: `web/src/lib/BarChart.svelte`, `web/src/lib/DotPlot.svelte`, `web/src/lib/BeforeAfter.svelte`, `web/src/lib/format.js`, `web/src/lib/format.test.js`

**Interfaces:**
- `format.js`: `pct(x, d = 1)`; `delta(x)` (signed, 3 dp); `scaleLinear(domain, range)`.
- `BarChart` props `{rows: {label, value, spread?: [min, max], highlight?}[], domain, unit}`: horizontal bars, value labels, seed range as a whisker; `<title>`/`<desc>` and a visually hidden table for screen readers.
- `DotPlot` props `{rows: {label, values: number[]}[]}` (seed values, e.g. 2000 vs 4000 updates per seed).
- `BeforeAfter` props `{indices: number[]}`: picker of showcase cards + search; two columns (pretrained / fine-tuned) of the top 8 with name and first rules lines; overlap count; "not available for this release" when `getPretrained` is null.

- [ ] **Step 1: Write failing tests**: `scaleLinear maps domain ends`, `delta formats sign`.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** — "Add charts and before-and-after".

### Task 7: Case study page (`/how-it-works`) and showcase cards

**Files:**
- Create: `web/src/routes/how-it-works/+page.svelte`
- Modify: `src/mtgemb/sitedata.py` (`showcase` list in metrics: six oracle IDs chosen by the lead, each with a one-line reason)

**Content (every number from `metrics.json`):** hero (pitch, `Search` that jumps to the explorer, showcase chips); The problem (Ghosts of the Innocent vs Furnace of Rath, shown as cards); The data (Scryfall + Tagger tags as labels, family-grouped fit/dev/sealed split, counts from metrics); The model (diagram: card text → e5-base-v2 → vector; contrastive pairs sharing a tag; lexical hard negatives); Before and after (`BeforeAfter`); Results (`BarChart` of dev nDCG@10 full and hard-lexical for lexical, pretrained E5, pretrained BGE, fine-tuned BGE, fine-tuned E5 with seed whiskers; promotion bounds as text; sealed one-time result; exam passes; `DotPlot` 2000 vs 4000); Limits (Deflecting Swat worked example with its shared tags; T021; proxy labels; ability matches exploratory); Built with / links.

- [ ] **Step 1:** Choose showcase cards by reading their neighbour lists (both columns readable; one honest mixed case); add to `site_metrics`; re-run `pytest -q tests/test_sitedata.py`.
- [ ] **Step 2:** Implement the page. **Step 3: Run** `npm run build` → PASS; prerendered `build/how-it-works.html` contains the hero text and chart SVG without JavaScript.
- [ ] **Step 4: Commit** — "Add case study page".

### Task 8: Optimise, check and document

**Files:**
- Create: `web/scripts/prune-build.mjs` (remove release folders other than `current.json`'s and its `site-` folder from `build/data/`)
- Modify: `web/package.json` (`"build": "vite build && node scripts/prune-build.mjs"`), `web/README.md`, `WRITE_UP.MD`

- [ ] **Step 1:** Implement prune; **Run** `npm run build` → `build/data/` holds only `current.json`, the current release and its site folder.
- [ ] **Step 2:** Serve `build/` (`npx vite preview` or `python -m http.server`), run Lighthouse (`npx lighthouse http://127.0.0.1:4173/how-it-works --only-categories=performance,accessibility --chrome-flags=--headless`) → performance ≥ 95, accessibility ≥ 95; fix and re-run if not.
- [ ] **Step 3:** Headless screenshots at 1440 and 360 px of `/`, `/?card=…`, `/how-it-works`, light and dark; check `scrollWidth <= 360` via `--dump-dom` script or Lighthouse's viewport audit.
- [ ] **Step 4:** README: how to build, regenerate site data, deploy the static folder to a VPS (any static server; SPA fallback not needed since pages are prerendered).
- [ ] **Step 5: Commit** — "Optimise and document site build".

### Task 9 (after reads out): follow the deployed model

- [ ] If promotes: export the new release with `status="promoted"` and a decision string that names and says no sealed run (the current `export()` hard-codes the text: add a `decision` argument, test it), run `site-data` for it, point `current.json` at it, rebuild. If not: nothing changes on the site; metrics show the x-1 confirmation as dev-proxy evidence only.
