# Release bundle schema `mtgemb-release-v2`

The static app reads one release directory, `<base>/data/<release_id>/`. The exporter (`mtgemb export`, P8) writes it. Fixtures written by `web/scripts/make-fixture.mjs` follow the same schema and set `"fixture": true`. A real release has `"fixture": false`.

All files are UTF-8 JSON. Card order is fixed by `ids.json.ids` (Oracle ID ascending). A card's integer index `i` is its position in that order. A card's shard number is `floor(i / shard_size)`.

## Loading order

The app loads `current.json`, then the selected `release.json`. It validates the v2 schema and fetches `names.json`, then `ids.json` once the names have arrived, so the larger, poorly compressing Oracle ID list does not share bandwidth with the lookup index. Name lookup becomes available as soon as `names.json` passes validation; it does not wait for `ids.json`. Card selection, share-link resolution and shard ID checks wait for `ids.json` validation. A failed or unsupported bundle displays the bundle error; fixture data is never substituted.

## `release.json` (loaded first)

```json
{
  "schema": "mtgemb-release-v2",
  "release_id": "2026-10-16-e5ft-ab12cd34",
  "fixture": false,
  "snapshot_date": "2026-10-16",
  "created_utc": "2026-10-20T12:00:00Z",
  "model": {"label": "fine-tuned e5-base-v2 (initial-1, seed 1)", "status": "promoted | experimental | pretrained-fallback",
            "backbone": "intfloat/e5-base-v2", "run_id": "confirm-1-...", "decision": "provisional tag-proxy promotion on selection-exposed development"},
  "n_cards": 27845, "k": 50, "shard_size": 256,
  "files": {"names.json": {"sha256": "...", "bytes": 123}, "ids.json": {"sha256": "...", "bytes": 123}, "cards/000.json": {"sha256": "...", "bytes": 1}, "neighbours/000.json": {"sha256": "...", "bytes": 1}},
  "attribution": "Card data and images © Wizards of the Coast; data from Scryfall (scryfall.com). Not affiliated with or endorsed by Wizards of the Coast or Scryfall. Tags from Scryfall Tagger."
}
```

`files` lists `names.json`, `ids.json`, and every card and neighbour shard. Each entry contains the file's SHA-256 and byte count.

## `names.json` (small startup lookup index)

```json
{"names": ["Card Name", "A // B"], "aliases": [["Face Name", 1], ["Other Face", 1]]}
```

`names[i]` is the card's full name: multi-face cards use `"A // B"`. `aliases` maps extra lookup strings (individual face names) to card indexes. This file has no IDs.

## `ids.json` (identity index, fetched after `names.json`)

```json
{"ids": ["oracle-id-0", "oracle-id-1", "..."]}
```

The list is in card-index order, has exactly `n_cards` entries, and contains unique, strictly ascending strings. Cards and neighbours use these indexes.

## `cards/NNN.json` (on demand, NNN = zero-padded shard, 3 digits)

An array of up to `shard_size` card objects for indexes `NNN*shard_size ...`:

```json
{"id": "oracle-id", "name": "A // B", "layout": "transform",
 "faces": [{"name": "A", "mana_cost": "{1}{R}", "type_line": "Creature - Human", "oracle_text": "..."}],
 "color_identity": "R", "mv": 2.0, "type_line": "Creature - Human // Creature - Werewolf",
 "is_creature": true, "image": "https://... or null", "scryfall_uri": "https://...",
 "has_text": true}
```

`card.id` must equal `ids.json.ids[index]`. `color_identity` is a WUBRG-ordered string, or `""` for colourless. `has_text: false` marks a textless card. Such a card has no neighbours, and the UI says so rather than showing "no similar cards".

## `neighbours/NNN.json` (on demand, same sharding)

An array aligned with the card shard. Element `j` is the neighbour list of card `NNN*shard_size + j`: up to `k` pairs `[neighbour_index, score]`, ordered by full-precision cosine descending, then by Oracle ID ascending. The stored score is that cosine rounded to 4 decimals, so equal rounded scores need not be in Oracle ID order: a ranking score, not a probability. The list never contains the card itself. A textless card has `[]`.

## Validation

- `schema` equals `mtgemb-release-v2`; v1 bundles are unsupported.
- `names.names.length == n_cards`, aliases contain valid indexes, and `names.json` has no `ids` key.
- `ids.ids.length == n_cards`; IDs are non-empty, unique and strictly ascending.
- Each shard's length matches the expected length (the last shard may be short), and every card ID matches its indexed ID.
- Neighbour indexes are in `[0, n_cards)`.
- The bundle checker verifies the hashes and byte counts listed in `release.json`.

On any failure, show a visible error. Never fall back to fixture data.
