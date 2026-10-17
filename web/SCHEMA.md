# Release bundle schema `mtgemb-release-v1`

The static app reads one release directory, `<base>/data/<release_id>/`. The exporter (`mtgemb export`, P8) writes it. Fixtures written by `web/scripts/make-fixture.mjs` follow the same schema and set `"fixture": true`. A real release has `"fixture": false`.

All files are UTF-8 JSON. Card order is fixed by `names.json.ids` (Oracle ID ascending). A card's integer index `i` is its position in that order. A card's shard number is `floor(i / shard_size)`.

## `release.json` (loaded first)

```json
{
  "schema": "mtgemb-release-v1",
  "release_id": "2026-10-16-e5ft-ab12cd34",
  "fixture": false,
  "snapshot_date": "2026-10-16",
  "created_utc": "2026-10-20T12:00:00Z",
  "model": {"label": "fine-tuned e5-base-v2 (initial-1, seed 1)", "status": "promoted | experimental | pretrained-fallback",
            "backbone": "intfloat/e5-base-v2", "run_id": "confirm-1-...", "decision": "provisional tag-proxy promotion on selection-exposed development"},
  "n_cards": 27845, "k": 50, "shard_size": 256,
  "files": {"names.json": {"sha256": "...", "bytes": 123}, "cards/000.json": {"sha256": "...", "bytes": 1}, "neighbours/000.json": {"sha256": "...", "bytes": 1}},
  "attribution": "Card data and images © Wizards of the Coast; data from Scryfall (scryfall.com). Not affiliated with or endorsed by Wizards of the Coast or Scryfall. Tags from Scryfall Tagger."
}
```

## `names.json` (loaded at startup)

```json
{"ids": ["oracle-id-0", "..."], "names": ["Card Name", "..."],
 "aliases": [["Face Name", 17], ["Other Face", 17]]}
```

`names[i]` is the card's full name: multi-face cards use `"A // B"`. `aliases` maps extra lookup strings (individual face names) to card indexes.

## `cards/NNN.json` (on demand, NNN = zero-padded shard, 3 digits)

An array of up to `shard_size` card objects for indexes `NNN*shard_size ...`:

```json
{"id": "oracle-id", "name": "A // B", "layout": "transform",
 "faces": [{"name": "A", "mana_cost": "{1}{R}", "type_line": "Creature — Human", "oracle_text": "..."}],
 "color_identity": "R", "mv": 2.0, "type_line": "Creature — Human // Creature — Werewolf",
 "is_creature": true, "image": "https://... or null", "scryfall_uri": "https://...",
 "has_text": true}
```

`color_identity` is a WUBRG-ordered string, or `""` for colourless. `has_text: false` marks a textless card. Such a card has no neighbours, and the UI says so rather than showing "no similar cards".

## `neighbours/NNN.json` (on demand, same sharding)

An array aligned with the card shard. Element `j` is the neighbour list of card `NNN*shard_size + j`: up to `k` pairs `[neighbour_index, score]`, ordered by score descending, then by Oracle ID ascending. The score is a cosine rounded to 4 decimals: a ranking score, not a probability. The list never contains the card itself. A textless card has `[]`.

## Validation the app must perform

- `schema` equals `mtgemb-release-v1`.
- `names.ids.length == names.names.length == n_cards`.
- Each shard's length matches the expected length (the last shard may be short).
- Neighbour indexes are in `[0, n_cards)`.

On any failure, show a visible error. Never fall back to fixture data. Hashes are checked by the release test script, not by the browser.
