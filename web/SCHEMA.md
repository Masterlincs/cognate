# Release bundle schemas `mtgemb-release-v2` and `mtgemb-release-v3`

The static app reads one release directory, `<base>/data/<release_id>/`. The exporter writes release bundles. Fixture bundles set `"fixture": true`; real releases set it to `false`. All files are UTF-8 JSON. Card order is fixed by `ids.json.ids` (Oracle ID ascending), and card index `i` is its position in that list. The shard number is `floor(i / shard_size)`.

The app accepts v2 and v3. A v2 bundle loads with the whole-card neighbour feature and no ability control. Version 1 and other schemas are unsupported. A failed bundle shows an error; the app never substitutes fixture data.

## Loading order (v2 and v3)

The app loads `current.json`, then `release.json`, then `names.json`. Lookup becomes available as soon as `names.json` passes validation. The larger `ids.json` is fetched after names; card selection, share-link resolution and shard ID checks wait for it. Card and neighbour shards, plus the selected card's v3 ability shard, load on demand and are cached by shard.

## Common files (v2 and v3)

### `release.json`

```json
{
  "schema": "mtgemb-release-v2",
  "release_id": "2026-10-16-e5ft-ab12cd34",
  "fixture": false,
  "snapshot_date": "2026-10-16",
  "created_utc": "2026-10-20T12:00:00Z",
  "model": {"label": "fine-tuned e5-base-v2", "status": "promoted", "backbone": "intfloat/e5-base-v2", "run_id": "confirm-e5-s1", "decision": "provisional tag-proxy promotion on selection-exposed development"},
  "n_cards": 27845, "k": 50, "shard_size": 256,
  "files": {"names.json": {"sha256": "...", "bytes": 123}, "ids.json": {"sha256": "...", "bytes": 123}, "cards/000.json": {"sha256": "...", "bytes": 1}, "neighbours/000.json": {"sha256": "...", "bytes": 1}},
  "attribution": "Card data and images © Wizards of the Coast; data from Scryfall (scryfall.com). Not affiliated with or endorsed by Wizards of the Coast or Scryfall. Tags from Scryfall Tagger."
}
```

`files` lists `names.json`, `ids.json`, and every card and neighbour shard. Each entry has the file's SHA-256 and byte count.

### `names.json` and `ids.json`

```json
{"names": ["Card Name", "A // B"], "aliases": [["Face Name", 1]]}
{"ids": ["oracle-id-0", "oracle-id-1"]}
```

`names[i]` is the card's full name; multi-face cards use `"A // B"`. `aliases` maps face names to card indexes. `names.json` has no IDs. IDs are in card-index order and are unique, non-empty and strictly ascending.

### `cards/NNN.json` and `neighbours/NNN.json`

Each shard is an array of up to `shard_size` elements aligned to card indexes `NNN*shard_size ...`:

```json
{"id": "oracle-id", "name": "A // B", "layout": "transform",
 "faces": [{"name": "A", "mana_cost": "{1}{R}", "type_line": "Creature - Human", "oracle_text": "..."}],
 "color_identity": "R", "mv": 2.0, "type_line": "Creature - Human // Creature - Werewolf",
 "is_creature": true, "image": null, "scryfall_uri": "https://...", "has_text": true}
```

`card.id` equals `ids.ids[index]`. `color_identity` is a WUBRG-ordered string or `""`. `has_text: false` marks a textless card. Such a card has no neighbours; the UI says that no neighbours were computed.

The matching element in `neighbours/NNN.json` is a list of up to `k` `[neighbour_index, score]` pairs. The list excludes the card itself. It is ranked by full-precision cosine descending, then Oracle ID ascending. Stored scores are rounded to four decimals; equal rounded scores need not be in Oracle ID order. A score is a ranking score, not a probability. Textless cards have `[]`.

## Additional v3 fields and shards

Version 3 keeps the v2 fields and files and adds `ability_k`, `abilities_status`, and every `abilities/NNN.json` shard to `files`:

```json
"schema": "mtgemb-release-v3",
"ability_k": 50,
"abilities_status": "exploratory: not evaluated on the tag proxy"
```

Each `abilities/NNN.json` is an array aligned with its card shard. An element is `[]` if the card has no rules text or fewer than two abilities; otherwise it is an ordered list with at least two entries:

```json
[
  {"text": "When this enters, draw a card.", "neighbours": [[17, 0.9123], [22, 0.8871]]},
  {"text": "Whenever you cast a spell, gain 1 life.", "neighbours": [[41, 0.9054], [17, 0.8792]]}
]
```

An ability is one non-empty line of the model input text (`no_reminder_v1`); face separators are not abilities. Each entry's neighbour list has up to `ability_k` pairs and excludes its card. Candidate cards are text cards and are ranked by the best matching ability line. Scores descend; because scores are rounded to four decimals, equal rounded scores need not be in index order. Indexes are in `[0, n_cards)`. Textless cards must have `[]`. Ability matches are exploratory and have not been evaluated on the tag proxy.

The ability index is stored in the `ability` URL parameter beside `card`. Missing, invalid or out-of-range values select Whole card.

## Validation

- v2 and v3 require positive `n_cards` and `shard_size`; v1 is unsupported.
- `names.names.length == n_cards`, aliases contain valid indexes, and `names.json` has no `ids` key.
- `ids.ids.length == n_cards`; IDs are non-empty, unique and strictly ascending.
- Shard lengths match the expected length (the final shard may be short), and each card ID matches its indexed ID.
- Neighbour and ability indexes are in `[0, n_cards)` and never refer to the source card. Neighbour scores are finite and descending.
- For v3, `ability_k` and `abilities_status` have the values above; the manifest includes every common shard and every ability shard with SHA-256 and byte count. Ability shard lengths align with card shards, entries have non-empty text and valid neighbour lists, and non-empty ability lists have at least two entries.
- Textless cards have no neighbours and, in v3, no ability entries.
- The bundle checker verifies hashes and byte counts for every manifest file required by the schema.

Any validation or loading failure displays a visible bundle error. Fixture data is never used as a fallback.

The checker validates `current.json` when it points to the bundle being checked. It can also validate a standalone release directory while `current.json` points to a different installed release.
