# Data card: MTG card corpus, splits and tag-proxy labels

Dataset `cbd5c12011c94e68`, processed from the Scryfall bulk snapshot of 2026-10-16 (`data/processed/2026-10-16_no_reminder_v1/dataset.json`). The data is not published; this repository holds only the code that rebuilds it.

## Sources and terms

- **Cards:** Scryfall bulk `oracle_cards` (one record per Oracle ID), downloaded 2026-10-16. The raw manifest hash is recorded in `dataset.json`.
- **Labels:** Scryfall Tagger `oracle_tags`, community-made functional tags (for example "removal" or "card draw").
- **Terms**, as recorded at acquisition:
  - Scryfall data is free for research and Magic software, with no paywalling, no plain repackaging or republishing, and no implied Scryfall endorsement.
  - Card text and images are © Wizards of the Coast, used under the Wizards of the Coast Fan Content Policy.
  - Not affiliated with or endorsed by Wizards of the Coast or Scryfall.

## Corpus

One row per Oracle ID. A card is included if it exists in paper, its vintage legality is not `not_legal`, and its layout is not a token, double-faced token, emblem or art card. This leaves out digital-only cards, unreleased previews, planes, schemes, vanguards and non-tournament novelty cards.

| Count | Value |
| --- | --- |
| Cards included | 31,066 |
| Excluded by policy | not paper 1,950; not tournament-legal 2,398; art series 2,244; tokens 926; double-faced tokens 80; emblems 89 |
| Cards with usable rules text | 30,745 (321 have none: 309 no rules text, 12 only parenthetical text) |
| Multi-face cards | 847 |
| Functional tags | 2,709 of 4,579 (cycle, metadata, errata, printing, flavour, name, set-relative and Un-set tags excluded) |
| Functional tag memberships | 175,457 |
| Card families | 28,999 (cards linked by identical text, self-name templates or meld) |

**Text representation** (`no_reminder_v1`): the Oracle rules text with every balanced parenthetical span removed (this removes reminder text, but also any parenthetical that matters to the rules), and faces joined by `\n//\n`.

## Splits and leakage controls

- **Families:** cards are grouped into families (identical text, self-name templates, meld pairs). A whole family goes to one split, so near-copies never straddle splits.
- **Assignment:** each family is assigned by a hash of its smallest Oracle ID and the seed `mtgemb-split-v1`. The targets are 80% fit, 10% development and 10% sealed.
- **Counts:**

| Split | Cards | With text | Use |
| --- | --- | --- | --- |
| Fit | 24,817 | 24,554 | training pairs, lexical fitting, tag statistics |
| Development | 3,049 | 3,019 | model and checkpoint selection; reported results (selection-exposed) |
| Sealed | 3,200 | — | one evaluation of the earlier model; never used for training, selection or any release |

- **Catalogue:** the released catalogue holds the fit and development cards only (27,866 cards). Sealed cards are not in any release.

## Labels (tag proxy)

- **Relevance:** a candidate card's relevance to an anchor is graded by the functional tags they share.
- **Benchmark:** benchmark `8bf2f48c0bba3b2c` (v3) has 3,016 development anchors in 2,853 families.
- **Not human judgements:** these labels are a proxy. They are community tags, not judgements of whether two cards play alike.
- **Known bias:** tags on cost or casting riders (for example alternative costs) count as much as main-effect tags. That lets rider matches outrank main-effect matches.

## Training pairs (`pairs_v1`, id `795ebb1ae7ce8251`)

- **How pairs are made:** pairs of fit cards that share a tag, built tag by tag:
  - tags held by 2 to 500 fit cards;
  - 3 seeded random matchings per tag;
  - at most 24 pairs per card and 48 per family;
  - no pairs within the same family.
- **Size:** 108,220 pairs from 1,853 eligible tags, covering 23,345 fit cards.
- **Hard negatives:** drawn from char 3–5 TF-IDF lexical neighbours that share no tag with the anchor.
- **Audit:** an automated audit of 120 pairs and 45 negatives passed its structural checks. Semantic quality is **unverified**.

## Human labels

None. A blinded pool of 1,256 items is prepared but unlabelled.
