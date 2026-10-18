import gzip
import json

import pandas as pd

from mtgemb import fresh
from mtgemb.util import sha256_file


def _card(oid, name, text, released="2026-01-01", layout="normal"):
    return {"oracle_id": oid, "name": name, "layout": layout, "games": ["paper"],
            "legalities": {"vintage": "legal"}, "oracle_text": text, "type_line": "Instant",
            "cmc": 1.0, "color_identity": [], "released_at": released}


def _raw(d, snapshot, cards, tags):
    d.mkdir()
    files = {}
    for key, rows in (("oracle_cards", cards), ("oracle_tags", tags)):
        p = d / f"{key}.jsonl.gz"
        with gzip.open(p, "wt", encoding="utf-8") as f:
            f.write("".join(json.dumps(r) + "\n" for r in rows))
        files[key] = {"path": p.name, "sha256": sha256_file(p)}
    (d / "manifest.json").write_text(json.dumps({"snapshot": snapshot, "files": files}))
    return d


def test_collect_keeps_only_new_settled_corpus_cards(tmp_path):
    old = [_card("o1", "Shock", "Shock deals 2 damage to any target."),
           _card("o2", "Art", "x", layout="art_series")]
    new = old + [
        _card("n1", "Zap", "Zap deals 2 damage to any target."),  # same template as Shock
        _card("n2", "Fresh Idea", "Draw a card."),
        _card("n3", "Too New", "Draw two cards.", released="2026-10-01"),
        _card("n4", "Art Two", "x", layout="art_series")]
    tags = [{"type": "oracle", "id": "burn", "taggings": [{"oracle_id": "n1"},
                                                          {"oracle_id": "o1"}]},
            {"type": "oracle", "id": "draw", "taggings": [{"oracle_id": "n2"},
                                                          {"oracle_id": "n3"}]},
            {"type": "oracle", "id": "cycle-xyz", "taggings": [{"oracle_id": "n2"}]}]
    ds = {"cards": pd.DataFrame({"oracle_id": ["o1"], "name": ["Shock"],
                                 "text": ["Shock deals 2 damage to any target."],
                                 "faces": [json.dumps([{"name": "Shock"}])],
                                 "family_id": ["famShock"]}),
          "tags": pd.DataFrame({"tag_id": ["burn", "draw", "cycle-xyz"],
                                "functional": [True, True, False]})}
    cards, mem, rep = fresh.collect(ds, _raw(tmp_path / "a", "2026-10-16", old, tags),
                                    _raw(tmp_path / "b", "2026-10-30", new, tags))
    assert cards["oracle_id"].tolist() == ["n1", "n2"]
    assert rep["new_ids"] == 4 and rep["excluded_by_policy"] == 1 and rep["too_recent"] == 1
    # Zap matches Shock's name-normalised text, so it joins Shock's family
    assert dict(zip(cards["oracle_id"], cards["family_id"], strict=True)) == {
        "n1": "famShock", "n2": "n2"}
    assert set(cards["split"]) == {"fresh"}
    # only fresh cards' functional, frozen-taxonomy tags
    assert sorted(map(tuple, mem.to_numpy())) == [("n1", "burn"), ("n2", "draw")]
