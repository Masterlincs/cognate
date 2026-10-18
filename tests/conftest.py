import pytest

from mtgemb import util


@pytest.fixture(autouse=True)
def _isolated_logs(tmp_path, monkeypatch):
    """log_event writes via util.RUNS: keep test events out of the real runs/*.jsonl."""
    monkeypatch.setattr(util, "RUNS", tmp_path / "runs-log")


@pytest.fixture
def selfname_cards():
    import json

    import pandas as pd

    from mtgemb.text import no_reminder_v1

    names = ["Hero", "Ally", "Trap", "Other", "Empty", "Broken", "Front // Back"]
    texts = ["Flying\nHero draws two cards.", "Ally gains 2 life.", "Trap loses 2 life.",
             'Create a token named Other.', "(Reminder)", "Broken (reminder",
             "Front gains 2 life."]
    rows = []
    for i, (name, text) in enumerate(zip(names, texts, strict=True)):
        faces = [{"name": name, "oracle_text": text, "type_line": "Creature"}]
        if i == 6:
            faces[0]["name"] = "Front"
            faces.append({"name": "Back", "oracle_text": "Back draws two cards.",
                          "type_line": "Creature"})
        v1 = no_reminder_v1([f["oracle_text"] for f in faces])
        rows.append({"oracle_id": f"00000000-0000-0000-0000-{i:012d}", "name": name,
                     "text": v1["text"], "text_status": v1["status"],
                     "faces": json.dumps(faces), "split": "dev" if i == 0 else "fit",
                     "family_id": f"f{i}"})
    return pd.DataFrame(rows, index=range(10, 17))
