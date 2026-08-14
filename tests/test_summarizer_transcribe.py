from pathlib import Path

import pytest

from scenelog.summarizer import _split_text
from scenelog.transcribe import transcribe_many_resilient


def test_whisper_style_single_newlines_are_split_into_chunks():
    text = "\n".join(f"第{index}段对白内容" for index in range(1000))

    chunks = _split_text(text, max_chunks=20)

    assert 1 < len(chunks) <= 20
    assert max(len(chunk) for chunk in chunks) < len(text)


def test_resilient_transcription_isolates_bad_item(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[list[str]] = []

    def fake_transcribe_many(items, output_dir):
        keys = [output_key for _, output_key in items]
        calls.append(keys)
        if "bad" in keys:
            raise RuntimeError("bad audio")
        return []

    monkeypatch.setattr(
        "scenelog.transcribe.transcribe_many",
        fake_transcribe_many,
    )
    items = [
        (tmp_path / "good-a.wav", "good-a"),
        (tmp_path / "bad.wav", "bad"),
        (tmp_path / "good-b.wav", "good-b"),
    ]

    successes, failures = transcribe_many_resilient(
        items,
        tmp_path,
        batch_size=3,
    )

    assert successes == ["good-a", "good-b"]
    assert failures == {"bad": "bad audio"}
    assert calls[0] == ["good-a", "bad", "good-b"]
