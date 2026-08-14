from pathlib import Path

import numpy as np
import soundfile as sf

from scenelog.people import PeopleStore
from scenelog.speaker import (
    match_transcript_speakers,
    parse_srt_precise,
    write_speaker_transcript,
)


class _FakeSpeakerEngine:
    def __init__(self, embeddings):
        self.embeddings = iter(embeddings)

    def extract(self, *_args, **_kwargs):
        return next(self.embeddings)


def _srt(path: Path) -> Path:
    path.write_text(
        "1\n00:00:00,000 --> 00:00:03,840\n第一句对白\n\n"
        "2\n00:00:03,840 --> 00:00:12,040\n第二句对白\n\n",
        encoding="utf-8",
    )
    return path


def test_parse_srt_precise_keeps_milliseconds(tmp_path: Path):
    segments = parse_srt_precise(_srt(tmp_path / "clip.srt"))

    assert segments[0]["start_seconds"] == 0.0
    assert segments[0]["end_seconds"] == 3.84
    assert segments[1]["start_seconds"] == 3.84
    assert segments[1]["end_seconds"] == 12.04


def test_speaker_matching_requires_threshold_and_margin(tmp_path: Path):
    srt_path = _srt(tmp_path / "clip.srt")
    audio_path = tmp_path / "clip.wav"
    sf.write(audio_path, np.zeros(13 * 16000, dtype=np.float32), 16000)
    profiles = [
        {
            "id": "person_0001",
            "name": "老刘",
            "voice_embeddings": [[1.0, 0.0]],
        },
        {
            "id": "person_0002",
            "name": "王书记",
            "voice_embeddings": [[0.0, 1.0]],
        },
    ]
    engine = _FakeSpeakerEngine(
        [
            [0.99, 0.01],
            [0.72, 0.69],
        ]
    )

    segments = match_transcript_speakers(
        audio_path,
        srt_path,
        profiles,
        engine,
        threshold=0.35,
        margin=0.08,
        single_threshold=0.55,
    )

    assert segments[0]["speaker_name"] == "老刘"
    assert segments[0]["confirmed"] is True
    assert segments[1]["speaker_name"] == ""
    assert segments[1]["confirmed"] is False
    assert segments[1]["reason"] == "候选人物分差不足"


def test_single_registered_voice_uses_stricter_threshold(tmp_path: Path):
    srt_path = _srt(tmp_path / "clip.srt")
    audio_path = tmp_path / "clip.wav"
    sf.write(audio_path, np.zeros(13 * 16000, dtype=np.float32), 16000)
    profiles = [
        {
            "id": "person_0001",
            "name": "老刘",
            "voice_embeddings": [[1.0, 0.0]],
        }
    ]
    engine = _FakeSpeakerEngine([[0.52, 0.85], [0.7, 0.1]])

    segments = match_transcript_speakers(
        audio_path,
        srt_path,
        profiles,
        engine,
        threshold=0.45,
        margin=0.1,
        single_threshold=0.55,
    )

    assert segments[0]["confirmed"] is False
    assert segments[0]["reason"] == "声纹相似度不足"
    assert segments[1]["confirmed"] is True


def test_named_transcript_keeps_unknown_segments(tmp_path: Path):
    srt_path, txt_path = write_speaker_transcript(
        [
            {
                "start_seconds": 0.0,
                "end_seconds": 3.84,
                "text": "第一句对白",
                "speaker_name": "老刘",
                "confirmed": True,
            },
            {
                "start_seconds": 3.84,
                "end_seconds": 12.04,
                "text": "第二句对白",
                "speaker_name": "",
                "confirmed": False,
            },
        ],
        tmp_path,
        "mat_test",
    )

    assert "老刘：第一句对白" in srt_path.read_text(encoding="utf-8")
    assert "未知说话人：第二句对白" in txt_path.read_text(encoding="utf-8")


def test_people_v2_migrates_voice_fields_without_dropping_face_data(tmp_path: Path):
    (tmp_path / "people.json").write_text(
        """{
          "version": 2,
          "mode": "registered_only",
          "next_id": 2,
          "people": {
            "person_0001": {
              "id": "person_0001",
              "name": "老刘",
              "reference_photos": ["people/person_0001/references/a.png"],
              "reference_embeddings": [[1.0, 0.0]],
              "embedding": [1.0, 0.0],
              "occurrences": []
            }
          },
          "events": {}
        }""",
        encoding="utf-8",
    )

    store = PeopleStore(tmp_path)
    profile = store.list_people()[0]

    assert store.data["version"] == 3
    assert profile["reference_count"] == 1
    assert profile["voice_count"] == 0
    assert profile["voice_duration"] == 0.0
