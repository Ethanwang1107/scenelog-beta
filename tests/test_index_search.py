from pathlib import Path

from scenelog.indexer import (
    build_identity_index,
    build_index,
    build_visual_index,
)
from scenelog.search import Searcher


def test_empty_transcript_removes_old_audio_records(tmp_path: Path):
    srt = tmp_path / "clip.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\n旧对白\n\n",
        encoding="utf-8",
    )
    build_index(srt, tmp_path, "mat_same", "clip.mov", "clip.mov")

    srt.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n[未识别出语音内容]\n\n",
        encoding="utf-8",
    )
    build_index(srt, tmp_path, "mat_same", "clip.mov", "clip.mov")

    assert Searcher(tmp_path, tmp_path).search("旧对白") == []


def test_visual_descriptions_are_searchable_with_timestamps(tmp_path: Path):
    build_visual_index(
        tmp_path,
        "mat_visual",
        "C001.MOV",
        "card-a/C001.MOV",
        [3.2, 9.8],
        ["院子里一名男子劈柴", "木柴堆放在墙边"],
    )

    results = Searcher(tmp_path, tmp_path).search("劈柴")

    assert len(results) == 1
    assert results[0]["rel_path"] == "card-a/C001.MOV"
    assert results[0]["start_time"] == "00:00:03"
    assert results[0]["source"] == "vision"


def test_updating_visual_records_does_not_remove_audio_records(tmp_path: Path):
    srt = tmp_path / "clip.srt"
    srt.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\n有人说准备出发\n\n",
        encoding="utf-8",
    )
    build_index(srt, tmp_path, "mat_same", "clip.mov", "clip.mov")
    build_visual_index(
        tmp_path,
        "mat_same",
        "clip.mov",
        "clip.mov",
        [4.0],
        ["众人在院门口整理行李"],
    )
    build_visual_index(
        tmp_path,
        "mat_same",
        "clip.mov",
        "clip.mov",
        [5.0],
        ["众人背着行李离开院子"],
    )

    searcher = Searcher(tmp_path, tmp_path)
    assert len(searcher.search("准备出发")) == 1
    assert searcher.search("整理行李") == []
    assert len(searcher.search("离开院子")) == 1


def test_identity_events_are_searchable_by_name_action_and_dialogue(tmp_path: Path):
    build_identity_index(
        tmp_path,
        "mat_identity",
        "C001.MOV",
        "card-a/C001.MOV",
        [
            {
                "event_id": "event_001",
                "start_timestamp": 1.2,
                "end_timestamp": 5.8,
                "people_ids": ["person_0001"],
                "people": [{"id": "person_0001", "name": "王书记"}],
                "description": {
                    "动作": "王书记带着两名男子走在乡间小路上",
                    "动作事件": [
                        {
                            "施动者": ["两名黑衣人"],
                            "动作": "控制并押走",
                            "承受者": ["老刘"],
                            "依据": "连续画面中两人控制老刘并向前移动",
                            "置信度": 0.93,
                        }
                    ],
                    "场景": "背景是蓝天和山脉",
                    "对白事实": "有人夸赞王书记的冲锋衣价值高",
                },
            }
        ],
    )

    result = Searcher(tmp_path, tmp_path).search("王书记 冲锋衣")[0]

    assert result["source"] == "identity"
    assert result["start_time"] == "00:00:01"
    assert result["end_time"] == "00:00:06"
    assert "走在乡间小路上" in result["text"]

    action_result = Searcher(tmp_path, tmp_path).search("老刘 押走")[0]

    assert action_result["source"] == "identity"
    assert action_result["material_id"] == "mat_identity"
    assert "两名黑衣人" in action_result["text"]
