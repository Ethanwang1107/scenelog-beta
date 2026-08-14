import json
from pathlib import Path

import pytest

from scenelog.identity_summary import (
    _dialogue_for_event,
    _ensure_grounded_action_facts,
    _normalize_event_description,
    _transcript_segments,
    describe_identity_events,
    generate_identity_summary,
)


class _Response:
    def __init__(self, payload: dict):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return {"response": json.dumps(self.payload, ensure_ascii=False)}


def _event(frame: str = "people/events/mat_test/event_001.jpg") -> dict:
    return {
        "event_id": "event_001",
        "start_timestamp": 1.0,
        "end_timestamp": 4.0,
        "people_ids": ["person_0001", "person_0002"],
        "people": [
            {"id": "person_0001", "name": "王书记"},
            {"id": "person_0002", "name": "老刘"},
        ],
        "labels": [
            {"tag": "P1", "person_id": "person_0001", "name": "王书记"},
            {"tag": "P2", "person_id": "person_0002", "name": "老刘"},
        ],
        "unknown_face_count": 1,
        "frame": frame,
    }


def test_srt_dialogue_is_aligned_to_identity_event_with_margin(tmp_path: Path):
    srt_path = tmp_path / "clip.srt"
    srt_path.write_text(
        "1\n00:00:00,000 --> 00:00:03,000\n第一句对白\n\n"
        "2\n00:00:05,000 --> 00:00:08,000\n第二句对白\n\n"
        "3\n00:00:20,000 --> 00:00:22,000\n无关对白\n\n",
        encoding="utf-8",
    )

    segments = _transcript_segments(srt_path)

    assert _dialogue_for_event(segments, 2.0, 5.0) == "第一句对白 第二句对白"
    assert _dialogue_for_event(segments, 10.0, 12.0) == ""


def test_confirmed_voice_overrides_visual_speaker_guess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    frame = tmp_path / "people/events/mat_test/event_001.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"jpeg")

    monkeypatch.setattr(
        "scenelog.identity_summary.requests.post",
        lambda *_args, **_kwargs: _Response(
            {
                "动作": "两人在路上交谈",
                "对白事实": "有人夸赞冲锋衣价值高",
                "说话人": "王书记",
                "说话人已确认": True,
            }
        ),
    )

    described = describe_identity_events(
        [_event()],
        tmp_path,
        None,
        "test-vlm",
        speaker_segments=[
            {
                "start_seconds": 0.0,
                "end_seconds": 3.84,
                "text": "领导，你这件冲锋衣都够我一年收入了呀",
                "person_id": "person_0002",
                "speaker_name": "老刘",
                "match_score": 0.81,
                "match_margin": 0.22,
                "confirmed": True,
            }
        ],
    )

    description = described[0]["description"]
    assert description["说话人"] == "老刘"
    assert description["说话人已确认"] is True
    assert described[0]["speaker_match"]["match_score"] == 0.81


def test_speaker_overlap_prefers_dialogue_covering_most_of_event():
    from scenelog.identity_summary import _confirmed_voice_for_event

    result = _confirmed_voice_for_event(
        [
            {
                "start_seconds": 0.0,
                "end_seconds": 3.84,
                "speaker_name": "老刘",
                "text": "第一句",
                "match_score": 0.81,
                "confirmed": True,
            },
            {
                "start_seconds": 3.84,
                "end_seconds": 12.04,
                "speaker_name": "王书记",
                "text": "第二句",
                "match_score": 0.79,
                "confirmed": True,
            },
        ],
        4.0,
        7.0,
    )

    assert result["speaker_name"] == "王书记"
    assert result["text"] == "第二句"


def test_identity_summary_deduplicates_voice_fact_across_visual_events(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = {}

    def fake_post(_url, json, **_kwargs):
        captured["prompt"] = json["prompt"]
        return _Response(
            {
                "摘要": "老刘说第一句对白，王书记说第二句对白。",
                "关键词": [],
            }
        )

    monkeypatch.setattr("scenelog.identity_summary.requests.post", fake_post)
    events = [
        {
            "people": [{"id": "person_0001", "name": "老刘"}],
            "speaker_match": {
                "speaker_name": "老刘",
                "text": "第一句对白",
            },
            "description": {
                "对白事实": "第一句对白",
                "说话人": "老刘",
                "说话人已确认": True,
                "动作事件": [],
            },
        },
        {
            "people": [{"id": "person_0001", "name": "老刘"}],
            "speaker_match": {
                "speaker_name": "老刘",
                "text": "第一句对白",
            },
            "description": {
                "对白事实": "第一句对白",
                "说话人": "老刘",
                "说话人已确认": True,
                "动作事件": [],
            },
        },
        {
            "people": [{"id": "person_0002", "name": "王书记"}],
            "speaker_match": {
                "speaker_name": "王书记",
                "text": "第二句对白",
            },
            "description": {
                "对白事实": "第二句对白",
                "说话人": "王书记",
                "说话人已确认": True,
                "动作事件": [],
            },
        },
    ]

    result = generate_identity_summary(
        events,
        "两人在路上交谈。",
        [],
        model="test-model",
    )

    prompt_events = json.loads(
        captured["prompt"].split("人物事件：\n", 1)[1].split(
            "\n\n基础摘要：", 1
        )[0]
    )
    confirmed = [
        event
        for event in prompt_events
        if event["说话人已确认"]
    ]
    assert [event["说话人"] for event in confirmed] == ["老刘", "王书记"]
    assert result["摘要"] == "老刘说第一句对白，王书记说第二句对白。"


def test_unconfirmed_or_unknown_speaker_is_removed():
    assert _normalize_event_description(
        {"说话人": "王书记", "说话人已确认": False},
        {"王书记"},
    )["说话人"] == ""
    assert _normalize_event_description(
        {"说话人": "陌生人", "说话人已确认": True},
        {"王书记"},
    ) == {
        "动作": "",
        "场景": "",
        "人物关系": "",
        "动作事件": [],
        "对白事实": "",
        "说话人": "",
        "说话人已确认": False,
    }
    assert _normalize_event_description(
        {"说话人": "王书记", "说话人已确认": True},
        {"王书记"},
    )["说话人"] == "王书记"
    assert _normalize_event_description(
        {
            "对白事实": "领导夸赞冲锋衣价值高",
            "说话人": "",
            "说话人已确认": False,
        },
        {"王书记"},
    )["对白事实"] == "有人夸赞冲锋衣价值高"


def test_describe_event_uses_name_mapping_and_clears_unconfirmed_speaker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    frame = tmp_path / "people/events/mat_test/event_001.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"jpeg")
    captured = {}

    def fake_post(_url, json, **_kwargs):
        captured.update(json)
        return _Response(
            {
                "动作": "王书记带着老刘和一名男子走在路上",
                "场景": "乡间小路，远处有山",
                "对白事实": "有人谈到冲锋衣价格",
                "说话人": "老刘",
                "说话人已确认": False,
            }
        )

    monkeypatch.setattr("scenelog.identity_summary.requests.post", fake_post)

    described = describe_identity_events(
        [_event()],
        tmp_path,
        None,
        "test-vlm",
    )

    assert "P1=王书记；P2=老刘" in captured["prompt"]
    assert described[0]["description"]["说话人"] == ""
    assert described[0]["description"]["说话人已确认"] is False
    assert described[0]["description"]["动作"].startswith("王书记")


def test_describe_event_sends_chronological_frames_and_maps_action_roles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    frames = []
    for index, timestamp in enumerate([9.0, 10.0, 11.0], 1):
        path = tmp_path / f"people/events/mat_test/frame_{index}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"jpeg-{index}".encode())
        frames.append(
            {
                "timestamp": timestamp,
                "frame": str(path.relative_to(tmp_path)),
                "labels": [
                    {
                        "tag": "P1",
                        "person_id": "person_0001",
                        "name": "老刘",
                    }
                ],
            }
        )
    event = {
        **_event(frames[1]["frame"]),
        "people_ids": ["person_0001"],
        "people": [{"id": "person_0001", "name": "老刘"}],
        "labels": [
            {
                "tag": "P1",
                "person_id": "person_0001",
                "name": "老刘",
            }
        ],
        "frames": frames,
        "start_timestamp": 9.0,
        "end_timestamp": 11.0,
    }
    captured = {}

    def fake_post(_url, json, **_kwargs):
        captured.update(json)
        return _Response(
            {
                "动作": "两名黑衣人控制一名男子",
                "动作事件": [
                    {
                        "施动者": ["两名黑衣人"],
                        "动作": "控制并押走",
                        "承受者": ["P1"],
                        "依据": "连续图片中P1被控制并发生移动",
                        "置信度": 0.93,
                    }
                ],
                "说话人": "",
                "说话人已确认": False,
            }
        )

    monkeypatch.setattr("scenelog.identity_summary.requests.post", fake_post)

    described = describe_identity_events(
        [event],
        tmp_path,
        None,
        "test-vlm",
    )

    assert len(captured["images"]) == 3
    assert captured["options"]["num_ctx"] == 8192
    assert "P1=老刘" in captured["prompt"]
    assert "00:00:09(core)、00:00:10(core)、00:00:11(core)" in captured["prompt"]
    assert described[0]["description"]["动作事件"] == [
        {
            "施动者": ["两名黑衣人"],
            "动作": "控制并押走",
            "承受者": ["老刘"],
            "依据": "连续图片中P1被控制并发生移动",
            "置信度": 0.93,
        }
    ]
    assert described[0]["description"]["动作"] == "两名黑衣人控制并押走老刘"


def test_event_prompt_does_not_include_registered_person_absent_from_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    first_frame = tmp_path / "people/events/mat_test/first.jpg"
    second_frame = tmp_path / "people/events/mat_test/second.jpg"
    first_frame.parent.mkdir(parents=True)
    first_frame.write_bytes(b"first")
    second_frame.write_bytes(b"second")
    events = [
        {
            **_event(str(first_frame.relative_to(tmp_path))),
            "people_ids": ["person_0001"],
            "people": [{"id": "person_0001", "name": "老刘"}],
            "labels": [{"tag": "P1", "person_id": "person_0001", "name": "老刘"}],
        },
        {
            **_event(str(second_frame.relative_to(tmp_path))),
            "event_id": "event_002",
            "people_ids": ["person_0002"],
            "people": [{"id": "person_0002", "name": "王书记"}],
            "labels": [{"tag": "P2", "person_id": "person_0002", "name": "王书记"}],
        },
    ]
    prompts = []

    def fake_post(_url, json, **_kwargs):
        prompts.append(json["prompt"])
        return _Response({})

    monkeypatch.setattr("scenelog.identity_summary.requests.post", fake_post)

    describe_identity_events(events, tmp_path, None, "test-vlm")

    assert "P1=老刘" in prompts[0]
    assert "王书记" not in prompts[0]
    assert "P2=王书记" in prompts[1]
    assert "老刘" not in prompts[1]


def test_single_frame_event_drops_role_sensitive_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    frame = tmp_path / "people/events/mat_test/event_001.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"jpeg")

    monkeypatch.setattr(
        "scenelog.identity_summary.requests.post",
        lambda *_args, **_kwargs: _Response(
            {
                "动作事件": [
                    {
                        "施动者": ["两名黑衣人"],
                        "动作": "控制并押走",
                        "承受者": ["P1"],
                        "依据": "单张图片",
                        "置信度": 0.9,
                    }
                ]
            }
        ),
    )
    event = {
        **_event(str(frame.relative_to(tmp_path))),
        "people_ids": ["person_0001"],
        "people": [{"id": "person_0001", "name": "老刘"}],
        "labels": [{"tag": "P1", "person_id": "person_0001", "name": "老刘"}],
    }

    described = describe_identity_events([event], tmp_path, None, "test-vlm")

    assert described[0]["description"]["动作事件"] == []
    assert described[0]["description"]["动作"] == ""


def test_role_sensitive_action_requires_structured_roles():
    normalized = _normalize_event_description(
        {
            "动作": "王书记控制并押走两名男子",
            "人物关系": "P1控制P2",
            "动作事件": [],
        },
        [{"name": "王书记"}],
        {"P1": "王书记"},
    )

    assert normalized["动作"] == ""
    assert normalized["人物关系"] == ""


def test_passive_action_roles_are_canonicalized_and_unknown_tags_are_removed():
    normalized = _normalize_event_description(
        {
            "动作事件": [
                {
                    "施动者": ["P1"],
                    "动作": "被按住",
                    "承受者": ["两名黑衣人", "P9"],
                    "依据": "连续画面中P1被两人按住",
                    "置信度": 0.9,
                }
            ]
        },
        [{"name": "老刘"}],
        {"P1": "老刘"},
    )

    assert normalized["动作事件"] == [
        {
            "施动者": ["两名黑衣人"],
            "动作": "按住",
            "承受者": ["老刘"],
            "依据": "连续画面中P1被两人按住",
            "置信度": 0.9,
        }
    ]
    assert normalized["动作"] == "两名黑衣人按住老刘"


def test_tag_with_parenthesized_name_normalizes_to_registered_name():
    normalized = _normalize_event_description(
        {
            "动作事件": [
                {
                    "施动者": ["两名黑衣人"],
                    "动作": "控制",
                    "承受者": ["P1（老刘）"],
                    "依据": "连续图片中两人抓住P1肩膀",
                    "置信度": 0.9,
                }
            ]
        },
        [{"name": "老刘"}],
        {"P1": "老刘"},
    )

    assert normalized["动作"] == "两名黑衣人控制老刘"


def test_first_complete_action_defaults_missing_confidence_and_drops_extras():
    normalized = _normalize_event_description(
        {
            "动作事件": [
                {
                    "施动者": "两名黑衣人",
                    "动作": "按住",
                    "承受者": "老刘",
                    "依据": "两人用手臂按住老刘",
                },
                {
                    "施动者": "一名旁观者",
                    "动作": "押送",
                    "承受者": "老刘",
                    "依据": "根据姿势推测",
                    "置信度": 0.9,
                },
            ]
        },
        [{"name": "老刘"}],
        {"P1": "老刘"},
    )

    assert normalized["动作事件"] == [
        {
            "施动者": ["两名黑衣人"],
            "动作": "按住",
            "承受者": ["老刘"],
            "依据": "两人用手臂按住老刘",
            "置信度": 0.7,
        }
    ]


def test_relationship_preserves_visible_group_when_actor_is_generic():
    normalized = _normalize_event_description(
        {
            "动作事件": [
                {
                    "施动者": ["一名男子"],
                    "动作": "按住",
                    "承受者": ["P1"],
                    "依据": "多名男子将P1按在地上",
                    "置信度": 0.82,
                }
            ],
            "人物关系": "两名黑衣人控制P1",
        },
        [{"name": "老刘"}],
        {"P1": "老刘"},
    )

    assert normalized["动作"] == "两名黑衣人按住老刘"
    assert normalized["动作事件"][0]["施动者"] == ["两名黑衣人"]


def test_relationship_restores_black_clothed_group_from_single_actor():
    normalized = _normalize_event_description(
        {
            "动作事件": [
                {
                    "施动者": ["一名黑衣人"],
                    "动作": "控制并押走",
                    "承受者": ["P1"],
                    "依据": "两人抓住P1肩膀并带离",
                    "置信度": 0.9,
                }
            ],
            "人物关系": "P1被两名黑衣人控制并押走",
        },
        [{"name": "老刘"}],
        {"P1": "老刘"},
    )

    assert normalized["动作"] == "两名黑衣人控制并押走老刘"


def test_conflicting_registered_target_uses_current_face_identity():
    normalized = _normalize_event_description(
        {
            "动作事件": [
                {
                    "施动者": ["两名黑衣人"],
                    "动作": "按住",
                    "承受者": ["王书记"],
                    "依据": "两人将被控制者按在地上",
                    "置信度": 0.9,
                }
            ]
        },
        [{"name": "老刘"}],
        {"P1": "老刘", "P2": "王书记"},
    )

    assert normalized["动作"] == "两名黑衣人按住老刘"


def test_passive_relationship_overrides_contradictory_role_array():
    normalized = _normalize_event_description(
        {
            "动作事件": [
                {
                    "施动者": ["P1"],
                    "动作": "控制并押走",
                    "承受者": ["一名男子"],
                    "依据": "右侧放大图显示P1的手臂接触该男子肩膀",
                    "置信度": 0.9,
                }
            ],
            "人物关系": "一名男子被两名黑衣人控制和押走",
        },
        [{"name": "老刘"}],
        {"P1": "老刘"},
    )

    assert normalized["动作"] == "两名黑衣人控制并押走老刘"
    assert normalized["动作事件"][0]["承受者"] == ["老刘"]


def test_role_sensitive_action_rejects_conclusion_only_evidence():
    normalized = _normalize_event_description(
        {
            "动作事件": [
                {
                    "施动者": ["两名黑衣人"],
                    "动作": "控制并押走",
                    "承受者": ["王书记"],
                    "依据": "图片显示王书记被控制并押走",
                    "置信度": 0.9,
                }
            ]
        },
        [{"name": "王书记"}],
        {"P1": "老刘", "P2": "王书记"},
    )

    assert normalized["动作事件"] == []
    assert normalized["动作"] == ""


def test_dialogue_content_cannot_become_action_event():
    normalized = _normalize_event_description(
        {
            "动作事件": [
                {
                    "施动者": ["领导"],
                    "动作": "说话",
                    "承受者": ["P1"],
                    "依据": "对白内容",
                    "置信度": 0.9,
                }
            ]
        },
        [{"name": "老刘"}],
        {"P1": "老刘"},
    )

    assert normalized["动作事件"] == []


def test_control_probe_binds_verified_registered_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    frames = []
    for index, timestamp in enumerate([9.0, 10.0], 1):
        path = tmp_path / f"people/events/mat_test/frame_{index}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        import cv2
        import numpy as np

        assert cv2.imwrite(
            str(path),
            np.zeros((640, 1280, 3), dtype=np.uint8),
        )
        frames.append(
            {
                "timestamp": timestamp,
                "frame": str(path.relative_to(tmp_path)),
                "labels": [
                    {
                        "tag": "P1",
                        "person_id": "person_0001",
                        "name": "老刘",
                        "bbox": [300, 500, 80, 100],
                    }
                ],
            }
        )
    event = {
        **_event(frames[0]["frame"]),
        "people_ids": ["person_0001"],
        "people": [{"id": "person_0001", "name": "老刘"}],
        "labels": frames[0]["labels"],
        "frames": frames,
        "unknown_face_count": 2,
        "start_timestamp": 9.0,
        "end_timestamp": 10.0,
    }
    responses = [
        _Response({"场景": "户外"}),
        _Response(
            {
                "核心动作观察": "两名黑衣人抓住并按住一人",
                "施动者人数衣着": "两名黑衣人",
                "承受者衣着": "穿灰色外套的人",
                "身体接触观察": "两人用手臂抓住并按住承受者肩膀",
                "前后位置变化": "承受者从左侧被带向右侧",
            }
        ),
        _Response(
            {
                "结论": "是",
                "黄色人物衣着": "灰色外套",
                "位置关系": "被两人按住",
                "依据": "黄色区域与承受者身体位置一致",
                "置信度": 0.91,
            }
        ),
        _Response(
            {
                "实际接触者": [
                    {"上衣": "黑色", "接触部位": "手臂", "动作": "推"},
                    {"上衣": "黑色", "接触部位": "肩膀", "动作": "拉"},
                ],
                "实际接触者数量": 2,
                "排除的旁观者": "一名绿衣男子",
                "依据": "两人实际接触承受者",
            }
        ),
    ]
    monkeypatch.setattr(
        "scenelog.identity_summary.requests.post",
        lambda *_args, **_kwargs: responses.pop(0),
    )

    described = describe_identity_events([event], tmp_path, None, "test-vlm")

    assert described[0]["description"]["动作事件"] == [
        {
            "施动者": ["两名黑衣人"],
            "动作": "控制并押走",
            "承受者": ["老刘"],
            "依据": "两人用手臂抓住并按住承受者肩膀",
            "置信度": 0.91,
        }
    ]
    assert described[0]["description"]["动作"] == "两名黑衣人控制并押走老刘"


def test_failed_control_probe_removes_main_model_control_guess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    frames = []
    for index, timestamp in enumerate([4.0, 5.0], 1):
        path = tmp_path / f"people/events/mat_test/frame_{index}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"jpeg-{index}".encode())
        frames.append(
            {
                "timestamp": timestamp,
                "frame": str(path.relative_to(tmp_path)),
                "labels": [
                    {
                        "tag": "P2",
                        "person_id": "person_0002",
                        "name": "王书记",
                    }
                ],
            }
        )
    event = {
        **_event(frames[0]["frame"]),
        "people_ids": ["person_0002"],
        "people": [{"id": "person_0002", "name": "王书记"}],
        "labels": frames[0]["labels"],
        "frames": frames,
        "unknown_face_count": 2,
        "start_timestamp": 4.0,
        "end_timestamp": 5.0,
    }
    responses = [
        _Response(
            {
                "动作事件": [
                    {
                        "施动者": ["两名黑衣人"],
                        "动作": "控制",
                        "承受者": ["P2"],
                        "依据": "手臂接触",
                        "置信度": 0.9,
                    }
                ]
            }
        ),
        _Response(
            {
                "核心动作观察": "三人并肩站立",
                "施动者人数衣着": "",
                "承受者衣着": "",
                "身体接触观察": "无",
                "前后位置变化": "无",
            }
        ),
    ]
    monkeypatch.setattr(
        "scenelog.identity_summary.requests.post",
        lambda *_args, **_kwargs: responses.pop(0),
    )

    described = describe_identity_events([event], tmp_path, None, "test-vlm")

    assert described[0]["description"]["动作事件"] == []
    assert described[0]["description"]["动作"] == ""


def test_high_confidence_action_fact_is_not_lost_from_summary():
    events = [
        {
            "description": {
                "动作事件": [
                    {
                        "施动者": ["两名黑衣人"],
                        "动作": "控制并押走",
                        "承受者": ["老刘"],
                        "依据": "连续图片",
                        "置信度": 0.91,
                    }
                ]
            }
        }
    ]

    result = _ensure_grounded_action_facts(
        "王书记与老刘在乡村小路上行走。",
        events,
    )

    assert result == (
        "王书记与老刘在乡村小路上行走。"
        "两名黑衣人控制并押走老刘。"
    )


def test_high_confidence_action_is_included_in_summary_keywords(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "scenelog.identity_summary.requests.post",
        lambda *_args, **_kwargs: _Response(
            {
                "摘要": "王书记在路边，老刘被两名黑衣人控制并押走。",
                "关键词": ["乡村小路"],
            }
        ),
    )
    event = {
        **_event(),
        "description": {
            "动作事件": [
                {
                    "施动者": ["两名黑衣人"],
                    "动作": "控制并押走",
                    "承受者": ["老刘"],
                    "依据": "连续图片",
                    "置信度": 0.9,
                }
            ],
            "说话人": "",
            "说话人已确认": False,
        },
    }

    result = generate_identity_summary(
        [event],
        "三名男子在乡村小路上。",
        [],
        model="test-text-model",
    )

    assert result["关键词"][:3] == ["王书记", "老刘", "控制并押走"]


def test_equivalent_control_action_is_not_appended_twice():
    events = [
        {
            "description": {
                "动作事件": [
                    {
                        "施动者": ["两名黑衣人"],
                        "动作": "按住",
                        "承受者": ["老刘"],
                        "依据": "连续图片",
                        "置信度": 0.9,
                    }
                ]
            }
        }
    ]

    result = _ensure_grounded_action_facts(
        "老刘被两名黑衣人控制。",
        events,
    )

    assert result == "老刘被两名黑衣人控制。"


def test_summary_retries_when_confirmed_speaker_is_only_anonymous(
    monkeypatch: pytest.MonkeyPatch,
):
    responses = iter(
        [
            {
                "摘要": "王书记和老刘在乡村小路交谈，有人夸赞冲锋衣价值高。",
                "关键词": ["乡村小路"],
            },
            {
                "摘要": "王书记和老刘走在乡村小路上，老刘夸赞王书记的冲锋衣价值高。",
                "关键词": ["冲锋衣"],
            },
        ]
    )
    monkeypatch.setattr(
        "scenelog.identity_summary.requests.post",
        lambda *_args, **_kwargs: _Response(next(responses)),
    )
    event = {
        "people": [
            {"id": "person_0001", "name": "老刘"},
            {"id": "person_0002", "name": "王书记"},
        ],
        "description": {
            "动作": "两人在乡村小路行走",
            "动作事件": [],
            "场景": "乡村小路",
            "对白事实": "有人夸赞冲锋衣价值高",
            "说话人": "老刘",
            "说话人已确认": True,
        },
    }

    result = generate_identity_summary(
        [event],
        "两名男子走在乡村小路上。",
        [],
        model="test-model",
    )

    assert "老刘夸赞" in result["摘要"]


def test_compound_action_keeps_missing_movement_fact():
    events = [
        {
            "description": {
                "动作事件": [
                    {
                        "施动者": ["两名黑衣人"],
                        "动作": "控制并押走",
                        "承受者": ["老刘"],
                        "依据": "连续图片",
                        "置信度": 0.9,
                    }
                ]
            }
        }
    ]

    result = _ensure_grounded_action_facts(
        "老刘被两名黑衣人控制。",
        events,
    )

    assert result.endswith("两名黑衣人控制并押走老刘。")


def test_grounded_action_replaces_conflicting_control_sentence():
    events = [
        {
            "description": {
                "动作事件": [
                    {
                        "施动者": ["两名黑衣人"],
                        "动作": "控制并押走",
                        "承受者": ["老刘"],
                        "依据": "连续图片",
                        "置信度": 0.9,
                    }
                ]
            }
        }
    ]

    result = _ensure_grounded_action_facts(
        "王书记走在乡村小路上。四名男子控制并押走老刘。",
        events,
    )

    assert result == "王书记走在乡村小路上。两名黑衣人控制并押走老刘。"


def test_summary_retries_when_registered_name_is_omitted(
    monkeypatch: pytest.MonkeyPatch,
):
    responses = [
        _Response(
            {
                "摘要": "王书记带着两名男子走在乡间小路上。",
                "关键词": ["王书记", "乡间小路"],
            }
        ),
        _Response(
            {
                "摘要": (
                    "王书记带着一名男子和老刘走在乡间小路上。"
                    "交谈中有人夸赞王书记的冲锋衣价值高。"
                ),
                "关键词": ["王书记", "老刘", "乡间小路", "冲锋衣"],
            }
        ),
    ]
    calls = []

    def fake_post(_url, json, **_kwargs):
        calls.append(json["prompt"])
        return responses.pop(0)

    monkeypatch.setattr("scenelog.identity_summary.requests.post", fake_post)
    event = {
        **_event(),
        "description": {
            "动作": "王书记带着老刘和一名男子走在乡间小路上",
            "场景": "蓝天和山脉",
            "人物关系": "",
            "对白事实": "有人夸赞王书记的冲锋衣价值高",
            "说话人": "",
            "说话人已确认": False,
        },
    }

    result = generate_identity_summary(
        [event],
        "乡村小路上三名男子并肩行走。",
        ["乡村", "行走"],
        model="test-text-model",
    )

    assert len(calls) == 2
    assert "遗漏已识别人物：老刘" in calls[1]
    assert "王书记" in result["摘要"]
    assert "老刘" in result["摘要"]
    assert "出现在素材中" not in result["摘要"]
    assert result["关键词"][:2] == ["王书记", "老刘"]


def test_summary_removes_unconfirmed_named_speaker_after_failed_rewrite(
    monkeypatch: pytest.MonkeyPatch,
):
    responses = [
        _Response(
            {
                "摘要": (
                    "王书记与老刘在乡村小路上行走。"
                    "王书记夸赞老刘的冲锋衣价值高。"
                ),
                "关键词": ["王书记", "老刘", "冲锋衣"],
            }
        ),
        _Response(
            {
                "摘要": (
                    "王书记与老刘在乡村小路上行走。"
                    "王书记夸赞老刘的冲锋衣价值高。"
                ),
                "关键词": ["王书记", "老刘", "冲锋衣"],
            }
        ),
    ]
    monkeypatch.setattr(
        "scenelog.identity_summary.requests.post",
        lambda *_args, **_kwargs: responses.pop(0),
    )
    event = {
        **_event(),
        "description": {
            "动作": "王书记与老刘在乡村小路上行走",
            "场景": "蓝天和山脉",
            "人物关系": "",
            "对白事实": "有人夸赞一件冲锋衣价值高",
            "说话人": "",
            "说话人已确认": False,
        },
    }

    result = generate_identity_summary(
        [event],
        "三名男子在乡村小路上行走。",
        ["乡村小路"],
        model="test-text-model",
    )

    assert result["摘要"] == (
        "王书记与老刘在乡村小路上行走。"
        "交谈中有人夸赞老刘的冲锋衣价值高。"
    )


def test_summary_treats_named_discussion_as_unconfirmed_speech(
    monkeypatch: pytest.MonkeyPatch,
):
    payload = {
        "摘要": (
            "王书记与老刘及另一名男子在乡村小路上行走。"
            "老刘继续讨论冲锋枪的话题。"
        ),
        "关键词": ["王书记", "老刘", "冲锋枪"],
    }
    monkeypatch.setattr(
        "scenelog.identity_summary.requests.post",
        lambda *_args, **_kwargs: _Response(payload),
    )
    event = {
        **_event(),
        "description": {
            "动作": "王书记与老刘在乡村小路上行走",
            "场景": "乡村小路",
            "人物关系": "",
            "对白事实": "交谈涉及冲锋枪",
            "说话人": "",
            "说话人已确认": False,
        },
    }

    result = generate_identity_summary(
        [event],
        "三名男子在乡村小路上行走。",
        [],
        model="test-text-model",
    )

    assert result["摘要"] == (
        "王书记与老刘及另一名男子在乡村小路上行走。"
        "交谈中有人讨论冲锋枪的话题。"
    )


def test_summary_uses_grounded_fallback_after_three_invalid_responses(
    monkeypatch: pytest.MonkeyPatch,
):
    payload = {
        "摘要": "王书记夸赞冲锋衣价值高。",
        "关键词": ["冲锋衣"],
    }
    calls = []

    def fake_post(*_args, **_kwargs):
        calls.append(1)
        return _Response(payload)

    monkeypatch.setattr("scenelog.identity_summary.requests.post", fake_post)
    event = {
        **_event(),
        "description": {
            "动作": "王书记与老刘在乡村小路上行走",
            "场景": "乡村小路，背景有蓝天和山脉",
            "人物关系": "",
            "对白事实": "有人夸赞冲锋衣价值高",
            "说话人": "",
            "说话人已确认": False,
        },
    }

    result = generate_identity_summary(
        [event],
        "乡村小路上三名男子并肩行走，背景是蓝天和山脉。",
        ["乡村小路"],
        model="test-text-model",
    )

    assert len(calls) == 3
    assert result["摘要"].startswith(
        "乡村小路上王书记、老刘和一名男子并肩行走，背景是蓝天和山脉。"
    )
    assert "交谈中有人提到" in result["摘要"]


def test_anonymous_speech_does_not_claim_possession(
    monkeypatch: pytest.MonkeyPatch,
):
    payload = {
        "摘要": (
            "王书记与老刘在乡村小路上行走。"
            "交谈中有人夸赞自己的冲锋衣价值高。"
        ),
        "关键词": ["王书记", "老刘", "冲锋衣"],
    }
    monkeypatch.setattr(
        "scenelog.identity_summary.requests.post",
        lambda *_args, **_kwargs: _Response(payload),
    )
    event = {
        **_event(),
        "description": {
            "动作": "王书记与老刘在乡村小路上行走",
            "场景": "乡村小路",
            "人物关系": "",
            "对白事实": "有人夸赞一件冲锋衣价值高",
            "说话人": "",
            "说话人已确认": False,
        },
    }

    result = generate_identity_summary(
        [event],
        "三名男子在乡村小路上行走。",
        [],
        model="test-text-model",
    )

    assert "交谈中有人夸赞一件冲锋衣价值高" in result["摘要"]
