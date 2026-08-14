from pathlib import Path

import pytest

from scenelog.pipeline import Pipeline
from scenelog.vision import visual_only_summary


class _FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "response": (
                '{"摘要":"夜晚城市道路上车流穿行",'
                '"关键词":["夜晚","城市道路","车流"]}'
            )
        }


def test_single_file_selector_accepts_unique_name_and_relative_path(tmp_path: Path):
    source = tmp_path / "source"
    first = source / "card-a" / "A001.MOV"
    second = source / "card-b" / "B001.MOV"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"one")
    second.write_bytes(b"two")

    pipeline = Pipeline(source, selected_file="A001.MOV")
    assert pipeline._select_media_files([first, second]) == [first]

    pipeline = Pipeline(source, selected_file="card-b/B001.MOV")
    assert pipeline._select_media_files([first, second]) == [second]


def test_single_file_selector_rejects_ambiguous_name(tmp_path: Path):
    source = tmp_path / "source"
    first = source / "card-a" / "A001.MOV"
    second = source / "card-b" / "A001.MOV"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"one")
    second.write_bytes(b"two")

    pipeline = Pipeline(source, selected_file="A001.MOV")

    with pytest.raises(ValueError, match="文件名不唯一"):
        pipeline._select_media_files([first, second])


def test_visual_only_summary_returns_searchable_keywords(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "scenelog.vision.requests.post",
        lambda *args, **kwargs: _FakeResponse(),
    )

    result = visual_only_summary(
        ["夜晚城市道路上车辆行驶", "高楼前车流持续穿行"],
        "test-model",
    )

    assert result == {
        "摘要": "夜晚城市道路上车流穿行",
        "关键词": ["夜晚", "城市道路", "车流"],
    }


def test_visual_only_summary_retries_keyword_generation(
    monkeypatch: pytest.MonkeyPatch,
):
    class _SummaryOnlyResponse(_FakeResponse):
        def json(self):
            return {"response": "北京街头繁忙交通景象持续"}

    monkeypatch.setattr(
        "scenelog.vision.requests.post",
        lambda *args, **kwargs: _SummaryOnlyResponse(),
    )
    monkeypatch.setattr(
        "scenelog.vision.generate_summary_from_text",
        lambda *args, **kwargs: {
            "摘要": "北京街头交通",
            "关键词": ["北京街头", "交通", "车流"],
        },
    )

    result = visual_only_summary(["北京街头车辆持续通行"], "test-model")

    assert result == {
        "摘要": "北京街头繁忙交通景象持续",
        "关键词": ["北京街头", "交通", "车流"],
    }


def test_summary_only_stale_cleanup_keeps_transcripts_frames_and_audio(
    tmp_path: Path,
):
    pipeline = Pipeline(tmp_path / "source", output_dir=tmp_path / "output")
    material_id = "mat_test"
    pipeline.cache_dir.mkdir(parents=True)
    pipeline.transcripts_dir.mkdir(parents=True)
    pipeline.frames_dir.mkdir(parents=True)
    audio_path = pipeline.cache_dir / f"{material_id}.wav"
    srt_path = pipeline.transcripts_dir / f"{material_id}.srt"
    txt_path = pipeline.transcripts_dir / f"{material_id}.txt"
    frame_path = pipeline.frames_dir / f"{material_id}_f0_1.0s.jpg"
    for path in [audio_path, srt_path, txt_path, frame_path]:
        path.write_bytes(b"cached")

    pipeline._remove_material_artifacts(material_id, ["summary", "excel"])

    assert all(path.exists() for path in [audio_path, srt_path, txt_path, frame_path])


@pytest.mark.parametrize("skipped_step", ["vad", "audio_extract"])
def test_build_record_visual_fallback_keeps_keywords(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    skipped_step: str,
):
    source = tmp_path / "source"
    source.mkdir()
    media = source / "clip.mov"
    media.write_bytes(b"video")
    pipeline = Pipeline(source, output_dir=tmp_path / "output")
    material_id = "mat_test"
    meta = {
        "file_name": media.name,
        "rel_path": media.name,
        "fingerprint": "fingerprint",
    }
    pipeline.state.init_record(material_id, meta)
    pipeline.state.set_step(material_id, skipped_step, "skipped")
    pipeline.state.set_vision(
        material_id,
        {
            "visual_summary": "城市道路上车辆穿行",
            "visual_keywords": ["城市道路", "车辆", "交通"],
        },
    )
    monkeypatch.setattr(
        "scenelog.pipeline.extract_metadata",
        lambda _path: {},
    )

    record = pipeline._build_record(material_id, meta)

    assert record["summary"] == {
        "摘要": "城市道路上车辆穿行",
        "关键词": ["城市道路", "车辆", "交通"],
    }


def test_build_record_uses_persisted_identity_summary_without_name_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "source"
    source.mkdir()
    media = source / "clip.mov"
    media.write_bytes(b"video")
    pipeline = Pipeline(source, output_dir=tmp_path / "output")
    material_id = "mat_test"
    meta = {
        "file_name": media.name,
        "rel_path": media.name,
        "fingerprint": "fingerprint",
    }
    pipeline.state.init_record(material_id, meta)
    pipeline.state.set_summary(material_id, "三名男子在乡村小路行走", ["乡村"])
    pipeline.state.set_identity_summary(
        material_id,
        [{"event_id": "event_001"}],
        "刘书记带着老李在乡村小路上行走。",
        ["刘书记", "老李", "乡村"],
    )
    pipeline.state.set_people(material_id, ["person_0001", "person_0002"])
    pipeline.people_store.data["people"] = {
        "person_0001": {"name": "刘书记"},
        "person_0002": {"name": "老李"},
    }
    monkeypatch.setattr("scenelog.pipeline.extract_metadata", lambda _path: {})

    record = pipeline._build_record(material_id, meta)

    assert record["people"] == ["刘书记", "老李"]
    assert record["summary"] == {
        "摘要": "刘书记带着老李在乡村小路上行走。",
        "关键词": ["刘书记", "老李", "乡村"],
    }


def test_update_identity_summary_persists_events_and_identity_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "source"
    source.mkdir()
    pipeline = Pipeline(source, output_dir=tmp_path / "output")
    material_id = "mat_test"
    pipeline.state.init_record(
        material_id,
        {
            "fingerprint": "fingerprint",
            "file_name": "clip.mov",
            "rel_path": "card-a/clip.mov",
        },
    )
    pipeline.state.set_summary(material_id, "三名男子在乡村小路行走", ["乡村"])
    event = {
        "event_id": "event_001",
        "start_timestamp": 1.0,
        "end_timestamp": 3.0,
        "people_ids": ["person_0001"],
        "people": [{"id": "person_0001", "name": "王书记"}],
        "labels": [{"tag": "P1", "person_id": "person_0001", "name": "王书记"}],
        "frame": "people/events/mat_test/event_001.jpg",
    }
    monkeypatch.setattr(
        pipeline.people_store,
        "material_events",
        lambda _material_id: [event],
    )
    monkeypatch.setattr(
        "scenelog.pipeline.describe_identity_events",
        lambda *args, **kwargs: [
            {
                **event,
                "description": {
                    "动作": "王书记带着两名男子行走",
                    "场景": "乡村小路",
                    "对白事实": "有人谈到冲锋衣",
                    "说话人": "",
                    "说话人已确认": False,
                },
            }
        ],
    )
    monkeypatch.setattr(
        "scenelog.pipeline.generate_identity_summary",
        lambda *args, **kwargs: {
            "摘要": "王书记带着两名男子走在乡村小路上。",
            "关键词": ["王书记", "乡村小路"],
        },
    )
    monkeypatch.setattr(pipeline, "_get_vlm_model", lambda: "test-vlm")

    pipeline._update_identity_summary(material_id, None)

    assert pipeline.state.get_summary(material_id)["摘要"] == (
        "王书记带着两名男子走在乡村小路上。"
    )
    results = __import__(
        "scenelog.search", fromlist=["Searcher"]
    ).Searcher(source, pipeline.scenelog_dir).search("王书记 冲锋衣")
    assert len(results) == 1
    assert results[0]["source"] == "identity"
    assert results[0]["start_time"] == "00:00:01"


def test_people_step_skips_without_registered_people(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "source"
    source.mkdir()
    pipeline = Pipeline(source, output_dir=tmp_path / "output")
    pipeline.frames_dir.mkdir(parents=True)
    material_id = "mat_test"
    (pipeline.frames_dir / f"{material_id}_f0_1.0s.jpg").write_bytes(b"frame")
    pipeline.state.init_record(
        material_id,
        {
            "fingerprint": "fingerprint",
            "file_name": "clip.mov",
            "rel_path": "clip.mov",
        },
    )
    monkeypatch.setattr(
        pipeline,
        "_get_face_engine",
        lambda: (_ for _ in ()).throw(AssertionError("model should not load")),
    )

    pipeline._process_people(
        material_id,
        {"file_name": "clip.mov", "rel_path": "clip.mov"},
        ["people"],
    )

    assert pipeline.state.get_step(material_id, "people") == "skipped"
    assert pipeline.state.get_people(material_id) == []


def test_people_step_uses_dense_frames_not_vlm_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = tmp_path / "source"
    source.mkdir()
    video = source / "clip.mov"
    video.write_bytes(b"video")
    pipeline = Pipeline(source, output_dir=tmp_path / "output")
    pipeline.people_store.data["people"]["person_0001"] = {
        "id": "person_0001",
        "name": "张三",
        "reference_embeddings": [[1.0, 0.0]],
        "reference_photos": ["reference.jpg"],
        "occurrences": [],
    }
    material_id = "mat_test"
    pipeline.state.init_record(
        material_id,
        {
            "fingerprint": "fingerprint",
            "file_name": video.name,
            "rel_path": video.name,
        },
    )
    dense_frame = tmp_path / "dense.jpg"
    dense_frame.write_bytes(b"frame")
    calls = {}

    monkeypatch.setattr(
        "scenelog.pipeline.extract_people_frames",
        lambda video_path, output_dir, mid, duration: (
            calls.update(
                {
                    "video_path": video_path,
                    "output_dir": output_dir,
                    "material_id": mid,
                    "duration": duration,
                }
            )
            or [(0.0, dense_frame)]
        ),
    )
    monkeypatch.setattr(pipeline, "_get_face_engine", lambda: object())
    monkeypatch.setattr(
        pipeline.people_store,
        "process_material",
        lambda *args: ["person_0001"],
    )

    pipeline._process_people(
        material_id,
        {"file_name": video.name, "rel_path": video.name},
        ["people"],
        video_path=video,
        duration=8.5,
        has_video=True,
    )

    assert calls["video_path"] == video
    assert calls["duration"] == 8.5
    assert calls["output_dir"] == pipeline.people_frames_dir
    assert pipeline.state.get_people(material_id) == ["person_0001"]
    assert pipeline.state.get_step(material_id, "people") == "success"
