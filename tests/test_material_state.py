from pathlib import Path

from scenelog.config import PIPELINE_VERSION
from scenelog.material_id import (
    compute_source_fingerprint,
    generate_material_id,
    material_meta,
)
from scenelog.state import ALL_STEPS, StateManager


def test_material_id_stays_stable_when_content_changes(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    media = source / "card-a" / "C001.MOV"
    media.parent.mkdir()
    media.write_bytes(b"first")

    material_id = generate_material_id(source, media)
    fingerprint = compute_source_fingerprint(source, media)

    media.write_bytes(b"second-content")

    assert generate_material_id(source, media) == material_id
    assert compute_source_fingerprint(source, media) != fingerprint


def test_same_file_name_in_different_directories_has_distinct_ids(tmp_path: Path):
    source = tmp_path / "source"
    first = source / "card-a" / "C001.MOV"
    second = source / "card-b" / "C001.MOV"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"one")
    second.write_bytes(b"two")

    assert generate_material_id(source, first) != generate_material_id(source, second)


def test_state_accepts_new_fingerprint_and_persists_visual_descriptions(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    media = source / "clip.mov"
    media.write_bytes(b"content")
    meta = material_meta(source, media)
    material_id = meta["material_id"]
    state = StateManager(tmp_path / "output")
    state.init_record(material_id, meta)
    state.set_vision(
        material_id,
        {
            "visual_summary": "院子里有人劈柴",
            "visual_descs": ["院子里一名男子劈柴", "木柴堆在墙边"],
            "frame_count": 2,
            "frame_points": [3.0, 9.0],
            "model": "llava:7b",
        },
    )

    state._states[material_id]["pipeline_version"] = "old"
    state._save()
    assert state.check_stale(material_id, meta["fingerprint"]) == ALL_STEPS

    state.accept_source_version(material_id, meta["fingerprint"], meta)
    restored = StateManager(tmp_path / "output")

    assert restored.check_stale(material_id, meta["fingerprint"]) == []
    assert restored.get(material_id)["pipeline_version"] == PIPELINE_VERSION
    assert restored.get_vision(material_id)["visual_descs"] == [
        "院子里一名男子劈柴",
        "木柴堆在墙边",
    ]


def test_invalidating_vision_clears_dependent_cached_values(tmp_path: Path):
    state = StateManager(tmp_path)
    material_id = "mat_test"
    state.set_vision(
        material_id,
        {
            "visual_summary": "旧摘要",
            "visual_descs": ["旧描述"],
            "frame_count": 1,
            "frame_points": [1.0],
            "model": "old-model",
        },
    )
    state.set_summary(material_id, "旧融合摘要", ["旧关键词"])

    state.invalidate_steps(material_id, ["vision", "summary", "index", "excel"])

    assert state.get_vision(material_id)["visual_descs"] == []
    assert state.get_summary(material_id) == {"摘要": "", "关键词": []}
    assert state.get_step(material_id, "vision") == "stale"


def test_speaker_state_is_persisted_and_invalidated_independently(tmp_path: Path):
    state = StateManager(tmp_path)
    material_id = "mat_test"
    segments = [
        {
            "start_seconds": 0.0,
            "end_seconds": 3.84,
            "text": "第一句对白",
            "speaker_name": "老刘",
            "confirmed": True,
        }
    ]
    state.set_speaker_segments(
        material_id,
        segments,
        "speaker_transcripts/mat_test.srt",
        "speaker_transcripts/mat_test.txt",
    )

    restored = StateManager(tmp_path)
    assert restored.get_speaker_segments(material_id) == segments

    restored.invalidate_steps(
        material_id,
        ["speaker", "summary", "index", "excel"],
    )
    record = restored.get(material_id)
    assert record["speaker_segments"] == []
    assert record["speaker_transcript_srt"] == ""
    assert record["speaker"] == "stale"


def test_invalidating_only_speaker_keeps_current_generated_summary(tmp_path: Path):
    state = StateManager(tmp_path)
    material_id = "mat_test"
    state.set_summary(material_id, "基础摘要", ["基础"])
    state.set_identity_summary(
        material_id,
        [{"event_id": "event_1"}],
        "老刘夸赞王书记的冲锋衣。",
        ["老刘", "王书记"],
    )

    state.invalidate_steps(material_id, ["speaker", "index", "excel"])

    assert state.get_summary(material_id)["摘要"] == "老刘夸赞王书记的冲锋衣。"
    assert state.get_identity_events(material_id) == [{"event_id": "event_1"}]


def test_step_errors_are_kept_per_step(tmp_path: Path):
    state = StateManager(tmp_path)
    material_id = "mat_test"
    state.set_step(material_id, "vision", "failed_retryable", "vision offline")
    state.set_step(material_id, "index", "success")

    record = state.get(material_id)
    assert record["errors"]["vision"] == "vision offline"
    assert record["last_error"] == "vision offline"


def test_manual_summary_override_survives_generated_summary_updates(tmp_path: Path):
    state = StateManager(tmp_path)
    material_id = "mat_test"
    state.set_summary(material_id, "自动摘要", ["自动", "关键词"])

    assert state.sync_manual_summary(
        material_id,
        "人工修正摘要",
        ["人工", "修正"],
    )
    state.set_summary(material_id, "新自动摘要", ["新自动"])

    assert state.get_summary(material_id) == {
        "摘要": "人工修正摘要",
        "关键词": ["人工", "修正"],
    }

    state.clear_manual_summary(material_id)

    assert state.get_summary(material_id) == {
        "摘要": "新自动摘要",
        "关键词": ["新自动"],
    }


def test_identity_summary_replaces_base_summary_and_keeps_base_for_fallback(
    tmp_path: Path,
):
    state = StateManager(tmp_path)
    material_id = "mat_test"
    state.set_summary(material_id, "乡村小路上三名男子并肩行走", ["乡村", "行走"])

    events = [
        {
            "event_id": "event_001",
            "people": [{"id": "person_0001", "name": "王书记"}],
            "description": {"动作": "王书记带着两名男子走在乡村小路上"},
        }
    ]
    state.set_identity_summary(
        material_id,
        events,
        "王书记带着两名男子走在乡村小路上。",
        ["王书记", "乡村小路"],
    )

    assert state.get_summary(material_id) == {
        "摘要": "王书记带着两名男子走在乡村小路上。",
        "关键词": ["王书记", "乡村小路"],
    }
    assert state.get_base_summary(material_id) == {
        "摘要": "乡村小路上三名男子并肩行走",
        "关键词": ["乡村", "行走"],
    }
    assert state.get_identity_events(material_id) == events

    state.clear_identity_summary(material_id)
    assert state.get_summary(material_id)["摘要"] == "乡村小路上三名男子并肩行走"


def test_manual_summary_stays_untouched_when_identity_summary_changes(tmp_path: Path):
    state = StateManager(tmp_path)
    material_id = "mat_test"
    state.set_summary(material_id, "自动摘要", ["关键词"])
    state.sync_manual_summary(material_id, "导演手工摘要", ["人工"])

    state.set_identity_summary(
        material_id,
        [{"event_id": "event_001"}],
        "张三带着众人出发。",
        ["张三", "出发"],
    )

    assert state.get_summary(material_id) == {
        "摘要": "导演手工摘要",
        "关键词": ["人工"],
    }
    assert state.get(material_id)["summary_text"] == "张三带着众人出发。"
    assert state.get(material_id)["base_summary_text"] == "自动摘要"

    state.clear_manual_summary(material_id)
    assert state.get_summary(material_id) == {
        "摘要": "张三带着众人出发。",
        "关键词": ["张三", "出发"],
    }


def test_syncing_generated_identity_keywords_does_not_create_manual_override(
    tmp_path: Path,
):
    state = StateManager(tmp_path)
    material_id = "mat_test"
    state.set_summary(material_id, "基础摘要", ["基础"])
    state.set_identity_summary(
        material_id,
        [{"event_id": "event_001"}],
        "王书记走在乡间小路上。",
        ["王书记", "乡间小路"],
    )

    changed = state.sync_manual_summary(
        material_id,
        "王书记走在乡间小路上。",
        ["王书记", "乡间小路"],
    )

    assert changed is False
    assert state.get(material_id)["manual_summary_keywords"] is None


def test_v03_state_only_invalidates_summary_and_excel_on_upgrade(tmp_path: Path):
    state = StateManager(tmp_path)
    material_id = "mat_test"
    state.init_record(
        material_id,
        {
            "fingerprint": "same-fingerprint",
            "file_name": "clip.mov",
            "rel_path": "clip.mov",
        },
    )
    state._states[material_id]["pipeline_version"] = "0.3.0"
    state._save()

    assert state.check_stale(material_id, "same-fingerprint") == [
        "summary",
        "people",
        "index",
        "excel",
    ]


def test_v04_state_only_adds_people_index_and_excel_on_upgrade(tmp_path: Path):
    state = StateManager(tmp_path)
    material_id = "mat_test"
    state.init_record(
        material_id,
        {
            "fingerprint": "same-fingerprint",
            "file_name": "clip.mov",
            "rel_path": "clip.mov",
        },
    )
    state._states[material_id]["pipeline_version"] = "0.4.0"
    state._save()

    assert state.check_stale(material_id, "same-fingerprint") == [
        "people",
        "index",
        "excel",
    ]


def test_v05_state_only_rematches_registered_people_on_upgrade(tmp_path: Path):
    state = StateManager(tmp_path)
    material_id = "mat_test"
    state.init_record(
        material_id,
        {
            "fingerprint": "same-fingerprint",
            "file_name": "clip.mov",
            "rel_path": "clip.mov",
        },
    )
    state._states[material_id]["pipeline_version"] = "0.5.0"
    state._save()

    assert state.check_stale(material_id, "same-fingerprint") == [
        "people",
        "index",
        "excel",
    ]


def test_v060_state_only_rescans_people_on_upgrade(tmp_path: Path):
    state = StateManager(tmp_path)
    material_id = "mat_test"
    state.init_record(
        material_id,
        {
            "fingerprint": "same-fingerprint",
            "file_name": "clip.mov",
            "rel_path": "clip.mov",
        },
    )
    state._states[material_id]["pipeline_version"] = "0.6.0"
    state._save()

    assert state.check_stale(material_id, "same-fingerprint") == [
        "people",
        "index",
        "excel",
    ]


def test_v061_state_rescans_people_for_identity_events_on_upgrade(tmp_path: Path):
    state = StateManager(tmp_path)
    material_id = "mat_test"
    state.init_record(
        material_id,
        {
            "fingerprint": "same-fingerprint",
            "file_name": "clip.mov",
            "rel_path": "clip.mov",
        },
    )
    state._states[material_id]["pipeline_version"] = "0.6.1"
    state._save()

    assert state.check_stale(material_id, "same-fingerprint") == [
        "people",
        "index",
        "excel",
    ]


def test_v062_state_rescans_people_for_identity_events_on_upgrade(tmp_path: Path):
    state = StateManager(tmp_path)
    material_id = "mat_test"
    state.init_record(
        material_id,
        {
            "fingerprint": "same-fingerprint",
            "file_name": "clip.mov",
            "rel_path": "clip.mov",
        },
    )
    state._states[material_id]["pipeline_version"] = "0.6.2"
    state._save()

    assert state.check_stale(material_id, "same-fingerprint") == [
        "people",
        "index",
        "excel",
    ]


def test_v070_state_rescans_people_for_multiframe_actions_on_upgrade(
    tmp_path: Path,
):
    state = StateManager(tmp_path)
    material_id = "mat_test"
    state.init_record(
        material_id,
        {
            "fingerprint": "same-fingerprint",
            "file_name": "clip.mov",
            "rel_path": "clip.mov",
        },
    )
    state._states[material_id]["pipeline_version"] = "0.7.0"
    state._save()

    assert state.check_stale(material_id, "same-fingerprint") == [
        "people",
        "index",
        "excel",
    ]


def test_v081_state_only_adds_speaker_and_downstream_outputs(tmp_path: Path):
    state = StateManager(tmp_path)
    material_id = "mat_test"
    state.init_record(
        material_id,
        {
            "fingerprint": "same-fingerprint",
            "file_name": "clip.mov",
            "rel_path": "clip.mov",
        },
    )
    state._states[material_id]["pipeline_version"] = "0.8.1"
    state._save()

    assert state.check_stale(material_id, "same-fingerprint") == [
        "speaker",
        "index",
        "excel",
    ]


