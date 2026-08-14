import json
from pathlib import Path

import numpy as np

from scenelog.indexer import build_people_index
import scenelog.people as people_module
from scenelog.people import (
    PeopleStore,
    _write_annotated_event_frame,
    cached_people_frames,
    extract_people_frames,
    people_scan_points,
)
from scenelog.search import Searcher


class _FakeEngine:
    def __init__(self, embeddings: dict[str, list[list[float]]]):
        self.embeddings = embeddings

    def extract(self, frame_path: Path) -> list[dict]:
        return [
            {
                "bbox": [0, 0, 80, 80],
                "score": 0.95,
                "embedding": embedding,
                "crop": np.zeros((80, 80, 3), dtype=np.uint8),
            }
            for embedding in self.embeddings.get(frame_path.name, [])
        ]


def _image(path: Path) -> Path:
    path.write_bytes(b"image")
    return path


def _stub_event_frames(monkeypatch):
    monkeypatch.setattr(
        people_module,
        "_write_annotated_event_frame",
        lambda _source, target, _labels: target.write_bytes(b"annotated"),
    )


def test_annotated_event_frame_includes_full_and_enlarged_views(tmp_path: Path):
    import cv2

    source = tmp_path / "source.jpg"
    target = tmp_path / "event.jpg"
    assert cv2.imwrite(
        str(source),
        np.zeros((720, 960, 3), dtype=np.uint8),
    )

    _write_annotated_event_frame(
        source,
        target,
        [{"tag": "P1", "bbox": [400, 250, 80, 100]}],
    )

    image = cv2.imread(str(target))
    assert image.shape[:2] == (640, 1280)


def test_registered_person_matches_across_materials_and_keeps_id_on_rerun(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(PeopleStore, "cv2_write", lambda *args: True)
    _stub_event_frames(monkeypatch)
    reference = _image(tmp_path / "zhangsan.jpg")
    frame_a = _image(tmp_path / "mat_a_f0_1.0s.jpg")
    frame_b = _image(tmp_path / "mat_b_f0_2.0s.jpg")
    engine = _FakeEngine(
        {
            reference.name: [[1.0, 0.0, 0.0]],
            frame_a.name: [[0.99, 0.05, 0.0]],
            frame_b.name: [[0.98, 0.08, 0.0]],
        }
    )
    store = PeopleStore(tmp_path / "output")

    person_id = store.add_person("张三", [reference], engine)
    first_ids = store.process_material(
        "mat_a", "a.mov", "a.mov", [(1.0, frame_a)], engine
    )
    second_ids = store.process_material(
        "mat_b", "b.mov", "b.mov", [(2.0, frame_b)], engine
    )
    rerun_ids = store.process_material(
        "mat_a", "a.mov", "a.mov", [(1.0, frame_a)], engine
    )

    assert person_id == "person_0001"
    assert first_ids == second_ids == rerun_ids == [person_id]
    profile = store.list_people()[0]
    assert profile["name"] == "张三"
    assert profile["reference_count"] == 1
    assert profile["sample_count"] == 2
    assert profile["material_count"] == 2


def test_continuous_matches_are_grouped_into_one_occurrence(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(PeopleStore, "cv2_write", lambda *args: True)
    _stub_event_frames(monkeypatch)
    reference = _image(tmp_path / "reference.jpg")
    frames = [
        _image(tmp_path / f"frame_{index}.jpg")
        for index in range(4)
    ]
    engine = _FakeEngine(
        {
            reference.name: [[1.0, 0.0]],
            frames[0].name: [[0.9, 0.1]],
            frames[1].name: [[0.99, 0.01]],
            frames[2].name: [[0.95, 0.05]],
            frames[3].name: [[0.92, 0.08]],
        }
    )
    store = PeopleStore(tmp_path / "output")
    store.add_person("张三", [reference], engine)

    store.process_material(
        "mat_a",
        "a.mov",
        "a.mov",
        list(zip([0.0, 1.0, 2.0, 8.0], frames)),
        engine,
    )

    occurrences = store.list_people()[0]["occurrences"]
    assert len(occurrences) == 2
    assert occurrences[0]["start_timestamp"] == 0.0
    assert occurrences[0]["timestamp"] == 1.0
    assert occurrences[0]["end_timestamp"] == 2.0
    assert occurrences[0]["scan_hits"] == 3
    assert occurrences[1]["timestamp"] == 8.0


def test_unknown_faces_are_ignored_and_never_create_profiles(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(PeopleStore, "cv2_write", lambda *args: True)
    _stub_event_frames(monkeypatch)
    reference = _image(tmp_path / "zhangsan.jpg")
    frame = _image(tmp_path / "mat_a_f0_1.0s.jpg")
    engine = _FakeEngine(
        {
            reference.name: [[1.0, 0.0, 0.0]],
            frame.name: [[0.0, 1.0, 0.0]],
        }
    )
    store = PeopleStore(tmp_path / "output")
    store.add_person("张三", [reference], engine)

    people_ids = store.process_material(
        "mat_a", "a.mov", "a.mov", [(1.0, frame)], engine
    )

    assert people_ids == []
    assert [profile["name"] for profile in store.list_people()] == ["张三"]
    assert store.list_people()[0]["sample_count"] == 0


def test_frame_matching_uses_global_best_assignment(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(PeopleStore, "cv2_write", lambda *args: True)
    _stub_event_frames(monkeypatch)
    first = _image(tmp_path / "first.jpg")
    second = _image(tmp_path / "second.jpg")
    frame = _image(tmp_path / "mat_a_f0_1.0s.jpg")
    engine = _FakeEngine(
        {
            first.name: [[1.0, 0.0]],
            second.name: [[0.0, 1.0]],
            frame.name: [
                [0.8, 0.6],
                [0.994987, 0.1],
            ],
        }
    )
    store = PeopleStore(tmp_path / "output")
    first_id = store.add_person("张三", [first], engine)
    second_id = store.add_person("李四", [second], engine)

    people_ids = store.process_material(
        "mat_a", "a.mov", "a.mov", [(1.0, frame)], engine
    )

    assert people_ids == [first_id, second_id]
    scores = {
        profile["id"]: profile["occurrences"][0]["match_score"]
        for profile in store.list_people()
    }
    assert scores[first_id] == 0.995
    assert scores[second_id] == 0.6


def test_add_person_requires_exactly_one_face_per_reference(tmp_path: Path):
    no_face = _image(tmp_path / "empty.jpg")
    group = _image(tmp_path / "group.jpg")
    engine = _FakeEngine(
        {
            group.name: [[1.0, 0.0], [0.0, 1.0]],
        }
    )
    store = PeopleStore(tmp_path / "output")

    try:
        store.add_person("张三", [no_face], engine)
        assert False, "expected missing-face validation"
    except ValueError as error:
        assert "未检测到清晰人脸" in str(error)

    try:
        store.add_person("张三", [group], engine)
        assert False, "expected multi-face validation"
    except ValueError as error:
        assert "必须只有一张人脸" in str(error)


def test_same_name_appends_reference_photos_and_uses_all_references(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(PeopleStore, "cv2_write", lambda *args: True)
    _stub_event_frames(monkeypatch)
    front = _image(tmp_path / "front.jpg")
    side = _image(tmp_path / "side.jpg")
    frame = _image(tmp_path / "mat_a_f0_1.0s.jpg")
    engine = _FakeEngine(
        {
            front.name: [[1.0, 0.0, 0.0]],
            side.name: [[0.0, 1.0, 0.0]],
            frame.name: [[0.0, 0.99, 0.05]],
        }
    )
    store = PeopleStore(tmp_path / "output")

    first_id = store.add_person("张三", [front], engine)
    second_id = store.add_person("张三", [side], engine)
    people_ids = store.process_material(
        "mat_a", "a.mov", "a.mov", [(1.0, frame)], engine
    )

    assert first_id == second_id == "person_0001"
    assert people_ids == ["person_0001"]
    assert store.list_people()[0]["reference_count"] == 2


def test_registered_people_support_rename_merge_and_delete(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(PeopleStore, "cv2_write", lambda *args: True)
    first = _image(tmp_path / "first.jpg")
    second = _image(tmp_path / "second.jpg")
    engine = _FakeEngine(
        {
            first.name: [[1.0, 0.0, 0.0]],
            second.name: [[0.0, 1.0, 0.0]],
        }
    )
    store = PeopleStore(tmp_path / "output")
    first_id = store.add_person("张三", [first], engine)
    second_id = store.add_person("李四", [second], engine)

    store.rename(first_id, "张老师")
    assert store.label(first_id) == "张老师"

    store.merge(second_id, first_id)
    assert [profile["id"] for profile in store.list_people()] == [first_id]
    assert store.list_people()[0]["reference_count"] == 2
    assert all(
        (store.output_dir / path).exists()
        for path in store.list_people()[0]["reference_photos"]
    )

    store.delete(first_id)
    assert store.list_people() == []


def test_identity_events_split_when_recognized_people_change(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(PeopleStore, "cv2_write", lambda *args: True)
    _stub_event_frames(monkeypatch)
    first = _image(tmp_path / "first.jpg")
    second = _image(tmp_path / "second.jpg")
    frames = [_image(tmp_path / f"frame_{index}.jpg") for index in range(4)]
    engine = _FakeEngine(
        {
            first.name: [[1.0, 0.0]],
            second.name: [[0.0, 1.0]],
            frames[0].name: [[0.99, 0.01]],
            frames[1].name: [[0.98, 0.02]],
            frames[2].name: [[0.98, 0.02], [0.02, 0.98]],
            frames[3].name: [[0.01, 0.99]],
        }
    )
    store = PeopleStore(tmp_path / "output")
    first_id = store.add_person("王书记", [first], engine)
    second_id = store.add_person("老刘", [second], engine)

    store.process_material(
        "mat_a",
        "a.mov",
        "a.mov",
        list(zip([0.0, 1.0, 2.0, 3.0], frames)),
        engine,
    )

    events = store.material_events("mat_a")
    assert [event["people_ids"] for event in events] == [
        [first_id],
        [first_id, second_id],
        [second_id],
    ]
    assert [(event["start_timestamp"], event["end_timestamp"]) for event in events] == [
        (0.0, 1.0),
        (2.0, 2.0),
        (3.0, 3.0),
    ]
    assert events[1]["labels"] == [
        {
            "tag": "P1",
            "person_id": first_id,
            "bbox": [0, 0, 80, 80],
                "match_score": 0.98,
            "name": "王书记",
        },
        {
            "tag": "P2",
            "person_id": second_id,
            "bbox": [0, 0, 80, 80],
                "match_score": 0.98,
            "name": "老刘",
        },
    ]
    assert [label["tag"] for label in events[2]["labels"]] == ["P2"]
    assert [frame["timestamp"] for frame in events[0]["frames"]] == [0.0, 1.0]
    assert all(
        (store.output_dir / frame["frame"]).is_file()
        for event in events
        for frame in event["frames"]
    )


def test_merge_and_delete_keep_identity_event_references_consistent(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(PeopleStore, "cv2_write", lambda *args: True)
    first = _image(tmp_path / "first.jpg")
    second = _image(tmp_path / "second.jpg")
    engine = _FakeEngine(
        {
            first.name: [[1.0, 0.0]],
            second.name: [[0.0, 1.0]],
        }
    )
    store = PeopleStore(tmp_path / "output")
    first_id = store.add_person("王书记", [first], engine)
    second_id = store.add_person("老刘", [second], engine)
    store.data["events"]["mat_a"] = [
        {
            "event_id": "event_001",
            "people_ids": [first_id, second_id],
            "labels": [
                {"tag": "P1", "person_id": first_id},
                {"tag": "P2", "person_id": second_id},
            ],
        }
    ]

    store.merge(second_id, first_id)
    event = store.material_events("mat_a")[0]
    assert event["people_ids"] == [first_id]
    assert [label["person_id"] for label in event["labels"]] == [first_id]

    store.delete(first_id)
    assert store.material_events("mat_a") == []


def test_legacy_auto_cluster_file_is_backed_up_and_replaced(tmp_path: Path):
    old_data = {
        "version": 1,
        "next_id": 2,
        "people": {"person_0001": {"auto_name": "人物01"}},
    }
    (tmp_path / "people.json").write_text(
        json.dumps(old_data, ensure_ascii=False),
        encoding="utf-8",
    )

    store = PeopleStore(tmp_path)

    assert store.list_people() == []
    assert (tmp_path / "people.auto_cluster.v1.json").exists()
    assert json.loads((tmp_path / "people.json").read_text(encoding="utf-8"))[
        "mode"
    ] == "registered_only"


def test_cached_people_frames_recovers_timestamps(tmp_path: Path):
    (tmp_path / "mat_test_f0_1.2s.jpg").write_bytes(b"frame")
    (tmp_path / "mat_test_f1_8.5s.jpg").write_bytes(b"frame")
    (tmp_path / "other_f0_2.0s.jpg").write_bytes(b"frame")

    frames = cached_people_frames(tmp_path, "mat_test")

    assert [(timestamp, path.name) for timestamp, path in frames] == [
        (1.2, "mat_test_f0_1.2s.jpg"),
        (8.5, "mat_test_f1_8.5s.jpg"),
    ]


def test_people_scan_points_cover_video_independently_from_vlm():
    assert people_scan_points(3.2, interval=1.0) == [0.0, 1.0, 2.0, 3.0]
    assert people_scan_points(0.2, interval=1.0) == [0.0]
    assert people_scan_points(0.0, interval=1.0) == []


def test_extract_people_frames_runs_one_dense_ffmpeg_pass(
    tmp_path: Path,
    monkeypatch,
):
    video = _image(tmp_path / "clip.mov")
    output = tmp_path / "people_frames"
    command = {}

    class Result:
        returncode = 0
        stderr = ""

    def fake_run(args, **kwargs):
        command["args"] = args
        target_dir = output / "mat_a"
        target_dir.mkdir(parents=True, exist_ok=True)
        for index in range(1, 5):
            _image(target_dir / f"frame_{index:06d}.jpg")
        return Result()

    monkeypatch.setattr("scenelog.people._find_bin", lambda _name: "ffmpeg")
    monkeypatch.setattr("scenelog.people.subprocess.run", fake_run)

    frames = extract_people_frames(
        video,
        output,
        "mat_a",
        duration=3.2,
        interval=1.0,
    )

    assert [timestamp for timestamp, _path in frames] == [0.0, 1.0, 2.0, 3.0]
    assert command["args"].count("ffmpeg") == 1
    assert "fps=1/1.000000" in command["args"][command["args"].index("-vf") + 1]


def test_people_index_supports_name_and_name_action_search(tmp_path: Path):
    profiles = [
        {
            "id": "person_0001",
            "name": "张三",
            "occurrences": [
                {
                    "material_id": "mat_a",
                    "start_timestamp": 3.0,
                    "timestamp": 4.0,
                    "end_timestamp": 6.0,
                }
            ],
        }
    ]
    build_people_index(
        tmp_path,
        "mat_a",
        "a.mov",
        "a.mov",
        profiles,
        [4.2],
        ["院子里一名男子劈柴"],
    )
    searcher = Searcher(tmp_path, tmp_path)

    by_name = searcher.search("张三")
    by_name_action = searcher.search("张三 劈柴")

    assert by_name[0]["source"] == "people"
    assert by_name[0]["start_time"] == "00:00:03"
    assert by_name[0]["end_time"] == "00:00:06"
    assert by_name_action[0]["text"] == "张三 院子里一名男子劈柴"
