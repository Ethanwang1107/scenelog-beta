"""本地人物识别与档案管理 — OpenCV YuNet + SFace。"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from scenelog.config import (
    PEOPLE_DETECTION_SCORE,
    PEOPLE_DETECTOR_MODEL,
    PEOPLE_DIR,
    PEOPLE_FILE,
    PEOPLE_MATCH_THRESHOLD,
    PEOPLE_MIN_FACE_SIZE,
    PEOPLE_OCCURRENCE_GAP_SECONDS,
    PEOPLE_RECOGNIZER_MODEL,
    PEOPLE_SCAN_INTERVAL_SECONDS,
)
from scenelog.media import _find_bin

logger = logging.getLogger(__name__)

MODEL_URLS = {
    Path(PEOPLE_DETECTOR_MODEL): (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
    Path(PEOPLE_RECOGNIZER_MODEL): (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec.onnx"
    ),
}


def missing_models() -> list[Path]:
    """返回尚未安装的人脸模型。"""
    return [path for path in MODEL_URLS if not path.expanduser().is_file()]


def install_models() -> list[Path]:
    """下载 OpenCV 官方人脸模型，返回安装路径。"""
    installed = []
    for configured_path, url in MODEL_URLS.items():
        path = configured_path.expanduser()
        if path.is_file():
            installed.append(path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("下载人物模型: %s", path.name)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "wb") as target, urlopen(url, timeout=120) as source:
                shutil.copyfileobj(source, target)
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
        installed.append(path)
    return installed


class FaceEngine:
    """从关键帧检测人脸并提取归一化 SFace 特征。"""

    def __init__(self):
        missing = missing_models()
        if missing:
            names = ", ".join(path.name for path in missing)
            raise RuntimeError(f"缺少人物模型: {names}，请先执行 scenelog people setup")

        import cv2

        self.cv2 = cv2
        self.detector = cv2.FaceDetectorYN.create(
            str(Path(PEOPLE_DETECTOR_MODEL).expanduser()),
            "",
            (320, 320),
            score_threshold=PEOPLE_DETECTION_SCORE,
            nms_threshold=0.3,
            top_k=5000,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(
            str(Path(PEOPLE_RECOGNIZER_MODEL).expanduser()),
            "",
        )

    def extract(self, frame_path: Path) -> list[dict]:
        """返回一帧内满足质量要求的人脸框和特征。"""
        image = self.cv2.imread(str(frame_path))
        if image is None:
            logger.warning("人物识别无法读取关键帧: %s", frame_path)
            return []
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(image)
        if faces is None:
            return []

        detections = []
        for face in faces:
            x, y, w, h = [int(round(value)) for value in face[:4]]
            if min(w, h) < PEOPLE_MIN_FACE_SIZE:
                continue
            try:
                aligned = self.recognizer.alignCrop(image, face)
                feature = self.recognizer.feature(aligned).flatten().astype(float)
            except Exception as exc:
                logger.debug("人脸特征提取失败: %s", exc)
                continue
            feature = _normalize(feature)
            if not feature:
                continue
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(width, x + w), min(height, y + h)
            crop = image[y1:y2, x1:x2]
            detections.append(
                {
                    "bbox": [x1, y1, max(0, x2 - x1), max(0, y2 - y1)],
                    "score": float(face[-1]),
                    "embedding": feature,
                    "crop": crop,
                }
            )
        return detections


class PeopleStore:
    """持久化用户预登记的关键人物和素材匹配记录。"""

    DATA_VERSION = 3

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.path = output_dir / PEOPLE_FILE
        self.people_dir = output_dir / PEOPLE_DIR
        self.data = {
            "version": self.DATA_VERSION,
            "mode": "registered_only",
            "next_id": 1,
            "people": {},
            "events": {},
        }
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("人物档案读取失败，将使用空档案: %s", self.path)
            return
        if loaded.get("version") in (2, self.DATA_VERSION) and isinstance(
            loaded.get("people"), dict
        ):
            loaded.setdefault("events", {})
            loaded["version"] = self.DATA_VERSION
            for person in loaded["people"].values():
                person.setdefault("voice_references", [])
                person.setdefault("voice_embeddings", [])
                person.setdefault("voice_duration", 0.0)
            self.data = loaded
            return

        backup_path = self.output_dir / "people.auto_cluster.v1.json"
        if not backup_path.exists():
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.path, backup_path)
        logger.warning(
            "检测到旧版自动聚类人物档案，已备份为 %s；新版仅识别预登记人物",
            backup_path.name,
        )
        self._save()

    def _save(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(self.output_dir), prefix=".people_", suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
                json.dump(self.data, handle, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def has_registered_people(self) -> bool:
        return bool(self.data["people"])

    def add_person(
        self,
        name: str,
        photo_paths: list[Path],
        engine: FaceEngine,
    ) -> str:
        """登记关键人物；同名时追加参考照片。"""
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("人物姓名不能为空")
        if not photo_paths:
            raise ValueError("至少需要一张人物参考照片")

        extracted = []
        for photo_path in photo_paths:
            path = photo_path.expanduser().resolve()
            if not path.is_file():
                raise ValueError(f"参考照片不存在: {photo_path}")
            detections = engine.extract(path)
            if not detections:
                raise ValueError(f"参考照片未检测到清晰人脸: {path.name}")
            if len(detections) != 1:
                raise ValueError(
                    f"参考照片必须只有一张人脸: {path.name}（检测到 {len(detections)} 张）"
                )
            extracted.append((path, detections[0]))

        person_id = self._find_by_name(normalized_name)
        if person_id:
            person = self.data["people"][person_id]
        else:
            number = int(self.data.get("next_id", 1))
            person_id = f"person_{number:04d}"
            self.data["next_id"] = number + 1
            person = {
                "id": person_id,
                "name": normalized_name,
                "reference_embeddings": [],
                "reference_photos": [],
                "embedding": [],
                "thumbnail": "",
                "occurrences": [],
                "voice_references": [],
                "voice_embeddings": [],
                "voice_duration": 0.0,
            }
            self.data["people"][person_id] = person

        person_dir = self.people_dir / person_id
        references_dir = person_dir / "references"
        references_dir.mkdir(parents=True, exist_ok=True)
        start_number = len(person.get("reference_photos", [])) + 1
        for offset, (source_path, detection) in enumerate(extracted):
            suffix = source_path.suffix.lower() or ".jpg"
            target = references_dir / f"reference_{start_number + offset:03d}{suffix}"
            shutil.copy2(source_path, target)
            person.setdefault("reference_photos", []).append(
                str(target.relative_to(self.output_dir))
            )
            person.setdefault("reference_embeddings", []).append(
                detection["embedding"]
            )

        person["name"] = normalized_name
        self._recompute_reference(person)
        if not person.get("thumbnail"):
            thumbnail_path = self.people_dir / f"{person_id}.jpg"
            crop = extracted[0][1].get("crop")
            if crop is not None and getattr(crop, "size", 0):
                if self.cv2_write(thumbnail_path, crop):
                    person["thumbnail"] = str(
                        thumbnail_path.relative_to(self.output_dir)
                    )
        self._save()
        return person_id

    def add_voice_samples(
        self,
        person_id: str,
        audio_paths: list[Path],
        engine,
    ):
        """Add cleaned single-speaker references to an existing person."""
        from scenelog.speaker import prepare_reference_audio

        if not audio_paths:
            raise ValueError("至少需要一段人物声音")
        person = self._require(person_id)
        voice_dir = self.people_dir / person_id / "voices"
        voice_dir.mkdir(parents=True, exist_ok=True)
        start_number = len(person.get("voice_references", [])) + 1
        added_paths = []
        added_embeddings = []
        added_duration = 0.0
        try:
            for offset, source_path in enumerate(audio_paths):
                target = voice_dir / f"voice_{start_number + offset:03d}.wav"
                duration = prepare_reference_audio(source_path, target)
                added_paths.append(str(target.relative_to(self.output_dir)))
                embedding = engine.extract(
                    target,
                    minimum_seconds=1.5,
                )
                added_embeddings.append(embedding)
                added_duration += duration
        except Exception:
            for relative in added_paths:
                (self.output_dir / relative).unlink(missing_ok=True)
            raise

        person.setdefault("voice_references", []).extend(added_paths)
        person.setdefault("voice_embeddings", []).extend(added_embeddings)
        person["voice_duration"] = round(
            float(person.get("voice_duration", 0.0)) + added_duration,
            2,
        )
        self._save()

    def process_material(
        self,
        material_id: str,
        file_name: str,
        rel_path: str,
        frames: list[tuple[float, Path]],
        engine: FaceEngine,
    ) -> list[str]:
        """只匹配已登记关键人物，未知人脸直接忽略。"""
        self.remove_material(material_id, save=False)
        if not self.has_registered_people():
            self._save()
            return []

        people_ids = []
        last_occurrences: dict[str, dict] = {}
        observations = []
        for timestamp, frame_path in frames:
            detections = engine.extract(frame_path)
            frame_matches = self._match_frame(detections)
            if frame_matches:
                observations.append(
                    {
                        "timestamp": float(timestamp),
                        "frame_path": frame_path,
                        "face_count": len(detections),
                        "matches": [
                            {
                                "person_id": person_id,
                                "bbox": detection["bbox"],
                                "match_score": float(match_score),
                                "detection_score": float(detection["score"]),
                            }
                            for detection, person_id, match_score in frame_matches
                        ],
                    }
                )
            for detection, person_id, match_score in frame_matches:
                person = self.data["people"][person_id]
                occurrence = {
                    "material_id": material_id,
                    "file_name": file_name,
                    "rel_path": rel_path,
                    "start_timestamp": float(timestamp),
                    "timestamp": float(timestamp),
                    "frame": frame_path.name,
                    "bbox": detection["bbox"],
                    "detection_score": round(detection["score"], 4),
                    "match_score": round(match_score, 4),
                }
                previous = last_occurrences.get(person_id)
                if (
                    previous
                    and timestamp - previous.get(
                        "end_timestamp",
                        previous["timestamp"],
                    )
                    <= PEOPLE_OCCURRENCE_GAP_SECONDS
                ):
                    previous_end = float(timestamp)
                    previous["end_timestamp"] = float(timestamp)
                    previous["scan_hits"] = previous.get("scan_hits", 1) + 1
                    if occurrence["match_score"] > previous["match_score"]:
                        previous.update(
                            {
                                "timestamp": occurrence["timestamp"],
                                "frame": occurrence["frame"],
                                "bbox": occurrence["bbox"],
                                "detection_score": occurrence["detection_score"],
                                "match_score": occurrence["match_score"],
                            }
                        )
                        previous["end_timestamp"] = previous_end
                else:
                    occurrence["end_timestamp"] = float(timestamp)
                    occurrence["scan_hits"] = 1
                    person.setdefault("occurrences", []).append(occurrence)
                    last_occurrences[person_id] = occurrence
                if person_id not in people_ids:
                    people_ids.append(person_id)

        self._save_material_events(material_id, observations)
        self._save()
        return people_ids

    def remove_material(self, material_id: str, save: bool = True):
        """移除某素材的旧匹配记录，但始终保留预登记人物。"""
        changed = False
        for person in self.data["people"].values():
            old = person.get("occurrences", [])
            current = [
                occurrence
                for occurrence in old
                if occurrence.get("material_id") != material_id
            ]
            if len(current) != len(old):
                person["occurrences"] = current
                changed = True
        if self.data.setdefault("events", {}).pop(material_id, None) is not None:
            changed = True
        shutil.rmtree(self.people_dir / "events" / material_id, ignore_errors=True)
        if save and changed:
            self._save()

    def list_people(self) -> list[dict]:
        result = []
        for person_id in sorted(self.data["people"], key=_person_number):
            person = self.data["people"][person_id]
            result.append(
                {
                    "id": person_id,
                    "name": person.get("name", person_id),
                    "reference_count": len(person.get("reference_photos", [])),
                    "sample_count": len(person.get("occurrences", [])),
                    "material_count": len(
                        {
                            occurrence.get("material_id")
                            for occurrence in person.get("occurrences", [])
                        }
                    ),
                    "thumbnail": person.get("thumbnail", ""),
                    "reference_photos": person.get("reference_photos", []),
                    "occurrences": person.get("occurrences", []),
                    "voice_count": len(person.get("voice_references", [])),
                    "voice_duration": float(person.get("voice_duration", 0.0)),
                    "voice_references": person.get("voice_references", []),
                    "voice_embeddings": person.get("voice_embeddings", []),
                }
            )
        return result

    def label(self, person_id: str) -> str:
        person = self.data["people"].get(person_id, {})
        return person.get("name") or person_id

    def labels(self, person_ids: list[str]) -> list[str]:
        return [
            self.label(person_id)
            for person_id in person_ids
            if person_id in self.data["people"]
        ]

    def rename(self, person_id: str, name: str):
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("人物姓名不能为空")
        duplicate_id = self._find_by_name(normalized_name)
        if duplicate_id and duplicate_id != person_id:
            raise ValueError(f"人物姓名已存在: {normalized_name}")
        person = self._require(person_id)
        person["name"] = normalized_name
        self._save()

    def merge(self, source_id: str, target_id: str):
        if source_id == target_id:
            return
        source = self._require(source_id)
        target = self._require(target_id)
        source_photos = source.get("reference_photos", [])
        source_embeddings = source.get("reference_embeddings", [])
        target_dir = self.people_dir / target_id / "references"
        target_dir.mkdir(parents=True, exist_ok=True)
        next_number = len(target.get("reference_photos", [])) + 1
        for offset, (photo, embedding) in enumerate(
            zip(source_photos, source_embeddings)
        ):
            source_path = self.output_dir / photo
            suffix = source_path.suffix.lower() or ".jpg"
            target_path = target_dir / f"reference_{next_number + offset:03d}{suffix}"
            if source_path.exists():
                shutil.copy2(source_path, target_path)
            target.setdefault("reference_photos", []).append(
                str(target_path.relative_to(self.output_dir))
            )
            target.setdefault("reference_embeddings", []).append(embedding)
        target.setdefault("occurrences", []).extend(source.get("occurrences", []))
        source_voices = source.get("voice_references", [])
        source_voice_embeddings = source.get("voice_embeddings", [])
        target_voice_dir = self.people_dir / target_id / "voices"
        target_voice_dir.mkdir(parents=True, exist_ok=True)
        next_voice_number = len(target.get("voice_references", [])) + 1
        for offset, (voice, embedding) in enumerate(
            zip(source_voices, source_voice_embeddings)
        ):
            source_path = self.output_dir / voice
            target_path = (
                target_voice_dir / f"voice_{next_voice_number + offset:03d}.wav"
            )
            if source_path.exists():
                shutil.copy2(source_path, target_path)
                target.setdefault("voice_references", []).append(
                    str(target_path.relative_to(self.output_dir))
                )
                target.setdefault("voice_embeddings", []).append(embedding)
        target["voice_duration"] = round(
            float(target.get("voice_duration", 0.0))
            + float(source.get("voice_duration", 0.0)),
            2,
        )
        self._recompute_reference(target)
        self._replace_event_person(source_id, target_id)
        self._delete_profile(source_id)
        self._save()

    def delete(self, person_id: str):
        self._require(person_id)
        self._replace_event_person(person_id, None)
        self._delete_profile(person_id)
        self._save()

    def material_people(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for person_id, person in self.data["people"].items():
            for occurrence in person.get("occurrences", []):
                material_id = occurrence.get("material_id", "")
                if material_id:
                    result.setdefault(material_id, [])
                    if person_id not in result[material_id]:
                        result[material_id].append(person_id)
        return result

    def people_for_material(self, material_id: str) -> list[str]:
        return self.material_people().get(material_id, [])

    def material_events(self, material_id: str) -> list[dict]:
        """Return chronological identity events with current profile names."""
        events = self.data.setdefault("events", {}).get(material_id, [])
        result = []
        for event in events:
            current = dict(event)
            current["people"] = [
                {
                    "id": person_id,
                    "name": self.label(person_id),
                }
                for person_id in event.get("people_ids", [])
                if person_id in self.data["people"]
            ]
            current["labels"] = [
                {
                    **label,
                    "name": self.label(label.get("person_id", "")),
                }
                for label in event.get("labels", [])
                if label.get("person_id") in self.data["people"]
            ]
            current["frames"] = [
                {
                    **frame,
                    "labels": [
                        {
                            **label,
                            "name": self.label(label.get("person_id", "")),
                        }
                        for label in frame.get("labels", [])
                        if label.get("person_id") in self.data["people"]
                    ],
                }
                for frame in event.get("frames", [])
            ]
            result.append(current)
        return sorted(result, key=lambda item: item.get("start_timestamp", 0))

    def _save_material_events(
        self,
        material_id: str,
        observations: list[dict],
    ):
        events_root = self.people_dir / "events" / material_id
        shutil.rmtree(events_root, ignore_errors=True)
        self.data.setdefault("events", {})[material_id] = []
        if not observations:
            return

        grouped = []
        for observation in sorted(
            observations,
            key=lambda item: item["timestamp"],
        ):
            person_set = {
                match["person_id"]
                for match in observation["matches"]
            }
            previous_person_set = (
                {
                    match["person_id"]
                    for match in grouped[-1][-1]["matches"]
                }
                if grouped
                else set()
            )
            if (
                not grouped
                or observation["timestamp"] - grouped[-1][-1]["timestamp"]
                > PEOPLE_OCCURRENCE_GAP_SECONDS
                or person_set != previous_person_set
            ):
                grouped.append([observation])
            else:
                grouped[-1].append(observation)

        events_root.mkdir(parents=True, exist_ok=True)
        events = []
        material_person_ids = sorted(
            {
                match["person_id"]
                for observation in observations
                for match in observation["matches"]
            },
            key=_person_number,
        )
        stable_tags = {
            person_id: f"P{index}"
            for index, person_id in enumerate(material_person_ids, 1)
        }
        for event_index, group in enumerate(grouped, 1):
            representative = max(
                group,
                key=lambda item: (
                    len(item["matches"]),
                    sum(match["match_score"] for match in item["matches"]),
                ),
            )
            people_ids = sorted(
                {
                    match["person_id"]
                    for observation in group
                    for match in observation["matches"]
                },
                key=_person_number,
            )
            labels = _event_labels(representative["matches"], stable_tags)
            target = events_root / f"event_{event_index:03d}.jpg"
            _write_annotated_event_frame(
                representative["frame_path"],
                target,
                labels,
            )
            sample_frames = []
            for sample_index, observation in enumerate(
                _sample_event_observations(group, representative),
                1,
            ):
                sample_labels = _event_labels(
                    observation["matches"],
                    stable_tags,
                )
                sample_target = (
                    events_root
                    / f"event_{event_index:03d}_frame_{sample_index:02d}.jpg"
                )
                raw_target = (
                    events_root
                    / f"event_{event_index:03d}_raw_{sample_index:02d}.jpg"
                )
                shutil.copy2(observation["frame_path"], raw_target)
                _write_annotated_event_frame(
                    observation["frame_path"],
                    sample_target,
                    sample_labels,
                )
                sample_frames.append(
                    {
                        "timestamp": observation["timestamp"],
                        "frame": str(sample_target.relative_to(self.output_dir)),
                        "raw_frame": str(raw_target.relative_to(self.output_dir)),
                        "labels": sample_labels,
                        "face_count": observation["face_count"],
                    }
                )
            events.append(
                {
                    "event_id": f"{material_id}_event_{event_index:03d}",
                    "start_timestamp": group[0]["timestamp"],
                    "end_timestamp": group[-1]["timestamp"],
                    "representative_timestamp": representative["timestamp"],
                    "people_ids": people_ids,
                    "labels": labels,
                    "unknown_face_count": max(
                        max(0, item["face_count"] - len(item["matches"]))
                        for item in group
                    ),
                    "frame": str(target.relative_to(self.output_dir)),
                    "frames": sample_frames,
                    "scan_frame_count": len(group),
                }
            )
        self.data["events"][material_id] = events

    def _match_frame(
        self,
        detections: list[dict],
    ) -> list[tuple[dict, str, float]]:
        """按全局最高相似度为同帧人脸分配不同的登记人物。"""
        candidates = []
        for detection_index, detection in enumerate(detections):
            for person_id, person in self.data["people"].items():
                references = person.get("reference_embeddings", [])
                if not references:
                    continue
                score = max(
                    _cosine(detection["embedding"], reference)
                    for reference in references
                )
                if score >= PEOPLE_MATCH_THRESHOLD:
                    candidates.append((score, detection_index, person_id))

        matched_detections = set()
        matched_people = set()
        matches = []
        for score, detection_index, person_id in sorted(
            candidates,
            reverse=True,
        ):
            if (
                detection_index in matched_detections
                or person_id in matched_people
            ):
                continue
            matched_detections.add(detection_index)
            matched_people.add(person_id)
            matches.append((detections[detection_index], person_id, score))
        return matches

    def _recompute_reference(self, person: dict):
        embeddings = [
            embedding
            for embedding in person.get("reference_embeddings", [])
            if embedding
        ]
        if not embeddings:
            person["embedding"] = []
            return
        centroid = np.asarray(embeddings, dtype=np.float32).mean(axis=0)
        person["embedding"] = _normalize(centroid)

    def _find_by_name(self, name: str) -> str:
        normalized = name.casefold()
        for person_id, person in self.data["people"].items():
            if str(person.get("name", "")).casefold() == normalized:
                return person_id
        return ""

    @staticmethod
    def cv2_write(path: Path, crop) -> bool:
        import cv2

        path.parent.mkdir(parents=True, exist_ok=True)
        return bool(cv2.imwrite(str(path), crop))

    def _delete_profile(self, person_id: str):
        person = self.data["people"].pop(person_id, None)
        if not person:
            return
        thumbnail = person.get("thumbnail")
        if thumbnail:
            (self.output_dir / thumbnail).unlink(missing_ok=True)
        shutil.rmtree(self.people_dir / person_id, ignore_errors=True)

    def _replace_event_person(
        self,
        source_id: str,
        target_id: str | None,
    ):
        """Replace or remove a profile ID from persisted identity events."""
        empty_materials = []
        for material_id, events in self.data.setdefault("events", {}).items():
            retained_events = []
            for event in events:
                people_ids = []
                for person_id in event.get("people_ids", []):
                    replacement = target_id if person_id == source_id else person_id
                    if replacement and replacement not in people_ids:
                        people_ids.append(replacement)
                labels = []
                seen_labels = set()
                for label in event.get("labels", []):
                    person_id = label.get("person_id")
                    replacement = target_id if person_id == source_id else person_id
                    if not replacement or replacement in seen_labels:
                        continue
                    labels.append({**label, "person_id": replacement})
                    seen_labels.add(replacement)
                frames = []
                for frame in event.get("frames", []):
                    frame_labels = []
                    frame_seen = set()
                    for label in frame.get("labels", []):
                        person_id = label.get("person_id")
                        replacement = (
                            target_id if person_id == source_id else person_id
                        )
                        if not replacement or replacement in frame_seen:
                            continue
                        frame_labels.append(
                            {**label, "person_id": replacement}
                        )
                        frame_seen.add(replacement)
                    frames.append({**frame, "labels": frame_labels})
                if people_ids:
                    retained_events.append(
                        {
                            **event,
                            "people_ids": people_ids,
                            "labels": labels,
                            "frames": frames,
                        }
                    )
            if retained_events:
                self.data["events"][material_id] = retained_events
            else:
                empty_materials.append(material_id)
        for material_id in empty_materials:
            self.data["events"].pop(material_id, None)

    def _require(self, person_id: str) -> dict:
        try:
            return self.data["people"][person_id]
        except KeyError as exc:
            raise ValueError(f"人物不存在: {person_id}") from exc


def _normalize(values) -> list[float]:
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0:
        return []
    return (vector / norm).astype(float).tolist()


def _cosine(first: list[float], second: list[float]) -> float:
    if len(first) != len(second) or not first:
        return -1.0
    return float(np.dot(first, second))


def _person_number(person_id: str) -> int:
    try:
        return int(person_id.rsplit("_", 1)[-1])
    except ValueError:
        return 0


def _event_labels(
    matches: list[dict],
    stable_tags: dict[str, str],
) -> list[dict]:
    labels = []
    for match in sorted(
        matches,
        key=lambda item: (
            item["bbox"][0],
            item["bbox"][1],
            _person_number(item["person_id"]),
        ),
    ):
        labels.append(
            {
                "tag": stable_tags[match["person_id"]],
                "person_id": match["person_id"],
                "bbox": match["bbox"],
                "match_score": round(match["match_score"], 4),
            }
        )
    return labels


def _sample_event_observations(
    group: list[dict],
    representative: dict,
    limit: int = 3,
) -> list[dict]:
    """Keep chronological start/middle/end evidence and the best face frame."""
    if len(group) <= limit:
        return list(group)
    indexes = {0, len(group) // 2, len(group) - 1}
    representative_index = group.index(representative)
    if representative_index not in indexes:
        middle = min(
            indexes,
            key=lambda index: abs(index - representative_index),
        )
        indexes.remove(middle)
        indexes.add(representative_index)
    return [group[index] for index in sorted(indexes)]


def _write_annotated_event_frame(
    source_path: Path,
    target_path: Path,
    labels: list[dict],
) -> None:
    """Draw identity labels and enlarge the surrounding body-action area."""
    import cv2

    image = cv2.imread(str(source_path))
    if image is None:
        raise RuntimeError(f"无法读取人物事件代表帧: {source_path}")
    for label in labels:
        x, y, width, height = [int(value) for value in label["bbox"]]
        tag = label["tag"]
        color = (48, 196, 255)
        cv2.rectangle(image, (x, y), (x + width, y + height), color, 3)
        text_size, baseline = cv2.getTextSize(
            tag,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            2,
        )
        text_width, text_height = text_size
        text_top = max(0, y - text_height - baseline - 8)
        cv2.rectangle(
            image,
            (x, text_top),
            (x + text_width + 12, y),
            color,
            -1,
        )
        cv2.putText(
            image,
            tag,
            (x + 6, y - baseline - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
    if labels:
        image = _identity_action_composite(image, labels)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target_path), image):
        raise RuntimeError(f"无法保存人物事件代表帧: {target_path}")


def _identity_action_composite(image, labels: list[dict]):
    """Pair the full frame with an enlarged crop around registered bodies."""
    import cv2

    image_height, image_width = image.shape[:2]
    left = cv2.resize(image, (640, 640), interpolation=cv2.INTER_AREA)
    boxes = [
        [int(value) for value in label["bbox"]]
        for label in labels
        if len(label.get("bbox", [])) == 4
    ]
    if not boxes:
        return left

    x_min = min(x - 4 * width for x, _y, width, _height in boxes)
    y_min = min(y - 2 * height for _x, y, _width, height in boxes)
    x_max = max(x + 5 * width for x, _y, width, _height in boxes)
    y_max = max(y + 5 * height for _x, y, _width, height in boxes)
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image_width, x_max)
    y_max = min(image_height, y_max)
    crop = image[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return left

    right = cv2.resize(crop, (640, 640), interpolation=cv2.INTER_CUBIC)
    composite = cv2.hconcat([left, right])
    cv2.rectangle(composite, (0, 0), (1280, 54), (20, 20, 20), -1)
    color = (48, 196, 255)
    cv2.putText(
        composite,
        "FULL VIEW",
        (18, 39),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        composite,
        "IDENTITY BODY AREA ENLARGED",
        (660, 39),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        color,
        2,
        cv2.LINE_AA,
    )
    return composite


def people_scan_points(
    duration: float,
    interval: float = PEOPLE_SCAN_INTERVAL_SECONDS,
) -> list[float]:
    """Generate dense timestamps independently from VLM keyframes."""
    if duration <= 0:
        return []
    points = []
    timestamp = 0.0
    while timestamp < duration:
        points.append(round(timestamp, 3))
        timestamp += interval
    return points


def extract_people_frames(
    video_path: Path,
    output_dir: Path,
    material_id: str,
    duration: float,
    interval: float = PEOPLE_SCAN_INTERVAL_SECONDS,
) -> list[tuple[float, Path]]:
    """Extract one frame per interval for face matching in a single ffmpeg run."""
    points = people_scan_points(duration, interval)
    if not points:
        return []

    material_dir = output_dir / material_id
    shutil.rmtree(material_dir, ignore_errors=True)
    material_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = material_dir / "frame_%06d.jpg"
    command = [
        _find_bin("ffmpeg"),
        "-y",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        (
            f"fps=1/{interval:.6f},"
            "scale=1280:-2:force_original_aspect_ratio=decrease"
        ),
        "-q:v",
        "3",
        str(output_pattern),
    ]
    timeout = max(120, int(duration * 3))
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        shutil.rmtree(material_dir, ignore_errors=True)
        raise RuntimeError("人物扫描抽帧超时") from exc
    if result.returncode != 0:
        shutil.rmtree(material_dir, ignore_errors=True)
        raise RuntimeError(f"人物扫描抽帧失败: {result.stderr.strip()}")

    paths = sorted(material_dir.glob("frame_*.jpg"))
    frames = [
        (points[index] if index < len(points) else round(index * interval, 3), path)
        for index, path in enumerate(paths)
    ]
    if not frames:
        shutil.rmtree(material_dir, ignore_errors=True)
        raise RuntimeError("人物扫描未抽取到有效画面")
    return frames


def cached_people_frames(
    cache_dir: Path,
    material_id: str,
) -> list[tuple[float, Path]]:
    """Read timestamped frame files kept by older versions."""
    frames = []
    for path in sorted(cache_dir.glob(f"{material_id}_f*_*.jpg")):
        try:
            timestamp = float(path.stem.rsplit("_", 1)[-1].removesuffix("s"))
        except ValueError:
            continue
        frames.append((timestamp, path))
    return frames
