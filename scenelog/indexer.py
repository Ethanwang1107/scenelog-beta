"""全文索引构建 — transcripts_index.jsonl"""

import json
import logging
import os
import tempfile
from pathlib import Path

from scenelog.config import INDEX_FILE
from scenelog.transcribe import parse_srt

logger = logging.getLogger(__name__)


def build_index(
    srt_path: Path,
    output_dir: Path,
    material_id: str,
    file_name: str,
    rel_path: str,
    speaker_segments: list[dict] | None = None,
) -> Path:
    """从 SRT 文件构建片段级全文索引。

    当前素材的语音记录会被完整替换；新转录为空时也会删除旧语音记录。

    返回索引文件路径。
    """
    segments = parse_srt(srt_path)
    segments = [s for s in segments if s["text"] != "[未识别出语音内容]"]

    speakers = {
        (
            str(segment.get("start_time", "")),
            str(segment.get("end_time", "")),
            str(segment.get("text", "")),
        ): str(segment.get("speaker_name", "")).strip()
        for segment in (speaker_segments or [])
        if segment.get("confirmed") is True
    }
    new_records = []
    for seg in segments:
        speaker = speakers.get(
            (seg["start_time"], seg["end_time"], seg["text"]),
            "",
        )
        text = f"{speaker}：{seg['text']}" if speaker else seg["text"]
        record = {
            "material_id": material_id,
            "file_name": file_name,
            "rel_path": rel_path,
            "start_time": seg["start_time"],
            "end_time": seg["end_time"],
            "text": text,
            "source": "audio",
        }
        if speaker:
            record["speaker_name"] = speaker
        new_records.append(record)

    index_path = _replace_source_records(
        output_dir, material_id, "audio", new_records
    )
    logger.info("语音索引更新完成: %d 条记录 → %s", len(new_records), index_path.name)
    return index_path


def build_visual_index(
    output_dir: Path,
    material_id: str,
    file_name: str,
    rel_path: str,
    frame_points: list[float],
    visual_descs: list[str],
) -> Path:
    """将逐帧画面描述写入全文索引，并替换当前素材的旧视觉记录。"""
    new_records = []
    for point, description in zip(frame_points, visual_descs):
        text = description.strip()
        if not text:
            continue
        timestamp = _format_seconds(point)
        new_records.append({
            "material_id": material_id,
            "file_name": file_name,
            "rel_path": rel_path,
            "start_time": timestamp,
            "end_time": timestamp,
            "text": text,
            "source": "vision",
        })

    index_path = _replace_source_records(
        output_dir, material_id, "vision", new_records
    )
    logger.info("视觉索引更新完成: %d 条记录 → %s", len(new_records), index_path.name)
    return index_path


def build_people_index(
    output_dir: Path,
    material_id: str,
    file_name: str,
    rel_path: str,
    people: list[dict],
    frame_points: list[float] | None = None,
    visual_descs: list[str] | None = None,
) -> Path:
    """写入人物出现记录，并附带相邻关键帧描述。"""
    points = frame_points or []
    descriptions = visual_descs or []
    new_records = []
    for person in people:
        label = person.get("name", "").strip()
        for occurrence in person.get("occurrences", []):
            if occurrence.get("material_id") != material_id:
                continue
            timestamp_seconds = float(occurrence.get("timestamp", 0))
            start_seconds = float(
                occurrence.get("start_timestamp", timestamp_seconds)
            )
            end_seconds = float(
                occurrence.get("end_timestamp", start_seconds)
            )
            description = _nearest_description(
                timestamp_seconds,
                points,
                descriptions,
            )
            text = " ".join(part for part in [label, description] if part)
            if not text:
                continue
            new_records.append({
                "material_id": material_id,
                "file_name": file_name,
                "rel_path": rel_path,
                "start_time": _format_seconds(start_seconds),
                "end_time": _format_seconds(end_seconds),
                "text": text,
                "source": "people",
                "person_id": person.get("id", ""),
            })

    index_path = _replace_source_records(
        output_dir, material_id, "people", new_records
    )
    logger.info("人物索引更新完成: %d 条记录 → %s", len(new_records), index_path.name)
    return index_path


def build_identity_index(
    output_dir: Path,
    material_id: str,
    file_name: str,
    rel_path: str,
    events: list[dict],
) -> Path:
    """Write identity-bound event descriptions as timestamped search records."""
    new_records = []
    for event in events:
        description = event.get("description", {})
        names = [
            person.get("name", "").strip()
            for person in event.get("people", [])
            if person.get("name", "").strip()
        ]
        action_parts = []
        for action_event in description.get("动作事件", []):
            if not isinstance(action_event, dict):
                continue
            action_parts.extend(
                str(value).strip()
                for key in ("施动者", "动作", "承受者")
                for value in (
                    action_event.get(key, [])
                    if isinstance(action_event.get(key, []), list)
                    else [action_event.get(key, "")]
                )
                if str(value).strip()
            )
        text = " ".join(
            dict.fromkeys(
                part
                for part in [
                    *names,
                    *action_parts,
                    description.get("动作", "").strip(),
                    description.get("场景", "").strip(),
                    description.get("对白事实", "").strip(),
                ]
                if part
            )
        )
        if not text:
            continue
        new_records.append(
            {
                "material_id": material_id,
                "file_name": file_name,
                "rel_path": rel_path,
                "start_time": _format_seconds(
                    float(event.get("start_timestamp", 0))
                ),
                "end_time": _format_seconds(
                    float(event.get("end_timestamp", 0))
                ),
                "text": text,
                "source": "identity",
                "people_ids": list(event.get("people_ids", [])),
                "event_id": event.get("event_id", ""),
            }
        )

    index_path = _replace_source_records(
        output_dir, material_id, "identity", new_records
    )
    logger.info("身份事件索引更新完成: %d 条记录 → %s", len(new_records), index_path.name)
    return index_path


def remove_material_from_index(output_dir: Path, material_id: str) -> Path:
    """删除某素材的全部语音、视觉、人物和身份事件索引记录。"""
    index_path = _replace_source_records(output_dir, material_id, "audio", [])
    index_path = _replace_source_records(output_dir, material_id, "vision", [])
    index_path = _replace_source_records(output_dir, material_id, "people", [])
    return _replace_source_records(output_dir, material_id, "identity", [])


def _replace_source_records(
    output_dir: Path,
    material_id: str,
    source: str,
    new_records: list[dict],
) -> Path:
    """原子替换某个素材、某种来源的全部索引记录。"""
    index_path = output_dir / INDEX_FILE
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(output_dir), prefix=".index_", suffix=".tmp"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            if index_path.exists():
                with open(index_path, "r", encoding="utf-8") as old:
                    for line in old:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                            same_record_set = (
                                rec.get("material_id") == material_id
                                and rec.get("source", "audio") == source
                            )
                            if not same_record_set:
                                f.write(line + "\n")
                        except json.JSONDecodeError:
                            continue

            for rec in new_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        os.replace(tmp_path, index_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    return index_path


def _format_seconds(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _nearest_description(
    timestamp: float,
    frame_points: list[float],
    visual_descs: list[str],
) -> str:
    candidates = [
        (abs(float(point) - timestamp), description.strip())
        for point, description in zip(frame_points, visual_descs)
        if description.strip()
    ]
    if not candidates:
        return ""
    return min(candidates, key=lambda item: item[0])[1]
