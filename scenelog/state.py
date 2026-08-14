"""步骤级状态管理 — JSONL 持久化 + 原子写入"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from scenelog.config import PIPELINE_VERSION, STATE_FILE

logger = logging.getLogger(__name__)

# 所有处理步骤
ALL_STEPS = [
    "metadata",
    "audio_extract",
    "vad",
    "transcription",
    "speaker",
    "vision",
    "people",
    "summary",
    "index",
    "excel",
]

# 有效状态枚举
VALID_STATUSES = {
    "pending", "processing", "success", "skipped",
    "failed_retryable", "failed_terminal", "stale",
}


class StateManager:
    """管理 material_id → 步骤级状态的 JSONL 持久化存储。"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.state_path = output_dir / STATE_FILE
        self._states: dict[str, dict] = {}
        self._load()

    def _load(self):
        """从 JSONL 加载所有状态到内存。"""
        if not self.state_path.exists():
            return
        with open(self.state_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    mid = record.get("material_id")
                    if mid:
                        self._states[mid] = record
                except json.JSONDecodeError:
                    logger.warning("状态文件行解析失败，已跳过")

    def _save(self):
        """原子写入全部状态到 JSONL。"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(self.output_dir), prefix=".state_", suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                for record in self._states.values():
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            os.replace(tmp_path, self.state_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def get(self, material_id: str) -> dict:
        """获取某素材的状态记录，不存在则返回默认模板。"""
        if material_id not in self._states:
            return self._default_state(material_id)
        return self._states[material_id]

    def has_record(self, material_id: str) -> bool:
        """判断素材是否已有持久化状态。"""
        return material_id in self._states

    def find_by_rel_path(self, rel_path: str) -> Optional[str]:
        """按相对路径查找 material_id。"""
        for material_id, record in self._states.items():
            if record.get("rel_path") == rel_path:
                return material_id
        return None

    def get_step(self, material_id: str, step: str) -> str:
        """获取某素材某步骤的状态。"""
        record = self.get(material_id)
        return record.get(step, "pending")

    def set_step(self, material_id: str, step: str, status: str, error: str = ""):
        """设置某素材某步骤的状态并持久化。"""
        if step not in ALL_STEPS:
            raise ValueError(f"无效步骤: {step}")
        if status not in VALID_STATUSES:
            raise ValueError(f"无效状态: {status}")

        record = self._states.setdefault(material_id, self._default_state(material_id))
        record[step] = status
        errors = record.setdefault("errors", {})
        if error:
            errors[step] = error
            record["last_error"] = error
        elif status in ("success", "skipped", "pending", "stale"):
            errors.pop(step, None)
            record["last_error"] = list(errors.values())[-1] if errors else ""

        # 更新 attempts
        if status in ("failed_retryable", "failed_terminal"):
            record["attempts"] = record.get("attempts", 0) + 1

        self._save()

    def set_step_bulk(self, material_id: str, updates: dict[str, str]):
        """批量设置步骤状态，一次写入。"""
        record = self._states.setdefault(material_id, self._default_state(material_id))
        for step, status in updates.items():
            if step not in ALL_STEPS:
                raise ValueError(f"无效步骤: {step}")
            if status not in VALID_STATUSES:
                raise ValueError(f"无效状态: {status}")
            record[step] = status
        self._save()

    def init_record(self, material_id: str, meta: dict):
        """初始化素材状态记录（仅当不存在时）。"""
        if material_id not in self._states:
            record = self._default_state(material_id)
            record.update({
                "source_fingerprint": meta.get("fingerprint", ""),
                "pipeline_version": PIPELINE_VERSION,
                "file_name": meta.get("file_name", ""),
                "rel_path": meta.get("rel_path", ""),
            })
            self._states[material_id] = record
            self._save()

    def invalidate_steps(self, material_id: str, steps: list[str]):
        """将步骤标记为 stale，并清除不能继续复用的派生结果。"""
        record = self._states.setdefault(material_id, self._default_state(material_id))
        errors = record.setdefault("errors", {})
        for step in steps:
            if step not in ALL_STEPS:
                raise ValueError(f"无效步骤: {step}")
            record[step] = "stale"
            errors.pop(step, None)

        if "vision" in steps:
            record["vision_summary"] = ""
            record["vision_keywords"] = []
            record["vision_descriptions"] = []
            record["vision_frame_count"] = 0
            record["vision_points"] = []
            record["vision_model"] = ""
        if "speaker" in steps:
            record["speaker_segments"] = []
            record["speaker_transcript_srt"] = ""
            record["speaker_transcript_txt"] = ""
        if "people" in steps:
            record["people_ids"] = []
            record["identity_events"] = []
            record["identity_summary_text"] = ""
            record["identity_summary_keywords"] = []
            record["summary_text"] = record.get("base_summary_text", "")
        if "summary" in steps:
            record["base_summary_text"] = ""
            record["summary_text"] = ""
            record["summary_keywords"] = []
            record["identity_events"] = []
            record["identity_summary_text"] = ""
            record["identity_summary_keywords"] = []

        record["last_error"] = list(errors.values())[-1] if errors else ""
        self._save()

    def prune(self, active_material_ids: set[str]) -> list[str]:
        """删除素材目录中已不存在的状态记录，返回被删除的 material_id。"""
        removed = [
            material_id
            for material_id in self._states
            if material_id not in active_material_ids
        ]
        if not removed:
            return []
        for material_id in removed:
            del self._states[material_id]
        self._save()
        return removed

    def accept_source_version(
        self,
        material_id: str,
        source_fingerprint: str,
        meta: Optional[dict] = None,
    ):
        """记录本轮已接受的源指纹和管线版本，防止重复 stale。"""
        record = self._states.setdefault(material_id, self._default_state(material_id))
        record["source_fingerprint"] = source_fingerprint
        record["pipeline_version"] = PIPELINE_VERSION
        if meta:
            record["file_name"] = meta.get("file_name", record.get("file_name", ""))
            record["rel_path"] = meta.get("rel_path", record.get("rel_path", ""))
        self._save()

    def check_stale(self, material_id: str, current_fingerprint: str) -> list[str]:
        """检查哪些步骤因源文件变化或版本变化而需要标记为 stale。

        返回需要标记为 stale 的步骤列表。
        """
        record = self.get(material_id)
        stale_steps = []

        # 源文件变化 → 所有步骤 stale
        if record.get("source_fingerprint") != current_fingerprint:
            stale_steps = ALL_STEPS[:]
        # pipeline 版本变化 → 下游步骤 stale
        elif record.get("pipeline_version") != PIPELINE_VERSION:
            if (
                record.get("pipeline_version") == "0.3.0"
                and PIPELINE_VERSION == "0.4.0"
            ):
                stale_steps = ["summary", "excel"]
            elif (
                record.get("pipeline_version") in ("0.3.0", "0.4.0")
                and PIPELINE_VERSION in (
                    "0.5.0", "0.6.0", "0.6.1", "0.6.2",
                    "0.7.0", "0.8.1", "0.9.0",
                )
            ):
                stale_steps = (
                    ["summary", "people", "index", "excel"]
                    if record.get("pipeline_version") == "0.3.0"
                    else ["people", "index", "excel"]
                )
            elif (
                record.get("pipeline_version") == "0.5.0"
                and PIPELINE_VERSION in (
                    "0.6.0", "0.6.1", "0.6.2", "0.7.0", "0.8.1",
                    "0.9.0",
                )
            ):
                stale_steps = ["people", "index", "excel"]
            elif (
                record.get("pipeline_version") == "0.6.0"
                and PIPELINE_VERSION in (
                    "0.6.1", "0.6.2", "0.7.0", "0.8.1", "0.9.0",
                )
            ):
                stale_steps = ["people", "index", "excel"]
            elif (
                record.get("pipeline_version") == "0.6.1"
                and PIPELINE_VERSION == "0.6.2"
            ):
                stale_steps = ["excel"]
            elif (
                record.get("pipeline_version") == "0.6.1"
                and PIPELINE_VERSION in ("0.7.0", "0.8.1", "0.9.0")
            ):
                stale_steps = ["people", "index", "excel"]
            elif (
                record.get("pipeline_version") == "0.6.2"
                and PIPELINE_VERSION in ("0.7.0", "0.8.1", "0.9.0")
            ):
                stale_steps = ["people", "index", "excel"]
            elif (
                record.get("pipeline_version") == "0.7.0"
                and PIPELINE_VERSION in ("0.8.1", "0.9.0")
            ):
                stale_steps = ["people", "index", "excel"]
            elif (
                record.get("pipeline_version") == "0.8.1"
                and PIPELINE_VERSION == "0.9.0"
            ):
                stale_steps = ["speaker", "index", "excel"]
            else:
                stale_steps = ALL_STEPS[:]

        return stale_steps

    def is_step_done(self, material_id: str, step: str) -> bool:
        """判断某步骤是否已完成（success 或 skipped）。"""
        status = self.get_step(material_id, step)
        return status in ("success", "skipped")

    def is_all_done(self, material_id: str) -> bool:
        """判断所有步骤是否都已完成。"""
        return all(self.is_step_done(material_id, s) for s in ALL_STEPS)

    def set_summary(self, material_id: str, summary_text: str, keywords: list[str]):
        """持久化摘要结果到状态记录。"""
        record = self._states.setdefault(material_id, self._default_state(material_id))
        record["base_summary_text"] = summary_text
        record["summary_text"] = summary_text
        record["summary_keywords"] = keywords
        record["identity_events"] = []
        record["identity_summary_text"] = ""
        record["identity_summary_keywords"] = []
        self._save()

    def get_base_summary(self, material_id: str) -> dict:
        """读取不含身份改写的模型基础摘要。"""
        record = self.get(material_id)
        return {
            "摘要": record.get("base_summary_text", record.get("summary_text", "")),
            "关键词": list(record.get("summary_keywords", [])),
        }

    def set_identity_summary(
        self,
        material_id: str,
        events: list[dict],
        summary_text: str,
        keywords: list[str],
    ):
        """保存身份事件和自然语言身份摘要，不改动人工覆盖值。"""
        record = self._states.setdefault(material_id, self._default_state(material_id))
        record["identity_events"] = events
        record["identity_summary_text"] = summary_text
        record["identity_summary_keywords"] = list(keywords)
        record["summary_text"] = summary_text or record.get("base_summary_text", "")
        self._save()

    def clear_identity_summary(self, material_id: str):
        """清除身份改写并恢复基础摘要。"""
        record = self._states.setdefault(material_id, self._default_state(material_id))
        record["identity_events"] = []
        record["identity_summary_text"] = ""
        record["identity_summary_keywords"] = []
        record["summary_text"] = record.get("base_summary_text", "")
        self._save()

    def get_identity_events(self, material_id: str) -> list[dict]:
        return list(self.get(material_id).get("identity_events", []))

    def get_summary(self, material_id: str) -> dict:
        """读取用于输出的摘要，优先返回人工修正值。"""
        record = self.get(material_id)
        return {
            "摘要": (
                record.get("manual_summary_text")
                if record.get("manual_summary_text") is not None
                else record.get("summary_text", "")
            ),
            "关键词": (
                record.get("manual_summary_keywords")
                if record.get("manual_summary_keywords") is not None
                else (
                    record.get("identity_summary_keywords", [])
                    or record.get("summary_keywords", [])
                )
            ),
        }

    def sync_manual_summary(
        self,
        material_id: str,
        summary_text: str,
        keywords: list[str],
    ) -> bool:
        """把 Excel 中发生变化的摘要字段同步为人工覆盖值。"""
        if material_id not in self._states:
            return False
        record = self._states[material_id]
        changed = False

        current_text = (
            record.get("manual_summary_text")
            if record.get("manual_summary_text") is not None
            else record.get("summary_text", "")
        )
        if summary_text != current_text:
            generated_text = record.get("summary_text", "")
            record["manual_summary_text"] = (
                None if summary_text == generated_text else summary_text
            )
            changed = True

        current_keywords = (
            record.get("manual_summary_keywords")
            if record.get("manual_summary_keywords") is not None
            else (
                record.get("identity_summary_keywords", [])
                or record.get("summary_keywords", [])
            )
        )
        if keywords != current_keywords:
            generated_keywords = (
                record.get("identity_summary_keywords", [])
                or record.get("summary_keywords", [])
            )
            record["manual_summary_keywords"] = (
                None if keywords == generated_keywords else keywords
            )
            changed = True

        if changed:
            self._save()
        return changed

    def clear_manual_summary(self, material_id: str):
        """清除摘要和关键词的人工覆盖值。"""
        if material_id not in self._states:
            return
        record = self._states[material_id]
        if (
            record.get("manual_summary_text") is None
            and record.get("manual_summary_keywords") is None
        ):
            return
        record["manual_summary_text"] = None
        record["manual_summary_keywords"] = None
        self._save()

    def set_vision(self, material_id: str, vision_data: dict):
        """持久化视觉理解结果。"""
        record = self._states.setdefault(material_id, self._default_state(material_id))
        record["vision_summary"] = vision_data.get("visual_summary", "")
        record["vision_keywords"] = vision_data.get("visual_keywords", [])
        record["vision_descriptions"] = vision_data.get("visual_descs", [])
        record["vision_frame_count"] = vision_data.get("frame_count", 0)
        record["vision_points"] = vision_data.get("frame_points", [])
        record["vision_model"] = vision_data.get("model", "")
        self._save()

    def get_vision(self, material_id: str) -> dict:
        """读取视觉理解结果。"""
        record = self.get(material_id)
        return {
            "visual_summary": record.get("vision_summary", ""),
            "visual_keywords": record.get("vision_keywords", []),
            "visual_descs": record.get("vision_descriptions", []),
            "frame_count": record.get("vision_frame_count", 0),
            "frame_points": record.get("vision_points", []),
            "model": record.get("vision_model", ""),
        }

    def set_people(self, material_id: str, people_ids: list[str]):
        """保存当前素材识别到的人物 ID。"""
        record = self._states.setdefault(material_id, self._default_state(material_id))
        record["people_ids"] = list(dict.fromkeys(people_ids))
        self._save()

    def get_people(self, material_id: str) -> list[str]:
        """读取当前素材识别到的人物 ID。"""
        return list(self.get(material_id).get("people_ids", []))

    def set_speaker_segments(
        self,
        material_id: str,
        segments: list[dict],
        transcript_srt: str = "",
        transcript_txt: str = "",
    ):
        """保存已登记人物的声纹匹配结果和独立逐字稿路径。"""
        record = self._states.setdefault(material_id, self._default_state(material_id))
        record["speaker_segments"] = list(segments)
        record["speaker_transcript_srt"] = transcript_srt
        record["speaker_transcript_txt"] = transcript_txt
        self._save()

    def get_speaker_segments(self, material_id: str) -> list[dict]:
        return list(self.get(material_id).get("speaker_segments", []))

    def clear_speaker_segments(self, material_id: str):
        record = self._states.setdefault(material_id, self._default_state(material_id))
        record["speaker_segments"] = []
        record["speaker_transcript_srt"] = ""
        record["speaker_transcript_txt"] = ""
        self._save()

    def set_auto_people_text(self, material_id: str, people_text: str):
        """记录上一次写入 Excel 的自动人物值，用于识别人工修改。"""
        record = self._states.setdefault(material_id, self._default_state(material_id))
        if record.get("auto_people_text", "") == people_text:
            return
        record["auto_people_text"] = people_text
        self._save()

    def _default_state(self, material_id: str) -> dict:
        return {
            "material_id": material_id,
            "source_fingerprint": "",
            "pipeline_version": PIPELINE_VERSION,
            "attempts": 0,
            "last_error": "",
            "errors": {},
            "base_summary_text": "",
            "summary_text": "",
            "summary_keywords": [],
            "identity_events": [],
            "identity_summary_text": "",
            "identity_summary_keywords": [],
            "manual_summary_text": None,
            "manual_summary_keywords": None,
            "vision_summary": "",
            "vision_keywords": [],
            "vision_descriptions": [],
            "vision_frame_count": 0,
            "vision_points": [],
            "vision_model": "",
            "people_ids": [],
            "speaker_segments": [],
            "speaker_transcript_srt": "",
            "speaker_transcript_txt": "",
            "auto_people_text": "",
            **{s: "pending" for s in ALL_STEPS},
        }
