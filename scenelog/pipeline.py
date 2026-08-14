"""处理管线编排 — 断点续传、步骤级状态、原子写入"""

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional

from scenelog import __version__
from scenelog.config import (
    CACHE_DIR,
    EXCEL_FILE,
    PEOPLE_ENABLED,
    PEOPLE_FRAMES_DIR,
    PIPELINE_VERSION,
    SCENELOG_DIR,
    SPEAKER_TRANSCRIPTS_DIR,
    TRANSCRIPTS_DIR,
    VISION_ENABLED,
    VISION_FRAMES_DIR,
    VISION_VLM_MODEL,
)
from scenelog.excel_writer import generate_excel, load_excel_edits
from scenelog.indexer import (
    build_identity_index,
    build_index,
    build_people_index,
    build_visual_index,
    remove_material_from_index,
)
from scenelog.identity_summary import (
    describe_identity_events,
    generate_identity_summary,
)
from scenelog.material_id import generate_material_id, material_meta
from scenelog.media import extract_audio, extract_metadata
from scenelog.people import FaceEngine, PeopleStore, extract_people_frames
from scenelog.preflight import PreflightError, run_preflight
from scenelog.scanner import scan_media
from scenelog.speaker import (
    SpeakerEngine,
    match_transcript_speakers,
    write_speaker_transcript,
)
from scenelog.state import ALL_STEPS, StateManager
from scenelog.summarizer import generate_summary
from scenelog.transcribe import transcribe, transcribe_many_resilient
from scenelog.vad import VAD_CLEAR_NON_SPEECH, detect_speech
from scenelog.vision import (
    _ensure_vlm_model,
    fuse_audio_visual,
    process_vision,
    visual_only_summary,
)

logger = logging.getLogger(__name__)


class Pipeline:
    """处理管线：扫描 → 预检 → 逐文件处理 → Excel 汇总"""

    def __init__(
        self,
        source_dir: Path,
        output_dir: Optional[Path] = None,
        force: bool = False,
        retry_failed: bool = False,
        rerun_step: Optional[str] = None,
        transcribe_all: bool = False,
        dry_run: bool = False,
        vision: bool = VISION_ENABLED,
        people: bool = PEOPLE_ENABLED,
        selected_file: Optional[str] = None,
    ):
        self.source_dir = source_dir
        self.scenelog_dir = output_dir or (source_dir / SCENELOG_DIR)
        self.force = force
        self.retry_failed = retry_failed
        self.rerun_step = rerun_step
        self.transcribe_all = transcribe_all
        self.dry_run = dry_run
        self.vision_enabled = vision
        self.people_enabled = people
        self.selected_file = selected_file

        # 子目录
        self.transcripts_dir = self.scenelog_dir / TRANSCRIPTS_DIR
        self.cache_dir = self.scenelog_dir / CACHE_DIR
        self.frames_dir = self.cache_dir / VISION_FRAMES_DIR
        self.people_frames_dir = self.cache_dir / PEOPLE_FRAMES_DIR
        self.speaker_transcripts_dir = self.scenelog_dir / SPEAKER_TRANSCRIPTS_DIR

        # 状态管理
        self.state = StateManager(self.scenelog_dir)

        # VLM 模型（延迟初始化）
        self._vlm_model: Optional[str] = None
        self._face_engine: Optional[FaceEngine] = None
        self._speaker_engine: Optional[SpeakerEngine] = None
        self.people_store = PeopleStore(self.scenelog_dir)

        # 收集所有素材的处理结果，供 Excel 汇总
        self._records: list[dict] = []

    def _get_vlm_model(self) -> Optional[str]:
        """延迟获取 VLM 模型名（启用 vision 时才拉模型）。"""
        if not self.vision_enabled:
            return None
        if self._vlm_model is None:
            self._vlm_model = VISION_VLM_MODEL or _ensure_vlm_model()
        return self._vlm_model

    def _get_face_engine(self) -> FaceEngine:
        if self._face_engine is None:
            self._face_engine = FaceEngine()
        return self._face_engine

    def _get_speaker_engine(self) -> SpeakerEngine:
        if self._speaker_engine is None:
            self._speaker_engine = SpeakerEngine()
        return self._speaker_engine

    def run(self):
        """执行完整处理管线。"""
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("scenelog v%s — 开始处理", __version__)
        logger.info("素材目录: %s", self.source_dir)
        logger.info("输出目录: %s", self.scenelog_dir)
        logger.info("画面理解: %s", "开启" if self.vision_enabled else "关闭")
        logger.info("人物识别: %s", "开启" if self.people_enabled else "关闭")
        logger.info("=" * 60)

        all_media_files = scan_media(self.source_dir, excluded_dirs=[self.scenelog_dir])
        if not all_media_files:
            logger.warning("未发现支持的素材文件")
            return
        media_files = self._select_media_files(all_media_files)

        try:
            run_preflight(
                self.source_dir,
                self.scenelog_dir,
                self.vision_enabled,
                self.people_enabled,
            )
        except PreflightError as e:
            logger.error(str(e))
            raise

        if self.dry_run:
            logger.info("[dry-run] 将处理 %d 个文件:", len(media_files))
            for f in media_files:
                logger.info("  %s", f.relative_to(self.source_dir))
            return

        self.scenelog_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        self.speaker_transcripts_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.vision_enabled:
            self.frames_dir.mkdir(parents=True, exist_ok=True)
        active_material_ids = {
            generate_material_id(self.source_dir, file_path)
            for file_path in all_media_files
        }
        for removed_material_id in self.state.prune(active_material_ids):
            self._remove_material_artifacts(removed_material_id)
            self.people_store.remove_material(removed_material_id)
            remove_material_from_index(self.scenelog_dir, removed_material_id)
        self._sync_excel_edits()
        self._acquire_lock()

        try:
            total = len(media_files)
            pending_transcriptions: list[tuple[Path, Path, str]] = []
            for i, file_path in enumerate(media_files):
                logger.info("[%d/%d] %s", i + 1, total, file_path.name)
                try:
                    record = self._process_one(file_path, pending_transcriptions)
                    if record is not None:
                        self._records.append(record)
                except Exception as e:
                    logger.error("处理失败 %s: %s", file_path.name, e)
                    mid = generate_material_id(self.source_dir, file_path)
                    self.state.set_step(mid, "metadata", "failed_retryable", str(e))
                    self._records.append(
                        self._build_record(
                            mid, material_meta(self.source_dir, file_path)
                        )
                    )

            if pending_transcriptions:
                logger.info(
                    "分批转录 %d 个素材（批内单次加载 Whisper 模型）...",
                    len(pending_transcriptions),
                )
                successes, failures = transcribe_many_resilient(
                    [(audio_path, output_key) for _, audio_path, output_key in pending_transcriptions],
                    self.transcripts_dir,
                )
                by_key = {
                    output_key: file_path
                    for file_path, _, output_key in pending_transcriptions
                }
                for output_key, error in failures.items():
                    file_path = by_key[output_key]
                    mid = generate_material_id(self.source_dir, file_path)
                    self.state.set_step(mid, "transcription", "failed_retryable", error)
                    self._records.append(
                        self._build_record(mid, material_meta(self.source_dir, file_path))
                    )
                for output_key in successes:
                    file_path = by_key[output_key]
                    mid = generate_material_id(self.source_dir, file_path)
                    self.state.set_step(mid, "transcription", "success")
                    try:
                        record = self._process_one(file_path)
                    except Exception as e:
                        logger.error("转录后处理失败 %s: %s", file_path.name, e)
                        mid = generate_material_id(self.source_dir, file_path)
                        self.state.set_step(mid, "summary", "failed_retryable", str(e))
                        record = self._build_record(
                            mid, material_meta(self.source_dir, file_path)
                        )
                    if record is not None:
                        self._records.append(record)

            output_records = []
            for file_path in all_media_files:
                material_id = generate_material_id(self.source_dir, file_path)
                if self.state.has_record(material_id):
                    output_records.append(
                        self._build_record(
                            material_id,
                            material_meta(self.source_dir, file_path),
                        )
                    )

            if output_records:
                logger.info("生成场记表...")
                generate_excel(
                    output_records,
                    self.scenelog_dir,
                    self.transcripts_dir,
                    self.scenelog_dir,
                )
                for record in output_records:
                    material_id = record.get("material_id", "")
                    if material_id:
                        people_text = ", ".join(record.get("people", []))
                        self.state.set_auto_people_text(material_id, people_text)
                        self.state.set_step(material_id, "excel", "success")
                logger.info("场记表已保存: %s", self.scenelog_dir / EXCEL_FILE)

        finally:
            self._release_lock()

        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info("处理完成！耗时 %.1f 分钟", elapsed / 60)
        logger.info("场记表: %s", self.scenelog_dir / EXCEL_FILE)
        logger.info("逐字稿: %s/", self.transcripts_dir)
        logger.info("=" * 60)

    def _process_one(
        self,
        file_path: Path,
        pending_transcriptions: Optional[list[tuple[Path, Path, str]]] = None,
    ) -> Optional[dict]:
        """处理单个素材文件，按步骤级状态决定执行哪些步骤。"""
        mid = generate_material_id(self.source_dir, file_path)
        meta = material_meta(self.source_dir, file_path)
        current_fingerprint = meta["fingerprint"]

        self.state.init_record(mid, meta)
        if self.rerun_step == "summary":
            self.state.clear_manual_summary(mid)

        stale_steps = self.state.check_stale(mid, current_fingerprint)
        if stale_steps:
            logger.info("  检测到变化，标记 %d 个步骤为 stale", len(stale_steps))
            self.state.invalidate_steps(mid, stale_steps)
            self._remove_material_artifacts(mid, stale_steps)
            if "index" in stale_steps:
                remove_material_from_index(self.scenelog_dir, mid)
            self.state.accept_source_version(mid, current_fingerprint, meta)

        steps_to_run = self._determine_steps(mid)

        if not steps_to_run:
            logger.info("  全部步骤已完成，跳过")
            return self._build_record(mid, meta)

        logger.info("  待执行步骤: %s", ", ".join(steps_to_run))

        record_data: dict = {"material_id": mid, "metadata": {}, "summary": {}, "state": {}}
        if not self.vision_enabled and "vision" in steps_to_run:
            self.state.set_step(mid, "vision", "skipped")
        if not self.people_enabled and "people" in steps_to_run:
            self.state.set_step(mid, "people", "skipped")

        # Step 1: 元数据
        if "metadata" in steps_to_run:
            try:
                record_data["metadata"] = extract_metadata(file_path)
                self.state.set_step(mid, "metadata", "success")
            except Exception as e:
                logger.error("  元数据提取失败: %s", e)
                self.state.set_step(mid, "metadata", "failed_terminal", str(e))
                self.state.accept_source_version(mid, current_fingerprint, meta)
                return self._build_record(mid, meta)
        else:
            try:
                record_data["metadata"] = extract_metadata(file_path)
            except Exception:
                record_data["metadata"] = {}

        record_data["metadata"]["file_name"] = meta["file_name"]
        record_data["metadata"]["rel_path"] = meta["rel_path"]

        metadata = record_data["metadata"]
        has_audio = metadata.get("has_audio", True)
        has_video = metadata.get("has_video", False)
        duration = metadata.get("duration", 0)
        can_process_vision = self.vision_enabled and has_video
        if "vision" in steps_to_run and not can_process_vision:
            self.state.set_step(mid, "vision", "skipped")

        # 无音轨 → 走纯画面路线（如果启用 vision）
        if not has_audio:
            return self._process_no_audio(file_path, mid, meta, record_data, steps_to_run)

        # Step 2: 音频抽取
        audio_path = self.cache_dir / f"{mid}.wav"
        output_key = mid
        srt_path = self.transcripts_dir / f"{output_key}.srt"
        txt_path = self.transcripts_dir / f"{output_key}.txt"

        if "audio_extract" in steps_to_run:
            try:
                extract_audio(file_path, audio_path)
                self.state.set_step(mid, "audio_extract", "success")
            except Exception as e:
                logger.error("  音频抽取失败: %s", e)
                self.state.set_step(mid, "audio_extract", "failed_terminal", str(e))
                record_data["summary"] = {"摘要": "[音频抽取失败]", "关键词": []}
                self.state.accept_source_version(mid, current_fingerprint, meta)
                record_data["state"] = self.state.get(mid)
                return record_data

        # Step 3: VAD
        is_silent = False
        vad_result = None
        if "vad" in steps_to_run:
            try:
                vad_result = detect_speech(audio_path, force_transcribe=self.transcribe_all)
                if vad_result.decision == VAD_CLEAR_NON_SPEECH:
                    is_silent = True
                    self.state.set_step(mid, "vad", "skipped")
                    if not self.vision_enabled:
                        for step in [
                            "transcription", "speaker", "vision", "summary", "index"
                        ]:
                            self.state.set_step(mid, step, "skipped")
                        self.state.set_summary(mid, "[无语音]", [])
                        record_data["summary"] = {"摘要": "[无语音]", "关键词": []}
                        self.state.accept_source_version(mid, current_fingerprint, meta)
                        record_data["state"] = self.state.get(mid)
                        return record_data
                else:
                    self.state.set_step(mid, "vad", "success")
            except Exception as e:
                logger.error("  VAD 失败: %s", e)
                self.state.set_step(mid, "vad", "failed_retryable", str(e))
                vad_result = None

        # 判断是否有有效语音（综合 VAD 结果和已有转录）
        has_speech = not is_silent
        if "vad" not in steps_to_run:
            existing_trans = self.state.get_step(mid, "transcription")
            has_speech = existing_trans == "success"

        # Step 4: 转录
        if "transcription" in steps_to_run:
            if is_silent and self.vision_enabled:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write("[未识别出语音内容]")
                with open(srt_path, "w", encoding="utf-8") as f:
                    f.write(f"1\n00:00:00,000 --> 00:00:01,000\n[未识别出语音内容]\n\n")
                self.state.set_step(mid, "transcription", "skipped")
                has_speech = False
            elif pending_transcriptions is not None:
                pending_transcriptions.append((file_path, audio_path, output_key))
                logger.info("  已加入批量转录队列")
                return None
            else:
                try:
                    transcribe(audio_path, self.transcripts_dir, output_key)
                    self.state.set_step(mid, "transcription", "success")
                except Exception as e:
                    logger.error("  转录失败: %s", e)
                    self.state.set_step(mid, "transcription", "failed_retryable", str(e))
                    record_data["summary"] = {"摘要": "[转录失败]", "关键词": []}
                    self.state.accept_source_version(mid, current_fingerprint, meta)
                    record_data["state"] = self.state.get(mid)
                    return record_data

        # Step 5: 已登记人物声纹识别
        self._process_speakers(
            mid,
            steps_to_run,
            audio_path,
            srt_path,
            has_speech,
        )

        # Step 6: 视觉理解
        vision_data = None
        if can_process_vision and "vision" in steps_to_run:
            try:
                vlm_model = self._get_vlm_model()
                srt_for_vision = srt_path if has_speech and srt_path.exists() else None
                vision_data = process_vision(
                    video_path=file_path,
                    srt_path=srt_for_vision,
                    cache_dir=self.frames_dir,
                    duration=duration,
                    has_speech=has_speech,
                    model=vlm_model,
                    cache_key=mid,
                )
                self.state.set_vision(mid, vision_data)
                self.state.set_step(mid, "vision", "success")
                logger.info("  画面理解完成: %d 帧, 模型=%s", vision_data["frame_count"], vision_data["model"])
            except Exception as e:
                logger.error("  画面理解失败: %s", e)
                self.state.set_step(mid, "vision", "failed_retryable", str(e))
                vision_data = None
        elif "vision" not in steps_to_run and can_process_vision:
            vision_data = self.state.get_vision(mid)
            if not vision_data.get("visual_summary") and not vision_data.get("frame_count"):
                vision_data = None

        self._process_people(
            mid,
            meta,
            steps_to_run,
            video_path=file_path,
            duration=duration,
            has_video=has_video,
        )
        if (
            "people" in steps_to_run
            or (
                "speaker" in steps_to_run
                and self.state.get_speaker_segments(mid)
            )
        ) and "summary" not in steps_to_run:
            self._update_identity_summary(mid, srt_path)

        # Step 7: 索引
        if "index" in steps_to_run:
            try:
                if srt_path.exists():
                    build_index(
                        srt_path,
                        self.scenelog_dir,
                        mid,
                        meta["file_name"],
                        meta["rel_path"],
                        self.state.get_speaker_segments(mid),
                    )
                else:
                    build_index(
                        self.transcripts_dir / f"{mid}.missing.srt",
                        self.scenelog_dir,
                        mid,
                        meta["file_name"],
                        meta["rel_path"],
                        self.state.get_speaker_segments(mid),
                    )
                if can_process_vision:
                    visual_index_data = vision_data or self.state.get_vision(mid)
                    build_visual_index(
                        self.scenelog_dir,
                        mid,
                        meta["file_name"],
                        meta["rel_path"],
                        visual_index_data.get("frame_points", []),
                        visual_index_data.get("visual_descs", []),
                    )
                people_data = self.people_store.list_people()
                visual_index_data = vision_data or self.state.get_vision(mid)
                build_people_index(
                    self.scenelog_dir,
                    mid,
                    meta["file_name"],
                    meta["rel_path"],
                    people_data,
                    visual_index_data.get("frame_points", []),
                    visual_index_data.get("visual_descs", []),
                )
                self.state.set_step(mid, "index", "success")
            except Exception as e:
                logger.error("  索引构建失败: %s", e)
                self.state.set_step(mid, "index", "failed_retryable", str(e))

        # Step 8: 摘要
        if "summary" in steps_to_run:
            try:
                if has_speech:
                    audio_summary = generate_summary(txt_path, duration)
                    if can_process_vision and vision_data and vision_data.get("visual_descs"):
                        vlm_model = self._get_vlm_model()
                        final_text = fuse_audio_visual(
                            audio_summary.get("摘要", ""),
                            vision_data["visual_descs"],
                            vlm_model,
                        )
                        summary = {"摘要": final_text, "关键词": audio_summary.get("关键词", [])}
                    else:
                        summary = audio_summary
                elif can_process_vision and vision_data and vision_data.get("visual_descs"):
                    if (
                        vision_data.get("visual_summary")
                        and vision_data.get("visual_keywords")
                    ):
                        summary = {
                            "摘要": vision_data["visual_summary"],
                            "关键词": vision_data["visual_keywords"],
                        }
                    else:
                        vlm_model = self._get_vlm_model()
                        summary = visual_only_summary(
                            vision_data["visual_descs"],
                            vlm_model,
                        )
                else:
                    summary = {"摘要": "[无语音]", "关键词": []}

                self.state.set_summary(mid, summary.get("摘要", ""), summary.get("关键词", []))
                self._update_identity_summary(mid, srt_path)
                record_data["summary"] = self.state.get_summary(mid)
                self.state.set_step(mid, "summary", "success")
            except Exception as e:
                logger.error("  摘要生成失败: %s", e)
                record_data["summary"] = {"摘要": "[摘要生成失败]", "关键词": []}
                self.state.set_summary(mid, "[摘要生成失败]", [])
                self.state.set_step(mid, "summary", "failed_retryable", str(e))

        if "summary" not in steps_to_run and not record_data.get("summary"):
            record_data["summary"] = self.state.get_summary(mid)

        record_data["state"] = self.state.get(mid)
        self.state.accept_source_version(mid, current_fingerprint, meta)
        record_data["state"] = self.state.get(mid)
        return record_data

    def _process_no_audio(
        self,
        file_path: Path,
        mid: str,
        meta: dict,
        record_data: dict,
        steps_to_run: list[str],
    ) -> dict:
        """处理无音轨素材：纯画面路线。"""
        for step in ["audio_extract", "vad", "transcription", "speaker"]:
            self.state.set_step(mid, step, "skipped")

        duration = record_data["metadata"].get("duration", 0)
        vision_data = None
        if self.vision_enabled and "vision" in steps_to_run:
            try:
                vlm_model = self._get_vlm_model()
                vision_data = process_vision(
                    video_path=file_path,
                    srt_path=None,
                    cache_dir=self.frames_dir,
                    duration=duration,
                    has_speech=False,
                    model=vlm_model,
                    cache_key=mid,
                )
                self.state.set_vision(mid, vision_data)
                self.state.set_step(mid, "vision", "success")
                logger.info("  画面理解完成: %d 帧, 模型=%s", vision_data["frame_count"], vision_data["model"])
            except Exception as e:
                logger.error("  画面理解失败: %s", e)
                self.state.set_step(mid, "vision", "failed_retryable", str(e))
                vision_data = None
        elif self.vision_enabled:
            vision_data = self.state.get_vision(mid)

        self._process_people(
            mid,
            meta,
            steps_to_run,
            video_path=file_path,
            duration=duration,
            has_video=record_data["metadata"].get("has_video", True),
        )
        if "people" in steps_to_run and "summary" not in steps_to_run:
            self._update_identity_summary(mid, None)

        if "index" in steps_to_run:
            try:
                visual_index_data = vision_data or self.state.get_vision(mid)
                build_index(
                    self.transcripts_dir / f"{mid}.missing.srt",
                    self.scenelog_dir,
                    mid,
                    meta["file_name"],
                    meta["rel_path"],
                    self.state.get_speaker_segments(mid),
                )
                if self.vision_enabled:
                    build_visual_index(
                        self.scenelog_dir,
                        mid,
                        meta["file_name"],
                        meta["rel_path"],
                        visual_index_data.get("frame_points", []),
                        visual_index_data.get("visual_descs", []),
                    )
                build_people_index(
                    self.scenelog_dir,
                    mid,
                    meta["file_name"],
                    meta["rel_path"],
                    self.people_store.list_people(),
                    visual_index_data.get("frame_points", []),
                    visual_index_data.get("visual_descs", []),
                )
                self.state.set_step(mid, "index", "success")
            except Exception as e:
                logger.error("  索引构建失败: %s", e)
                self.state.set_step(mid, "index", "failed_retryable", str(e))

        if "summary" in steps_to_run:
            try:
                if self.vision_enabled and vision_data and vision_data.get("visual_descs"):
                    if (
                        vision_data.get("visual_summary")
                        and vision_data.get("visual_keywords")
                    ):
                        summary = {
                            "摘要": vision_data["visual_summary"],
                            "关键词": vision_data["visual_keywords"],
                        }
                    else:
                        vlm_model = self._get_vlm_model()
                        summary = visual_only_summary(
                            vision_data["visual_descs"],
                            vlm_model,
                        )
                else:
                    summary = {"摘要": "[无音轨]", "关键词": []}
                self.state.set_summary(mid, summary.get("摘要", ""), summary.get("关键词", []))
                self._update_identity_summary(mid, None)
                record_data["summary"] = self.state.get_summary(mid)
                self.state.set_step(mid, "summary", "success")
            except Exception as e:
                logger.error("  摘要生成失败: %s", e)
                record_data["summary"] = {"摘要": "[无音轨]", "关键词": []}
                self.state.set_summary(mid, "[无音轨]", [])
                self.state.set_step(mid, "summary", "failed_retryable", str(e))
        else:
            record_data["summary"] = self.state.get_summary(mid)

        record_data["state"] = self.state.get(mid)
        self.state.accept_source_version(mid, meta["fingerprint"], meta)
        record_data["state"] = self.state.get(mid)
        return record_data

    def _determine_steps(self, material_id: str) -> list[str]:
        """根据 force/retry_failed/rerun_step 决定需要执行的步骤。"""
        if self.force:
            return list(ALL_STEPS)

        if self.rerun_step:
            if self.rerun_step not in ALL_STEPS:
                raise ValueError(f"无效步骤: {self.rerun_step}，可选: {', '.join(ALL_STEPS)}")
            return self._expand_step_dependencies([self.rerun_step])

        if self.retry_failed:
            steps = []
            for step in ALL_STEPS:
                status = self.state.get_step(material_id, step)
                if status in ("failed_retryable", "stale", "pending"):
                    steps.append(step)
            return self._expand_step_dependencies(steps)

        steps = []
        for step in ALL_STEPS:
            status = self.state.get_step(material_id, step)
            if status in ("pending", "stale"):
                steps.append(step)
            elif status == "failed_retryable":
                steps.append(step)
            elif status == "failed_terminal":
                pass
            elif status in ("success", "skipped"):
                pass
        return self._expand_step_dependencies(steps)

    def _expand_step_dependencies(self, steps: list[str]) -> list[str]:
        """扩展下游依赖，同时保持管线执行顺序。"""
        required = set(steps)
        if "metadata" in required:
            required.update(ALL_STEPS)
        if "audio_extract" in required:
            required.update(
                [
                    "vad", "transcription", "speaker", "vision",
                    "people", "summary", "index", "excel",
                ]
            )
        if "vad" in required:
            required.update(
                [
                    "transcription", "speaker", "vision",
                    "people", "summary", "index", "excel",
                ]
            )
        if "transcription" in required:
            required.update(
                ["speaker", "vision", "people", "summary", "index", "excel"]
            )
        if "speaker" in required:
            required.update(["index", "excel"])
        if "vision" in required:
            required.update(["people", "summary", "index", "excel"])
        if "people" in required:
            required.update(["index", "excel"])
        if "summary" in required or "index" in required:
            required.add("excel")
        return [step for step in ALL_STEPS if step in required]

    def _build_record(self, mid: str, meta: dict) -> dict:
        """从已有状态构建记录（用于跳过的素材）。"""
        people_names = self.people_store.labels(self.state.get_people(mid))
        summary = self.state.get_summary(mid)
        if not summary.get("摘要"):
            st = self.state.get(mid)
            if st.get("vad") == "skipped":
                vision = self.state.get_vision(mid) if self.vision_enabled else {}
                if vision.get("visual_summary"):
                    summary = {
                        "摘要": vision["visual_summary"],
                        "关键词": vision.get("visual_keywords", []),
                    }
                else:
                    summary = {"摘要": "[无语音]", "关键词": []}
            elif st.get("transcription") == "failed_retryable":
                summary = {"摘要": "[转录失败]", "关键词": []}
            elif st.get("audio_extract") == "failed_terminal":
                summary = {"摘要": "[音频抽取失败]", "关键词": []}
            elif st.get("audio_extract") == "skipped":
                vision = self.state.get_vision(mid) if self.vision_enabled else {}
                if vision.get("visual_summary"):
                    summary = {
                        "摘要": vision["visual_summary"],
                        "关键词": vision.get("visual_keywords", []),
                    }
                else:
                    summary = {"摘要": "[无音轨]", "关键词": []}

        if (
            not self.state.get(mid).get("summary_text")
            and summary.get("摘要")
        ):
            self.state.set_summary(
                mid,
                summary.get("摘要", ""),
                summary.get("关键词", []),
            )
        summary = self.state.get_summary(mid)

        try:
            file_path = self.source_dir / meta["rel_path"]
            md = extract_metadata(file_path)
        except Exception:
            md = {}
        md["file_name"] = meta["file_name"]
        md["rel_path"] = meta["rel_path"]

        return {
            "material_id": mid,
            "file_name": meta["file_name"],
            "rel_path": meta["rel_path"],
            "metadata": md,
            "summary": summary,
            "people": people_names,
            "state": self.state.get(mid),
        }

    def _update_identity_summary(
        self,
        material_id: str,
        srt_path: Path | None,
    ):
        """将人物事件、画面和同期对白融合为身份感知摘要。"""
        events = self.people_store.material_events(material_id)
        if not events:
            self.state.clear_identity_summary(material_id)
            self._write_identity_index(material_id, [])
            return
        base_summary = self.state.get_base_summary(material_id)
        try:
            described_events = describe_identity_events(
                events,
                self.scenelog_dir,
                srt_path,
                self._get_vlm_model(),
                speaker_segments=self.state.get_speaker_segments(material_id),
            )
            result = generate_identity_summary(
                described_events,
                base_summary.get("摘要", ""),
                base_summary.get("关键词", []),
            )
            self.state.set_identity_summary(
                material_id,
                described_events,
                result.get("摘要", ""),
                result.get("关键词", []),
            )
            self._write_identity_index(material_id, described_events)
            logger.info(
                "  身份感知摘要完成: %d 个事件",
                len(described_events),
            )
        except Exception as exc:
            logger.warning("  身份感知摘要失败，保留基础摘要: %s", exc)
            self.state.clear_identity_summary(material_id)
            self._write_identity_index(material_id, [])

    def _write_identity_index(
        self,
        material_id: str,
        events: list[dict],
    ):
        record = self.state.get(material_id)
        build_identity_index(
            self.scenelog_dir,
            material_id,
            record.get("file_name", ""),
            record.get("rel_path", ""),
            events,
        )

    def _process_speakers(
        self,
        material_id: str,
        steps_to_run: list[str],
        audio_path: Path,
        srt_path: Path,
        has_speech: bool,
    ):
        """Match Whisper segments to registered voices without changing raw SRT."""
        if "speaker" not in steps_to_run:
            return
        profiles = self.people_store.list_people()
        if (
            not has_speech
            or not audio_path.is_file()
            or not srt_path.is_file()
            or not any(profile.get("voice_count") for profile in profiles)
        ):
            self.state.clear_speaker_segments(material_id)
            self.state.set_step(material_id, "speaker", "skipped")
            return
        try:
            segments = match_transcript_speakers(
                audio_path,
                srt_path,
                profiles,
                self._get_speaker_engine(),
            )
            named_srt, named_txt = write_speaker_transcript(
                segments,
                self.speaker_transcripts_dir,
                material_id,
            )
            self.state.set_speaker_segments(
                material_id,
                segments,
                str(named_srt.relative_to(self.scenelog_dir)),
                str(named_txt.relative_to(self.scenelog_dir)),
            )
            self.state.set_step(material_id, "speaker", "success")
            confirmed = sum(
                1 for segment in segments if segment.get("confirmed")
            )
            logger.info(
                "  声纹识别完成: %d/%d 个对白片段确认说话人",
                confirmed,
                len(segments),
            )
        except Exception as exc:
            logger.warning("  声纹识别失败，保留原始逐字稿: %s", exc)
            self.state.clear_speaker_segments(material_id)
            self.state.set_step(
                material_id,
                "speaker",
                "failed_retryable",
                str(exc),
            )

    def _process_people(
        self,
        material_id: str,
        meta: dict,
        steps_to_run: list[str],
        video_path: Path | None = None,
        duration: float = 0,
        has_video: bool = True,
    ):
        """密集扫描视频，只匹配用户预登记的关键人物。"""
        if "people" not in steps_to_run:
            return
        if not self.people_enabled:
            self.state.set_step(material_id, "people", "skipped")
            return
        if not self.people_store.has_registered_people():
            self.people_store.remove_material(material_id)
            self.state.set_people(material_id, [])
            self.state.set_step(material_id, "people", "skipped")
            logger.info("  未登记关键人物，跳过人物识别")
            return
        if not has_video or video_path is None or duration <= 0:
            self.people_store.remove_material(material_id)
            self.state.set_people(material_id, [])
            self.state.set_step(material_id, "people", "skipped")
            logger.info("  素材无有效视频画面，跳过人物识别")
            return
        try:
            frames = extract_people_frames(
                video_path,
                self.people_frames_dir,
                material_id,
                duration,
            )
            logger.info("  人物识别独立扫描: %d 帧", len(frames))
            people_ids = self.people_store.process_material(
                material_id,
                meta["file_name"],
                meta["rel_path"],
                frames,
                self._get_face_engine(),
            )
            self.state.set_people(material_id, people_ids)
            self.state.set_step(material_id, "people", "success")
            logger.info("  人物识别完成: %d 人", len(people_ids))
        except Exception as exc:
            logger.error("  人物识别失败: %s", exc)
            self.state.set_step(
                material_id,
                "people",
                "failed_retryable",
                str(exc),
            )
        finally:
            shutil.rmtree(
                self.people_frames_dir / material_id,
                ignore_errors=True,
            )

    def _acquire_lock(self):
        lock_file = self.scenelog_dir / ".running"
        try:
            fd = os.open(lock_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError as e:
            raise RuntimeError("检测到另一个 scenelog 任务正在运行中") from e
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))

    def _release_lock(self):
        lock_file = self.scenelog_dir / ".running"
        if lock_file.exists():
            lock_file.unlink()

    def _remove_material_artifacts(
        self,
        material_id: str,
        stale_steps: Optional[list[str]] = None,
    ):
        """清理会因源变化而失效的单素材派生产物。"""
        steps = set(stale_steps or ALL_STEPS)
        paths = []
        if "audio_extract" in steps:
            paths.append(self.cache_dir / f"{material_id}.wav")
        if "transcription" in steps:
            paths.extend(
                [
                    self.transcripts_dir / f"{material_id}.srt",
                    self.transcripts_dir / f"{material_id}.txt",
                ]
            )
        if "speaker" in steps:
            paths.extend(
                [
                    self.speaker_transcripts_dir / f"{material_id}.srt",
                    self.speaker_transcripts_dir / f"{material_id}.txt",
                ]
            )
        for path in paths:
            if path.exists():
                path.unlink()
        if "vision" in steps and self.frames_dir.exists():
            for frame_path in self.frames_dir.glob(f"{material_id}_f*.jpg"):
                frame_path.unlink()

    def _select_media_files(self, media_files: list[Path]) -> list[Path]:
        """按相对路径或唯一文件名筛选单个素材。"""
        if not self.selected_file:
            return media_files

        selection = self.selected_file.replace("\\", "/").lstrip("./")
        selected_path = Path(self.selected_file).expanduser()
        if selected_path.is_absolute():
            try:
                selection = selected_path.resolve().relative_to(
                    self.source_dir
                ).as_posix()
            except ValueError as e:
                raise ValueError("--file 必须位于素材目录内") from e

        exact_matches = [
            file_path
            for file_path in media_files
            if file_path.relative_to(self.source_dir).as_posix() == selection
        ]
        if exact_matches:
            return exact_matches

        name_matches = [
            file_path
            for file_path in media_files
            if file_path.name == Path(selection).name
        ]
        if len(name_matches) == 1:
            return name_matches
        if len(name_matches) > 1:
            choices = ", ".join(
                file_path.relative_to(self.source_dir).as_posix()
                for file_path in name_matches
            )
            raise ValueError(f"文件名不唯一，请使用相对路径: {choices}")
        raise ValueError(f"未找到素材: {self.selected_file}")

    def _sync_excel_edits(self):
        """将用户在 Excel 中修改的摘要和关键词同步到状态文件。"""
        excel_path = self.scenelog_dir / EXCEL_FILE
        synced = 0
        for rel_path, edits in load_excel_edits(excel_path).items():
            material_id = self.state.find_by_rel_path(rel_path)
            if material_id and self.state.sync_manual_summary(
                material_id,
                edits.get("内容摘要", ""),
                edits.get("关键词", []),
            ):
                synced += 1
        if synced:
            logger.info("已保留 %d 条 Excel 人工摘要/关键词修正", synced)
