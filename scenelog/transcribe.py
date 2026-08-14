"""whisper.cpp（Apple Silicon Metal）转录 — 生成带时间戳逐字稿。"""

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from scenelog.config import (
    WHISPER_BATCH_SIZE,
    WHISPER_CPP_BINARY,
    WHISPER_CPP_MODEL,
    WHISPER_CPP_THREADS,
    WHISPER_LANGUAGE,
    WHISPER_TIMEOUT_PER_FILE,
)

logger = logging.getLogger(__name__)


def transcribe(
    audio_path: Path,
    output_dir: Path,
    file_stem: str,
) -> tuple[Path, Path]:
    """使用 Apple Silicon Metal 版 whisper.cpp 转录单个音频。"""
    return transcribe_many([(audio_path, file_stem)], output_dir)[0]


def transcribe_many(
    items: list[tuple[Path, str]],
    output_dir: Path,
) -> list[tuple[Path, Path]]:
    """使用一次 whisper.cpp 进程转录多个音频。"""
    if not items:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    cli_path = Path(WHISPER_CPP_BINARY).expanduser()
    model_path = Path(WHISPER_CPP_MODEL).expanduser()
    if not cli_path.is_file() or not os.access(cli_path, os.X_OK):
        raise RuntimeError(
            f"未找到可执行的 whisper.cpp CLI: {cli_path}。"
            "请设置 SCENELOG_WHISPER_CPP_BIN。"
        )
    if not model_path.is_file():
        raise RuntimeError(
            f"未找到 whisper.cpp 模型: {model_path}。"
            "请设置 SCENELOG_WHISPER_CPP_MODEL。"
        )

    file_stems = [file_stem for _, file_stem in items]
    if len(file_stems) != len(set(file_stems)):
        raise ValueError("批量转录的输出文件名不能重复")

    logger.info(
        "开始批量 Metal 转录: %d 个音频 (whisper.cpp, 模型=%s)",
        len(items),
        model_path.name,
    )

    tmp_dir = Path(tempfile.mkdtemp(dir=output_dir, prefix=".transcribe_"))
    processed: list[tuple[Path, Path, list[dict]]] = []
    try:
        cmd = [
            str(cli_path),
            "-m", str(model_path),
            "-l", WHISPER_LANGUAGE,
            "-t", str(WHISPER_CPP_THREADS),
            "-osrt",
            "-otxt",
            "-np",
        ]
        for index, (audio_path, _) in enumerate(items):
            tmp_prefix = tmp_dir / str(index)
            cmd.extend(["-f", str(audio_path), "-of", str(tmp_prefix)])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(WHISPER_TIMEOUT_PER_FILE, WHISPER_TIMEOUT_PER_FILE * len(items)),
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"whisper.cpp 批量转录失败: {detail}")

        for index, (_, file_stem) in enumerate(items):
            tmp_prefix = tmp_dir / str(index)
            tmp_srt = tmp_prefix.with_suffix(".srt")
            tmp_txt = tmp_prefix.with_suffix(".txt")
            if not tmp_srt.is_file() or not tmp_txt.is_file():
                raise RuntimeError(
                    f"whisper.cpp 未生成 SRT/TXT 输出: {file_stem}"
                )

            segments = parse_srt(tmp_srt)
            full_text = tmp_txt.read_text(encoding="utf-8").strip()
            if not full_text or not segments:
                _write_no_speech(tmp_srt, tmp_txt)
                segments = []
            processed.append((tmp_srt, tmp_txt, segments))

        # 仅在所有素材的 SRT/TXT 都成功生成后替换现有产物。
        for (tmp_srt, tmp_txt, segments), (_, file_stem) in zip(processed, items):
            srt_path = output_dir / f"{file_stem}.srt"
            txt_path = output_dir / f"{file_stem}.txt"
            os.replace(tmp_srt, srt_path)
            os.replace(tmp_txt, txt_path)
            if segments:
                logger.info(
                    "转录完成: %s (%d 个片段, whisper.cpp Metal)",
                    srt_path.name,
                    len(segments),
                )
            else:
                logger.info("转录完成: %s → 未识别出语音内容", srt_path.name)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return [
        (output_dir / f"{file_stem}.srt", output_dir / f"{file_stem}.txt")
        for _, file_stem in items
    ]


def transcribe_many_resilient(
    items: list[tuple[Path, str]],
    output_dir: Path,
    batch_size: int = WHISPER_BATCH_SIZE,
) -> tuple[list[str], dict[str, str]]:
    """分小批转录；批次失败时降级逐文件重试并隔离失败素材。"""
    successes: list[str] = []
    failures: dict[str, str] = {}
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")

    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        try:
            transcribe_many(batch, output_dir)
        except Exception as batch_error:
            logger.warning(
                "Whisper 批次失败，降级逐文件重试 (%d 个): %s",
                len(batch),
                batch_error,
            )
            for item in batch:
                _, output_key = item
                try:
                    transcribe_many([item], output_dir)
                    successes.append(output_key)
                except Exception as item_error:
                    failures[output_key] = str(item_error)
        else:
            successes.extend(output_key for _, output_key in batch)

    return successes, failures


def _write_no_speech(srt_path: Path, txt_path: Path) -> None:
    """写入没有有效语音时的占位逐字稿。"""
    placeholder = "[未识别出语音内容]"
    txt_path.write_text(placeholder, encoding="utf-8")
    srt_path.write_text(
        f"1\n00:00:00,000 --> 00:00:01,000\n{placeholder}\n\n",
        encoding="utf-8",
    )


def parse_srt(srt_path: Path) -> list[dict]:
    """解析 SRT 文件为片段列表。

    每个片段包含: index, start_time, end_time, text
    """
    if not srt_path.exists():
        return []

    segments = []
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    blocks = content.split("\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        try:
            index = int(lines[0])
        except ValueError:
            continue

        time_line = lines[1]
        times = time_line.split(" --> ")
        if len(times) != 2:
            continue

        start_time = _parse_srt_time(times[0].strip())
        end_time = _parse_srt_time(times[1].strip())
        text = " ".join(lines[2:]).strip()

        if text:
            segments.append({
                "index": index,
                "start_time": start_time,
                "end_time": end_time,
                "text": text,
            })

    return segments


def _parse_srt_time(time_str: str) -> str:
    """将 SRT 时间戳转为 HH:MM:SS 格式。

    "00:01:23,456" → "00:01:23"
    """
    if "," in time_str:
        time_str = time_str.split(",")[0]
    return time_str
