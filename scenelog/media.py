"""ffprobe 元数据提取 + ffmpeg 音轨抽取"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path

from scenelog.config import AUDIO_CHANNELS, AUDIO_SAMPLE_RATE
from scenelog.dependencies import find_executable

logger = logging.getLogger(__name__)


def _find_bin(name: str) -> str:
    """返回标准系统安装的可执行文件路径。"""
    path = find_executable(name)
    if path:
        return path
    raise RuntimeError(
        f"未找到 {name}。请安装标准系统版本，或设置 "
        f"SCENELOG_{name.upper().replace('-', '_')}_BIN。"
    )


def extract_metadata(file_path: Path) -> dict:
    """使用 ffprobe 提取素材元数据。

    返回包含：时长、分辨率、帧率、编码、拍摄时间、内嵌时间码等。
    """
    cmd = [
        _find_bin("ffprobe"),
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(file_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe 失败: {result.stderr.strip()}")

        data = json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffprobe 超时")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"ffprobe 输出解析失败: {e}")

    fmt = data.get("format", {})
    streams = data.get("streams", [])

    # 找视频流和音频流
    video_stream = None
    audio_stream = None
    for s in streams:
        if s.get("codec_type") == "video" and video_stream is None:
            video_stream = s
        elif s.get("codec_type") == "audio" and audio_stream is None:
            audio_stream = s

    duration = float(fmt.get("duration", 0))
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    frames = int(duration * 24)  # 估算

    metadata = {
        "duration": duration,
        "duration_str": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
        "duration_frames": frames,
        "file_size": int(fmt.get("size", 0)),
        "format_name": fmt.get("format_name", ""),
        "bit_rate": int(fmt.get("bit_rate", 0)) if fmt.get("bit_rate") else 0,
        "creation_time": _extract_creation_time(fmt),
        "has_video": video_stream is not None,
        "has_audio": audio_stream is not None,
        "audio_codec": audio_stream.get("codec_name", "") if audio_stream else "",
        "audio_channels": audio_stream.get("channels", 0) if audio_stream else 0,
        "audio_sample_rate": int(audio_stream.get("sample_rate", 0)) if audio_stream else 0,
        "video_codec": video_stream.get("codec_name", "") if video_stream else "",
        "width": video_stream.get("width", 0) if video_stream else 0,
        "height": video_stream.get("height", 0) if video_stream else 0,
        "frame_rate": _parse_frame_rate(video_stream) if video_stream else "",
        "timecode": _extract_timecode(fmt, video_stream),
    }

    logger.debug("元数据提取完成: %s (%.1fs)", file_path.name, duration)
    return metadata


def extract_audio(file_path: Path, output_path: Path) -> Path:
    """使用 ffmpeg 抽取音轨并降采样到 16kHz 单声道 WAV。

    返回输出文件路径。若素材无音轨，抛出 RuntimeError。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        _find_bin("ffmpeg"),
        "-y",
        "-v", "error",
        "-i", str(file_path),
        "-map", "0:a:0",          # 取第一条音轨
        "-ac", str(AUDIO_CHANNELS),
        "-ar", str(AUDIO_SAMPLE_RATE),
        "-sample_fmt", "s16",
        str(output_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "Stream map '0:a:0'" in stderr and "does not contain any stream" in stderr:
                raise RuntimeError("素材无音轨")
            raise RuntimeError(f"ffmpeg 音频抽取失败: {stderr}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffmpeg 音频抽取超时")

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("音频抽取输出文件为空")

    logger.debug("音频抽取完成: %s → %s", file_path.name, output_path.name)
    return output_path


def _extract_creation_time(fmt: dict) -> str:
    """从 format tags 中提取拍摄时间。"""
    tags = fmt.get("tags", {})
    for key in ("creation_time", "com.apple.quicktime.creationdate",
                "date", "DateTimeOriginal"):
        if key in tags:
            return tags[key]
    return ""


def _parse_frame_rate(video_stream: dict) -> str:
    """解析帧率为可读字符串。"""
    r_frame_rate = video_stream.get("r_frame_rate", "")
    avg_frame_rate = video_stream.get("avg_frame_rate", "")

    def _eval_fraction(s: str) -> float:
        if not s or "/" not in s:
            return 0.0
        parts = s.split("/")
        try:
            num, den = int(parts[0]), int(parts[1])
            return num / den if den != 0 else 0.0
        except (ValueError, ZeroDivisionError):
            return 0.0

    fps = _eval_fraction(avg_frame_rate) or _eval_fraction(r_frame_rate)
    if fps > 0:
        return f"{fps:.2f} fps"
    return r_frame_rate or ""


def _extract_timecode(fmt: dict, video_stream: dict | None) -> str:
    """提取内嵌时间码。"""
    tags = fmt.get("tags", {}) if fmt else {}
    for key in ("timecode", "TIMECODE", "com.apple.quicktime.timecode"):
        if key in tags:
            return tags[key]
    if video_stream:
        stream_tags = video_stream.get("tags", {})
        for key in ("timecode", "TIMECODE"):
            if key in stream_tags:
                return stream_tags[key]
    return ""
