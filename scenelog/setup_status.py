"""Installation diagnostics for the desktop onboarding flow."""

from __future__ import annotations

import json
import os
import platform
import shutil
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from scenelog import __version__
from scenelog.config import (
    PEOPLE_DETECTOR_MODEL,
    PEOPLE_RECOGNIZER_MODEL,
    SPEAKER_MODEL_DIR,
    WHISPER_CPP_BINARY,
    WHISPER_CPP_MODEL,
)
from scenelog.dependencies import find_executable


def _item(
    key: str,
    label: str,
    ready: bool,
    detail: str,
    required: bool = True,
) -> dict:
    return {
        "key": key,
        "label": label,
        "ready": ready,
        "detail": detail,
        "required": required,
    }


def _ollama_status() -> tuple[bool, set[str], str]:
    executable = find_executable("ollama")
    if not executable:
        return False, set(), "未安装 Ollama"
    try:
        with urlopen("http://127.0.0.1:11434/api/tags", timeout=2) as response:
            models = {
                str(item.get("name", ""))
                for item in json.loads(response.read()).get("models", [])
            }
    except (URLError, TimeoutError, json.JSONDecodeError, OSError):
        return False, set(), "已安装，服务尚未启动"
    return True, models, f"服务运行中，已安装 {len(models)} 个模型"


def installation_status() -> dict:
    """Return user-facing readiness information without changing the system."""
    ffmpeg = find_executable("ffmpeg")
    ffprobe = find_executable("ffprobe")
    whisper_binary = Path(WHISPER_CPP_BINARY).expanduser()
    whisper_model = Path(WHISPER_CPP_MODEL).expanduser()
    ollama_ready, ollama_models, ollama_detail = _ollama_status()
    text_model_ready = any(
        model == "qwen2.5" or model.startswith("qwen2.5:")
        for model in ollama_models
    )
    vision_models = ("qwen2.5vl:3b", "qwen2.5vl:7b", "llava:7b", "llava")
    vision_model = next(
        (model for model in vision_models if model in ollama_models),
        "",
    )
    face_models = [
        Path(PEOPLE_DETECTOR_MODEL).expanduser(),
        Path(PEOPLE_RECOGNIZER_MODEL).expanduser(),
    ]
    speaker_model = Path(SPEAKER_MODEL_DIR).expanduser()
    free_gb = shutil.disk_usage(Path.home()).free / (1024 ** 3)

    checks = [
        _item(
            "ffmpeg",
            "音视频工具",
            bool(ffmpeg and ffprobe),
            "FFmpeg 与 FFprobe 已就绪" if ffmpeg and ffprobe else "需要安装 FFmpeg",
        ),
        _item(
            "whisper_binary",
            "Whisper 转录引擎",
            whisper_binary.is_file() and os.access(whisper_binary, os.X_OK),
            str(whisper_binary)
            if whisper_binary.is_file()
            else "需要安装 whisper-cli",
        ),
        _item(
            "whisper_model",
            "Whisper 中文模型",
            whisper_model.is_file(),
            str(whisper_model) if whisper_model.is_file() else "需要下载转录模型",
        ),
        _item("ollama", "本地模型服务", ollama_ready, ollama_detail),
        _item(
            "text_model",
            "文字摘要模型",
            text_model_ready,
            "qwen2.5 已就绪" if text_model_ready else "需要下载 qwen2.5",
        ),
        _item(
            "vision_model",
            "画面理解模型",
            bool(vision_model),
            f"{vision_model} 已就绪" if vision_model else "需要下载视觉模型",
            required=False,
        ),
        _item(
            "face_models",
            "人物识别模型",
            all(path.is_file() for path in face_models),
            "YuNet 与 SFace 已就绪"
            if all(path.is_file() for path in face_models)
            else "使用人物识别前需要下载",
            required=False,
        ),
        _item(
            "speaker_model",
            "声纹识别模型",
            speaker_model.is_dir() and any(speaker_model.iterdir()),
            "ECAPA-TDNN 已就绪"
            if speaker_model.is_dir() and any(speaker_model.iterdir())
            else "添加声纹时自动下载",
            required=False,
        ),
    ]
    required = [item for item in checks if item["required"]]
    return {
        "version": __version__,
        "platform": {
            "system": platform.system(),
            "release": platform.mac_ver()[0] or platform.release(),
            "machine": platform.machine(),
        },
        "free_disk_gb": round(free_gb, 1),
        "ready": all(item["ready"] for item in required),
        "completed": sum(1 for item in checks if item["ready"]),
        "total": len(checks),
        "checks": checks,
    }
