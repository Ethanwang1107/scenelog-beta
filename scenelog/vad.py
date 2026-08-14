"""Silero VAD 多级语音检测决策"""

import logging
from pathlib import Path
from typing import Optional

from scenelog.config import (
    VAD_CRITICAL_RATIO,
    VAD_MIN_SILENCE_DURATION,
    VAD_MIN_SPEECH_DURATION,
    VAD_SPEECH_RATIO_THRESHOLD,
    VAD_THRESHOLD,
)

logger = logging.getLogger(__name__)

# VAD 决策枚举
VAD_CLEAR_SPEECH = "clear_speech"           # 明确有语音
VAD_CLEAR_NON_SPEECH = "clear_non_speech"   # 明确无语音
VAD_CRITICAL = "critical"                   # 临界样本


class VADResult:
    """VAD 检测结果"""

    def __init__(
        self,
        decision: str,
        speech_intervals: list[tuple[float, float]],
        total_speech_duration: float,
        speech_ratio: float,
        confidence: float,
        reason: str,
    ):
        self.decision = decision
        self.speech_intervals = speech_intervals
        self.total_speech_duration = total_speech_duration
        self.speech_ratio = speech_ratio
        self.confidence = confidence
        self.reason = reason

    def has_speech(self) -> bool:
        return self.decision in (VAD_CLEAR_SPEECH, VAD_CRITICAL)

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "speech_intervals": [
                {"start": round(s, 2), "end": round(e, 2)}
                for s, e in self.speech_intervals
            ],
            "total_speech_duration": round(self.total_speech_duration, 2),
            "speech_ratio": round(self.speech_ratio, 4),
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
        }


_model: Optional[tuple] = None


def _get_model():
    """从本地 Torch Hub 缓存加载 Silero VAD 模型。"""
    global _model
    if _model is None:
        import torch
        repo_dir = Path(torch.hub.get_dir()) / "snakers4_silero-vad_master"
        if not repo_dir.is_dir():
            raise RuntimeError(
                "本地未缓存 Silero VAD 模型。联网时执行 INSTALL.md 中的预下载命令，"
                "再离线运行 scenelog。"
            )
        logger.info("加载 Silero VAD 模型...")
        model, utils = torch.hub.load(
            repo_or_dir=str(repo_dir),
            model="silero_vad",
            source="local",
            force_reload=False,
            trust_repo=True,
        )
        _model = (model, utils)
    return _model


def _load_audio(audio_path: Path) -> tuple["torch.Tensor", int]:
    """使用 soundfile 加载音频，返回 (tensor, sample_rate)。"""
    import numpy as np
    import soundfile as sf
    import torch

    data, sr = sf.read(str(audio_path), dtype="float32")
    # 转单声道
    if data.ndim > 1:
        data = data.mean(axis=1)
    # 重采样到 16kHz
    if sr != 16000:
        import scipy.signal
        num_samples = int(len(data) * 16000 / sr)
        data = scipy.signal.resample(data, num_samples)
        sr = 16000
    tensor = torch.from_numpy(data).float().unsqueeze(0)
    return tensor, sr


def detect_speech(audio_path: Path, force_transcribe: bool = False) -> VADResult:
    """对音频文件执行 VAD 多级决策。

    Args:
        audio_path: 16kHz 单声道 WAV 音频文件路径
        force_transcribe: 若为 True，强制返回 clear_speech

    Returns:
        VADResult 包含决策、语音区间、统计信息
    """
    if force_transcribe:
        duration = _get_audio_duration(audio_path)
        return VADResult(
            decision=VAD_CLEAR_SPEECH,
            speech_intervals=[(0.0, duration)],
            total_speech_duration=duration,
            speech_ratio=1.0,
            confidence=1.0,
            reason="用户强制转录 (--transcribe-all)",
        )

    model, utils = _get_model()
    get_speech_timestamps = utils[0]

    # 加载音频
    wav, sr = _load_audio(audio_path)
    audio_duration = wav.shape[1] / sr

    # 获取语音时间戳
    speech_timestamps = get_speech_timestamps(
        wav.squeeze(),
        model,
        threshold=VAD_THRESHOLD,
        min_speech_duration_ms=int(VAD_MIN_SPEECH_DURATION * 1000),
        min_silence_duration_ms=int(VAD_MIN_SILENCE_DURATION * 1000),
    )

    # 计算语音统计
    total_speech = sum(
        (ts["end"] - ts["start"]) / 16000 for ts in speech_timestamps
    )
    speech_ratio = total_speech / audio_duration if audio_duration > 0 else 0.0

    intervals = [
        (ts["start"] / 16000, ts["end"] / 16000)
        for ts in speech_timestamps
    ]

    # 计算置信度（基于语音段数量和平均长度）
    if speech_timestamps:
        avg_seg_len = total_speech / len(speech_timestamps)
        confidence = min(1.0, avg_seg_len / 2.0)
    else:
        confidence = 0.0

    # 多级决策
    if speech_ratio < VAD_SPEECH_RATIO_THRESHOLD:
        decision = VAD_CLEAR_NON_SPEECH
        reason = f"语音占比 {speech_ratio:.4f} < {VAD_SPEECH_RATIO_THRESHOLD}（明确无语音阈值）"
    elif speech_ratio < VAD_CRITICAL_RATIO:
        decision = VAD_CRITICAL
        reason = f"语音占比 {speech_ratio:.4f} 在 [{VAD_SPEECH_RATIO_THRESHOLD}, {VAD_CRITICAL_RATIO}) 临界区间"
    else:
        decision = VAD_CLEAR_SPEECH
        reason = f"语音占比 {speech_ratio:.4f} ≥ {VAD_CRITICAL_RATIO}（明确有语音阈值）"

    logger.debug(
        "VAD: %s → %s (ratio=%.4f, speech=%.1fs, segments=%d)",
        audio_path.name, decision, speech_ratio, total_speech, len(speech_timestamps),
    )

    return VADResult(
        decision=decision,
        speech_intervals=intervals,
        total_speech_duration=total_speech,
        speech_ratio=speech_ratio,
        confidence=confidence,
        reason=reason,
    )


def _get_audio_duration(audio_path: Path) -> float:
    """获取音频时长（秒）。"""
    import soundfile as sf
    info = sf.info(str(audio_path))
    return info.duration
