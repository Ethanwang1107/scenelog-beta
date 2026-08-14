"""Registered-person speaker embeddings and conservative transcript attribution."""

from __future__ import annotations

import logging
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from scenelog.config import (
    AUDIO_SAMPLE_RATE,
    SPEAKER_MATCH_MARGIN,
    SPEAKER_MATCH_THRESHOLD,
    SPEAKER_MIN_REFERENCE_SECONDS,
    SPEAKER_MIN_SEGMENT_SECONDS,
    SPEAKER_MODEL_DIR,
    SPEAKER_MODEL_SOURCE,
    SPEAKER_SINGLE_MATCH_THRESHOLD,
)
from scenelog.dependencies import find_executable
from scenelog.vad import detect_speech

logger = logging.getLogger(__name__)

SUPPORTED_VOICE_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".flac",
    ".mov",
    ".mp4",
}


def install_speaker_model() -> Path:
    """Download and initialize the configured ECAPA speaker model."""
    model_dir = Path(SPEAKER_MODEL_DIR).expanduser()
    from speechbrain.inference.speaker import EncoderClassifier

    EncoderClassifier.from_hparams(
        source=SPEAKER_MODEL_SOURCE,
        savedir=str(model_dir),
        run_opts={"device": "cpu"},
    )
    return model_dir


class SpeakerEngine:
    """Extract normalized ECAPA-TDNN speaker embeddings."""

    def __init__(self):
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as exc:
            raise RuntimeError(
                "缺少 SpeechBrain 声纹依赖，请重新安装 scenelog"
            ) from exc

        model_dir = Path(SPEAKER_MODEL_DIR).expanduser()
        source = str(model_dir) if (model_dir / "hyperparams.yaml").is_file() else (
            SPEAKER_MODEL_SOURCE
        )
        try:
            self.classifier = EncoderClassifier.from_hparams(
                source=source,
                savedir=str(model_dir),
                run_opts={"device": "cpu"},
            )
        except Exception as exc:
            raise RuntimeError(
                "声纹模型不可用，请联网执行 scenelog people voice-setup"
            ) from exc

    def extract(
        self,
        audio_path: Path,
        start_seconds: float | None = None,
        end_seconds: float | None = None,
        minimum_seconds: float = SPEAKER_MIN_SEGMENT_SECONDS,
    ) -> list[float]:
        """Extract one normalized embedding from a 16 kHz mono audio interval."""
        waveform, sample_rate = load_audio(
            audio_path,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
        )
        duration = len(waveform) / sample_rate
        if duration < minimum_seconds:
            raise ValueError(
                f"有效声音片段过短（{duration:.1f} 秒），至少需要 {minimum_seconds:.1f} 秒"
            )
        if _rms(waveform) < 0.003:
            raise ValueError("声音片段音量过低，无法建立可靠声纹")

        import torch

        tensor = torch.from_numpy(waveform).float().unsqueeze(0)
        with torch.inference_mode():
            embedding = self.classifier.encode_batch(tensor).flatten()
        values = embedding.detach().cpu().numpy().astype(np.float32)
        normalized = _normalize(values)
        if not normalized:
            raise RuntimeError("声纹模型未生成有效特征")
        return normalized


def prepare_reference_audio(source_path: Path, target_path: Path) -> float:
    """Convert uploaded audio/video to 16 kHz mono WAV and keep speech only."""
    source = source_path.expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"声音样本不存在: {source_path}")
    if source.suffix.lower() not in SUPPORTED_VOICE_EXTENSIONS:
        raise ValueError(f"不支持的声音格式: {source.name}")

    ffmpeg = find_executable("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("未找到 ffmpeg，无法处理声音样本")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="scenelog_voice_") as temp_dir:
        converted = Path(temp_dir) / "converted.wav"
        result = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-v",
                "error",
                "-i",
                str(source),
                "-map",
                "0:a:0",
                "-ac",
                "1",
                "-ar",
                str(AUDIO_SAMPLE_RATE),
                "-sample_fmt",
                "s16",
                str(converted),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0 or not converted.is_file():
            raise ValueError(
                f"无法读取声音样本 {source.name}: "
                f"{result.stderr.strip() or '没有可用音轨'}"
            )

        vad = detect_speech(converted)
        if not vad.speech_intervals:
            raise ValueError(f"声音样本未检测到清晰语音: {source.name}")
        speech_duration = sum(end - start for start, end in vad.speech_intervals)
        if speech_duration < SPEAKER_MIN_REFERENCE_SECONDS:
            raise ValueError(
                f"有效语音只有 {speech_duration:.1f} 秒，"
                f"至少需要 {SPEAKER_MIN_REFERENCE_SECONDS:.1f} 秒"
            )

        _write_joined_intervals(
            converted,
            target_path,
            vad.speech_intervals,
            maximum_seconds=60.0,
        )
    return min(speech_duration, 60.0)


def match_transcript_speakers(
    audio_path: Path,
    srt_path: Path,
    profiles: list[dict],
    engine: SpeakerEngine,
    threshold: float = SPEAKER_MATCH_THRESHOLD,
    margin: float = SPEAKER_MATCH_MARGIN,
    single_threshold: float = SPEAKER_SINGLE_MATCH_THRESHOLD,
) -> list[dict]:
    """Match each Whisper segment to a registered voice using two gates."""
    references = []
    for profile in profiles:
        embeddings = [
            embedding
            for embedding in profile.get("voice_embeddings", [])
            if embedding
        ]
        if not embeddings:
            continue
        references.append(
            {
                "person_id": profile["id"],
                "name": profile["name"],
                "embeddings": embeddings,
            }
        )
    if not references:
        return []

    matched = []
    for segment in parse_srt_precise(srt_path):
        duration = segment["end_seconds"] - segment["start_seconds"]
        current = {
            **segment,
            "person_id": "",
            "speaker_name": "",
            "match_score": 0.0,
            "match_margin": 0.0,
            "confirmed": False,
            "reason": "",
        }
        if duration < SPEAKER_MIN_SEGMENT_SECONDS:
            current["reason"] = "片段过短"
            matched.append(current)
            continue
        try:
            embedding = engine.extract(
                audio_path,
                segment["start_seconds"],
                segment["end_seconds"],
            )
        except (ValueError, RuntimeError) as exc:
            current["reason"] = str(exc)
            matched.append(current)
            continue

        scores = sorted(
            (
                max(_cosine(embedding, reference) for reference in item["embeddings"]),
                item,
            )
            for item in references
        )
        best_score, best = scores[-1]
        has_multiple_candidates = len(scores) > 1
        second_score = scores[-2][0] if has_multiple_candidates else best_score
        score_margin = best_score - second_score if has_multiple_candidates else 0.0
        current["match_score"] = round(float(best_score), 4)
        current["match_margin"] = round(float(score_margin), 4)
        accepted = (
            best_score >= threshold and score_margin >= margin
            if has_multiple_candidates
            else best_score >= single_threshold
        )
        if accepted:
            current.update(
                {
                    "person_id": best["person_id"],
                    "speaker_name": best["name"],
                    "confirmed": True,
                    "reason": "声纹匹配通过",
                }
            )
        elif best_score < (threshold if has_multiple_candidates else single_threshold):
            current["reason"] = "声纹相似度不足"
        else:
            current["reason"] = "候选人物分差不足"
        matched.append(current)
    return matched


def write_speaker_transcript(
    segments: list[dict],
    output_dir: Path,
    material_id: str,
) -> tuple[Path, Path]:
    """Write a separate named transcript without modifying Whisper output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    srt_path = output_dir / f"{material_id}.srt"
    txt_path = output_dir / f"{material_id}.txt"
    srt_blocks = []
    text_lines = []
    for index, segment in enumerate(segments, 1):
        name = segment.get("speaker_name") if segment.get("confirmed") else "未知说话人"
        text = str(segment.get("text", "")).strip()
        srt_blocks.append(
            f"{index}\n"
            f"{_format_srt_time(segment['start_seconds'])} --> "
            f"{_format_srt_time(segment['end_seconds'])}\n"
            f"{name}：{text}"
        )
        text_lines.append(f"{name}：{text}")
    srt_path.write_text("\n\n".join(srt_blocks) + ("\n\n" if srt_blocks else ""), encoding="utf-8")
    txt_path.write_text("\n".join(text_lines), encoding="utf-8")
    return srt_path, txt_path


def parse_srt_precise(srt_path: Path) -> list[dict]:
    """Parse SRT while preserving millisecond timestamps."""
    if not srt_path.is_file():
        return []
    blocks = re.split(r"\n\s*\n", srt_path.read_text(encoding="utf-8").strip())
    result = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3 or " --> " not in lines[1]:
            continue
        start, end = lines[1].split(" --> ", 1)
        try:
            start_seconds = _srt_time_to_seconds(start)
            end_seconds = _srt_time_to_seconds(end)
        except ValueError:
            continue
        text = " ".join(line.strip() for line in lines[2:] if line.strip())
        if not text or text == "[未识别出语音内容]":
            continue
        result.append(
            {
                "index": len(result) + 1,
                "start_time": _display_time(start_seconds),
                "end_time": _display_time(end_seconds),
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "text": text,
            }
        )
    return result


def load_audio(
    path: Path,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> tuple[np.ndarray, int]:
    """Load an interval as mono float32 and resample to 16 kHz."""
    info = sf.info(str(path))
    source_rate = int(info.samplerate)
    start_frame = max(0, int((start_seconds or 0.0) * source_rate))
    stop_frame = (
        min(info.frames, int(end_seconds * source_rate))
        if end_seconds is not None
        else info.frames
    )
    if stop_frame <= start_frame:
        return np.asarray([], dtype=np.float32), AUDIO_SAMPLE_RATE
    data, sample_rate = sf.read(
        str(path),
        start=start_frame,
        stop=stop_frame,
        dtype="float32",
        always_2d=False,
    )
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sample_rate != AUDIO_SAMPLE_RATE and len(data):
        from scipy.signal import resample_poly

        common = math.gcd(sample_rate, AUDIO_SAMPLE_RATE)
        data = resample_poly(
            data,
            AUDIO_SAMPLE_RATE // common,
            sample_rate // common,
        ).astype(np.float32)
        sample_rate = AUDIO_SAMPLE_RATE
    return np.asarray(data, dtype=np.float32), sample_rate


def _write_joined_intervals(
    source_path: Path,
    target_path: Path,
    intervals: list[tuple[float, float]],
    maximum_seconds: float,
):
    waveform, sample_rate = load_audio(source_path)
    pieces = []
    remaining = int(maximum_seconds * sample_rate)
    for start, end in intervals:
        start_index = max(0, int(start * sample_rate))
        end_index = min(len(waveform), int(end * sample_rate))
        piece = waveform[start_index:end_index]
        if len(piece) > remaining:
            piece = piece[:remaining]
        if len(piece):
            pieces.append(piece)
            remaining -= len(piece)
        if remaining <= 0:
            break
    if not pieces:
        raise ValueError("声音样本没有可用语音")
    joined = np.concatenate(pieces)
    sf.write(str(target_path), joined, sample_rate, subtype="PCM_16")


def _cosine(first: list[float], second: list[float]) -> float:
    a = np.asarray(first, dtype=np.float32)
    b = np.asarray(second, dtype=np.float32)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else -1.0


def _normalize(values: np.ndarray) -> list[float]:
    norm = float(np.linalg.norm(values))
    if not norm or not np.isfinite(norm):
        return []
    return (values / norm).astype(float).tolist()


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else 0.0


def _srt_time_to_seconds(value: str) -> float:
    match = re.fullmatch(r"(\d+):(\d+):(\d+)[,.](\d+)", value.strip())
    if not match:
        raise ValueError(value)
    hours, minutes, seconds, milliseconds = (int(item) for item in match.groups())
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000


def _format_srt_time(seconds: float) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def _display_time(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
