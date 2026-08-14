"""画面理解 — 声音引导抽帧 + PySceneDetect 场景切换抽帧 + VLM 描述"""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import requests

from scenelog.config import (
    VISION_DEDUP_SECONDS,
    VISION_LONG_THRESHOLD,
    VISION_MAX_FRAMES_LONG,
    VISION_MAX_FRAMES_SHORT,
    VISION_VLM_MODEL,
    VISION_VLM_BASE_URL,
)
from scenelog.dependencies import find_executable
from scenelog.media import _find_bin
from scenelog.summarizer import generate_summary_from_text
from scenelog.transcribe import parse_srt

logger = logging.getLogger(__name__)

VLM_PROMPT_SINGLE = (
    "你是纪录片场记，请用一句简洁中文描述这一帧（20字以内）。\n"
    "只关注三件事：场景在哪里、画面里有谁、人物或主体正在做什么。\n"
    "没有人物时，如实描述环境和正在发生的事物或变化；不要臆测人物。\n"
    "严禁输出：这段视频、画面中、可以看到、包括、镜头用途、镜头类型、"
    "空镜、转场、建立镜头、人物行为镜头、访谈镜头、特写。\n"
    "好例子：\n"
    "夜晚城市街道上车流穿行\n"
    "办公室内一名工作人员收拾物品\n"
    "室外田野中三人边走边交谈\n"
    "室内桌面摆放木纹板材\n"
    "夕阳下城市高楼与天空同框\n"
    "只输出这一句描述，不要解释。"
)

VLM_PROMPT_FUSE = """你是纪录片场记。把语音内容和多帧画面描述归纳成一句通顺摘要（50字以内）。

语音：{audio_summary}
画面描述：{visual_descs}

规则：
- 只说明场景、人物和动作，并自然融入语音的核心事件；
- 多帧描述不一致时，优先选择最能代表当前素材的主场景和主事件，不要把冲突内容硬拼成一句；
- 不要出现「画面：」「语音：」「画面场景」「语音核心信息」等标记词；
- 不要说「这段视频」「画面中可以看到」；
- 严禁输出任何镜头用途或类型标签，包括「空镜」「转场」「建立镜头」「人物行为镜头」「访谈镜头」。
好例子：
夜晚街道上车流穿行，旁白谈论手机外观
办公室内工作人员准备下班，并谈及离开时间
田野中三人边走边讨论冲锋衣性能
只输出这一句。"""

VLM_PROMPT_VISUAL_ONLY = """你是纪录片场记。把下面多帧画面描述归纳为一句通顺摘要（35字以内）。

画面描述：{visual_descs}

规则：
- 只说明场景、人物和动作；
- 多帧内容重复时只说一次；描述不一致时，选择最能代表素材的主场景，不要罗列或拼接矛盾内容；
- 不要解释、不要列点、不要说"这段视频展示了"；
- 严禁输出任何镜头用途或类型标签，包括「空镜」「转场」「建立镜头」「人物行为镜头」「访谈镜头」。
- 必须同时给出 3-5 个能用于检索的关键词，优先选择地点、人物、动作和显著物体。

严格按 JSON 格式输出，不要有额外文字：
{{"摘要":"夜晚城市街道上车流穿行","关键词":["夜晚","城市街道","车流"]}}"""


def detect_scene_changes(video_path: Path) -> list[float]:
    """使用 PySceneDetect 检测镜头切换点，返回切换时间点列表（秒）。"""
    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import ContentDetector
    except ImportError:
        logger.warning("PySceneDetect 未安装，跳过场景切换检测")
        return []

    try:
        video = open_video(str(video_path))
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=30.0))
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()

        cut_points = []
        for i, (start, _) in enumerate(scene_list):
            t = start.get_seconds()
            if t > 0.5:
                cut_points.append(t)
        logger.debug("场景检测完成: %s → %d 个切换点", video_path.name, len(cut_points))
        return cut_points
    except Exception as e:
        logger.warning("场景切换检测失败: %s", e)
        return []


def get_speech_midpoints(srt_path: Path) -> list[float]:
    """从 SRT 解析每个语音片段的中间时间点（秒）。"""
    segments = parse_srt(srt_path)
    midpoints = []
    for seg in segments:
        start_s = _srt_time_to_seconds(seg["start_time"])
        end_s = _srt_time_to_seconds(seg["end_time"])
        mid = (start_s + end_s) / 2.0
        midpoints.append(mid)
    return midpoints


def _srt_time_to_seconds(time_str: str) -> float:
    """HH:MM:SS → 秒"""
    parts = time_str.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    return 0.0


def dedup_frame_points(points: list[float], min_gap: float = VISION_DEDUP_SECONDS) -> list[float]:
    """对抽帧点去重：相隔小于 min_gap 秒的只保留一个。"""
    if not points:
        return []
    sorted_pts = sorted(set(points))
    result = [sorted_pts[0]]
    for p in sorted_pts[1:]:
        if p - result[-1] >= min_gap:
            result.append(p)
    return result


def cap_frame_points(points: list[float], duration: float) -> list[float]:
    """根据视频时长限制抽帧总数，超出则均匀采样。"""
    if duration >= VISION_LONG_THRESHOLD:
        max_frames = VISION_MAX_FRAMES_LONG
    else:
        max_frames = VISION_MAX_FRAMES_SHORT

    if len(points) <= max_frames:
        return points

    step = len(points) / max_frames
    sampled = [points[int(i * step)] for i in range(max_frames)]
    return sampled


def extract_frame(video_path: Path, timestamp: float, output_path: Path) -> Path:
    """用 ffmpeg 在指定时间点抽取一帧为 JPG。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _find_bin("ffmpeg"),
        "-y",
        "-v", "error",
        "-ss", f"{timestamp:.3f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-q:v", "2",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 抽帧失败 @{timestamp:.1f}s: {result.stderr.strip()}")
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"抽帧输出为空 @{timestamp:.1f}s")
    return output_path


def _ensure_vlm_model() -> str:
    """确保本地有可用的多模态模型，返回模型名。"""
    ollama = find_executable("ollama")
    if ollama is None:
        raise RuntimeError("未找到 Ollama，请安装标准系统版本: brew install --cask ollama")

    models_to_try = [
        m
        for m in [
            VISION_VLM_MODEL,
            "qwen2.5vl:3b",
            "qwen2.5vl:7b",
            "llava:7b",
            "llava",
        ]
        if m
    ]
    try:
        result = subprocess.run([ollama, "list"], capture_output=True, text=True, timeout=30)
        installed = set()
        for line in result.stdout.splitlines()[1:]:
            parts = line.split()
            if parts:
                installed.add(parts[0])
    except Exception:
        installed = set()

    for model in models_to_try:
        if model in installed:
            return model

    for model in models_to_try:
        logger.info("正在拉取 VLM 模型: %s", model)
        try:
            result = subprocess.run(
                [ollama, "pull", model],
                capture_output=True,
                text=True,
                timeout=1800,
            )
            if result.returncode == 0:
                return model
        except Exception:
            continue

    raise RuntimeError("未能找到或拉取可用的本地多模态模型 (qwen2.5vl/llava)")


def describe_frames(
    frame_paths: list[Path],
    model: str,
    base_url: str = VISION_VLM_BASE_URL,
) -> list[str]:
    """逐帧调用 VLM 生成中文画面描述。"""
    descriptions = []
    for i, fp in enumerate(frame_paths):
        logger.info("  VLM 推理帧 %d/%d: %s", i + 1, len(frame_paths), fp.name)
        try:
            img_b64 = base64.b64encode(fp.read_bytes()).decode("ascii")
            resp = requests.post(
                f"{base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": VLM_PROMPT_SINGLE,
                    "images": [img_b64],
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 60},
                },
                timeout=300,
            )
            resp.raise_for_status()
            desc = resp.json().get("response", "").strip()
            descriptions.append(desc)
        except Exception as e:
            logger.error("  VLM 推理失败 @%s: %s", fp.name, e)
            descriptions.append("")
    if frame_paths and not any(descriptions):
        raise RuntimeError("所有关键帧的 VLM 推理均失败")
    return descriptions


def fuse_audio_visual(
    audio_summary: str,
    visual_descs: list[str],
    model: str,
    base_url: str = VISION_VLM_BASE_URL,
) -> str:
    """将语音摘要和画面描述融合为一段最终摘要。"""
    vis_text = "；".join(d for d in visual_descs if d)
    if not vis_text:
        return audio_summary
    if not audio_summary or audio_summary.startswith("["):
        return vis_text[:80]

    prompt = VLM_PROMPT_FUSE.format(audio_summary=audio_summary, visual_descs=vis_text)
    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 150},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        raise RuntimeError(f"音画融合失败: {e}") from e


def visual_only_summary(
    visual_descs: list[str],
    model: str,
    base_url: str = VISION_VLM_BASE_URL,
) -> dict:
    """无语音素材：根据多帧画面描述生成摘要和关键词。"""
    vis_text = "；".join(d for d in visual_descs if d)
    if not vis_text:
        return {"摘要": "[画面理解失败]", "关键词": []}

    prompt = VLM_PROMPT_VISUAL_ONLY.format(visual_descs=vis_text)
    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.3, "num_predict": 200},
            },
            timeout=120,
        )
        resp.raise_for_status()
        result = resp.json().get("response", "").strip()
        parsed = _parse_visual_summary(result)
        if not parsed["关键词"]:
            keyword_result = generate_summary_from_text(
                f"以下是纪录片素材的多帧画面描述，请提取3-5个便于检索的"
                f"地点、人物、动作或显著物体关键词：\n{vis_text}",
                ollama_base_url=base_url,
            )
            parsed["关键词"] = keyword_result.get("关键词", [])
        if parsed["摘要"]:
            if not parsed["关键词"]:
                parsed["关键词"] = [parsed["摘要"][:20]]
            return parsed
        return {"摘要": vis_text[:80], "关键词": [vis_text[:20]]}
    except Exception as e:
        raise RuntimeError(f"纯视觉摘要生成失败: {e}") from e


def _parse_visual_summary(response_text: str) -> dict:
    try:
        data = json.loads(response_text)
    except (json.JSONDecodeError, TypeError):
        return {"摘要": response_text[:80], "关键词": []}
    summary = str(data.get("摘要", "")).strip()
    keywords = data.get("关键词", [])
    if not isinstance(keywords, list):
        keywords = []
    return {
        "摘要": summary,
        "关键词": [
            str(keyword).strip()
            for keyword in keywords
            if str(keyword).strip()
        ][:5],
    }


def process_vision(
    video_path: Path,
    srt_path: Path | None,
    cache_dir: Path,
    duration: float,
    has_speech: bool,
    model: str | None = None,
    cache_key: str | None = None,
) -> dict:
    """对单个素材执行画面理解流程。

    Args:
        video_path: 源视频路径
        srt_path: 逐字稿 SRT 路径（有语音时提供）
        cache_dir: 帧缓存目录
        duration: 视频时长（秒）
        has_speech: 是否有有效语音
        model: VLM 模型名（None 则自动检测）

    Returns:
        {"frame_points": [...], "visual_descs": [...], "visual_summary": "...", "frame_count": N}
    """
    if model is None:
        model = _ensure_vlm_model()

    # 1. 收集抽帧点
    speech_pts = []
    scene_pts = []

    if has_speech and srt_path and srt_path.exists():
        speech_pts = get_speech_midpoints(srt_path)
        logger.debug("语音定位抽帧点: %s", speech_pts)

    scene_pts = detect_scene_changes(video_path)
    logger.debug("场景切换抽帧点: %s", scene_pts)

    # 2. 合并 + 去重 + 上限
    all_pts = sorted(set(speech_pts + scene_pts))

    # 确保至少覆盖开头和中间
    if duration > 0:
        all_pts = [duration * 0.1] + all_pts + [duration * 0.9]
        all_pts = [p for p in all_pts if 0 <= p < duration]

    all_pts = dedup_frame_points(all_pts)
    all_pts = cap_frame_points(all_pts, duration)
    logger.info("  最终抽帧点 (%d 帧): %s", len(all_pts), [f"{p:.1f}s" for p in all_pts])

    # 3. 抽帧
    material_stem = cache_key or video_path.stem
    frame_paths = []
    for i, ts in enumerate(all_pts):
        fp = cache_dir / f"{material_stem}_f{i}_{ts:.1f}s.jpg"
        if fp.exists() and fp.stat().st_size > 0:
            frame_paths.append(fp)
        else:
            try:
                extract_frame(video_path, ts, fp)
                frame_paths.append(fp)
            except Exception as e:
                logger.error("  抽帧失败 @%.1fs: %s", ts, e)

    if not frame_paths:
        raise RuntimeError("所有关键帧抽取均失败")

    # 4. VLM 逐帧描述
    visual_descs = describe_frames(frame_paths, model)

    # 5. 生成视觉摘要
    if has_speech:
        visual_summary = "；".join(d for d in visual_descs if d)[:120]
        visual_keywords = []
    else:
        visual_result = visual_only_summary(visual_descs, model)
        visual_summary = visual_result["摘要"]
        visual_keywords = visual_result["关键词"]

    return {
        "frame_points": all_pts,
        "visual_descs": visual_descs,
        "visual_summary": visual_summary,
        "visual_keywords": visual_keywords,
        "frame_count": len(frame_paths),
        "model": model,
    }
