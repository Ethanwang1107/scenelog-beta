"""Ollama 摘要 + 关键词生成"""

import json
import logging
import re
from pathlib import Path

import requests

from scenelog.config import SUMMARY_CHUNK_DURATION, SUMMARY_MAX_CHUNKS

logger = logging.getLogger(__name__)

SUMMARY_PROMPT = """你是一位纪录片场记助手。请根据以下逐字稿片段，生成：
1. 一句话内容摘要（不超过 50 字）
2. 3-5 个关键词标签

严格按以下 JSON 格式输出，不要有任何额外文字：
{{"摘要":"...","关键词":["...","..."]}}

逐字稿：
{text}"""

# 关键词为空时的强化重试提示
RETRY_PROMPT = """你是一位纪录片场记助手。请根据以下逐字稿片段，生成：
1. 一句话内容摘要（不超过 50 字）
2. 3-5 个关键词标签

注意：关键词是必须提供的，不能为空数组。请从逐字稿中提取核心主题词。

严格按以下 JSON 格式输出，不要有任何额外文字：
{{"摘要":"...","关键词":["关键词1","关键词2","关键词3"]}}

逐字稿：
{text}"""


def generate_summary(
    txt_path: Path,
    duration: float,
    ollama_base_url: str = "http://127.0.0.1:11434",
    model: str = "qwen2.5",
) -> dict:
    """为转录文本生成摘要和关键词。

    长音频按静音/时长分块，分别摘要后合并。

    Args:
        txt_path: 转录 TXT 文件路径
        duration: 音频时长（秒）
        ollama_base_url: Ollama 服务地址
        model: 模型名称

    Returns:
        {"摘要": "...", "关键词": ["...", "..."]}
    """
    if not txt_path.exists():
        raise RuntimeError(f"逐字稿不存在: {txt_path}")

    with open(txt_path, "r", encoding="utf-8") as f:
        full_text = f.read().strip()

    if not full_text or full_text == "[未识别出语音内容]":
        return {"摘要": "[无语音]", "关键词": []}

    # 短文本直接摘要
    if duration <= SUMMARY_CHUNK_DURATION or len(full_text) < 500:
        return _call_ollama(full_text, ollama_base_url, model)

    # 长文本分块摘要
    chunks = _split_text(full_text, SUMMARY_MAX_CHUNKS)
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        logger.debug("摘要分块 %d/%d", i + 1, len(chunks))
        result = _call_ollama(chunk, ollama_base_url, model)
        chunk_summaries.append(result.get("摘要", ""))

    # 合并分块摘要
    merged_text = " ".join(chunk_summaries)
    if not merged_text.strip():
        return {"摘要": "[摘要生成失败]", "关键词": []}

    # 对合并摘要再做一次总结
    merge_prompt = f"""以下是纪录片素材各片段的摘要，请合并为一句总摘要（不超过 50 字），并提取 3-5 个关键词。

严格按 JSON 格式输出：{{"摘要":"...","关键词":["...","..."]}}

各片段摘要：
{merged_text}"""

    return _call_ollama(merge_prompt, ollama_base_url, model)


def generate_summary_from_text(
    text: str,
    ollama_base_url: str = "http://127.0.0.1:11434",
    model: str = "qwen2.5",
) -> dict:
    """直接从文本生成摘要和关键词，供画面描述等非逐字稿内容复用。"""
    if not text.strip():
        return {"摘要": "", "关键词": []}
    return _call_ollama(text, ollama_base_url, model)


def _call_ollama(prompt_text: str, base_url: str, model: str, retry: bool = True) -> dict:
    """调用 Ollama API 获取摘要，关键词为空时自动重试一次。"""
    url = f"{base_url}/api/generate"

    payload = {
        "model": model,
        "prompt": SUMMARY_PROMPT.format(text=prompt_text),
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 256,
        },
    }

    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        response_text = data.get("response", "").strip()
    except requests.RequestException as e:
        logger.error("Ollama 请求失败: %s", e)
        raise RuntimeError(f"Ollama 摘要请求失败: {e}") from e

    # 解析 JSON 响应
    result = _parse_summary_response(response_text)
    if not result.get("摘要"):
        raise RuntimeError("Ollama 返回了空摘要")

    # 关键词为空且有摘要时，用强化提示重试一次
    if retry and result.get("摘要") and not result.get("关键词"):
        logger.warning("摘要已生成但关键词为空，使用强化提示重试...")
        retry_payload = {
            "model": model,
            "prompt": RETRY_PROMPT.format(text=prompt_text),
            "stream": False,
            "options": {
                "temperature": 0.5,
                "num_predict": 256,
            },
        }
        try:
            resp2 = requests.post(url, json=retry_payload, timeout=300)
            resp2.raise_for_status()
            data2 = resp2.json()
            retry_text = data2.get("response", "").strip()
            retry_result = _parse_summary_response(retry_text)
            # 保留原摘要，只替换关键词
            if retry_result.get("关键词"):
                result["关键词"] = retry_result["关键词"]
                logger.info("重试成功，获得 %d 个关键词", len(result["关键词"]))
            else:
                logger.warning("重试后关键词仍为空")
        except requests.RequestException as e:
            logger.error("关键词重试请求失败: %s", e)

    return result


def _parse_summary_response(text: str) -> dict:
    """从 Ollama 响应中提取 JSON。"""
    # 尝试直接解析
    try:
        result = json.loads(text)
        return _normalize_summary_result(result)
    except json.JSONDecodeError:
        pass

    # 尝试提取 JSON 块
    json_match = re.search(r'\{[^{}]*"摘要"[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            return _normalize_summary_result(result)
        except json.JSONDecodeError:
            pass

    # 兜底：把整个响应当摘要
    logger.warning("无法解析 Ollama JSON 响应，使用原始文本")
    return {"摘要": text[:100], "关键词": []}


def _normalize_summary_result(result: dict) -> dict:
    summary = result.get("摘要", "")
    keywords = result.get("关键词", [])
    if not isinstance(summary, str):
        summary = str(summary)
    if not isinstance(keywords, list):
        keywords = []
    normalized_keywords = [
        str(keyword).strip()
        for keyword in keywords
        if str(keyword).strip()
    ]
    return {
        "摘要": summary.strip(),
        "关键词": normalized_keywords[:5],
    }


def _split_text(text: str, max_chunks: int) -> list[str]:
    """将长文本按段落/句子边界分块。"""
    units = [
        unit.strip()
        for unit in re.split(r"(?:\r?\n)+|(?<=[。！？!?])", text)
        if unit.strip()
    ]
    if not units:
        return []

    target_chars = max(500, min(2000, (len(text) + max_chunks - 1) // max_chunks))
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for unit in units:
        if current and current_length + len(unit) > target_chars:
            chunks.append(" ".join(current))
            current = []
            current_length = 0
        current.append(unit)
        current_length += len(unit)
    if current:
        chunks.append(" ".join(current))

    if len(chunks) <= max_chunks:
        return chunks

    merged: list[str] = []
    group_size = (len(chunks) + max_chunks - 1) // max_chunks
    for start in range(0, len(chunks), group_size):
        merged.append(" ".join(chunks[start : start + group_size]))
    return merged[:max_chunks]
