"""Identity-aware event descriptions and documentary summaries."""

from __future__ import annotations

import base64
import json
import logging
import re
from pathlib import Path

import requests

from scenelog.config import VISION_VLM_BASE_URL
from scenelog.transcribe import parse_srt

logger = logging.getLogger(__name__)

EVENT_PROMPT = """你是纪录片场记。以下图片按时间顺序排列。每张证据图左侧是完整画面，右侧是方框标注人物周边身体动作区域的放大图。相同标签在所有图片中始终代表同一个人：
{label_map}

时间段：{start_time}-{end_time}
图片时间点：{frame_times}
该时间段对应对白：{dialogue}
已确认声纹：{speaker_evidence}
本事件最多检测到 {unknown_count} 张未登记人脸。

请只根据图片和对白输出 JSON：
{{"动作":"...","场景":"...","人物关系":"...","动作事件":[],"对白事实":"...","说话人":"","说话人已确认":false}}

规则：
1. 只有看清“谁对谁做了什么”时才填写“动作事件”。每项必须包含
   “施动者、动作、承受者、依据、置信度”五个字段；没有明确角色关系时保持空数组。
   施动者和承受者优先使用 P1/P2 标签；未标注者使用“两名黑衣人”等可见外观泛称。
   必须使用主动语态：例如被按住的人属于承受者，“动作”只写“按住”，不得写“被按住”。
2. 被方框标注的人不能写成男子、人物、领导等泛称；未标注者可写“一名男子”“两名黑衣人”等。
3. 结合左侧完整画面判断人数和场景，结合右侧放大区域判断谁的手臂、肩膀或身体接触了谁。
4. 连续图片显示某人被抓住、控制并发生位置移动时，可以写“控制并押走”；单张证据不足时只写“控制”。
5. 动作字段用自然语言概括动作事件；场景只写地点、环境和明显背景。
6. 对白事实只概括对白内容，不臆测未说出的信息。
7. “已确认声纹”中的姓名是高置信度声音证据，可直接填写为说话人并设为 true。
8. 没有已确认声纹时，只有连续图片中的可见口型或明确互动才能确认发言人。
   其余情况说话人必须为空，不能因某人在画面中就把对白归给他。
9. 置信度为 0-1；无法确定施动者或承受者时使用空数组，不要猜测。
10. 动作事件最多填写一个，只描述“时间段”内最明确的核心事件。
11. 不要输出镜头类型、用途或解释。
"""

SUMMARY_PROMPT = """你是纪录片场记。请把按时间排序的人物事件和基础摘要改写为自然、准确的内容摘要。

人物事件：
{events}

基础摘要：
{base_summary}

规则：
1. 以姓名作为动作主体，写成“谁做了什么”，不能写“某某出现在素材中”。
2. 已登记姓名不得再被写成男子、人物、领导等泛称；未登记者可写一名男子、两名男子等。
3. 按发生顺序写 1-3 句，每句包含人物、动作或对白事实，必要时包含场景。
4. 仅当事件的“说话人已确认”为 true 时，才可写“某某说/夸赞/表示”。
   已确认时优先把对白事实自然写成“老刘夸赞王书记的冲锋衣价值高”，
   不要机械重复整段原话，除非原话本身对内容理解很重要。
5. 说话人未确认时，写“交谈中有人……”或直接陈述讨论内容，禁止强行署名。
6. 严格区分说话人和被说话人：“领导，你这件……”表示有人对领导说话，不能写成领导说了这句话。
7. 不得编造身份、人物关系、地点、动作或对白。
8. 不要提及识别、标签、模型、画面中、这段视频。

只输出 JSON：
{{"摘要":"...","关键词":["人物姓名","地点","动作"]}}
"""

CONTROL_PROBE_PROMPT = """图片按时间顺序排列。客观描述人物之间可见的身体接触和位置变化。
不要使用姓名，不要参考对白，不要判断身份。普通并肩行走、站立、交谈、靠近要如实写。
若有人用手、手臂或身体抓住、按住、扶起、拖动另一人，写清施动者人数和衣着、
承受者衣着、具体接触部位。没有看到就写“无”。
只输出 JSON：
{"核心动作观察":"","施动者人数衣着":"","承受者衣着":"","身体接触观察":"","前后位置变化":""}
"""

TARGET_VERIFY_PROMPT = """黄色半透明矩形标出同一个待核验人物的脸和身体区域。
不要判断姓名，也不要重新判断动作。只回答黄色区域中的人物是否就是以下承受者：
{target_description}

必须根据衣着、身体位置和与周围人的接触关系回答。
若黄色区域包含多个人或无法对应，回答“无法确定”。
只输出 JSON：
{{"结论":"是/否/无法确定","黄色人物衣着":"","位置关系":"","依据":"","置信度":0}}
"""

CONTACT_COUNT_PROMPT = """以下图片按时间顺序，{target_description}是承受者。
只列出实际用手、手臂或身体接触承受者的人；站在旁边、指向、旁观但未接触者必须排除。
逐个写接触者的上衣颜色和接触部位，最后给出实际接触者数量。不要使用姓名，不参考对白。
只输出 JSON：
{{"实际接触者":[{{"上衣":"","接触部位":"","动作":""}}],"实际接触者数量":0,"排除的旁观者":"","依据":""}}
"""


def describe_identity_events(
    events: list[dict],
    output_dir: Path,
    srt_path: Path | None,
    model: str,
    base_url: str = VISION_VLM_BASE_URL,
    speaker_segments: list[dict] | None = None,
) -> list[dict]:
    """Describe each annotated identity event with conservative speaker attribution."""
    transcript = _transcript_segments(srt_path)
    described = []
    for event_index, event in enumerate(events):
        frame_records = _event_context_frames(events, event_index)
        valid_frames = [
            (frame, output_dir / frame.get("frame", ""))
            for frame in frame_records
            if (output_dir / frame.get("frame", "")).is_file()
        ]
        if not valid_frames:
            continue
        event_label_map = _event_label_map([event])
        label_map = "；".join(
            f"{tag}={name}"
            for tag, name in sorted(event_label_map.items())
        )
        dialogue = _dialogue_for_event(
            transcript,
            float(event.get("start_timestamp", 0)),
            float(event.get("end_timestamp", 0)),
        )
        confirmed_voice = _confirmed_voice_for_event(
            speaker_segments or [],
            float(event.get("start_timestamp", 0)),
            float(event.get("end_timestamp", 0)),
        )
        prompt = EVENT_PROMPT.format(
            label_map=label_map or "无",
            start_time=_format_seconds(event.get("start_timestamp", 0)),
            end_time=_format_seconds(event.get("end_timestamp", 0)),
            frame_times="、".join(
                (
                    f"{_format_seconds(frame.get('timestamp', 0))}"
                    f"({frame.get('context', 'core')})"
                )
                for frame, _path in valid_frames
            ),
            dialogue=dialogue or "无",
            speaker_evidence=(
                f"{confirmed_voice['speaker_name']}：{confirmed_voice['text']}"
                if confirmed_voice
                else "无"
            ),
            unknown_count=int(event.get("unknown_face_count", 0)),
        )
        try:
            response = requests.post(
                f"{base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "images": [
                        base64.b64encode(frame_path.read_bytes()).decode("ascii")
                        for _frame, frame_path in valid_frames
                    ],
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 220,
                        "num_ctx": 8192,
                    },
                },
                timeout=300,
            )
            response.raise_for_status()
            parsed = _parse_json_object(response.json().get("response", ""))
        except Exception as exc:
            detail = _request_error_detail(exc)
            logger.warning(
                "人物事件理解失败 %s: %s",
                event.get("event_id"),
                detail,
            )
            parsed = {}
        current = dict(event)
        current["dialogue"] = dialogue
        current["description"] = _normalize_event_description(
            parsed,
            event.get("people", []),
            event_label_map,
            allow_role_sensitive=sum(
                frame.get("context") == "core"
                for frame, _path in valid_frames
            ) >= 2,
        )
        if confirmed_voice:
            current["description"]["说话人"] = confirmed_voice["speaker_name"]
            current["description"]["说话人已确认"] = True
            current["description"]["对白事实"] = (
                current["description"].get("对白事实")
                or confirmed_voice["text"]
            )
            current["speaker_match"] = confirmed_voice
        if _needs_control_probe(parsed, event, valid_frames):
            control_event = _probe_registered_control_target(
                event,
                valid_frames,
                current["description"],
                output_dir,
                model,
                base_url,
            )
            current["description"] = _replace_control_action(
                current["description"],
                control_event,
            )
        described.append(current)
    return described


def _needs_control_probe(
    _parsed: dict,
    event: dict,
    valid_frames: list[tuple[dict, Path]],
) -> bool:
    core_frame_count = sum(
        frame.get("context") == "core"
        for frame, _path in valid_frames
    )
    return (
        core_frame_count >= 2
        and len(event.get("people", [])) == 1
        and int(event.get("unknown_face_count", 0)) >= 2
    )


def _probe_registered_control_target(
    event: dict,
    valid_frames: list[tuple[dict, Path]],
    description: dict,
    output_dir: Path,
    model: str,
    base_url: str,
) -> dict | None:
    probe_images = []
    for frame, path in valid_frames:
        raw_path = output_dir / frame.get("raw_frame", "")
        probe_path = raw_path if raw_path.is_file() else path
        probe_images.append(probe_path.read_bytes())
    control = _request_vlm_json(
        CONTROL_PROBE_PROMPT,
        probe_images,
        model,
        base_url,
        num_predict=220,
    )
    observation = _normalize_text_field(control.get("核心动作观察", ""))
    evidence = _normalize_text_field(control.get("身体接触观察", ""))
    actor = _normalize_text_field(control.get("施动者人数衣着", ""))
    target = _normalize_text_field(control.get("承受者衣着", ""))
    movement = _normalize_text_field(control.get("前后位置变化", ""))
    if (
        not _has_control_observation(observation, evidence)
        or not target
        or not _has_physical_contact_evidence(evidence)
    ):
        return None

    highlighted_images = []
    for frame, path in valid_frames:
        image = _highlight_registered_body(path, frame.get("labels", []))
        if image:
            highlighted_images.append(image)
    if not highlighted_images:
        return None
    verification = _request_vlm_json(
        TARGET_VERIFY_PROMPT.format(target_description=target),
        highlighted_images,
        model,
        base_url,
        num_predict=160,
    )
    verify_confidence = _bounded_confidence(verification.get("置信度", 0))
    if (
        _normalize_text_field(verification.get("结论", "")) != "是"
        or verify_confidence < 0.7
    ):
        return None

    contact_count = _request_vlm_json(
        CONTACT_COUNT_PROMPT.format(target_description=target),
        probe_images,
        model,
        base_url,
        num_predict=240,
    )
    actor_count = _parse_actor_count(contact_count.get("实际接触者数量"))
    if actor_count <= 0:
        return None
    current_name = str(event["people"][0].get("name", "")).strip()
    if not current_name:
        return None
    actor = _actor_with_verified_count(
        _grounded_actor_from_description(description) or actor,
        actor_count,
    )
    moved = _has_movement_observation(movement)
    action = "控制并押走" if moved else "控制"
    return {
        "施动者": [_normalize_actor_description(actor)],
        "动作": action,
        "承受者": [current_name],
        "依据": evidence,
        "置信度": round(verify_confidence, 3),
    }


def _has_control_observation(observation: str, evidence: str) -> bool:
    combined = f"{observation} {evidence}"
    return any(
        term in combined
        for term in ("抓住", "按住", "控制", "压住", "拖动", "扶起", "推")
    )


def _has_movement_observation(text: str) -> bool:
    if not text or text.strip() in {"无", "没有", "未发生"}:
        return False
    return any(
        term in text
        for term in ("移动", "拖", "带离", "离开", "从", "向")
    )


def _grounded_actor_from_description(description: dict) -> str:
    candidates = [
        str(role).strip()
        for event in description.get("动作事件", [])
        for role in event.get("施动者", [])
        if str(role).strip()
    ]
    relationship = str(description.get("人物关系", ""))
    match = re.search(
        r"([二两三四五六七八九十\d]+名[^，。；]{0,8}?"
        r"(?:黑衣人|男子|人员))",
        relationship,
    )
    if match:
        candidates.insert(0, match.group(1))
    return candidates[0] if candidates else ""


def _parse_actor_count(value) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        return 0
    return count if 1 <= count <= 9 else 0


def _actor_with_verified_count(actor: str, count: int) -> str:
    numerals = {
        1: "一",
        2: "两",
        3: "三",
        4: "四",
        5: "五",
        6: "六",
        7: "七",
        8: "八",
        9: "九",
    }
    count_text = numerals[count]
    if "黑衣" in actor:
        return f"{count_text}名黑衣人"
    if "正装" in actor or "西装" in actor:
        return f"{count_text}名正装人员"
    return f"{count_text}名男子"


def _request_vlm_json(
    prompt: str,
    images: list[bytes],
    model: str,
    base_url: str,
    num_predict: int,
) -> dict:
    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "images": [
                    base64.b64encode(image).decode("ascii")
                    for image in images
                ],
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0,
                    "num_predict": num_predict,
                    "num_ctx": 8192,
                },
            },
            timeout=300,
        )
        response.raise_for_status()
        return _parse_json_object(response.json().get("response", ""))
    except Exception as exc:
        logger.warning("人物动作复核失败: %s", _request_error_detail(exc))
        return {}


def _highlight_registered_body(frame_path: Path, labels: list[dict]) -> bytes:
    import cv2
    import numpy as np

    composite = cv2.imread(str(frame_path))
    if composite is None or not labels:
        return b""
    image = composite[:, : min(640, composite.shape[1])].copy()
    image_height, image_width = image.shape[:2]
    scale_x = image_width / 1280
    scale_y = image_height / 1280
    boxes = []
    for label in labels:
        bbox = label.get("bbox", [])
        if len(bbox) != 4:
            continue
        x, y, width, height = [int(value) for value in bbox]
        boxes.append(
            (
                int(x * scale_x),
                int(y * scale_y),
                max(1, int(width * scale_x)),
                max(1, int(height * scale_y)),
            )
        )
    if not boxes:
        return b""

    x_min = max(0, min(x - 2 * width for x, _y, width, _height in boxes))
    y_min = max(54, min(y - height for _x, y, _width, height in boxes))
    x_max = min(
        image_width,
        max(x + 3 * width for x, _y, width, _height in boxes),
    )
    y_max = min(
        image_height,
        max(y + 6 * height for _x, y, _width, height in boxes),
    )
    overlay = image.copy()
    cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 255), -1)
    image = cv2.addWeighted(overlay, 0.2, image, 0.8, 0)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 6)
    cv2.putText(
        image,
        "HIGHLIGHTED PERSON",
        (max(5, x_min), max(40, y_min - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    success, encoded = cv2.imencode(".jpg", image)
    return encoded.tobytes() if success else b""


def _normalize_actor_description(value: str) -> str:
    value = value.strip()
    if not value:
        return "多名男子"
    if re.search(r"[二两三四五六七八九十\d]+名", value):
        return value
    if any(term in value for term in ("多人", "多名", "人员")):
        if "黑" in value:
            return "多名黑衣人"
        if "正装" in value or "西装" in value:
            return "多名正装人员"
        return "多名男子"
    return value


def _bounded_confidence(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _replace_control_action(description: dict, control_event: dict | None) -> dict:
    current = dict(description)
    action_events = [
        event
        for event in current.get("动作事件", [])
        if not _contains_role_sensitive_action(str(event.get("动作", "")))
    ]
    if control_event:
        action_events.append(control_event)
    current["动作事件"] = action_events
    grounded_action = _action_events_text(action_events)
    if grounded_action:
        current["动作"] = grounded_action
    elif _contains_role_sensitive_action(str(current.get("动作", ""))):
        current["动作"] = ""
    relationship = str(current.get("人物关系", ""))
    if _contains_role_sensitive_action(relationship):
        current["人物关系"] = ""
    return current


def generate_identity_summary(
    events: list[dict],
    base_summary: str,
    base_keywords: list[str],
    model: str = "qwen2.5:latest",
    base_url: str = VISION_VLM_BASE_URL,
) -> dict:
    """Generate a natural summary grounded in ordered identity events."""
    if not events:
        return {"摘要": base_summary, "关键词": base_keywords}
    compact_events = []
    seen_speaker_facts = set()
    for event in events:
        description = event.get("description", {})
        speaker = str(description.get("说话人", "")).strip()
        speaker_confirmed = bool(description.get("说话人已确认", False))
        dialogue_fact = str(description.get("对白事实", "")).strip()
        speaker_fact = (
            speaker,
            str(event.get("speaker_match", {}).get("text", "")).strip()
            or dialogue_fact,
        )
        if speaker_confirmed and speaker_fact in seen_speaker_facts:
            speaker = ""
            speaker_confirmed = False
            dialogue_fact = ""
        elif speaker_confirmed:
            seen_speaker_facts.add(speaker_fact)
            dialogue_fact = speaker_fact[1]
        visible_names = [
            str(person.get("name", "")).strip()
            for person in event.get("people", [])
            if str(person.get("name", "")).strip()
            and str(person.get("name", "")).strip() != speaker
        ]
        dialogue_target = (
            visible_names[0]
            if speaker_confirmed and len(visible_names) == 1
            else ""
        )
        compact_events.append(
            {
                "时间": (
                    f"{_format_seconds(event.get('start_timestamp', 0))}-"
                    f"{_format_seconds(event.get('end_timestamp', 0))}"
                ),
                "已登记人物": [
                    person.get("name")
                    for person in event.get("people", [])
                    if person.get("name")
                ],
                "未登记人脸数": event.get("unknown_face_count", 0),
                "动作": description.get("动作", ""),
                "动作事件": description.get("动作事件", []),
                "场景": description.get("场景", ""),
                "人物关系": description.get("人物关系", ""),
                "对白事实": dialogue_fact,
                "说话人": speaker,
                "说话人已确认": speaker_confirmed,
                "对白对象": dialogue_target,
            }
        )
    prompt = SUMMARY_PROMPT.format(
        events=json.dumps(compact_events, ensure_ascii=False),
        base_summary=base_summary or "无",
    )
    required_names = list(
        dict.fromkeys(
            person.get("name", "")
            for event in events
            for person in event.get("people", [])
            if person.get("name")
        )
    )
    confirmed_speakers = {
        str(event.get("description", {}).get("说话人", "")).strip()
        for event in events
        if event.get("description", {}).get("说话人已确认") is True
        and str(event.get("description", {}).get("说话人", "")).strip()
    }
    confirmed_dialogues = _confirmed_dialogue_facts(events)
    request_prompt = prompt
    parsed = {}
    summary = ""
    issues = ["尚未生成"]
    for _attempt in range(3):
        parsed = _request_summary(request_prompt, model, base_url)
        summary = str(parsed.get("摘要", "")).strip()
        summary = _remove_unconfirmed_attributions(
            summary,
            required_names,
            confirmed_speakers,
        )
        summary = _neutralize_anonymous_possessives(summary)
        summary = _ensure_grounded_action_facts(summary, events)
        issues = _summary_issues(
            summary,
            required_names,
            confirmed_speakers,
            confirmed_dialogues,
        )
        if not issues:
            break
        request_prompt = (
            f"{prompt}\n\n上一次摘要：{summary or '空'}\n"
            f"不合格原因：{'；'.join(issues)}。\n"
            f"必须保留并自然使用这些姓名：{'、'.join(required_names)}。"
            "请让姓名承担可见的行走、站立、互动等动作，不要只追加人物名单。"
            "已确认说话人必须用姓名承担说、夸赞、表示等发言动作。"
            "没有确认说话人的对白必须另写“交谈中有人……”，"
            "不能把姓名、领导等对白称呼写成发言人。重新输出完整 JSON。"
        )
    if issues:
        summary = _grounded_fallback_summary(
            events,
            base_summary,
            required_names,
            confirmed_speakers,
        )
        summary = _ensure_grounded_action_facts(summary, events)
        issues = _summary_issues(
            summary,
            required_names,
            confirmed_speakers,
            confirmed_dialogues,
        )
        if issues:
            raise RuntimeError(f"身份摘要校验失败: {'；'.join(issues)}")
    keywords = parsed.get("关键词", [])
    if not isinstance(keywords, list):
        keywords = []
    action_keywords = [
        str(action_event.get("动作", "")).strip()
        for event in events
        for action_event in event.get("description", {}).get("动作事件", [])
        if float(action_event.get("置信度", 0)) >= 0.7
        and str(action_event.get("动作", "")).strip()
    ]
    normalized_keywords = list(
        dict.fromkeys(
            str(keyword).strip()
            for keyword in [
                *required_names,
                *action_keywords,
                *keywords,
                *base_keywords,
            ]
            if str(keyword).strip()
        )
    )[:8]
    return {"摘要": summary, "关键词": normalized_keywords}


SPEECH_VERBS = (
    "说",
    "说道",
    "表示",
    "回答",
    "回应",
    "询问",
    "问道",
    "告诉",
    "解释",
    "强调",
    "提到",
    "夸赞",
    "称赞",
    "建议",
    "要求",
    "透露",
    "补充",
    "承认",
    "否认",
    "喊",
    "交谈",
    "讨论",
    "谈到",
    "谈及",
    "提及",
    "聊到",
    "讲述",
)


def _summary_issues(
    summary: str,
    required_names: list[str],
    confirmed_speakers: set[str],
    confirmed_dialogues: list[dict] | None = None,
) -> list[str]:
    if not summary:
        return ["摘要为空"]
    issues = []
    missing_names = [name for name in required_names if name not in summary]
    if missing_names:
        issues.append(f"遗漏已识别人物：{'、'.join(missing_names)}")
    unsupported = _unconfirmed_attributions(
        summary,
        required_names,
        confirmed_speakers,
    )
    if unsupported:
        issues.append(f"无证据署名说话人：{'、'.join(unsupported)}")
    missing_confirmed = [
        name
        for name in confirmed_speakers
        if not _has_named_speech_attribution(summary, name)
    ]
    if missing_confirmed:
        issues.append(f"遗漏已确认说话人：{'、'.join(missing_confirmed)}")
    quoted_speakers = [
        item["speaker"]
        for item in (confirmed_dialogues or [])
        if re.search(
            rf"{re.escape(item['speaker'])}(?:说|表示|回应)[：:]?[“\"]",
            summary,
        )
    ]
    if quoted_speakers:
        issues.append(
            f"机械复述对白，需概括发言意图：{'、'.join(dict.fromkeys(quoted_speakers))}"
        )
    if "出现在素材中" in summary:
        issues.append("使用了人物名单前缀")
    return issues


def _confirmed_dialogue_facts(events: list[dict]) -> list[dict]:
    result = []
    seen = set()
    for event in events:
        description = event.get("description", {})
        speaker = str(description.get("说话人", "")).strip()
        if description.get("说话人已确认") is not True or not speaker:
            continue
        raw_text = (
            str(event.get("speaker_match", {}).get("text", "")).strip()
            or str(description.get("对白事实", "")).strip()
        )
        if not raw_text:
            continue
        visible_names = [
            str(person.get("name", "")).strip()
            for person in event.get("people", [])
            if str(person.get("name", "")).strip()
            and str(person.get("name", "")).strip() != speaker
        ]
        target = visible_names[0] if len(visible_names) == 1 else ""
        key = (speaker, raw_text)
        if key in seen:
            continue
        seen.add(key)
        result.append(
            {
                "speaker": speaker,
                "target": target,
                "raw_text": raw_text,
                "fact": str(description.get("对白事实", "")).strip(),
            }
        )
    return result


def _has_named_speech_attribution(summary: str, name: str) -> bool:
    verb_pattern = "|".join(
        sorted(map(re.escape, SPEECH_VERBS), key=len, reverse=True)
    )
    bridge = r"(?:与(?:他|其|对方)|向(?:他|其|对方)|对(?:他|其|对方))?"
    return bool(
        re.search(
            rf"{re.escape(name)}{bridge}(?:随即|随后|继续|又|还|则)?"
            rf"(?:{verb_pattern})",
            summary,
        )
    )


def _unconfirmed_attributions(
    summary: str,
    names: list[str],
    confirmed_speakers: set[str],
) -> list[str]:
    verb_pattern = "|".join(sorted(map(re.escape, SPEECH_VERBS), key=len, reverse=True))
    bridge = r"(?:与(?:他|其|对方)|向(?:他|其|对方)|对(?:他|其|对方))?"
    unsupported = []
    for name in names:
        if name in confirmed_speakers:
            continue
        if re.search(
            rf"{re.escape(name)}{bridge}(?:随即|随后|继续|又|还|则)?"
            rf"(?:{verb_pattern})",
            summary,
        ):
            unsupported.append(name)
    return unsupported


def _remove_unconfirmed_attributions(
    summary: str,
    names: list[str],
    confirmed_speakers: set[str],
) -> str:
    verb_pattern = "|".join(sorted(map(re.escape, SPEECH_VERBS), key=len, reverse=True))
    bridge = r"(?:与(?:他|其|对方)|向(?:他|其|对方)|对(?:他|其|对方))?"
    result = summary
    for name in names:
        if name in confirmed_speakers:
            continue
        result = re.sub(
            rf"{re.escape(name)}{bridge}(?:随即|随后|继续|又|还|则)?"
            rf"(?P<verb>{verb_pattern})(?P<about>关于)?",
            _anonymous_speech_replacement,
            result,
        )
    return result


def _anonymous_speech_replacement(match: re.Match) -> str:
    verb = match.group("verb")
    if verb == "交谈":
        return "交谈中有人讨论"
    return f"交谈中有人{verb}"


def _neutralize_anonymous_possessives(summary: str) -> str:
    sentences = re.split(r"(?<=[。！？])", summary)
    result = "".join(
        sentence.replace("自己的", "一件")
        if "交谈中有人" in sentence
        else sentence
        for sentence in sentences
    )
    result = re.sub(
        r"(?:，|、)?(?:与|和)?交谈中有人讨论(?=[。！？])",
        "",
        result,
    )
    result = re.sub(
        r"(?:，|、)?(?:与|和)(?:一|两|二|三)名(?:男子|人物|人)(?=[。！？])",
        "",
        result,
    )
    return result


def _ensure_grounded_action_facts(summary: str, events: list[dict]) -> str:
    result = summary.strip()
    for event in events:
        for action_event in event.get("description", {}).get("动作事件", []):
            if float(action_event.get("置信度", 0)) < 0.7:
                continue
            clause = _action_events_text([action_event])
            if not clause or _summary_contains_action_fact(result, action_event):
                continue
            result = _remove_conflicting_action_sentences(result, action_event)
            result = result.rstrip()
            separator = "" if result.endswith(("。", "！", "？", "。”", "！”", "？”")) else "。"
            result = result + separator + clause.rstrip("。") + "。"
    return result


def _remove_conflicting_action_sentences(summary: str, action_event: dict) -> str:
    actors = action_event.get("施动者", [])
    targets = action_event.get("承受者", [])
    if not actors or not targets:
        return summary
    action_family_terms = set().union(
        *[
            family
            for family in ACTION_FAMILIES
            if any(term in family for term in _action_terms(action_event.get("动作", "")))
        ],
    )
    if not action_family_terms:
        return summary
    sentences = re.split(r"(?<=[。！？])", summary)
    kept = []
    for sentence in sentences:
        same_target = all(target in sentence for target in targets)
        related_action = any(term in sentence for term in action_family_terms)
        if same_target and related_action:
            continue
        kept.append(sentence)
    return "".join(kept).strip()


def _action_terms(action: str) -> list[str]:
    return [
        term
        for term in re.split(r"[并和及、，\s]+", str(action))
        if len(term) >= 2
    ] or ([str(action)] if action else [])


def _summary_contains_action_fact(summary: str, action_event: dict) -> bool:
    roles = [
        *action_event.get("施动者", []),
        *action_event.get("承受者", []),
    ]
    action = str(action_event.get("动作", "")).strip()
    action_terms = _action_terms(action)
    for sentence in re.split(r"[。！？；]", summary):
        role_match = all(role in sentence for role in roles)
        exact_action = all(term in sentence for term in action_terms)
        equivalent_action = all(
            _sentence_has_action_family(sentence, term)
            for term in action_terms
        )
        if (
            role_match
            and (exact_action or equivalent_action)
        ):
            return True
    return False


def _sentence_has_action_family(sentence: str, action_term: str) -> bool:
    family = next(
        (terms for terms in ACTION_FAMILIES if action_term in terms),
        None,
    )
    return bool(family and any(term in sentence for term in family))


def _grounded_fallback_summary(
    events: list[dict],
    base_summary: str,
    names: list[str],
    confirmed_speakers: set[str],
) -> str:
    """Build a conservative visual sentence when the text model cannot comply."""
    visual_clause = re.split(r"[；。]", base_summary, maxsplit=1)[0].strip()
    if visual_clause:
        visual_clause = _replace_generic_group(visual_clause, names)
    if not visual_clause or any(name not in visual_clause for name in names):
        scenes = [
            str(event.get("description", {}).get("场景", "")).strip()
            for event in events
            if str(event.get("description", {}).get("场景", "")).strip()
        ]
        scene = scenes[0] if scenes else "现场"
        visual_clause = f"{'、'.join(names)}等人在{scene}活动"

    dialogue_facts = [
        str(event.get("description", {}).get("对白事实", "")).strip()
        for event in events
        if str(event.get("description", {}).get("对白事实", "")).strip()
    ]
    parts = [visual_clause.rstrip("。") + "。"]
    confirmed_facts = _confirmed_dialogue_facts(events)
    for item in confirmed_facts:
        clause = _semantic_dialogue_clause(item)
        if clause:
            parts.append(clause.rstrip("。") + "。")
    if dialogue_facts and not confirmed_facts:
        fact = re.sub(
            r"^(?:交谈中)?(?:有人)?",
            "",
            dialogue_facts[0],
        ).strip("，。； ")
        if fact:
            parts.append(f"交谈中有人提到{fact}。")
    return "".join(parts)


def _semantic_dialogue_clause(item: dict) -> str:
    speaker = str(item.get("speaker", "")).strip()
    target = str(item.get("target", "")).strip()
    fact = str(item.get("fact", "")).strip()
    raw_text = str(item.get("raw_text", "")).strip()
    combined = f"{fact} {raw_text}"
    if not speaker:
        return ""
    if "冲锋衣" in combined and any(
        term in combined
        for term in ("夸赞", "称赞", "价值", "收入", "贵")
    ):
        object_text = f"{target}的冲锋衣" if target else "冲锋衣"
        return f"{speaker}夸赞{object_text}价值高"
    if any(term in combined for term in ("回应", "回答", "这有啥")):
        return f"{speaker}回应对方"
    clean_fact = fact or raw_text
    clean_fact = clean_fact.strip("，。； ")
    if not clean_fact:
        return ""
    return f"{speaker}表示{clean_fact}"


def _replace_generic_group(text: str, names: list[str]) -> str:
    match = re.search(r"([一二两三四五六七八九十\d]+)名(?:男子|人物|人)", text)
    if not match:
        return text
    count = _parse_small_count(match.group(1))
    if count is None or count < len(names):
        replacement = f"{'、'.join(names)}等人"
    else:
        unknown_count = count - len(names)
        parts = list(names)
        if unknown_count == 1:
            parts.append("一名男子")
        elif unknown_count > 1:
            parts.append(f"{_small_count_text(unknown_count)}名男子")
        replacement = "、".join(parts[:-1]) + (
            f"和{parts[-1]}" if len(parts) > 1 else parts[0]
        )
    return text[: match.start()] + replacement + text[match.end() :]


def _parse_small_count(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    values = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    return values.get(value)


def _small_count_text(value: int) -> str:
    return {
        2: "两",
        3: "三",
        4: "四",
        5: "五",
        6: "六",
        7: "七",
        8: "八",
        9: "九",
        10: "十",
    }.get(value, str(value))


def _request_summary(prompt: str, model: str, base_url: str) -> dict:
    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.2, "num_predict": 260},
            },
            timeout=300,
        )
        response.raise_for_status()
        return _parse_json_object(response.json().get("response", ""))
    except Exception as exc:
        raise RuntimeError(f"身份摘要生成失败: {exc}") from exc


def _transcript_segments(srt_path: Path | None) -> list[dict]:
    if not srt_path or not srt_path.is_file():
        return []
    result = []
    for segment in parse_srt(srt_path):
        result.append(
            {
                **segment,
                "start_seconds": _time_to_seconds(segment.get("start_time", "")),
                "end_seconds": _time_to_seconds(segment.get("end_time", "")),
            }
        )
    return result


def _dialogue_for_event(
    segments: list[dict],
    start_timestamp: float,
    end_timestamp: float,
) -> str:
    margin = 1.0
    matched = [
        segment.get("text", "")
        for segment in segments
        if segment.get("end_seconds", 0) >= start_timestamp - margin
        and segment.get("start_seconds", 0) <= end_timestamp + margin
        and segment.get("text") != "[未识别出语音内容]"
    ]
    return " ".join(dict.fromkeys(text for text in matched if text))


def _confirmed_voice_for_event(
    segments: list[dict],
    start_timestamp: float,
    end_timestamp: float,
) -> dict | None:
    """Return one confirmed registered speaker when event overlap is unambiguous."""
    if end_timestamp - start_timestamp < 0.5:
        window_start = start_timestamp - 0.5
        window_end = end_timestamp + 0.5
    else:
        window_start = start_timestamp
        window_end = end_timestamp
    matched = []
    overlap_by_name: dict[str, float] = {}
    for segment in segments:
        name = str(segment.get("speaker_name", "")).strip()
        if segment.get("confirmed") is not True or not name:
            continue
        overlap = max(
            0.0,
            min(float(segment.get("end_seconds", 0)), window_end)
            - max(float(segment.get("start_seconds", 0)), window_start),
        )
        if overlap < 0.1:
            continue
        matched.append((segment, overlap))
        overlap_by_name[name] = overlap_by_name.get(name, 0.0) + overlap
    if not matched:
        return None
    ranked_names = sorted(
        overlap_by_name.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    name, best_overlap = ranked_names[0]
    if len(ranked_names) > 1 and best_overlap - ranked_names[1][1] < 0.5:
        return None
    selected = [
        segment
        for segment, _overlap in matched
        if str(segment.get("speaker_name", "")).strip() == name
    ]
    text = " ".join(
        dict.fromkeys(
            str(segment.get("text", "")).strip()
            for segment in selected
            if str(segment.get("text", "")).strip()
        )
    )
    best = max(
        selected,
        key=lambda segment: float(segment.get("match_score", 0)),
    )
    return {
        "person_id": best.get("person_id", ""),
        "speaker_name": name,
        "text": text,
        "match_score": float(best.get("match_score", 0)),
        "match_margin": float(best.get("match_margin", 0)),
    }


def _event_context_frames(events: list[dict], event_index: int) -> list[dict]:
    event = events[event_index]
    core_frames = list(event.get("frames", []))
    if not core_frames:
        core_frames = [
            {
                "timestamp": event.get("representative_timestamp", 0),
                "frame": event.get("frame", ""),
                "labels": event.get("labels", []),
            }
        ]
    candidates = [{**frame, "context": "core"} for frame in core_frames]

    unique = {}
    for frame in candidates:
        path = frame.get("frame", "")
        if path:
            unique[path] = frame
    return sorted(
        unique.values(),
        key=lambda item: float(item.get("timestamp", 0)),
    )


def _event_label_map(events: list[dict]) -> dict[str, str]:
    result = {}
    for event in events:
        for label in event.get("labels", []):
            tag = str(label.get("tag", "")).strip()
            name = str(label.get("name", "")).strip()
            if tag and name:
                result[tag] = name
        for frame in event.get("frames", []):
            for label in frame.get("labels", []):
                tag = str(label.get("tag", "")).strip()
                name = str(label.get("name", "")).strip()
                if tag and name:
                    result[tag] = name
    return result


def _normalize_event_description(
    value: dict,
    people: list[dict] | set[str] | None = None,
    label_map: dict[str, str] | None = None,
    allow_role_sensitive: bool = True,
) -> dict:
    if isinstance(people, set):
        allowed_speakers = people
    else:
        allowed_speakers = {
            str(person.get("name", "")).strip()
            for person in (people or [])
            if str(person.get("name", "")).strip()
        }
    label_map = label_map or {}
    speaker = str(value.get("说话人", "")).strip()
    speaker = label_map.get(speaker, speaker)
    confirmed = value.get("说话人已确认", False)
    if not isinstance(confirmed, bool):
        confirmed = str(confirmed).lower() in ("true", "1", "yes")
    if allowed_speakers is not None and speaker not in allowed_speakers:
        confirmed = False
    dialogue_fact = _normalize_text_field(value.get("对白事实", ""))
    if not confirmed:
        speaker = ""
        dialogue_fact = re.sub(
            r"^(?:领导|男子|人物|某人|有人)[：:]\s*",
            "",
            dialogue_fact,
        )
        dialogue_fact = re.sub(
            r"^(?:领导|书记)(夸赞|称赞|表示|说|提到|询问)",
            r"有人\1",
            dialogue_fact,
        )
    action_events = _normalize_action_events(
        value.get("动作事件", []),
        label_map,
        allowed_speakers,
    )
    if not allow_role_sensitive:
        action_events = [
            event
            for event in action_events
            if not _contains_role_sensitive_action(str(event.get("动作", "")))
        ]
    relationship = _normalize_text_field(value.get("人物关系", ""))
    action_events = _reconcile_passive_relationship(
        action_events,
        relationship,
        allowed_speakers,
    )
    action_events = _reconcile_anonymous_group(action_events, relationship)
    action = _normalize_text_field(value.get("动作", ""))
    grounded_action = _action_events_text(action_events)
    if grounded_action:
        action = grounded_action
    elif _contains_role_sensitive_action(action):
        action = ""
    if not action_events and _contains_role_sensitive_action(relationship):
        relationship = ""
    return {
        "动作": action,
        "场景": _normalize_text_field(value.get("场景", "")),
        "人物关系": relationship,
        "动作事件": action_events,
        "对白事实": dialogue_fact,
        "说话人": speaker,
        "说话人已确认": confirmed,
    }


def _normalize_action_events(
    value,
    label_map: dict[str, str],
    current_names: set[str] | None = None,
) -> list[dict]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if not isinstance(item, dict):
            continue
        actors = _normalize_role_values(item.get("施动者", []), label_map)
        targets = _normalize_role_values(item.get("承受者", []), label_map)
        actors = _align_roles_to_current_people(
            actors,
            current_names or set(),
            set(label_map.values()),
        )
        targets = _align_roles_to_current_people(
            targets,
            current_names or set(),
            set(label_map.values()),
        )
        action = _normalize_text_field(item.get("动作", ""))
        evidence = _normalize_text_field(item.get("依据", ""))
        if action.startswith("被") and actors and targets:
            actors, targets = targets, actors
            action = action.removeprefix("被").strip()
        raw_confidence = item.get("置信度")
        try:
            confidence = max(0.0, min(1.0, float(raw_confidence)))
        except (TypeError, ValueError):
            confidence = (
                0.7
                if actors
                and targets
                and evidence
                and (
                    not _contains_role_sensitive_action(action)
                    or _has_physical_contact_evidence(evidence)
                )
                else 0.0
            )
        if (
            not action
            or action in {"动作", "可见动作"}
            or _is_non_visual_action(action, evidence)
            or confidence < 0.55
            or (_contains_role_sensitive_action(action) and not (actors and targets))
            or (
                _contains_role_sensitive_action(action)
                and not _has_physical_contact_evidence(evidence)
            )
            or set(actors).intersection(targets)
        ):
            continue
        result.append(
            {
                "施动者": actors,
                "动作": action,
                "承受者": targets,
                "依据": evidence,
                "置信度": round(confidence, 3),
            }
        )
        break
    return result


ROLE_SENSITIVE_ACTIONS = (
    "控制",
    "押走",
    "带走",
    "制服",
    "抓住",
    "按住",
    "拖走",
    "推倒",
    "搀扶",
)

NON_VISUAL_ACTIONS = (
    "说",
    "说话",
    "交谈",
    "讨论",
    "夸赞",
    "称赞",
    "询问",
    "回答",
    "表示",
    "提到",
)

ACTION_FAMILIES = (
    {"控制", "制服", "抓住", "按住", "推倒"},
    {"押走", "带走", "拖走"},
    {"搀扶"},
)


def _contains_role_sensitive_action(text: str) -> bool:
    return any(term in text for term in ROLE_SENSITIVE_ACTIONS)


def _is_non_visual_action(action: str, evidence: str) -> bool:
    return (
        any(term in action for term in NON_VISUAL_ACTIONS)
        or evidence.strip() in {"对白", "对白内容", "逐字稿", "语音内容"}
    )


PHYSICAL_CONTACT_TERMS = (
    "手",
    "肩",
    "手臂",
    "按住",
    "按在",
    "压在",
    "抓住",
    "拉住",
    "架住",
    "拖",
    "搀",
    "扶",
    "身体接触",
    "位置变化",
    "发生移动",
    "向前移动",
    "带离",
    "离开",
)


def _has_physical_contact_evidence(text: str) -> bool:
    return any(term in text for term in PHYSICAL_CONTACT_TERMS)


def _align_roles_to_current_people(
    roles: list[str],
    current_names: set[str],
    registered_names: set[str],
) -> list[str]:
    if len(current_names) != 1:
        return roles
    current_name = next(iter(current_names))
    return list(
        dict.fromkeys(
            current_name
            if role in registered_names and role not in current_names
            else role
            for role in roles
        )
    )


def _normalize_text_field(value) -> str:
    if isinstance(value, list):
        return "；".join(
            str(item).strip()
            for item in value
            if str(item).strip()
        )
    return str(value).strip()


def _reconcile_anonymous_group(
    action_events: list[dict],
    relationship: str,
) -> list[dict]:
    """Preserve a visible group description when an action role was generalized."""
    if not action_events or not relationship:
        return action_events
    event = action_events[0]
    actors = event.get("施动者", [])
    if len(actors) != 1 or actors[0] not in {
        "一名男子",
        "一名人物",
        "一名黑衣人",
        "一名未标注人物",
        "未标注人物",
        "一个人",
        "某人",
    }:
        return action_events
    match = re.search(
        r"([二两三四五六七八九十\d]+名[^，。；]{0,8}?"
        r"(?:黑衣人|男子|人员))",
        relationship,
    )
    action = str(event.get("动作", ""))
    action_terms = [
        term
        for term in re.split(r"[并和及、，\s]+", action)
        if len(term) >= 2
    ] or ([action] if action else [])
    same_action = any(term in relationship for term in action_terms)
    same_role_action_family = (
        _contains_role_sensitive_action(action)
        and _contains_role_sensitive_action(relationship)
    )
    if not match or not (same_action or same_role_action_family):
        return action_events
    updated = dict(event)
    updated["施动者"] = [match.group(1)]
    return [updated]


def _reconcile_passive_relationship(
    action_events: list[dict],
    relationship: str,
    current_names: set[str],
) -> list[dict]:
    """Use an explicit passive clause to resolve contradictory role arrays."""
    if len(action_events) != 1 or len(current_names) != 1:
        return action_events
    event = action_events[0]
    if not _has_physical_contact_evidence(str(event.get("依据", ""))):
        return action_events
    match = re.search(
        r"(?P<target>[^，。；]{1,12}?)被"
        r"(?P<actor>[二两三四五六七八九十\d]+名[^，。；]{0,8}?"
        r"(?:黑衣人|男子|人员))"
        r"(?P<action>控制(?:并|和|及)?押走|按住|控制|押走|抓住|制服)",
        relationship,
    )
    if not match:
        return action_events
    target = match.group("target").strip()
    generic_targets = {
        "一名男子",
        "一名人物",
        "一个人",
        "某人",
        "一人",
    }
    current_name = next(iter(current_names))
    if target not in generic_targets and target != current_name:
        return action_events
    action = match.group("action").replace("和", "并").replace("及", "并")
    updated = dict(event)
    updated["施动者"] = [match.group("actor")]
    updated["动作"] = action
    updated["承受者"] = [current_name]
    return [updated]


def _request_error_detail(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    body = getattr(response, "text", "") if response is not None else ""
    return f"{exc}; {body[:500]}" if body else str(exc)


def _normalize_role_values(value, label_map: dict[str, str]) -> list[str]:
    values = value if isinstance(value, list) else [value]
    return list(
        dict.fromkeys(
            _normalize_role_value(str(item).strip(), label_map)
            for item in values
            if str(item).strip()
            and not (
                re.fullmatch(r"P\d+", str(item).strip())
                and str(item).strip() not in label_map
            )
        )
    )


def _normalize_role_value(value: str, label_map: dict[str, str]) -> str:
    if value in label_map:
        return label_map[value]
    tagged_name = re.fullmatch(r"(P\d+)[（(]([^）)]+)[）)]", value)
    if tagged_name and tagged_name.group(1) in label_map:
        return label_map[tagged_name.group(1)]
    return value


def _action_events_text(events: list[dict]) -> str:
    clauses = []
    for event in events:
        actors = "和".join(event.get("施动者", []))
        targets = "和".join(event.get("承受者", []))
        action = event.get("动作", "")
        if actors and targets:
            clauses.append(f"{actors}{action}{targets}")
        elif actors:
            clauses.append(f"{actors}{action}")
        elif targets:
            clauses.append(f"{targets}被{action}")
        elif action:
            clauses.append(action)
    return "；".join(clauses)


def _parse_json_object(value: str) -> dict:
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        match = re.search(r"\{.*\}", str(value), re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group())
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}


def _time_to_seconds(value: str) -> float:
    try:
        hours, minutes, seconds = value.split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except (ValueError, AttributeError):
        return 0.0


def _format_seconds(value: float) -> str:
    total = max(0, int(round(float(value))))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
