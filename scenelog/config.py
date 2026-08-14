"""全局配置常量"""

import os

# 版本号，用于 pipeline_version 和 stale 检测
PIPELINE_VERSION = "0.9.0"

# 输出目录名
SCENELOG_DIR = "_scenelog"
TRANSCRIPTS_DIR = "transcripts"
STATE_FILE = "state.jsonl"
INDEX_FILE = "transcripts_index.jsonl"
CACHE_DIR = ".cache"
EXCEL_FILE = "场记表.xlsx"
PEOPLE_FILE = "people.json"
PEOPLE_DIR = "people"
PEOPLE_FRAMES_DIR = "people_frames"
SPEAKER_TRANSCRIPTS_DIR = "speaker_transcripts"

# 支持的素材格式
SUPPORTED_EXTENSIONS = {".mov", ".mp4", ".m4a", ".wav"}

# 音频参数
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1

# VAD 参数
VAD_THRESHOLD = 0.5
VAD_MIN_SPEECH_DURATION = 0.3       # 最短语音段 (秒)
VAD_MIN_SILENCE_DURATION = 0.5      # 静音间隔 (秒)
VAD_SPEECH_RATIO_THRESHOLD = 0.02   # 语音占比阈值：低于此值视为「明确无语音」
VAD_CRITICAL_RATIO = 0.05           # 临界阈值：低于此值视为「临界样本」

# 转录参数
WHISPER_LANGUAGE = "zh"
WHISPER_CPP_ROOT = os.path.expanduser(
    "~/Library/Application Support/scenelog/whisper.cpp"
)
WHISPER_CPP_BINARY = os.environ.get(
    "SCENELOG_WHISPER_CPP_BIN",
    os.path.join(WHISPER_CPP_ROOT, "whisper-cli"),
)
WHISPER_CPP_MODEL = os.environ.get(
    "SCENELOG_WHISPER_CPP_MODEL",
    os.path.join(WHISPER_CPP_ROOT, "models", "ggml-large-v3-turbo.bin"),
)
WHISPER_CPP_THREADS = int(os.environ.get("SCENELOG_WHISPER_CPP_THREADS", "4"))
WHISPER_BATCH_SIZE = int(os.environ.get("SCENELOG_WHISPER_BATCH_SIZE", "8"))
WHISPER_TIMEOUT_PER_FILE = int(
    os.environ.get("SCENELOG_WHISPER_TIMEOUT_PER_FILE", "1800")
)

# 摘要参数
SUMMARY_CHUNK_DURATION = 300  # 长音频分块时长 (秒)
SUMMARY_MAX_CHUNKS = 20       # 最多分块数

# 磁盘安全阈值 (GB)
DISK_SAFE_MIN_GB = 5.0

# 缓存保留期 (天)
CACHE_MAX_AGE_DAYS = 30
CACHE_MAX_SIZE_GB = 50.0

# 视觉理解参数
VISION_ENABLED = True                    # 默认开启画面理解
VISION_DEDUP_SECONDS = 3.0               # 抽帧去重最小间隔（秒）
VISION_LONG_THRESHOLD = 60.0             # 长视频阈值（秒）
VISION_MAX_FRAMES_SHORT = 3              # 短视频最多抽帧数
VISION_MAX_FRAMES_LONG = 5               # 长视频最多抽帧数
VISION_VLM_BASE_URL = "http://127.0.0.1:11434"
VISION_VLM_MODEL = os.environ.get("SCENELOG_VLM_MODEL", "")  # 留空则自动选择
VISION_FRAMES_DIR = "frames"             # 抽帧缓存子目录名

# 人物识别参数
PEOPLE_ENABLED = True
PEOPLE_MODEL_DIR = os.path.expanduser(
    "~/Library/Application Support/scenelog/face_models"
)
PEOPLE_DETECTOR_MODEL = os.environ.get(
    "SCENELOG_FACE_DETECTOR_MODEL",
    os.path.join(PEOPLE_MODEL_DIR, "face_detection_yunet_2023mar.onnx"),
)
PEOPLE_RECOGNIZER_MODEL = os.environ.get(
    "SCENELOG_FACE_RECOGNIZER_MODEL",
    os.path.join(PEOPLE_MODEL_DIR, "face_recognition_sface_2021dec.onnx"),
)
PEOPLE_MATCH_THRESHOLD = float(
    os.environ.get("SCENELOG_FACE_MATCH_THRESHOLD", "0.45")
)
PEOPLE_MIN_FACE_SIZE = int(os.environ.get("SCENELOG_FACE_MIN_SIZE", "40"))
PEOPLE_DETECTION_SCORE = float(
    os.environ.get("SCENELOG_FACE_DETECTION_SCORE", "0.8")
)
PEOPLE_SCAN_INTERVAL_SECONDS = max(
    0.25,
    float(os.environ.get("SCENELOG_FACE_SCAN_INTERVAL", "1.0")),
)
PEOPLE_OCCURRENCE_GAP_SECONDS = max(
    PEOPLE_SCAN_INTERVAL_SECONDS,
    float(os.environ.get("SCENELOG_FACE_OCCURRENCE_GAP", "2.5")),
)

# 声纹识别参数
SPEAKER_MODEL_SOURCE = os.environ.get(
    "SCENELOG_SPEAKER_MODEL",
    "speechbrain/spkrec-ecapa-voxceleb",
)
SPEAKER_MODEL_DIR = os.path.expanduser(
    os.environ.get(
        "SCENELOG_SPEAKER_MODEL_DIR",
        "~/Library/Application Support/scenelog/speaker_models/ecapa-voxceleb",
    )
)
SPEAKER_MIN_REFERENCE_SECONDS = float(
    os.environ.get("SCENELOG_SPEAKER_MIN_REFERENCE_SECONDS", "3.0")
)
SPEAKER_MIN_SEGMENT_SECONDS = float(
    os.environ.get("SCENELOG_SPEAKER_MIN_SEGMENT_SECONDS", "1.5")
)
SPEAKER_MATCH_THRESHOLD = float(
    os.environ.get("SCENELOG_SPEAKER_MATCH_THRESHOLD", "0.45")
)
SPEAKER_MATCH_MARGIN = float(
    os.environ.get("SCENELOG_SPEAKER_MATCH_MARGIN", "0.10")
)
SPEAKER_SINGLE_MATCH_THRESHOLD = float(
    os.environ.get("SCENELOG_SPEAKER_SINGLE_MATCH_THRESHOLD", "0.55")
)

# 日志级别
LOG_LEVEL = os.environ.get("SCENELOG_LOG_LEVEL", "INFO")
