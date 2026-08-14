"""Local web interface for scenelog."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from scenelog import __version__
from scenelog.config import EXCEL_FILE, SCENELOG_DIR

ASSET_DIR = Path(__file__).with_name("web_assets")
MAX_REQUEST_BYTES = 60 * 1024 * 1024
MAX_LOG_LINES = 2000
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VOICE_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".mov", ".mp4"}


def _output_dir(source_dir: Path) -> Path:
    return source_dir / SCENELOG_DIR


def _source_path(raw_path: str) -> Path:
    if not raw_path or not raw_path.strip():
        raise ValueError("请先选择素材目录")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"素材目录不存在: {path}")
    return path


def _profile_payload(source_dir: Path) -> list[dict]:
    from scenelog.people import PeopleStore

    store = PeopleStore(_output_dir(source_dir))
    profiles = []
    for profile in store.list_people():
        profiles.append(
            {
                "id": profile["id"],
                "name": profile["name"],
                "reference_count": profile["reference_count"],
                "sample_count": profile["sample_count"],
                "material_count": profile["material_count"],
                "voice_count": profile.get("voice_count", 0),
                "voice_duration": profile.get("voice_duration", 0.0),
                "thumbnail_url": (
                    "/api/people/image?"
                    f"source_dir={_quote(str(source_dir))}&"
                    f"path={_quote(profile['thumbnail'])}"
                    if profile.get("thumbnail")
                    else ""
                ),
            }
        )
    return profiles


def _quote(value: str) -> str:
    from urllib.parse import quote

    return quote(value, safe="")


def _project_payload(source_dir: Path) -> dict:
    from scenelog.scanner import scan_media
    from scenelog.state import ALL_STEPS, StateManager

    output_dir = _output_dir(source_dir)
    media_files = scan_media(source_dir, excluded_dirs=[output_dir])
    state = StateManager(output_dir)
    states = list(state._states.values())
    failed = sum(
        1
        for record in states
        if any(
            record.get(step) in ("failed_retryable", "failed_terminal")
            for step in ALL_STEPS
        )
    )
    completed = sum(
        1
        for record in states
        if record.get("excel") in ("success", "skipped")
    )
    return {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "media_count": len(media_files),
        "processed_count": completed,
        "failed_count": failed,
        "people": _profile_payload(source_dir),
        "excel_ready": (output_dir / EXCEL_FILE).is_file(),
        "index_ready": (output_dir / "transcripts_index.jsonl").is_file(),
    }


@dataclass
class ProcessTask:
    """One local processing process and its captured output."""

    lock: threading.RLock = field(default_factory=threading.RLock)
    process: subprocess.Popen | None = None
    task_id: str = ""
    status: str = "idle"
    source_dir: str = ""
    started_at: float | None = None
    ended_at: float | None = None
    return_code: int | None = None
    logs: list[str] = field(default_factory=list)

    def start(self, source_dir: Path, options: dict) -> dict:
        with self.lock:
            if self.process and self.process.poll() is None:
                raise ValueError("已有处理任务正在运行")

            if getattr(sys, "frozen", False):
                command = [
                    sys.executable,
                    "--scenelog-cli",
                    "process",
                    str(source_dir),
                ]
            else:
                command = [
                    sys.executable,
                    "-m",
                    "scenelog.cli",
                    "process",
                    str(source_dir),
                ]
            if not options.get("vision", True):
                command.append("--no-vision")
            if not options.get("people", True):
                command.append("--no-people")
            if options.get("retry_failed"):
                command.append("--retry-failed")
            if options.get("dry_run"):
                command.append("--dry-run")
            rerun = str(options.get("rerun") or "").strip()
            if rerun:
                command.extend(["--rerun", rerun])
            selected_file = str(options.get("selected_file") or "").strip()
            if selected_file:
                command.extend(["--file", selected_file])

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            self.process = subprocess.Popen(
                command,
                cwd=str(source_dir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self.task_id = uuid.uuid4().hex
            self.status = "running"
            self.source_dir = str(source_dir)
            self.started_at = time.time()
            self.ended_at = None
            self.return_code = None
            self.logs = [f"$ {' '.join(command)}"]
            threading.Thread(target=self._capture, daemon=True).start()
            return self.snapshot()

    def _capture(self):
        process = self.process
        if process is None:
            return
        assert process.stdout is not None
        for line in process.stdout:
            with self.lock:
                self.logs.append(line.rstrip())
                if len(self.logs) > MAX_LOG_LINES:
                    self.logs = self.logs[-MAX_LOG_LINES:]
        return_code = process.wait()
        with self.lock:
            self.return_code = return_code
            self.ended_at = time.time()
            if self.status == "stopping":
                self.status = "stopped"
            else:
                self.status = "completed" if return_code == 0 else "failed"

    def stop(self) -> dict:
        with self.lock:
            if not self.process or self.process.poll() is not None:
                raise ValueError("当前没有正在运行的任务")
            self.status = "stopping"
            self.logs.append("正在停止任务，已完成的步骤会保留。")
            self.process.terminate()
            return self.snapshot()

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "task_id": self.task_id,
                "status": self.status,
                "source_dir": self.source_dir,
                "started_at": self.started_at,
                "ended_at": self.ended_at,
                "return_code": self.return_code,
                "logs": list(self.logs),
            }


TASK = ProcessTask()


class ScenelogWebHandler(BaseHTTPRequestHandler):
    server_version = "scenelog-local"

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                return self._send_asset("index.html")
            if parsed.path.startswith("/assets/"):
                return self._send_asset(parsed.path.removeprefix("/assets/"))
            if parsed.path == "/api/project":
                source = _source_path(_one(query, "source_dir"))
                return self._json(_project_payload(source))
            if parsed.path == "/api/task":
                return self._json(TASK.snapshot())
            if parsed.path == "/api/setup/status":
                from scenelog.setup_status import installation_status

                return self._json(installation_status())
            if parsed.path == "/api/search":
                return self._search(query)
            if parsed.path == "/api/people/image":
                return self._person_image(query)
            if parsed.path == "/api/download/excel":
                return self._download_excel(query)
            self.send_error(HTTPStatus.NOT_FOUND)
        except (ValueError, OSError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._json({"error": f"本地服务错误: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            data = self._read_json()
            if parsed.path == "/api/folder-picker":
                return self._folder_picker()
            if parsed.path == "/api/people":
                return self._add_person(data)
            if parsed.path == "/api/people/voice":
                return self._add_person_voice(data)
            if parsed.path == "/api/people/delete":
                return self._delete_person(data)
            if parsed.path == "/api/process":
                source = _source_path(str(data.get("source_dir", "")))
                return self._json(TASK.start(source, data.get("options") or {}))
            if parsed.path == "/api/process/stop":
                return self._json(TASK.stop())
            self.send_error(HTTPStatus.NOT_FOUND)
        except (ValueError, OSError) as exc:
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._json({"error": f"本地服务错误: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length > MAX_REQUEST_BYTES:
            raise ValueError("上传内容过大，请减少照片数量或尺寸")
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("请求内容不是有效 JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("请求格式错误")
        return value

    def _folder_picker(self):
        script = (
            'POSIX path of (choose folder with prompt '
            '"选择包含视频素材的文件夹")'
        )
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            if "User canceled" in result.stderr:
                return self._json({"cancelled": True})
            raise ValueError(result.stderr.strip() or "无法打开文件夹选择器")
        path = str(Path(result.stdout.strip()).resolve())
        return self._json({"source_dir": path})

    def _add_person(self, data: dict):
        from scenelog.cli import _mark_people_stale
        from scenelog.people import FaceEngine, PeopleStore

        source = _source_path(str(data.get("source_dir", "")))
        name = str(data.get("name", "")).strip()
        photos = data.get("photos")
        if not isinstance(photos, list) or not photos:
            raise ValueError("请至少选择一张人物照片")
        if len(photos) > 8:
            raise ValueError("每次最多上传 8 张人物照片")

        with tempfile.TemporaryDirectory(prefix="scenelog_people_") as temp_dir:
            paths = []
            for index, photo in enumerate(photos, 1):
                if not isinstance(photo, dict):
                    raise ValueError("人物照片格式错误")
                original_name = Path(str(photo.get("name", ""))).name
                suffix = Path(original_name).suffix.lower()
                if suffix not in PHOTO_EXTENSIONS:
                    raise ValueError(f"不支持的照片格式: {original_name}")
                content = str(photo.get("data", ""))
                if "," in content:
                    content = content.split(",", 1)[1]
                try:
                    decoded = base64.b64decode(content, validate=True)
                except ValueError as exc:
                    raise ValueError(f"照片读取失败: {original_name}") from exc
                if len(decoded) > 15 * 1024 * 1024:
                    raise ValueError(f"单张照片不能超过 15 MB: {original_name}")
                target = Path(temp_dir) / f"photo_{index:02d}{suffix}"
                target.write_bytes(decoded)
                paths.append(target)

            store = PeopleStore(_output_dir(source))
            person_id = store.add_person(name, paths, FaceEngine())
            _mark_people_stale(store.output_dir)

        return self._json(
            {
                "person_id": person_id,
                "message": f"已登记 {name}",
                "people": _profile_payload(source),
            },
            HTTPStatus.CREATED,
        )

    def _delete_person(self, data: dict):
        from scenelog.cli import _mark_people_stale
        from scenelog.people import PeopleStore

        source = _source_path(str(data.get("source_dir", "")))
        person_id = str(data.get("person_id", ""))
        store = PeopleStore(_output_dir(source))
        store.delete(person_id)
        _mark_people_stale(store.output_dir)
        return self._json({"people": _profile_payload(source)})

    def _add_person_voice(self, data: dict):
        from scenelog.cli import _mark_speaker_stale
        from scenelog.people import PeopleStore
        from scenelog.speaker import SpeakerEngine

        source = _source_path(str(data.get("source_dir", "")))
        person_id = str(data.get("person_id", "")).strip()
        samples = data.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError("请至少选择一段人物声音")
        if len(samples) > 5:
            raise ValueError("每次最多上传 5 段声音样本")

        with tempfile.TemporaryDirectory(prefix="scenelog_voice_") as temp_dir:
            paths = []
            for index, sample in enumerate(samples, 1):
                if not isinstance(sample, dict):
                    raise ValueError("声音样本格式错误")
                original_name = Path(str(sample.get("name", ""))).name
                suffix = Path(original_name).suffix.lower()
                if suffix not in VOICE_EXTENSIONS:
                    raise ValueError(f"不支持的声音格式: {original_name}")
                content = str(sample.get("data", ""))
                if "," in content:
                    content = content.split(",", 1)[1]
                try:
                    decoded = base64.b64decode(content, validate=True)
                except ValueError as exc:
                    raise ValueError(f"声音读取失败: {original_name}") from exc
                if len(decoded) > 40 * 1024 * 1024:
                    raise ValueError(f"单个声音文件不能超过 40 MB: {original_name}")
                target = Path(temp_dir) / f"voice_{index:02d}{suffix}"
                target.write_bytes(decoded)
                paths.append(target)

            store = PeopleStore(_output_dir(source))
            store.add_voice_samples(person_id, paths, SpeakerEngine())
            _mark_speaker_stale(store.output_dir)

        profile = next(
            item
            for item in store.list_people()
            if item["id"] == person_id
        )
        return self._json(
            {
                "message": (
                    f"已为 {profile['name']} 建立 "
                    f"{profile['voice_duration']:.0f} 秒声纹"
                ),
                "people": _profile_payload(source),
            },
            HTTPStatus.CREATED,
        )

    def _search(self, query: dict):
        from scenelog.search import Searcher

        source = _source_path(_one(query, "source_dir"))
        text = _one(query, "q").strip()
        if not text:
            raise ValueError("请输入搜索内容")
        results = Searcher(source).search(text, limit=50)
        return self._json({"query": text, "results": results})

    def _person_image(self, query: dict):
        source = _source_path(_one(query, "source_dir"))
        relative = Path(_one(query, "path"))
        root = _output_dir(source).resolve()
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError("无效的人物图片路径") from exc
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        return self._send_file(target, cache=True)

    def _download_excel(self, query: dict):
        source = _source_path(_one(query, "source_dir"))
        target = _output_dir(source) / EXCEL_FILE
        if not target.is_file():
            raise ValueError("场记表尚未生成")
        return self._send_file(target, download_name=EXCEL_FILE)

    def _send_asset(self, name: str):
        target = (ASSET_DIR / name).resolve()
        try:
            target.relative_to(ASSET_DIR.resolve())
        except ValueError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._send_file(target, cache=False)

    def _send_file(
        self,
        path: Path,
        download_name: str | None = None,
        cache: bool = False,
    ):
        content = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "private, max-age=3600" if cache else "no-store")
        if download_name:
            encoded = _quote(download_name)
            self.send_header(
                "Content-Disposition",
                f"attachment; filename*=UTF-8''{encoded}",
            )
        self.end_headers()
        self.wfile.write(content)

    def _json(self, data: dict, status: HTTPStatus = HTTPStatus.OK):
        content = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format_string, *args):
        return


def _one(query: dict, name: str) -> str:
    values = query.get(name, [])
    return values[0] if values else ""


def create_server(
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    """Create the local HTTP server for browser and desktop launchers."""
    return ThreadingHTTPServer((host, port), ScenelogWebHandler)


def stop_server(server: ThreadingHTTPServer) -> None:
    """Stop the local server and any active processing child."""
    if TASK.process and TASK.process.poll() is None:
        TASK.process.terminate()
    server.shutdown()
    server.server_close()


def run_server(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True):
    """Run the local-only web application."""
    server = create_server(host, port)
    url = f"http://{host}:{port}"
    print(f"scenelog v{__version__} 本地界面已启动: {url}")
    print("按 Control-C 停止服务")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在关闭本地界面")
    finally:
        if TASK.process and TASK.process.poll() is None:
            TASK.process.terminate()
        server.server_close()
