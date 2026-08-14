import json
import threading
from http.client import HTTPConnection
from pathlib import Path

import pytest

from scenelog.web import (
    ASSET_DIR,
    ProcessTask,
    ScenelogWebHandler,
    ThreadingHTTPServer,
    _project_payload,
    _source_path,
)


def test_source_path_requires_existing_directory(tmp_path):
    assert _source_path(str(tmp_path)) == tmp_path.resolve()
    with pytest.raises(ValueError, match="请先选择"):
        _source_path("")
    with pytest.raises(ValueError, match="不存在"):
        _source_path(str(tmp_path / "missing"))


def test_project_payload_counts_media_and_people(tmp_path):
    (tmp_path / "clip.mov").write_bytes(b"video")
    (tmp_path / "ignore.txt").write_text("text", encoding="utf-8")

    payload = _project_payload(tmp_path)

    assert payload["source_dir"] == str(tmp_path)
    assert payload["media_count"] == 1
    assert payload["processed_count"] == 0
    assert payload["failed_count"] == 0
    assert payload["people"] == []
    assert payload["excel_ready"] is False


def test_process_task_builds_cli_command(monkeypatch, tmp_path):
    started = {}

    class FakeStdout:
        def __iter__(self):
            return iter(["第一行\n", "第二行\n"])

    class FakeProcess:
        stdout = FakeStdout()

        def __init__(self, command, **kwargs):
            started["command"] = command
            started["kwargs"] = kwargs

        def poll(self):
            return 0

        def wait(self):
            return 0

    monkeypatch.setattr("scenelog.web.subprocess.Popen", FakeProcess)
    task = ProcessTask()
    task.start(
        tmp_path,
        {
            "vision": False,
            "people": True,
            "retry_failed": True,
            "rerun": "people",
            "selected_file": "A001.MOV",
        },
    )

    for thread in threading.enumerate():
        if thread.name != "MainThread" and thread.daemon:
            thread.join(timeout=1)

    command = started["command"]
    assert command[1:4] == ["-m", "scenelog.cli", "process"]
    assert "--no-vision" in command
    assert "--no-people" not in command
    assert "--retry-failed" in command
    assert command[-4:] == ["--rerun", "people", "--file", "A001.MOV"]
    assert task.snapshot()["status"] == "completed"
    assert task.snapshot()["logs"][-2:] == ["第一行", "第二行"]


@pytest.fixture()
def web_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), ScenelogWebHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_web_serves_frontend_and_project_api(web_server, tmp_path):
    (tmp_path / "clip.mp4").write_bytes(b"video")
    host, port = web_server
    connection = HTTPConnection(host, port, timeout=5)

    connection.request("GET", "/")
    response = connection.getresponse()
    html = response.read().decode("utf-8")
    assert response.status == 200
    assert "scenelog 本地工作台" in html

    connection.request("GET", "/assets/styles.css")
    response = connection.getresponse()
    css = response.read().decode("utf-8")
    assert response.status == 200
    assert ".workspace" in css

    connection.request("GET", "/api/setup/status")
    response = connection.getresponse()
    setup = json.loads(response.read())
    assert response.status == 200
    assert setup["version"] == "0.10.0"
    assert setup["total"] == 8
    assert {item["key"] for item in setup["checks"]} >= {
        "ffmpeg",
        "whisper_model",
        "ollama",
        "vision_model",
    }

    from urllib.parse import urlencode

    connection.request("GET", f"/api/project?{urlencode({'source_dir': tmp_path})}")
    response = connection.getresponse()
    payload = json.loads(response.read())
    assert response.status == 200
    assert payload["media_count"] == 1
    connection.close()


def test_frontend_assets_exist():
    assert (ASSET_DIR / "index.html").is_file()
    assert (ASSET_DIR / "styles.css").is_file()
    assert (ASSET_DIR / "app.js").is_file()
    assert (ASSET_DIR / "scenelog-mark.svg").is_file()
    assert (ASSET_DIR / "favicon.svg").is_file()


def test_frozen_process_task_uses_desktop_cli_route(monkeypatch, tmp_path):
    started = {}

    class FakeProcess:
        stdout = []

        def __init__(self, command, **kwargs):
            started["command"] = command

        def poll(self):
            return 0

        def wait(self):
            return 0

    monkeypatch.setattr("scenelog.web.subprocess.Popen", FakeProcess)
    monkeypatch.setattr("scenelog.web.sys.frozen", True, raising=False)
    monkeypatch.setattr("scenelog.web.sys.executable", "/Applications/Scenelog")
    task = ProcessTask()
    task.start(tmp_path, {"vision": True, "people": True})

    for thread in threading.enumerate():
        if thread.name != "MainThread" and thread.daemon:
            thread.join(timeout=1)

    assert started["command"][:3] == [
        "/Applications/Scenelog",
        "--scenelog-cli",
        "process",
    ]
