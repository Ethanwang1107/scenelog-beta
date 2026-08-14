"""Native desktop launcher for the local Scenelog web application."""

from __future__ import annotations

import socket
import sys
import threading
from contextlib import closing

from scenelog.web import create_server, stop_server


def _available_port(host: str = "127.0.0.1") -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--scenelog-cli":
        from scenelog.cli import main as cli_main

        cli_main(args=sys.argv[2:])
        return

    try:
        import webview
    except ImportError as exc:
        raise SystemExit(
            "桌面运行组件未安装，请执行: python -m pip install '.[desktop]'"
        ) from exc

    host = "127.0.0.1"
    server = create_server(host, _available_port(host))
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    url = f"http://{server.server_address[0]}:{server.server_address[1]}"

    try:
        webview.create_window(
            "Scenelog",
            url,
            width=1280,
            height=860,
            min_size=(920, 680),
            background_color="#f5f5f2",
            text_select=True,
        )
        webview.start()
    finally:
        stop_server(server)
        server_thread.join(timeout=3)


if __name__ == "__main__":
    main()
