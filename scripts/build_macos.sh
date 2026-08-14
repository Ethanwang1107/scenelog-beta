#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
VERSION="${SCENELOG_RELEASE_VERSION:-0.10.0}"
ARCH="$(uname -m)"
APP_PATH="$ROOT_DIR/dist/Scenelog.app"
DMG_PATH="$ROOT_DIR/dist/Scenelog-${VERSION}-${ARCH}.dmg"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "错误: macOS App 只能在 macOS 上构建。" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "错误: 找不到 Python: $PYTHON_BIN" >&2
  exit 1
fi

if ! "$PYTHON_BIN" -c "import PyInstaller, webview" >/dev/null 2>&1; then
  echo "缺少构建依赖，请先执行:" >&2
  echo "  $PYTHON_BIN -m pip install -e '.[build]'" >&2
  exit 1
fi

cd "$ROOT_DIR"
rm -rf build/Scenelog "$APP_PATH" "$DMG_PATH"
"$PYTHON_BIN" -m PyInstaller \
  --noconfirm \
  --clean \
  packaging/macos/Scenelog.spec

if [[ -n "${APPLE_SIGN_IDENTITY:-}" ]]; then
  codesign \
    --force \
    --deep \
    --options runtime \
    --timestamp \
    --entitlements packaging/macos/entitlements.plist \
    --sign "$APPLE_SIGN_IDENTITY" \
    "$APP_PATH"
  codesign --verify --deep --strict --verbose=2 "$APP_PATH"
else
  echo "提示: 未设置 APPLE_SIGN_IDENTITY，生成的是未公证 Beta App。"
fi

if command -v create-dmg >/dev/null 2>&1; then
  create-dmg \
    --volname "Scenelog" \
    --window-pos 200 120 \
    --window-size 720 430 \
    --icon-size 112 \
    --icon "Scenelog.app" 180 210 \
    --app-drop-link 540 210 \
    "$DMG_PATH" \
    "$APP_PATH"
else
  DMG_STAGE="$ROOT_DIR/build/dmg-stage"
  rm -rf "$DMG_STAGE"
  mkdir -p "$DMG_STAGE"
  cp -R "$APP_PATH" "$DMG_STAGE/"
  ln -s /Applications "$DMG_STAGE/Applications"
  hdiutil create \
    -volname "Scenelog" \
    -srcfolder "$DMG_STAGE" \
    -ov \
    -format UDZO \
    "$DMG_PATH"
fi

echo
echo "构建完成:"
echo "  App: $APP_PATH"
if [[ -f "$DMG_PATH" ]]; then
  echo "  DMG: $DMG_PATH"
fi
