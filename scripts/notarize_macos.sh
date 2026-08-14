#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "用法: $0 dist/Scenelog-版本-架构.dmg" >&2
  exit 1
fi

DMG_PATH="$1"
PROFILE="${APPLE_NOTARY_PROFILE:-scenelog-notary}"

if [[ ! -f "$DMG_PATH" ]]; then
  echo "错误: 找不到 DMG: $DMG_PATH" >&2
  exit 1
fi

xcrun notarytool submit "$DMG_PATH" \
  --keychain-profile "$PROFILE" \
  --wait
xcrun stapler staple "$DMG_PATH"
xcrun stapler validate "$DMG_PATH"
spctl --assess --type open --context context:primary-signature -v "$DMG_PATH"

echo "Apple 公证与装订完成: $DMG_PATH"
