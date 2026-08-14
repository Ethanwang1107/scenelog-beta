# Scenelog macOS 发布手册

## 当前发布形态

- 产品版本：`0.10.0`
- 处理管线版本：`0.9.0`
- 首发架构：Apple Silicon (`arm64`)
- 最低系统：macOS 13
- 当前安装形式：未公证的 Apple Silicon Beta DMG
- 正式发布目标：Developer ID 签名并经过 Apple 公证的 DMG
- 数据边界：视频、照片、声纹、逐字稿和场记表仅保存在用户电脑

产品版本与处理管线版本分别维护。仅修改桌面壳、安装流程或发布工程时，
不要提升 `PIPELINE_VERSION`，避免已有素材被无意义地重新处理。

## 本地构建

```zsh
cd /path/to/scenelog-beta
source .venv/bin/activate
python -m pip install -e '.[build]'
./scripts/build_macos.sh
```

输出文件：

```text
dist/Scenelog.app
dist/Scenelog-0.10.0-arm64.dmg
```

没有 Developer ID 时会生成临时签名 App。可以作为公开 Beta 附带明确风险说明发布，
但不要把未经过 Apple 公证的版本包装成正式稳定版。

## 发布前验证

```zsh
python -m pytest -q
dist/Scenelog.app/Contents/MacOS/Scenelog --scenelog-cli --version
dist/Scenelog.app/Contents/MacOS/Scenelog \
  --scenelog-cli people voice-setup
codesign --verify --deep --strict --verbose=2 dist/Scenelog.app
```

还必须在一台没有 Python、Homebrew 和 Trae 的 Apple Silicon Mac 上验证：

1. 从 DMG 拖入 Applications。
2. Beta 包按 README 说明通过 Gatekeeper；正式包不应被 Gatekeeper 拦截。
3. 工作台能显示环境诊断。
4. 选择素材目录、登记人物并完成一条素材处理。
5. 退出应用后本地服务和处理子进程停止。
6. 升级 App 后已有 `_scenelog` 数据和模型仍然保留。

## Developer ID 签名

需要 Apple Developer Program 账号和 `Developer ID Application` 证书。

查看本机可用身份：

```zsh
security find-identity -v -p codesigning
```

使用正式证书构建：

```zsh
export APPLE_SIGN_IDENTITY='Developer ID Application: 公司或姓名 (TEAMID)'
./scripts/build_macos.sh
```

## Apple 公证

首次配置钥匙串凭据：

```zsh
xcrun notarytool store-credentials scenelog-notary \
  --apple-id '你的 Apple ID' \
  --team-id 'TEAMID' \
  --password 'App 专用密码'
```

提交、公证、装订并验证：

```zsh
export APPLE_NOTARY_PROFILE=scenelog-notary
./scripts/notarize_macos.sh dist/Scenelog-0.10.0-arm64.dmg
```

公证成功后重新生成校验文件：

```zsh
shasum -a 256 dist/Scenelog-0.10.0-arm64.dmg \
  > dist/Scenelog-0.10.0-arm64.dmg.sha256
```

## 网站发布

推荐第一阶段使用 GitHub Releases 存储安装包，Vercel 或 Cloudflare Pages 托管官网。
Release 至少上传：

- `Scenelog-0.10.0-arm64.dmg`
- `Scenelog-0.10.0-arm64.dmg.sha256`
- 本版本更新说明
- 系统要求和已知限制

网站下载页必须明确：

- 仅支持 Apple Silicon 和 macOS 13 及以上。
- App 本体约数百 MB，首次运行还需要下载本地 AI 模型。
- 素材不上传云端。
- 当前仍需 Ollama、FFmpeg 和 Whisper 运行环境；工作台会显示缺失项。
- 提供隐私政策、用户协议、第三方开源许可和反馈渠道。

## 后续发布门槛

公开测试前还需要完成：

1. Developer ID 签名和 Apple 公证。
2. 全新 Mac 的非开发者安装测试。
3. 首次运行的一键依赖与模型安装。
4. 自定义 App 图标和 DMG 视觉。
5. 30 条真实素材量化验收。
6. 官网、隐私政策、用户协议和开源许可清单。
