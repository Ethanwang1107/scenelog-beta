# Scenelog 测试版

Scenelog 是一个本地离线的纪录片素材智能场记工具。它会在你的 Mac 上处理视频文件夹，生成场记表、逐字稿、画面描述、人物信息和可搜索索引。

当前版本是公开 Beta。它适合愿意按安装说明配置本地 AI 环境的 Apple Silicon Mac 用户试用。视频、照片、逐字稿、声纹和场记表都保存在本机，不会上传到云端。

> 重要：当前 Beta 包尚未完成 Apple Developer ID 正式签名和 Apple 公证。macOS 第一次打开时会显示安全提示。正式面向更广泛用户前，仍建议完成签名、公证和全新 Mac 安装测试。

## 下载 Beta

在 GitHub Release 中下载：

- `Scenelog-0.10.0-arm64.dmg`
- `Scenelog-0.10.0-arm64.dmg.sha256`

当前只支持：

- Apple Silicon Mac：M1 / M2 / M3 / M4
- macOS 13 或更高版本

## 安装

1. 下载 `Scenelog-0.10.0-arm64.dmg`。
2. 双击打开 DMG。
3. 将 `Scenelog.app` 拖到 `Applications`。
4. 从 `Applications` 打开 Scenelog。

## 如果提示无法验证开发者

当前测试版没有 Apple Developer ID 正式签名和公证，第一次打开时 macOS 可能提示：

```text
“Scenelog”无法打开，因为无法验证开发者。
```

这是未公证 Beta 包的正常现象。可以这样打开：

1. 打开 `系统设置`。
2. 进入 `隐私与安全性`。
3. 找到 Scenelog 被拦截的提示。
4. 点击 `仍要打开`。
5. 再确认一次打开。

也可以右键点击 `Scenelog.app`，选择 `打开`，再确认打开。

## 首次运行需要准备的本地环境

Scenelog 的 AI 处理在本机完成。第一次使用前需要安装或下载：

- FFmpeg
- whisper.cpp 和 Whisper 模型
- Ollama
- `qwen2.5`
- 视觉模型，例如 `qwen2.5vl:3b` 或 `llava:7b`

打开 Scenelog 后，工作台顶部会显示环境诊断，告诉你哪些项目已经准备好、哪些还缺失。详细安装步骤见 [INSTALL.md](INSTALL.md)。

## 从源码运行

```zsh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python -m pytest -q
scenelog web
```

## 基本使用

1. 打开 Scenelog。
2. 点击 `选择文件夹`，选择一个包含视频素材的目录。
3. 可选：登记关键人物照片或声纹。
4. 点击 `开始处理`。
5. 处理完成后下载 `场记表.xlsx`，或在页面中搜索人物、动作、对白和画面内容。

## 版本信息

- App 版本：`0.10.0`
- AI 处理管线版本：`0.9.0`
- 当前状态：Apple Silicon 公开 Beta

## 隐私与安全

- Scenelog 默认只监听 `127.0.0.1`，不会对外开放网络服务。
- 素材、人物照片、声纹、逐字稿和场记表保存在用户本机。
- 使用 Ollama、Whisper、SpeechBrain、OpenCV 等本地模型和依赖。
- 请不要在 Issue 中上传未脱敏的视频、照片、逐字稿、声纹或私人素材。

安全问题请查看 [SECURITY.md](SECURITY.md)。

## 反馈

如果遇到打不开、缺依赖、处理失败、识别结果不准或搜索不好用，可以开 GitHub Issue。请尽量提供：

- Mac 型号和系统版本
- Scenelog 页面顶部环境诊断截图
- 运行日志里最后 20 行
- 出问题的操作步骤

