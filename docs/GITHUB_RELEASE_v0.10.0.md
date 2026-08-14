# Scenelog v0.10.0 Apple Silicon Beta

这是 Scenelog 的 macOS Apple Silicon 公开 Beta，适合愿意按安装说明配置本地 AI 环境的用户试用。

> 当前 Beta 包尚未完成 Apple Developer ID 正式签名和 Apple 公证。macOS 第一次打开时可能显示“无法验证开发者”。如果你不熟悉 macOS 安全设置，建议先从源码运行或等待后续签名公证版本。

## 下载

请下载：

- `Scenelog-0.10.0-arm64.dmg`
- `Scenelog-0.10.0-arm64.dmg.sha256`

SHA-256：

```text
b6b434476e1538479f04591b0950aad909b93150ef9e44dae7cce6c95c4a27c2
```

## 系统要求

- Apple Silicon Mac：M1 / M2 / M3 / M4
- macOS 13 或更高版本
- 本地可用磁盘空间建议 10 GB 以上

## 重要说明

当前 Beta 没有 Apple Developer ID 正式签名和 Apple 公证。第一次打开时 macOS 可能提示无法验证开发者。

如果被拦截，请打开：

```text
系统设置 > 隐私与安全性 > 仍要打开 Scenelog
```

也可以右键 `Scenelog.app`，选择 `打开`。

## 首次运行依赖

Scenelog 是本地离线工具，素材不会上传云端。首次处理素材前，需要准备：

- FFmpeg
- whisper.cpp 和 Whisper 模型
- Ollama
- `qwen2.5`
- 视觉模型，例如 `qwen2.5vl:3b` 或 `llava:7b`

打开 App 后，页面顶部会显示环境诊断，缺什么会直接提示。

## 本版新增

- macOS 桌面 App 壳，不再要求用户从终端启动网页。
- 首次运行环境诊断。
- App 内部可执行完整 Scenelog 处理管线。
- DMG 安装包和 SHA-256 校验文件。
- 发布脚本、签名脚本和公证脚本。

## 隐私说明

- 素材、照片、声纹、逐字稿和场记表保存在用户本机。
- 本地工作台默认只监听 `127.0.0.1`。
- 请不要在 GitHub Issue 中上传未脱敏的私人素材。

## 已验证

- 自动化测试：`99 passed`
- App 内部 CLI：`scenelog, version 0.10.0`
- App 内 SpeechBrain ECAPA-TDNN 声纹模型加载成功
- DMG 挂载验证通过

## 已知限制

- 当前只支持 Apple Silicon。
- 当前包未做 Apple Developer ID 签名和公证。
- 首次运行仍需要按安装说明配置本地 AI 依赖。
- 尚未完成全新无开发环境 Mac 的外部安装测试。

