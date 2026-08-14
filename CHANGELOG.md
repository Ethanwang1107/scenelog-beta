# 更新日志

## 0.10.0 - 2026-08-13

首个 macOS 安装与发布工程版本。

### 新增

- 使用 pywebview 打开的原生 macOS 桌面窗口。
- 打包状态下可执行完整处理管线的内部 CLI 路由。
- 工作台首次运行环境诊断，分别显示基础与可选 AI 能力状态。
- PyInstaller App 构建、DMG 生成、Developer ID 签名与 Apple 公证脚本。
- Apple Silicon、macOS 13+ 的应用元数据和运行权限配置。
- 产品版本与 AI 管线版本分离，桌面升级不会触发已有素材重新处理。

### 验证

- 自动化测试：`99 passed`。
- App 内部 CLI：`scenelog, version 0.10.0`。
- App 内 SpeechBrain ECAPA-TDNN 声纹模型加载成功。
- App 静态资源、Mach-O 依赖和临时签名结构验证通过。
- 生成 Apple Silicon DMG 和 SHA-256 校验文件。

### 已知限制

- 当前候选包尚未使用 Developer ID 正式签名，也未提交 Apple 公证。
- 当前只构建 Apple Silicon 版本。
- 首次运行仍需按环境诊断安装 FFmpeg、Whisper 和 Ollama/模型。
- 尚未完成全新无开发环境 Mac 的外部安装测试。
