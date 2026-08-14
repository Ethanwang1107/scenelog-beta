# GitHub 上传清单

## 仓库里应该包含

- `scenelog/`：Scenelog 源码
- `tests/`：自动化测试
- `packaging/`：macOS App 打包配置
- `scripts/`：构建、签名和公证脚本
- `README.md`：公开 Beta 用户说明
- `INSTALL.md`：详细安装依赖说明
- `RELEASE.md`：发布流程说明
- `CHANGELOG.md`：版本日志
- `SECURITY.md`：安全披露和本地数据边界说明
- `CONTRIBUTING.md`：反馈和贡献说明
- `pyproject.toml`：Python 包配置
- `.gitignore`：避免上传本地缓存、素材和构建产物

## Release 附件应该包含

这些文件放在本地 `release-assets/`，上传 GitHub Release 时作为附件上传：

- `Scenelog-0.10.0-arm64.dmg`
- `Scenelog-0.10.0-arm64.dmg.sha256`

不要把 DMG commit 到 git 仓库历史里。
`release-assets/` 已写入 `.gitignore`，只作为本地附件暂存目录。

## 不应该上传

- `.venv/`
- `build/`
- `dist/`
- `_scenelog/`
- 原始视频素材
- 场记表输出
- 人物照片
- 声纹文件
- 模型缓存
- 任何个人路径、账号、密钥或私人素材

## 建议 GitHub Release 设置

- Tag：`v0.10.0`
- Title：`Scenelog v0.10.0 Apple Silicon Beta`
- 勾选：`Set as a pre-release`
- Release 正文：使用 `docs/GITHUB_RELEASE_v0.10.0.md`

