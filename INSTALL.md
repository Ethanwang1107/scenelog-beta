# macOS 独立安装与离线运行

`scenelog` 运行时不使用 Trae 内置二进制。首次安装和模型下载需要联网；完成本页预热后，处理本地素材时不需要网络，也不需要启动 Trae。

## 1. 安装系统依赖

在 macOS Terminal 中执行：

```zsh
xcode-select --install
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install ffmpeg python@3.12 whisper-cpp
brew install --cask ollama
```

启动 Ollama 应用一次，使本地服务运行：

```zsh
open -a Ollama
```

## 2. 安装 Python 依赖

在本项目目录执行：

```zsh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## 3. 下载离线模型

```zsh
mkdir -p "$HOME/.local/share/scenelog/models"
curl -L \
  -o "$HOME/.local/share/scenelog/models/ggml-large-v3-turbo.bin" \
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin"
ollama pull qwen2.5
ollama pull llava:7b
scenelog people setup
python - <<'PY'
import torch
torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
PY
```

`qwen2.5` 用于文字摘要。画面理解会优先使用已安装的
`qwen2.5vl:3b` / `qwen2.5vl:7b`，其次使用 `llava:7b` / `llava`。
若始终使用 `--no-vision`，可不下载视觉模型。
`scenelog people setup` 会下载 OpenCV 官方 YuNet 与 SFace 模型（约 37 MB）；
完成后人物识别可离线运行。若使用 `--no-people`，可跳过这一步。

## 4. 配置 whisper.cpp

将 Homebrew 的 `whisper-cli` 和本地模型路径写入 `~/.zshrc`：

```zsh
cat >> ~/.zshrc <<'EOF'
export SCENELOG_WHISPER_CPP_BIN="$(command -v whisper-cli)"
export SCENELOG_WHISPER_CPP_MODEL="$HOME/.local/share/scenelog/models/ggml-large-v3-turbo.bin"
EOF
source ~/.zshrc
```

也可以把这两个环境变量写入任意启动脚本。它们不会指向 Trae。

## 5. 独立终端验证

重新打开 Terminal 后，在项目目录执行：

```zsh
source .venv/bin/activate
python -m scenelog.cli process "$HOME/Desktop/测试素材" \
  --output "$HOME/Desktop/测试素材/_scenelog_standalone"
```

要显式排除 IDE 注入的 PATH，可使用：

```zsh
env -i \
  HOME="$HOME" \
  PATH="$PWD/.venv/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin" \
  SCENELOG_WHISPER_CPP_BIN="$SCENELOG_WHISPER_CPP_BIN" \
  SCENELOG_WHISPER_CPP_MODEL="$SCENELOG_WHISPER_CPP_MODEL" \
  python -m scenelog.cli process "$HOME/Desktop/测试素材" \
  --output "$HOME/Desktop/测试素材/_scenelog_standalone"
```

首次离线运行前请确认上述模型都已下载，且 Ollama 服务已经启动。`scenelog` 会在预检阶段报告缺少的 ffmpeg、whisper.cpp、Ollama 服务或模型，而不会回退到 Trae。

## 6. 启动本地网页

激活虚拟环境后执行：

```zsh
cd /path/to/scenelog-beta
source .venv/bin/activate
scenelog web
```

浏览器会自动打开 `http://127.0.0.1:8765`。网页只监听本机地址，不会把
视频、照片、逐字稿或场记表上传到网络。

在网页中按以下顺序操作：

1. 点击「选择文件夹」，载入素材目录。
2. 输入人物姓名并选择 1-3 张参考照片，点击「登记人物」。
3. 确认关键人物名单后，点击「开始处理」。
4. 处理完成后下载场记表，或在页面底部搜索人物、对白和画面内容。

终端窗口需要保持打开。要关闭网页服务，在运行命令的终端按 `Control-C`。
如果 `8765` 端口已被占用，可以改用：

```zsh
scenelog web --port 8766
```

## 7. v0.8.1 / 关键人物与动作角色

标准顺序是「先登记关键人物，再处理素材」。首次使用先下载人物模型：

```zsh
scenelog people setup
```

为关键人物准备 1-3 张照片，照片中必须只有这个人一张清晰人脸。建议包含
正脸、左右侧脸或不同光线。登记一名人物：

```zsh
scenelog people add "$HOME/Desktop/测试素材" "张三" \
  "$HOME/Desktop/人物照片/张三正脸.jpg" \
  "$HOME/Desktop/人物照片/张三侧脸.jpg"
```

继续登记其他关键人物：

```zsh
scenelog people add "$HOME/Desktop/测试素材" "李四" \
  "$HOME/Desktop/人物照片/李四.jpg"
```

同一姓名再次执行 `people add` 会追加参考照片，不会创建重复人物。查看当前
登记名单：

```zsh
scenelog people list "$HOME/Desktop/测试素材"
```

确认名单后再处理素材：

```zsh
scenelog process "$HOME/Desktop/测试素材"
```

系统只识别已登记姓名，其他出现在素材中的陌生人会被忽略，不会产生
「人物01」等未知人物档案。结果会写入 `场记表.xlsx` 的「人物」列和人物索引。
只要素材命中预登记人物，系统会按时间形成身份事件，用标注代表帧理解人物
动作，并将姓名自然写入「内容摘要」，例如“王书记带着两名男子走在乡间小路上”，
不会只写“王书记出现在素材中”。同期对白只有在画面证据能确认发言人时才署名；
无法确认时会写“交谈中有人……”。用户已经在 Excel 手工改写的摘要仍会优先保留。
v0.8.1 会为同一条素材中的登记人物分配稳定 `P1/P2` 标签，并把一个身份事件的
开始、中间、结束等核心连续帧交给视觉模型。系统先用无姓名原始帧确认抓住、
按住、拖动等可见身体接触，再用高亮身体区域核验登记人物是否为承受者，并单独
复核实际接触者数量，最后输出「施动者、动作、承受者」结构化事实。衣着颜色无法
稳定判断时会使用“两名正装人员”等保守称谓，不会为了贴合预期强行猜成黑衣人。
因此“两名正装人员控制并押走老刘”不会再退化成“一名男子被制服”。置信度不低于
0.7 的动作事实会被确定性补入摘要和关键词，并写入身份事件索引；可以组合搜索
`老刘 押走`。此版本仍不会仅凭普通逐字稿猜测是谁说话，说话人归属将在 v0.8.2
通过说话人分离和主动说话人检测单独解决。
人物识别不再依赖 VLM 的少量语义关键帧，而是默认每 1 秒独立扫描一帧；
同一人物连续出现时会合并为一次出现记录。可通过
`SCENELOG_FACE_SCAN_INTERVAL` 调整扫描间隔，例如设为 `0.5` 可提高短暂出镜
人物的召回率，但会增加处理时间。

如果素材已经处理过，后来才登记人物，可只重跑人物步骤：

```zsh
scenelog process "$HOME/Desktop/测试素材" --rerun people
```

从 v0.7 升级到 v0.8.1 时也执行同一命令。系统会复用转录、基础抽帧和画面描述，
只重建人物事件、身份摘要、身份索引与 Excel。

也可以只重跑一条素材：

```zsh
scenelog process "$HOME/Desktop/测试素材" \
  --file "卡1/A001.MOV" \
  --rerun people
```

修改姓名或删除登记人物：

```zsh
scenelog people name "$HOME/Desktop/测试素材" person_0001 "张老师"
scenelog people delete "$HOME/Desktop/测试素材" person_0002
```

按人物或人物加行为搜索：

```zsh
scenelog search "$HOME/Desktop/测试素材" "张三"
scenelog search "$HOME/Desktop/测试素材" "张三 劈柴"
```

可以直接在 `场记表.xlsx` 修改「内容摘要、关键词、人物、地点、可用点」，
后续续跑会保留人工修改。若未登记任何人物，处理流程会正常运行并跳过人物识别。

人物识别默认逐秒扫描，不依赖少量 VLM 关键帧。背影、严重遮挡、运动模糊、
过小人脸以及两次扫描之间极短出镜的人物仍可能漏检；增加不同角度参考照片或将
`SCENELOG_FACE_SCAN_INTERVAL` 调低到 `0.5` 可以提高覆盖率，但会增加处理时间。

## 8. 运行测试

安装测试依赖并运行：

```zsh
python -m pip install -e '.[test]'
pytest -q
```

逐字稿、抽帧缓存、人物档案、状态和索引均使用稳定 `material_id` 关联，
避免不同目录下同名素材互相覆盖。`scenelog search` 同时检索语音、
画面描述、人物出现和带姓名/动作/对白事实的身份事件。
