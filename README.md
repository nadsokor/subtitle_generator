# 视频自动字幕（本地 Web）

本地运行的可视化工具：上传视频/音频，选择 Whisper 模型与目标语言，自动生成 `.srt` 字幕文件；可选翻译成另一语言。**数据均在本地处理，不上传任何服务器。**

## 功能

- **本地 Web 端**：浏览器打开页面，上传文件即可使用；启动服务时可自动打开浏览器
- **输入**：主流视频/音频格式（MP4、MKV、AVI、MOV、WebM、MP3、WAV 等）
- **输出**：标准 `.srt` 字幕文件，可直接用于播放器或剪辑软件
- **Whisper 模型**：可选 tiny、base、small、medium、large 等各版本（含 turbo）；首次使用某模型时自动下载
- **语言**：支持指定字幕语言或自动检测
- **翻译**：可选将字幕翻译成另一种语言
  - **Google 翻译**：免密钥
  - **DeepL**、**OpenAI**、**Gemini**、**Moonshot**：需在页面或环境变量中配置 API Key
  - **OpenAI / Gemini / Moonshot** 支持选择模型及「自定义」输入模型名
  - **OpenAI Reasoning Effort**：可选配置 `minimal / low / medium / high`
  - **Gemini Thinking Level**：可选配置 `MINIMAL / LOW / MEDIUM / HIGH`
  - **Gemini Base URL（可选）**：支持接入兼容 Gemini 原生 `v1beta` 路径的中转站（如 `/v1beta/models/{model}:generateContent`）
  - **批量翻译**：Google / DeepL / OpenAI / Gemini / Moonshot 会自动分批请求，减少请求次数并提升效率
  - **每批条数可配置**：可在页面中自定义「每批条数」（留空用默认），按所选翻译 API 生效
  - **多文件并行翻译**：在 OpenAI / Gemini 下可并行处理多个文件，默认并行数为 3（可调整）
  - **翻译风格与规则**：可选填写说明（OpenAI / Gemini / Moonshot 生效），如语气、专有名词保留等
- **ffmpeg 集成**：若未检测到系统 ffmpeg，可在页面「一键下载并安装」到应用目录，无需手动配置 PATH
- **任务进度**：网页端展示下载模型、转写、翻译的进度条与预计剩余时间
- **并行任务明细面板**：多文件并行时实时展示每个文件的排队/运行/完成/失败状态与进度
- **API 请求日志**：自动写入 `logs/api_requests.log`，包含请求路径、状态码、耗时及关键提交参数（敏感字段脱敏）

## 环境要求

- **Python 3.8+**（使用 Gemini 翻译需 **Python 3.9+**）
- **ffmpeg**：用于处理音视频。可二选一：
  - 系统已安装并加入 PATH（推荐：`brew install ffmpeg` / `apt install ffmpeg` 等）
  - 或使用应用内「一键安装」：未检测到时在首页点击按钮，自动下载到项目 `.ffmpeg/` 并仅对当前进程生效
- 支持 **Windows / macOS / Linux**

## 安装与运行

```bash
# 进入项目目录
cd auto_subbed

# 创建虚拟环境（推荐）
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 安装依赖（首次较慢，含 PyTorch）
pip install -r requirements.txt

# 启动服务（会自动打开浏览器）
python app.py
```

浏览器访问：**http://127.0.0.1:8765**

指定端口或不用自动打开浏览器时：

```bash
uvicorn app:app --host 0.0.0.0 --port 8765
```

### GPU 加速

转写会自动使用 GPU（若可用）：**NVIDIA（CUDA）** 或 **Apple Silicon（MPS）**，否则使用 CPU。

- **NVIDIA 显卡（Windows / Linux）**：需安装带 CUDA 的 PyTorch（先装好 [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) 或使用 PyTorch 自带 CUDA 的包）。安装依赖后执行：
  ```bash
  pip uninstall torch -y
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  ```
  （`cu121` 为 CUDA 12.1；其他版本见 [PyTorch 官网](https://pytorch.org/get-started/locally/)）
- 若已装 CUDA 版 PyTorch 仍走 CPU，可强制指定设备：**Windows** 在启动前执行 `set AUTO_SUBBED_DEVICE=cuda`，**Linux/macOS** 执行 `export AUTO_SUBBED_DEVICE=cuda`，再运行 `python app.py`。
- **Apple Silicon（M1/M2/M3 等）**：用 `pip install -r requirements.txt` 安装的 PyTorch 通常已支持 MPS，无需额外步骤。
- 启动后若使用 GPU，转写会明显更快。

## 使用说明

1. **ffmpeg**：若页面顶部提示未检测到 ffmpeg，可点击「一键下载并安装 ffmpeg」等待完成（约 1～2 分钟）
2. 在页面中**选择或拖拽**视频/音频文件
3. 选择 **Whisper 模型**：体积越大精度越高、速度越慢（如 `base` 平衡速度与效果）
   - **识别引擎**：可选 Whisper 原版、faster-whisper（更快）或 Purfview XXL（exe）
4. 选择 **字幕语言**：若已知语种可指定，否则选「自动检测」
5. （可选）**翻译**：
   - 在「翻译 API」中选 Google / DeepL / OpenAI / Gemini / Moonshot，在「翻译成」中选目标语言
   - 选 OpenAI / Gemini / Moonshot 时可在「API 配置」中填写 API Key、选择模型（含「自定义」输入模型名），以及「翻译风格与规则」
   - 选 OpenAI 时可额外设置 Reasoning Effort（`minimal / low / medium / high`，留空为默认）
   - 选 Gemini 时可额外设置 Thinking Level（`MINIMAL / LOW / MEDIUM / HIGH`，留空为默认）
   - 选 DeepL 时填写 DeepL API Key；配置会保存在浏览器本地，下次自动带出
   - 可在「每批条数」中设置批量请求的每批字幕条数（如 8/12/20）；不填则使用默认值
   - 可在「并行文件数」中设置 OpenAI / Gemini 的多文件并发数（默认 3，建议 1~5）
6. 点击 **「生成字幕」**，等待任务完成（页面会显示进度条与预计剩余时间）
7. 完成后会自动下载 `.srt` 文件（若启用了翻译，文件名为 `原名.语言码.srt`，如 `demo.zh.srt`）
8. 若需排查问题，可查看 `logs/api_requests.log`（自动轮转清理历史）

### 翻译 API 说明

- **Google 翻译**：无需配置，直接可用（需网络）
- **DeepL**：在 [DeepL 开发者](https://www.deepl.com/pro-api) 获取 API Key，在页面「翻译」→ 选 DeepL 后出现的输入框中填写，或设置环境变量 `DEEPL_API_KEY` / `DEEPL_AUTH_KEY`
- **OpenAI**：在 [OpenAI API](https://platform.openai.com/api-keys) 创建 Key，在页面填写或设置 `OPENAI_API_KEY`；可选环境变量 `OPENAI_TRANSLATE_MODEL` 指定默认模型
- **Gemini**：在 [Gemini API](https://ai.google.dev/gemini-api/docs/api-key) 创建 Key，在页面填写或设置 `GEMINI_API_KEY`；可选环境变量 `GEMINI_TRANSLATE_MODEL` 指定默认模型
  - 若通过中转站访问 Gemini，可在页面填写 `Gemini Base URL`，或设置环境变量 `GEMINI_BASE_URL`（示例：`https://your-relay.example.com`）
  - 中转站需兼容 Gemini 原生接口路径 `/v1beta/models/{model}:generateContent`，并支持 `Authorization: Bearer <token>` 认证
- **Moonshot**：在 [Moonshot 平台](https://platform.moonshot.cn/docs/overview) 获取 API Key，在页面填写或设置 `MOONSHOT_API_KEY`；可选 `MOONSHOT_BASE_URL`（默认 `https://api.moonshot.cn/v1`）与 `MOONSHOT_TRANSLATE_MODEL`（默认 `kimi-k2-turbo-preview`，`kimi-k2.5` 会自动适配 `temperature=1`）

## 项目结构

```
auto_subbed/
├── app.py              # FastAPI 后端：ffmpeg 检测/安装、上传、Whisper 转写、翻译、SRT 生成、异步任务与进度
├── requirements.txt
├── README.md
├── .gitignore
├── .ffmpeg/            # 可选，一键安装的 ffmpeg 所在目录（按平台分目录）
├── logs/               # 运行日志目录（自动创建，按大小轮转并清理历史）
└── static/
    └── index.html      # 前端页面
```

## 日志与排查

- 默认日志文件：`logs/api_requests.log`
- 日志内容：`/api/*` 请求起止、状态码、耗时、客户端信息，以及 `/api/transcribe`、`/api/translate` 的关键业务参数
- 敏感字段（如 API Key）会自动脱敏，不会记录明文
- 轮转策略：按文件大小滚动，超出后自动生成历史文件并按保留份数删除旧日志
- 可通过环境变量调整：
  - `AUTO_SUBBED_API_LOG_MAX_BYTES`：单个日志文件最大字节数（默认 `10485760`，即 10MB）
  - `AUTO_SUBBED_API_LOG_BACKUP_COUNT`：保留历史日志份数（默认 `10`）

## Windows 常见问题

### `OSError: [WinError 1114] 动态链接库(DLL)初始化例程失败`（c10.dll 等）

多为 PyTorch 在 Windows 上依赖的运行时缺失或冲突，可依次尝试：

1. **安装 Visual C++ 运行库**  
   安装 [Microsoft Visual C++ Redistributable（最新 x64）](https://aka.ms/vs/17/release/vc_redist.x64.exe)，安装后重启终端再运行 `python app.py`。

2. **重装 PyTorch（CPU 版）**  
   先卸再装，使用官方 CPU 构建，避免与 CUDA 等 DLL 冲突：
   ```bash
   pip uninstall torch -y
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```
   若需要 **GPU 加速**且已安装 NVIDIA 驱动与 CUDA，可在解决上述问题后改用：`pip install torch --index-url https://download.pytorch.org/whl/cu121`（见上方「GPU 加速」）。

3. **用 Conda 安装 PyTorch（若上面仍报错）**  
   在 Conda 环境中安装 PyTorch 有时更稳定：
   ```bash
   conda install pytorch cpuonly -c pytorch
   ```

4. **确认 Python 版本**  
   建议使用 64 位 Python 3.10 或 3.11，在 [python.org](https://www.python.org/downloads/) 下载安装时勾选 “Add Python to PATH”。

## 技术说明

- **后端**：FastAPI + OpenAI Whisper（`openai-whisper`）+ 翻译（`deep-translator`：Google，`deepl`：DeepL，`openai`：OpenAI/Moonshot Chat Completions，`google-genai`：Gemini）
- **Purfview XXL（可选）**：将 `faster-whisper-xxl.exe` 放到 `app.py` 同级的 `.models/purfview-xxl/`，选择引擎为 `Purfview XXL（exe）` 即可调用
- **前端**：单页 HTML + 原生 JS，无构建步骤；配置与 API Key 使用 localStorage 持久化
- **ffmpeg**：优先使用系统 PATH；若无则从 BtbN（Windows/Linux）或 evermeet（macOS）下载并解压到 `.ffmpeg/<平台>`，仅当前进程 PATH 生效
- **任务流程**：提交后异步执行（下载模型 → 转写 → 可选翻译），前端轮询任务状态并展示进度与预计剩余时间
- **视频/音频**：由 ffmpeg 转码与切分，Whisper 按片段转写后合并；翻译时每条字幕按所选 API 翻译并保持原时间轴

## 许可证

本项目仅供学习与个人使用。Whisper 使用请遵守 OpenAI 相关条款。
