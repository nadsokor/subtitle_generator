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
  - **DeepL**、**OpenAI**：需在页面或环境变量中配置 API Key
  - **OpenAI** 支持选择模型（含 gpt-5、gpt-4o-mini 等）及「自定义」输入模型名
  - **翻译风格与规则**：可选填写说明（仅 OpenAI 生效），如语气、专有名词保留等
- **ffmpeg 集成**：若未检测到系统 ffmpeg，可在页面「一键下载并安装」到应用目录，无需手动配置 PATH
- **任务进度**：网页端展示下载模型、转写、翻译的进度条与预计剩余时间

## 环境要求

- **Python 3.8+**
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
- **Apple Silicon（M1/M2/M3 等）**：用 `pip install -r requirements.txt` 安装的 PyTorch 通常已支持 MPS，无需额外步骤。
- 启动后若使用 GPU，转写会明显更快。

## 使用说明

1. **ffmpeg**：若页面顶部提示未检测到 ffmpeg，可点击「一键下载并安装 ffmpeg」等待完成（约 1～2 分钟）
2. 在页面中**选择或拖拽**视频/音频文件
3. 选择 **Whisper 模型**：体积越大精度越高、速度越慢（如 `base` 平衡速度与效果）
4. 选择 **字幕语言**：若已知语种可指定，否则选「自动检测」
5. （可选）**翻译**：
   - 在「翻译 API」中选 Google / DeepL / OpenAI，在「翻译成」中选目标语言
   - 选 OpenAI 时可在「API 配置」中填写 API Key、选择模型（含「自定义」输入模型名），以及「翻译风格与规则」
   - 选 DeepL 时填写 DeepL API Key；配置会保存在浏览器本地，下次自动带出
6. 点击 **「生成字幕」**，等待任务完成（页面会显示进度条与预计剩余时间）
7. 完成后会自动下载 `.srt` 文件（若启用了翻译，文件名为 `原名.语言码.srt`，如 `demo.zh.srt`）

### 翻译 API 说明

- **Google 翻译**：无需配置，直接可用（需网络）
- **DeepL**：在 [DeepL 开发者](https://www.deepl.com/pro-api) 获取 API Key，在页面「翻译」→ 选 DeepL 后出现的输入框中填写，或设置环境变量 `DEEPL_API_KEY` / `DEEPL_AUTH_KEY`
- **OpenAI**：在 [OpenAI API](https://platform.openai.com/api-keys) 创建 Key，在页面填写或设置 `OPENAI_API_KEY`；可选环境变量 `OPENAI_TRANSLATE_MODEL` 指定默认模型

## 项目结构

```
auto_subbed/
├── app.py              # FastAPI 后端：ffmpeg 检测/安装、上传、Whisper 转写、翻译、SRT 生成、异步任务与进度
├── requirements.txt
├── README.md
├── .gitignore
├── .ffmpeg/            # 可选，一键安装的 ffmpeg 所在目录（按平台分目录）
└── static/
    └── index.html      # 前端页面
```

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

- **后端**：FastAPI + OpenAI Whisper（`openai-whisper`）+ 翻译（`deep-translator`：Google / DeepL；OpenAI 使用 Chat Completions）
- **前端**：单页 HTML + 原生 JS，无构建步骤；配置与 API Key 使用 localStorage 持久化
- **ffmpeg**：优先使用系统 PATH；若无则从 BtbN（Windows/Linux）或 evermeet（macOS）下载并解压到 `.ffmpeg/<平台>`，仅当前进程 PATH 生效
- **任务流程**：提交后异步执行（下载模型 → 转写 → 可选翻译），前端轮询任务状态并展示进度与预计剩余时间
- **视频/音频**：由 ffmpeg 转码与切分，Whisper 按片段转写后合并；翻译时每条字幕按所选 API 翻译并保持原时间轴

## 许可证

本项目仅供学习与个人使用。Whisper 使用请遵守 OpenAI 相关条款。
