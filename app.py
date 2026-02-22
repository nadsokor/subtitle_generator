"""
视频自动生成多语字幕 - 本地 Web 服务
支持：上传视频 → 选择 Whisper 模型与语言 → 生成 .srt，可选翻译成另一语言
"""
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import uuid
import zipfile
import json
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Any, Optional, List, Tuple
from urllib.parse import quote

import torch
import whisper
import deepl
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
from openai import OpenAI
from openai import RateLimitError as OpenAIRateLimitError
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

try:
    from google import genai
except Exception:
    genai = None

# 允许的视频/音频格式（ffmpeg 可处理）
ALLOWED_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v",
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma",
}

# Whisper 可用模型（按体积与精度）
WHISPER_MODELS = [
    "tiny", "tiny.en",
    "base", "base.en",
    "small", "small.en",
    "medium", "medium.en",
    "large-v1", "large-v2", "large-v3", "large",
    "large-v3-turbo", "turbo",
]

TRANSCRIBE_ENGINES = {
    "whisper",
    "faster-whisper",
    "purfview-xxl",
}

FASTER_WHISPER_MODEL_PREFIX = "Systran/faster-whisper-"

PURFVIEW_XXL_EXE_NAME = "faster-whisper-xxl.exe"

# 常用语言：code -> 显示名（与 Whisper LANGUAGES 一致）
LANGUAGES = [
    ("auto", "自动检测"),
    ("zh", "中文"),
    ("en", "English"),
    ("ja", "日本語"),
    ("ko", "한국어"),
    ("fr", "Français"),
    ("de", "Deutsch"),
    ("es", "Español"),
    ("ru", "Русский"),
    ("pt", "Português"),
    ("it", "Italiano"),
    ("nl", "Nederlands"),
    ("pl", "Polski"),
    ("tr", "Türkçe"),
    ("vi", "Tiếng Việt"),
    ("th", "ไทย"),
    ("id", "Indonesia"),
    ("ar", "العربية"),
    ("hi", "हिन्दी"),
]

# 翻译 API 选项：id -> 显示名
TRANSLATION_APIS = [
    ("none", "不翻译"),
    ("google", "Google 翻译"),
    ("deepl", "DeepL"),
    ("openai", "OpenAI"),
    ("gemini", "Gemini"),
    ("moonshot", "Moonshot"),
]

# 部分语言在翻译 API 中的目标代码（与 Whisper 不一致时）
TRANSLATOR_TARGET_MAP = {
    "zh": "zh-CN",  # Google 使用 zh-CN
}

# OpenAI/Gemini 翻译时目标语言英文名（用于 prompt）
LLM_TARGET_LANG_NAMES = {
    "zh": "Simplified Chinese", "en": "English", "ja": "Japanese", "ko": "Korean",
    "fr": "French", "de": "German", "es": "Spanish", "ru": "Russian", "pt": "Portuguese",
    "it": "Italian", "nl": "Dutch", "pl": "Polish", "tr": "Turkish", "vi": "Vietnamese",
    "th": "Thai", "id": "Indonesian", "ar": "Arabic", "hi": "Hindi",
}

# 前端可选的 OpenAI 翻译模型
OPENAI_TRANSLATE_MODELS = [
    ("gpt-5", "gpt-5"),
    ("gpt-4o-mini", "gpt-4o-mini（推荐）"),
    ("gpt-4o", "gpt-4o"),
    ("gpt-4-turbo", "gpt-4-turbo"),
    ("gpt-4", "gpt-4"),
    ("gpt-3.5-turbo", "gpt-3.5-turbo"),
]

# 前端可选的 Gemini 翻译模型
GEMINI_TRANSLATE_MODELS = [
    ("gemini-2.5-flash", "gemini-2.5-flash（推荐）"),
    ("gemini-2.5-pro", "gemini-2.5-pro"),
    ("gemini-2.0-flash", "gemini-2.0-flash"),
]

# 前端可选的 Moonshot 翻译模型
MOONSHOT_TRANSLATE_MODELS = [
    ("moonshot-v1-8k", "moonshot-v1-8k"),
    ("moonshot-v1-32k", "moonshot-v1-32k"),
    ("moonshot-v1-128k", "moonshot-v1-128k"),
]

app = FastAPI(title="视频自动字幕", description="本地 Whisper 多语字幕生成")

# 静态文件（前端）
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ffmpeg 应用内安装目录（未检测到系统 ffmpeg 时可下载到此）
APP_DIR = Path(__file__).resolve().parent
FFMPEG_APP_DIR = APP_DIR / ".ffmpeg"
_FFMPEG_BIN_DIR: Optional[Path] = None  # 当前使用的 ffmpeg 目录（应用内或已加入 PATH 的由调用方判断）


def _platform_tag() -> str:
    """返回当前平台标签，用于下载对应构建。"""
    machine = platform.machine().lower()
    if sys.platform == "win32":
        return "win64"
    if sys.platform == "darwin":
        return "macos_arm64" if machine in ("arm64", "aarch64") else "macos64"
    if sys.platform == "linux":
        return "linux64" if machine in ("x86_64", "amd64") else f"linux_{machine}"
    return "unknown"


def get_ffmpeg_bin_dir() -> Optional[Path]:
    """返回应用内已安装的 ffmpeg 所在目录；若未安装则返回 None。"""
    tag = _platform_tag()
    base = FFMPEG_APP_DIR / tag
    if sys.platform == "win32":
        # BtbN: 解压后为 ffmpeg-master-latest-win64-gpl/bin/
        ffmpeg_exe = base / "bin" / "ffmpeg.exe"
    else:
        ffmpeg_exe = base / "ffmpeg"
    if ffmpeg_exe.exists():
        return ffmpeg_exe.parent
    return None


def _check_system_ffmpeg() -> bool:
    """检测系统 PATH 中是否有 ffmpeg 和 ffprobe。"""
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def check_ffmpeg_available() -> Tuple[bool, str, Optional[str]]:
    """
    检测 ffmpeg 是否可用。
    返回 (available, source, path_or_error)
    source: "system" | "bundled" | None
    """
    if _check_system_ffmpeg():
        return True, "system", None
    bin_dir = get_ffmpeg_bin_dir()
    if bin_dir:
        return True, "bundled", str(bin_dir)
    return False, "", "未检测到 ffmpeg。请安装到系统 PATH，或使用下方「一键安装」下载到应用目录。"


def _apply_ffmpeg_path() -> None:
    """将应用内的 ffmpeg 目录加入当前进程的 PATH（仅当存在应用内安装时）。"""
    bin_dir = get_ffmpeg_bin_dir()
    if bin_dir:
        path = str(bin_dir)
        env_path = os.environ.get("PATH", "")
        if path not in env_path.split(os.pathsep):
            os.environ["PATH"] = path + os.pathsep + env_path


def _download_file(url: str, dest: Path, progress_cb: Optional[Callable[[int, int], None]] = None) -> None:
    """下载 url 到 dest 文件。progress_cb(downloaded, total) 可选。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "auto_subbed/1.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", "0") or "0")
        downloaded = 0
        while True:
            chunk = resp.read(1024 * 512)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if progress_cb and total:
                progress_cb(downloaded, total)


def ensure_ffmpeg_installed() -> Tuple[bool, str]:
    """
    若系统无 ffmpeg 且应用内未安装，则下载并解压到应用目录，并更新 PATH。
    返回 (success, message)。
    """
    if _check_system_ffmpeg():
        return True, "系统已存在 ffmpeg，无需安装。"
    if get_ffmpeg_bin_dir():
        _apply_ffmpeg_path()
        return True, "已使用应用内 ffmpeg。"

    tag = _platform_tag()
    base = FFMPEG_APP_DIR / tag
    base.mkdir(parents=True, exist_ok=True)

    # 各平台下载地址
    base_url_btbn = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest"
    if sys.platform == "win32":
        url = f"{base_url_btbn}/ffmpeg-master-latest-win64-gpl.zip"
        archive = base / "ffmpeg.zip"
        try:
            _download_file(url, archive)
            with zipfile.ZipFile(archive, "r") as z:
                # 解压后通常为 ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe
                for name in z.namelist():
                    if "ffmpeg.exe" in name or "ffprobe.exe" in name:
                        z.extract(name, base)
            # 统一到 base/bin/
            extracted = next(base.iterdir()) if list(base.iterdir()) else None
            if extracted and extracted.is_dir() and (extracted / "bin").exists():
                bin_dir = base / "bin"
                bin_dir.mkdir(exist_ok=True)
                for exe in ("ffmpeg.exe", "ffprobe.exe"):
                    src = extracted / "bin" / exe
                    if src.exists():
                        shutil.copy2(src, bin_dir / exe)
                shutil.rmtree(extracted, ignore_errors=True)
            archive.unlink(missing_ok=True)
        except Exception as e:
            shutil.rmtree(base, ignore_errors=True)
            return False, f"下载或解压失败: {e}"
    elif sys.platform == "darwin":
        # evermeet 提供 ffmpeg 和 ffprobe 两个 zip，解压后得到单个可执行文件
        ffmpeg_zip = base / "ffmpeg.zip"
        ffprobe_zip = base / "ffprobe.zip"
        try:
            _download_file("https://evermeet.cx/ffmpeg/get/zip", ffmpeg_zip)
            _download_file("https://evermeet.cx/ffmpeg/get/ffprobe/zip", ffprobe_zip)
            for zpath, final_name in ((ffmpeg_zip, "ffmpeg"), (ffprobe_zip, "ffprobe")):
                with zipfile.ZipFile(zpath, "r") as z:
                    for name in z.namelist():
                        if name.endswith("/"):
                            continue
                        z.extract(name, base)
                        src = base / name
                        if src.is_file():
                            dst = base / final_name
                            if src != dst:
                                shutil.move(str(src), str(dst))
                        elif src.is_dir():
                            for f in src.rglob("*"):
                                if f.is_file():
                                    shutil.move(str(f), base / f.name)
                            shutil.rmtree(src, ignore_errors=True)
                zpath.unlink(missing_ok=True)
            for exe in ("ffmpeg", "ffprobe"):
                p = base / exe
                if p.exists():
                    os.chmod(p, 0o755)
                    try:
                        subprocess.run(["xattr", "-d", "com.apple.quarantine", str(p)], capture_output=True, check=False)
                    except Exception:
                        pass
        except Exception as e:
            shutil.rmtree(base, ignore_errors=True)
            return False, f"下载或解压失败: {e}"
    elif sys.platform == "linux":
        url = f"{base_url_btbn}/ffmpeg-master-latest-linux64-gpl.tar.xz"
        archive = base / "ffmpeg.tar.xz"
        try:
            _download_file(url, archive)
            with tarfile.open(archive, "r:xz") as t:
                t.extractall(base)
            # 解压后为 ffmpeg-master-latest-linux64-gpl/bin/ffmpeg
            extracted = next((d for d in base.iterdir() if d.is_dir()), None)
            if extracted and (extracted / "bin").exists():
                bin_dir = base / "bin"
                bin_dir.mkdir(exist_ok=True)
                for exe in ("ffmpeg", "ffprobe"):
                    src = extracted / "bin" / exe
                    if src.exists():
                        shutil.copy2(src, bin_dir / exe)
                        os.chmod(bin_dir / exe, 0o755)
                shutil.rmtree(extracted, ignore_errors=True)
            archive.unlink(missing_ok=True)
        except Exception as e:
            shutil.rmtree(base, ignore_errors=True)
            return False, f"下载或解压失败: {e}"
    else:
        return False, f"当前平台 {sys.platform} 暂不支持自动安装 ffmpeg，请手动安装并加入 PATH。"

    if not get_ffmpeg_bin_dir():
        return False, "安装后未找到 ffmpeg 可执行文件。"
    _apply_ffmpeg_path()
    return True, "ffmpeg 已安装到应用目录并已加入 PATH。"


# 启动时若已有应用内 ffmpeg，则加入 PATH
_apply_ffmpeg_path()

# 简单任务存储（内存）
JOB_STORE: Dict[str, Dict[str, Any]] = {}
JOB_LOCK = threading.Lock()
JOB_CANCEL: Dict[str, threading.Event] = {}


class JobCanceled(Exception):
    pass


def _init_job(job_id: str, filename: str, filename_original: Optional[str] = None) -> None:
    with JOB_LOCK:
        JOB_STORE[job_id] = {
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "message": "等待开始",
            "filename": filename,
            "filename_original": filename_original,
            "srt": None,
            "srt_original": None,
            "error": None,
            "eta_seconds": None,
        }
        JOB_CANCEL[job_id] = threading.Event()


def _update_job(job_id: str, **kwargs: Any) -> None:
    with JOB_LOCK:
        if job_id in JOB_STORE:
            JOB_STORE[job_id].update(kwargs)


def _get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with JOB_LOCK:
        return JOB_STORE.get(job_id)


def _get_cancel_event(job_id: str) -> Optional[threading.Event]:
    with JOB_LOCK:
        return JOB_CANCEL.get(job_id)


def get_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def format_timestamp_srt(seconds: float) -> str:
    """将秒数转为 SRT 时间轴格式 00:00:00,000"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _content_disposition(filename: str) -> str:
    """构造兼容中文文件名的 Content-Disposition。"""
    fallback = "".join(ch if 32 <= ord(ch) < 127 and ch not in {'"', "\\"} else "_" for ch in (filename or "subtitle.srt"))
    if not fallback:
        fallback = "subtitle.srt"
    encoded = quote(filename or "subtitle.srt", safe="")
    return f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{encoded}"


def _safe_filename_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", (value or "").strip())
    token = token.strip("._-")
    return token or "model"


def _translated_suffix(
    translate_to: str,
    translation_api: str,
    openai_model: str,
    gemini_model: str,
    moonshot_model: str,
) -> str:
    parts = [translate_to]
    api_name = (translation_api or "").strip().lower()
    if api_name and api_name not in ("none", "openai", "gemini", "moonshot"):
        parts.append(_safe_filename_token(api_name))
    if api_name == "openai":
        model_name = (openai_model or "").strip() or os.environ.get("OPENAI_TRANSLATE_MODEL", "gpt-4o-mini")
        parts.append(_safe_filename_token(model_name))
    if api_name == "gemini":
        model_name = (gemini_model or "").strip() or os.environ.get("GEMINI_TRANSLATE_MODEL", "gemini-2.5-flash")
        parts.append(_safe_filename_token(model_name))
    if api_name == "moonshot":
        model_name = (moonshot_model or "").strip() or os.environ.get("MOONSHOT_TRANSLATE_MODEL", "moonshot-v1-8k")
        parts.append(_safe_filename_token(model_name))
    return "." + ".".join(parts)


def segments_to_srt(segments: list) -> str:
    """Whisper 的 segments 转为 SRT 文本"""
    lines = []
    for i, seg in enumerate(segments, 1):
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        lines.append(str(i))
        lines.append(f"{format_timestamp_srt(start)} --> {format_timestamp_srt(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def srt_to_segments(srt_text: str) -> list:
    """解析 SRT 文本为 segments 列表，每项 { start, end, text }。"""
    segments = []
    # 按空行分块，兼容 \n 与 \r\n
    blocks = re.split(r"\n\s*\n", (srt_text or "").strip())
    for block in blocks:
        lines = [ln.strip() for ln in block.strip().split("\n") if ln.strip()]
        if len(lines) < 2:
            continue
        # 第一行：序号（可忽略）；第二行：00:00:00,000 --> 00:00:02,500
        time_line = lines[1]
        m = re.match(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})", time_line)
        if not m:
            continue
        start = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + int(m.group(4)) / 1000.0
        end = int(m.group(5)) * 3600 + int(m.group(6)) * 60 + int(m.group(7)) + int(m.group(8)) / 1000.0
        text = "\n".join(lines[2:]).strip()
        if text:
            segments.append({"start": start, "end": end, "text": text})
    return segments


def _format_eta(seconds: int) -> str:
    """将秒数格式化为「X 分 Y 秒」或「X 秒」"""
    if seconds < 60:
        return f"{seconds} 秒"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m} 分 {s} 秒" if s else f"{m} 分钟"
    h, m = divmod(m, 60)
    return f"{h} 小时 {m} 分" if h else f"{m} 分钟"


def _whisper_device() -> str:
    """
    选择 Whisper 运行设备。优先读环境变量 AUTO_SUBBED_DEVICE（cuda / mps / cpu），
    未设置时：CUDA → Apple MPS → CPU。Windows 上若已装 CUDA 版 PyTorch 仍走 CPU，
    可设置 AUTO_SUBBED_DEVICE=cuda 强制使用 GPU。
    """
    env_device = (os.environ.get("AUTO_SUBBED_DEVICE") or "").strip().lower()
    if env_device in ("cuda", "mps", "cpu"):
        return env_device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _faster_whisper_device() -> str:
    env_device = (os.environ.get("AUTO_SUBBED_DEVICE") or "").strip().lower()
    if env_device in ("cuda", "cpu"):
        return env_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _faster_whisper_compute_type(device: str) -> str:
    return "float16" if device == "cuda" else "float32"


def _faster_whisper_model_path(model_name: str) -> str:
    """将模型名映射为 faster-whisper 仓库名或本地路径。"""
    if os.path.exists(model_name):
        return model_name
    if model_name in WHISPER_MODELS:
        return f"{FASTER_WHISPER_MODEL_PREFIX}{model_name}"
    return model_name


def _purfview_xxl_exe_path() -> str:
    """返回 Purfview faster-whisper-xxl.exe 路径。"""
    env = (os.environ.get("PURFVIEW_XXL_EXE") or "").strip()
    if env and os.path.exists(env):
        return env
    return str(Path(__file__).resolve().parent / ".models" / "purfview-xxl" / PURFVIEW_XXL_EXE_NAME)


def _run_purfview_xxl_transcribe(
    input_path: str,
    model_name: str,
    language: str,
    work_dir: str,
    *,
    job_id: str,
    cancel_event: Optional[threading.Event] = None,
) -> str:
    """调用 Purfview faster-whisper-xxl.exe 转写，返回 SRT 文本。"""
    exe_path = _purfview_xxl_exe_path()
    if not os.path.exists(exe_path):
        raise RuntimeError(f"未找到 {PURFVIEW_XXL_EXE_NAME}，请放到 {Path(exe_path).parent}")
    if sys.platform != "win32":
        raise RuntimeError("Purfview XXL 仅支持 Windows")
    cmd = [exe_path, input_path, "-m", model_name, "-o", work_dir]
    if language and language != "auto":
        cmd += ["-l", language]
    proc = subprocess.Popen(
        cmd,
        cwd=str(Path(exe_path).parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        universal_newlines=True,
    )
    last_line = ""
    try:
        if proc.stdout:
            for raw_line in proc.stdout:
                if cancel_event and cancel_event.is_set():
                    proc.terminate()
                    raise JobCanceled("任务已取消")
                line = (raw_line or "").strip()
                if line:
                    line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line).strip()
                if not line:
                    continue
                last_line = line
                _update_job(
                    job_id,
                    stage="transcribing",
                    progress=None,
                    message=f"Purfview: {line}",
                    eta_seconds=None,
                )
        proc.wait()
    finally:
        if proc.stdout:
            proc.stdout.close()
    srt_files = sorted(Path(work_dir).glob("*.srt"), key=lambda p: p.stat().st_mtime)
    if not srt_files:
        raise RuntimeError(f"Purfview XXL 未生成 SRT 文件：{last_line or '未知错误'}")
    return srt_files[-1].read_text(encoding="utf-8", errors="replace")


def _get_whisper_cache_dir() -> str:
    default = os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(os.getenv("XDG_CACHE_HOME", default), "whisper")


def _ensure_model_downloaded(
    model_name: str,
    progress_cb: Callable[[int, Optional[int]], None],
) -> None:
    """progress_cb(percent, eta_seconds)"""
    if not hasattr(whisper, "_MODELS") or model_name not in whisper._MODELS:
        progress_cb(100, 0)
        return
    url = whisper._MODELS[model_name]
    download_root = _get_whisper_cache_dir()
    os.makedirs(download_root, exist_ok=True)
    download_target = os.path.join(download_root, os.path.basename(url))
    if os.path.exists(download_target):
        progress_cb(100, 0)
        return
    progress_cb(0, None)
    start = time.time()
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        total = int(source.info().get("Content-Length", "0") or "0")
        downloaded = 0
        while True:
            buffer = source.read(1024 * 512)
            if not buffer:
                break
            output.write(buffer)
            downloaded += len(buffer)
            if total > 0:
                pct = int(downloaded / total * 100)
                eta = None
                if downloaded > 0 and pct < 100:
                    elapsed = time.time() - start
                    eta = int(elapsed / downloaded * (total - downloaded))
                progress_cb(pct, eta)
    progress_cb(100, 0)


def _run_cmd(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _get_media_duration(path: str) -> Optional[float]:
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                path,
            ],
            stderr=subprocess.DEVNULL,
        )
        return float(out.decode().strip())
    except Exception:
        return None


def _split_audio(input_path: str, work_dir: str, segment_seconds: int = 30) -> List[str]:
    wav_path = os.path.join(work_dir, "audio.wav")
    _run_cmd(["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000", "-vn", wav_path])
    pattern = os.path.join(work_dir, "chunk_%04d.wav")
    _run_cmd(
        [
            "ffmpeg",
            "-y",
            "-i",
            wav_path,
            "-f",
            "segment",
            "-segment_time",
            str(segment_seconds),
            "-c",
            "copy",
            pattern,
        ]
    )
    chunks = sorted(str(p) for p in Path(work_dir).glob("chunk_*.wav"))
    return chunks


def _process_job(
    job_id: str,
    input_path: str,
    model_name: str,
    language: str,
    engine: str,
    translate_to: str,
    translation_api: str,
    openai_api_key: str,
    openai_base_url: str,
    deepl_api_key: str,
    openai_model: str,
    openai_reasoning_effort: str,
    gemini_api_key: str,
    gemini_model: str,
    gemini_thinking_level: str,
    moonshot_api_key: str,
    moonshot_base_url: str,
    moonshot_model: str,
    translation_rules: str,
    translation_batch_size: int,
    base_name: str = "",
) -> None:
    work_dir = None
    try:
        cancel_event = _get_cancel_event(job_id)
        _update_job(job_id, status="running", stage="downloading", progress=0, message="下载模型中 0%", eta_seconds=None)

        def download_progress(pct: int, eta_sec: Optional[int]) -> None:
            msg = f"下载模型中 {pct}%"
            if eta_sec is not None and eta_sec > 0:
                msg += f"（约剩余 {_format_eta(eta_sec)}）"
            _update_job(job_id, stage="downloading", progress=pct, message=msg, eta_seconds=eta_sec)

        if cancel_event and cancel_event.is_set():
            raise JobCanceled("任务已取消")
        if engine == "whisper":
            _ensure_model_downloaded(model_name, download_progress)
            _update_job(job_id, stage="transcribing", progress=0, message="转写中 0%", eta_seconds=None)
            _device = _whisper_device()
            model = whisper.load_model(model_name, device=_device)
            transcribe_options = {"word_timestamps": False}
            if language and language != "auto":
                transcribe_options["language"] = language
        elif engine == "faster-whisper":
            _update_job(job_id, stage="downloading", progress=0, message="加载模型中…", eta_seconds=None)
            fw_device = _faster_whisper_device()
            fw_compute = _faster_whisper_compute_type(fw_device)
            model_root = Path(__file__).resolve().parent / ".models" / "faster-whisper"
            os.makedirs(model_root, exist_ok=True)
            fw_name = _faster_whisper_model_path(model_name)
            model = WhisperModel(
                fw_name,
                device=fw_device,
                compute_type=fw_compute,
                download_root=str(model_root),
            )
            transcribe_options = {
                "language": None if (not language or language == "auto") else language,
                "task": "transcribe",
                "word_timestamps": False,
            }
            _update_job(job_id, stage="transcribing", progress=0, message="转写中 0%", eta_seconds=None)
        else:
            _update_job(job_id, stage="transcribing", progress=None, message="转写中（Purfview XXL 不支持进度）", eta_seconds=None)

        work_dir = tempfile.mkdtemp(prefix="auto_subbed_")
        if engine == "purfview-xxl":
            srt_original_content = _run_purfview_xxl_transcribe(
                input_path,
                model_name,
                language,
                work_dir,
                job_id=job_id,
                cancel_event=cancel_event,
            )
            all_segments = srt_to_segments(srt_original_content)
        else:
            chunks = _split_audio(input_path, work_dir)
            if not chunks:
                raise RuntimeError("无法切分音频，请检查输入文件或 ffmpeg")

            all_segments = []
            total_chunks = len(chunks)
            offset = 0.0
            transcribe_start = time.time()
            for idx, chunk in enumerate(chunks, 1):
                if cancel_event and cancel_event.is_set():
                    raise JobCanceled("任务已取消")
                if engine == "whisper":
                    result = model.transcribe(chunk, **transcribe_options)
                    for seg in result.get("segments") or []:
                        seg["start"] = (seg.get("start") or 0) + offset
                        seg["end"] = (seg.get("end") or 0) + offset
                        all_segments.append(seg)
                else:
                    segments, _info = model.transcribe(chunk, **transcribe_options)
                    for seg in segments:
                        all_segments.append({
                            "start": (seg.start or 0) + offset,
                            "end": (seg.end or 0) + offset,
                            "text": seg.text or "",
                        })

                duration = _get_media_duration(chunk) or 0
                if duration <= 0:
                    duration = 30.0
                offset += duration

                progress = int(idx / total_chunks * 100)
                elapsed = time.time() - transcribe_start
                eta_sec = None
                if idx < total_chunks and elapsed > 0:
                    eta_sec = int(elapsed / idx * (total_chunks - idx))
                msg = f"转写中 {progress}%"
                if eta_sec is not None and eta_sec > 0:
                    msg += f"（约剩余 {_format_eta(eta_sec)}）"
                _update_job(job_id, stage="transcribing", progress=progress, message=msg, eta_seconds=eta_sec)

        if cancel_event and cancel_event.is_set():
            raise JobCanceled("任务已取消")
        if engine != "purfview-xxl":
            srt_original_content = segments_to_srt(all_segments)
        if translation_api and translation_api != "none" and translate_to and translate_to != "none":
            name = (base_name or Path(input_path).stem).strip() or "subtitle"
            filename_original = f"{name}.srt"
            _update_job(job_id, srt_original=srt_original_content, filename_original=filename_original)
            _update_job(job_id, stage="translating", progress=0, message="翻译中 0%", eta_seconds=None)

            translate_start = time.time()

            def translate_progress(done: int, total: int) -> None:
                pct = int(done / total * 100) if total else 100
                eta_sec = None
                if done > 0 and total and done < total:
                    elapsed = time.time() - translate_start
                    eta_sec = int(elapsed / done * (total - done))
                msg = f"翻译中 {pct}%"
                if eta_sec is not None and eta_sec > 0:
                    msg += f"（约剩余 {_format_eta(eta_sec)}）"
                _update_job(job_id, stage="translating", progress=pct, message=msg, eta_seconds=eta_sec)

            all_segments = translate_segments(
                all_segments,
                target_lang=translate_to,
                api_name=translation_api,
                source_lang=None,
                openai_api_key=openai_api_key or None,
                openai_base_url=openai_base_url or None,
                deepl_api_key=deepl_api_key or None,
                openai_model=openai_model or None,
                openai_reasoning_effort=openai_reasoning_effort or None,
                gemini_api_key=gemini_api_key or None,
                gemini_model=gemini_model or None,
                gemini_thinking_level=gemini_thinking_level or None,
                moonshot_api_key=moonshot_api_key or None,
                moonshot_base_url=moonshot_base_url or None,
                moonshot_model=moonshot_model or None,
                translation_rules=translation_rules or None,
                translation_batch_size=translation_batch_size,
                progress_cb=translate_progress,
                status_cb=lambda msg: _update_job(job_id, stage="translating", message=msg, eta_seconds=None),
                cancel_event=cancel_event,
            )

        srt_content = segments_to_srt(all_segments)
        _update_job(job_id, status="done", stage="done", progress=100, message="完成", srt=srt_content, eta_seconds=0)
    except JobCanceled as e:
        _update_job(job_id, status="canceled", stage="canceled", progress=0, error=str(e), message="已取消", eta_seconds=None)
    except Exception as e:
        _update_job(job_id, status="error", stage="error", progress=0, error=str(e), message=f"失败: {e}", eta_seconds=None)
    finally:
        try:
            os.unlink(input_path)
        except Exception:
            pass
        if work_dir:
            try:
                shutil.rmtree(work_dir)
            except Exception:
                pass


def _process_translate_only_job(
    job_id: str,
    srt_path: str,
    translate_to: str,
    translation_api: str,
    openai_api_key: str,
    openai_base_url: str,
    deepl_api_key: str,
    openai_model: str,
    openai_reasoning_effort: str,
    gemini_api_key: str,
    gemini_model: str,
    gemini_thinking_level: str,
    moonshot_api_key: str,
    moonshot_base_url: str,
    moonshot_model: str,
    translation_rules: str,
    translation_batch_size: int,
    base_name: str,
) -> None:
    """仅翻译：读取 SRT 文件，翻译后写回 SRT，不涉及转写。"""
    try:
        cancel_event = _get_cancel_event(job_id)
        with open(srt_path, "r", encoding="utf-8", errors="replace") as f:
            srt_text = f.read()
        segments = srt_to_segments(srt_text)
        if not segments:
            raise RuntimeError("SRT 文件无有效字幕块，请检查格式")

        _update_job(job_id, status="running", stage="translating", progress=0, message="翻译中 0%", eta_seconds=None)
        translate_start = time.time()

        def translate_progress(done: int, total: int) -> None:
            pct = int(done / total * 100) if total else 100
            eta_sec = None
            if done > 0 and total and done < total:
                elapsed = time.time() - translate_start
                eta_sec = int(elapsed / done * (total - done))
            msg = f"翻译中 {pct}%"
            if eta_sec is not None and eta_sec > 0:
                msg += f"（约剩余 {_format_eta(eta_sec)}）"
            _update_job(job_id, stage="translating", progress=pct, message=msg, eta_seconds=eta_sec)

        translated = translate_segments(
            segments,
            target_lang=translate_to,
            api_name=translation_api,
            source_lang=None,
            openai_api_key=openai_api_key or None,
            openai_base_url=openai_base_url or None,
            deepl_api_key=deepl_api_key or None,
            openai_model=openai_model or None,
            openai_reasoning_effort=openai_reasoning_effort or None,
            gemini_api_key=gemini_api_key or None,
            gemini_model=gemini_model or None,
            gemini_thinking_level=gemini_thinking_level or None,
            moonshot_api_key=moonshot_api_key or None,
            moonshot_base_url=moonshot_base_url or None,
            moonshot_model=moonshot_model or None,
            translation_rules=translation_rules or None,
            translation_batch_size=translation_batch_size,
            progress_cb=translate_progress,
            status_cb=lambda msg: _update_job(job_id, stage="translating", message=msg, eta_seconds=None),
            cancel_event=cancel_event,
        )
        srt_content = segments_to_srt(translated)
        _update_job(job_id, status="done", stage="done", progress=100, message="完成", srt=srt_content, eta_seconds=0)
    except JobCanceled as e:
        _update_job(job_id, status="canceled", stage="canceled", progress=0, error=str(e), message="已取消", eta_seconds=None)
    except Exception as e:
        _update_job(job_id, status="error", stage="error", progress=0, error=str(e), message=f"失败: {e}", eta_seconds=None)
    finally:
        try:
            os.unlink(srt_path)
        except Exception:
            pass


def _translator_target_code(lang: str) -> str:
    """转为各翻译 API 通用的目标语言代码"""
    return TRANSLATOR_TARGET_MAP.get(lang, lang)


def _resolve_batch_size(value: Optional[int], default: int, *, min_value: int = 1, max_value: int = 200) -> int:
    """将用户输入的每批条数解析为安全整数，<=0 时使用默认值。"""
    try:
        n = int(value) if value is not None else 0
    except Exception:
        n = 0
    if n <= 0:
        n = default
    n = max(min_value, n)
    n = min(max_value, n)
    return n


def _normalize_gemini_thinking_level(value: Optional[str]) -> Optional[str]:
    """将 Gemini thinking level 规范化为 SDK 支持值。"""
    raw = (value or "").strip().upper()
    if not raw or raw in {"AUTO", "DEFAULT", "UNSPECIFIED", "THINKING_LEVEL_UNSPECIFIED"}:
        return None
    aliases = {
        "MIN": "MINIMAL",
        "NONE": "MINIMAL",
        "OFF": "MINIMAL",
        "DISABLED": "MINIMAL",
    }
    raw = aliases.get(raw, raw)
    allowed = {"MINIMAL", "LOW", "MEDIUM", "HIGH"}
    if raw in allowed:
        return raw
    raise HTTPException(
        status_code=400,
        detail="Gemini Thinking Level 不合法。可用值：MINIMAL / LOW / MEDIUM / HIGH（留空为默认）。",
    )


def _normalize_openai_reasoning_effort(value: Optional[str]) -> Optional[str]:
    """将 OpenAI reasoning effort 规范化。"""
    raw = (value or "").strip().lower()
    if not raw or raw in {"auto", "default", "none", "off"}:
        return None
    aliases = {
        "min": "minimal",
    }
    raw = aliases.get(raw, raw)
    allowed = {"minimal", "low", "medium", "high"}
    if raw in allowed:
        return raw
    raise HTTPException(
        status_code=400,
        detail="OpenAI reasoning effort 不合法。可用值：minimal / low / medium / high（留空为默认）。",
    )


def translate_segments(
    segments: list,
    target_lang: str,
    api_name: str,
    source_lang: str | None = None,
    *,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
    deepl_api_key: str | None = None,
    openai_model: str | None = None,
    openai_reasoning_effort: str | None = None,
    gemini_api_key: str | None = None,
    gemini_model: str | None = None,
    gemini_thinking_level: str | None = None,
    moonshot_api_key: str | None = None,
    moonshot_base_url: str | None = None,
    moonshot_model: str | None = None,
    translation_rules: str | None = None,
    translation_batch_size: int | None = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    status_cb: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> list:
    """
    将 segments 中每条 text 翻译成目标语言，时间轴不变。
    api_name: 'google' | 'deepl' | 'openai' | 'gemini' | 'moonshot'
    translation_rules: 可选，自定义风格与规则（OpenAI/Gemini/Moonshot 支持）。
    """
    if not segments or api_name == "none":
        return segments
    total = len(segments)

    # DeepL: 使用单客户端 + 批量请求，避免逐条调用过慢
    if api_name == "deepl":
        key = (deepl_api_key or "").strip() or os.environ.get("DEEPL_AUTH_KEY") or os.environ.get("DEEPL_API_KEY")
        if not key:
            raise HTTPException(
                status_code=500,
                detail="翻译失败（deepl）: 请填写 DeepL API Key，或在服务端设置 DEEPL_API_KEY",
            )
        deepl_target = "ZH" if target_lang in ("zh", "zh-CN") else target_lang.upper()
        client = deepl.DeepLClient(key)
        kwargs = {"target_lang": deepl_target}
        if source_lang:
            kwargs["source_lang"] = source_lang.upper()

        out: List[Optional[dict]] = [None] * total
        pending: List[Tuple[int, dict, str]] = []
        done = 0
        for idx, seg in enumerate(segments, 1):
            if cancel_event and cancel_event.is_set():
                raise JobCanceled("任务已取消")
            text = (seg.get("text") or "").strip()
            if not text:
                out[idx - 1] = seg
                done += 1
                if progress_cb:
                    progress_cb(done, total)
                continue
            pending.append((idx, seg, text))

        # 每批条数支持用户自定义；DeepL 单批上限保持 50
        batch_max_items = _resolve_batch_size(translation_batch_size, 50, max_value=50)
        batch_max_chars = 12000
        pos = 0
        while pos < len(pending):
            if cancel_event and cancel_event.is_set():
                raise JobCanceled("任务已取消")
            batch_meta: List[Tuple[int, dict, str]] = []
            batch_texts: List[str] = []
            chars = 0
            while pos < len(pending) and len(batch_meta) < batch_max_items:
                m = pending[pos]
                t = m[2]
                if batch_meta and chars + len(t) > batch_max_chars:
                    break
                batch_meta.append(m)
                batch_texts.append(t)
                chars += len(t)
                pos += 1
            try:
                result = client.translate_text(batch_texts, **kwargs)
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"翻译失败（deepl）: {e!s}",
                ) from e

            if isinstance(result, list):
                translated_items = result
            else:
                translated_items = [result]
            if len(translated_items) != len(batch_meta):
                raise HTTPException(
                    status_code=500,
                    detail="翻译失败（deepl）: 返回结果数量与请求不一致",
                )
            for (idx, seg, _), item in zip(batch_meta, translated_items):
                translated = item.text if hasattr(item, "text") else str(item)
                out[idx - 1] = {**seg, "text": translated}
                done += 1
                if progress_cb:
                    progress_cb(done, total)

        return [x if x is not None else segments[i] for i, x in enumerate(out)]

    # OpenAI: 使用单客户端 + 批量请求，减少请求次数提升速度
    if api_name == "openai":
        key = (openai_api_key or "").strip() or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise HTTPException(
                status_code=500,
                detail="翻译失败（openai）: 请填写 OpenAI API Key，或在服务端设置 OPENAI_API_KEY",
            )
        model = (openai_model or "").strip() or os.environ.get("OPENAI_TRANSLATE_MODEL", "gpt-4o-mini")
        model_lc = model.lower()
        gpt5_mode = "gpt-5" in model_lc
        reasoning_effort = _normalize_openai_reasoning_effort(openai_reasoning_effort)
        lang_name = LLM_TARGET_LANG_NAMES.get(
            target_lang if target_lang != "zh-CN" else "zh", "English"
        )
        rules = (translation_rules or "").strip()
        base_url = (openai_base_url or "").strip() or os.environ.get("OPENAI_BASE_URL")
        client = OpenAI(api_key=key, base_url=base_url or None)

        out: List[Optional[dict]] = [None] * total
        pending: List[Tuple[int, dict, str]] = []
        done = 0
        for idx, seg in enumerate(segments, 1):
            if cancel_event and cancel_event.is_set():
                raise JobCanceled("任务已取消")
            text = (seg.get("text") or "").strip()
            if not text:
                out[idx - 1] = seg
                done += 1
                if progress_cb:
                    progress_cb(done, total)
                continue
            pending.append((idx, seg, text))

        default_batch_items = 6 if gpt5_mode else 12
        batch_max_items = _resolve_batch_size(translation_batch_size, default_batch_items, max_value=200)
        batch_max_chars = 2500 if gpt5_mode else 5000

        def _openai_chat_create(**kwargs: Any) -> Any:
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort
            try:
                return client.chat.completions.create(**kwargs)
            except TypeError as e:
                # 兼容较老 SDK / 不支持该参数的中转
                if "reasoning_effort" in str(e) and "reasoning_effort" in kwargs:
                    kwargs.pop("reasoning_effort", None)
                    return client.chat.completions.create(**kwargs)
                raise
            except Exception as e:
                # 兼容仅服务端不支持该字段的中转实现
                if "reasoning_effort" in str(e).lower() and "reasoning_effort" in kwargs:
                    kwargs.pop("reasoning_effort", None)
                    return client.chat.completions.create(**kwargs)
                raise

        def _parse_openai_batch_json(raw_text: str, expected_len: int) -> List[str]:
            raw = (raw_text or "").strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE).strip()
            raw = re.sub(r"\s*```$", "", raw).strip()
            parsed: Any
            try:
                parsed = json.loads(raw)
            except Exception:
                # 尝试提取 JSON 对象或数组
                m_obj = re.search(r"\{[\s\S]*\}", raw)
                m_arr = re.search(r"\[[\s\S]*\]", raw)
                if m_obj:
                    parsed = json.loads(m_obj.group(0))
                elif m_arr:
                    parsed = json.loads(m_arr.group(0))
                else:
                    raise
            if isinstance(parsed, dict):
                items = parsed.get("items")
            elif isinstance(parsed, list):
                items = parsed
            else:
                raise ValueError("OpenAI 返回既不是 JSON 对象也不是数组")
            if not isinstance(items, list) or len(items) != expected_len:
                raise ValueError("OpenAI 返回 items 数量与请求不一致")
            return [str(x).strip() if x is not None else "" for x in items]

        def _single_translate(src_text: str) -> str:
            single_system = f"You are a translator. Output only the translation in {lang_name}, no explanation."
            if rules:
                single_system = (
                    f"You are a translator. Style and rules you must follow:\n{rules}\n\n"
                    f"Output only the translation in {lang_name}, no explanation."
                )
            for attempt in range(2):
                try:
                    one_resp = _openai_chat_create(
                        model=model,
                        messages=[
                            {"role": "system", "content": single_system},
                            {"role": "user", "content": src_text},
                        ],
                        max_tokens=1024,
                        temperature=0,
                    )
                    return (one_resp.choices[0].message.content or "").strip() or src_text
                except OpenAIRateLimitError as e:
                    if attempt < 1:
                        time.sleep(2 ** (attempt + 1))
                    else:
                        raise HTTPException(
                            status_code=429,
                            detail="OpenAI 请求过于频繁（限流）。请稍后再试，或在 platform.openai.com/account/limits 查看并提升用量/限流额度。",
                        ) from e
                except Exception:
                    if attempt < 1:
                        time.sleep(2 ** (attempt + 1))
                    else:
                        return src_text
            return src_text

        def _call_openai_batch(batch_texts: List[str]) -> List[str]:
            expected_len = len(batch_texts)
            if cancel_event and cancel_event.is_set():
                raise JobCanceled("任务已取消")
            system_content = (
                f"You are a translator. Translate every string to {lang_name}. "
                "Return ONLY valid JSON object with this schema: "
                "{\"items\": [\"translated string 1\", \"translated string 2\", ...]}. "
                "The items length and order must exactly match input. "
                "No markdown, no explanation, no extra keys."
            )
            if rules:
                system_content = (
                    f"You are a translator. Style and rules you must follow:\n{rules}\n\n"
                    f"Translate every string to {lang_name}. Return ONLY valid JSON object with schema "
                    "{\"items\": [...]}. items length and order must exactly match input. "
                    "No markdown, no explanation, no extra keys."
                )
            user_content = json.dumps(batch_texts, ensure_ascii=False)
            for attempt in range(2):
                try:
                    kwargs = dict(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": user_content},
                        ],
                        max_tokens=4096,
                        temperature=0,
                    )
                    # gpt-5 默认走非 strict，减少中转兼容问题
                    if gpt5_mode:
                        try:
                            resp = _openai_chat_create(
                                **kwargs,
                                response_format={"type": "json_object"},
                            )
                        except Exception:
                            resp = _openai_chat_create(**kwargs)
                    else:
                        # 其他模型优先 strict 结构化输出
                        try:
                            resp = _openai_chat_create(
                                **kwargs,
                                response_format={
                                    "type": "json_schema",
                                    "json_schema": {
                                        "name": "translation_batch",
                                        "strict": True,
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "items": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                    "minItems": expected_len,
                                                    "maxItems": expected_len,
                                                }
                                            },
                                            "required": ["items"],
                                            "additionalProperties": False,
                                        },
                                    },
                                },
                            )
                        except Exception:
                            try:
                                resp = _openai_chat_create(
                                    **kwargs,
                                    response_format={"type": "json_object"},
                                )
                            except Exception:
                                resp = _openai_chat_create(**kwargs)

                    finish_reason = (resp.choices[0].finish_reason or "").lower()
                    if finish_reason == "length":
                        raise ValueError("OpenAI 输出被截断（finish_reason=length）")
                    raw = (resp.choices[0].message.content or "").strip()
                    return _parse_openai_batch_json(raw, expected_len)
                except OpenAIRateLimitError as e:
                    if attempt < 1:
                        time.sleep(2 ** (attempt + 1))
                    else:
                        raise HTTPException(
                            status_code=429,
                            detail="OpenAI 请求过于频繁（限流）。请稍后再试，或在 platform.openai.com/account/limits 查看并提升用量/限流额度。",
                        ) from e
                except Exception as e:
                    err_msg = str(e)
                    if "insufficient_quota" in err_msg.lower() or "quota" in err_msg.lower():
                        raise HTTPException(
                            status_code=402,
                            detail="OpenAI 报错与额度/配额有关（可能与账单预算无关）：请到 platform.openai.com/account/limits 检查「用量上限」与「限流」，或确认当前 Key 所属组织是否为付费账户。",
                        ) from e
                    if attempt < 1:
                        time.sleep(2 ** (attempt + 1))
                    else:
                        raise RuntimeError(f"OpenAI 批量解析失败：{e!s}") from e

            raise RuntimeError("OpenAI 批量解析失败：未获取到有效返回")

        def _translate_batch_with_split(batch_meta: List[Tuple[int, dict, str]]) -> List[str]:
            if not batch_meta:
                return []
            texts = [x[2] for x in batch_meta]
            try:
                return _call_openai_batch(texts)
            except RuntimeError:
                # 先拆半批重试，最后才逐条兜底
                if len(batch_meta) == 1:
                    return [_single_translate(batch_meta[0][2])]
                mid = len(batch_meta) // 2
                left = _translate_batch_with_split(batch_meta[:mid])
                right = _translate_batch_with_split(batch_meta[mid:])
                return left + right

        pos = 0
        batch_no = 0
        while pos < len(pending):
            if cancel_event and cancel_event.is_set():
                raise JobCanceled("任务已取消")
            batch_meta: List[Tuple[int, dict, str]] = []
            chars = 0
            while pos < len(pending) and len(batch_meta) < batch_max_items:
                m = pending[pos]
                t = m[2]
                if batch_meta and chars + len(t) > batch_max_chars:
                    break
                batch_meta.append(m)
                chars += len(t)
                pos += 1

            batch_no += 1
            if status_cb:
                status_cb(f"第 {batch_no} 批等待返回中…")
            translated_items = _translate_batch_with_split(batch_meta)
            for (idx, seg, src_text), translated in zip(batch_meta, translated_items):
                out[idx - 1] = {**seg, "text": translated or src_text}
                done += 1
                if progress_cb:
                    progress_cb(done, total)

        return [x if x is not None else segments[i] for i, x in enumerate(out)]

    # Moonshot: 使用 OpenAI 兼容接口 + 批量请求
    if api_name == "moonshot":
        key = (moonshot_api_key or "").strip() or os.environ.get("MOONSHOT_API_KEY")
        if not key:
            raise HTTPException(
                status_code=500,
                detail="翻译失败（moonshot）: 请填写 Moonshot API Key，或在服务端设置 MOONSHOT_API_KEY",
            )
        model = (moonshot_model or "").strip() or os.environ.get("MOONSHOT_TRANSLATE_MODEL", "moonshot-v1-8k")
        lang_name = LLM_TARGET_LANG_NAMES.get(
            target_lang if target_lang != "zh-CN" else "zh", "English"
        )
        rules = (translation_rules or "").strip()
        base_url = (moonshot_base_url or "").strip() or os.environ.get("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1")
        client = OpenAI(api_key=key, base_url=base_url or None)

        out: List[Optional[dict]] = [None] * total
        pending: List[Tuple[int, dict, str]] = []
        done = 0
        for idx, seg in enumerate(segments, 1):
            if cancel_event and cancel_event.is_set():
                raise JobCanceled("任务已取消")
            text = (seg.get("text") or "").strip()
            if not text:
                out[idx - 1] = seg
                done += 1
                if progress_cb:
                    progress_cb(done, total)
                continue
            pending.append((idx, seg, text))

        batch_max_items = _resolve_batch_size(translation_batch_size, 12, max_value=200)
        batch_max_chars = 5000

        def _parse_batch_json(raw_text: str, expected_len: int) -> List[str]:
            raw = (raw_text or "").strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE).strip()
            raw = re.sub(r"\s*```$", "", raw).strip()
            parsed: Any
            try:
                parsed = json.loads(raw)
            except Exception:
                m_obj = re.search(r"\{[\s\S]*\}", raw)
                m_arr = re.search(r"\[[\s\S]*\]", raw)
                if m_obj:
                    parsed = json.loads(m_obj.group(0))
                elif m_arr:
                    parsed = json.loads(m_arr.group(0))
                else:
                    raise
            if isinstance(parsed, dict):
                items = parsed.get("items")
            elif isinstance(parsed, list):
                items = parsed
            else:
                raise ValueError("Moonshot 返回既不是 JSON 对象也不是数组")
            if not isinstance(items, list) or len(items) != expected_len:
                raise ValueError("Moonshot 返回 items 数量与请求不一致")
            return [str(x).strip() if x is not None else "" for x in items]

        def _single_translate(src_text: str) -> str:
            single_system = f"You are a translator. Output only the translation in {lang_name}, no explanation."
            if rules:
                single_system = (
                    f"You are a translator. Style and rules you must follow:\n{rules}\n\n"
                    f"Output only the translation in {lang_name}, no explanation."
                )
            for attempt in range(2):
                try:
                    one_resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": single_system},
                            {"role": "user", "content": src_text},
                        ],
                        max_tokens=1024,
                        temperature=0,
                    )
                    return (one_resp.choices[0].message.content or "").strip() or src_text
                except OpenAIRateLimitError as e:
                    if attempt < 1:
                        time.sleep(2 ** (attempt + 1))
                    else:
                        raise HTTPException(
                            status_code=429,
                            detail="Moonshot 请求过于频繁（限流）。请稍后再试，或检查平台配额限制。",
                        ) from e
                except Exception:
                    if attempt < 1:
                        time.sleep(2 ** (attempt + 1))
                    else:
                        return src_text
            return src_text

        def _call_batch(batch_texts: List[str]) -> List[str]:
            expected_len = len(batch_texts)
            if cancel_event and cancel_event.is_set():
                raise JobCanceled("任务已取消")
            system_content = (
                f"You are a translator. Translate every string to {lang_name}. "
                "Return ONLY valid JSON object with this schema: "
                "{\"items\": [\"translated string 1\", \"translated string 2\", ...]}. "
                "The items length and order must exactly match input. "
                "No markdown, no explanation, no extra keys."
            )
            if rules:
                system_content = (
                    f"You are a translator. Style and rules you must follow:\n{rules}\n\n"
                    f"Translate every string to {lang_name}. Return ONLY valid JSON object with schema "
                    "{\"items\": [...]}. items length and order must exactly match input. "
                    "No markdown, no explanation, no extra keys."
                )
            user_content = json.dumps(batch_texts, ensure_ascii=False)
            for attempt in range(2):
                try:
                    kwargs = dict(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": user_content},
                        ],
                        max_tokens=4096,
                        temperature=0,
                    )
                    try:
                        resp = client.chat.completions.create(
                            **kwargs,
                            response_format={
                                "type": "json_schema",
                                "json_schema": {
                                    "name": "translation_batch",
                                    "strict": True,
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "items": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                                "minItems": expected_len,
                                                "maxItems": expected_len,
                                            }
                                        },
                                        "required": ["items"],
                                        "additionalProperties": False,
                                    },
                                },
                            },
                        )
                    except Exception:
                        try:
                            resp = client.chat.completions.create(
                                **kwargs,
                                response_format={"type": "json_object"},
                            )
                        except Exception:
                            resp = client.chat.completions.create(**kwargs)

                    finish_reason = (resp.choices[0].finish_reason or "").lower()
                    if finish_reason == "length":
                        raise ValueError("Moonshot 输出被截断（finish_reason=length）")
                    raw = (resp.choices[0].message.content or "").strip()
                    return _parse_batch_json(raw, expected_len)
                except OpenAIRateLimitError as e:
                    if attempt < 1:
                        time.sleep(2 ** (attempt + 1))
                    else:
                        raise HTTPException(
                            status_code=429,
                            detail="Moonshot 请求过于频繁（限流）。请稍后再试，或检查平台配额限制。",
                        ) from e
                except Exception as e:
                    err_msg = str(e).lower()
                    if "insufficient_quota" in err_msg or "quota" in err_msg:
                        raise HTTPException(
                            status_code=402,
                            detail="Moonshot 报错与额度/配额有关，请检查平台账户额度与限流配置。",
                        ) from e
                    if attempt < 1:
                        time.sleep(2 ** (attempt + 1))
                    else:
                        raise RuntimeError(f"Moonshot 批量解析失败：{e!s}") from e
            raise RuntimeError("Moonshot 批量解析失败：未获取到有效返回")

        def _translate_batch_with_split(batch_meta: List[Tuple[int, dict, str]]) -> List[str]:
            if not batch_meta:
                return []
            texts = [x[2] for x in batch_meta]
            try:
                return _call_batch(texts)
            except RuntimeError:
                if len(batch_meta) == 1:
                    return [_single_translate(batch_meta[0][2])]
                mid = len(batch_meta) // 2
                left = _translate_batch_with_split(batch_meta[:mid])
                right = _translate_batch_with_split(batch_meta[mid:])
                return left + right

        pos = 0
        batch_no = 0
        while pos < len(pending):
            if cancel_event and cancel_event.is_set():
                raise JobCanceled("任务已取消")
            batch_meta: List[Tuple[int, dict, str]] = []
            chars = 0
            while pos < len(pending) and len(batch_meta) < batch_max_items:
                m = pending[pos]
                t = m[2]
                if batch_meta and chars + len(t) > batch_max_chars:
                    break
                batch_meta.append(m)
                chars += len(t)
                pos += 1

            batch_no += 1
            if status_cb:
                status_cb(f"第 {batch_no} 批等待 Moonshot 返回中…")
            translated_items = _translate_batch_with_split(batch_meta)
            for (idx, seg, src_text), translated in zip(batch_meta, translated_items):
                out[idx - 1] = {**seg, "text": translated or src_text}
                done += 1
                if progress_cb:
                    progress_cb(done, total)

        return [x if x is not None else segments[i] for i, x in enumerate(out)]

    # Gemini: 使用单客户端 + 批量请求（规则与 OpenAI 非 gpt-5 对齐）
    if api_name == "gemini":
        if genai is None:
            raise HTTPException(
                status_code=500,
                detail="翻译失败（gemini）: 未安装 google-genai，或当前 Python 版本不兼容（Gemini 需 Python 3.9+）。请先 pip install -r requirements.txt。",
            )
        key = (gemini_api_key or "").strip() or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise HTTPException(
                status_code=500,
                detail="翻译失败（gemini）: 请填写 Gemini API Key，或在服务端设置 GEMINI_API_KEY",
            )
        model = (gemini_model or "").strip() or os.environ.get("GEMINI_TRANSLATE_MODEL", "gemini-2.5-flash")
        thinking_level = _normalize_gemini_thinking_level(gemini_thinking_level)
        lang_name = LLM_TARGET_LANG_NAMES.get(
            target_lang if target_lang != "zh-CN" else "zh", "English"
        )
        rules = (translation_rules or "").strip()
        client = genai.Client(api_key=key)

        out: List[Optional[dict]] = [None] * total
        pending: List[Tuple[int, dict, str]] = []
        done = 0
        for idx, seg in enumerate(segments, 1):
            if cancel_event and cancel_event.is_set():
                raise JobCanceled("任务已取消")
            text = (seg.get("text") or "").strip()
            if not text:
                out[idx - 1] = seg
                done += 1
                if progress_cb:
                    progress_cb(done, total)
                continue
            pending.append((idx, seg, text))

        # 每批条数支持用户自定义；默认与 OpenAI 非 gpt-5 一致
        batch_max_items = _resolve_batch_size(translation_batch_size, 12, max_value=200)
        batch_max_chars = 5000

        def _extract_gemini_text(resp: Any) -> str:
            text = getattr(resp, "text", None)
            if text:
                return str(text)
            try:
                chunks: List[str] = []
                candidates = getattr(resp, "candidates", None) or []
                for cand in candidates:
                    content = getattr(cand, "content", None)
                    if not content:
                        continue
                    for part in (getattr(content, "parts", None) or []):
                        ptxt = getattr(part, "text", None)
                        if ptxt:
                            chunks.append(str(ptxt))
                return "\n".join(chunks).strip()
            except Exception:
                return ""

        def _parse_gemini_batch_json(raw_text: str, expected_len: int) -> List[str]:
            raw = (raw_text or "").strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE).strip()
            raw = re.sub(r"\s*```$", "", raw).strip()
            parsed: Any
            try:
                parsed = json.loads(raw)
            except Exception:
                m_obj = re.search(r"\{[\s\S]*\}", raw)
                m_arr = re.search(r"\[[\s\S]*\]", raw)
                if m_obj:
                    parsed = json.loads(m_obj.group(0))
                elif m_arr:
                    parsed = json.loads(m_arr.group(0))
                else:
                    raise
            if isinstance(parsed, dict):
                items = parsed.get("items")
            elif isinstance(parsed, list):
                items = parsed
            else:
                raise ValueError("Gemini 返回既不是 JSON 对象也不是数组")
            if not isinstance(items, list) or len(items) != expected_len:
                raise ValueError("Gemini 返回 items 数量与请求不一致")
            return [str(x).strip() if x is not None else "" for x in items]

        def _single_translate(src_text: str) -> str:
            single_prompt = (
                f"Translate the following text to {lang_name}. "
                "Output only the translation text, no explanation.\n\n"
                f"Text:\n{src_text}"
            )
            if rules:
                single_prompt = (
                    f"You are a translator. Follow these style/rules strictly:\n{rules}\n\n"
                    f"Translate the following text to {lang_name}. "
                    "Output only the translation text, no explanation.\n\n"
                    f"Text:\n{src_text}"
                )
            for attempt in range(2):
                try:
                    config_one: Dict[str, Any] = {"temperature": 0, "max_output_tokens": 2048}
                    if thinking_level:
                        config_one["thinking_config"] = {"thinking_level": thinking_level}
                    one_resp = client.models.generate_content(
                        model=model,
                        contents=single_prompt,
                        config=config_one,
                    )
                    return _extract_gemini_text(one_resp).strip() or src_text
                except Exception as e:
                    msg = str(e).lower()
                    if "429" in msg or "resource_exhausted" in msg or "rate limit" in msg or "too many requests" in msg:
                        if attempt < 1:
                            time.sleep(2 ** (attempt + 1))
                        else:
                            raise HTTPException(
                                status_code=429,
                                detail="Gemini 请求过于频繁（限流）。请稍后再试，或在 Google AI Studio / Gemini 控制台查看配额。",
                            ) from e
                    if attempt < 1:
                        time.sleep(2 ** (attempt + 1))
                    else:
                        return src_text
            return src_text

        def _call_gemini_batch(batch_texts: List[str]) -> List[str]:
            expected_len = len(batch_texts)
            if cancel_event and cancel_event.is_set():
                raise JobCanceled("任务已取消")
            schema = {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": expected_len,
                        "maxItems": expected_len,
                    }
                },
                "required": ["items"],
                "additionalProperties": False,
            }
            prompt = (
                f"You are a translator. Translate every string to {lang_name}. "
                "Return ONLY valid JSON object with this schema: "
                "{\"items\": [\"translated string 1\", \"translated string 2\", ...]}. "
                "The items length and order must exactly match input. "
                "No markdown, no explanation, no extra keys.\n\n"
                f"Input JSON array:\n{json.dumps(batch_texts, ensure_ascii=False)}"
            )
            if rules:
                prompt = (
                    f"You are a translator. Follow these style/rules strictly:\n{rules}\n\n"
                    f"Translate every string to {lang_name}. Return ONLY valid JSON object with schema "
                    "{\"items\": [...]}. items length and order must exactly match input. "
                    "No markdown, no explanation, no extra keys.\n\n"
                    f"Input JSON array:\n{json.dumps(batch_texts, ensure_ascii=False)}"
                )

            for attempt in range(2):
                try:
                    config_base: Dict[str, Any] = {
                        "temperature": 0,
                        "max_output_tokens": 8192,
                        "response_mime_type": "application/json",
                    }
                    if thinking_level:
                        config_base["thinking_config"] = {"thinking_level": thinking_level}
                    try:
                        resp = client.models.generate_content(
                            model=model,
                            contents=prompt,
                            config={**config_base, "response_json_schema": schema},
                        )
                    except Exception:
                        resp = client.models.generate_content(
                            model=model,
                            contents=prompt,
                            config=config_base,
                        )
                    raw = _extract_gemini_text(resp)
                    return _parse_gemini_batch_json(raw, expected_len)
                except Exception as e:
                    err_msg = str(e)
                    err_lc = err_msg.lower()
                    if "insufficient_quota" in err_lc or "quota" in err_lc:
                        raise HTTPException(
                            status_code=402,
                            detail="Gemini 报错与额度/配额有关：请检查 Google AI Studio / Gemini 控制台中的项目配额与账单状态。",
                        ) from e
                    if "429" in err_lc or "resource_exhausted" in err_lc or "rate limit" in err_lc or "too many requests" in err_lc:
                        if attempt < 1:
                            time.sleep(2 ** (attempt + 1))
                        else:
                            raise HTTPException(
                                status_code=429,
                                detail="Gemini 请求过于频繁（限流）。请稍后再试，或在 Google AI Studio / Gemini 控制台查看配额。",
                            ) from e
                    if attempt < 1:
                        time.sleep(2 ** (attempt + 1))
                    else:
                        raise RuntimeError(f"Gemini 批量解析失败：{e!s}") from e
            raise RuntimeError("Gemini 批量解析失败：未获取到有效返回")

        def _translate_batch_with_split(batch_meta: List[Tuple[int, dict, str]]) -> List[str]:
            if not batch_meta:
                return []
            texts = [x[2] for x in batch_meta]
            try:
                return _call_gemini_batch(texts)
            except RuntimeError:
                if len(batch_meta) == 1:
                    return [_single_translate(batch_meta[0][2])]
                mid = len(batch_meta) // 2
                if status_cb:
                    status_cb(f"当前批次过大，自动拆分为 {mid} + {len(batch_meta) - mid} 条重试…")
                left = _translate_batch_with_split(batch_meta[:mid])
                right = _translate_batch_with_split(batch_meta[mid:])
                return left + right

        pos = 0
        batch_no = 0
        while pos < len(pending):
            if cancel_event and cancel_event.is_set():
                raise JobCanceled("任务已取消")
            batch_meta: List[Tuple[int, dict, str]] = []
            chars = 0
            while pos < len(pending) and len(batch_meta) < batch_max_items:
                m = pending[pos]
                t = m[2]
                if batch_meta and chars + len(t) > batch_max_chars:
                    break
                batch_meta.append(m)
                chars += len(t)
                pos += 1

            batch_no += 1
            if status_cb:
                status_cb(f"第 {batch_no} 批等待 Gemini 返回中…")
            translated_items = _translate_batch_with_split(batch_meta)
            for (idx, seg, src_text), translated in zip(batch_meta, translated_items):
                out[idx - 1] = {**seg, "text": translated or src_text}
                done += 1
                if progress_cb:
                    progress_cb(done, total)

        return [x if x is not None else segments[i] for i, x in enumerate(out)]

    out = []
    if api_name == "google":
        target = _translator_target_code(target_lang)
        translator = GoogleTranslator(
            source=source_lang or "auto",
            target=target,
        )
        batch_max_items = _resolve_batch_size(translation_batch_size, 20, max_value=100)
        pending: List[Tuple[int, dict, str]] = []
        done = 0
        for idx, seg in enumerate(segments, 1):
            if cancel_event and cancel_event.is_set():
                raise JobCanceled("任务已取消")
            text = (seg.get("text") or "").strip()
            if not text:
                out.append(seg)
                done += 1
                if progress_cb:
                    progress_cb(done, total)
                continue
            pending.append((idx, seg, text))

        pos = 0
        while pos < len(pending):
            if cancel_event and cancel_event.is_set():
                raise JobCanceled("任务已取消")
            batch_meta = pending[pos: pos + batch_max_items]
            batch_texts = [x[2] for x in batch_meta]
            pos += len(batch_meta)
            try:
                if hasattr(translator, "translate_batch"):
                    translated_items = translator.translate_batch(batch_texts)
                else:
                    translated_items = [translator.translate(t) for t in batch_texts]
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"翻译失败（{api_name}）: {e!s}",
                ) from e
            if not isinstance(translated_items, list):
                translated_items = [str(translated_items)]
            if len(translated_items) != len(batch_meta):
                raise HTTPException(
                    status_code=500,
                    detail="翻译失败（google）: 返回结果数量与请求不一致",
                )
            for (_, seg, src_text), translated in zip(batch_meta, translated_items):
                out.append({**seg, "text": (translated or src_text)})
                done += 1
                if progress_cb:
                    progress_cb(done, total)
        return out

    for idx, seg in enumerate(segments, 1):
        if cancel_event and cancel_event.is_set():
            raise JobCanceled("任务已取消")
        text = (seg.get("text") or "").strip()
        if not text:
            out.append(seg)
            if progress_cb:
                progress_cb(idx, total)
            continue
        out.append({**seg, "text": text})
        if progress_cb:
            progress_cb(idx, total)
    return out


@app.get("/", response_class=HTMLResponse)
async def index():
    """返回前端页面"""
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="前端页面未找到")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/models")
async def list_models():
    """列出可选的 Whisper 模型"""
    return {"models": WHISPER_MODELS}


@app.get("/api/languages")
async def list_languages():
    """列出可选语言"""
    return {"languages": [{"code": c, "name": n} for c, n in LANGUAGES]}


@app.get("/api/translation_apis")
async def list_translation_apis():
    """列出可选的翻译 API"""
    return {"apis": [{"id": i, "name": n} for i, n in TRANSLATION_APIS]}


@app.get("/api/openai_models")
async def list_openai_models():
    """列出可选的 OpenAI 翻译模型"""
    return {"models": [{"id": i, "name": n} for i, n in OPENAI_TRANSLATE_MODELS]}


@app.get("/api/gemini_models")
async def list_gemini_models():
    """列出可选的 Gemini 翻译模型"""
    return {"models": [{"id": i, "name": n} for i, n in GEMINI_TRANSLATE_MODELS]}


@app.get("/api/moonshot_models")
async def list_moonshot_models():
    """列出可选的 Moonshot 翻译模型"""
    return {"models": [{"id": i, "name": n} for i, n in MOONSHOT_TRANSLATE_MODELS]}


@app.get("/api/ffmpeg/status")
async def ffmpeg_status():
    """检测 ffmpeg 是否可用，以及来源（系统 / 应用内）。"""
    available, source, path_or_error = check_ffmpeg_available()
    return {
        "available": available,
        "source": source if available else None,
        "path": path_or_error if available else None,
        "error": None if available else path_or_error,
    }


@app.post("/api/ffmpeg/install")
async def ffmpeg_install():
    """若未检测到 ffmpeg，则下载并安装到应用目录，并配置当前进程 PATH。"""
    success, message = ensure_ffmpeg_installed()
    return {"success": success, "message": message}


@app.post("/api/transcribe")
async def transcribe_video(
    file: UploadFile = File(...),
    model_name: str = Form("base"),
    language: str = Form("auto"),
    engine: str = Form("whisper"),
    translate_to: str = Form("none"),
    translation_api: str = Form("none"),
    openai_api_key: str = Form(""),
    openai_base_url: str = Form(""),
    deepl_api_key: str = Form(""),
    openai_model: str = Form(""),
    openai_reasoning_effort: str = Form(""),
    gemini_api_key: str = Form(""),
    gemini_model: str = Form(""),
    gemini_thinking_level: str = Form(""),
    moonshot_api_key: str = Form(""),
    moonshot_base_url: str = Form(""),
    moonshot_model: str = Form(""),
    translation_rules: str = Form(""),
    translation_batch_size: int = Form(0),
):
    """
    上传视频/音频，使用 Whisper 转写，可选翻译成另一语言，返回 SRT 文件。
    language: 识别语言；translate_to: 翻译目标语言（none 表示不翻译）；
    translation_api: 翻译引擎（none / google / deepl / openai / gemini / moonshot）。
    """
    ext = get_extension(file.filename or "")
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式。支持: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    if model_name not in WHISPER_MODELS:
        raise HTTPException(status_code=400, detail=f"不支持的模型: {model_name}")
    if engine not in TRANSCRIBE_ENGINES:
        raise HTTPException(status_code=400, detail=f"不支持的识别引擎: {engine}")
    if translation_api and translation_api != "none" and (not translate_to or translate_to == "none"):
        raise HTTPException(status_code=400, detail="已选择翻译 API，请同时选择“翻译成”目标语言")
    if translate_to and translate_to != "none" and (not translation_api or translation_api == "none"):
        raise HTTPException(status_code=400, detail="已选择“翻译成”目标语言，请同时选择翻译 API")

    available, _, path_or_error = check_ffmpeg_available()
    if not available:
        raise HTTPException(
            status_code=400,
            detail=path_or_error or "未检测到 ffmpeg。请安装到系统 PATH 或在首页点击「一键安装 ffmpeg」。",
        )

    # 保存上传到临时文件（保留原扩展名以便 ffmpeg 识别）
    suffix = ext or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        input_path = tmp.name

    base_name = Path(file.filename or "video").stem
    out_suffix = _translated_suffix(
        translate_to,
        translation_api,
        openai_model,
        gemini_model,
        moonshot_model,
    ) if (translate_to and translate_to != "none") else ""
    filename = f"{base_name}{out_suffix}.srt"
    filename_original = f"{base_name}.srt" if (translate_to and translate_to != "none") else None
    job_id = str(uuid.uuid4())
    _init_job(job_id, filename, filename_original=filename_original)

    thread = threading.Thread(
        target=_process_job,
        args=(
            job_id,
            input_path,
            model_name,
            language,
            engine,
            translate_to,
            translation_api,
            openai_api_key,
            openai_base_url,
            deepl_api_key,
            openai_model,
            openai_reasoning_effort,
            gemini_api_key,
            gemini_model,
            gemini_thinking_level,
            moonshot_api_key,
            moonshot_base_url,
            moonshot_model,
            translation_rules,
            translation_batch_size,
            base_name,
        ),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id}


@app.post("/api/translate")
async def translate_only(
    file: UploadFile = File(...),
    translate_to: str = Form(...),
    translation_api: str = Form(...),
    openai_api_key: str = Form(""),
    openai_base_url: str = Form(""),
    deepl_api_key: str = Form(""),
    openai_model: str = Form(""),
    openai_reasoning_effort: str = Form(""),
    gemini_api_key: str = Form(""),
    gemini_model: str = Form(""),
    gemini_thinking_level: str = Form(""),
    moonshot_api_key: str = Form(""),
    moonshot_base_url: str = Form(""),
    moonshot_model: str = Form(""),
    translation_rules: str = Form(""),
    translation_batch_size: int = Form(0),
):
    """
    仅翻译：上传 .srt 字幕文件，翻译成目标语言后返回新 SRT。不进行语音转写。
    """
    ext = get_extension(file.filename or "")
    if ext != ".srt":
        raise HTTPException(status_code=400, detail="仅支持 .srt 字幕文件")
    if not translate_to or translate_to == "none":
        raise HTTPException(status_code=400, detail="请选择翻译目标语言")
    if not translation_api or translation_api == "none":
        raise HTTPException(status_code=400, detail="请选择翻译 API")

    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".srt") as tmp:
        content = await file.read()
        tmp.write(content)
        srt_path = tmp.name

    base_name = Path(file.filename or "subtitle").stem
    out_suffix = _translated_suffix(
        translate_to,
        translation_api,
        openai_model,
        gemini_model,
        moonshot_model,
    )
    filename = f"{base_name}{out_suffix}.srt"
    job_id = str(uuid.uuid4())
    _init_job(job_id, filename)

    thread = threading.Thread(
        target=_process_translate_only_job,
        args=(
            job_id,
            srt_path,
            translate_to,
            translation_api,
            openai_api_key,
            openai_base_url,
            deepl_api_key,
            openai_model,
            openai_reasoning_effort,
            gemini_api_key,
            gemini_model,
            gemini_thinking_level,
            moonshot_api_key,
            moonshot_base_url,
            moonshot_model,
            translation_rules,
            translation_batch_size,
            base_name,
        ),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    out = {
        "job_id": job_id,
        "status": job.get("status"),
        "stage": job.get("stage"),
        "progress": job.get("progress"),
        "message": job.get("message"),
        "filename": job.get("filename"),
        "error": job.get("error"),
        "eta_seconds": job.get("eta_seconds"),
    }
    if job.get("filename_original") and job.get("srt_original") is not None:
        out["filename_original"] = job.get("filename_original")
    return out


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    if job.get("status") in ("done", "error", "canceled"):
        return {"success": True, "status": job.get("status")}
    ev = _get_cancel_event(job_id)
    if ev:
        ev.set()
    _update_job(job_id, status="canceled", stage="canceled", progress=0, message="已取消", eta_seconds=None)
    return {"success": True, "status": "canceled"}


@app.get("/api/jobs/{job_id}/download")
async def download_job(job_id: str, file: str = "translated"):
    """
    file=translated（默认）下载翻译后的 SRT；file=original 下载未翻译的原文 SRT（仅当任务包含翻译时可用）。
    """
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    if file == "original":
        srt_content = job.get("srt_original")
        filename = job.get("filename_original")
        if srt_content is None or not filename:
            raise HTTPException(status_code=400, detail="该任务无未翻译版本")
    else:
        if job.get("status") != "done":
            raise HTTPException(status_code=400, detail="任务未完成")
        srt_content = job.get("srt") or ""
        filename = job.get("filename") or "subtitle.srt"
    return Response(
        content=(srt_content or "").encode("utf-8"),
        media_type="application/x-subrip; charset=utf-8",
        headers={"Content-Disposition": _content_disposition(filename)},
    )


if __name__ == "__main__":
    import threading
    import webbrowser
    import uvicorn

    port = 8765
    url = f"http://127.0.0.1:{port}"

    def open_browser():
        import time
        time.sleep(1.2)
        webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=port)
