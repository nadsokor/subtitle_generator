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
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Any, Optional, List, Tuple

import torch
import whisper
import deepl
from deep_translator import GoogleTranslator
from openai import OpenAI
from openai import RateLimitError as OpenAIRateLimitError
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

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
]

# 部分语言在翻译 API 中的目标代码（与 Whisper 不一致时）
TRANSLATOR_TARGET_MAP = {
    "zh": "zh-CN",  # Google 使用 zh-CN
}

# OpenAI 翻译时目标语言英文名（用于 prompt）
OPENAI_TARGET_LANG_NAMES = {
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


def _update_job(job_id: str, **kwargs: Any) -> None:
    with JOB_LOCK:
        if job_id in JOB_STORE:
            JOB_STORE[job_id].update(kwargs)


def _get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with JOB_LOCK:
        return JOB_STORE.get(job_id)


def get_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def format_timestamp_srt(seconds: float) -> str:
    """将秒数转为 SRT 时间轴格式 00:00:00,000"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


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
    """优先 CUDA，其次 Apple MPS，否则 CPU。"""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
    translate_to: str,
    translation_api: str,
    openai_api_key: str,
    deepl_api_key: str,
    openai_model: str,
    translation_rules: str,
    base_name: str = "",
) -> None:
    work_dir = None
    try:
        _update_job(job_id, status="running", stage="downloading", progress=0, message="下载模型中 0%", eta_seconds=None)

        def download_progress(pct: int, eta_sec: Optional[int]) -> None:
            msg = f"下载模型中 {pct}%"
            if eta_sec is not None and eta_sec > 0:
                msg += f"（约剩余 {_format_eta(eta_sec)}）"
            _update_job(job_id, stage="downloading", progress=pct, message=msg, eta_seconds=eta_sec)

        _ensure_model_downloaded(model_name, download_progress)

        _update_job(job_id, stage="transcribing", progress=0, message="转写中 0%", eta_seconds=None)
        _device = _whisper_device()
        model = whisper.load_model(model_name, device=_device)
        transcribe_options = {"word_timestamps": False}
        if language and language != "auto":
            transcribe_options["language"] = language

        work_dir = tempfile.mkdtemp(prefix="auto_subbed_")
        chunks = _split_audio(input_path, work_dir)
        if not chunks:
            raise RuntimeError("无法切分音频，请检查输入文件或 ffmpeg")

        all_segments = []
        total_chunks = len(chunks)
        offset = 0.0
        transcribe_start = time.time()
        for idx, chunk in enumerate(chunks, 1):
            result = model.transcribe(chunk, **transcribe_options)
            for seg in result.get("segments") or []:
                seg["start"] = (seg.get("start") or 0) + offset
                seg["end"] = (seg.get("end") or 0) + offset
                all_segments.append(seg)

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
                deepl_api_key=deepl_api_key or None,
                openai_model=openai_model or None,
                translation_rules=translation_rules or None,
                progress_cb=translate_progress,
            )

        srt_content = segments_to_srt(all_segments)
        _update_job(job_id, status="done", stage="done", progress=100, message="完成", srt=srt_content, eta_seconds=0)
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
    deepl_api_key: str,
    openai_model: str,
    translation_rules: str,
    base_name: str,
) -> None:
    """仅翻译：读取 SRT 文件，翻译后写回 SRT，不涉及转写。"""
    try:
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
            deepl_api_key=deepl_api_key or None,
            openai_model=openai_model or None,
            translation_rules=translation_rules or None,
            progress_cb=translate_progress,
        )
        srt_content = segments_to_srt(translated)
        _update_job(job_id, status="done", stage="done", progress=100, message="完成", srt=srt_content, eta_seconds=0)
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


def translate_segments(
    segments: list,
    target_lang: str,
    api_name: str,
    source_lang: str | None = None,
    *,
    openai_api_key: str | None = None,
    deepl_api_key: str | None = None,
    openai_model: str | None = None,
    translation_rules: str | None = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> list:
    """
    将 segments 中每条 text 翻译成目标语言，时间轴不变。
    api_name: 'google' | 'deepl' | 'openai'
    translation_rules: 可选，自定义风格与规则（目前仅 OpenAI 支持）。
    """
    if not segments or api_name == "none":
        return segments
    out = []
    total = len(segments)
    for idx, seg in enumerate(segments, 1):
        text = (seg.get("text") or "").strip()
        if not text:
            out.append(seg)
            if progress_cb:
                progress_cb(idx, total)
            continue
        try:
            if api_name == "google":
                target = _translator_target_code(target_lang)
                translator = GoogleTranslator(
                    source=source_lang or "auto",
                    target=target,
                )
                translated = translator.translate(text)
            elif api_name == "deepl":
                key = (deepl_api_key or "").strip() or os.environ.get("DEEPL_AUTH_KEY") or os.environ.get("DEEPL_API_KEY")
                if not key:
                    raise ValueError("请填写 DeepL API Key，或在服务端设置 DEEPL_API_KEY")
                # 使用官方 deepl 库（POST + Authorization 头），兼容 2025-03 后 DeepL 弃用 GET/auth_key 的变更
                deepl_target = "ZH" if target_lang in ("zh", "zh-CN") else target_lang.upper()
                client = deepl.DeepLClient(key)
                kwargs = {"target_lang": deepl_target}
                if source_lang:
                    kwargs["source_lang"] = source_lang.upper()
                result = client.translate_text(text, **kwargs)
                translated = result.text if hasattr(result, "text") else str(result)
            elif api_name == "openai":
                key = (openai_api_key or "").strip() or os.environ.get("OPENAI_API_KEY")
                if not key:
                    raise ValueError("请填写 OpenAI API Key，或在服务端设置 OPENAI_API_KEY")
                model = (openai_model or "").strip() or os.environ.get("OPENAI_TRANSLATE_MODEL", "gpt-4o-mini")
                lang_name = OPENAI_TARGET_LANG_NAMES.get(
                    target_lang if target_lang != "zh-CN" else "zh", "English"
                )
                rules = (translation_rules or "").strip()
                system_content = f"You are a translator. Output only the translation in {lang_name}, no explanation."
                if rules:
                    system_content = f"You are a translator. Style and rules you must follow:\n{rules}\n\nOutput only the translation in {lang_name}, no explanation."
                client = OpenAI(api_key=key)
                translated = None
                for attempt in range(4):
                    try:
                        resp = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": system_content},
                                {"role": "user", "content": text},
                            ],
                            max_tokens=1024,
                        )
                        translated = (resp.choices[0].message.content or "").strip() or text
                        break
                    except OpenAIRateLimitError as e:
                        if attempt < 3:
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
                        raise
                if translated is None:
                    translated = text
            else:
                translated = text
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"翻译失败（{api_name}）: {e!s}",
            )
        out.append({**seg, "text": translated})
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
    translate_to: str = Form("none"),
    translation_api: str = Form("none"),
    openai_api_key: str = Form(""),
    deepl_api_key: str = Form(""),
    openai_model: str = Form(""),
    translation_rules: str = Form(""),
):
    """
    上传视频/音频，使用 Whisper 转写，可选翻译成另一语言，返回 SRT 文件。
    language: 识别语言；translate_to: 翻译目标语言（none 表示不翻译）；
    translation_api: 翻译引擎（none / google / deepl / openai）。
    """
    ext = get_extension(file.filename or "")
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式。支持: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    if model_name not in WHISPER_MODELS:
        raise HTTPException(status_code=400, detail=f"不支持的模型: {model_name}")

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
    out_suffix = f".{translate_to}" if (translate_to and translate_to != "none") else ""
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
            translate_to,
            translation_api,
            openai_api_key,
            deepl_api_key,
            openai_model,
            translation_rules,
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
    deepl_api_key: str = Form(""),
    openai_model: str = Form(""),
    translation_rules: str = Form(""),
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
    filename = f"{base_name}.{translate_to}.srt"
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
            deepl_api_key,
            openai_model,
            translation_rules,
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


@app.get("/api/jobs/{job_id}/download")
async def download_job(job_id: str, file: str = "translated"):
    """
    file=translated（默认）下载翻译后的 SRT；file=original 下载未翻译的原文 SRT（仅当任务包含翻译时可用）。
    """
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    if job.get("status") != "done":
        raise HTTPException(status_code=400, detail="任务未完成")
    if file == "original":
        srt_content = job.get("srt_original")
        filename = job.get("filename_original")
        if srt_content is None or not filename:
            raise HTTPException(status_code=400, detail="该任务无未翻译版本")
    else:
        srt_content = job.get("srt") or ""
        filename = job.get("filename") or "subtitle.srt"
    return Response(
        content=(srt_content or "").encode("utf-8"),
        media_type="application/x-subrip; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
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
