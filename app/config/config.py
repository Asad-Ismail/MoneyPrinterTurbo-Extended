import copy
import os
import shutil
import socket
from typing import Any

import toml
from loguru import logger

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
config_file = f"{root_dir}/config.toml"

DEFAULT_PROFILE_NAME = "default"
DEFAULT_PRESET_ID = "youtube-explainer"
DEFAULT_JAPANESE_UI_LANGUAGE = "ja"
DEFAULT_JAPANESE_VIDEO_LANGUAGE = "ja-JP"
DEFAULT_JAPANESE_VOICE_NAME = "ja-JP-NanamiNeural-Female"
LEGACY_REPLACEABLE_VOICES = {"", "en-AU-NatashaNeural-Female"}


def default_tts_server_for_platform() -> str:
    return "windows-sapi" if os.name == "nt" else "azure-tts-v1"

DEFAULT_GROUPED_CONFIG = {
    "project": {
        "preset_id": DEFAULT_PRESET_ID,
        "profile_name": DEFAULT_PROFILE_NAME,
        "video_source": "pexels",
        "video_aspect": "9:16",
        "video_language": "",
    },
    "llm": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "base_url": "",
        "api_key_env": "",
        "timeout_sec": 60,
        "retry_count": 5,
    },
    "pipeline": {
        "subtitle_provider": "edge",
        "resume_from": "auto",
        "cache_enabled": True,
        "reuse_intermediate": True,
        "material_directory": "",
    },
    "performance": {
        "run_mode": "stable",
        "compute_profile": "cpu-safe",
        "asset_policy": "balanced",
        "preview_enabled": True,
        "preview_resolution": "720x1280",
        "preview_fps": 24,
        "max_concurrent_tasks": 2,
        "n_threads": 2,
        "verbose": False,
        "enable_chatterbox": False,
        "allow_voice_clone": False,
    },
    "style": {
        "tts_server": default_tts_server_for_platform(),
        "subtitle_density": "balanced",
        "audio_mix_profile": "speech-first",
        "font_name": "MicrosoftYaHeiBold.ttc",
        "font_size": 60,
        "text_fore_color": "#FFFFFF",
        "highlight_color": "#ff0000",
        "max_chars_per_line": 40,
        "max_lines_per_subtitle": 2,
    },
    "quality": {
        "enable_checks": True,
        "max_audio_subtitle_delta_sec": 1.2,
        "max_subtitle_chars_per_line": 40,
        "max_subtitle_lines": 2,
        "max_material_duplication_rate": 0.35,
        "max_bgm_to_voice_ratio": 0.35,
        "ng_words": [
            "guaranteed profit",
            "risk free",
            "絶対儲かる",
            "必ず勝てる",
            "元本保証",
        ],
    },
}

BUILTIN_PRESETS = {
    "shorts-basic": {
        "label": "Shorts Basic",
        "video_type": "shorts",
        "visible_sections": ["project", "llm", "performance", "style"],
        "default_config": {
            "project": {"preset_id": "shorts-basic", "video_aspect": "9:16"},
            "performance": {
                "run_mode": "fast-preview",
                "compute_profile": "cpu-safe",
                "asset_policy": "minimal",
                "preview_resolution": "720x1280",
                "preview_fps": 24,
                "max_concurrent_tasks": 1,
            },
            "style": {
                "subtitle_density": "short",
                "audio_mix_profile": "speech-first",
                "font_size": 66,
                "max_chars_per_line": 24,
            },
            "quality": {
                "max_subtitle_chars_per_line": 24,
                "max_subtitle_lines": 2,
            },
        },
    },
    "youtube-explainer": {
        "label": "YouTube Explainer",
        "video_type": "explainer",
        "visible_sections": ["project", "llm", "pipeline", "performance", "style", "quality"],
        "default_config": {
            "project": {"preset_id": "youtube-explainer", "video_aspect": "16:9"},
            "performance": {
                "run_mode": "stable",
                "compute_profile": "cpu-safe",
                "asset_policy": "balanced",
                "max_concurrent_tasks": 2,
            },
            "style": {
                "subtitle_density": "balanced",
                "audio_mix_profile": "speech-first",
                "font_size": 60,
                "max_chars_per_line": 40,
            },
        },
    },
    "news-summary": {
        "label": "News Summary",
        "video_type": "news",
        "visible_sections": ["project", "llm", "performance", "style", "quality"],
        "default_config": {
            "project": {"preset_id": "news-summary", "video_aspect": "9:16"},
            "performance": {
                "run_mode": "stable",
                "compute_profile": "cpu-safe",
                "asset_policy": "minimal",
                "max_concurrent_tasks": 1,
            },
            "style": {
                "subtitle_density": "balanced",
                "audio_mix_profile": "speech-first",
                "font_size": 64,
                "max_chars_per_line": 30,
            },
            "quality": {
                "max_subtitle_chars_per_line": 30,
            },
        },
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    if not override:
        return base
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def is_japanese_language(language: str | None) -> bool:
    normalized = (language or "").strip().lower().replace("_", "-")
    return normalized == "ja" or normalized.startswith("ja-")


def default_video_language_for_ui(ui_language: str | None) -> str:
    if is_japanese_language(ui_language):
        return DEFAULT_JAPANESE_VIDEO_LANGUAGE
    return ""


def default_voice_name_for_languages(
    ui_language: str | None, video_language: str | None
) -> str:
    if is_japanese_language(video_language) or is_japanese_language(ui_language):
        return DEFAULT_JAPANESE_VOICE_NAME
    return ""


def _apply_language_defaults(cfg: dict[str, Any], grouped: dict[str, Any]) -> None:
    ui_cfg = cfg.setdefault("ui", {})
    project_cfg = grouped.setdefault("project", {})
    style_cfg = grouped.setdefault("style", {})
    quality_cfg = grouped.setdefault("quality", {})

    ui_language = ui_cfg.get("language", "")
    video_language = project_cfg.get("video_language", "")
    if not (is_japanese_language(ui_language) or is_japanese_language(video_language)):
        return

    project_cfg.setdefault("video_language", DEFAULT_JAPANESE_VIDEO_LANGUAGE)
    if not project_cfg.get("video_language"):
        project_cfg["video_language"] = DEFAULT_JAPANESE_VIDEO_LANGUAGE

    current_voice = ui_cfg.get("voice_name", "").strip()
    if current_voice in LEGACY_REPLACEABLE_VOICES:
        ui_cfg["voice_name"] = DEFAULT_JAPANESE_VOICE_NAME

    style_cfg["max_chars_per_line"] = min(style_cfg.get("max_chars_per_line", 40), 26)
    quality_cfg["max_subtitle_chars_per_line"] = min(
        quality_cfg.get("max_subtitle_chars_per_line", 40), 26
    )


def _apply_windows_tts_defaults(cfg: dict[str, Any], grouped: dict[str, Any]) -> None:
    if os.name != "nt":
        return

    performance_cfg = grouped.setdefault("performance", {})
    if performance_cfg.get("compute_profile", "cpu-safe") != "cpu-safe":
        return

    ui_cfg = cfg.setdefault("ui", {})
    style_cfg = grouped.setdefault("style", {})
    azure_cfg = cfg.setdefault("azure", {})

    current_tts_server = (
        style_cfg.get("tts_server")
        or ui_cfg.get("tts_server")
        or default_tts_server_for_platform()
    ).strip().lower()

    # Windows + CPU-safe の既定経路では Edge TTS V1 が不安定なため、
    # 明示的な Azure Speech キー未設定時は内蔵音声へ寄せる。
    if current_tts_server == "azure-tts-v2":
        return
    if azure_cfg.get("speech_key", "").strip() and azure_cfg.get("speech_region", "").strip():
        return
    if current_tts_server in ("", "azure-tts-v1"):
        ui_cfg["tts_server"] = "windows-sapi"
        style_cfg["tts_server"] = "windows-sapi"


def _load_toml(path: str) -> dict[str, Any]:
    try:
        return toml.load(path)
    except Exception as e:
        logger.warning(f"load config failed: {str(e)}, try to load as utf-8-sig")
        with open(path, mode="r", encoding="utf-8-sig") as fp:
            return toml.loads(fp.read())


def _ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def presets_dir() -> str:
    return os.path.join(root_dir, "presets")


def profiles_dir() -> str:
    return os.path.join(root_dir, "profiles")


def state_dir() -> str:
    return os.path.join(root_dir, "state")


def last_profile_path() -> str:
    return os.path.join(state_dir(), "last_used_profile.toml")


def _legacy_defaults() -> dict[str, Any]:
    return {
        "app": {
            "video_source": "pexels",
            "hide_config": False,
            "pexels_api_keys": [],
            "pixabay_api_keys": [],
            "llm_provider": "openai",
            "openai_api_key": "",
            "openai_base_url": "",
            "openai_model_name": "gpt-4o-mini",
            "groq_api_key": "",
            "groq_base_url": "https://api.groq.com/openai/v1",
            "groq_model_name": "llama-3.1-8b-instant",
            "openrouter_api_key": "",
            "openrouter_base_url": "https://openrouter.ai/api/v1",
            "openrouter_model_name": "openai/gpt-4o-mini",
            "moonshot_api_key": "",
            "moonshot_base_url": "https://api.moonshot.cn/v1",
            "moonshot_model_name": "moonshot-v1-8k",
            "oneapi_api_key": "",
            "oneapi_base_url": "",
            "oneapi_model_name": "",
            "g4f_model_name": "gpt-3.5-turbo",
            "azure_api_key": "",
            "azure_base_url": "",
            "azure_model_name": "gpt-35-turbo",
            "azure_api_version": "2024-02-15-preview",
            "gemini_api_key": "",
            "gemini_model_name": "gemini-1.0-pro",
            "qwen_api_key": "",
            "qwen_model_name": "qwen-max",
            "deepseek_api_key": "",
            "deepseek_base_url": "https://api.deepseek.com",
            "deepseek_model_name": "deepseek-chat",
            "cloudflare_api_key": "",
            "cloudflare_account_id": "",
            "cloudflare_model_name": "",
            "ernie_api_key": "",
            "ernie_secret_key": "",
            "ernie_base_url": "",
            "pollinations_api_key": "",
            "pollinations_base_url": "https://pollinations.ai/api/v1",
            "pollinations_model_name": "openai-fast",
            "subtitle_provider": "edge",
            "material_directory": "",
            "verbose": False,
            "max_concurrent_tasks": 2,
        },
        "whisper": {
            "model_size": "large-v3",
            "device": "CPU",
            "compute_type": "int8",
        },
        "proxy": {},
        "azure": {"speech_key": "", "speech_region": ""},
        "siliconflow": {"api_key": ""},
        "ui": {
            "hide_log": False,
            "language": "en",
            "tts_server": default_tts_server_for_platform(),
            "voice_name": "",
            "font_name": "MicrosoftYaHeiBold.ttc",
            "text_fore_color": "#FFFFFF",
            "font_size": 60,
            "enable_word_highlighting": False,
            "highlight_color": "#ff0000",
            "max_chars_per_line": 40,
            "max_lines_per_subtitle": 2,
        },
    }


def _build_grouped_from_legacy(cfg: dict[str, Any]) -> dict[str, Any]:
    app_cfg = cfg.get("app", {})
    ui_cfg = cfg.get("ui", {})
    whisper_cfg = cfg.get("whisper", {})
    project_cfg = copy.deepcopy(cfg.get("project", {}))
    llm_cfg = copy.deepcopy(cfg.get("llm", {}))
    pipeline_cfg = copy.deepcopy(cfg.get("pipeline", {}))
    performance_cfg = copy.deepcopy(cfg.get("performance", {}))
    style_cfg = copy.deepcopy(cfg.get("style", {}))
    quality_cfg = copy.deepcopy(cfg.get("quality", {}))

    project_cfg.setdefault("preset_id", app_cfg.get("preset_id", DEFAULT_PRESET_ID))
    project_cfg.setdefault("profile_name", app_cfg.get("profile_name", DEFAULT_PROFILE_NAME))
    project_cfg.setdefault("video_source", app_cfg.get("video_source", "pexels"))
    project_cfg.setdefault("video_aspect", app_cfg.get("video_aspect", "9:16"))
    project_cfg.setdefault("video_language", app_cfg.get("video_language", ""))

    provider = llm_cfg.get("provider") or app_cfg.get("llm_provider", "openai")
    llm_cfg.setdefault("provider", provider)
    llm_cfg.setdefault("timeout_sec", app_cfg.get("llm_timeout_sec", 60))
    llm_cfg.setdefault("retry_count", app_cfg.get("llm_retry_count", 5))
    llm_cfg.setdefault("api_key_env", app_cfg.get("llm_api_key_env", ""))
    llm_cfg.setdefault("base_url", app_cfg.get(f"{provider}_base_url", ""))
    llm_cfg.setdefault("model", app_cfg.get(f"{provider}_model_name", ""))

    pipeline_cfg.setdefault("subtitle_provider", app_cfg.get("subtitle_provider", "edge"))
    pipeline_cfg.setdefault("resume_from", app_cfg.get("resume_from", "auto"))
    pipeline_cfg.setdefault("cache_enabled", app_cfg.get("cache_enabled", True))
    pipeline_cfg.setdefault("reuse_intermediate", app_cfg.get("reuse_intermediate", True))
    pipeline_cfg.setdefault("material_directory", app_cfg.get("material_directory", ""))

    performance_cfg.setdefault("run_mode", app_cfg.get("run_mode", "stable"))
    performance_cfg.setdefault("compute_profile", app_cfg.get("compute_profile", "cpu-safe"))
    performance_cfg.setdefault("asset_policy", app_cfg.get("asset_policy", "balanced"))
    performance_cfg.setdefault("preview_enabled", app_cfg.get("preview_enabled", True))
    performance_cfg.setdefault("preview_resolution", app_cfg.get("preview_resolution", "720x1280"))
    performance_cfg.setdefault("preview_fps", app_cfg.get("preview_fps", 24))
    performance_cfg.setdefault("max_concurrent_tasks", app_cfg.get("max_concurrent_tasks", 2))
    performance_cfg.setdefault("n_threads", app_cfg.get("n_threads", 2))
    performance_cfg.setdefault("verbose", app_cfg.get("verbose", False))
    performance_cfg.setdefault("enable_chatterbox", ui_cfg.get("tts_server") == "chatterbox")
    performance_cfg.setdefault("allow_voice_clone", app_cfg.get("allow_voice_clone", False))

    style_cfg.setdefault("tts_server", ui_cfg.get("tts_server", default_tts_server_for_platform()))
    style_cfg.setdefault("subtitle_density", app_cfg.get("subtitle_density", "balanced"))
    style_cfg.setdefault("audio_mix_profile", app_cfg.get("audio_mix_profile", "speech-first"))
    style_cfg.setdefault("font_name", ui_cfg.get("font_name", "MicrosoftYaHeiBold.ttc"))
    style_cfg.setdefault("font_size", ui_cfg.get("font_size", 60))
    style_cfg.setdefault("text_fore_color", ui_cfg.get("text_fore_color", "#FFFFFF"))
    style_cfg.setdefault("highlight_color", ui_cfg.get("highlight_color", "#ff0000"))
    style_cfg.setdefault("max_chars_per_line", ui_cfg.get("max_chars_per_line", 40))
    style_cfg.setdefault("max_lines_per_subtitle", ui_cfg.get("max_lines_per_subtitle", 2))

    quality_cfg.setdefault("enable_checks", app_cfg.get("enable_quality_checks", True))
    quality_cfg.setdefault("max_audio_subtitle_delta_sec", app_cfg.get("max_audio_subtitle_delta_sec", 1.2))
    quality_cfg.setdefault("max_subtitle_chars_per_line", ui_cfg.get("max_chars_per_line", 40))
    quality_cfg.setdefault("max_subtitle_lines", ui_cfg.get("max_lines_per_subtitle", 2))
    quality_cfg.setdefault("max_material_duplication_rate", app_cfg.get("max_material_duplication_rate", 0.35))
    quality_cfg.setdefault("max_bgm_to_voice_ratio", app_cfg.get("max_bgm_to_voice_ratio", 0.35))
    quality_cfg.setdefault("ng_words", app_cfg.get("ng_words", DEFAULT_GROUPED_CONFIG["quality"]["ng_words"]))

    if performance_cfg.get("compute_profile") == "cpu-safe":
        whisper_cfg.setdefault("device", "CPU")
        whisper_cfg.setdefault("compute_type", "int8")

    return {
        "project": project_cfg,
        "llm": llm_cfg,
        "pipeline": pipeline_cfg,
        "performance": performance_cfg,
        "style": style_cfg,
        "quality": quality_cfg,
    }


def normalize_config_dict(cfg: dict[str, Any]) -> dict[str, Any]:
    merged = deep_merge(copy.deepcopy(_legacy_defaults()), cfg)
    grouped = _build_grouped_from_legacy(merged)
    preset_id = grouped["project"].get("preset_id", DEFAULT_PRESET_ID)
    preset = BUILTIN_PRESETS.get(preset_id, BUILTIN_PRESETS[DEFAULT_PRESET_ID])
    normalized_grouped = deep_merge(copy.deepcopy(DEFAULT_GROUPED_CONFIG), preset.get("default_config", {}))
    normalized_grouped = deep_merge(normalized_grouped, grouped)
    _apply_language_defaults(merged, normalized_grouped)
    _apply_windows_tts_defaults(merged, normalized_grouped)

    merged["project"] = normalized_grouped["project"]
    merged["llm"] = normalized_grouped["llm"]
    merged["pipeline"] = normalized_grouped["pipeline"]
    merged["performance"] = normalized_grouped["performance"]
    merged["style"] = normalized_grouped["style"]
    merged["quality"] = normalized_grouped["quality"]

    sync_legacy_sections(merged)
    return merged


def sync_legacy_sections(cfg: dict[str, Any]):
    project_cfg = cfg["project"]
    llm_cfg = cfg["llm"]
    pipeline_cfg = cfg["pipeline"]
    perf_cfg = cfg["performance"]
    style_cfg = cfg["style"]
    quality_cfg = cfg["quality"]
    app_cfg = cfg.setdefault("app", {})
    ui_cfg = cfg.setdefault("ui", {})
    whisper_cfg = cfg.setdefault("whisper", {})

    provider = llm_cfg.get("provider", "openai")
    app_cfg["video_source"] = project_cfg.get("video_source", "pexels")
    app_cfg["preset_id"] = project_cfg.get("preset_id", DEFAULT_PRESET_ID)
    app_cfg["profile_name"] = project_cfg.get("profile_name", DEFAULT_PROFILE_NAME)
    app_cfg["video_language"] = project_cfg.get("video_language", "")
    app_cfg["llm_provider"] = provider
    app_cfg["run_mode"] = perf_cfg.get("run_mode", "stable")
    app_cfg["compute_profile"] = perf_cfg.get("compute_profile", "cpu-safe")
    app_cfg["asset_policy"] = perf_cfg.get("asset_policy", "balanced")
    app_cfg["resume_from"] = pipeline_cfg.get("resume_from", "auto")
    app_cfg["cache_enabled"] = pipeline_cfg.get("cache_enabled", True)
    app_cfg["reuse_intermediate"] = pipeline_cfg.get("reuse_intermediate", True)
    app_cfg["material_directory"] = pipeline_cfg.get("material_directory", "")
    app_cfg["subtitle_provider"] = pipeline_cfg.get("subtitle_provider", "edge")
    app_cfg["llm_timeout_sec"] = llm_cfg.get("timeout_sec", 60)
    app_cfg["llm_retry_count"] = llm_cfg.get("retry_count", 5)
    app_cfg["llm_api_key_env"] = llm_cfg.get("api_key_env", "")
    app_cfg["max_concurrent_tasks"] = perf_cfg.get("max_concurrent_tasks", 2)
    app_cfg["n_threads"] = perf_cfg.get("n_threads", 2)
    app_cfg["verbose"] = perf_cfg.get("verbose", False)
    app_cfg["preview_enabled"] = perf_cfg.get("preview_enabled", True)
    app_cfg["preview_resolution"] = perf_cfg.get("preview_resolution", "720x1280")
    app_cfg["preview_fps"] = perf_cfg.get("preview_fps", 24)
    app_cfg["enable_quality_checks"] = quality_cfg.get("enable_checks", True)
    app_cfg["max_audio_subtitle_delta_sec"] = quality_cfg.get("max_audio_subtitle_delta_sec", 1.2)
    app_cfg["max_material_duplication_rate"] = quality_cfg.get("max_material_duplication_rate", 0.35)
    app_cfg["max_bgm_to_voice_ratio"] = quality_cfg.get("max_bgm_to_voice_ratio", 0.35)
    app_cfg["ng_words"] = quality_cfg.get("ng_words", [])
    app_cfg["enable_chatterbox"] = perf_cfg.get("enable_chatterbox", False)
    app_cfg["allow_voice_clone"] = perf_cfg.get("allow_voice_clone", False)

    app_cfg[f"{provider}_base_url"] = llm_cfg.get("base_url", "")
    app_cfg[f"{provider}_model_name"] = llm_cfg.get("model", "")

    ui_cfg["tts_server"] = style_cfg.get("tts_server", default_tts_server_for_platform())
    ui_cfg["font_name"] = style_cfg.get("font_name", "MicrosoftYaHeiBold.ttc")
    ui_cfg["font_size"] = style_cfg.get("font_size", 60)
    ui_cfg["text_fore_color"] = style_cfg.get("text_fore_color", "#FFFFFF")
    ui_cfg["highlight_color"] = style_cfg.get("highlight_color", "#ff0000")
    ui_cfg["max_chars_per_line"] = style_cfg.get("max_chars_per_line", 40)
    ui_cfg["max_lines_per_subtitle"] = style_cfg.get("max_lines_per_subtitle", 2)

    if perf_cfg.get("compute_profile") == "cpu-safe":
        whisper_cfg["device"] = "CPU"
        whisper_cfg["compute_type"] = "int8"
        if ui_cfg.get("tts_server") == "chatterbox":
            ui_cfg["tts_server"] = default_tts_server_for_platform()
    else:
        whisper_cfg.setdefault("device", "cuda")
        whisper_cfg.setdefault("compute_type", "float16")


def ensure_runtime_dirs():
    _ensure_dir(presets_dir())
    _ensure_dir(profiles_dir())
    _ensure_dir(state_dir())


def ensure_default_presets():
    ensure_runtime_dirs()
    for preset_id, preset in BUILTIN_PRESETS.items():
        preset_path = os.path.join(presets_dir(), f"{preset_id}.toml")
        if not os.path.exists(preset_path):
            with open(preset_path, "w", encoding="utf-8") as f:
                f.write(toml.dumps(preset))


def load_config():
    ensure_default_presets()
    if os.path.isdir(config_file):
        shutil.rmtree(config_file)

    if not os.path.isfile(config_file):
        example_file = f"{root_dir}/config.example.toml"
        if os.path.isfile(example_file):
            shutil.copyfile(example_file, config_file)
            logger.info("copy config.example.toml to config.toml")

    logger.info(f"load config from file: {config_file}")
    raw_cfg = _load_toml(config_file)
    return normalize_config_dict(raw_cfg)


def save_config():
    global _cfg, app, whisper, proxy, azure, siliconflow, ui, project, llm, pipeline, performance, style, quality
    _cfg["app"] = app
    _cfg["whisper"] = whisper
    _cfg["proxy"] = proxy
    _cfg["azure"] = azure
    _cfg["siliconflow"] = siliconflow
    _cfg["ui"] = ui
    _cfg["project"] = project
    _cfg["llm"] = llm
    _cfg["pipeline"] = pipeline
    _cfg["performance"] = performance
    _cfg["style"] = style
    _cfg["quality"] = quality
    sync_legacy_sections(_cfg)
    with open(config_file, "w", encoding="utf-8") as f:
        f.write(toml.dumps(_cfg))


def _snapshot_config() -> dict[str, Any]:
    return {
        "project": copy.deepcopy(project),
        "llm": copy.deepcopy(llm),
        "pipeline": copy.deepcopy(pipeline),
        "performance": copy.deepcopy(performance),
        "style": copy.deepcopy(style),
        "quality": copy.deepcopy(quality),
        "ui": copy.deepcopy(ui),
        "app": {
            "hide_config": app.get("hide_config", False),
            "pexels_api_keys": copy.deepcopy(app.get("pexels_api_keys", [])),
            "pixabay_api_keys": copy.deepcopy(app.get("pixabay_api_keys", [])),
            "openai_api_key": app.get("openai_api_key", ""),
            "groq_api_key": app.get("groq_api_key", ""),
            "openrouter_api_key": app.get("openrouter_api_key", ""),
            "moonshot_api_key": app.get("moonshot_api_key", ""),
            "oneapi_api_key": app.get("oneapi_api_key", ""),
            "azure_api_key": app.get("azure_api_key", ""),
            "gemini_api_key": app.get("gemini_api_key", ""),
            "qwen_api_key": app.get("qwen_api_key", ""),
            "deepseek_api_key": app.get("deepseek_api_key", ""),
            "cloudflare_api_key": app.get("cloudflare_api_key", ""),
            "cloudflare_account_id": app.get("cloudflare_account_id", ""),
            "ernie_api_key": app.get("ernie_api_key", ""),
            "ernie_secret_key": app.get("ernie_secret_key", ""),
            "ernie_base_url": app.get("ernie_base_url", ""),
            "pollinations_api_key": app.get("pollinations_api_key", ""),
        },
        "azure": copy.deepcopy(azure),
        "siliconflow": copy.deepcopy(siliconflow),
        "whisper": copy.deepcopy(whisper),
    }


def save_profile(profile_name: str, snapshot: dict[str, Any] | None = None):
    ensure_runtime_dirs()
    name = profile_name.strip() or DEFAULT_PROFILE_NAME
    profile_path = os.path.join(profiles_dir(), f"{name}.toml")
    data = snapshot or _snapshot_config()
    data.setdefault("project", {})["profile_name"] = name
    with open(profile_path, "w", encoding="utf-8") as f:
        f.write(toml.dumps(data))
    save_last_used_profile(name)
    return profile_path


def load_profile(profile_name: str) -> dict[str, Any] | None:
    name = profile_name.strip()
    if not name:
        return None
    profile_path = os.path.join(profiles_dir(), f"{name}.toml")
    if not os.path.exists(profile_path):
        return None
    return _load_toml(profile_path)


def apply_profile(profile_name: str) -> bool:
    global _cfg, app, whisper, proxy, azure, siliconflow, ui, project, llm, pipeline, performance, style, quality
    profile_cfg = load_profile(profile_name)
    if not profile_cfg:
        return False
    _cfg = normalize_config_dict(deep_merge(copy.deepcopy(_cfg), profile_cfg))
    app = _cfg.get("app", {})
    whisper = _cfg.get("whisper", {})
    proxy = _cfg.get("proxy", {})
    azure = _cfg.get("azure", {})
    siliconflow = _cfg.get("siliconflow", {})
    ui = _cfg.get("ui", {})
    project = _cfg.get("project", {})
    llm = _cfg.get("llm", {})
    pipeline = _cfg.get("pipeline", {})
    performance = _cfg.get("performance", {})
    style = _cfg.get("style", {})
    quality = _cfg.get("quality", {})
    save_last_used_profile(project.get("profile_name", profile_name))
    save_config()
    return True


def apply_preset(preset_id: str) -> bool:
    global _cfg, app, whisper, proxy, azure, siliconflow, ui, project, llm, pipeline, performance, style, quality
    preset = BUILTIN_PRESETS.get(preset_id)
    if not preset:
        return False
    merged = copy.deepcopy(_cfg)
    grouped_override = copy.deepcopy(preset.get("default_config", {}))
    grouped_override.setdefault("project", {})["preset_id"] = preset_id
    for section, section_data in grouped_override.items():
        merged.setdefault(section, {})
        deep_merge(merged[section], section_data)
    _cfg = normalize_config_dict(merged)
    app = _cfg.get("app", {})
    whisper = _cfg.get("whisper", {})
    proxy = _cfg.get("proxy", {})
    azure = _cfg.get("azure", {})
    siliconflow = _cfg.get("siliconflow", {})
    ui = _cfg.get("ui", {})
    project = _cfg.get("project", {})
    llm = _cfg.get("llm", {})
    pipeline = _cfg.get("pipeline", {})
    performance = _cfg.get("performance", {})
    style = _cfg.get("style", {})
    quality = _cfg.get("quality", {})
    save_config()
    return True


def list_presets() -> list[dict[str, Any]]:
    ensure_default_presets()
    presets = []
    for preset_id, preset in BUILTIN_PRESETS.items():
        presets.append(
            {
                "id": preset_id,
                "label": preset.get("label", preset_id),
                "video_type": preset.get("video_type", ""),
                "visible_sections": preset.get("visible_sections", []),
            }
        )
    return presets


def list_profiles() -> list[str]:
    ensure_runtime_dirs()
    names = []
    for name in os.listdir(profiles_dir()):
        if name.endswith(".toml"):
            names.append(name[:-5])
    names.sort()
    return names


def save_last_used_profile(profile_name: str):
    ensure_runtime_dirs()
    with open(last_profile_path(), "w", encoding="utf-8") as f:
        f.write(toml.dumps({"project": {"profile_name": profile_name}}))


def get_last_used_profile() -> str:
    path = last_profile_path()
    if not os.path.exists(path):
        return ""
    data = _load_toml(path)
    return data.get("project", {}).get("profile_name", "")


def get_llm_provider_config(provider: str | None = None) -> dict[str, Any]:
    active_provider = (provider or llm.get("provider") or app.get("llm_provider", "openai")).lower()
    base_url = llm.get("base_url", "")
    model = llm.get("model", "")
    api_key_env = llm.get("api_key_env", "")
    config_data = {
        "provider": active_provider,
        "model": model or app.get(f"{active_provider}_model_name", ""),
        "base_url": base_url or app.get(f"{active_provider}_base_url", ""),
        "api_key": app.get(f"{active_provider}_api_key", ""),
        "api_key_env": api_key_env,
        "timeout_sec": llm.get("timeout_sec", 60),
        "retry_count": llm.get("retry_count", 5),
    }
    if config_data["api_key_env"]:
        config_data["api_key"] = os.getenv(config_data["api_key_env"], config_data["api_key"])
    return config_data


_cfg = load_config()
app = _cfg.get("app", {})
whisper = _cfg.get("whisper", {})
proxy = _cfg.get("proxy", {})
azure = _cfg.get("azure", {})
siliconflow = _cfg.get("siliconflow", {})
ui = _cfg.get("ui", {"hide_log": False})
project = _cfg.get("project", {})
llm = _cfg.get("llm", {})
pipeline = _cfg.get("pipeline", {})
performance = _cfg.get("performance", {})
style = _cfg.get("style", {})
quality = _cfg.get("quality", {})

hostname = socket.gethostname()

log_level = _cfg.get("log_level", "DEBUG")
listen_host = _cfg.get("listen_host", "0.0.0.0")
listen_port = _cfg.get("listen_port", 8080)
project_name = _cfg.get("project_name", "MoneyPrinterTurbo")
project_description = _cfg.get(
    "project_description",
    "<a href='https://github.com/harry0703/MoneyPrinterTurbo'>https://github.com/harry0703/MoneyPrinterTurbo</a>",
)
project_version = _cfg.get("project_version", "1.2.6")
reload_debug = False

imagemagick_path = app.get("imagemagick_path", "")
if imagemagick_path and os.path.isfile(imagemagick_path):
    os.environ["IMAGEMAGICK_BINARY"] = imagemagick_path

ffmpeg_path = app.get("ffmpeg_path", "")
if ffmpeg_path and os.path.isfile(ffmpeg_path):
    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path

logger.info(f"{project_name} v{project_version}")
