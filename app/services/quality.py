import json
import os
import re
from collections import Counter
from typing import Any

from loguru import logger

from app.config import config
from app.utils import utils


def _parse_srt_entries(subtitle_path: str) -> list[dict[str, Any]]:
    if not subtitle_path or not os.path.exists(subtitle_path):
        return []
    with open(subtitle_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    blocks = re.split(r"\n\s*\n", raw)
    entries = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        start_text, end_text = [item.strip() for item in lines[1].split("-->")]
        text_lines = lines[2:]
        entries.append(
            {
                "start": _parse_srt_time(start_text),
                "end": _parse_srt_time(end_text),
                "text_lines": text_lines,
                "text": " ".join(text_lines),
            }
        )
    return entries


def _parse_srt_time(value: str) -> float:
    hours, minutes, rest = value.split(":")
    seconds, millis = rest.split(",")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000


def _check_audio_subtitle_delta(audio_duration: float, subtitle_entries: list[dict[str, Any]]) -> dict[str, Any]:
    if not audio_duration or not subtitle_entries:
        return {"name": "audio_subtitle_delta", "level": "warning", "message": "音声または字幕が不足しているため差分確認をスキップしました。"}
    subtitle_duration = subtitle_entries[-1]["end"]
    delta = round(abs(audio_duration - subtitle_duration), 3)
    threshold = config.quality.get("max_audio_subtitle_delta_sec", 1.2)
    level = "error" if delta > threshold else "ok"
    return {
        "name": "audio_subtitle_delta",
        "level": level,
        "delta_sec": delta,
        "threshold_sec": threshold,
        "message": f"音声と字幕の差分は {delta} 秒です。",
    }


def _is_japanese_language(language: str = "") -> bool:
    normalized = (language or "").strip().lower().replace("_", "-")
    return normalized == "ja" or normalized.startswith("ja-")


def _resolve_subtitle_limits(video_language: str = "") -> tuple[int, int]:
    max_chars = config.quality.get("max_subtitle_chars_per_line", 40)
    max_lines = config.quality.get("max_subtitle_lines", 2)
    if _is_japanese_language(video_language):
        max_chars = min(max_chars, 26)
    return max_chars, max_lines


def _check_subtitle_density(
    subtitle_entries: list[dict[str, Any]], video_language: str = ""
) -> dict[str, Any]:
    max_chars, max_lines = _resolve_subtitle_limits(video_language)
    worst_chars = 0
    worst_lines = 0
    for entry in subtitle_entries:
        worst_lines = max(worst_lines, len(entry["text_lines"]))
        for line in entry["text_lines"]:
            worst_chars = max(worst_chars, len(line))

    level = "ok"
    if worst_lines > max_lines or worst_chars > max_chars:
        level = "warning"
    if worst_lines > max_lines + 1 or worst_chars > max_chars + 20:
        level = "error"
    return {
        "name": "subtitle_density",
        "level": level,
        "max_chars": worst_chars,
        "max_lines": worst_lines,
        "allowed_chars": max_chars,
        "allowed_lines": max_lines,
        "message": f"字幕の最大行数は {worst_lines}、最大文字数は {worst_chars} です。"
        + (" 日本語では短めの字幕を推奨します。" if _is_japanese_language(video_language) else ""),
    }


def _check_audio_mix(voice_volume: float, bgm_volume: float) -> dict[str, Any]:
    ratio = 0 if voice_volume <= 0 else round(bgm_volume / voice_volume, 3)
    threshold = config.quality.get("max_bgm_to_voice_ratio", 0.35)
    level = "warning" if ratio > threshold else "ok"
    return {
        "name": "audio_mix",
        "level": level,
        "ratio": ratio,
        "threshold": threshold,
        "message": f"BGM/音声比率は {ratio} です。",
    }


def _check_material_duplication(materials: list[str]) -> dict[str, Any]:
    if not materials:
        return {"name": "material_duplication", "level": "warning", "message": "素材が見つからないため重複率を確認できません。"}
    names = [os.path.basename(material) for material in materials]
    counts = Counter(names)
    duplicates = sum(count - 1 for count in counts.values() if count > 1)
    duplication_rate = round(duplicates / max(len(names), 1), 3)
    threshold = config.quality.get("max_material_duplication_rate", 0.35)
    level = "warning" if duplication_rate > threshold else "ok"
    return {
        "name": "material_duplication",
        "level": level,
        "duplication_rate": duplication_rate,
        "threshold": threshold,
        "message": f"素材重複率は {duplication_rate} です。",
    }


def _check_ng_words(video_script: str) -> dict[str, Any]:
    lowered = (video_script or "").lower()
    hits = []
    for word in config.quality.get("ng_words", []):
        if word.lower() in lowered:
            hits.append(word)
    level = "warning" if hits else "ok"
    return {
        "name": "ng_words",
        "level": level,
        "hits": hits,
        "message": "NG ワードは見つかりませんでした。" if not hits else f"NG ワード候補: {', '.join(hits)}",
    }


def run_quality_checks(task_id: str, params, audio_duration: float, subtitle_path: str, materials: list[str], video_script: str) -> dict[str, Any]:
    entries = _parse_srt_entries(subtitle_path)
    video_language = getattr(params, "video_language", "") or config.project.get("video_language", "")
    checks = [
        _check_audio_subtitle_delta(audio_duration, entries),
        _check_subtitle_density(entries, video_language),
        _check_audio_mix(getattr(params, "voice_volume", 1.0), getattr(params, "bgm_volume", 0.2)),
        _check_material_duplication(materials or []),
        _check_ng_words(video_script),
    ]
    blocking = any(check["level"] == "error" for check in checks)
    report = {
        "task_id": task_id,
        "checks": checks,
        "summary": {
            "ok": len([c for c in checks if c["level"] == "ok"]),
            "warning": len([c for c in checks if c["level"] == "warning"]),
            "error": len([c for c in checks if c["level"] == "error"]),
        },
        "blocking": blocking,
    }
    report_path = os.path.join(utils.task_dir(task_id), "quality-report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"quality report saved: {report_path}")
    return report
