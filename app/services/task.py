import json
import math
import os
import os.path
import re
from os import path

from loguru import logger

from app.config import config
from app.models import const
from app.models.schema import VideoConcatMode, VideoParams
from app.services import llm, material, quality, subtitle, video, voice
from app.services import state as sm
from app.utils import utils


STAGE_ORDER = ["script", "audio", "subtitle", "assets", "compose", "quality_check"]


def _artifact_dir(task_id: str) -> str:
    return path.join(utils.task_dir(task_id), "artifacts")


def _artifact_path(task_id: str, stage: str) -> str:
    artifact_dir = _artifact_dir(task_id)
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir, exist_ok=True)
    return path.join(artifact_dir, f"{stage}.json")


def _save_stage_artifact(task_id: str, stage: str, status: str, metadata: dict | None = None):
    with open(_artifact_path(task_id, stage), "w", encoding="utf-8") as f:
        json.dump({"stage": stage, "status": status, "metadata": metadata or {}}, f, ensure_ascii=False, indent=2)


def _load_stage_artifact(task_id: str, stage: str) -> dict:
    artifact_path = _artifact_path(task_id, stage)
    if not os.path.exists(artifact_path):
        return {}
    with open(artifact_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _should_resume_from(params: VideoParams, stage: str) -> bool:
    if not getattr(params, "reuse_intermediate", False):
        return False
    resume_from = (getattr(params, "resume_from", "auto") or "auto").lower()
    if resume_from == "auto":
        return True
    if resume_from not in STAGE_ORDER:
        return False
    return STAGE_ORDER.index(stage) < STAGE_ORDER.index(resume_from)


def _load_script_bundle(task_id: str):
    script_file = path.join(utils.task_dir(task_id), "script.json")
    if not os.path.exists(script_file):
        return None, None
    with open(script_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("script"), data.get("search_terms")


def _get_audio_duration_from_file(audio_file: str) -> int:
    return math.ceil(voice.get_audio_duration_from_file(audio_file))


def generate_script(task_id, params):
    logger.info("\n\n## generating video script")
    if _should_resume_from(params, "script"):
        cached_script, _ = _load_script_bundle(task_id)
        if cached_script:
            logger.info("reuse cached script from previous run")
            return cached_script
    video_script = params.video_script.strip()
    if not video_script:
        video_script = llm.generate_script(
            video_subject=params.video_subject,
            language=params.video_language,
            paragraph_number=params.paragraph_number,
        )
    else:
        logger.debug(f"video script: \n{video_script}")

    if not video_script:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        logger.error("failed to generate video script.")
        return None

    _save_stage_artifact(task_id, "script", "completed", {"script_file": path.join(utils.task_dir(task_id), "script.json")})
    return video_script


def generate_terms(task_id, params, video_script):
    logger.info("\n\n## generating video terms")
    if _should_resume_from(params, "audio"):
        _, cached_terms = _load_script_bundle(task_id)
        if cached_terms:
            logger.info("reuse cached search terms from previous run")
            return cached_terms
    video_terms = params.video_terms
    if not video_terms:
        video_terms = llm.generate_terms(
            video_subject=params.video_subject, video_script=video_script, amount=5
        )
    else:
        if isinstance(video_terms, str):
            video_terms = [term.strip() for term in re.split(r"[,，]", video_terms)]
        elif isinstance(video_terms, list):
            video_terms = [term.strip() for term in video_terms]
        else:
            raise ValueError("video_terms must be a string or a list of strings.")

        logger.debug(f"video terms: {utils.to_json(video_terms)}")

    if not video_terms:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        logger.error("failed to generate video terms.")
        return None

    return video_terms


def save_script_data(task_id, video_script, video_terms, params):
    script_file = path.join(utils.task_dir(task_id), "script.json")
    script_data = {
        "script": video_script,
        "search_terms": video_terms,
        "params": params,
    }

    with open(script_file, "w", encoding="utf-8") as f:
        f.write(utils.to_json(script_data))
    _save_stage_artifact(task_id, "script", "completed", {"script_file": script_file, "search_terms": video_terms})


def generate_audio(task_id, params, video_script):
    logger.info("\n\n## generating audio")
    audio_file = path.join(utils.task_dir(task_id), "audio.mp3")
    if _should_resume_from(params, "audio") and os.path.exists(audio_file):
        logger.info("reuse cached audio from previous run")
        return audio_file, _get_audio_duration_from_file(audio_file), None
    sub_maker = voice.tts(
        text=video_script,
        voice_name=voice.parse_voice_name(params.voice_name),
        voice_rate=params.voice_rate,
        voice_file=audio_file,
    )
    if sub_maker is None:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        logger.error(
            """failed to generate audio:
1. check if the language of the voice matches the language of the video script.
2. check if the network is available. If you are in China, it is recommended to use a VPN and enable the global traffic mode.
        """.strip()
        )
        return None, None, None

    # Get the actual audio file path (might be .wav if MP3 conversion failed)
    actual_audio_file = getattr(sub_maker, '_actual_audio_file', audio_file)
    if actual_audio_file != audio_file:
        logger.info(f"Audio file saved as: {actual_audio_file} (instead of {audio_file})")
        audio_file = actual_audio_file

    audio_duration = math.ceil(voice.get_audio_duration(sub_maker))
    _save_stage_artifact(task_id, "audio", "completed", {"audio_file": audio_file, "audio_duration": audio_duration})
    return audio_file, audio_duration, sub_maker


def generate_subtitle(task_id, params, video_script, sub_maker, audio_file):
    if not params.subtitle_enabled:
        return ""

    subtitle_path = path.join(utils.task_dir(task_id), "subtitle.srt")
    if _should_resume_from(params, "subtitle") and os.path.exists(subtitle_path):
        logger.info("reuse cached subtitle from previous run")
        return subtitle_path
    subtitle_provider = config.app.get("subtitle_provider", "edge").strip().lower()
    logger.info(f"\n\n## generating subtitle, provider: {subtitle_provider}")

    # Check if Chatterbox TTS was used by examining the voice name
    is_chatterbox = voice.is_chatterbox_voice(params.voice_name)
    
    subtitle_fallback = False
    if subtitle_provider == "edge":
        if is_chatterbox and sub_maker and sub_maker.subs:
            # Use specialized Chatterbox subtitle function for word-level timestamps
            logger.info("Using Chatterbox-optimized subtitle generation")
            voice.create_chatterbox_subtitle(
                sub_maker=sub_maker, text=video_script, subtitle_file=subtitle_path
            )
        else:
            # Use standard subtitle function for Azure TTS
            voice.create_subtitle(
                text=video_script, sub_maker=sub_maker, subtitle_file=subtitle_path
            )
        
        if not os.path.exists(subtitle_path):
            subtitle_fallback = True
            logger.warning("subtitle file not found, fallback to whisper")

    if subtitle_provider == "whisper" or subtitle_fallback:
        subtitle.create(audio_file=audio_file, subtitle_file=subtitle_path)
        logger.info("\n\n## correcting subtitle")
        subtitle.correct(subtitle_file=subtitle_path, video_script=video_script)

    # Generate enhanced subtitles if word highlighting is enabled
    if getattr(params, 'enable_word_highlighting', False):
        logger.info("\n\n## generating enhanced subtitles for word highlighting")
        enhanced_subtitle_path = path.join(utils.task_dir(task_id), "subtitle_enhanced.json")
        enhanced_subtitles = subtitle.create_enhanced_subtitles(
            audio_file=audio_file, 
            subtitle_file=enhanced_subtitle_path,
            params=params
        )
        if enhanced_subtitles:
            # Store both paths for later use
            params._enhanced_subtitle_path = enhanced_subtitle_path
            logger.info(f"enhanced subtitles created: {enhanced_subtitle_path}")

    subtitle_lines = subtitle.file_to_subtitles(subtitle_path)
    if not subtitle_lines:
        logger.warning(f"subtitle file is invalid: {subtitle_path}")
        return ""

    _save_stage_artifact(task_id, "subtitle", "completed", {"subtitle_path": subtitle_path})
    return subtitle_path


def get_video_materials(task_id, params, video_terms, audio_duration):
    if _should_resume_from(params, "assets"):
        artifact = _load_stage_artifact(task_id, "assets")
        cached_materials = artifact.get("metadata", {}).get("materials", [])
        if cached_materials and all(os.path.exists(item) or item.startswith("http") for item in cached_materials):
            logger.info("reuse cached materials from previous run")
            return cached_materials
    if params.video_source == "local":
        logger.info("\n\n## preprocess local materials")
        materials = video.preprocess_video(
            materials=params.video_materials, clip_duration=params.video_clip_duration
        )
        if not materials:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            logger.error(
                "no valid materials found, please check the materials and try again."
            )
            return None
        material_urls = [material_info.url for material_info in materials]
        _save_stage_artifact(task_id, "assets", "completed", {"materials": material_urls})
        return material_urls
    else:
        logger.info(f"\n\n## downloading videos from {params.video_source}")
        downloaded_videos = material.download_videos(
            task_id=task_id,
            search_terms=video_terms,
            source=params.video_source,
            video_aspect=params.video_aspect,
            video_contact_mode=params.video_concat_mode,
            audio_duration=audio_duration * params.video_count,
            max_clip_duration=params.video_clip_duration,
        )
        if not downloaded_videos:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            logger.error(
                "failed to download videos, maybe the network is not available. if you are in China, please use a VPN."
            )
            return None
        _save_stage_artifact(task_id, "assets", "completed", {"materials": downloaded_videos})
        return downloaded_videos


def generate_final_videos(
    task_id, params, downloaded_videos, audio_file, subtitle_path, video_script=""
):
    final_video_paths = []
    combined_video_paths = []

    if _should_resume_from(params, "compose"):
        artifact = _load_stage_artifact(task_id, "compose")
        cached_videos = artifact.get("metadata", {}).get("videos", [])
        cached_combined = artifact.get("metadata", {}).get("combined_videos", [])
        if cached_videos and all(path.exists(video_path) for video_path in cached_videos):
            logger.info("reuse cached composed videos from previous run")
            return cached_videos, cached_combined
    
    # Force random mode for multiple videos to ensure variety
    # Semantic mode would produce identical videos, which doesn't make sense for multiple generation
    video_concat_mode = params.video_concat_mode
    if params.video_count > 1 and video_concat_mode.value == "semantic":
        logger.info(f"🔄 Multiple videos requested ({params.video_count}), forcing random concatenation mode for variety")
        logger.info("   ℹ️  Semantic mode would produce identical videos, which is not useful for multiple generation")
        video_concat_mode = VideoConcatMode.random
    
    video_transition_mode = params.video_transition_mode

    _progress = 50
    for i in range(params.video_count):
        index = i + 1
        combined_video_path = path.join(
            utils.task_dir(task_id), f"combined-{index}.mp4"
        )
        logger.info(f"\n\n## combining video: {index} => {combined_video_path}")
        video.combine_videos(
            combined_video_path=combined_video_path,
            video_paths=downloaded_videos,
            audio_file=audio_file,
            video_aspect=params.video_aspect,
            video_concat_mode=video_concat_mode,
            video_transition_mode=video_transition_mode,
            max_clip_duration=params.video_clip_duration,
            threads=params.n_threads,
            script=video_script,
            params=params,
        )

        _progress += 50 / params.video_count / 2
        sm.state.update_task(task_id, progress=_progress)

        final_video_path = path.join(utils.task_dir(task_id), f"final-{index}.mp4")

        logger.info(f"\n\n## generating video: {index} => {final_video_path}")
        video.generate_video(
            video_path=combined_video_path,
            audio_path=audio_file,
            subtitle_path=subtitle_path,
            output_file=final_video_path,
            params=params,
        )

        _progress += 50 / params.video_count / 2
        sm.state.update_task(task_id, progress=_progress)

        final_video_paths.append(final_video_path)
        combined_video_paths.append(combined_video_path)

    _save_stage_artifact(
        task_id,
        "compose",
        "completed",
        {"videos": final_video_paths, "combined_videos": combined_video_paths},
    )
    return final_video_paths, combined_video_paths


def start(task_id, params: VideoParams, stop_at: str = "video"):
    logger.info(f"start task: {task_id}, stop_at: {stop_at}")
    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=5)

    if type(params.video_concat_mode) is str:
        params.video_concat_mode = VideoConcatMode(params.video_concat_mode)
    if not getattr(params, "compute_profile", ""):
        params.compute_profile = config.performance.get("compute_profile", "cpu-safe")
    if not getattr(params, "run_mode", ""):
        params.run_mode = config.performance.get("run_mode", "stable")
    if not getattr(params, "resume_from", ""):
        params.resume_from = config.pipeline.get("resume_from", "auto")
    if getattr(params, "n_threads", None) is None:
        params.n_threads = config.performance.get("n_threads", 2)

    # 1. Generate script
    video_script = generate_script(task_id, params)
    if not video_script or "Error: " in video_script:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        return

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=10)

    if stop_at == "script":
        sm.state.update_task(
            task_id, state=const.TASK_STATE_COMPLETE, progress=100, script=video_script
        )
        return {"script": video_script}

    # 2. Generate terms
    video_terms = ""
    if params.video_source != "local":
        video_terms = generate_terms(task_id, params, video_script)
        if not video_terms:
            sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
            return

    save_script_data(task_id, video_script, video_terms, params)

    if stop_at == "terms":
        sm.state.update_task(
            task_id, state=const.TASK_STATE_COMPLETE, progress=100, terms=video_terms
        )
        return {"script": video_script, "terms": video_terms}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=20)

    # 3. Generate audio
    audio_file, audio_duration, sub_maker = generate_audio(
        task_id, params, video_script
    )
    if not audio_file:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        return

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=30)

    if stop_at == "audio":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            audio_file=audio_file,
        )
        return {"audio_file": audio_file, "audio_duration": audio_duration}

    # 4. Generate subtitle
    subtitle_path = generate_subtitle(
        task_id, params, video_script, sub_maker, audio_file
    )

    if stop_at == "subtitle":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            subtitle_path=subtitle_path,
        )
        return {"subtitle_path": subtitle_path}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=40)

    # 5. Get video materials
    downloaded_videos = get_video_materials(
        task_id, params, video_terms, audio_duration
    )
    if not downloaded_videos:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        return

    if stop_at == "materials":
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            materials=downloaded_videos,
        )
        return {"materials": downloaded_videos}

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=50)

    # 6. Generate final videos
    final_video_paths, combined_video_paths = generate_final_videos(
        task_id, params, downloaded_videos, audio_file, subtitle_path, video_script
    )

    if not final_video_paths:
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED)
        return

    logger.success(
        f"task {task_id} finished, generated {len(final_video_paths)} videos."
    )
    quality_report = quality.run_quality_checks(
        task_id=task_id,
        params=params,
        audio_duration=audio_duration,
        subtitle_path=subtitle_path,
        materials=downloaded_videos,
        video_script=video_script,
    )
    _save_stage_artifact(task_id, "quality_check", "completed", quality_report)

    kwargs = {
        "videos": final_video_paths,
        "combined_videos": combined_video_paths,
        "script": video_script,
        "terms": video_terms,
        "audio_file": audio_file,
        "audio_duration": audio_duration,
        "subtitle_path": subtitle_path,
        "materials": downloaded_videos,
        "quality_report": quality_report,
    }
    if quality_report.get("blocking"):
        sm.state.update_task(task_id, state=const.TASK_STATE_FAILED, progress=100, **kwargs)
        logger.error("quality checks failed with blocking errors")
        return kwargs
    sm.state.update_task(
        task_id, state=const.TASK_STATE_COMPLETE, progress=100, **kwargs
    )
    return kwargs


if __name__ == "__main__":
    task_id = "task_id"
    params = VideoParams(
        video_subject="金钱的作用",
        voice_name="zh-CN-XiaoyiNeural-Female",
        voice_rate=1.0,
    )
    start(task_id, params, stop_at="video")
