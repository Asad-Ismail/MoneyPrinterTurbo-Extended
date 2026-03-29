import asyncio
import base64
import os
import re
import subprocess
from datetime import datetime
from typing import Optional, Union
from xml.sax.saxutils import unescape

# Suppress warnings and handle CUDA library conflicts
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", message=".*audio is shorter than 30s.*")
os.environ["PYANNOTE_CACHE"] = "/tmp/pyannote"
# Handle CUDA library conflicts gracefully
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"  # Reduce CUDA memory issues
# Set environment variable to handle cuDNN version mismatches
os.environ["CUDNN_LOGINFO_DBG"] = "0"  # Suppress cuDNN debug info

import edge_tts
import requests
from edge_tts import SubMaker
try:
    from edge_tts.submaker import mktimestamp
except ImportError:
    # edge_tts 7.x では mktimestamp が公開されないため、100ns 単位を SRT 時刻へ変換する
    def mktimestamp(offset):
        total_ms = max(int(round(offset / 10000)), 0)
        hours = total_ms // 3600000
        minutes = (total_ms % 3600000) // 60000
        seconds = (total_ms % 60000) // 1000
        millis = total_ms % 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"
from loguru import logger
from moviepy.video.tools import subtitles

from app.config import config
from app.utils import utils

# Import Chatterbox TTS and WhisperX if available
try:
    from chatterbox.tts import ChatterboxTTS
    import whisperx
    import torch
    import torchaudio
    CHATTERBOX_AVAILABLE = True
    logger.info("Chatterbox TTS and WhisperX are available")
except ImportError as e:
    CHATTERBOX_AVAILABLE = False
    logger.warning(f"Chatterbox TTS or WhisperX not available: {e}")

# Global Chatterbox model instance
chatterbox_model = None
whisperx_model = None
_last_tts_error = ""
WINDOWS_SAPI_PREFIX = "windows:"


def clear_last_tts_error():
    global _last_tts_error
    _last_tts_error = ""


def get_last_tts_error() -> str:
    return _last_tts_error


def _set_last_tts_error(message: str):
    global _last_tts_error
    _last_tts_error = message.strip()


def get_edge_tts_version() -> str:
    return getattr(edge_tts, "__version__", "unknown")


def is_edge_tts_outdated() -> bool:
    version = get_edge_tts_version()
    try:
        major = int(version.split(".")[0])
    except (TypeError, ValueError, AttributeError):
        return False
    return major < 7


def _resolve_edge_tts_proxy() -> Optional[str]:
    proxy_cfg = config.proxy
    if isinstance(proxy_cfg, dict):
        return proxy_cfg.get("https") or proxy_cfg.get("http") or None
    if isinstance(proxy_cfg, str) and proxy_cfg.strip():
        return proxy_cfg.strip()
    return None


def _build_edge_tts_error_message(error: Exception) -> str:
    message = str(error).strip()
    proxy = _resolve_edge_tts_proxy()
    details = [f"Edge TTS に接続できませんでした: {message}"]
    if "403" in message and "Invalid response status" in message:
        if is_edge_tts_outdated():
            details.append(
                f"現在の edge_tts バージョンは {get_edge_tts_version()} です。古い版では接続拒否されることがあります。"
            )
            details.append("edge_tts を再インストールして最新版へ更新してください。")
        else:
            details.append(
                f"現在の edge_tts バージョンは {get_edge_tts_version()} です。最新版でもネットワーク条件によって接続拒否されることがあります。"
            )
        if not proxy:
            details.append("必要なら config.toml の [proxy] に https または http プロキシを設定してください。")
        details.append("この環境では Edge の読み上げエンドポイント自体が拒否している可能性があります。")
        if os.name == "nt":
            details.append("Windows では内蔵音声へ自動フォールバックします。")
    return " ".join(details)


def _ensure_parent_dir(file_path: str):
    parent_dir = os.path.dirname(file_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)


def ensure_submaker_compatibility(sub_maker):
    """Ensure SubMaker has required attributes for compatibility with different edge_tts versions"""
    if not hasattr(sub_maker, 'subs'):
        sub_maker.subs = []
    if not hasattr(sub_maker, 'offset'):
        sub_maker.offset = []
    return sub_maker


def _append_submaker_timing(sub_maker, chunk: dict):
    if chunk["type"] not in ("WordBoundary", "SentenceBoundary"):
        return

    if hasattr(sub_maker, "feed"):
        try:
            sub_maker.feed(chunk)
        except Exception:
            logger.debug("submaker feed fallback to legacy timing list")

    sub_maker.subs.append(chunk["text"])
    sub_maker.offset.append((chunk["offset"], chunk["offset"] + chunk["duration"]))


def get_siliconflow_voices() -> list[str]:
    """
    获取硅基流动的声音列表

    Returns:
        声音列表，格式为 ["siliconflow:FunAudioLLM/CosyVoice2-0.5B:alex", ...]
    """
    # 硅基流动的声音列表和对应的性别（用于显示）
    voices_with_gender = [
        ("FunAudioLLM/CosyVoice2-0.5B", "alex", "Male"),
        ("FunAudioLLM/CosyVoice2-0.5B", "anna", "Female"),
        ("FunAudioLLM/CosyVoice2-0.5B", "bella", "Female"),
        ("FunAudioLLM/CosyVoice2-0.5B", "benjamin", "Male"),
        ("FunAudioLLM/CosyVoice2-0.5B", "charles", "Male"),
        ("FunAudioLLM/CosyVoice2-0.5B", "claire", "Female"),
        ("FunAudioLLM/CosyVoice2-0.5B", "david", "Male"),
        ("FunAudioLLM/CosyVoice2-0.5B", "diana", "Female"),
    ]

    # 添加siliconflow:前缀，并格式化为显示名称
    return [
        f"siliconflow:{model}:{voice}-{gender}"
        for model, voice, gender in voices_with_gender
    ]


def get_chatterbox_voices() -> list[str]:
    """
    获取Chatterbox TTS的声音列表

    Returns:
        声音列表，格式为 ["chatterbox:default:Default Voice", "chatterbox:clone:Voice Clone", ...]
    """
    if not CHATTERBOX_AVAILABLE:
        return []
    
    voices = [
        "chatterbox:default:Default Voice-Neutral",
        "chatterbox:clone:Voice Clone-Custom"
    ]
    
    # Add reference audio files from reference_audio directory if available
    reference_audio_dir = os.path.join(utils.root_dir(), "reference_audio")
    if os.path.exists(reference_audio_dir):
        for file in os.listdir(reference_audio_dir):
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                name = os.path.splitext(file)[0]
                voices.append(f"chatterbox:clone:{name}-Custom")
    
    return voices


def is_windows_sapi_voice(voice_name: str) -> bool:
    """Windows 内蔵音声の識別子かどうかを判定する"""
    return voice_name.startswith(WINDOWS_SAPI_PREFIX)


def _powershell_single_quote(value: str) -> str:
    return value.replace("'", "''")


def _run_powershell_script(script: str) -> subprocess.CompletedProcess:
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-EncodedCommand",
            encoded,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def _locale_hint_from_voice_name(voice_name: str) -> str:
    if is_windows_sapi_voice(voice_name):
        return ""
    match = re.match(r"^([a-z]{2}-[A-Z]{2})", parse_voice_name(voice_name))
    return match.group(1) if match else ""


def _windows_locale_keyword(locale_hint: str) -> str:
    normalized = (locale_hint or "").lower()
    if normalized.startswith("ja"):
        return "Japanese"
    if normalized.startswith("en"):
        return "English"
    if normalized.startswith("zh"):
        return "Chinese"
    if normalized.startswith("ko"):
        return "Korean"
    return ""


def _build_approximate_submaker(text: str, audio_file: str) -> SubMaker:
    """
    Windows 内蔵音声は単語境界を返さないため、文章単位で概算タイムラインを作る。
    """
    cleaned_text = _format_text(text)
    script_lines = utils.split_string_by_punctuations(cleaned_text)
    if not script_lines:
        script_lines = [cleaned_text.strip() or text.strip()]

    total_duration = max(get_audio_duration_from_file(audio_file), 0.1)
    total_offset = int(total_duration * 10000000)
    weights = [max(len(re.sub(r"\s+", "", line)), 1) for line in script_lines]
    total_weight = max(sum(weights), 1)

    sub_maker = ensure_submaker_compatibility(SubMaker())
    cursor = 0
    for index, line in enumerate(script_lines):
        if index == len(script_lines) - 1:
            end_offset = total_offset
        else:
            span = max(int(total_offset * (weights[index] / total_weight)), 1)
            end_offset = min(cursor + span, total_offset)
        if end_offset <= cursor:
            end_offset = cursor + 1
        sub_maker.subs.append(line)
        sub_maker.offset.append((cursor, end_offset))
        cursor = end_offset
    return sub_maker


def get_windows_sapi_voices() -> list[str]:
    """
    Windows の SAPI 音声一覧を返す。
    """
    if os.name != "nt":
        return []

    script = """
$ErrorActionPreference = 'Stop'
$voice = New-Object -ComObject SAPI.SpVoice
foreach ($item in @($voice.GetVoices())) {
    $desc = $item.GetDescription()
    if ($desc) {
        Write-Output $desc
    }
}
"""
    result = _run_powershell_script(script)
    if result.returncode != 0:
        logger.warning(f"failed to list Windows voices: {result.stderr.strip() or result.stdout.strip()}")
        return []

    voices = []
    for line in result.stdout.splitlines():
        desc = line.strip()
        if desc:
            voices.append(f"{WINDOWS_SAPI_PREFIX}{desc}")
    return voices


def _should_fallback_to_windows_sapi(selected_tts_server: str) -> bool:
    error = get_last_tts_error()
    if os.name != "nt":
        return False
    if selected_tts_server not in ("", "azure-tts-v1"):
        return False
    return "403" in error and "Invalid response status" in error


def windows_sapi_tts(
    text: str, voice_name: str, voice_rate: float, voice_file: str
) -> Union[SubMaker, None]:
    """
    Windows 標準の SAPI 音声で wav を生成する。
    """
    if os.name != "nt":
        _set_last_tts_error("Windows 内蔵音声は Windows でのみ利用できます。")
        return None

    text = text.strip()
    wav_file = os.path.splitext(voice_file)[0] + ".wav"
    _ensure_parent_dir(wav_file)
    exact_voice = ""
    if is_windows_sapi_voice(voice_name):
        exact_voice = voice_name[len(WINDOWS_SAPI_PREFIX) :]
    locale_hint = _locale_hint_from_voice_name(voice_name)
    locale_keyword = _windows_locale_keyword(locale_hint)
    rate = max(min(round((voice_rate - 1.0) * 10), 10), -10)

    script = f"""
$ErrorActionPreference = 'Stop'
$text = @'
{text}
'@
$outFile = '{_powershell_single_quote(os.path.abspath(wav_file))}'
$requestedVoice = '{_powershell_single_quote(exact_voice)}'
$localeKeyword = '{_powershell_single_quote(locale_keyword)}'
$voice = New-Object -ComObject SAPI.SpVoice
$voices = @($voice.GetVoices())
if ($voices.Count -eq 0) {{
    throw 'Windows の音声が見つかりません。'
}}
$selected = $null
if ($requestedVoice) {{
    $selected = $voices | Where-Object {{ $_.GetDescription() -eq $requestedVoice }} | Select-Object -First 1
}}
if (-not $selected -and $localeKeyword) {{
    $selected = $voices | Where-Object {{ $_.GetDescription() -like ('*' + $localeKeyword + '*') }} | Select-Object -First 1
}}
if (-not $selected) {{
    $selected = $voices | Select-Object -First 1
}}
$voice.Voice = $selected
$voice.Rate = {rate}
$stream = New-Object -ComObject SAPI.SpFileStream
$stream.Open($outFile, 3, $false)
$voice.AudioOutputStream = $stream
$null = $voice.Speak($text)
$stream.Close()
Write-Output ('VOICE=' + $selected.GetDescription())
"""

    result = _run_powershell_script(script)
    if result.returncode != 0 or not os.path.exists(wav_file) or os.path.getsize(wav_file) == 0:
        message = result.stderr.strip() or result.stdout.strip() or "Windows 内蔵音声での音声生成に失敗しました。"
        _set_last_tts_error(f"Windows 内蔵音声での音声生成に失敗しました: {message}")
        logger.error(get_last_tts_error())
        return None

    sub_maker = _build_approximate_submaker(text=text, audio_file=wav_file)
    sub_maker._actual_audio_file = wav_file
    clear_last_tts_error()
    selected_voice_log = result.stdout.strip()
    if selected_voice_log:
        logger.info(f"Windows SAPI voice: {selected_voice_log}")
    logger.info(f"completed with Windows SAPI, output file: {wav_file}")
    return sub_maker


def get_all_azure_voices(filter_locals=None) -> list[str]:
    azure_voices_str = """
Name: af-ZA-AdriNeural
Gender: Female

Name: af-ZA-WillemNeural
Gender: Male

Name: am-ET-AmehaNeural
Gender: Male

Name: am-ET-MekdesNeural
Gender: Female

Name: ar-AE-FatimaNeural
Gender: Female

Name: ar-AE-HamdanNeural
Gender: Male

Name: ar-BH-AliNeural
Gender: Male

Name: ar-BH-LailaNeural
Gender: Female

Name: ar-DZ-AminaNeural
Gender: Female

Name: ar-DZ-IsmaelNeural
Gender: Male

Name: ar-EG-SalmaNeural
Gender: Female

Name: ar-EG-ShakirNeural
Gender: Male

Name: ar-IQ-BasselNeural
Gender: Male

Name: ar-IQ-RanaNeural
Gender: Female

Name: ar-JO-SanaNeural
Gender: Female

Name: ar-JO-TaimNeural
Gender: Male

Name: ar-KW-FahedNeural
Gender: Male

Name: ar-KW-NouraNeural
Gender: Female

Name: ar-LB-LaylaNeural
Gender: Female

Name: ar-LB-RamiNeural
Gender: Male

Name: ar-LY-ImanNeural
Gender: Female

Name: ar-LY-OmarNeural
Gender: Male

Name: ar-MA-JamalNeural
Gender: Male

Name: ar-MA-MounaNeural
Gender: Female

Name: ar-OM-AbdullahNeural
Gender: Male

Name: ar-OM-AyshaNeural
Gender: Female

Name: ar-QA-AmalNeural
Gender: Female

Name: ar-QA-MoazNeural
Gender: Male

Name: ar-SA-HamedNeural
Gender: Male

Name: ar-SA-ZariyahNeural
Gender: Female

Name: ar-SY-AmanyNeural
Gender: Female

Name: ar-SY-LaithNeural
Gender: Male

Name: ar-TN-HediNeural
Gender: Male

Name: ar-TN-ReemNeural
Gender: Female

Name: ar-YE-MaryamNeural
Gender: Female

Name: ar-YE-SalehNeural
Gender: Male

Name: az-AZ-BabekNeural
Gender: Male

Name: az-AZ-BanuNeural
Gender: Female

Name: bg-BG-BorislavNeural
Gender: Male

Name: bg-BG-KalinaNeural
Gender: Female

Name: bn-BD-NabanitaNeural
Gender: Female

Name: bn-BD-PradeepNeural
Gender: Male

Name: bn-IN-BashkarNeural
Gender: Male

Name: bn-IN-TanishaaNeural
Gender: Female

Name: bs-BA-GoranNeural
Gender: Male

Name: bs-BA-VesnaNeural
Gender: Female

Name: ca-ES-EnricNeural
Gender: Male

Name: ca-ES-JoanaNeural
Gender: Female

Name: cs-CZ-AntoninNeural
Gender: Male

Name: cs-CZ-VlastaNeural
Gender: Female

Name: cy-GB-AledNeural
Gender: Male

Name: cy-GB-NiaNeural
Gender: Female

Name: da-DK-ChristelNeural
Gender: Female

Name: da-DK-JeppeNeural
Gender: Male

Name: de-AT-IngridNeural
Gender: Female

Name: de-AT-JonasNeural
Gender: Male

Name: de-CH-JanNeural
Gender: Male

Name: de-CH-LeniNeural
Gender: Female

Name: de-DE-AmalaNeural
Gender: Female

Name: de-DE-ConradNeural
Gender: Male

Name: de-DE-FlorianMultilingualNeural
Gender: Male

Name: de-DE-KatjaNeural
Gender: Female

Name: de-DE-KillianNeural
Gender: Male

Name: de-DE-SeraphinaMultilingualNeural
Gender: Female

Name: el-GR-AthinaNeural
Gender: Female

Name: el-GR-NestorasNeural
Gender: Male

Name: en-AU-NatashaNeural
Gender: Female

Name: en-AU-WilliamNeural
Gender: Male

Name: en-CA-ClaraNeural
Gender: Female

Name: en-CA-LiamNeural
Gender: Male

Name: en-GB-LibbyNeural
Gender: Female

Name: en-GB-MaisieNeural
Gender: Female

Name: en-GB-RyanNeural
Gender: Male

Name: en-GB-SoniaNeural
Gender: Female

Name: en-GB-ThomasNeural
Gender: Male

Name: en-HK-SamNeural
Gender: Male

Name: en-HK-YanNeural
Gender: Female

Name: en-IE-ConnorNeural
Gender: Male

Name: en-IE-EmilyNeural
Gender: Female

Name: en-IN-NeerjaExpressiveNeural
Gender: Female

Name: en-IN-NeerjaNeural
Gender: Female

Name: en-IN-PrabhatNeural
Gender: Male

Name: en-KE-AsiliaNeural
Gender: Female

Name: en-KE-ChilembaNeural
Gender: Male

Name: en-NG-AbeoNeural
Gender: Male

Name: en-NG-EzinneNeural
Gender: Female

Name: en-NZ-MitchellNeural
Gender: Male

Name: en-NZ-MollyNeural
Gender: Female

Name: en-PH-JamesNeural
Gender: Male

Name: en-PH-RosaNeural
Gender: Female

Name: en-SG-LunaNeural
Gender: Female

Name: en-SG-WayneNeural
Gender: Male

Name: en-TZ-ElimuNeural
Gender: Male

Name: en-TZ-ImaniNeural
Gender: Female

Name: en-US-AnaNeural
Gender: Female

Name: en-US-AndrewMultilingualNeural
Gender: Male

Name: en-US-AndrewNeural
Gender: Male

Name: en-US-AriaNeural
Gender: Female

Name: en-US-AvaMultilingualNeural
Gender: Female

Name: en-US-AvaNeural
Gender: Female

Name: en-US-BrianMultilingualNeural
Gender: Male

Name: en-US-BrianNeural
Gender: Male

Name: en-US-ChristopherNeural
Gender: Male

Name: en-US-EmmaMultilingualNeural
Gender: Female

Name: en-US-EmmaNeural
Gender: Female

Name: en-US-EricNeural
Gender: Male

Name: en-US-GuyNeural
Gender: Male

Name: en-US-JennyNeural
Gender: Female

Name: en-US-MichelleNeural
Gender: Female

Name: en-US-RogerNeural
Gender: Male

Name: en-US-SteffanNeural
Gender: Male

Name: en-ZA-LeahNeural
Gender: Female

Name: en-ZA-LukeNeural
Gender: Male

Name: es-AR-ElenaNeural
Gender: Female

Name: es-AR-TomasNeural
Gender: Male

Name: es-BO-MarceloNeural
Gender: Male

Name: es-BO-SofiaNeural
Gender: Female

Name: es-CL-CatalinaNeural
Gender: Female

Name: es-CL-LorenzoNeural
Gender: Male

Name: es-CO-GonzaloNeural
Gender: Male

Name: es-CO-SalomeNeural
Gender: Female

Name: es-CR-JuanNeural
Gender: Male

Name: es-CR-MariaNeural
Gender: Female

Name: es-CU-BelkysNeural
Gender: Female

Name: es-CU-ManuelNeural
Gender: Male

Name: es-DO-EmilioNeural
Gender: Male

Name: es-DO-RamonaNeural
Gender: Female

Name: es-EC-AndreaNeural
Gender: Female

Name: es-EC-LuisNeural
Gender: Male

Name: es-ES-AlvaroNeural
Gender: Male

Name: es-ES-ElviraNeural
Gender: Female

Name: es-ES-XimenaNeural
Gender: Female

Name: es-GQ-JavierNeural
Gender: Male

Name: es-GQ-TeresaNeural
Gender: Female

Name: es-GT-AndresNeural
Gender: Male

Name: es-GT-MartaNeural
Gender: Female

Name: es-HN-CarlosNeural
Gender: Male

Name: es-HN-KarlaNeural
Gender: Female

Name: es-MX-DaliaNeural
Gender: Female

Name: es-MX-JorgeNeural
Gender: Male

Name: es-NI-FedericoNeural
Gender: Male

Name: es-NI-YolandaNeural
Gender: Female

Name: es-PA-MargaritaNeural
Gender: Female

Name: es-PA-RobertoNeural
Gender: Male

Name: es-PE-AlexNeural
Gender: Male

Name: es-PE-CamilaNeural
Gender: Female

Name: es-PR-KarinaNeural
Gender: Female

Name: es-PR-VictorNeural
Gender: Male

Name: es-PY-MarioNeural
Gender: Male

Name: es-PY-TaniaNeural
Gender: Female

Name: es-SV-LorenaNeural
Gender: Female

Name: es-SV-RodrigoNeural
Gender: Male

Name: es-US-AlonsoNeural
Gender: Male

Name: es-US-PalomaNeural
Gender: Female

Name: es-UY-MateoNeural
Gender: Male

Name: es-UY-ValentinaNeural
Gender: Female

Name: es-VE-PaolaNeural
Gender: Female

Name: es-VE-SebastianNeural
Gender: Male

Name: et-EE-AnuNeural
Gender: Female

Name: et-EE-KertNeural
Gender: Male

Name: fa-IR-DilaraNeural
Gender: Female

Name: fa-IR-FaridNeural
Gender: Male

Name: fi-FI-HarriNeural
Gender: Male

Name: fi-FI-NooraNeural
Gender: Female

Name: fil-PH-AngeloNeural
Gender: Male

Name: fil-PH-BlessicaNeural
Gender: Female

Name: fr-BE-CharlineNeural
Gender: Female

Name: fr-BE-GerardNeural
Gender: Male

Name: fr-CA-AntoineNeural
Gender: Male

Name: fr-CA-JeanNeural
Gender: Male

Name: fr-CA-SylvieNeural
Gender: Female

Name: fr-CA-ThierryNeural
Gender: Male

Name: fr-CH-ArianeNeural
Gender: Female

Name: fr-CH-FabriceNeural
Gender: Male

Name: fr-FR-DeniseNeural
Gender: Female

Name: fr-FR-EloiseNeural
Gender: Female

Name: fr-FR-HenriNeural
Gender: Male

Name: fr-FR-RemyMultilingualNeural
Gender: Male

Name: fr-FR-VivienneMultilingualNeural
Gender: Female

Name: ga-IE-ColmNeural
Gender: Male

Name: ga-IE-OrlaNeural
Gender: Female

Name: gl-ES-RoiNeural
Gender: Male

Name: gl-ES-SabelaNeural
Gender: Female

Name: gu-IN-DhwaniNeural
Gender: Female

Name: gu-IN-NiranjanNeural
Gender: Male

Name: he-IL-AvriNeural
Gender: Male

Name: he-IL-HilaNeural
Gender: Female

Name: hi-IN-MadhurNeural
Gender: Male

Name: hi-IN-SwaraNeural
Gender: Female

Name: hr-HR-GabrijelaNeural
Gender: Female

Name: hr-HR-SreckoNeural
Gender: Male

Name: hu-HU-NoemiNeural
Gender: Female

Name: hu-HU-TamasNeural
Gender: Male

Name: id-ID-ArdiNeural
Gender: Male

Name: id-ID-GadisNeural
Gender: Female

Name: is-IS-GudrunNeural
Gender: Female

Name: is-IS-GunnarNeural
Gender: Male

Name: it-IT-DiegoNeural
Gender: Male

Name: it-IT-ElsaNeural
Gender: Female

Name: it-IT-GiuseppeMultilingualNeural
Gender: Male

Name: it-IT-IsabellaNeural
Gender: Female

Name: iu-Cans-CA-SiqiniqNeural
Gender: Female

Name: iu-Cans-CA-TaqqiqNeural
Gender: Male

Name: iu-Latn-CA-SiqiniqNeural
Gender: Female

Name: iu-Latn-CA-TaqqiqNeural
Gender: Male

Name: ja-JP-KeitaNeural
Gender: Male

Name: ja-JP-NanamiNeural
Gender: Female

Name: jv-ID-DimasNeural
Gender: Male

Name: jv-ID-SitiNeural
Gender: Female

Name: ka-GE-EkaNeural
Gender: Female

Name: ka-GE-GiorgiNeural
Gender: Male

Name: kk-KZ-AigulNeural
Gender: Female

Name: kk-KZ-DauletNeural
Gender: Male

Name: km-KH-PisethNeural
Gender: Male

Name: km-KH-SreymomNeural
Gender: Female

Name: kn-IN-GaganNeural
Gender: Male

Name: kn-IN-SapnaNeural
Gender: Female

Name: ko-KR-HyunsuMultilingualNeural
Gender: Male

Name: ko-KR-InJoonNeural
Gender: Male

Name: ko-KR-SunHiNeural
Gender: Female

Name: lo-LA-ChanthavongNeural
Gender: Male

Name: lo-LA-KeomanyNeural
Gender: Female

Name: lt-LT-LeonasNeural
Gender: Male

Name: lt-LT-OnaNeural
Gender: Female

Name: lv-LV-EveritaNeural
Gender: Female

Name: lv-LV-NilsNeural
Gender: Male

Name: mk-MK-AleksandarNeural
Gender: Male

Name: mk-MK-MarijaNeural
Gender: Female

Name: ml-IN-MidhunNeural
Gender: Male

Name: ml-IN-SobhanaNeural
Gender: Female

Name: mn-MN-BataaNeural
Gender: Male

Name: mn-MN-YesuiNeural
Gender: Female

Name: mr-IN-AarohiNeural
Gender: Female

Name: mr-IN-ManoharNeural
Gender: Male

Name: ms-MY-OsmanNeural
Gender: Male

Name: ms-MY-YasminNeural
Gender: Female

Name: mt-MT-GraceNeural
Gender: Female

Name: mt-MT-JosephNeural
Gender: Male

Name: my-MM-NilarNeural
Gender: Female

Name: my-MM-ThihaNeural
Gender: Male

Name: nb-NO-FinnNeural
Gender: Male

Name: nb-NO-PernilleNeural
Gender: Female

Name: ne-NP-HemkalaNeural
Gender: Female

Name: ne-NP-SagarNeural
Gender: Male

Name: nl-BE-ArnaudNeural
Gender: Male

Name: nl-BE-DenaNeural
Gender: Female

Name: nl-NL-ColetteNeural
Gender: Female

Name: nl-NL-FennaNeural
Gender: Female

Name: nl-NL-MaartenNeural
Gender: Male

Name: pl-PL-MarekNeural
Gender: Male

Name: pl-PL-ZofiaNeural
Gender: Female

Name: ps-AF-GulNawazNeural
Gender: Male

Name: ps-AF-LatifaNeural
Gender: Female

Name: pt-BR-AntonioNeural
Gender: Male

Name: pt-BR-FranciscaNeural
Gender: Female

Name: pt-BR-ThalitaMultilingualNeural
Gender: Female

Name: pt-PT-DuarteNeural
Gender: Male

Name: pt-PT-RaquelNeural
Gender: Female

Name: ro-RO-AlinaNeural
Gender: Female

Name: ro-RO-EmilNeural
Gender: Male

Name: ru-RU-DmitryNeural
Gender: Male

Name: ru-RU-SvetlanaNeural
Gender: Female

Name: si-LK-SameeraNeural
Gender: Male

Name: si-LK-ThiliniNeural
Gender: Female

Name: sk-SK-LukasNeural
Gender: Male

Name: sk-SK-ViktoriaNeural
Gender: Female

Name: sl-SI-PetraNeural
Gender: Female

Name: sl-SI-RokNeural
Gender: Male

Name: so-SO-MuuseNeural
Gender: Male

Name: so-SO-UbaxNeural
Gender: Female

Name: sq-AL-AnilaNeural
Gender: Female

Name: sq-AL-IlirNeural
Gender: Male

Name: sr-RS-NicholasNeural
Gender: Male

Name: sr-RS-SophieNeural
Gender: Female

Name: su-ID-JajangNeural
Gender: Male

Name: su-ID-TutiNeural
Gender: Female

Name: sv-SE-MattiasNeural
Gender: Male

Name: sv-SE-SofieNeural
Gender: Female

Name: sw-KE-RafikiNeural
Gender: Male

Name: sw-KE-ZuriNeural
Gender: Female

Name: sw-TZ-DaudiNeural
Gender: Male

Name: sw-TZ-RehemaNeural
Gender: Female

Name: ta-IN-PallaviNeural
Gender: Female

Name: ta-IN-ValluvarNeural
Gender: Male

Name: ta-LK-KumarNeural
Gender: Male

Name: ta-LK-SaranyaNeural
Gender: Female

Name: ta-MY-KaniNeural
Gender: Female

Name: ta-MY-SuryaNeural
Gender: Male

Name: ta-SG-AnbuNeural
Gender: Male

Name: ta-SG-VenbaNeural
Gender: Female

Name: te-IN-MohanNeural
Gender: Male

Name: te-IN-ShrutiNeural
Gender: Female

Name: th-TH-NiwatNeural
Gender: Male

Name: th-TH-PremwadeeNeural
Gender: Female

Name: tr-TR-AhmetNeural
Gender: Male

Name: tr-TR-EmelNeural
Gender: Female

Name: uk-UA-OstapNeural
Gender: Male

Name: uk-UA-PolinaNeural
Gender: Female

Name: ur-IN-GulNeural
Gender: Female

Name: ur-IN-SalmanNeural
Gender: Male

Name: ur-PK-AsadNeural
Gender: Male

Name: ur-PK-UzmaNeural
Gender: Female

Name: uz-UZ-MadinaNeural
Gender: Female

Name: uz-UZ-SardorNeural
Gender: Male

Name: vi-VN-HoaiMyNeural
Gender: Female

Name: vi-VN-NamMinhNeural
Gender: Male

Name: zh-CN-XiaoxiaoNeural
Gender: Female

Name: zh-CN-XiaoyiNeural
Gender: Female

Name: zh-CN-YunjianNeural
Gender: Male

Name: zh-CN-YunxiNeural
Gender: Male

Name: zh-CN-YunxiaNeural
Gender: Male

Name: zh-CN-YunyangNeural
Gender: Male

Name: zh-CN-liaoning-XiaobeiNeural
Gender: Female

Name: zh-CN-shaanxi-XiaoniNeural
Gender: Female

Name: zh-HK-HiuGaaiNeural
Gender: Female

Name: zh-HK-HiuMaanNeural
Gender: Female

Name: zh-HK-WanLungNeural
Gender: Male

Name: zh-TW-HsiaoChenNeural
Gender: Female

Name: zh-TW-HsiaoYuNeural
Gender: Female

Name: zh-TW-YunJheNeural
Gender: Male

Name: zu-ZA-ThandoNeural
Gender: Female

Name: zu-ZA-ThembaNeural
Gender: Male


Name: en-US-AvaMultilingualNeural-V2
Gender: Female

Name: en-US-AndrewMultilingualNeural-V2
Gender: Male

Name: en-US-EmmaMultilingualNeural-V2
Gender: Female

Name: en-US-BrianMultilingualNeural-V2
Gender: Male

Name: de-DE-FlorianMultilingualNeural-V2
Gender: Male

Name: de-DE-SeraphinaMultilingualNeural-V2
Gender: Female

Name: fr-FR-RemyMultilingualNeural-V2
Gender: Male

Name: fr-FR-VivienneMultilingualNeural-V2
Gender: Female

Name: zh-CN-XiaoxiaoMultilingualNeural-V2
Gender: Female
    """.strip()
    voices = []
    # 定义正则表达式模式，用于匹配 Name 和 Gender 行
    pattern = re.compile(r"Name:\s*(.+)\s*Gender:\s*(.+)\s*", re.MULTILINE)
    # 使用正则表达式查找所有匹配项
    matches = pattern.findall(azure_voices_str)

    for name, gender in matches:
        # 应用过滤条件
        if filter_locals and any(
            name.lower().startswith(fl.lower()) for fl in filter_locals
        ):
            voices.append(f"{name}-{gender}")
        elif not filter_locals:
            voices.append(f"{name}-{gender}")

    voices.sort()
    return voices


def parse_voice_name(name: str):
    # zh-CN-XiaoyiNeural-Female
    # zh-CN-YunxiNeural-Male
    # zh-CN-XiaoxiaoMultilingualNeural-V2-Female
    name = name.replace("-Female", "").replace("-Male", "").strip()
    return name


def is_azure_v2_voice(voice_name: str):
    voice_name = parse_voice_name(voice_name)
    if voice_name.endswith("-V2"):
        return voice_name.replace("-V2", "").strip()
    return ""


def is_siliconflow_voice(voice_name: str):
    """检查是否是硅基流动的声音"""
    return voice_name.startswith("siliconflow:")


def is_chatterbox_voice(voice_name: str):
    """检查是否是Chatterbox的声音"""
    return voice_name.startswith("chatterbox:")


def tts(
    text: str,
    voice_name: str,
    voice_rate: float,
    voice_file: str,
    voice_volume: float = 1.0,
) -> Union[SubMaker, None]:
    clear_last_tts_error()
    selected_tts_server = (
        config.style.get("tts_server")
        or config.ui.get("tts_server")
        or "azure-tts-v1"
    ).strip().lower()
    if selected_tts_server == "windows-sapi" or is_windows_sapi_voice(voice_name):
        return windows_sapi_tts(text, voice_name, voice_rate, voice_file)
    if is_azure_v2_voice(voice_name):
        return azure_tts_v2(text, voice_name, voice_file)
    elif is_siliconflow_voice(voice_name):
        # 从voice_name中提取模型和声音
        # 格式: siliconflow:model:voice-Gender
        parts = voice_name.split(":")
        if len(parts) >= 3:
            model = parts[1]
            # 移除性别后缀，例如 "alex-Male" -> "alex"
            voice_with_gender = parts[2]
            voice = voice_with_gender.split("-")[0]
            # 构建完整的voice参数，格式为 "model:voice"
            full_voice = f"{model}:{voice}"
            return siliconflow_tts(
                text, model, full_voice, voice_rate, voice_file, voice_volume
            )
        else:
            logger.error(f"Invalid siliconflow voice name format: {voice_name}")
            return None
    elif is_chatterbox_voice(voice_name):
        # Chatterbox TTS with WhisperX timestamps
        # 格式: chatterbox:type:name-Gender
        return chatterbox_tts(text, voice_name, voice_rate, voice_file, voice_volume)
    sub_maker = azure_tts_v1(text, voice_name, voice_rate, voice_file)
    if sub_maker:
        return sub_maker

    if _should_fallback_to_windows_sapi(selected_tts_server):
        logger.warning("azure-tts-v1 が 403 を返したため、Windows 内蔵音声へフォールバックします")
        fallback_sub_maker = windows_sapi_tts(text, voice_name, voice_rate, voice_file)
        if fallback_sub_maker:
            return fallback_sub_maker

    return None


def convert_rate_to_percent(rate: float) -> str:
    if rate == 1.0:
        return "+0%"
    percent = round((rate - 1.0) * 100)
    if percent > 0:
        return f"+{percent}%"
    else:
        return f"{percent}%"


def azure_tts_v1(
    text: str, voice_name: str, voice_rate: float, voice_file: str
) -> Union[SubMaker, None]:
    voice_name = parse_voice_name(voice_name)
    text = text.strip()
    rate_str = convert_rate_to_percent(voice_rate)
    _ensure_parent_dir(voice_file)
    proxy = _resolve_edge_tts_proxy()
    last_error = ""
    for i in range(3):
        try:
            logger.info(f"start, voice name: {voice_name}, try: {i + 1}")

            async def _do() -> SubMaker:
                communicate_kwargs = {"rate": rate_str}
                if proxy:
                    communicate_kwargs["proxy"] = proxy
                communicate = edge_tts.Communicate(text, voice_name, **communicate_kwargs)
                sub_maker = ensure_submaker_compatibility(edge_tts.SubMaker())
                with open(voice_file, "wb") as file:
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            file.write(chunk["data"])
                        elif chunk["type"] in ("WordBoundary", "SentenceBoundary"):
                            _append_submaker_timing(sub_maker, chunk)
                return sub_maker

            sub_maker = asyncio.run(_do())
            if not sub_maker or not sub_maker.subs:
                logger.warning("failed, sub_maker is None or sub_maker.subs is None")
                continue

            clear_last_tts_error()
            logger.info(f"completed, output file: {voice_file}")
            return sub_maker
        except Exception as e:
            last_error = _build_edge_tts_error_message(e)
            _set_last_tts_error(last_error)
            logger.error(f"failed, error: {last_error}")
    if last_error:
        _set_last_tts_error(last_error)
    return None


def siliconflow_tts(
    text: str,
    model: str,
    voice: str,
    voice_rate: float,
    voice_file: str,
    voice_volume: float = 1.0,
) -> Union[SubMaker, None]:
    """
    使用硅基流动的API生成语音

    Args:
        text: 要转换为语音的文本
        model: 模型名称，如 "FunAudioLLM/CosyVoice2-0.5B"
        voice: 声音名称，如 "FunAudioLLM/CosyVoice2-0.5B:alex"
        voice_rate: 语音速度，范围[0.25, 4.0]
        voice_file: 输出的音频文件路径
        voice_volume: 语音音量，范围[0.6, 5.0]，需要转换为硅基流动的增益范围[-10, 10]

    Returns:
        SubMaker对象或None
    """
    text = text.strip()
    _ensure_parent_dir(voice_file)
    api_key = config.siliconflow.get("api_key", "")

    if not api_key:
        logger.error("SiliconFlow API key is not set")
        return None

    # 将voice_volume转换为硅基流动的增益范围
    # 默认voice_volume为1.0，对应gain为0
    gain = voice_volume - 1.0
    # 确保gain在[-10, 10]范围内
    gain = max(-10, min(10, gain))

    url = "https://api.siliconflow.cn/v1/audio/speech"

    payload = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": "mp3",
        "sample_rate": 32000,
        "stream": False,
        "speed": voice_rate,
        "gain": gain,
    }

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for i in range(3):  # 尝试3次
        try:
            logger.info(
                f"start siliconflow tts, model: {model}, voice: {voice}, try: {i + 1}"
            )

            response = requests.post(url, json=payload, headers=headers)

            if response.status_code == 200:
                # 保存音频文件
                with open(voice_file, "wb") as f:
                    f.write(response.content)

                # 创建一个空的SubMaker对象
                sub_maker = ensure_submaker_compatibility(SubMaker())

                # 获取音频文件的实际长度
                try:
                    # 尝试使用moviepy获取音频长度
                    from moviepy import AudioFileClip

                    audio_clip = AudioFileClip(voice_file)
                    audio_duration = audio_clip.duration
                    audio_clip.close()

                    # 将音频长度转换为100纳秒单位（与edge_tts兼容）
                    audio_duration_100ns = int(audio_duration * 10000000)

                    # 使用文本分割来创建更准确的字幕
                    # 将文本按标点符号分割成句子
                    sentences = utils.split_string_by_punctuations(text)

                    if sentences:
                        # 计算每个句子的大致时长（按字符数比例分配）
                        total_chars = sum(len(s) for s in sentences)
                        char_duration = (
                            audio_duration_100ns / total_chars if total_chars > 0 else 0
                        )

                        current_offset = 0
                        for sentence in sentences:
                            if not sentence.strip():
                                continue

                            # 计算当前句子的时长
                            sentence_chars = len(sentence)
                            sentence_duration = int(sentence_chars * char_duration)

                            # 添加到SubMaker
                            sub_maker.subs.append(sentence)
                            sub_maker.offset.append(
                                (current_offset, current_offset + sentence_duration)
                            )

                            # 更新偏移量
                            current_offset += sentence_duration
                    else:
                        # 如果无法分割，则使用整个文本作为一个字幕
                        sub_maker.subs = [text]
                        sub_maker.offset = [(0, audio_duration_100ns)]

                except Exception as e:
                    logger.warning(f"Failed to create accurate subtitles: {str(e)}")
                    # 回退到简单的字幕
                    sub_maker.subs = [text]
                    # 使用音频文件的实际长度，如果无法获取，则假设为10秒
                    sub_maker.offset = [
                        (
                            0,
                            audio_duration_100ns
                            if "audio_duration_100ns" in locals()
                            else 10000000,
                        )
                    ]

                logger.success(f"siliconflow tts succeeded: {voice_file}")
                return sub_maker
            else:
                logger.error(
                    f"siliconflow tts failed with status code {response.status_code}: {response.text}"
                )
        except Exception as e:
            logger.error(f"siliconflow tts failed: {str(e)}")

    return None


def preprocess_text_for_chatterbox(text: str) -> str:
    """
    Preprocess text to improve Chatterbox TTS quality and prevent garbled audio
    
    Fixes common issues:
    - Converts numbers to words
    - Simplifies technical terms
    - Shortens complex sentences
    - Reduces punctuation variety
    - Handles contractions properly
    """
    if not text:
        return text
    
    # Clean and normalize text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Convert number ranges to words
    number_ranges = {
        r'\b150-300\b': 'one hundred fifty to three hundred',
        r'\b75-150\b': 'seventy five to one hundred fifty',
        r'\b150\b': 'one hundred fifty',
        r'\b300\b': 'three hundred',
        r'\b75\b': 'seventy five',
        r'\b6-7\b': 'six to seven',
        r'\b30\b': 'thirty',
        r'\b5\b': 'five',
        r'\b2\b': 'two'
    }
    
    for pattern, replacement in number_ranges.items():
        text = re.sub(pattern, replacement, text)
    
    # Simplify technical terms
    technical_replacements = {
        r'\bmetabolism\b': 'how your body burns calories',
        r'\bantioxidants\b': 'healthy compounds',
        r'\bquinoa\b': 'healthy grains',
        r'\bcardiovascular\b': 'heart and blood vessel',
        r'\bWorld Health Organization\b': 'health experts',
        r'\bvigorous cardio\b': 'intense exercise',
        r'\bmoderate cardio\b': 'gentle exercise',
        r'\bstrengthening activities\b': 'muscle building exercises',
        r'\bresistance bands\b': 'exercise bands',
        r'\bcomplex carbs\b': 'healthy carbs'
    }
    
    for pattern, replacement in technical_replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Reduce excessive punctuation
    text = re.sub(r'!!+', '!', text)  # Multiple exclamations
    text = re.sub(r'\?\?+', '?', text)  # Multiple questions
    text = re.sub(r'\.\.+', '.', text)  # Multiple periods
    
    # Simplify contractions for better pronunciation
    contractions = {
        r"\byou're\b": 'you are',
        r"\bdon't\b": 'do not',
        r"\blet's\b": 'let us',
        r"\bwhen's\b": 'when is',
        r"\bthat's\b": 'that is',
        r"\bit's\b": 'it is'
    }
    
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Break very long sentences at natural pause points
    # Split sentences longer than 120 characters
    sentences = re.split(r'([.!?])', text)
    processed_sentences = []
    
    for i in range(0, len(sentences), 2):
        if i + 1 < len(sentences):
            sentence = sentences[i]
            punctuation = sentences[i + 1]
            
            if len(sentence) > 120:
                # Try to split at commas or other natural breaks
                parts = sentence.split(', ')
                if len(parts) > 1:
                    # Rejoin as separate sentences
                    for j, part in enumerate(parts):
                        if j == len(parts) - 1:
                            processed_sentences.append(part + punctuation)
                        else:
                            processed_sentences.append(part.strip() + '.')
                else:
                    processed_sentences.append(sentence + punctuation)
            else:
                processed_sentences.append(sentence + punctuation)
        else:
            processed_sentences.append(sentences[i])
    
    text = ' '.join(processed_sentences)
    
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def chunk_text_for_chatterbox(text: str, max_chunk_size: int = 300) -> list:
    """
    Split text into optimal chunks for Chatterbox TTS processing
    
    Args:
        text: Input text to chunk
        max_chunk_size: Maximum characters per chunk
        
    Returns:
        List of text chunks
    """
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    sentences = re.split(r'([.!?])', text)
    current_chunk = ""
    
    for i in range(0, len(sentences), 2):
        if i + 1 < len(sentences):
            sentence = sentences[i].strip()
            punctuation = sentences[i + 1]
            full_sentence = sentence + punctuation
            
            # If adding this sentence would exceed chunk size, start new chunk
            if len(current_chunk) + len(full_sentence) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = full_sentence
            else:
                current_chunk += " " + full_sentence if current_chunk else full_sentence
        else:
            # Handle case where there's a sentence without punctuation at the end
            if sentences[i].strip():
                if len(current_chunk) + len(sentences[i]) > max_chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentences[i].strip()
                else:
                    current_chunk += " " + sentences[i].strip() if current_chunk else sentences[i].strip()
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def chatterbox_tts(
    text: str,
    voice_name: str,
    voice_rate: float,
    voice_file: str,
    voice_volume: float = 1.0,
) -> Union[SubMaker, None]:
    """
    使用Chatterbox TTS + WhisperX生成语音和精确的单词时间戳

    Args:
        text: 要转换为语音的文本
        voice_name: 声音名称，格式: "chatterbox:type:name-Gender"
        voice_rate: 语音速度（暂不支持调整）
        voice_file: 输出的音频文件路径
        voice_volume: 语音音量（暂不支持调整）

    Returns:
        SubMaker对象或None
    """
    if not CHATTERBOX_AVAILABLE:
        logger.error("Chatterbox TTS is not available. Please install chatterbox-tts and whisperx.")
        return None

    text = text.strip()
    if not text:
        logger.error("Text is empty")
        return None

    # Preprocess text to improve TTS quality
    original_text = text
    text = preprocess_text_for_chatterbox(text)
    
    # Check if text needs chunking (configurable threshold via CHATTERBOX_CHUNK_THRESHOLD)
    # Higher threshold reduces chunking frequency which can affect speech pacing
    chunk_threshold = int(os.environ.get("CHATTERBOX_CHUNK_THRESHOLD", "600"))
    if len(text) > chunk_threshold:
        logger.warning(f"Text is too long ({len(text)} chars) for single-pass Chatterbox TTS")
        logger.info("Automatically chunking text for better quality...")
        return chatterbox_tts_chunked(text, voice_name, voice_rate, voice_file, voice_volume)
    
    logger.info(f"Chatterbox TTS input: '{text[:100]}...' (original: {len(original_text)} → processed: {len(text)} chars)")

    # 解析voice_name: chatterbox:type:name-Gender
    parts = voice_name.split(":")
    if len(parts) < 3:
        logger.error(f"Invalid Chatterbox voice name format: {voice_name}")
        return None

    voice_type = parts[1]  # "default" or "clone"
    voice_info = parts[2]  # "name-Gender"
    voice_base_name = voice_info.split("-")[0]

    # 获取设备 - Use CPU by default to avoid cuDNN version conflicts
    # Set CHATTERBOX_DEVICE=cuda environment variable to force GPU usage
    force_device = os.environ.get("CHATTERBOX_DEVICE", "cpu").lower()
    if force_device == "cuda" and torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using GPU device: {device} (forced via CHATTERBOX_DEVICE)")
    else:
        device = "cpu"
        logger.info(f"Using CPU device (safe mode - set CHATTERBOX_DEVICE=cuda to use GPU)")

    global chatterbox_model, whisperx_model

    try:
        # 1. 加载Chatterbox TTS模型
        if chatterbox_model is None:
            logger.info("Loading Chatterbox TTS model...")
            try:
                chatterbox_model = ChatterboxTTS.from_pretrained(device=device)
                logger.info("Chatterbox TTS model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Chatterbox TTS model: {e}")
                if device == "cuda":
                    logger.info("Falling back to CPU mode...")
                    device = "cpu"
                    chatterbox_model = ChatterboxTTS.from_pretrained(device=device)
                    logger.info("Chatterbox TTS model loaded successfully on CPU")
                else:
                    raise

        # 2. 生成语音
        logger.info(f"Generating speech with Chatterbox TTS, type: {voice_type}")
        
        audio_prompt_path = None
        if voice_type == "clone" and voice_base_name != "Voice Clone":
            # 查找参考音频文件
            reference_audio_dir = os.path.join(utils.root_dir(), "reference_audio")
            for ext in ['.wav', '.mp3', '.flac', '.m4a']:
                potential_path = os.path.join(reference_audio_dir, voice_base_name + ext)
                if os.path.exists(potential_path):
                    audio_prompt_path = potential_path
                    break
            
            if audio_prompt_path:
                logger.info(f"Using voice cloning with reference: {audio_prompt_path}")
            else:
                logger.warning(f"Reference audio not found for {voice_base_name}, using default voice")

        # 生成语音 (with improved pacing control)
        # Lower cfg_weight for slower, more natural pacing
        # Environment variable CHATTERBOX_CFG_WEIGHT can override (default 0.2 for very slow speech)
        cfg_weight = float(os.environ.get("CHATTERBOX_CFG_WEIGHT", "0.2"))
        logger.info(f"Using cfg_weight={cfg_weight} for speech pacing control")
        
        if audio_prompt_path:
            wav = chatterbox_model.generate(text, audio_prompt_path=audio_prompt_path, cfg_weight=cfg_weight)
        else:
            wav = chatterbox_model.generate(text, cfg_weight=cfg_weight)

        # 保存为临时WAV文件
        temp_wav_file = voice_file.replace('.mp3', '_temp.wav')
        torchaudio.save(temp_wav_file, wav, 24000)

        # 3. 使用WhisperX获取精确的单词时间戳
        logger.info("Generating word timestamps with WhisperX")
        
        if whisperx_model is None:
            logger.info("Loading WhisperX model...")
            # Use appropriate compute type for CPU
            compute_type = "int8" if device == "cpu" else "float16"
            try:
                whisperx_model = whisperx.load_model("base", device, compute_type=compute_type)
                logger.info(f"WhisperX model loaded successfully on {device} with {compute_type}")
            except Exception as e:
                logger.error(f"Failed to load WhisperX model on {device}: {e}")
                if device == "cuda":
                    logger.info("Falling back to CPU for WhisperX...")
                    device = "cpu"
                    compute_type = "int8"
                    whisperx_model = whisperx.load_model("base", device, compute_type=compute_type)
                    logger.info(f"WhisperX model loaded successfully on CPU with {compute_type}")
                else:
                    raise

        # 转录音频获取单词时间戳
        audio = whisperx.load_audio(temp_wav_file)
        result = whisperx_model.transcribe(audio, batch_size=16)

        # Validate transcription result
        transcription_failed = False
        if not result or "segments" not in result or not result["segments"]:
            logger.warning("WhisperX transcription failed or returned empty result")
            logger.debug(f"WhisperX result: {result}")
            transcription_failed = True
        else:
            # Log transcribed text for validation
            transcribed_text = " ".join([segment.get("text", "") for segment in result["segments"]]).strip()
            logger.info(f"WhisperX transcribed: '{transcribed_text[:100]}...' (length: {len(transcribed_text)} chars)")
            
            # Check if transcription matches input text reasonably well
            text_similarity = len(set(text.lower().split()) & set(transcribed_text.lower().split())) / max(len(text.split()), 1)
            logger.debug(f"Text similarity score: {text_similarity:.2f}")
            
            if text_similarity < 0.3:
                logger.warning(f"Transcription seems inaccurate (similarity: {text_similarity:.2f})")
                if text_similarity < 0.1:
                    logger.error(f"Transcription quality too poor (similarity: {text_similarity:.2f}), falling back to sentence-level timing")
                    transcription_failed = True

        # 加载对齐模型 (only if transcription is good)
        if not transcription_failed:
            try:
                model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
                result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
            except Exception as e:
                logger.error(f"WhisperX alignment failed: {e}")
                transcription_failed = True

        # 4. 创建SubMaker并填充时间戳
        sub_maker = ensure_submaker_compatibility(SubMaker())
        
        # Process word-level timestamps from WhisperX alignment (only if transcription is good)
        word_count = 0
        if not transcription_failed and "segments" in result and result["segments"]:
            # Debug: Log the WhisperX result structure
            logger.debug(f"WhisperX result keys: {list(result.keys())}")
            logger.debug(f"Number of segments: {len(result['segments'])}")
            if result["segments"]:
                logger.debug(f"First segment keys: {list(result['segments'][0].keys())}")
            
            for segment in result["segments"]:
                # Check if this segment has word-level alignments
                if "words" in segment and segment["words"]:
                    for word_info in segment["words"]:
                        word = word_info.get("word", "").strip()
                        start = word_info.get("start", None)
                        end = word_info.get("end", None)
                        
                        # Skip words without proper timing or empty words
                        if word and start is not None and end is not None and start < end:
                            # 转换为100纳秒单位（与edge_tts兼容）
                            start_100ns = int(start * 10000000)
                            end_100ns = int(end * 10000000)
                            
                            sub_maker.subs.append(word)
                            sub_maker.offset.append((start_100ns, end_100ns))
                            word_count += 1
                        else:
                            logger.debug(f"Skipping invalid word: '{word}', start: {start}, end: {end}")
            
            logger.info(f"Processed {word_count} word-level timestamps from WhisperX")
        else:
            logger.warning("Skipping word-level processing due to transcription issues")
        
        # 如果没有获取到单词级时间戳，回退到句子级 (enhanced fallback)
        if not sub_maker.subs or transcription_failed:
            if transcription_failed:
                logger.info("Using sentence-level timing due to poor transcription quality")
            else:
                logger.warning("No word-level timestamps found, falling back to sentence-level")
                
            sentences = utils.split_string_by_punctuations(text)
            audio_duration = len(wav[0]) / 24000  # 采样率24000Hz
            
            if sentences:
                total_chars = sum(len(s) for s in sentences)
                char_duration = (audio_duration * 10000000) / total_chars if total_chars > 0 else 0
                
                current_offset = 0
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                    
                    sentence_chars = len(sentence)
                    sentence_duration = int(sentence_chars * char_duration)
                    
                    sub_maker.subs.append(sentence.strip())
                    sub_maker.offset.append((current_offset, current_offset + sentence_duration))
                    current_offset += sentence_duration
                    
                logger.info(f"Generated {len(sub_maker.subs)} sentence-level timestamps")
            else:
                # 最后的回退方案
                audio_duration_100ns = int(audio_duration * 10000000)
                sub_maker.subs = [text]
                sub_maker.offset = [(0, audio_duration_100ns)]
                logger.info("Using single timestamp for entire text")

        # 5. 转换音频格式为MP3（如果需要）
        final_audio_file = voice_file
        if voice_file.endswith('.mp3'):
            try:
                from moviepy import AudioFileClip
                logger.info("Converting WAV to MP3...")
                audio_clip = AudioFileClip(temp_wav_file)
                audio_clip.write_audiofile(voice_file, logger=None)  # Removed verbose parameter
                audio_clip.close()
                os.remove(temp_wav_file)  # 删除临时WAV文件
                logger.info("Audio conversion to MP3 completed")
                final_audio_file = voice_file
            except Exception as e:
                logger.warning(f"Failed to convert to MP3, keeping WAV format: {e}")
                # Keep the WAV file with original extension
                final_audio_file = voice_file.replace('.mp3', '.wav')
                os.rename(temp_wav_file, final_audio_file)
                logger.info(f"Saved as WAV: {final_audio_file}")
        else:
            final_audio_file = voice_file
            os.rename(temp_wav_file, voice_file)

        # Log subtitle information for debugging
        if sub_maker.subs:
            logger.info(f"Generated {len(sub_maker.subs)} subtitle entries")
            logger.debug(f"First few subtitle entries: {sub_maker.subs[:5]}")
            logger.debug(f"First few timing offsets: {sub_maker.offset[:5]}")
            
            # Validate subtitle timing
            total_audio_duration = len(wav[0]) / 24000
            last_subtitle_time = sub_maker.offset[-1][1] / 10000000 if sub_maker.offset else 0
            logger.info(f"Audio duration: {total_audio_duration:.2f}s, Last subtitle time: {last_subtitle_time:.2f}s")
            
            # Final quality check
            if transcription_failed:
                logger.warning("⚠️  Chatterbox TTS transcription had quality issues. Consider:")
                logger.warning("   • Using shorter, simpler text")
                logger.warning("   • Trying Azure TTS for better accuracy")
                logger.warning("   • Using CPU mode (set CHATTERBOX_DEVICE=cpu)")
        else:
            logger.warning("No subtitles generated!")

        logger.success(f"Chatterbox TTS completed with {len(sub_maker.subs)} word/sentence timestamps")
        logger.info(f"Output file: {final_audio_file}")
        
        # Store the actual file path for downstream processing
        sub_maker._actual_audio_file = final_audio_file
        sub_maker._transcription_quality_warning = transcription_failed
        
        return sub_maker

    except Exception as e:
        logger.error(f"Chatterbox TTS failed: {str(e)}")
        # 清理临时文件
        temp_wav_file = voice_file.replace('.mp3', '_temp.wav')
        if os.path.exists(temp_wav_file):
            os.remove(temp_wav_file)
        return None


def chatterbox_tts_chunked(
    text: str,
    voice_name: str,
    voice_rate: float,
    voice_file: str,
    voice_volume: float = 1.0,
) -> Union[SubMaker, None]:
    """
    Handle long texts by chunking them into smaller pieces for Chatterbox TTS
    
    This prevents garbled audio that occurs when text is too long
    """
    logger.info("🔄 Starting chunked Chatterbox TTS processing")
    
    # Split text into optimal chunks
    chunks = chunk_text_for_chatterbox(text, max_chunk_size=300)
    logger.info(f"Split text into {len(chunks)} chunks (max 300 chars each)")
    
    if len(chunks) == 1:
        # If only one chunk, use regular processing
        return chatterbox_tts(chunks[0], voice_name, voice_rate, voice_file, voice_volume)
    
    # Generate audio for each chunk
    temp_audio_files = []
    all_sub_makers = []
    cumulative_duration = 0.0
    
    try:
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
            
            # Create temporary file for this chunk
            chunk_file = voice_file.replace('.mp3', f'_chunk_{i}.mp3')
            
            # Generate TTS for this chunk
            chunk_result = chatterbox_tts(chunk, voice_name, voice_rate, chunk_file, voice_volume)
            
            if chunk_result:
                chunk_audio_file = getattr(chunk_result, '_actual_audio_file', chunk_file)
                temp_audio_files.append(chunk_audio_file)
                
                # Adjust timing offsets to account for previous chunks
                adjusted_subs = []
                adjusted_offsets = []
                
                cumulative_offset_100ns = int(cumulative_duration * 10000000)
                
                for sub, (start, end) in zip(chunk_result.subs, chunk_result.offset):
                    adjusted_subs.append(sub)
                    adjusted_offsets.append((
                        start + cumulative_offset_100ns,
                        end + cumulative_offset_100ns
                    ))
                
                # Create adjusted SubMaker for this chunk
                adjusted_sub_maker = ensure_submaker_compatibility(SubMaker())
                adjusted_sub_maker.subs = adjusted_subs
                adjusted_sub_maker.offset = adjusted_offsets
                all_sub_makers.append(adjusted_sub_maker)
                
                # Update cumulative duration
                if chunk_result.offset:
                    chunk_duration = chunk_result.offset[-1][1] / 10000000
                    cumulative_duration += chunk_duration
                    
                logger.info(f"Chunk {i+1} completed: {len(chunk_result.subs)} entries, {chunk_duration:.2f}s")
            else:
                logger.error(f"Failed to generate chunk {i+1}")
                return None
        
        # Combine all audio files
        logger.info("🎵 Combining audio chunks...")
        combined_audio = combine_audio_files(temp_audio_files, voice_file)
        
        if combined_audio:
            # Combine all SubMakers
            final_sub_maker = ensure_submaker_compatibility(SubMaker())
            final_sub_maker.subs = []
            final_sub_maker.offset = []
            
            for sub_maker in all_sub_makers:
                final_sub_maker.subs.extend(sub_maker.subs)
                final_sub_maker.offset.extend(sub_maker.offset)
            
            # Set metadata
            final_sub_maker._actual_audio_file = combined_audio
            final_sub_maker._transcription_quality_warning = any(
                getattr(sm, '_transcription_quality_warning', False) for sm in all_sub_makers
            )
            
            logger.success(f"✅ Chunked TTS completed: {len(final_sub_maker.subs)} total entries, {cumulative_duration:.2f}s")
            logger.info(f"Combined audio file: {combined_audio}")
            
            return final_sub_maker
        else:
            logger.error("Failed to combine audio chunks")
            return None
            
    finally:
        # Clean up temporary chunk files
        for temp_file in temp_audio_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    logger.debug(f"Cleaned up temporary file: {temp_file}")
                except Exception as e:
                    logger.warning(f"Could not remove temporary file {temp_file}: {e}")


def combine_audio_files(audio_files: list, output_file: str) -> str:
    """
    Combine multiple audio files into a single file
    
    Args:
        audio_files: List of audio file paths to combine
        output_file: Output file path
        
    Returns:
        Path to combined audio file or None if failed
    """
    try:
        from moviepy import AudioFileClip, concatenate_audioclips
        
        logger.info(f"Combining {len(audio_files)} audio files...")
        
        # Load all audio clips
        audio_clips = []
        for audio_file in audio_files:
            if os.path.exists(audio_file):
                clip = AudioFileClip(audio_file)
                audio_clips.append(clip)
                logger.debug(f"Loaded audio clip: {audio_file} ({clip.duration:.2f}s)")
            else:
                logger.warning(f"Audio file not found: {audio_file}")
        
        if not audio_clips:
            logger.error("No valid audio clips to combine")
            return None
        
        # Concatenate all clips
        final_audio = concatenate_audioclips(audio_clips)
        
        # Write combined audio
        final_audio.write_audiofile(output_file, logger=None)
        
        # Clean up clips
        for clip in audio_clips:
            clip.close()
        final_audio.close()
        
        logger.info(f"Successfully combined audio: {output_file} ({final_audio.duration:.2f}s)")
        return output_file
        
    except Exception as e:
        logger.error(f"Failed to combine audio files: {e}")
        return None


def azure_tts_v2(text: str, voice_name: str, voice_file: str) -> Union[SubMaker, None]:
    voice_name = is_azure_v2_voice(voice_name)
    if not voice_name:
        logger.error(f"invalid voice name: {voice_name}")
        raise ValueError(f"invalid voice name: {voice_name}")
    text = text.strip()
    _ensure_parent_dir(voice_file)

    def _format_duration_to_offset(duration) -> int:
        if isinstance(duration, str):
            time_obj = datetime.strptime(duration, "%H:%M:%S.%f")
            milliseconds = (
                (time_obj.hour * 3600000)
                + (time_obj.minute * 60000)
                + (time_obj.second * 1000)
                + (time_obj.microsecond // 1000)
            )
            return milliseconds * 10000

        if isinstance(duration, int):
            return duration

        return 0

    for i in range(3):
        try:
            logger.info(f"start, voice name: {voice_name}, try: {i + 1}")

            import azure.cognitiveservices.speech as speechsdk

            sub_maker = ensure_submaker_compatibility(SubMaker())

            def speech_synthesizer_word_boundary_cb(evt: speechsdk.SessionEventArgs):
                # print('WordBoundary event:')
                # print('\tBoundaryType: {}'.format(evt.boundary_type))
                # print('\tAudioOffset: {}ms'.format((evt.audio_offset + 5000)))
                # print('\tDuration: {}'.format(evt.duration))
                # print('\tText: {}'.format(evt.text))
                # print('\tTextOffset: {}'.format(evt.text_offset))
                # print('\tWordLength: {}'.format(evt.word_length))

                duration = _format_duration_to_offset(str(evt.duration))
                offset = _format_duration_to_offset(evt.audio_offset)
                sub_maker.subs.append(evt.text)
                sub_maker.offset.append((offset, offset + duration))

            # Creates an instance of a speech config with specified subscription key and service region.
            speech_key = config.azure.get("speech_key", "")
            service_region = config.azure.get("speech_region", "")
            if not speech_key or not service_region:
                logger.error("Azure speech key or region is not set")
                return None

            audio_config = speechsdk.audio.AudioOutputConfig(
                filename=voice_file, use_default_speaker=True
            )
            speech_config = speechsdk.SpeechConfig(
                subscription=speech_key, region=service_region
            )
            speech_config.speech_synthesis_voice_name = voice_name
            # speech_config.set_property(property_id=speechsdk.PropertyId.SpeechServiceResponse_RequestSentenceBoundary,
            #                            value='true')
            speech_config.set_property(
                property_id=speechsdk.PropertyId.SpeechServiceResponse_RequestWordBoundary,
                value="true",
            )

            speech_config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Audio48Khz192KBitRateMonoMp3
            )
            speech_synthesizer = speechsdk.SpeechSynthesizer(
                audio_config=audio_config, speech_config=speech_config
            )
            speech_synthesizer.synthesis_word_boundary.connect(
                speech_synthesizer_word_boundary_cb
            )

            result = speech_synthesizer.speak_text_async(text).get()
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.success(f"azure v2 speech synthesis succeeded: {voice_file}")
                return sub_maker
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                logger.error(
                    f"azure v2 speech synthesis canceled: {cancellation_details.reason}"
                )
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    logger.error(
                        f"azure v2 speech synthesis error: {cancellation_details.error_details}"
                    )
            logger.info(f"completed, output file: {voice_file}")
        except Exception as e:
            logger.error(f"failed, error: {str(e)}")
    return None


def _format_text(text: str) -> str:
    # text = text.replace("\n", " ")
    text = text.replace("[", " ")
    text = text.replace("]", " ")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace("{", " ")
    text = text.replace("}", " ")
    text = text.strip()
    return text


def create_chatterbox_subtitle(sub_maker: SubMaker, text: str, subtitle_file: str):
    """
    Create subtitle file optimized for Chatterbox TTS timestamps
    Handles both word-level and sentence-level timestamps intelligently
    """
    if not sub_maker.subs or not sub_maker.offset:
        logger.warning("No subtitle data available")
        return

    def mktimestamp(seconds: float) -> str:
        """Convert seconds to SRT timestamp format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

    def formatter(idx: int, start_time: float, end_time: float, sub_text: str) -> str:
        """Format subtitle entry for SRT file"""
        start_t = mktimestamp(start_time)
        end_t = mktimestamp(end_time)
        return f"{idx}\n{start_t} --> {end_t}\n{sub_text}\n"

    try:
        subtitle_entries = []
        subtitle_index = 1

        # Detect if we have word-level or sentence-level timestamps
        avg_sub_length = sum(len(sub) for sub in sub_maker.subs) / len(sub_maker.subs)
        is_word_level = avg_sub_length < 15  # Average word length is typically < 15 chars
        
        if is_word_level:
            logger.info("Processing word-level timestamps for subtitle grouping")
            current_phrase = []
            current_start_time = None
            current_end_time = None

            # Group words into phrases
            for i, (word, (start_100ns, end_100ns)) in enumerate(zip(sub_maker.subs, sub_maker.offset)):
                start_time = start_100ns / 10000000  # Convert to seconds
                end_time = end_100ns / 10000000

                if current_start_time is None:
                    current_start_time = start_time

                current_phrase.append(word)
                current_end_time = end_time

                # End phrase on punctuation or every 8-10 words
                is_punctuation_end = word.rstrip().endswith(('.', '!', '?', '。', '！', '？'))
                is_comma_pause = word.rstrip().endswith((',', '，'))
                is_long_phrase = len(current_phrase) >= 8
                is_last_word = i == len(sub_maker.subs) - 1

                if is_punctuation_end or is_long_phrase or is_last_word or (is_comma_pause and len(current_phrase) >= 4):
                    if current_phrase:
                        phrase_text = ' '.join(current_phrase).strip()
                        # Clean up spacing around punctuation
                        phrase_text = re.sub(r'\s+([,.!?。，！？])', r'\1', phrase_text)
                        
                        subtitle_entry = formatter(
                            idx=subtitle_index,
                            start_time=current_start_time,
                            end_time=current_end_time,
                            sub_text=phrase_text
                        )
                        subtitle_entries.append(subtitle_entry)
                        subtitle_index += 1

                    # Reset for next phrase
                    current_phrase = []
                    current_start_time = None
        else:
            logger.info("Processing sentence-level timestamps directly")
            # Use sentence-level timestamps as-is
            for i, (sentence, (start_100ns, end_100ns)) in enumerate(zip(sub_maker.subs, sub_maker.offset)):
                start_time = start_100ns / 10000000
                end_time = end_100ns / 10000000
                
                subtitle_entry = formatter(
                    idx=subtitle_index,
                    start_time=start_time,
                    end_time=end_time,
                    sub_text=sentence.strip()
                )
                subtitle_entries.append(subtitle_entry)
                subtitle_index += 1

        # Write subtitle file
        if subtitle_entries:
            with open(subtitle_file, "w", encoding="utf-8") as file:
                file.write("\n".join(subtitle_entries))
            
            logger.success(f"Chatterbox subtitle file created: {subtitle_file} with {len(subtitle_entries)} entries")
        else:
            logger.warning("No subtitle entries created")

    except Exception as e:
        logger.error(f"Failed to create Chatterbox subtitle: {str(e)}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")


def create_subtitle(sub_maker: SubMaker, text: str, subtitle_file: str):
    """
    优化字幕文件
    1. 将字幕文件按照标点符号分割成多行
    2. 逐行匹配字幕文件中的文本
    3. 生成新的字幕文件
    
    Note: This function is optimized for Azure TTS phrase-level chunks.
    For Chatterbox TTS word-level timestamps, use create_chatterbox_subtitle instead.
    """

    text = _format_text(text)

    def formatter(idx: int, start_time: float, end_time: float, sub_text: str) -> str:
        """
        1
        00:00:00,000 --> 00:00:02,360
        跑步是一项简单易行的运动
        """
        start_t = mktimestamp(start_time).replace(".", ",")
        end_t = mktimestamp(end_time).replace(".", ",")
        return f"{idx}\n{start_t} --> {end_t}\n{sub_text}\n"

    start_time = -1.0
    sub_items = []
    sub_index = 0

    script_lines = utils.split_string_by_punctuations(text)

    def match_line(_sub_line: str, _sub_index: int):
        if len(script_lines) <= _sub_index:
            return ""

        _line = script_lines[_sub_index]
        if _sub_line == _line:
            return script_lines[_sub_index].strip()

        _sub_line_ = re.sub(r"[^\w\s]", "", _sub_line)
        _line_ = re.sub(r"[^\w\s]", "", _line)
        if _sub_line_ == _line_:
            return _line_.strip()

        _sub_line_ = re.sub(r"\W+", "", _sub_line)
        _line_ = re.sub(r"\W+", "", _line)
        if _sub_line_ == _line_:
            return _line.strip()

        return ""

    sub_line = ""

    try:
        for _, (offset, sub) in enumerate(zip(sub_maker.offset, sub_maker.subs)):
            _start_time, end_time = offset
            if start_time < 0:
                start_time = _start_time

            sub = unescape(sub)
            sub_line += sub
            sub_text = match_line(sub_line, sub_index)
            if sub_text:
                sub_index += 1
                line = formatter(
                    idx=sub_index,
                    start_time=start_time,
                    end_time=end_time,
                    sub_text=sub_text,
                )
                sub_items.append(line)
                start_time = -1.0
                sub_line = ""

        if len(sub_items) == len(script_lines):
            with open(subtitle_file, "w", encoding="utf-8") as file:
                file.write("\n".join(sub_items) + "\n")
            try:
                sbs = subtitles.file_to_subtitles(subtitle_file, encoding="utf-8")
                duration = max([tb for ((ta, tb), txt) in sbs])
                logger.info(
                    f"completed, subtitle file created: {subtitle_file}, duration: {duration}"
                )
            except Exception as e:
                logger.error(f"failed, error: {str(e)}")
                os.remove(subtitle_file)
        else:
            logger.warning(
                f"failed, sub_items len: {len(sub_items)}, script_lines len: {len(script_lines)}"
            )
            fallback_srt = ""
            if hasattr(sub_maker, "get_srt"):
                try:
                    fallback_srt = sub_maker.get_srt()
                except Exception as fallback_error:
                    logger.error(f"failed to build fallback srt: {fallback_error}")
            if fallback_srt:
                with open(subtitle_file, "w", encoding="utf-8") as file:
                    file.write(fallback_srt)
                logger.info(f"completed, subtitle file created with fallback srt: {subtitle_file}")

    except Exception as e:
        logger.error(f"failed, error: {str(e)}")


def get_audio_duration(sub_maker: SubMaker):
    """
    获取音频时长
    """
    if not sub_maker.offset:
        return 0.0
    return sub_maker.offset[-1][1] / 10000000


def get_audio_duration_from_file(audio_file: str) -> float:
    if not os.path.exists(audio_file):
        return 0.0
    try:
        from moviepy import AudioFileClip

        clip = AudioFileClip(audio_file)
        duration = clip.duration or 0.0
        clip.close()
        return duration
    except Exception as e:
        logger.warning(f"failed to read audio duration from file: {e}")
        return 0.0


# Note: This module contains TTS functions for Azure TTS V1/V2, SiliconFlow TTS, and Chatterbox TTS
# All functions are optimized for production use with proper error handling and fallbacks
