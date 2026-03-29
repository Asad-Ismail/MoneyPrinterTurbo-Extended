import asyncio
import unittest
import os
import sys
from pathlib import Path
from unittest.mock import patch

# add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.utils import utils
from app.services import voice as vs
from app.config import config

temp_dir = utils.storage_dir("temp")

text_en = """
What is the meaning of life? 
This question has puzzled philosophers, scientists, and thinkers of all kinds for centuries. 
Throughout history, various cultures and individuals have come up with their interpretations and beliefs around the purpose of life. 
Some say it's to seek happiness and self-fulfillment, while others believe it's about contributing to the welfare of others and making a positive impact in the world. 
Despite the myriad of perspectives, one thing remains clear: the meaning of life is a deeply personal concept that varies from one person to another. 
It's an existential inquiry that encourages us to reflect on our values, desires, and the essence of our existence.
"""

text_zh = """
预计未来3天深圳冷空气活动频繁，未来两天持续阴天有小雨，出门带好雨具；
10-11日持续阴天有小雨，日温差小，气温在13-17℃之间，体感阴凉；
12日天气短暂好转，早晚清凉；
"""

voice_rate=1.0
voice_volume=1.0
RUN_LIVE_TTS_TESTS = os.getenv("RUN_LIVE_TTS_TESTS", "").lower() in {"1", "true", "yes"}
                    
class TestVoiceService(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self._original_proxy = config.proxy
        vs.clear_last_tts_error()
    
    def tearDown(self):
        self.loop.close()
        config.proxy = self._original_proxy
        vs.clear_last_tts_error()

    def test_azure_tts_v1_uses_proxy_from_config(self):
        captured = {}
        config.proxy = {"https": "http://127.0.0.1:7890"}

        class FakeCommunicate:
            def __init__(self, text, voice, **kwargs):
                captured["text"] = text
                captured["voice"] = voice
                captured["kwargs"] = kwargs

            async def stream(self):
                yield {"type": "audio", "data": b"fake-audio"}
                yield {"type": "WordBoundary", "text": "hello", "offset": 0, "duration": 1}

        voice_name = "zh-CN-XiaoyiNeural-Female"
        voice_file = f"{temp_dir}/tts-proxy-test.mp3"
        if os.path.exists(voice_file):
            os.remove(voice_file)

        with patch.object(vs.edge_tts, "Communicate", FakeCommunicate):
            sub_maker = vs.azure_tts_v1(
                text="hello world",
                voice_name=voice_name,
                voice_file=voice_file,
                voice_rate=voice_rate,
            )

        self.assertIsNotNone(sub_maker)
        self.assertEqual(captured["kwargs"]["proxy"], "http://127.0.0.1:7890")
        self.assertTrue(os.path.exists(voice_file))
        os.remove(voice_file)

    def test_azure_tts_v1_records_diagnostic_on_403(self):
        class FakeCommunicate:
            def __init__(self, *args, **kwargs):
                pass

            async def stream(self):
                raise Exception("403, message='Invalid response status'")
                yield

        with patch.object(vs.edge_tts, "Communicate", FakeCommunicate):
            sub_maker = vs.azure_tts_v1(
                text="hello world",
                voice_name="zh-CN-XiaoyiNeural-Female",
                voice_file=f"{temp_dir}/tts-403-test.mp3",
                voice_rate=voice_rate,
            )

        self.assertIsNone(sub_maker)
        self.assertIn("403", vs.get_last_tts_error())
        self.assertIn("edge_tts", vs.get_last_tts_error())

    def test_tts_falls_back_to_windows_sapi_after_edge_403(self):
        original_tts_server = config.style.get("tts_server")
        config.style["tts_server"] = "azure-tts-v1"
        try:
            with patch.object(vs, "azure_tts_v1", return_value=None), patch.object(
                vs, "_should_fallback_to_windows_sapi", return_value=True
            ), patch.object(vs, "windows_sapi_tts", return_value="fallback-sub-maker") as fallback:
                result = vs.tts(
                    text="こんにちは",
                    voice_name="ja-JP-NanamiNeural-Female",
                    voice_rate=1.0,
                    voice_file=f"{temp_dir}/tts-fallback.wav",
                )

            self.assertEqual(result, "fallback-sub-maker")
            fallback.assert_called_once()
        finally:
            config.style["tts_server"] = original_tts_server

    def test_build_approximate_submaker_uses_audio_duration(self):
        with patch.object(vs, "get_audio_duration_from_file", return_value=6.0):
            sub_maker = vs._build_approximate_submaker(
                text="一つ目の文です。二つ目の文です。",
                audio_file="dummy.wav",
            )

        self.assertEqual(len(sub_maker.subs), 2)
        self.assertEqual(sub_maker.subs[0], "一つ目の文です")
        self.assertEqual(sub_maker.offset[-1][1], 60000000)

    def test_azure_tts_v1_successful_stream_keeps_subs(self):
        class FakeCommunicate:
            def __init__(self, *args, **kwargs):
                pass

            async def stream(self):
                yield {"type": "audio", "data": b"fake-audio"}
                yield {"type": "WordBoundary", "text": "hello", "offset": 0, "duration": 1}
                yield {"type": "WordBoundary", "text": "world", "offset": 1, "duration": 1}

        voice_file = f"{temp_dir}/tts-success-test.mp3"
        if os.path.exists(voice_file):
            os.remove(voice_file)

        with patch.object(vs.edge_tts, "Communicate", FakeCommunicate):
            sub_maker = vs.azure_tts_v1(
                text="hello world",
                voice_name="zh-CN-XiaoyiNeural-Female",
                voice_file=voice_file,
                voice_rate=voice_rate,
            )

        self.assertIsNotNone(sub_maker)
        self.assertEqual(sub_maker.subs, ["hello", "world"])
        self.assertEqual(len(sub_maker.offset), 2)
        self.assertEqual(vs.get_last_tts_error(), "")
        self.assertTrue(os.path.exists(voice_file))
        os.remove(voice_file)
    
    @unittest.skipUnless(RUN_LIVE_TTS_TESTS, "live TTS test requires external credentials/network")
    def test_siliconflow(self):
        voice_name = "siliconflow:FunAudioLLM/CosyVoice2-0.5B:alex-Male"
        voice_name = vs.parse_voice_name(voice_name)
        
        async def _do():
            parts = voice_name.split(":")
            if len(parts) >= 3:
                model = parts[1]
                # 移除性别后缀，例如 "alex-Male" -> "alex"
                voice_with_gender = parts[2]
                voice = voice_with_gender.split("-")[0]
                # 构建完整的voice参数，格式为 "model:voice"
                full_voice = f"{model}:{voice}"
                voice_file = f"{temp_dir}/tts-siliconflow-{voice}.mp3"
                subtitle_file = f"{temp_dir}/tts-siliconflow-{voice}.srt"
                sub_maker = vs.siliconflow_tts(
                    text=text_zh, model=model, voice=full_voice, voice_file=voice_file, voice_rate=voice_rate, voice_volume=voice_volume
                )
                if not sub_maker:
                    self.fail("siliconflow tts failed")
                vs.create_subtitle(sub_maker=sub_maker, text=text_zh, subtitle_file=subtitle_file)
                audio_duration = vs.get_audio_duration(sub_maker)
                print(f"voice: {voice_name}, audio duration: {audio_duration}s")
            else:
                self.fail("siliconflow invalid voice name")

        self.loop.run_until_complete(_do())
    
    def test_azure_tts_v1(self):
        if not RUN_LIVE_TTS_TESTS:
            self.skipTest("live TTS test requires external network")
        voice_name = "zh-CN-XiaoyiNeural-Female"
        voice_name = vs.parse_voice_name(voice_name)
        print(voice_name)
        
        voice_file = f"{temp_dir}/tts-azure-v1-{voice_name}.mp3"
        subtitle_file = f"{temp_dir}/tts-azure-v1-{voice_name}.srt"
        sub_maker = vs.azure_tts_v1(
            text=text_zh, voice_name=voice_name, voice_file=voice_file, voice_rate=voice_rate
        )
        if not sub_maker:
            self.fail("azure tts v1 failed")
        vs.create_subtitle(sub_maker=sub_maker, text=text_zh, subtitle_file=subtitle_file)
        self.assertTrue(os.path.exists(subtitle_file))
        audio_duration = vs.get_audio_duration(sub_maker)
        print(f"voice: {voice_name}, audio duration: {audio_duration}s")

    def test_azure_tts_v2(self):
        if not RUN_LIVE_TTS_TESTS:
            self.skipTest("live TTS test requires external credentials")
        voice_name = "zh-CN-XiaoxiaoMultilingualNeural-V2-Female"
        voice_name = vs.parse_voice_name(voice_name)
        print(voice_name)

        async def _do():
            voice_file = f"{temp_dir}/tts-azure-v2-{voice_name}.mp3"
            subtitle_file = f"{temp_dir}/tts-azure-v2-{voice_name}.srt"
            sub_maker = vs.azure_tts_v2(
                text=text_zh, voice_name=voice_name, voice_file=voice_file
            )
            if not sub_maker:
                self.fail("azure tts v2 failed")
            vs.create_subtitle(sub_maker=sub_maker, text=text_zh, subtitle_file=subtitle_file)
            audio_duration = vs.get_audio_duration(sub_maker)
            print(f"voice: {voice_name}, audio duration: {audio_duration}s")

        self.loop.run_until_complete(_do())

if __name__ == "__main__":
    # python -m unittest test.services.test_voice.TestVoiceService.test_azure_tts_v1
    # python -m unittest test.services.test_voice.TestVoiceService.test_azure_tts_v2
    unittest.main() 
