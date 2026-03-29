import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import quality


class DummyParams:
    voice_volume = 1.0
    bgm_volume = 0.5
    video_language = ""


class TestQualityChecks(unittest.TestCase):
    def test_srt_parser_and_report(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            subtitle_path = os.path.join(tmp_dir, "sample.srt")
            with open(subtitle_path, "w", encoding="utf-8") as f:
                f.write(
                    "1\n00:00:00,000 --> 00:00:01,000\n短い字幕\n\n"
                    "2\n00:00:01,000 --> 00:00:03,500\nこれは少し長めの字幕行です\n"
                )
            report = quality.run_quality_checks(
                task_id="unit-test-quality",
                params=DummyParams(),
                audio_duration=3.4,
                subtitle_path=subtitle_path,
                materials=["a.mp4", "a.mp4", "b.mp4"],
                video_script="This is not guaranteed profit content.",
            )
            self.assertIn("checks", report)
            self.assertTrue(any(item["name"] == "audio_subtitle_delta" for item in report["checks"]))
            self.assertTrue(any(item["name"] == "material_duplication" for item in report["checks"]))

    def test_japanese_subtitle_density_uses_stricter_limit(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            subtitle_path = os.path.join(tmp_dir, "ja.srt")
            with open(subtitle_path, "w", encoding="utf-8") as f:
                f.write(
                    "1\n00:00:00,000 --> 00:00:02,000\nこれは日本語字幕の一行としては少し長めに作ってあるサンプルです\n"
                )

            params = DummyParams()
            params.video_language = "ja-JP"
            report = quality.run_quality_checks(
                task_id="unit-test-quality-ja",
                params=params,
                audio_duration=2.0,
                subtitle_path=subtitle_path,
                materials=["a.mp4"],
                video_script="通常の説明文です。",
            )
            subtitle_check = next(
                item for item in report["checks"] if item["name"] == "subtitle_density"
            )
            self.assertEqual(subtitle_check["allowed_chars"], 26)
            self.assertEqual(subtitle_check["level"], "warning")


if __name__ == "__main__":
    unittest.main()
