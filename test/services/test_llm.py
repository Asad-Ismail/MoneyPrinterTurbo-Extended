import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.services import llm


class TestLlmService(unittest.TestCase):
    def test_health_check_for_openrouter(self):
        config.llm["provider"] = "openrouter"
        config.llm["model"] = "openai/gpt-4o-mini"
        config.llm["base_url"] = "https://openrouter.ai/api/v1"
        config.app["openrouter_api_key"] = ""
        result = llm.health_check("openrouter")
        self.assertEqual(result["provider"], "openrouter")
        self.assertFalse(result["ok"])
        self.assertIn("api_key", result["missing"])

    def test_japanese_script_prompt_adds_language_rules(self):
        prompt = llm._build_script_prompt(
            video_subject="日本の物価上昇",
            language="ja-JP",
            paragraph_number=2,
        )
        self.assertIn("language: ja-JP", prompt)
        self.assertIn("natural spoken Japanese", prompt)
        self.assertIn("easy to subtitle", prompt)


if __name__ == "__main__":
    unittest.main()
