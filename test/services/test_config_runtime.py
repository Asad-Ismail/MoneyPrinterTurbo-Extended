import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config


class TestConfigRuntime(unittest.TestCase):
    def test_normalize_config_adds_grouped_sections(self):
        raw = {
            "app": {
                "llm_provider": "groq",
                "groq_model_name": "llama-3.1-8b-instant",
                "groq_base_url": "https://api.groq.com/openai/v1",
                "compute_profile": "cpu-safe",
            },
            "ui": {"tts_server": "chatterbox"},
        }
        normalized = config.normalize_config_dict(raw)
        self.assertEqual(normalized["llm"]["provider"], "groq")
        self.assertEqual(normalized["performance"]["compute_profile"], "cpu-safe")
        self.assertEqual(normalized["ui"]["tts_server"], "azure-tts-v1")
        self.assertIn("quality", normalized)

    def test_provider_config_reads_env_override(self):
        previous = os.environ.get("TEST_GROQ_KEY")
        os.environ["TEST_GROQ_KEY"] = "env-secret"
        try:
            config.llm["provider"] = "groq"
            config.llm["api_key_env"] = "TEST_GROQ_KEY"
            config.app["groq_api_key"] = "file-secret"
            provider_cfg = config.get_llm_provider_config("groq")
            self.assertEqual(provider_cfg["api_key"], "env-secret")
        finally:
            config.llm["api_key_env"] = ""
            if previous is None:
                os.environ.pop("TEST_GROQ_KEY", None)
            else:
                os.environ["TEST_GROQ_KEY"] = previous


if __name__ == "__main__":
    unittest.main()
