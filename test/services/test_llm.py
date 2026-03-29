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


if __name__ == "__main__":
    unittest.main()
