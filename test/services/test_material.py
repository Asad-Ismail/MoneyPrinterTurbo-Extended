import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from tenacity import wait_none

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.modules.setdefault("sentence_transformers", MagicMock())
sys.modules.setdefault("sklearn", MagicMock())
sys.modules.setdefault("sklearn.metrics.pairwise", MagicMock())

import requests
from app.services import material


class TestMaterialHttpRetry(unittest.TestCase):
    def setUp(self):
        self._wait = material._http_get.retry.wait
        material._http_get.retry.wait = wait_none()

    def tearDown(self):
        material._http_get.retry.wait = self._wait

    def test_search_pexels_retries_timeout_then_succeeds(self):
        ok = MagicMock()
        ok.json.return_value = {"videos": []}
        calls = {"n": 0}

        def fake_get(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] < 2:
                raise requests.exceptions.Timeout()
            return ok

        with patch.object(material, "get_api_key", return_value="k"), patch(
            "app.services.material.requests.get", side_effect=fake_get
        ):
            items = material.search_videos_pexels("cats", minimum_duration=1)

        self.assertEqual(items, [])
        self.assertEqual(calls["n"], 2)

    def test_http_get_reraises_after_retries(self):
        with patch(
            "app.services.material.requests.get",
            side_effect=requests.exceptions.Timeout(),
        ):
            with self.assertRaises(requests.exceptions.Timeout):
                material._http_get("https://example.com", timeout=1)


if __name__ == "__main__":
    unittest.main()
