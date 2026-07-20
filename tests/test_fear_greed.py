import json

import requests

from layer1_eyes.fear_greed import FearGreedIndex


# 4. Fear & Greed uses the cache when the API is unavailable.
def test_uses_cache_on_api_error(tmp_path):
    cache_path = tmp_path / "fear_greed.json"
    cache_path.write_text(json.dumps({"value": 42.0, "previous_value": 38.0}), encoding="utf-8")

    class BrokenSession:
        def get(self, *a, **k):
            raise requests.ConnectionError("boom")

    index = FearGreedIndex(session=BrokenSession(), cache_path=str(cache_path))
    index.poll(now=1000.0)
    assert index.normalized() == 0.42
    assert index.change_24h() == 4.0


def test_neutral_fallback_when_no_cache_and_api_fails(tmp_path):
    class BrokenSession:
        def get(self, *a, **k):
            raise requests.ConnectionError("boom")

    index = FearGreedIndex(session=BrokenSession(), cache_path=str(tmp_path / "missing.json"))
    index.poll(now=1000.0)
    assert index.normalized() == 0.5
    assert index.change_24h() == 0.0


def test_successful_fetch_updates_and_writes_cache(tmp_path):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"value": "70"}, {"value": "60"}]}

    class FakeSession:
        def get(self, *a, **k):
            return FakeResponse()

    cache_path = tmp_path / "fear_greed.json"
    index = FearGreedIndex(session=FakeSession(), cache_path=str(cache_path))
    index.poll(now=1000.0)
    assert index.normalized() == 0.7
    assert index.change_24h() == 10.0
    assert cache_path.exists()
