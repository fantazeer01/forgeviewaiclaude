import requests

from layer1_eyes.news_feed import NewsFeed


# 3. News feed returns neutral sentiment when the API is unavailable.
def test_neutral_sentiment_without_api_key():
    feed = NewsFeed(api_key="")
    feed.poll(now=1000.0)
    assert feed.sentiment_1h() == 0.0
    assert feed.count_1h() == 0
    assert feed.has_major_1h() is False


def test_neutral_sentiment_on_fetch_error(tmp_path):
    class BrokenSession:
        def get(self, *a, **k):
            raise requests.ConnectionError("boom")

    feed = NewsFeed(session=BrokenSession(), api_key="fake-token", cache_path=str(tmp_path / "news.json"))
    feed.poll(now=1000.0)
    assert feed.sentiment_1h() == 0.0


def test_sentiment_computed_from_votes(tmp_path):
    import datetime

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            return {"results": [
                {"title": "Bull run", "source": {"domain": "coindesk"},
                 "currencies": [{"code": "BTC"}], "votes": {"positive": 8, "negative": 2},
                 "published_at": now},
            ]}

    class FakeSession:
        def get(self, *a, **k):
            return FakeResponse()

    feed = NewsFeed(session=FakeSession(), api_key="fake-token", cache_path=str(tmp_path / "news.json"))
    feed.poll(now=1000.0)
    assert feed.sentiment_1h() == 0.6  # (8-2)/10
    assert feed.count_1h() == 1
    assert feed.has_major_1h() is True
