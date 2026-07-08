import datetime

from core.macro_events import MacroEventsFetcher


def make_response(mocker, payload):
    resp = mocker.Mock()
    resp.raise_for_status = mocker.Mock()
    resp.json.return_value = payload
    return resp


def iso(dt):
    return dt.isoformat()


def test_filters_to_usd_high_impact_only(mocker):
    now = datetime.datetime.now(datetime.timezone.utc)
    future = now + datetime.timedelta(days=1)
    mocker.patch(
        "core.macro_events.requests.get",
        return_value=make_response(mocker, [
            {"title": "Non-USD event", "country": "EUR", "impact": "High", "date": iso(future)},
            {"title": "Low impact USD", "country": "USD", "impact": "Low", "date": iso(future)},
            {"title": "NFP", "country": "USD", "impact": "High", "date": iso(future)},
        ]),
    )
    events = MacroEventsFetcher().fetch_next_events(3)
    assert len(events) == 1
    assert events[0]["title"] == "NFP"


def test_prefers_future_events_sorted_ascending(mocker):
    now = datetime.datetime.now(datetime.timezone.utc)
    later = now + datetime.timedelta(days=2)
    sooner = now + datetime.timedelta(hours=1)
    mocker.patch(
        "core.macro_events.requests.get",
        return_value=make_response(mocker, [
            {"title": "Later Event", "country": "USD", "impact": "High", "date": iso(later)},
            {"title": "Sooner Event", "country": "USD", "impact": "High", "date": iso(sooner)},
        ]),
    )
    events = MacroEventsFetcher().fetch_next_events(3)
    assert [e["title"] for e in events] == ["Sooner Event", "Later Event"]


def test_falls_back_to_latest_past_events_when_none_are_future(mocker):
    now = datetime.datetime.now(datetime.timezone.utc)
    older = now - datetime.timedelta(days=5)
    newer = now - datetime.timedelta(days=1)
    mocker.patch(
        "core.macro_events.requests.get",
        return_value=make_response(mocker, [
            {"title": "Older", "country": "USD", "impact": "High", "date": iso(older)},
            {"title": "Newer", "country": "USD", "impact": "High", "date": iso(newer)},
        ]),
    )
    events = MacroEventsFetcher().fetch_next_events(3)
    # both are in the past (clock-mismatch fallback case) -- still returns real
    # data, just the latest-dated entries rather than nothing
    assert [e["title"] for e in events] == ["Older", "Newer"]


def test_respects_n_limit(mocker):
    now = datetime.datetime.now(datetime.timezone.utc)
    mocker.patch(
        "core.macro_events.requests.get",
        return_value=make_response(mocker, [
            {"title": f"Event {i}", "country": "USD", "impact": "High",
             "date": iso(now + datetime.timedelta(hours=i))}
            for i in range(5)
        ]),
    )
    events = MacroEventsFetcher().fetch_next_events(3)
    assert len(events) == 3


def test_returns_empty_list_when_no_matching_events(mocker):
    mocker.patch(
        "core.macro_events.requests.get",
        return_value=make_response(mocker, [
            {"title": "Not USD", "country": "EUR", "impact": "High", "date": "2026-01-01T00:00:00+00:00"},
        ]),
    )
    assert MacroEventsFetcher().fetch_next_events(3) == []


def test_returns_none_on_request_exception(mocker):
    mocker.patch("core.macro_events.requests.get", side_effect=RuntimeError("network down"))
    assert MacroEventsFetcher().fetch_next_events(3) is None


def test_skips_malformed_event_rows(mocker):
    now = datetime.datetime.now(datetime.timezone.utc)
    future = now + datetime.timedelta(days=1)
    mocker.patch(
        "core.macro_events.requests.get",
        return_value=make_response(mocker, [
            {"title": "", "country": "USD", "impact": "High", "date": iso(future)},
            {"title": "Missing date", "country": "USD", "impact": "High"},
            {"title": "Bad date", "country": "USD", "impact": "High", "date": "not-a-date"},
            {"title": "Valid", "country": "USD", "impact": "High", "date": iso(future)},
        ]),
    )
    events = MacroEventsFetcher().fetch_next_events(3)
    assert [e["title"] for e in events] == ["Valid"]
