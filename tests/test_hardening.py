"""Slack 429 재시도, 비밀 정보 로그 노출 방지, WAF 오판 방지 테스트."""

from __future__ import annotations

import logging
import types

import pytest

import gsai_notifier as gn


class _Resp:
    def __init__(self, status_code=200, content=b"", text="", headers=None):
        self.status_code = status_code
        self.content = content
        self.text = text
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _ConnErr(Exception):
    pass


class _RequestException(Exception):
    pass


@pytest.fixture
def sleeps(monkeypatch):
    calls: list[float] = []
    monkeypatch.setattr(gn, "_sleep", calls.append)
    monkeypatch.setattr(gn, "_rand", lambda: 0.0)
    return calls


@pytest.fixture
def fake_requests(monkeypatch):
    fake = types.SimpleNamespace(
        exceptions=types.SimpleNamespace(
            ConnectionError=_ConnErr, RequestException=_RequestException
        ),
        RequestException=_RequestException,
        get=None,
        post=None,
    )
    monkeypatch.setattr(gn, "requests", fake)
    return fake


# --- 429 ---


def test_429_waits_retry_after_then_succeeds(fake_requests, sleeps):
    responses = iter([_Resp(429, headers={"Retry-After": "2"}), _Resp(200)])
    resp = gn._request_with_retry(lambda *a, **k: next(responses), retries=3, backoff=0.5)
    assert resp.status_code == 200
    assert sleeps == [2.0]


def test_429_without_header_uses_backoff(fake_requests, sleeps):
    responses = iter([_Resp(429), _Resp(429, headers={"Retry-After": "soon"}), _Resp(200)])
    gn._request_with_retry(lambda *a, **k: next(responses), retries=3, backoff=0.5)
    assert sleeps == [0.5, 1.0]


def test_429_retry_after_is_capped(fake_requests, sleeps):
    responses = iter([_Resp(429, headers={"Retry-After": "3600"}), _Resp(200)])
    gn._request_with_retry(
        lambda *a, **k: next(responses), retries=3, backoff=0.5, max_backoff=30.0
    )
    assert sleeps == [30.0]


def test_persistent_429_returns_last_response(fake_requests, sleeps):
    calls = {"n": 0}

    def always_429(*_a, **_k):
        calls["n"] += 1
        return _Resp(429, headers={"Retry-After": "1"})

    resp = gn._request_with_retry(always_429, retries=2, backoff=0.5)
    assert resp.status_code == 429
    assert calls["n"] == 3


def test_other_4xx_still_not_retried(fake_requests, sleeps):
    calls = {"n": 0}

    def bad_request(*_a, **_k):
        calls["n"] += 1
        return _Resp(400)

    assert gn._request_with_retry(bad_request, retries=3).status_code == 400
    assert calls["n"] == 1
    assert sleeps == []


# --- Slack 오류에 webhook URL이 실리지 않음 ---

SECRET_HOOK = "https://hooks.slack.com/services/T000/B000/SECRETTOKEN123"


def test_slack_connection_error_hides_webhook_url(fake_requests, sleeps):
    def boom(url, **_k):
        raise _ConnErr(f"Max retries exceeded with url: {url}")

    fake_requests.post = boom
    with pytest.raises(RuntimeError) as info:
        gn.send_to_slack(SECRET_HOOK, {"text": "hi"}, dry_run=False)

    assert "SECRETTOKEN123" not in str(info.value)
    assert info.value.__cause__ is None
    assert info.value.__suppress_context__ is True


def test_slack_non_retryable_request_error_hides_webhook_url(fake_requests, sleeps):
    def boom(url, **_k):
        raise _RequestException(f"Invalid URL {url}")

    fake_requests.post = boom
    with pytest.raises(RuntimeError) as info:
        gn.send_to_slack(SECRET_HOOK, {"text": "hi"}, dry_run=False)
    assert "SECRETTOKEN123" not in str(info.value)


def test_slack_http_error_includes_reason_not_url(fake_requests, sleeps):
    fake_requests.post = lambda url, **_k: _Resp(400, text="invalid_blocks")
    with pytest.raises(RuntimeError, match="Slack HTTP 400: invalid_blocks") as info:
        gn.send_to_slack(SECRET_HOOK, {"text": "hi"}, dry_run=False)
    assert "SECRETTOKEN123" not in str(info.value)


def test_slack_failure_log_line_does_not_contain_webhook(tmp_path, monkeypatch, caplog, sleeps):
    """실제 실행 경로(run_once → LOG.exception)에서도 URL이 로그에 남지 않습니다."""
    import json

    def boom(url, **_k):
        raise _ConnErr(f"Max retries exceeded with url: {url}")

    fake = types.SimpleNamespace(
        exceptions=types.SimpleNamespace(ConnectionError=_ConnErr), get=None, post=boom
    )
    monkeypatch.setattr(gn, "requests", fake)
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps({"version": 1, "feeds": {"https://feed.a/": {"last_id": "old"}}}),
        encoding="utf-8",
    )
    parsed = types.SimpleNamespace(
        feed={"title": "T", "link": ""},
        entries=[{"id": "new", "link": "https://x/new", "title": "n"}, {"id": "old"}],
    )
    monkeypatch.setattr(gn, "fetch_feed", lambda url, *, verify_ssl: parsed)
    cfg = gn.Config(
        slack_webhook_url=SECRET_HOOK,
        feed_urls=["https://feed.a/"],
        state_file=state_file,
        initial_notify_count=0,
        max_items_per_message=10,
        on_state_miss="skip",
        verify_ssl=True,
        dry_run=False,
        notion_token="secret_notion_token_value",
        notion_page_id="0123456789abcdef0123456789abcdef",
    )
    with caplog.at_level(logging.DEBUG, logger="gsai_notifier"):
        assert gn.run_once(cfg) == 3

    logged = caplog.text + "\n".join(
        str(r.exc_info[1]) for r in caplog.records if r.exc_info and r.exc_info[1]
    )
    assert "SECRETTOKEN123" not in logged
    assert "hooks.slack.com/services" not in logged


def test_notion_log_line_has_no_token_prefix(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(gn, "send_to_slack", lambda *a, **k: None)
    monkeypatch.setattr(gn, "send_to_notion", lambda **k: None)
    cfg = gn.Config(
        slack_webhook_url="x",
        feed_urls=[],
        state_file=tmp_path / "s.json",
        initial_notify_count=0,
        max_items_per_message=10,
        on_state_miss="skip",
        verify_ssl=True,
        dry_run=True,
        notion_token="ntn_SUPERSECRET",
        notion_page_id="https://www.notion.so/My-Page-0123456789abcdef0123456789abcdef",
    )
    with caplog.at_level(logging.DEBUG, logger="gsai_notifier"):
        gn._dispatch_items(
            cfg,
            "T",
            [{"id": "a", "title": "a", "link": "https://x/a"}],
            feed_url="f",
            site_url="",
            summary=gn._RunSummary(),
        )
    assert "ntn_SUPER" not in caplog.text
    assert "My-Page" not in caplog.text


# --- WAF 오판 방지 ---


def _install_parser(monkeypatch, entries):
    parsed = types.SimpleNamespace(entries=entries, feed={}, get=lambda k, d=None: d)
    monkeypatch.setattr(gn, "feedparser", types.SimpleNamespace(parse=lambda _c: parsed))


def test_legit_feed_mentioning_snucert_is_accepted(fake_requests, sleeps, monkeypatch):
    """보안 공지 본문이 snucert.snu.ac.kr을 링크해도 정상 피드로 처리합니다."""
    _install_parser(monkeypatch, entries=[{"id": "a"}])
    body = b'<rss><item><description><a href="https://snucert.snu.ac.kr/">x</a></description></item></rss>'
    fake_requests.get = lambda *a, **k: _Resp(200, content=body)

    parsed = gn.fetch_feed("https://feed", verify_ssl=True)
    assert parsed.entries == [{"id": "a"}]


def test_real_waf_page_still_detected(fake_requests, sleeps, monkeypatch):
    _install_parser(monkeypatch, entries=[])
    page = b'<meta http-equiv="refresh" content="0;url=http://snucert.snu.ac.kr/waf/error.html">'
    fake_requests.get = lambda *a, **k: _Resp(200, content=page)

    with pytest.raises(RuntimeError, match="방화벽"):
        gn.fetch_feed("https://feed", verify_ssl=True)


def test_empty_feed_without_waf_marker_reports_no_entries(fake_requests, sleeps, monkeypatch):
    _install_parser(monkeypatch, entries=[])
    fake_requests.get = lambda *a, **k: _Resp(200, content=b"<rss></rss>")
    with pytest.raises(RuntimeError, match="항목이 없습니다"):
        gn.fetch_feed("https://feed", verify_ssl=True)
