"""_request_with_retry 및 fetch_feed/send_to_slack 재시도 동작의 오프라인 테스트.

이 테스트들은 requests/feedparser가 설치되지 않은 샌드박스에서도 동작하도록
gn.requests / gn.feedparser 를 가짜 객체로 monkeypatch하고, gn._sleep 을
가로채 실제로 대기하지 않게 합니다.
"""

from __future__ import annotations

import types

import pytest

import gsai_notifier as gn


class _FakeResponse:
    """requests.Response 유사 객체. status_code/content 와 raise_for_status 제공."""

    def __init__(self, status_code: int = 200, content: bytes = b"") -> None:
        self.status_code = status_code
        self.content = content
        self.raised = False

    def raise_for_status(self) -> None:
        self.raised = True
        if self.status_code >= 400:
            raise _FakeHTTPError(f"HTTP {self.status_code}")


class _FakeHTTPError(Exception):
    """requests.exceptions.HTTPError 대용(raise_for_status 실패)."""


class _FakeConnectionError(Exception):
    """requests.exceptions.ConnectionError 대용(일시적 연결 오류)."""


class _FakeTimeout(Exception):
    """requests.exceptions.Timeout 대용(일시적 타임아웃)."""


def _make_fake_requests() -> types.SimpleNamespace:
    """gn.requests 를 대체할 가짜 모듈 유사 객체를 만듭니다.

    .exceptions 에 ConnectionError/Timeout 을 노출해 _retryable_exceptions()가
    이들을 재시도 대상으로 인식하도록 합니다.
    """
    exceptions = types.SimpleNamespace(
        ConnectionError=_FakeConnectionError,
        Timeout=_FakeTimeout,
    )
    return types.SimpleNamespace(exceptions=exceptions, get=None, post=None)


@pytest.fixture
def no_sleep(monkeypatch):
    """실제 대기 없이 _sleep 호출 인자를 기록합니다."""
    calls: list[float] = []

    def _fake_sleep(seconds: float) -> None:
        calls.append(seconds)

    monkeypatch.setattr(gn, "_sleep", _fake_sleep)
    return calls


@pytest.fixture
def fake_requests(monkeypatch):
    """gn.requests 를 가짜 객체로 교체합니다."""
    fake = _make_fake_requests()
    monkeypatch.setattr(gn, "requests", fake)
    return fake


# --- _request_with_retry 단위 테스트 ---


def test_transient_then_success_returns_response(fake_requests, no_sleep):
    """일시적 예외 2회 후 성공하면 최종 응답을 반환하고 호출 횟수가 맞아야 함."""
    attempts = {"n": 0}
    ok = _FakeResponse(200, b"ok")

    def flaky(*_a, **_k):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise fake_requests.exceptions.ConnectionError("boom")
        return ok

    resp = gn._request_with_retry(flaky, "http://x", retries=3, backoff=0.5)
    assert resp is ok
    assert attempts["n"] == 3
    # 2번의 재시도 -> 2번의 sleep, 지수 백오프(0.5, 1.0)
    assert no_sleep == [0.5, 1.0]


def test_persistent_5xx_eventually_raises(fake_requests, no_sleep):
    """지속적인 5xx는 재시도 후 마지막 응답을 반환하고, raise_for_status에서 실패."""
    calls = {"n": 0}

    def always_500(*_a, **_k):
        calls["n"] += 1
        return _FakeResponse(503, b"unavailable")

    resp = gn._request_with_retry(always_500, retries=3, backoff=0.5)
    # 최초 1회 + 3회 재시도 = 4회 호출
    assert calls["n"] == 4
    # 최종 응답은 5xx이며 raise_for_status에서 에러
    assert resp.status_code == 503
    with pytest.raises(_FakeHTTPError):
        resp.raise_for_status()
    # 3회 재시도 -> 3회 sleep
    assert no_sleep == [0.5, 1.0, 2.0]


def test_5xx_then_success_is_retried(fake_requests, no_sleep):
    """5xx 응답 후 200이면 재시도하여 최종 200을 반환."""
    seq = [_FakeResponse(500), _FakeResponse(200, b"ok")]
    calls = {"n": 0}

    def flaky(*_a, **_k):
        r = seq[calls["n"]]
        calls["n"] += 1
        return r

    resp = gn._request_with_retry(flaky, retries=3, backoff=0.5)
    assert resp.status_code == 200
    assert calls["n"] == 2
    assert no_sleep == [0.5]


def test_4xx_not_retried(fake_requests, no_sleep):
    """4xx 응답은 재시도하지 않고 즉시 반환(호출 1회, sleep 0회)."""
    calls = {"n": 0}

    def client_error(*_a, **_k):
        calls["n"] += 1
        return _FakeResponse(404, b"not found")

    resp = gn._request_with_retry(client_error, retries=3, backoff=0.5)
    assert resp.status_code == 404
    assert calls["n"] == 1
    assert no_sleep == []


def test_non_transient_exception_not_retried(fake_requests, no_sleep):
    """재시도 대상이 아닌 예외는 첫 호출에서 즉시 전파(호출 1회)."""
    calls = {"n": 0}

    def boom(*_a, **_k):
        calls["n"] += 1
        raise ValueError("non-transient")

    with pytest.raises(ValueError):
        gn._request_with_retry(boom, retries=3, backoff=0.5)
    assert calls["n"] == 1
    assert no_sleep == []


def test_persistent_transient_exception_raises_last(fake_requests, no_sleep):
    """지속적 타임아웃은 시도 소진 후 마지막 예외를 전파."""
    calls = {"n": 0}

    def always_timeout(*_a, **_k):
        calls["n"] += 1
        raise fake_requests.exceptions.Timeout("slow")

    with pytest.raises(_FakeTimeout):
        gn._request_with_retry(always_timeout, retries=2, backoff=0.5)
    assert calls["n"] == 3  # 1 + 2 재시도
    assert no_sleep == [0.5, 1.0]


# --- fetch_feed 통합(재시도 + WAF/빈 항목 검사 유지) ---


def _install_fake_feedparser(monkeypatch, entries, feed=None, bozo=False):
    parsed = types.SimpleNamespace(
        entries=entries,
        feed=feed or {},
        get=lambda k, default=None: {"bozo": bozo}.get(k, default),
    )

    fake_fp = types.SimpleNamespace(parse=lambda _content: parsed)
    monkeypatch.setattr(gn, "feedparser", fake_fp)
    return parsed


def test_fetch_feed_retries_then_succeeds(fake_requests, no_sleep, monkeypatch):
    """fetch_feed는 일시적 get 실패를 재시도한 뒤 성공하고 항목을 반환."""
    _install_fake_feedparser(monkeypatch, entries=[{"id": "a", "title": "T"}])

    attempts = {"n": 0}

    def flaky_get(*_a, **_k):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise fake_requests.exceptions.ConnectionError("net")
        return _FakeResponse(200, b"<rss/>")

    fake_requests.get = flaky_get

    parsed = gn.fetch_feed("http://feed", verify_ssl=True)
    assert attempts["n"] == 2
    assert parsed.entries == [{"id": "a", "title": "T"}]
    assert no_sleep == [0.5]


def test_fetch_feed_waf_block_not_retried(fake_requests, no_sleep, monkeypatch):
    """WAF 차단(200 본문)은 일시적 오류가 아니므로 재시도하지 않고 즉시 실패."""
    _install_fake_feedparser(monkeypatch, entries=[{"id": "a"}])
    calls = {"n": 0}

    def blocked(*_a, **_k):
        calls["n"] += 1
        return _FakeResponse(200, b"... waf/error ...")

    fake_requests.get = blocked

    with pytest.raises(RuntimeError, match="방화벽"):
        gn.fetch_feed("http://feed", verify_ssl=True)
    assert calls["n"] == 1
    assert no_sleep == []


def test_fetch_feed_empty_entries_not_retried(fake_requests, no_sleep, monkeypatch):
    """빈 항목 응답은 재시도하지 않고 RuntimeError."""
    _install_fake_feedparser(monkeypatch, entries=[])
    calls = {"n": 0}

    def ok_empty(*_a, **_k):
        calls["n"] += 1
        return _FakeResponse(200, b"<rss/>")

    fake_requests.get = ok_empty

    with pytest.raises(RuntimeError, match="항목이 없습니다"):
        gn.fetch_feed("http://feed", verify_ssl=True)
    assert calls["n"] == 1
    assert no_sleep == []


def test_fetch_feed_retries_5xx_then_applies_checks(fake_requests, no_sleep, monkeypatch):
    """5xx 재시도 후 200이 오면 WAF/빈 항목 검사를 정상 적용."""
    _install_fake_feedparser(monkeypatch, entries=[{"id": "x", "title": "ok"}])
    seq = [_FakeResponse(500), _FakeResponse(200, b"<rss/>")]
    calls = {"n": 0}

    def flaky(*_a, **_k):
        r = seq[calls["n"]]
        calls["n"] += 1
        return r

    fake_requests.get = flaky

    parsed = gn.fetch_feed("http://feed", verify_ssl=True)
    assert calls["n"] == 2
    assert parsed.entries[0]["id"] == "x"
    assert no_sleep == [0.5]


# --- send_to_slack 통합 ---


def test_send_to_slack_retries_then_succeeds(fake_requests, no_sleep):
    """비 dry_run에서 일시적 post 실패를 재시도한 뒤 성공."""
    attempts = {"n": 0}

    def flaky_post(*_a, **_k):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise fake_requests.exceptions.Timeout("slow")
        return _FakeResponse(200, b"ok")

    fake_requests.post = flaky_post

    gn.send_to_slack("http://hook", {"text": "hi"}, dry_run=False)
    assert attempts["n"] == 3
    assert no_sleep == [0.5, 1.0]


def test_send_to_slack_persistent_5xx_raises(fake_requests, no_sleep):
    """지속적 5xx는 재시도 소진 후 raise_for_status에서 실패."""
    calls = {"n": 0}

    def always_500(*_a, **_k):
        calls["n"] += 1
        return _FakeResponse(502, b"bad gateway")

    fake_requests.post = always_500

    with pytest.raises(_FakeHTTPError):
        gn.send_to_slack("http://hook", {"text": "hi"}, dry_run=False)
    assert calls["n"] == 4  # 1 + 3 재시도
    assert no_sleep == [0.5, 1.0, 2.0]


def test_send_to_slack_dry_run_never_calls_network(fake_requests, no_sleep):
    """dry_run 모드는 네트워크를 호출하지 않음."""

    def should_not_call(*_a, **_k):
        raise AssertionError("dry_run에서 post가 호출되면 안 됩니다.")

    fake_requests.post = should_not_call

    payload = {"attachments": [{"blocks": [{"type": "divider"}]}]}
    gn.send_to_slack("http://hook", payload, dry_run=True)
    assert no_sleep == []


def test_retryable_exceptions_fallback_without_requests(monkeypatch):
    """requests가 None이면 OSError로 폴백하여 재시도 대상 튜플에 포함."""
    monkeypatch.setattr(gn, "requests", None)
    types_tuple = gn._retryable_exceptions()
    assert gn._TransientHTTPError in types_tuple
    assert OSError in types_tuple
