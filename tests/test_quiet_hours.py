"""조용한 시간(QUIET_HOURS) 테스트.

배경: GitHub Actions가 예약 실행을 평균 7~9시간 늦게 돌려 공지가 새벽에 도착했습니다.
매시간 예약으로 바꾸고, 밤에는 아무것도 보내지 않은 채 다음 실행으로 미룹니다.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

import gsai_notifier as gn


def _kst(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 11, 26, hour, minute, tzinfo=gn._KST)


# --- 파싱 ---


@pytest.mark.parametrize(
    "raw, expected",
    [("22-8", (22, 8)), (" 23 - 7 ", (23, 7)), ("9-18", (9, 18)), ("", None), (None, None)],
)
def test_parse_quiet_hours(raw, expected):
    assert gn._parse_quiet_hours(raw) == expected


@pytest.mark.parametrize("raw", ["22", "22-8-1", "a-b", "24-8", "22-24", "-1-8", "8-8"])
def test_parse_quiet_hours_rejects_invalid(raw):
    with pytest.raises(ValueError, match="QUIET_HOURS"):
        gn._parse_quiet_hours(raw)


# --- 시간대 판정 ---


@pytest.mark.parametrize(
    "hour, quiet",
    [(21, False), (22, True), (23, True), (0, True), (7, True), (8, False), (12, False)],
)
def test_in_quiet_hours_wrapping_midnight(hour, quiet):
    assert gn._in_quiet_hours(hour, (22, 8)) is quiet


@pytest.mark.parametrize("hour, quiet", [(8, False), (9, True), (17, True), (18, False)])
def test_in_quiet_hours_same_day(hour, quiet):
    assert gn._in_quiet_hours(hour, (9, 18)) is quiet


def test_no_quiet_hours_never_quiet():
    assert not any(gn._in_quiet_hours(h, None) for h in range(24))


# --- load_config ---


def test_load_config_reads_quiet_hours(monkeypatch):
    monkeypatch.setenv("DRY_RUN", "true")
    monkeypatch.setenv("QUIET_HOURS", "22-8")
    monkeypatch.setenv("ENV_FILE", "/nonexistent/.env")
    assert gn.load_config().quiet_hours == (22, 8)


def test_load_config_empty_quiet_hours_means_none(monkeypatch):
    # 워크플로우는 수동 실행에서 QUIET_HOURS를 빈 문자열로 넘깁니다.
    monkeypatch.setenv("DRY_RUN", "true")
    monkeypatch.setenv("QUIET_HOURS", "")
    monkeypatch.setenv("ENV_FILE", "/nonexistent/.env")
    assert gn.load_config().quiet_hours is None


def test_load_config_invalid_quiet_hours_is_config_error(monkeypatch):
    monkeypatch.setenv("DRY_RUN", "true")
    monkeypatch.setenv("QUIET_HOURS", "late")
    monkeypatch.setenv("ENV_FILE", "/nonexistent/.env")
    with pytest.raises(ValueError):
        gn.load_config()


# --- run_once ---


class _Parsed:
    def __init__(self, entries):
        self.feed = {"title": "T", "link": "https://example.com"}
        self.entries = entries


def _cfg(state_file: Path, quiet_hours) -> gn.Config:
    return gn.Config(
        slack_webhook_url="https://hooks.slack.com/x",
        feed_urls=["https://feed.a/"],
        state_file=state_file,
        initial_notify_count=0,
        max_items_per_message=10,
        on_state_miss="skip",
        verify_ssl=True,
        dry_run=True,
        notion_token=None,
        notion_page_id=None,
        quiet_hours=quiet_hours,
    )


@pytest.fixture
def feed_with_new_post(tmp_path, monkeypatch):
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps({"version": 1, "feeds": {"https://feed.a/": {"last_id": "old"}}}),
        encoding="utf-8",
    )
    fetched: list[str] = []
    sent: list[dict] = []

    def fake_fetch(url, *, verify_ssl):
        fetched.append(url)
        return _Parsed([{"id": "new", "link": "https://x/new", "title": "새 글"}, {"id": "old"}])

    monkeypatch.setattr(gn, "fetch_feed", fake_fetch)
    monkeypatch.setattr(gn, "send_to_slack", lambda w, p, *, dry_run: sent.append(p))
    monkeypatch.setattr(gn, "send_to_notion", lambda **k: None)
    return state_file, fetched, sent


def test_quiet_hours_skip_everything_and_leave_state(feed_with_new_post, monkeypatch, caplog):
    state_file, fetched, sent = feed_with_new_post
    before = state_file.read_text(encoding="utf-8")
    monkeypatch.setattr(gn, "_now_kst", lambda: _kst(3, 30))

    with caplog.at_level("INFO", logger="gsai_notifier"):
        assert gn.run_once(_cfg(state_file, (22, 8))) == 0

    assert fetched == [] and sent == []
    assert state_file.read_text(encoding="utf-8") == before
    assert "조용한 시간" in caplog.text


def test_post_from_night_is_sent_in_the_morning(feed_with_new_post, monkeypatch):
    state_file, _fetched, sent = feed_with_new_post
    cfg = _cfg(state_file, (22, 8))

    monkeypatch.setattr(gn, "_now_kst", lambda: _kst(23, 10))
    assert gn.run_once(cfg) == 0
    assert sent == []

    monkeypatch.setattr(gn, "_now_kst", lambda: _kst(8, 5))
    assert gn.run_once(cfg) == 0
    assert len(sent) == 1
    assert (
        json.loads(state_file.read_text(encoding="utf-8"))["feeds"]["https://feed.a/"]["last_id"]
        == "new"
    )


def test_daytime_runs_normally(feed_with_new_post, monkeypatch):
    _state_file, fetched, sent = feed_with_new_post
    monkeypatch.setattr(gn, "_now_kst", lambda: _kst(14, 0))
    assert gn.run_once(_cfg(_state_file, (22, 8))) == 0
    assert fetched == ["https://feed.a/"] and len(sent) == 1


def test_without_quiet_hours_runs_at_night(feed_with_new_post, monkeypatch):
    _state_file, _fetched, sent = feed_with_new_post
    monkeypatch.setattr(gn, "_now_kst", lambda: _kst(3, 0))
    assert gn.run_once(_cfg(_state_file, None)) == 0
    assert len(sent) == 1


def test_now_kst_is_utc_plus_9():
    assert gn._now_kst().utcoffset().total_seconds() == 9 * 3600
