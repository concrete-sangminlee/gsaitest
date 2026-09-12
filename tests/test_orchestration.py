"""오케스트레이션 계층(_dispatch_items/_process_feed/run_once) 통합 테스트.

IO 경계(fetch_feed/send_to_slack/send_to_notion)만 monkeypatch로 대체하고,
_dispatch_items/_process_feed/run_once 자체는 실제 로직을 그대로 실행합니다.
상태는 tmp_path의 state.json으로 검증합니다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import gsai_notifier as gn


class FakeParsed:
    """feedparser.parse 결과를 흉내내는 가벼운 객체.

    .feed 는 title/link 를 .get 으로 제공하는 dict, .entries 는 dict 항목 리스트.
    """

    def __init__(self, entries, *, title="테스트 피드", link="https://example.com"):
        self.feed = {"title": title, "link": link}
        self.entries = entries


def _entry(uid: str, *, title: str | None = None, link: str | None = None) -> dict:
    """id/link/title 을 가진 평범한 dict 항목을 만듭니다."""
    e: dict = {"id": uid, "title": title if title is not None else f"글 {uid}"}
    if link is not None:
        e["link"] = link
    return e


def _make_config(state_file: Path, **overrides) -> gn.Config:
    """테스트용 Config. dry_run 기본 True 로 Slack IO 를 우회합니다."""
    params = dict(
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
    )
    params.update(overrides)
    return gn.Config(**params)


@pytest.fixture
def recorder(monkeypatch):
    """send_to_slack/send_to_notion 호출을 기록하는 레코더를 설치합니다."""

    slack_calls: list[dict] = []
    notion_calls: list[dict] = []

    def fake_slack(webhook_url, payload, *, dry_run):
        slack_calls.append({"webhook": webhook_url, "payload": payload, "dry_run": dry_run})

    def fake_notion(*, token, page_id, feed_title, items, dry_run=False):
        notion_calls.append(
            {
                "token": token,
                "page_id": page_id,
                "feed_title": feed_title,
                "items": list(items),
                "dry_run": dry_run,
            }
        )

    monkeypatch.setattr(gn, "send_to_slack", fake_slack)
    monkeypatch.setattr(gn, "send_to_notion", fake_notion)

    return {"slack": slack_calls, "notion": notion_calls}


def _install_feeds(monkeypatch, mapping: dict):
    """feed_url -> FakeParsed 또는 Exception 을 반환하는 fetch_feed 를 설치합니다."""

    def fake_fetch(url, *, verify_ssl):
        result = mapping[url]
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(gn, "fetch_feed", fake_fetch)


def _read_state(state_file: Path) -> dict:
    return json.loads(state_file.read_text(encoding="utf-8"))


# --- run_once 종료 코드 집계 ---


def test_fetch_failure_yields_exit_2(tmp_path, monkeypatch, recorder):
    state_file = tmp_path / "state.json"
    cfg = _make_config(state_file, feed_urls=["https://feed.a/"])
    _install_feeds(monkeypatch, {"https://feed.a/": RuntimeError("boom")})

    assert gn.run_once(cfg) == 2
    # 가져오기 실패해도 상태 파일은 저장됩니다.
    assert state_file.exists()
    assert recorder["slack"] == []


def test_slack_failure_on_new_entries_yields_exit_3(tmp_path, monkeypatch, recorder):
    state_file = tmp_path / "state.json"
    # 기존 last_id 를 심어 정상 new-entries 경로를 타게 합니다.
    state_file.write_text(
        json.dumps(
            {"version": 1, "feeds": {"https://feed.a/": {"last_id": "old", "updated_at": "x"}}}
        ),
        encoding="utf-8",
    )
    cfg = _make_config(state_file, feed_urls=["https://feed.a/"])
    parsed = FakeParsed([_entry("new1"), _entry("old")])
    _install_feeds(monkeypatch, {"https://feed.a/": parsed})

    def boom_slack(*_a, **_k):
        raise RuntimeError("slack down")

    monkeypatch.setattr(gn, "send_to_slack", boom_slack)

    assert gn.run_once(cfg) == 3
    # Slack 실패 시 상태(last_id)는 전진하지 않습니다.
    state = _read_state(state_file)
    assert state["feeds"]["https://feed.a/"]["last_id"] == "old"


def test_last_non_zero_wins_aggregation(tmp_path, monkeypatch, recorder):
    state_file = tmp_path / "state.json"
    # feed.b 는 정상 new-entries 경로에서 Slack 실패(3), feed.a 는 fetch 실패(2).
    state_file.write_text(
        json.dumps(
            {
                "version": 1,
                "feeds": {"https://feed.b/": {"last_id": "b_old", "updated_at": "x"}},
            }
        ),
        encoding="utf-8",
    )
    cfg = _make_config(state_file, feed_urls=["https://feed.a/", "https://feed.b/"])
    parsed_b = FakeParsed([_entry("b_new"), _entry("b_old")])
    _install_feeds(
        monkeypatch,
        {"https://feed.a/": RuntimeError("boom"), "https://feed.b/": parsed_b},
    )

    def boom_slack(*_a, **_k):
        raise RuntimeError("slack down")

    monkeypatch.setattr(gn, "send_to_slack", boom_slack)

    # feed.a -> 2, feed.b -> 3 순서이므로 마지막 non-zero(3)가 이깁니다.
    assert gn.run_once(cfg) == 3


def test_last_non_zero_wins_reverse_order(tmp_path, monkeypatch, recorder):
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps(
            {
                "version": 1,
                "feeds": {"https://feed.b/": {"last_id": "b_old", "updated_at": "x"}},
            }
        ),
        encoding="utf-8",
    )
    cfg = _make_config(state_file, feed_urls=["https://feed.b/", "https://feed.a/"])
    parsed_b = FakeParsed([_entry("b_new"), _entry("b_old")])
    _install_feeds(
        monkeypatch,
        {"https://feed.a/": RuntimeError("boom"), "https://feed.b/": parsed_b},
    )

    def boom_slack(*_a, **_k):
        raise RuntimeError("slack down")

    monkeypatch.setattr(gn, "send_to_slack", boom_slack)

    # feed.b -> 3, feed.a -> 2 순서이므로 마지막 non-zero(2)가 이깁니다.
    assert gn.run_once(cfg) == 2


def test_success_after_failure_keeps_failure_code(tmp_path, monkeypatch, recorder):
    state_file = tmp_path / "state.json"
    cfg = _make_config(state_file, feed_urls=["https://feed.a/", "https://feed.b/"])
    # feed.a fetch 실패(2), feed.b 는 첫 실행(성공, 0).
    parsed_b = FakeParsed([_entry("b1")])
    _install_feeds(
        monkeypatch,
        {"https://feed.a/": RuntimeError("boom"), "https://feed.b/": parsed_b},
    )

    # 성공한 feed.b 뒤에도 실패 코드 2가 유지되어야 합니다.
    assert gn.run_once(cfg) == 2


# --- 첫 실행 동작 ---


def test_first_run_no_notify_sets_bookmark_without_slack(tmp_path, monkeypatch, recorder):
    state_file = tmp_path / "state.json"
    cfg = _make_config(state_file, feed_urls=["https://feed.a/"], initial_notify_count=0)
    parsed = FakeParsed([_entry("newest"), _entry("older")])
    _install_feeds(monkeypatch, {"https://feed.a/": parsed})

    assert gn.run_once(cfg) == 0
    # Slack 전송이 일어나지 않아야 합니다.
    assert recorder["slack"] == []
    # last_id 는 최신 항목 uid 로 설정되어야 합니다.
    state = _read_state(state_file)
    assert state["feeds"]["https://feed.a/"]["last_id"] == "newest"


def test_first_run_with_notify_count_sends(tmp_path, monkeypatch, recorder):
    state_file = tmp_path / "state.json"
    cfg = _make_config(state_file, feed_urls=["https://feed.a/"], initial_notify_count=2)
    parsed = FakeParsed([_entry("n1"), _entry("n2"), _entry("n3")])
    _install_feeds(monkeypatch, {"https://feed.a/": parsed})

    assert gn.run_once(cfg) == 0
    # initial_notify_count 만큼(청크 1건) Slack 전송이 발생해야 합니다.
    assert len(recorder["slack"]) == 1
    state = _read_state(state_file)
    assert state["feeds"]["https://feed.a/"]["last_id"] == "n1"


# --- 정상 new-entries 경로 ---


def test_new_entries_send_then_advance_last_id(tmp_path, monkeypatch, recorder):
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps(
            {"version": 1, "feeds": {"https://feed.a/": {"last_id": "old", "updated_at": "x"}}}
        ),
        encoding="utf-8",
    )
    cfg = _make_config(state_file, feed_urls=["https://feed.a/"])
    parsed = FakeParsed([_entry("new2"), _entry("new1"), _entry("old")])
    _install_feeds(monkeypatch, {"https://feed.a/": parsed})

    assert gn.run_once(cfg) == 0
    # 새 글 2건이 한 청크로 전송됩니다.
    assert len(recorder["slack"]) == 1
    # 성공 후 last_id 가 최신으로 전진합니다.
    state = _read_state(state_file)
    assert state["feeds"]["https://feed.a/"]["last_id"] == "new2"


def test_no_new_entries_advances_bookmark_without_send(tmp_path, monkeypatch, recorder):
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps(
            {"version": 1, "feeds": {"https://feed.a/": {"last_id": "top", "updated_at": "x"}}}
        ),
        encoding="utf-8",
    )
    cfg = _make_config(state_file, feed_urls=["https://feed.a/"])
    parsed = FakeParsed([_entry("top"), _entry("below")])
    _install_feeds(monkeypatch, {"https://feed.a/": parsed})

    assert gn.run_once(cfg) == 0
    assert recorder["slack"] == []
    state = _read_state(state_file)
    assert state["feeds"]["https://feed.a/"]["last_id"] == "top"


# --- Notion 베스트에포트 ---


def test_notion_failure_does_not_fail_run_or_block_advance(tmp_path, monkeypatch, recorder):
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps(
            {"version": 1, "feeds": {"https://feed.a/": {"last_id": "old", "updated_at": "x"}}}
        ),
        encoding="utf-8",
    )
    cfg = _make_config(
        state_file,
        feed_urls=["https://feed.a/"],
        notion_token="tok",
        notion_page_id="pid",
    )
    parsed = FakeParsed([_entry("new1"), _entry("old")])
    _install_feeds(monkeypatch, {"https://feed.a/": parsed})

    # send_to_slack 은 정상, send_to_notion 만 예외를 던지도록 교체합니다.
    slack_calls: list = []

    def ok_slack(webhook_url, payload, *, dry_run):
        slack_calls.append(payload)

    def boom_notion(**_k):
        raise RuntimeError("notion down")

    monkeypatch.setattr(gn, "send_to_slack", ok_slack)
    monkeypatch.setattr(gn, "send_to_notion", boom_notion)

    # Notion 실패는 run 을 실패시키지 않습니다(exit 0 유지).
    assert gn.run_once(cfg) == 0
    assert len(slack_calls) == 1
    # 상태 전진도 막지 않습니다.
    state = _read_state(state_file)
    assert state["feeds"]["https://feed.a/"]["last_id"] == "new1"


# --- Slack 실패 시 모든 경로에서 exit 3 + 상태 미전진 ---


def test_slack_failure_on_first_run_yields_exit_3(tmp_path, monkeypatch, recorder):
    state_file = tmp_path / "state.json"
    # last_id 가 없는(첫 실행) 상태 + initial_notify_count>0 로 dispatch 를 태웁니다.
    cfg = _make_config(state_file, feed_urls=["https://feed.a/"], initial_notify_count=2)
    parsed = FakeParsed([_entry("n1"), _entry("n2"), _entry("n3")])
    _install_feeds(monkeypatch, {"https://feed.a/": parsed})

    def boom_slack(*_a, **_k):
        raise RuntimeError("slack down")

    monkeypatch.setattr(gn, "send_to_slack", boom_slack)

    assert gn.run_once(cfg) == 3
    # Slack 실패에도 state.json 은 저장됩니다.
    assert state_file.exists()
    state = _read_state(state_file)
    # 첫 실행 기준점(last_id)이 전진하지 않아야 합니다(피드가 기록되지 않음).
    assert "https://feed.a/" not in state["feeds"]


def test_slack_failure_on_state_miss_send_yields_exit_3(tmp_path, monkeypatch, recorder):
    state_file = tmp_path / "state.json"
    # last_id 가 피드에 없는(상태 불일치) 상황을 만듭니다.
    state_file.write_text(
        json.dumps(
            {"version": 1, "feeds": {"https://feed.a/": {"last_id": "missing", "updated_at": "x"}}}
        ),
        encoding="utf-8",
    )
    cfg = _make_config(state_file, feed_urls=["https://feed.a/"], on_state_miss="send")
    parsed = FakeParsed([_entry("a2"), _entry("a1")])
    _install_feeds(monkeypatch, {"https://feed.a/": parsed})

    def boom_slack(*_a, **_k):
        raise RuntimeError("slack down")

    monkeypatch.setattr(gn, "send_to_slack", boom_slack)

    assert gn.run_once(cfg) == 3
    assert state_file.exists()
    state = _read_state(state_file)
    # 상태 불일치 재설정 전에 실패하므로 last_id 는 전진하지 않고 그대로 유지됩니다.
    assert state["feeds"]["https://feed.a/"]["last_id"] == "missing"


# --- 실행 요약 로그 (#6) ---


def test_run_summary_log_counts_items_sent(tmp_path, monkeypatch, recorder, caplog):
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps(
            {"version": 1, "feeds": {"https://feed.a/": {"last_id": "old", "updated_at": "x"}}}
        ),
        encoding="utf-8",
    )
    cfg = _make_config(state_file, feed_urls=["https://feed.a/"])
    parsed = FakeParsed([_entry("new2"), _entry("new1"), _entry("old")])
    _install_feeds(monkeypatch, {"https://feed.a/": parsed})

    with caplog.at_level("INFO", logger="gsai_notifier"):
        assert gn.run_once(cfg) == 0

    summaries = [r for r in caplog.records if r.getMessage().startswith("run summary:")]
    assert len(summaries) == 1
    msg = summaries[0].getMessage()
    assert "feeds=1" in msg
    assert "items_sent=2" in msg
    assert "fetch_failures=0" in msg
    assert "slack_failures=0" in msg
    assert "exit=0" in msg


def test_run_summary_log_counts_fetch_failure(tmp_path, monkeypatch, recorder, caplog):
    state_file = tmp_path / "state.json"
    cfg = _make_config(state_file, feed_urls=["https://feed.a/"])
    _install_feeds(monkeypatch, {"https://feed.a/": RuntimeError("boom")})

    with caplog.at_level("INFO", logger="gsai_notifier"):
        assert gn.run_once(cfg) == 2

    summaries = [r for r in caplog.records if r.getMessage().startswith("run summary:")]
    assert len(summaries) == 1
    msg = summaries[0].getMessage()
    assert "feeds=1" in msg
    assert "items_sent=0" in msg
    assert "fetch_failures=1" in msg
    assert "slack_failures=0" in msg
    assert "exit=2" in msg
