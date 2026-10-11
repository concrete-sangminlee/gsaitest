"""state.json 쓰기 억제 및 발행 시각 복구의 통합(run_once) 검증.

배경: 새 글이 없는데도 매 실행 `updated_at`을 갱신하면 state.json이 항상
변경되어 워크플로우의 "변경 시에만 커밋" 판정이 무력화되고 의미 없는 커밋이
쌓입니다. 아래 테스트는 실질 변화가 없을 때 파일이 그대로 남는지, 그리고
기준 글이 사라진 경우 발행 시각으로 새 글이 복구되는지를 확인합니다.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import gsai_notifier as gn

NOW = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)


class FakeParsed:
    def __init__(self, entries, *, title="테스트 피드", link="https://example.com"):
        self.feed = {"title": title, "link": link}
        self.entries = entries


def _entry(uid: str, published: datetime | None = None) -> dict:
    e: dict = {"id": uid, "link": f"https://x/{uid}", "title": f"글 {uid}"}
    if published is not None:
        e["published_parsed"] = published.timetuple()[:6]
    return e


def _make_config(state_file: Path, **overrides) -> gn.Config:
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
    slack_calls: list[dict] = []

    def fake_slack(webhook_url, payload, *, dry_run):
        slack_calls.append({"payload": payload})

    monkeypatch.setattr(gn, "send_to_slack", fake_slack)
    monkeypatch.setattr(gn, "send_to_notion", lambda **kw: None)
    return {"slack": slack_calls}


def _install_feeds(monkeypatch, mapping: dict):
    def fake_fetch(url, *, verify_ssl):
        result = mapping[url]
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(gn, "fetch_feed", fake_fetch)


def _write_state(state_file: Path, feed_state: dict) -> None:
    state_file.write_text(
        json.dumps({"version": 1, "feeds": {"https://feed.a/": feed_state}}),
        encoding="utf-8",
    )


def _read_state(state_file: Path) -> dict:
    return json.loads(state_file.read_text(encoding="utf-8"))


# --- 불필요한 쓰기 억제 ---


def test_no_new_entries_leaves_state_byte_identical(tmp_path, monkeypatch, recorder):
    """새 글이 없고 기준점도 그대로면 state.json 내용이 바뀌지 않아야 합니다."""
    state_file = tmp_path / "state.json"
    _write_state(
        state_file,
        {"last_id": "top", "last_published": NOW.isoformat(), "updated_at": "2020-01-01T00:00:00"},
    )
    cfg = _make_config(state_file)
    _install_feeds(
        monkeypatch,
        {"https://feed.a/": FakeParsed([_entry("top", NOW), _entry("below", NOW - timedelta(1))])},
    )

    before = _read_state(state_file)
    assert gn.run_once(cfg) == 0
    after = _read_state(state_file)

    assert recorder["slack"] == []
    # updated_at 이 그대로여야 합니다(= 커밋이 생기지 않습니다).
    assert after["feeds"]["https://feed.a/"] == before["feeds"]["https://feed.a/"]


def test_repeated_runs_do_not_churn_state(tmp_path, monkeypatch, recorder):
    """여러 번 실행해도 변화가 없으면 피드 레코드가 계속 동일해야 합니다."""
    state_file = tmp_path / "state.json"
    _write_state(
        state_file, {"last_id": "top", "last_published": NOW.isoformat(), "updated_at": "x"}
    )
    cfg = _make_config(state_file)
    _install_feeds(monkeypatch, {"https://feed.a/": FakeParsed([_entry("top", NOW)])})

    snapshots = []
    for _ in range(3):
        assert gn.run_once(cfg) == 0
        snapshots.append(_read_state(state_file)["feeds"]["https://feed.a/"])

    assert snapshots[0] == snapshots[1] == snapshots[2]


def test_first_run_backfills_last_published_once(tmp_path, monkeypatch, recorder):
    """과거 상태 파일(last_published 없음)에는 한 번 기록하고, 그 뒤로는 멈춥니다."""
    state_file = tmp_path / "state.json"
    _write_state(state_file, {"last_id": "top", "updated_at": "x"})  # 과거 포맷
    cfg = _make_config(state_file)
    _install_feeds(monkeypatch, {"https://feed.a/": FakeParsed([_entry("top", NOW)])})

    assert gn.run_once(cfg) == 0
    first = _read_state(state_file)["feeds"]["https://feed.a/"]
    assert first["last_published"] == NOW.isoformat()  # 백필됨

    assert gn.run_once(cfg) == 0
    second = _read_state(state_file)["feeds"]["https://feed.a/"]
    assert second == first  # 두 번째 실행은 쓰지 않습니다.
    assert recorder["slack"] == []


def test_new_entries_still_advance_state(tmp_path, monkeypatch, recorder):
    """실제 새 글이 있을 때는 기존처럼 전송하고 기준점을 전진시킵니다."""
    state_file = tmp_path / "state.json"
    _write_state(
        state_file,
        {"last_id": "old", "last_published": (NOW - timedelta(days=2)).isoformat()},
    )
    cfg = _make_config(state_file)
    _install_feeds(
        monkeypatch,
        {"https://feed.a/": FakeParsed([_entry("new", NOW), _entry("old", NOW - timedelta(2))])},
    )

    assert gn.run_once(cfg) == 0
    assert len(recorder["slack"]) == 1
    rec = _read_state(state_file)["feeds"]["https://feed.a/"]
    assert rec["last_id"] == "new"
    assert rec["last_published"] == NOW.isoformat()


# --- 발행 시각 복구 (누락 방지) ---


def test_deleted_bookmark_recovers_and_sends_new_posts(tmp_path, monkeypatch, recorder):
    """기준 글이 삭제되어도 발행 시각으로 새 글을 찾아 전송해야 합니다.

    기존 동작(ON_STATE_MISS=skip)은 이 경우 아무것도 보내지 않고 기준점만
    재설정하여 그 사이 글들을 영구히 놓쳤습니다.
    """
    state_file = tmp_path / "state.json"
    _write_state(
        state_file,
        {
            "last_id": "deleted-post",
            "last_published": (NOW - timedelta(days=2)).isoformat(),
        },
    )
    cfg = _make_config(state_file)
    _install_feeds(
        monkeypatch,
        {"https://feed.a/": FakeParsed([_entry("n2", NOW), _entry("n1", NOW - timedelta(days=1))])},
    )

    assert gn.run_once(cfg) == 0

    assert len(recorder["slack"]) == 1
    sent = json.dumps(recorder["slack"][0]["payload"], ensure_ascii=False)
    assert "글 n1" in sent and "글 n2" in sent
    rec = _read_state(state_file)["feeds"]["https://feed.a/"]
    assert rec["last_id"] == "n2"


def test_recovery_does_not_resend_already_sent_posts(tmp_path, monkeypatch, recorder):
    """복구 경로에서도 sent_articles 이력이 중복을 막아야 합니다."""
    state_file = tmp_path / "state.json"
    # sent_articles 의 시각은 실제 현재 시각과 비교되어 정리되므로(보관 기간),
    # 고정 NOW 가 아니라 실제 시각을 기준으로 기록해야 합니다.
    already = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    state_file.write_text(
        json.dumps(
            {
                "version": 1,
                "feeds": {
                    "https://feed.a/": {
                        "last_id": "deleted-post",
                        "last_published": (NOW - timedelta(days=2)).isoformat(),
                    }
                },
                # n1 은 이미 보낸 글로 기록되어 있습니다.
                "sent_articles": {"n1": already, "https://x/n1": already},
            }
        ),
        encoding="utf-8",
    )
    cfg = _make_config(state_file)
    _install_feeds(
        monkeypatch,
        {"https://feed.a/": FakeParsed([_entry("n2", NOW), _entry("n1", NOW - timedelta(days=1))])},
    )

    assert gn.run_once(cfg) == 0

    sent = json.dumps(recorder["slack"], ensure_ascii=False)
    assert "글 n2" in sent
    assert "글 n1" not in sent  # 이미 보낸 글은 다시 가지 않습니다.


def test_without_last_published_state_miss_still_skips(tmp_path, monkeypatch, recorder):
    """last_published 가 없으면 기존 ON_STATE_MISS=skip 동작을 유지합니다."""
    state_file = tmp_path / "state.json"
    _write_state(state_file, {"last_id": "deleted-post", "updated_at": "x"})
    cfg = _make_config(state_file)
    _install_feeds(
        monkeypatch,
        {"https://feed.a/": FakeParsed([_entry("n2", NOW), _entry("n1", NOW - timedelta(1))])},
    )

    assert gn.run_once(cfg) == 0
    assert recorder["slack"] == []  # skip 정책대로 전송하지 않습니다.
    assert _read_state(state_file)["feeds"]["https://feed.a/"]["last_id"] == "n2"


def test_recovery_counted_in_run_summary(tmp_path, monkeypatch, recorder, caplog):
    state_file = tmp_path / "state.json"
    _write_state(
        state_file,
        {"last_id": "deleted", "last_published": (NOW - timedelta(days=2)).isoformat()},
    )
    cfg = _make_config(state_file)
    _install_feeds(monkeypatch, {"https://feed.a/": FakeParsed([_entry("n1", NOW)])})

    with caplog.at_level("INFO", logger="gsai_notifier"):
        assert gn.run_once(cfg) == 0

    summary = [r.getMessage() for r in caplog.records if "run summary" in r.getMessage()]
    assert summary and "state_recoveries=1" in summary[0]


def test_no_recovery_count_on_normal_run(tmp_path, monkeypatch, recorder, caplog):
    state_file = tmp_path / "state.json"
    _write_state(
        state_file,
        {"last_id": "old", "last_published": (NOW - timedelta(days=2)).isoformat()},
    )
    cfg = _make_config(state_file)
    _install_feeds(
        monkeypatch,
        {"https://feed.a/": FakeParsed([_entry("new", NOW), _entry("old", NOW - timedelta(2))])},
    )

    with caplog.at_level("INFO", logger="gsai_notifier"):
        assert gn.run_once(cfg) == 0

    summary = [r.getMessage() for r in caplog.records if "run summary" in r.getMessage()]
    assert summary and "state_recoveries=0" in summary[0]
