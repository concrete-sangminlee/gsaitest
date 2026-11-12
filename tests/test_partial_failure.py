"""부분 실패 시 재전송/누락이 없는지 검증합니다.

배경: 한 실행 안에서 일부 글은 Slack에 전달됐는데 그 뒤에 실패(뒤쪽 청크 실패,
다른 피드 장애, 실행 중단)가 나면, 전달된 글의 기록이 남지 않아 다음 실행에서
같은 글이 다시 전송될 수 있습니다. 아래 테스트는 (1) 청크 단위로 즉시 이력이
남는지, (2) 피드마다 state.json이 저장되는지, (3) 손상된 state.json에서는 아무것도
보내지 않고 멈추는지를 확인합니다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import gsai_notifier as gn


class FakeParsed:
    def __init__(self, entries, *, title="테스트 피드", link="https://example.com"):
        self.feed = {"title": title, "link": link}
        self.entries = entries


def _entry(uid: str) -> dict:
    return {"id": uid, "link": f"https://x/{uid}", "title": f"글 {uid}"}


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


def _install_feeds(monkeypatch, mapping: dict):
    def fake_fetch(url, *, verify_ssl):
        result = mapping[url]
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(gn, "fetch_feed", fake_fetch)


def _write_state(state_file: Path, feeds: dict) -> None:
    state_file.write_text(json.dumps({"version": 1, "feeds": feeds}), encoding="utf-8")


def _read_state(state_file: Path) -> dict:
    return json.loads(state_file.read_text(encoding="utf-8"))


def _titles(payloads: list[dict]) -> list[str]:
    text = json.dumps(payloads, ensure_ascii=False)
    return sorted({t for t in ("글 new1", "글 new2", "글 new3") if t in text})


@pytest.fixture(autouse=True)
def no_notion(monkeypatch):
    monkeypatch.setattr(gn, "send_to_notion", lambda **kw: None)


# --- 청크 단위 기록 ---


def test_later_chunk_failure_does_not_resend_delivered_chunks(tmp_path, monkeypatch):
    """3개 청크 중 2개 전달 후 실패 → 다음 실행에서는 남은 1개만 전송합니다."""
    state_file = tmp_path / "state.json"
    _write_state(state_file, {"https://feed.a/": {"last_id": "old", "updated_at": "x"}})
    cfg = _make_config(state_file, max_items_per_message=1)
    parsed = FakeParsed([_entry("new3"), _entry("new2"), _entry("new1"), _entry("old")])
    _install_feeds(monkeypatch, {"https://feed.a/": parsed})

    run1: list[dict] = []

    def flaky_slack(webhook_url, payload, *, dry_run):
        if len(run1) >= 2:
            raise RuntimeError("slack down")
        run1.append(payload)

    monkeypatch.setattr(gn, "send_to_slack", flaky_slack)
    assert gn.run_once(cfg) == 3
    assert _titles(run1) == ["글 new1", "글 new2"]

    run2: list[dict] = []
    monkeypatch.setattr(gn, "send_to_slack", lambda w, p, *, dry_run: run2.append(p))
    assert gn.run_once(cfg) == 0

    assert _titles(run2) == ["글 new3"]  # 이미 전달된 new1, new2는 다시 가지 않습니다.
    assert _read_state(state_file)["feeds"]["https://feed.a/"]["last_id"] == "new3"


def test_delivered_chunk_is_in_history_even_when_run_fails(tmp_path, monkeypatch):
    state_file = tmp_path / "state.json"
    _write_state(state_file, {"https://feed.a/": {"last_id": "old", "updated_at": "x"}})
    cfg = _make_config(state_file, max_items_per_message=1)
    _install_feeds(
        monkeypatch,
        {"https://feed.a/": FakeParsed([_entry("new2"), _entry("new1"), _entry("old")])},
    )
    calls = {"n": 0}

    def flaky_slack(webhook_url, payload, *, dry_run):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("slack down")

    monkeypatch.setattr(gn, "send_to_slack", flaky_slack)
    assert gn.run_once(cfg) == 3

    history = _read_state(state_file)["sent_articles"]
    assert "new1" in history and "https://x/new1" in history
    assert "new2" not in history


# --- 피드 단위 저장 ---


def test_state_saved_after_each_feed_even_if_run_aborts(tmp_path, monkeypatch):
    """앞 피드에서 보낸 뒤 실행이 중단돼도(예: 취소) 그 전송 기록은 디스크에 남습니다."""
    state_file = tmp_path / "state.json"
    _write_state(
        state_file,
        {
            "https://feed.a/": {"last_id": "old", "updated_at": "x"},
            "https://feed.b/": {"last_id": "old", "updated_at": "x"},
        },
    )
    cfg = _make_config(state_file, feed_urls=["https://feed.a/", "https://feed.b/"])
    _install_feeds(
        monkeypatch,
        {
            "https://feed.a/": FakeParsed([_entry("new1"), _entry("old")]),
            "https://feed.b/": KeyboardInterrupt(),  # 실행 중단을 흉내냅니다.
        },
    )
    sent: list[dict] = []
    monkeypatch.setattr(gn, "send_to_slack", lambda w, p, *, dry_run: sent.append(p))

    with pytest.raises(KeyboardInterrupt):
        gn.run_once(cfg)

    assert len(sent) == 1
    state = _read_state(state_file)
    assert state["feeds"]["https://feed.a/"]["last_id"] == "new1"
    assert "new1" in state["sent_articles"]


def test_fetch_failure_on_one_feed_keeps_other_feeds_progress(tmp_path, monkeypatch):
    """한 피드 장애(exit 2)여도 다른 피드의 전송 기록은 저장됩니다(워크플로우가 커밋)."""
    state_file = tmp_path / "state.json"
    _write_state(
        state_file,
        {
            "https://feed.a/": {"last_id": "old", "updated_at": "x"},
            "https://feed.b/": {"last_id": "old", "updated_at": "x"},
        },
    )
    cfg = _make_config(state_file, feed_urls=["https://feed.a/", "https://feed.b/"])
    _install_feeds(
        monkeypatch,
        {
            "https://feed.a/": FakeParsed([_entry("new1"), _entry("old")]),
            "https://feed.b/": RuntimeError("timeout"),
        },
    )
    monkeypatch.setattr(gn, "send_to_slack", lambda w, p, *, dry_run: None)

    assert gn.run_once(cfg) == 2
    state = _read_state(state_file)
    assert state["feeds"]["https://feed.a/"]["last_id"] == "new1"
    assert state["feeds"]["https://feed.b/"]["last_id"] == "old"


# --- 손상된 state.json ---


@pytest.mark.parametrize(
    "content",
    [
        "",
        "<<<<<<< HEAD\n{}\n=======\n{}\n>>>>>>> main\n",
        "[1, 2, 3]",
        '{"version": 1, "feeds": []}',
    ],
    ids=["empty", "conflict-markers", "not-a-dict", "feeds-not-a-dict"],
)
def test_corrupt_state_stops_without_sending_or_overwriting(tmp_path, monkeypatch, content):
    state_file = tmp_path / "state.json"
    state_file.write_text(content, encoding="utf-8")
    cfg = _make_config(state_file, initial_notify_count=1)
    _install_feeds(monkeypatch, {"https://feed.a/": FakeParsed([_entry("new1")])})
    sent: list[dict] = []
    monkeypatch.setattr(gn, "send_to_slack", lambda w, p, *, dry_run: sent.append(p))

    assert gn.run_once(cfg) == 4
    assert sent == []
    assert state_file.read_text(encoding="utf-8") == content  # 파일을 건드리지 않습니다.


def test_missing_state_file_is_still_a_normal_first_run(tmp_path, monkeypatch):
    state_file = tmp_path / "state.json"
    cfg = _make_config(state_file)
    _install_feeds(monkeypatch, {"https://feed.a/": FakeParsed([_entry("new1")])})
    monkeypatch.setattr(gn, "send_to_slack", lambda w, p, *, dry_run: None)

    assert gn.run_once(cfg) == 0
    assert _read_state(state_file)["feeds"]["https://feed.a/"]["last_id"] == "new1"


def test_load_state_strict_raises_on_invalid_json(tmp_path):
    path = tmp_path / "state.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(gn.StateFileError):
        gn.load_state(path, strict=True)
    # 기본(non-strict) 동작은 그대로입니다.
    assert gn.load_state(path) == {"version": 1, "feeds": {}}


def test_load_state_strict_accepts_legacy_flat_format(tmp_path):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"https://feed.a/": "abc"}), encoding="utf-8")
    state = gn.load_state(path, strict=True)
    assert state["feeds"]["https://feed.a/"]["last_id"] == "abc"


def test_non_dict_sent_articles_is_ignored(tmp_path, monkeypatch):
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps(
            {
                "version": 1,
                "feeds": {"https://feed.a/": {"last_id": "new1"}},
                "sent_articles": ["broken"],
            }
        ),
        encoding="utf-8",
    )
    cfg = _make_config(state_file)
    _install_feeds(monkeypatch, {"https://feed.a/": FakeParsed([_entry("new1")])})
    assert gn.run_once(cfg) == 0
    assert _read_state(state_file)["sent_articles"] == {}
