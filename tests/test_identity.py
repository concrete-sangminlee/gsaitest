"""글 식별자 테스트: 글 번호 주소 우선, guid 충돌 시 누락 방지, 기존 state 무중단 이전.

배경: 이 사이트의 guid는 제목을 앞 30자에서 자른 주소입니다(실측 70개 중 62개가 정확히
30자). 앞부분이 같은 새 글은 guid가 같아져 '이미 보낸 글'로 판정되고 조용히 누락될 수
있습니다. 글 번호 주소(https://gsai.snu.ac.kr/18486/)는 유일하므로 이것으로 식별합니다.
"""

from __future__ import annotations

import json
from pathlib import Path

import gsai_notifier as gn

SITE = "https://gsai.snu.ac.kr"


def _post(num: int, guid: str, title: str | None = None) -> dict:
    return {"id": f"{SITE}/{guid}/", "link": f"{SITE}/{num}/", "title": title or f"글 {num}"}


class _Parsed:
    def __init__(self, entries):
        self.feed = {"title": "협동과정 인공지능 전공", "link": f"{SITE}/"}
        self.entries = entries


def _cfg(state_file: Path) -> gn.Config:
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
    )


def _run(monkeypatch, state_file: Path, entries) -> list[str]:
    """한 번 실행하고, Slack으로 보낸 글 제목 목록을 돌려줍니다."""
    sent: list[str] = []

    def fake_slack(webhook_url, payload, *, dry_run):
        for b in payload["attachments"][0]["blocks"]:
            if b["type"] == "section":
                sent.append(b["text"]["text"].split("*")[1])

    monkeypatch.setattr(gn, "fetch_feed", lambda url, *, verify_ssl: _Parsed(entries))
    monkeypatch.setattr(gn, "send_to_slack", fake_slack)
    monkeypatch.setattr(gn, "send_to_notion", lambda **k: None)
    assert gn.run_once(_cfg(state_file)) == 0
    return sent


def _state(state_file: Path) -> dict:
    return json.loads(state_file.read_text(encoding="utf-8"))


# --- entry_uid ---


def test_uid_prefers_numeric_permalink_over_guid():
    assert (
        gn.entry_uid(_post(18486, "2026학년도-2학기-파안장학문화재단-석박사")) == f"{SITE}/18486/"
    )


def test_uid_accepts_permalink_without_trailing_slash_and_http():
    assert gn.entry_uid({"id": "g", "link": "http://example.com/12"}) == "http://example.com/12"


def test_uid_falls_back_to_guid_for_slug_links():
    e = {"id": "guid-1", "link": f"{SITE}/some-slug/"}
    assert gn.entry_uid(e) == "guid-1"


def test_uid_ignores_numeric_paths_that_are_not_post_permalinks():
    for link in (f"{SITE}/2026/11/26/post/", f"{SITE}/18486/?p=1", f"{SITE}/category/1/"):
        assert gn.entry_uid({"id": "guid-1", "link": link}) == "guid-1"


# --- guid 충돌 ---


def test_new_post_sharing_guid_with_sent_post_is_not_dropped(tmp_path, monkeypatch):
    """앞 30자가 같은 두 공지: 두 번째 공지도 보내야 합니다(이전 코드는 조용히 누락)."""
    guid = "2026년-하반기-BK21-4단계-대학원혁"
    first = _post(18535, guid, "2026년 하반기 BK21 … 지원사업 공고")
    second = _post(18578, guid, "2026년 하반기 BK21 … 지원사업 공고 (추가)")
    older = _post(18500, "older-post")
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps({"version": 1, "feeds": {"https://feed.a/": {"last_id": f"{SITE}/18500/"}}}),
        encoding="utf-8",
    )

    assert _run(monkeypatch, state_file, [first, older]) == [first["title"]]
    assert _run(monkeypatch, state_file, [second, first, older]) == [second["title"]]


def test_new_post_sharing_guid_with_bookmark_is_not_dropped(tmp_path, monkeypatch):
    guid = "같은-앞부분"
    bookmark = _post(100, guid)
    newer = _post(101, guid)
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps({"version": 1, "feeds": {"https://feed.a/": {"last_id": f"{SITE}/100/"}}}),
        encoding="utf-8",
    )
    assert _run(monkeypatch, state_file, [newer, bookmark]) == ["글 101"]


# --- 기존 state.json(guid 기준점)에서 이전 ---


def _legacy_state(state_file: Path, last_guid: str, sent: dict | None = None) -> None:
    state_file.write_text(
        json.dumps(
            {
                "version": 1,
                "feeds": {"https://feed.a/": {"last_id": f"{SITE}/{last_guid}/"}},
                "sent_articles": sent or {},
            }
        ),
        encoding="utf-8",
    )


def test_legacy_bookmark_migrates_silently_without_sending(tmp_path, monkeypatch, caplog):
    state_file = tmp_path / "state.json"
    _legacy_state(state_file, "top-guid")
    entries = [_post(200, "top-guid"), _post(199, "below-guid")]

    with caplog.at_level("WARNING", logger="gsai_notifier"):
        assert _run(monkeypatch, state_file, entries) == []
    assert "복구" not in caplog.text  # 발행 시각 복구 경로를 타지 않습니다.
    assert _state(state_file)["feeds"]["https://feed.a/"]["last_id"] == f"{SITE}/200/"

    before = state_file.read_text(encoding="utf-8")
    assert _run(monkeypatch, state_file, entries) == []
    assert state_file.read_text(encoding="utf-8") == before  # 이전 후에는 바뀌지 않습니다.


def test_legacy_bookmark_with_new_post_sends_only_the_new_one(tmp_path, monkeypatch, caplog):
    state_file = tmp_path / "state.json"
    _legacy_state(state_file, "top-guid")
    entries = [_post(201, "new-guid"), _post(200, "top-guid"), _post(199, "below-guid")]

    with caplog.at_level("INFO", logger="gsai_notifier"):
        assert _run(monkeypatch, state_file, entries) == ["글 201"]
    assert "state_recoveries=0" in caplog.text
    assert _state(state_file)["feeds"]["https://feed.a/"]["last_id"] == f"{SITE}/201/"


def test_legacy_history_still_prevents_resending(tmp_path, monkeypatch):
    """이전 전 이력(guid+link 키)에 있는 글은 이전 후에도 다시 보내지 않습니다."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    state_file = tmp_path / "state.json"
    _legacy_state(
        state_file,
        "deleted-bookmark",  # 기준 글이 피드에서 사라진 상황
        sent={f"{SITE}/sent-guid/": now, f"{SITE}/300/": now},
    )
    state = _state(state_file)
    state["feeds"]["https://feed.a/"]["last_published"] = "2026-01-01T00:00:00+00:00"
    state_file.write_text(json.dumps(state), encoding="utf-8")
    sent_post = _post(300, "sent-guid")
    sent_post["published_parsed"] = (2026, 11, 20, 0, 0, 0, 0, 0, 0)
    new_post = _post(301, "new-guid")
    new_post["published_parsed"] = (2026, 11, 21, 0, 0, 0, 0, 0, 0)

    assert _run(monkeypatch, state_file, [new_post, sent_post]) == ["글 301"]


def test_history_records_one_key_per_post_after_switch(tmp_path, monkeypatch):
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps({"version": 1, "feeds": {"https://feed.a/": {"last_id": f"{SITE}/1/"}}}),
        encoding="utf-8",
    )
    _run(monkeypatch, state_file, [_post(2, "g2"), _post(1, "g1")])
    assert list(_state(state_file)["sent_articles"]) == [f"{SITE}/2/"]
