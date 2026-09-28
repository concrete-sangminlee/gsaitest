"""_entry_already_sent / _mark_entries_sent 중복 방지 로직 테스트."""

from __future__ import annotations

import gsai_notifier as gn


def test_already_sent_by_uid_in_sent_ids():
    entry = {"id": "UID1", "link": "http://x"}
    assert gn._entry_already_sent(entry, {"UID1"}, {}) is True


def test_already_sent_by_uid_in_sent_articles():
    entry = {"id": "UID1", "link": "http://x"}
    assert gn._entry_already_sent(entry, set(), {"UID1": "ts"}) is True


def test_already_sent_by_link_in_sent_ids():
    # uid는 미기록이지만 link가 sent_ids에 있으면 이미 보낸 것으로 판단합니다.
    entry = {"id": "UID1", "link": "http://x"}
    assert gn._entry_already_sent(entry, {"http://x"}, {}) is True


def test_already_sent_by_link_in_sent_articles():
    entry = {"id": "UID1", "link": "http://x"}
    assert gn._entry_already_sent(entry, set(), {"http://x": "ts"}) is True


def test_not_already_sent():
    entry = {"id": "UID1", "link": "http://x"}
    assert gn._entry_already_sent(entry, {"other"}, {"another": "ts"}) is False


def test_not_already_sent_empty_link_ignored():
    # link가 비어 있으면 link 기반 검사가 오탐하지 않아야 합니다.
    entry = {"id": "UID1", "link": ""}
    assert gn._entry_already_sent(entry, set(), {}) is False


def test_mark_entries_sent_records_uid_and_link():
    entries = [{"id": "UID1", "link": "http://x"}]
    sent_ids: set = set()
    sent_articles: dict[str, str] = {}
    gn._mark_entries_sent(entries, sent_ids, sent_articles)

    assert "UID1" in sent_ids
    assert "http://x" in sent_ids
    assert "UID1" in sent_articles
    assert "http://x" in sent_articles
    # 타임스탬프가 ISO 형식으로 기록되어야 합니다.
    from datetime import datetime

    datetime.fromisoformat(sent_articles["UID1"])
    datetime.fromisoformat(sent_articles["http://x"])


def test_mark_entries_sent_without_link():
    entries = [{"id": "UID1"}]
    sent_ids: set = set()
    sent_articles: dict[str, str] = {}
    gn._mark_entries_sent(entries, sent_ids, sent_articles)

    assert sent_ids == {"UID1"}
    assert set(sent_articles) == {"UID1"}


def test_mark_then_already_sent_roundtrip():
    entry = {"id": "UID1", "link": "http://x"}
    sent_ids: set = set()
    sent_articles: dict[str, str] = {}
    assert gn._entry_already_sent(entry, sent_ids, sent_articles) is False
    gn._mark_entries_sent([entry], sent_ids, sent_articles)
    assert gn._entry_already_sent(entry, sent_ids, sent_articles) is True
