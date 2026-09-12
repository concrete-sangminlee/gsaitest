"""entry_* 헬퍼 및 compute_new_entries 순수 로직 테스트 (평범한 dict 사용)."""

from __future__ import annotations

import gsai_notifier as gn

# --- entry_uid 우선순위: id > guid > link > title|published ---


def test_entry_uid_prefers_id():
    entry = {"id": "ID1", "guid": "GUID1", "link": "L1", "title": "T", "published": "P"}
    assert gn.entry_uid(entry) == "ID1"


def test_entry_uid_falls_back_to_guid():
    entry = {"guid": "GUID1", "link": "L1", "title": "T", "published": "P"}
    assert gn.entry_uid(entry) == "GUID1"


def test_entry_uid_falls_back_to_link():
    entry = {"link": "L1", "title": "T", "published": "P"}
    assert gn.entry_uid(entry) == "L1"


def test_entry_uid_falls_back_to_title_published():
    entry = {"title": " Hello ", "published": " 2024-01-01 "}
    assert gn.entry_uid(entry) == "Hello|2024-01-01"


def test_entry_uid_uses_updated_when_no_published():
    entry = {"title": "Hello", "updated": "2024-02-02"}
    assert gn.entry_uid(entry) == "Hello|2024-02-02"


def test_entry_uid_strips_whitespace_on_id():
    entry = {"id": "  ID1  "}
    assert gn.entry_uid(entry) == "ID1"


def test_entry_uid_skips_blank_id():
    # 공백뿐인 id는 무시하고 다음 후보(link)로 넘어갑니다.
    entry = {"id": "   ", "link": "L1"}
    assert gn.entry_uid(entry) == "L1"


# --- entry_title / entry_link / entry_pub ---


def test_entry_title_strips():
    assert gn.entry_title({"title": "  Title  "}) == "Title"


def test_entry_title_missing():
    assert gn.entry_title({}) == ""


def test_entry_link_strips():
    assert gn.entry_link({"link": "  http://x  "}) == "http://x"


def test_entry_link_missing():
    assert gn.entry_link({}) == ""


def test_entry_pub_prefers_published():
    assert gn.entry_pub({"published": " 2024-01-01 ", "updated": "2024-02-02"}) == "2024-01-01"


def test_entry_pub_falls_back_to_updated():
    assert gn.entry_pub({"updated": " 2024-02-02 "}) == "2024-02-02"


def test_entry_pub_missing():
    assert gn.entry_pub({}) == ""


# --- compute_new_entries ---


def _e(uid: str) -> dict:
    return {"id": uid}


def test_compute_new_entries_empty():
    new, newest, found = gn.compute_new_entries([], None)
    assert new == []
    assert newest is None
    assert found is True


def test_compute_new_entries_last_seen_none():
    entries = [_e("c"), _e("b"), _e("a")]
    new, newest, found = gn.compute_new_entries(entries, None)
    assert new == []
    assert newest == "c"
    assert found is True


def test_compute_new_entries_found_returns_oldest_first():
    # 피드는 최신순(c,b,a). last_seen=a이면 새 글은 c,b이고 오래된 순(b,c)으로 반환됩니다.
    entries = [_e("c"), _e("b"), _e("a")]
    new, newest, found = gn.compute_new_entries(entries, "a")
    assert [gn.entry_uid(e) for e in new] == ["b", "c"]
    assert newest == "c"
    assert found is True


def test_compute_new_entries_all_new_when_last_seen_at_bottom():
    entries = [_e("c"), _e("b"), _e("a")]
    new, newest, found = gn.compute_new_entries(entries, "c")
    # last_seen이 첫 항목이면 새 글이 없습니다.
    assert new == []
    assert newest == "c"
    assert found is True


def test_compute_new_entries_not_found():
    entries = [_e("c"), _e("b"), _e("a")]
    new, newest, found = gn.compute_new_entries(entries, "zzz")
    assert new == []
    assert newest == "c"
    assert found is False
