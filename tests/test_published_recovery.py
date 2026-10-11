"""발행 시각 기반 복구 로직 테스트.

기준 글(last_id)이 피드에서 사라지면 위치 기반 탐색은 실패합니다. 이때
`last_published`가 있으면 발행 시각으로 새 글을 복구해 누락을 막습니다.
이 모듈은 그 복구 경로와 보조 헬퍼들을 검증합니다.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import gsai_notifier as gn


def _entry(uid: str, published: datetime | None = None) -> dict:
    e: dict = {"id": uid, "link": f"https://x/{uid}", "title": uid}
    if published is not None:
        e["published_parsed"] = published.timetuple()[:6]
    return e


def _iso(dt: datetime) -> str:
    return dt.isoformat()


NOW = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)


# --- entry_published_dt ---


def test_published_dt_from_published_parsed():
    e = _entry("a", NOW)
    assert gn.entry_published_dt(e) == NOW


def test_published_dt_falls_back_to_updated_parsed():
    e = {"id": "a", "updated_parsed": NOW.timetuple()[:6]}
    assert gn.entry_published_dt(e) == NOW


def test_published_dt_prefers_published_over_updated():
    older = NOW - timedelta(days=5)
    e = {
        "id": "a",
        "published_parsed": NOW.timetuple()[:6],
        "updated_parsed": older.timetuple()[:6],
    }
    assert gn.entry_published_dt(e) == NOW


def test_published_dt_none_when_absent():
    assert gn.entry_published_dt({"id": "a"}) is None


def test_published_dt_none_on_malformed():
    assert gn.entry_published_dt({"id": "a", "published_parsed": ("x", "y")}) is None


def test_published_dt_accepts_struct_time():
    import time

    struct = time.struct_time(NOW.timetuple())
    assert gn.entry_published_dt({"id": "a", "published_parsed": struct}) == NOW


# --- _newest_published_iso ---


def test_newest_published_picks_max():
    entries = [_entry("a", NOW - timedelta(days=2)), _entry("b", NOW)]
    assert gn._newest_published_iso(entries) == _iso(NOW)


def test_newest_published_ignores_undated_entries():
    entries = [_entry("a"), _entry("b", NOW)]
    assert gn._newest_published_iso(entries) == _iso(NOW)


def test_newest_published_none_when_no_dates():
    assert gn._newest_published_iso([_entry("a"), _entry("b")]) is None


def test_newest_published_empty():
    assert gn._newest_published_iso([]) is None


# --- _new_entries_by_published ---


def test_by_published_returns_only_newer_oldest_first():
    cutoff = NOW - timedelta(days=3)
    entries = [
        _entry("new2", NOW),
        _entry("new1", NOW - timedelta(days=1)),
        _entry("old", NOW - timedelta(days=5)),
    ]
    got = gn._new_entries_by_published(entries, _iso(cutoff))
    assert [e["id"] for e in got] == ["new1", "new2"]


def test_by_published_excludes_entry_exactly_at_cutoff():
    entries = [_entry("same", NOW)]
    assert gn._new_entries_by_published(entries, _iso(NOW)) == []


def test_by_published_none_without_cutoff():
    assert gn._new_entries_by_published([_entry("a", NOW)], None) is None


def test_by_published_none_on_malformed_cutoff():
    assert gn._new_entries_by_published([_entry("a", NOW)], "not-a-date") is None


def test_by_published_none_when_any_entry_undated():
    # 일부만 날짜가 있으면 섣불리 판단하지 않고 포기합니다(과다 전송 방지).
    entries = [_entry("a", NOW), _entry("b")]
    assert gn._new_entries_by_published(entries, _iso(NOW - timedelta(days=1))) is None


def test_by_published_accepts_naive_cutoff_as_utc():
    cutoff = (NOW - timedelta(days=1)).replace(tzinfo=None)
    got = gn._new_entries_by_published([_entry("a", NOW)], cutoff.isoformat())
    assert [e["id"] for e in got] == ["a"]


# --- compute_new_entries 복구 경로 ---


def test_recovers_when_last_id_missing_but_published_known():
    entries = [_entry("c", NOW), _entry("b", NOW - timedelta(days=1))]
    cutoff = _iso(NOW - timedelta(days=2))
    new, newest_id, found = gn.compute_new_entries(entries, "deleted", cutoff)
    assert found is True  # 복구 성공 → ON_STATE_MISS 경로로 빠지지 않습니다.
    assert [e["id"] for e in new] == ["b", "c"]
    assert newest_id == "c"


def test_recovery_sends_nothing_when_all_entries_are_old():
    entries = [_entry("b", NOW - timedelta(days=5))]
    new, _newest, found = gn.compute_new_entries(entries, "deleted", _iso(NOW))
    assert found is True
    assert new == []


def test_no_recovery_without_last_published_keeps_state_miss():
    entries = [_entry("c", NOW), _entry("b", NOW - timedelta(days=1))]
    new, newest_id, found = gn.compute_new_entries(entries, "deleted")
    assert found is False
    assert new == []
    assert newest_id == "c"


def test_no_recovery_when_entries_lack_dates():
    entries = [_entry("c"), _entry("b")]
    _new, _newest, found = gn.compute_new_entries(entries, "deleted", _iso(NOW))
    assert found is False


def test_positional_match_wins_over_published_recovery():
    # 기준 글이 피드에 있으면 발행 시각은 쓰이지 않고 위치 기준이 우선입니다.
    entries = [_entry("c", NOW), _entry("b", NOW - timedelta(days=1)), _entry("a", NOW)]
    new, _newest, found = gn.compute_new_entries(entries, "b", _iso(NOW - timedelta(days=10)))
    assert found is True
    assert [e["id"] for e in new] == ["c"]


def test_last_published_ignored_on_first_run():
    entries = [_entry("c", NOW)]
    new, newest_id, found = gn.compute_new_entries(entries, None, _iso(NOW - timedelta(days=9)))
    assert (new, newest_id, found) == ([], "c", True)


def test_recovery_logs_warning(caplog):
    entries = [_entry("c", NOW)]
    with caplog.at_level("WARNING", logger="gsai_notifier"):
        gn.compute_new_entries(entries, "deleted", _iso(NOW - timedelta(days=1)))
    assert any("발행 시각으로 복구" in r.getMessage() for r in caplog.records)


# --- _feed_state_record / _feed_state_changed ---


def test_feed_state_record_includes_last_published():
    rec = gn._feed_state_record("c", [_entry("c", NOW)])
    assert rec["last_id"] == "c"
    assert rec["last_published"] == _iso(NOW)
    assert "updated_at" in rec


def test_feed_state_record_omits_last_published_without_dates():
    rec = gn._feed_state_record("c", [_entry("c")])
    assert "last_published" not in rec


def test_feed_state_unchanged_ignores_updated_at():
    current = {"last_id": "c", "last_published": _iso(NOW), "updated_at": "old"}
    desired = {"last_id": "c", "last_published": _iso(NOW), "updated_at": "new"}
    assert gn._feed_state_changed(current, desired) is False


def test_feed_state_changed_on_last_id():
    current = {"last_id": "b", "updated_at": "x"}
    desired = {"last_id": "c", "updated_at": "y"}
    assert gn._feed_state_changed(current, desired) is True


def test_feed_state_changed_when_backfilling_last_published():
    # 과거 상태 파일에는 last_published가 없으므로 한 번은 기록되어야 합니다.
    current = {"last_id": "c", "updated_at": "x"}
    desired = {"last_id": "c", "last_published": _iso(NOW), "updated_at": "y"}
    assert gn._feed_state_changed(current, desired) is True
