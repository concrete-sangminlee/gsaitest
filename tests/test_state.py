"""load_state / save_state / _prune_sent_articles 테스트 (tmp_path 사용)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import gsai_notifier as gn

# --- load_state ---


def test_load_state_missing_file(tmp_path):
    path = tmp_path / "nope.json"
    state = gn.load_state(path)
    assert state == {"version": 1, "feeds": {}}


def test_load_state_empty_file(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("   \n", encoding="utf-8")
    state = gn.load_state(path)
    assert state == {"version": 1, "feeds": {}}


def test_load_state_invalid_json_returns_default(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    # 예외를 던지지 않고 기본 상태를 반환해야 합니다.
    state = gn.load_state(path)
    assert state == {"version": 1, "feeds": {}}


def test_load_state_legacy_flat_dict_migration(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"https://a/feed": "id-1"}), encoding="utf-8")
    state = gn.load_state(path)
    assert state["version"] == 1
    assert "https://a/feed" in state["feeds"]
    entry = state["feeds"]["https://a/feed"]
    assert entry["last_id"] == "id-1"
    assert "updated_at" in entry


def test_load_state_valid_v1_roundtrip(tmp_path):
    path = tmp_path / "state.json"
    data = {
        "version": 1,
        "feeds": {"https://a/feed": {"last_id": "x", "updated_at": "2024-01-01T00:00:00+00:00"}},
        "sent_articles": {"x": "2024-01-01T00:00:00+00:00"},
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    state = gn.load_state(path)
    assert state == data


def test_load_state_non_dict_returns_default(tmp_path):
    path = tmp_path / "list.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    state = gn.load_state(path)
    assert state == {"version": 1, "feeds": {}}


# --- save_state ---


def test_save_state_atomic_write(tmp_path):
    path = tmp_path / "sub" / "state.json"
    data = {"version": 1, "feeds": {"u": {"last_id": "y"}}}
    gn.save_state(path, data)

    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert json.loads(text) == data
    # 임시 파일이 남아있지 않아야 합니다.
    assert not (tmp_path / "sub" / "state.json.tmp").exists()


def test_save_state_roundtrip_with_load(tmp_path):
    path = tmp_path / "state.json"
    data = {
        "version": 1,
        "feeds": {"https://a/feed": {"last_id": "abc", "updated_at": "2024-01-01T00:00:00+00:00"}},
    }
    gn.save_state(path, data)
    assert gn.load_state(path) == data


# --- _prune_sent_articles ---


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def test_prune_drops_old_keeps_recent():
    now = datetime.now(timezone.utc)
    recent = _iso(now - timedelta(days=1))
    old = _iso(now - timedelta(days=10))
    sent = {"recent": recent, "old": old}
    pruned = gn._prune_sent_articles(sent)
    assert "recent" in pruned
    assert "old" not in pruned
    assert pruned["recent"] == recent


def test_prune_boundary_keeps_within_7_days():
    now = datetime.now(timezone.utc)
    # 7일보다 약간 이내인 항목은 유지되어야 합니다.
    within = _iso(now - timedelta(days=6, hours=23))
    pruned = gn._prune_sent_articles({"within": within})
    assert "within" in pruned


def test_prune_tolerates_malformed_timestamps():
    now = datetime.now(timezone.utc)
    recent = _iso(now - timedelta(hours=1))
    sent = {"good": recent, "bad": "not-a-date", "none": None}
    pruned = gn._prune_sent_articles(sent)
    assert pruned == {"good": recent}


def test_prune_empty():
    assert gn._prune_sent_articles({}) == {}
