"""format_slack_payload Block Kit 구조 테스트 (실제 함수 출력에 대해 검증)."""

from __future__ import annotations

import gsai_notifier as gn


def _block_types(payload: dict) -> list[str]:
    return [b["type"] for b in payload["attachments"][0]["blocks"]]


def _make_payload(items, *, index=1, total=1):
    return gn.format_slack_payload(
        "테스트 피드",
        items,
        feed_url="https://gsai.snu.ac.kr/feed/",
        site_url="https://gsai.snu.ac.kr",
        index=index,
        total=total,
    )


def test_top_level_structure_and_unfurl_flags():
    items = [{"title": "글1", "link": "http://x/1", "published": "2024-01-01"}]
    payload = _make_payload(items)

    assert payload["unfurl_links"] is False
    assert payload["unfurl_media"] is False
    assert isinstance(payload["attachments"], list)
    assert payload["attachments"][0]["color"] == "#003876"


def test_fallback_text_includes_count():
    items = [{"title": "a", "link": "http://x/1"}, {"title": "b", "link": "http://x/2"}]
    payload = _make_payload(items)
    assert "2" in payload["text"]
    assert "테스트 피드" in payload["text"]


def test_expected_block_types_present():
    items = [{"title": "글1", "link": "http://x/1", "published": "2024-01-01"}]
    types = _block_types(_make_payload(items))
    assert "header" in types
    assert "context" in types
    assert "divider" in types
    assert "section" in types
    assert "actions" in types


def test_subtitle_hides_index_total_when_total_is_one():
    items = [{"title": "글1", "link": "http://x/1"}]
    payload = _make_payload(items, index=1, total=1)
    context_block = payload["attachments"][0]["blocks"][1]
    subtitle = context_block["elements"][0]["text"]
    assert "(1/1)" not in subtitle
    assert "/" not in subtitle


def test_subtitle_shows_index_total_when_total_gt_one():
    items = [{"title": "글1", "link": "http://x/1"}]
    payload = _make_payload(items, index=2, total=3)
    context_block = payload["attachments"][0]["blocks"][1]
    subtitle = context_block["elements"][0]["text"]
    assert "(2/3)" in subtitle


def _sections(payload: dict) -> list[dict]:
    return [b for b in payload["attachments"][0]["blocks"] if b["type"] == "section"]


def test_section_has_button_accessory_when_link_present():
    items = [{"title": "글1", "link": "http://x/1"}]
    section = _sections(_make_payload(items))[0]
    assert "accessory" in section
    assert section["accessory"]["type"] == "button"
    assert section["accessory"]["url"] == "http://x/1"


def test_section_has_no_accessory_when_link_absent():
    items = [{"title": "글1"}]
    section = _sections(_make_payload(items))[0]
    assert "accessory" not in section


def test_one_section_per_item():
    items = [
        {"title": "글1", "link": "http://x/1"},
        {"title": "글2", "link": "http://x/2"},
        {"title": "글3"},
    ]
    sections = _sections(_make_payload(items))
    assert len(sections) == 3


def test_section_text_includes_title():
    items = [{"title": "특별한제목", "link": "http://x/1"}]
    section = _sections(_make_payload(items))[0]
    assert "특별한제목" in section["text"]["text"]
