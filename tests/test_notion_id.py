"""_normalize_notion_page_id 정규화 로직 테스트."""

from __future__ import annotations

import gsai_notifier as gn

_HEX = "27e2cbf5657380319715fa24fb5d4d15"


def test_notion_so_url_with_trailing_hex_after_hyphen():
    url = f"https://www.notion.so/Notice-{_HEX}"
    assert gn._normalize_notion_page_id(url) == _HEX


def test_notion_so_url_query_stripped():
    url = f"https://www.notion.so/Notice-{_HEX}?v=abc123"
    assert gn._normalize_notion_page_id(url) == _HEX


def test_notion_so_url_fragment_stripped():
    url = f"https://www.notion.so/Notice-{_HEX}#section"
    assert gn._normalize_notion_page_id(url) == _HEX


def test_notion_so_url_no_hyphen_finds_hex():
    url = f"https://www.notion.so/{_HEX}"
    assert gn._normalize_notion_page_id(url) == _HEX


def test_hyphenated_uuid_to_hex():
    uuid = "27e2cbf5-6573-8031-9715-fa24fb5d4d15"
    assert gn._normalize_notion_page_id(uuid) == _HEX


def test_already_32_hex_passthrough():
    assert gn._normalize_notion_page_id(_HEX) == _HEX


def test_non_matching_returned_as_is():
    assert gn._normalize_notion_page_id("not-a-page-id") == "not-a-page-id"


def test_empty_returns_empty():
    assert gn._normalize_notion_page_id("") == ""
    assert gn._normalize_notion_page_id(None) == ""


def test_strips_surrounding_whitespace():
    assert gn._normalize_notion_page_id(f"  {_HEX}  ") == _HEX
