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


def test_multi_hyphen_slug_url_normalizes_to_hex():
    url = f"https://www.notion.so/My-Cool-Page-Title-{_HEX}"
    assert gn._normalize_notion_page_id(url) == _HEX


def test_hyphenated_uuid_in_notion_url_normalizes():
    uuid = "27e2cbf5-6573-8031-9715-fa24fb5d4d15"
    url = f"https://www.notion.so/My-Cool-Page-{uuid}"
    assert gn._normalize_notion_page_id(url) == _HEX


def test_uppercase_hex_normalizes_to_lowercase():
    upper = _HEX.upper()
    assert gn._normalize_notion_page_id(upper) == _HEX
    url = f"https://www.notion.so/Notice-{upper}"
    assert gn._normalize_notion_page_id(url) == _HEX


def test_non_matching_still_returns_unchanged():
    assert gn._normalize_notion_page_id("not-a-page-id") == "not-a-page-id"


def test_long_hex_run_not_truncated_to_first_32():
    """33자 이상 연속 hex 런은 앞 32자로 잘리지 않고 원본(passthrough)을 반환해야 함."""
    long_hex = "a" * 40  # 40자 연속 hex
    result = gn._normalize_notion_page_id(long_hex)
    # 앞 32자를 잘못 반환하면 안 됩니다.
    assert result != long_hex[:32]
    # 경계 매칭/32자 검사에 모두 걸리지 않으므로 원본(공백 제거본)이 그대로 반환됩니다.
    assert result == long_hex


def test_thirty_three_hex_run_not_truncated():
    """정확히 33자 연속 hex 런도 앞 32자로 잘리지 않아야 함."""
    hex33 = "27e2cbf5657380319715fa24fb5d4d15a"  # 32자 + 1
    result = gn._normalize_notion_page_id(hex33)
    assert result != _HEX
    assert result == hex33


def test_bounded_32_hex_in_context_still_normalizes():
    """비-hex 문맥으로 둘러싸인 32자 hex는 정상적으로 정규화됨(회귀 방지)."""
    # 앞뒤가 hex가 아닌 문자(하이픈/슬래시)로 경계지어져 있습니다.
    url = f"https://www.notion.so/page-{_HEX}-view"
    assert gn._normalize_notion_page_id(url) == _HEX
