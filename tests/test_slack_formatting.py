"""Slack 메시지 표시 품질 테스트: 이스케이프, 서식 기호 무력화, 게시판 표시, 헤더 정리."""

from __future__ import annotations

import gsai_notifier as gn


def _payload(items, feed_title="협동과정 인공지능 전공"):
    return gn.format_slack_payload(
        feed_title,
        items,
        feed_url="https://gsai.snu.ac.kr/feed/",
        site_url="https://gsai.snu.ac.kr/",
        index=1,
        total=1,
    )


def _blocks(payload):
    return payload["attachments"][0]["blocks"]


def _sections(payload):
    return [b["text"]["text"] for b in _blocks(payload) if b["type"] == "section"]


def _entry(title, *, tags=(), link="https://gsai.snu.ac.kr/18627/"):
    e = {"id": "x", "title": title, "link": link}
    if tags:
        e["tags"] = [{"term": t, "scheme": None, "label": None} for t in tags]
    return e


# --- 이스케이프 ---


def test_ampersand_and_angle_brackets_are_escaped():
    p = _payload([_entry("SNU-LG AI Tech & Talent <공지>")])
    text = _sections(p)[0]
    assert "Tech &amp; Talent &lt;공지&gt;" in text
    assert "<공지>" not in text


def test_mention_like_title_cannot_ping_channel():
    p = _payload([_entry("<!channel> 긴급")])
    assert "<!channel>" not in _sections(p)[0]
    assert "<!channel>" not in p["text"]


def test_tilde_and_asterisk_do_not_break_formatting():
    # 마감일 표기의 '~'가 두 번이면 그 사이가 취소선이 되고, '*'는 굵게 구간을 끊습니다.
    p = _payload([_entry("신청(~11/30), 면접(~12/5) *필독*")])
    text = _sections(p)[0]
    assert text.startswith("> *신청(～11/30), 면접(～12/5) ＊필독＊*")
    assert "~" not in text
    assert text.count("*") == 2  # 우리가 감싼 굵게 표시만 남습니다.


def test_plain_korean_title_unchanged():
    p = _payload([_entry("2027학년도 1학기 대학원생 교내장학금 신청 안내")])
    assert "> *2027학년도 1학기 대학원생 교내장학금 신청 안내*" in _sections(p)[0]


def test_fallback_text_is_escaped_but_keeps_tilde():
    # 맨 위 한 줄은 굵게 감싸지 않으므로 이스케이프만 합니다.
    p = _payload([_entry("A & B (~11/30)")])
    assert p["text"] == "📢 A &amp; B (~11/30)"


# --- 게시판(카테고리) 표시 ---


def test_entry_categories_drops_wordpress_default():
    e = _entry("t", tags=["분류되지 않음", "장학/국제"])
    assert gn.entry_categories(e) == ["장학/국제"]


def test_entry_categories_keeps_order_and_dedups():
    e = _entry("t", tags=["채용/행사/세미나", "Uncategorized", "장학/국제", "채용/행사/세미나"])
    assert gn.entry_categories(e) == ["채용/행사/세미나", "장학/국제"]


def test_entry_categories_tolerates_missing_or_odd_tags():
    assert gn.entry_categories({"title": "t"}) == []
    assert gn.entry_categories({"tags": [{"term": None}, "bad", {}]}) == []


def test_section_shows_category_and_date():
    e = _entry("장학금 안내", tags=["분류되지 않음", "장학/국제"])
    e["published_parsed"] = (2026, 11, 21, 1, 0, 0, 0, 0, 0)
    text = _sections(_payload([e]))[0]
    assert text == "> *장학금 안내*\n> 🏷️ 장학/국제  ·  📅 2026. 11. 21"


def test_section_without_category_shows_only_date():
    e = _entry("공지")
    e["published_parsed"] = (2026, 11, 21, 1, 0, 0, 0, 0, 0)
    assert _sections(_payload([e]))[0] == "> *공지*\n> 📅 2026. 11. 21"


def test_section_without_category_or_date_is_title_only():
    assert _sections(_payload([_entry("공지")]))[0] == "> *공지*"


# --- 헤더 정리 ---


def test_clean_feed_title_strips_wordpress_archive_suffix():
    assert gn._clean_feed_title("장학/국제 Archives - 협동과정 인공지능 전공") == "장학/국제"
    assert gn._clean_feed_title("협동과정 인공지능 전공") == "협동과정 인공지능 전공"
    assert gn._clean_feed_title(" Archives - 사이트") == "Archives - 사이트"


def test_header_uses_clean_board_name():
    p = _payload([_entry("t")], feed_title="장학/국제 Archives - 협동과정 인공지능 전공")
    header = _blocks(p)[0]
    assert header["type"] == "header"
    assert header["text"]["text"] == "📢 장학/국제 새 글"


def test_header_is_truncated_to_slack_limit():
    p = _payload([_entry("t")], feed_title="가" * 200)
    text = _blocks(p)[0]["text"]["text"]
    assert len(text) <= 150
    assert text.endswith("…")


def test_process_feed_passes_clean_title_to_notion(tmp_path, monkeypatch):
    """Slack뿐 아니라 Notion·로그도 정리된 게시판 이름을 씁니다."""
    import json

    class Parsed:
        feed = {"title": "장학/국제 Archives - 협동과정 인공지능 전공", "link": ""}
        entries = [{"id": "new", "link": "https://x/new", "title": "n"}, {"id": "old"}]

    state = tmp_path / "state.json"
    state.write_text(json.dumps({"version": 1, "feeds": {"f": {"last_id": "old"}}}), "utf-8")
    seen: list[str] = []
    monkeypatch.setattr(gn, "fetch_feed", lambda url, *, verify_ssl: Parsed())
    monkeypatch.setattr(gn, "send_to_slack", lambda *a, **k: None)
    monkeypatch.setattr(gn, "send_to_notion", lambda **k: seen.append(k["feed_title"]))
    cfg = gn.Config(
        slack_webhook_url="x",
        feed_urls=["f"],
        state_file=state,
        initial_notify_count=0,
        max_items_per_message=10,
        on_state_miss="skip",
        verify_ssl=True,
        dry_run=True,
        notion_token="t",
        notion_page_id="p",
    )
    assert gn.run_once(cfg) == 0
    assert seen == ["장학/국제"]
