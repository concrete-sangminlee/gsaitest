#!/usr/bin/env python3
"""
GSAI(https://gsai.snu.ac.kr) RSS/Atom 피드의 새 글을 감지해 Slack으로 알리는 스크립트입니다.

핵심 아이디어(비유):
- RSS 피드는 “신문 배달 목록”이고,
- 상태 파일(state.json)은 “어제 어디까지 읽었는지 표시해둔 책갈피”입니다.
- 매번 피드를 읽고 책갈피 이후의 항목만 Slack으로 보내면 됩니다.

필수 환경변수:
- SLACK_WEBHOOK_URL: Slack Incoming Webhook URL (DRY_RUN=true면 없어도 됨)

주요 환경변수:
- FEED_URLS: 감시할 피드 URL(쉼표 구분), 기본값: https://gsai.snu.ac.kr/feed/
- STATE_FILE: 마지막으로 본 글 ID 저장 경로, 기본값: .state/state.json

실행 예:
  python gsai_notifier.py
  DRY_RUN=true python gsai_notifier.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    import feedparser
except ImportError:  # pragma: no cover
    # feedparser는 fetch_feed 안에서만 필요합니다. 순수 로직/테스트가
    # 이 라이브러리 없이도 모듈을 import할 수 있도록 지연 처리합니다.
    feedparser = None

try:
    import requests
except ImportError:  # pragma: no cover
    # requests는 fetch_feed/send_to_slack 안에서만 필요합니다.
    requests = None

try:
    # 로컬에서 .env 파일을 쓰고 싶은 경우를 지원합니다.
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

try:
    from notion_client import Client
except ImportError:
    Client = None


# 피드 항목을 다루는 순수 헬퍼들이 feedparser에 강하게 묶이지 않도록
# 가벼운 타입 별칭을 사용합니다. 런타임에는 feedparser의 항목(dict 유사)이
# 그대로 들어오며, 테스트에서는 평범한 dict를 넣어도 동작합니다.
Entry = Mapping[str, Any]


LOG = logging.getLogger("gsai_notifier")

_SENT_ARTICLES_MAX_AGE_DAYS = 7

# SNU 방화벽이 봇 UA를 차단하므로 브라우저 UA를 사용합니다(USER_AGENT로 재정의 가능).
_USER_AGENT = os.getenv(
    "USER_AGENT",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
)

_ACCEPT_HEADER = (
    "application/rss+xml, application/atom+xml, application/xml, text/xml;q=0.9, */*;q=0.1"
)


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    v = value.strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _parse_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value.strip())
    except Exception:
        return default


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _chunked(items: list[Any], size: int) -> Iterable[list[Any]]:
    if size <= 0:
        yield items
        return
    for i in range(0, len(items), size):
        yield items[i : i + size]


@dataclass(frozen=True)
class Config:
    slack_webhook_url: str | None
    feed_urls: list[str]
    state_file: Path
    initial_notify_count: int
    max_items_per_message: int
    on_state_miss: str  # "skip" | "send"
    verify_ssl: bool
    dry_run: bool
    notion_token: str | None
    notion_page_id: str | None


def load_config() -> Config:
    # 1) .env(옵션) 로드
    # Cursor 환경에서는 .env 파일 생성이 제한될 수 있어 env.example을 참고해 직접 export해도 됩니다.
    if load_dotenv is not None:
        env_file = os.getenv("ENV_FILE", ".env")
        if Path(env_file).exists():
            load_dotenv(env_file, override=False)

    # 2) 환경변수 파싱
    slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL") or None
    feed_urls_raw = os.getenv("FEED_URLS", "https://gsai.snu.ac.kr/feed/")
    feed_urls = [u.strip() for u in feed_urls_raw.split(",") if u.strip()]

    state_file = Path(os.getenv("STATE_FILE", ".state/state.json"))
    initial_notify_count = _parse_int(os.getenv("INITIAL_NOTIFY_COUNT"), 0)
    max_items_per_message = _parse_int(os.getenv("MAX_ITEMS_PER_MESSAGE"), 10)
    on_state_miss = (os.getenv("ON_STATE_MISS", "skip") or "skip").strip().lower()
    verify_ssl = _parse_bool(os.getenv("VERIFY_SSL"), True)
    dry_run = _parse_bool(os.getenv("DRY_RUN"), False)

    if not feed_urls:
        raise ValueError("FEED_URLS가 비어 있습니다. 최소 1개 피드 URL이 필요합니다.")

    if on_state_miss not in {"skip", "send"}:
        raise ValueError("ON_STATE_MISS는 skip 또는 send 여야 합니다.")

    if not dry_run and not slack_webhook_url:
        raise ValueError("SLACK_WEBHOOK_URL이 필요합니다. (또는 DRY_RUN=true로 테스트)")

    notion_token = os.getenv("NOTION_TOKEN")
    notion_page_id = os.getenv("NOTION_PAGE_ID")

    return Config(
        slack_webhook_url=slack_webhook_url,
        feed_urls=feed_urls,
        state_file=state_file,
        initial_notify_count=max(0, initial_notify_count),
        max_items_per_message=max(1, max_items_per_message),
        on_state_miss=on_state_miss,
        verify_ssl=verify_ssl,
        dry_run=dry_run,
        notion_token=notion_token,
        notion_page_id=notion_page_id,
    )


def load_state(path: Path) -> dict[str, Any]:
    """
    상태 파일 포맷(v1):
    {
      "version": 1,
      "feeds": {
        "<feed_url>": {"last_id": "...", "updated_at": "..."}
      }
    }
    """
    default_state: dict[str, Any] = {"version": 1, "feeds": {}}
    if not path.exists():
        return default_state

    # 빈 파일/깨진 JSON 등으로 봇이 멈추지 않도록 방어적으로 읽습니다.
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as e:
        LOG.warning("상태 파일 읽기 실패(%s). 초기 상태로 시작합니다: %s", path, e)
        return default_state

    if not raw.strip():
        LOG.info("상태 파일이 비어 있습니다(%s). 초기 상태로 시작합니다.", path)
        return default_state

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        LOG.warning("상태 파일 JSON 파싱 실패(%s). 초기 상태로 시작합니다: %s", path, e)
        return default_state

    # 과거에 단순 dict 형태로 저장했을 경우를 흡수
    if isinstance(data, dict) and "feeds" not in data and "version" not in data:
        feeds = {k: {"last_id": v, "updated_at": _now_iso()} for k, v in data.items()}
        return {"version": 1, "feeds": feeds}

    if not isinstance(data, dict) or "feeds" not in data:
        return default_state

    return data


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def fetch_feed(url: str, *, verify_ssl: bool) -> Any:
    if requests is None or feedparser is None:  # pragma: no cover
        raise RuntimeError(
            "requests/feedparser 라이브러리가 설치되지 않았습니다. pip install -r requirements.txt"
        )

    # SNU 방화벽(WAF)이 봇 성격의 User-Agent를 차단하므로 일반 브라우저로 요청합니다.
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": _ACCEPT_HEADER,
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    resp = requests.get(url, headers=headers, timeout=20, verify=verify_ssl)
    resp.raise_for_status()

    # 방화벽 차단 페이지는 200으로 내려오므로 본문을 직접 확인합니다.
    if b"waf/error" in resp.content or b"snucert.snu.ac.kr" in resp.content:
        raise RuntimeError(f"방화벽(WAF)에 차단되었습니다. User-Agent 확인이 필요합니다: {url}")

    parsed = feedparser.parse(resp.content)

    # 항목이 0개면 정상 응답이 아닐 가능성이 큽니다(차단/장애/포맷 변경).
    # 조용히 "새 글 없음"으로 넘어가면 알림이 멈춘 걸 알 수 없으므로 실패로 처리합니다.
    if not parsed.entries:
        reason = f" ({parsed.get('bozo_exception')})" if parsed.get("bozo") else ""
        raise RuntimeError(f"피드에 항목이 없습니다{reason}: {url}")

    return parsed


def entry_uid(entry: Entry) -> str:
    # WordPress RSS는 보통 guid / link가 안정적입니다.
    for key in ("id", "guid", "link"):
        v = entry.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    title = (entry.get("title") or "").strip()
    published = (entry.get("published") or entry.get("updated") or "").strip()
    return f"{title}|{published}"


def entry_title(entry: Entry) -> str:
    return (entry.get("title") or "").strip()


def entry_link(entry: Entry) -> str:
    return (entry.get("link") or "").strip()


def entry_pub(entry: Entry) -> str:
    return (entry.get("published") or entry.get("updated") or "").strip()


def _prune_sent_articles(sent_articles: dict[str, str]) -> dict[str, str]:
    """오래된 전송 이력을 제거합니다."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=_SENT_ARTICLES_MAX_AGE_DAYS)
    pruned: dict[str, str] = {}
    for key, ts in sent_articles.items():
        try:
            dt = datetime.fromisoformat(ts)
            if dt >= cutoff:
                pruned[key] = ts
        except (ValueError, TypeError):
            pass
    return pruned


def _entry_already_sent(
    entry: Entry,
    sent_ids: set,
    sent_articles: dict[str, str],
) -> bool:
    """한 번이라도 보낸 글인지 uid와 link 모두로 확인합니다."""
    uid = entry_uid(entry)
    link = entry_link(entry)
    if uid in sent_ids or uid in sent_articles:
        return True
    if link and (link in sent_ids or link in sent_articles):
        return True
    return False


def _mark_entries_sent(
    entries: Iterable[Entry],
    sent_ids: set,
    sent_articles: dict[str, str],
) -> None:
    """보낸 글의 uid와 link를 모두 기록합니다."""
    now = _now_iso()
    for e in entries:
        uid = entry_uid(e)
        link = entry_link(e)
        sent_ids.add(uid)
        sent_articles[uid] = now
        if link:
            sent_ids.add(link)
            sent_articles[link] = now


def compute_new_entries(
    entries: list[Entry],
    last_seen_id: str | None,
) -> tuple[list[Entry], str | None, bool]:
    """
    returns: (new_entries_oldest_first, newest_id, last_seen_found_in_feed)
    """
    if not entries:
        return [], None, True

    newest_id = entry_uid(entries[0])
    if last_seen_id is None:
        return [], newest_id, True

    new_entries: list[Entry] = []
    found = False
    for e in entries:
        if entry_uid(e) == last_seen_id:
            found = True
            break
        new_entries.append(e)

    if not found:
        return [], newest_id, False

    new_entries.reverse()  # 오래된 글부터 알림
    return new_entries, newest_id, True


def _format_date_kr(pub_str: str, entry: Entry) -> str:
    """날짜 문자열을 한국식 포맷으로 변환합니다."""
    try:
        parsed_time = entry.get("published_parsed") or entry.get("updated_parsed")
        if parsed_time:
            y, mo, d, h, mi, s = parsed_time[:6]
            dt = datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc)
            kst = timezone(timedelta(hours=9))
            dt_kst = dt.astimezone(kst)
            return dt_kst.strftime("%Y. %m. %d")
    except Exception:
        pass
    return pub_str[:10] if pub_str else ""


def _now_kst_str() -> str:
    """현재 KST 시각을 포맷팅합니다."""
    kst = timezone(timedelta(hours=9))
    now = datetime.now(kst)
    return now.strftime("%Y. %m. %d  %H:%M KST")


def format_slack_payload(
    feed_title: str,
    items: list[Entry],
    *,
    feed_url: str,
    site_url: str,
    index: int,
    total: int,
) -> dict[str, Any]:
    """Slack Block Kit 기반 프리미엄 페이로드를 생성합니다."""
    count = len(items)
    subtitle = f"🔔 {count}건의 새로운 글"
    if total > 1:
        subtitle += f"  ({index}/{total})"

    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"📢 {feed_title} 새 글",
                "emoji": True,
            },
        },
        {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": subtitle},
            ],
        },
        {"type": "divider"},
    ]

    for e in items:
        title = entry_title(e) or "(제목 없음)"
        link = entry_link(e)
        pub = entry_pub(e)
        date_str = _format_date_kr(pub, e)

        section_text = f"> *{title}*"
        if date_str:
            section_text += f"\n> 📅 {date_str}"

        section: dict[str, Any] = {
            "type": "section",
            "text": {"type": "mrkdwn", "text": section_text},
        }
        if link:
            section["accessory"] = {
                "type": "button",
                "text": {"type": "plain_text", "text": "확인하기", "emoji": True},
                "url": link,
                "style": "primary",
            }
        blocks.append(section)

    # 하단 영역
    view_url = site_url or feed_url.split("/feed")[0] or feed_url
    blocks.extend(
        [
            {"type": "divider"},
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "📋 전체 글 보기", "emoji": True},
                        "url": view_url,
                    },
                ],
            },
            {
                "type": "context",
                "elements": [
                    {"type": "mrkdwn", "text": f"🏫 GSAI Notice Bot  ｜  {_now_kst_str()}"},
                ],
            },
        ]
    )

    fallback = f"📢 {feed_title} 새 글 {count}건"
    return {
        "text": fallback,
        "unfurl_links": False,
        "unfurl_media": False,
        "attachments": [{"color": "#003876", "blocks": blocks}],
    }


def send_to_slack(webhook_url: str, payload: dict[str, Any], *, dry_run: bool) -> None:
    if dry_run:
        for att in payload.get("attachments", []):
            for block in att.get("blocks", []):
                btype = block.get("type")
                if btype == "header":
                    print(f"\n  {block['text']['text']}")
                elif btype == "context":
                    for el in block["elements"]:
                        print(f"  {el.get('text', '')}")
                elif btype == "section":
                    print(f"  {block['text']['text']}")
                elif btype == "divider":
                    print(f"  {'─' * 40}")
                elif btype == "actions":
                    for el in block["elements"]:
                        print(f"         [ {el['text']['text']} ]")
        print()
        return

    if requests is None:  # pragma: no cover
        raise RuntimeError(
            "requests 라이브러리가 설치되지 않았습니다. pip install -r requirements.txt"
        )

    resp = requests.post(webhook_url, json=payload, timeout=20)
    resp.raise_for_status()


def _normalize_notion_page_id(page_id_or_url: str) -> str:
    """
    Notion 페이지 ID를 정규화합니다.
    URL 형식: https://www.notion.so/Notice-27e2cbf5657380319715fa24fb5d4d15
    -> 페이지 ID: 27e2cbf5657380319715fa24fb5d4d15 (하이픈 제거, 32자 hex)
    """
    s = (page_id_or_url or "").strip()
    if not s:
        return ""

    # URL에서 페이지 ID 추출
    if "notion.so" in s:
        # URL에서 쿼리 파라미터와 프래그먼트 제거
        if "?" in s:
            s = s.split("?")[0]
        if "#" in s:
            s = s.split("#")[0]

        # 마지막 하이픈 이후 부분이 페이지 ID
        # 예: https://www.notion.so/Notice-27e2cbf5657380319715fa24fb5d4d15
        # -> ['https://www.notion.so/Notice', '27e2cbf5657380319715fa24fb5d4d15']
        parts = s.split("-")
        if len(parts) >= 2:
            # 마지막 부분이 페이지 ID
            page_id = parts[-1]
            # 32자 hex 문자열인지 확인
            if len(page_id) == 32 and all(c in "0123456789abcdef" for c in page_id.lower()):
                return page_id

        # 하이픈이 없는 경우: URL 경로의 마지막 부분에서 32자 hex 찾기
        # 예: https://www.notion.so/27e2cbf5657380319715fa24fb5d4d15
        match = re.search(r"([0-9a-f]{32})", s.lower())
        if match:
            return match.group(1)

    # 이미 페이지 ID인 경우 (하이픈 제거)
    # 예: 27e2cbf5-6573-8031-9715-fa24fb5d4d15 -> 27e2cbf5657380319715fa24fb5d4d15
    cleaned = s.replace("-", "")
    if len(cleaned) == 32 and all(c in "0123456789abcdef" for c in cleaned.lower()):
        return cleaned

    # 그 외의 경우 원본 반환 (에러는 호출하는 쪽에서 처리)
    return s


def send_to_notion(
    *,
    token: str,
    page_id: str,
    feed_title: str,
    items: list[Entry],
    dry_run: bool = False,
) -> None:
    """
    Notion 페이지에 새 글 목록을 블록으로 추가합니다.
    """
    if Client is None:
        raise RuntimeError(
            "notion-client 라이브러리가 설치되지 않았습니다. pip install notion-client"
        )

    if dry_run:
        print(f"[DRY-RUN] Notion 전송: {len(items)}개 글")
        return

    normalized_page_id = _normalize_notion_page_id(page_id)
    if not normalized_page_id:
        raise ValueError(f"유효하지 않은 Notion 페이지 ID: {page_id}")

    # 정규화된 페이지 ID가 32자 hex가 아니면 에러
    if len(normalized_page_id) != 32 or not all(
        c in "0123456789abcdef" for c in normalized_page_id.lower()
    ):
        raise ValueError(
            f"정규화된 페이지 ID가 유효하지 않습니다: {normalized_page_id} (원본: {page_id})"
        )

    LOG.debug("페이지 ID 정규화: %s -> %s", page_id[:50], normalized_page_id)
    client = Client(auth=token)

    # 각 글을 Notion 블록으로 추가
    blocks: list[dict[str, Any]] = []

    # 헤더 블록
    header_text = f"📰 {feed_title} 새 글 {len(items)}개"

    blocks.append(
        {
            "object": "block",
            "type": "heading_2",
            "heading_2": {"rich_text": [{"type": "text", "text": {"content": header_text}}]},
        }
    )

    # 구분선 추가
    blocks.append({"object": "block", "type": "divider", "divider": {}})

    # 각 글을 callout 블록으로 추가 (더 예쁘게 표시)
    for idx, entry in enumerate(items):
        title = entry_title(entry) or "(제목 없음)"
        link = entry_link(entry)
        pub = entry_pub(entry)

        # 날짜 포맷팅 (간단하게)
        date_str = ""
        if pub:
            try:
                # feedparser의 published_parsed를 사용하거나 문자열 파싱
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    y, mo, d, h, mi, s = entry.published_parsed[:6]
                    dt = datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc)
                    date_str = dt.strftime("%Y-%m-%d %H:%M")
                else:
                    # 문자열에서 간단히 추출
                    date_str = pub[:16] if len(pub) >= 16 else pub
            except Exception:
                date_str = pub[:16] if len(pub) >= 16 else pub

        # Callout 블록의 rich_text 구성
        rich_text_parts: list[dict[str, Any]] = []

        # 제목 (bold)
        rich_text_parts.append(
            {
                "type": "text",
                "text": {"content": title},
                "annotations": {"bold": True},
            }
        )

        # 날짜가 있으면 추가
        if date_str:
            rich_text_parts.append(
                {
                    "type": "text",
                    "text": {"content": f"\n📅 {date_str}"},
                    "annotations": {"bold": False},
                }
            )

        # 링크가 있으면 별도 줄로 추가
        if link:
            rich_text_parts.append(
                {
                    "type": "text",
                    "text": {"content": "\n🔗 "},
                    "annotations": {"bold": False},
                }
            )
            rich_text_parts.append(
                {
                    "type": "text",
                    "text": {"content": "원문 보기", "link": {"url": link}},
                    "annotations": {"bold": False},
                }
            )

        # Callout 블록 생성 (색상: blue)
        blocks.append(
            {
                "object": "block",
                "type": "callout",
                "callout": {
                    "rich_text": rich_text_parts,
                    "icon": {"emoji": "📰"},
                    "color": "blue",
                },
            }
        )

        # 마지막 글이 아니면 구분선 추가 (선택적)
        if idx < len(items) - 1:
            blocks.append({"object": "block", "type": "divider", "divider": {}})

    # Notion API로 블록 추가 (한 번에 최대 100개까지 가능)
    try:
        LOG.debug("Notion API 호출: page_id=%s, blocks=%d개", normalized_page_id, len(blocks))
        client.blocks.children.append(block_id=normalized_page_id, children=blocks)
        LOG.info("Notion 전송 완료: %d개 글을 페이지에 추가했습니다.", len(items))
    except Exception as e:
        error_msg = str(e)
        LOG.error("Notion API 실패: %s", error_msg, exc_info=True)
        # 더 자세한 에러 정보 출력
        if hasattr(e, "response") and hasattr(e.response, "text"):
            LOG.error("Notion API 응답: %s", e.response.text[:500])
        raise RuntimeError(f"Notion API 실패: {error_msg}") from e


def _dispatch_items(
    cfg: Config,
    feed_title: str,
    items: list[Entry],
    *,
    feed_url: str,
    site_url: str,
) -> int:
    """
    항목들을 max_items_per_message 단위로 나눠 Slack으로 보내고,
    이어서 Notion으로도(설정된 경우) 베스트에포트로 전송합니다.

    Notion 전송 실패는 Slack 경로를 실패시키지 않습니다(경고만 남김).
    Slack 전송이 실패하면 예외가 그대로 전파됩니다.

    반환값: Slack으로 전송한 항목 수(요약 로그용).
    """
    chunks = list(_chunked(items, cfg.max_items_per_message))
    for idx, chunk in enumerate(chunks, start=1):
        payload = format_slack_payload(
            feed_title,
            chunk,
            feed_url=feed_url,
            site_url=site_url,
            index=idx,
            total=len(chunks),
        )
        send_to_slack(cfg.slack_webhook_url or "", payload, dry_run=cfg.dry_run)

        # Notion 전송
        if cfg.notion_token and cfg.notion_page_id:
            LOG.info(
                "Notion 전송 시도: token=%s..., page_id=%s",
                cfg.notion_token[:10] if cfg.notion_token else "None",
                cfg.notion_page_id[:20] if cfg.notion_page_id else "None",
            )
            try:
                send_to_notion(
                    token=str(cfg.notion_token),
                    page_id=str(cfg.notion_page_id),
                    feed_title=feed_title,
                    items=chunk,
                    dry_run=cfg.dry_run,
                )
            except Exception as e:
                LOG.warning("Notion 전송 실패 (Slack은 정상 전송됨): %s", e, exc_info=True)
        else:
            if not cfg.notion_token:
                LOG.warning("Notion 전송 스킵: NOTION_TOKEN이 설정되지 않았습니다.")
            if not cfg.notion_page_id:
                LOG.warning("Notion 전송 스킵: NOTION_PAGE_ID가 설정되지 않았습니다.")

    # Slack 전송이 성공적으로 끝난 항목 수를 반환합니다.
    return len(items)


@dataclass
class _RunSummary:
    """한 번의 run_once 실행 동안의 집계 값을 담는 가변 카운터입니다."""

    feeds_checked: int = 0
    items_sent: int = 0
    fetch_failures: int = 0
    slack_failures: int = 0


def _process_feed(
    cfg: Config,
    feed_url: str,
    feeds: dict[str, Any],
    sent_ids: set,
    sent_articles: dict[str, str],
    summary: _RunSummary,
) -> int:
    """
    한 피드를 처리하고 이 피드의 종료 코드 기여분을 반환합니다.
    0 성공, 2 가져오기 실패, 3 Slack 전송 실패.

    전송한 항목 수는 summary.items_sent 에 누적됩니다.
    """
    LOG.info("피드 확인: %s", feed_url)
    try:
        parsed = fetch_feed(feed_url, verify_ssl=cfg.verify_ssl)
    except Exception as e:
        LOG.exception("피드 가져오기 실패: %s (%s)", feed_url, e)
        return 2

    feed_title = (parsed.feed.get("title") or feed_url).strip()
    site_url = (parsed.feed.get("link") or "").strip()
    entries = list(parsed.entries or [])

    feed_state = feeds.get(feed_url) or {}
    last_seen_id = feed_state.get("last_id")

    # 첫 실행: 기준점만 찍거나, 옵션으로 최신 N개를 보냅니다.
    if last_seen_id is None:
        initial_items: list[Entry] = []
        if entries and cfg.initial_notify_count > 0:
            initial_items = [
                e
                for e in reversed(entries[: cfg.initial_notify_count])
                if not _entry_already_sent(e, sent_ids, sent_articles)
            ]
            try:
                summary.items_sent += _dispatch_items(
                    cfg, feed_title, initial_items, feed_url=feed_url, site_url=site_url
                )
            except Exception as e:
                LOG.exception("Slack 전송 실패: %s (%s)", feed_title, e)
                return 3

        _mark_entries_sent(initial_items, sent_ids, sent_articles)
        newest_id = entry_uid(entries[0]) if entries else None
        feeds[feed_url] = {"last_id": newest_id, "updated_at": _now_iso()}
        LOG.info("첫 실행 기준점 저장: %s", newest_id)
        return 0

    new_entries, newest_id, found = compute_new_entries(entries, last_seen_id)
    # 다른 피드 또는 이전 실행에서 이미 보낸 글 제거 (uid+link 이중 검사)
    new_entries = [e for e in new_entries if not _entry_already_sent(e, sent_ids, sent_articles)]

    if not found:
        LOG.warning(
            "상태 불일치: 마지막 ID가 피드에 없습니다. last_id=%s, feed=%s",
            last_seen_id,
            feed_url,
        )
        items: list[Entry] = []
        if cfg.on_state_miss == "send" and entries:
            # 피드에 보이는 항목을 모두 새 글로 간주(다른 피드에서 보낸 건 제외)
            items = [
                e for e in reversed(entries) if not _entry_already_sent(e, sent_ids, sent_articles)
            ]
            try:
                summary.items_sent += _dispatch_items(
                    cfg, feed_title, items, feed_url=feed_url, site_url=site_url
                )
            except Exception as e:
                LOG.exception("Slack 전송 실패: %s (%s)", feed_title, e)
                return 3

        _mark_entries_sent(items, sent_ids, sent_articles)
        # 어쨌든 최신 기준점으로 재설정(다음 실행부터 정상 동작)
        feeds[feed_url] = {"last_id": newest_id, "updated_at": _now_iso()}
        return 0

    if not new_entries:
        LOG.info("새 글 없음: %s", feed_title)
        # sent_uids 필터링으로 비워진 경우에도 상태를 갱신해야
        # 다음 실행에서 같은 글을 다시 보내지 않습니다.
        if newest_id:
            feeds[feed_url] = {"last_id": newest_id, "updated_at": _now_iso()}
        return 0

    # 새 글이 있으면, Slack 전송 성공 후 상태를 업데이트합니다(누락 방지).
    try:
        summary.items_sent += _dispatch_items(
            cfg, feed_title, new_entries, feed_url=feed_url, site_url=site_url
        )
    except Exception as e:
        LOG.exception("Slack 전송 실패: %s (%s)", feed_title, e)
        return 3

    _mark_entries_sent(new_entries, sent_ids, sent_articles)
    feeds[feed_url] = {"last_id": newest_id, "updated_at": _now_iso()}
    LOG.info("상태 업데이트: %s -> %s", last_seen_id, newest_id)
    return 0


def run_once(cfg: Config) -> int:
    state = load_state(cfg.state_file)
    feeds: dict[str, Any] = state.setdefault("feeds", {})

    # 영구 전송 이력: 이전 실행에서 보낸 글도 중복 방지 (uid+link 모두 기록)
    sent_articles: dict[str, str] = _prune_sent_articles(state.get("sent_articles", {}))

    overall_exit = 0
    sent_ids: set = set()  # 피드 간 중복 알림 방지 (uid + link)
    summary = _RunSummary()

    for feed_url in cfg.feed_urls:
        summary.feeds_checked += 1
        exit_code = _process_feed(cfg, feed_url, feeds, sent_ids, sent_articles, summary)
        if exit_code == 2:
            summary.fetch_failures += 1
        elif exit_code == 3:
            summary.slack_failures += 1
        if exit_code:
            overall_exit = exit_code

    state["sent_articles"] = sent_articles
    save_state(cfg.state_file, state)

    LOG.info(
        "run summary: feeds=%d items_sent=%d fetch_failures=%d slack_failures=%d exit=%d",
        summary.feeds_checked,
        summary.items_sent,
        summary.fetch_failures,
        summary.slack_failures,
        overall_exit,
    )
    return overall_exit


def main() -> int:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    try:
        cfg = load_config()
    except Exception as e:
        print(f"[config error] {e}", file=sys.stderr)
        return 1
    return run_once(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
