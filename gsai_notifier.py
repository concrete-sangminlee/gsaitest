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
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import feedparser
import requests

try:
    # 로컬에서 .env 파일을 쓰고 싶은 경우를 지원합니다.
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]


LOG = logging.getLogger("gsai_notifier")


def _parse_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    v = value.strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _parse_int(value: Optional[str], default: int) -> int:
    if value is None:
        return default
    try:
        return int(value.strip())
    except Exception:
        return default


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _chunked(items: List[Any], size: int) -> Iterable[List[Any]]:
    if size <= 0:
        yield items
        return
    for i in range(0, len(items), size):
        yield items[i : i + size]


@dataclass(frozen=True)
class Config:
    slack_webhook_url: Optional[str]
    feed_urls: List[str]
    state_file: Path
    initial_notify_count: int
    max_items_per_message: int
    on_state_miss: str  # "skip" | "send"
    verify_ssl: bool
    dry_run: bool


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

    return Config(
        slack_webhook_url=slack_webhook_url,
        feed_urls=feed_urls,
        state_file=state_file,
        initial_notify_count=max(0, initial_notify_count),
        max_items_per_message=max(1, max_items_per_message),
        on_state_miss=on_state_miss,
        verify_ssl=verify_ssl,
        dry_run=dry_run,
    )


def load_state(path: Path) -> Dict[str, Any]:
    """
    상태 파일 포맷(v1):
    {
      "version": 1,
      "feeds": {
        "<feed_url>": {"last_id": "...", "updated_at": "..."}
      }
    }
    """
    default_state: Dict[str, Any] = {"version": 1, "feeds": {}}
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


def save_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def fetch_feed(url: str, *, verify_ssl: bool) -> feedparser.FeedParserDict:
    headers = {
        "User-Agent": "gsai-slack-bot/1.0 (+https://gsai.snu.ac.kr)",
        "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml;q=0.9, */*;q=0.1",
    }
    resp = requests.get(url, headers=headers, timeout=20, verify=verify_ssl)
    resp.raise_for_status()
    parsed = feedparser.parse(resp.content)
    return parsed


def entry_uid(entry: feedparser.FeedParserDict) -> str:
    # WordPress RSS는 보통 guid / link가 안정적입니다.
    for key in ("id", "guid", "link"):
        v = entry.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    title = (entry.get("title") or "").strip()
    published = (entry.get("published") or entry.get("updated") or "").strip()
    return f"{title}|{published}"


def entry_title(entry: feedparser.FeedParserDict) -> str:
    return (entry.get("title") or "").strip()


def entry_link(entry: feedparser.FeedParserDict) -> str:
    return (entry.get("link") or "").strip()


def entry_pub(entry: feedparser.FeedParserDict) -> str:
    return (entry.get("published") or entry.get("updated") or "").strip()


def compute_new_entries(
    entries: List[feedparser.FeedParserDict],
    last_seen_id: Optional[str],
) -> Tuple[List[feedparser.FeedParserDict], Optional[str], bool]:
    """
    returns: (new_entries_oldest_first, newest_id, last_seen_found_in_feed)
    """
    if not entries:
        return [], None, True

    newest_id = entry_uid(entries[0])
    if last_seen_id is None:
        return [], newest_id, True

    new_entries: List[feedparser.FeedParserDict] = []
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


def format_slack_text(feed_title: str, items: List[feedparser.FeedParserDict], *, index: int, total: int) -> str:
    # Slack 링크 포맷: <url|text>
    header = f":newspaper: *{feed_title}* 새 글 {len(items)}개"
    if total > 1:
        header += f" ({index}/{total})"

    lines: List[str] = [header]
    for e in items:
        title = entry_title(e) or "(제목 없음)"
        link = entry_link(e)
        pub = entry_pub(e)
        if link:
            line = f"- <{link}|{title}>"
        else:
            line = f"- {title}"
        if pub:
            line += f"  _({pub})_"
        lines.append(line)
    return "\n".join(lines)


def send_to_slack(webhook_url: str, text: str, *, dry_run: bool) -> None:
    if dry_run:
        print(text)
        return

    payload = {
        "text": text,
        "mrkdwn": True,
        "unfurl_links": False,
        "unfurl_media": False,
    }
    resp = requests.post(webhook_url, json=payload, timeout=20)
    resp.raise_for_status()


def run_once(cfg: Config) -> int:
    state = load_state(cfg.state_file)
    feeds: Dict[str, Any] = state.setdefault("feeds", {})

    overall_exit = 0

    for feed_url in cfg.feed_urls:
        LOG.info("피드 확인: %s", feed_url)
        try:
            parsed = fetch_feed(feed_url, verify_ssl=cfg.verify_ssl)
        except Exception as e:
            overall_exit = 2
            LOG.exception("피드 가져오기 실패: %s (%s)", feed_url, e)
            continue

        feed_title = (parsed.feed.get("title") or feed_url).strip()
        entries = list(parsed.entries or [])

        feed_state = feeds.get(feed_url) or {}
        last_seen_id = feed_state.get("last_id")

        # 첫 실행: 기준점만 찍거나, 옵션으로 최신 N개를 보냅니다.
        if last_seen_id is None:
            if entries and cfg.initial_notify_count > 0:
                initial_items = list(reversed(entries[: cfg.initial_notify_count]))
                chunks = list(_chunked(initial_items, cfg.max_items_per_message))
                for idx, chunk in enumerate(chunks, start=1):
                    text = format_slack_text(feed_title, chunk, index=idx, total=len(chunks))
                    send_to_slack(cfg.slack_webhook_url or "", text, dry_run=cfg.dry_run)

            newest_id = entry_uid(entries[0]) if entries else None
            feeds[feed_url] = {"last_id": newest_id, "updated_at": _now_iso()}
            LOG.info("첫 실행 기준점 저장: %s", newest_id)
            continue

        new_entries, newest_id, found = compute_new_entries(entries, last_seen_id)

        if not found:
            LOG.warning(
                "상태 불일치: 마지막 ID가 피드에 없습니다. last_id=%s, feed=%s",
                last_seen_id,
                feed_url,
            )
            if cfg.on_state_miss == "send" and entries:
                # 피드에 보이는 항목을 모두 새 글로 간주(중복 가능)
                items = list(reversed(entries))
                chunks = list(_chunked(items, cfg.max_items_per_message))
                for idx, chunk in enumerate(chunks, start=1):
                    text = format_slack_text(feed_title, chunk, index=idx, total=len(chunks))
                    send_to_slack(cfg.slack_webhook_url or "", text, dry_run=cfg.dry_run)

            # 어쨌든 최신 기준점으로 재설정(다음 실행부터 정상 동작)
            feeds[feed_url] = {"last_id": newest_id, "updated_at": _now_iso()}
            continue

        if not new_entries:
            LOG.info("새 글 없음: %s", feed_title)
            continue

        # 새 글이 있으면, Slack 전송 성공 후 상태를 업데이트합니다(누락 방지).
        chunks = list(_chunked(new_entries, cfg.max_items_per_message))
        try:
            for idx, chunk in enumerate(chunks, start=1):
                text = format_slack_text(feed_title, chunk, index=idx, total=len(chunks))
                send_to_slack(cfg.slack_webhook_url or "", text, dry_run=cfg.dry_run)
        except Exception as e:
            overall_exit = 3
            LOG.exception("Slack 전송 실패: %s (%s)", feed_title, e)
            continue

        feeds[feed_url] = {"last_id": newest_id, "updated_at": _now_iso()}
        LOG.info("상태 업데이트: %s -> %s", last_seen_id, newest_id)

    save_state(cfg.state_file, state)
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


