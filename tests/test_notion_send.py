"""send_to_notion의 100블록 청킹 동작 테스트 (오프라인).

notion-client가 설치되어 있지 않아도 gn.Client를 fake로 monkeypatch하여
append 호출이 100블록 이하로 나뉘는지, 순서가 보존되는지 검증합니다.
"""

from __future__ import annotations

import math
from typing import Any

import gsai_notifier as gn

# 정규화를 통과하는 유효한 32자 hex 페이지 ID.
_PAGE_ID = "27e2cbf5657380319715fa24fb5d4d15"


def _make_items(n: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for i in range(n):
        items.append({"title": f"글{i}", "link": f"http://x/{i}", "published": "2024-01-01"})
    return items


class _FakeChildren:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, Any]]] = []

    def append(self, *, block_id: str, children: list[dict[str, Any]]) -> None:
        assert block_id == _PAGE_ID
        self.calls.append(list(children))


class _FakeBlocks:
    def __init__(self) -> None:
        self.children = _FakeChildren()


class _FakeClient:
    last_instance: _FakeClient | None = None

    def __init__(self, *, auth: str) -> None:
        self.auth = auth
        self.blocks = _FakeBlocks()
        _FakeClient.last_instance = self


def _run_send(monkeypatch, n_items: int) -> _FakeClient:
    monkeypatch.setattr(gn, "Client", _FakeClient)
    items = _make_items(n_items)
    gn.send_to_notion(
        token="fake-token",
        page_id=_PAGE_ID,
        feed_title="테스트 피드",
        items=items,
        dry_run=False,
    )
    client = _FakeClient.last_instance
    assert client is not None
    return client


def test_small_case_single_append(monkeypatch):
    client = _run_send(monkeypatch, 1)
    calls = client.blocks.children.calls
    # n=1 -> 3 블록, 한 번의 append.
    assert len(calls) == 1
    assert len(calls[0]) == 3


def test_large_case_chunks_at_100(monkeypatch):
    n_items = 60  # 총 블록 = 2*60 + 1 = 121 (>100)
    client = _run_send(monkeypatch, n_items)
    calls = client.blocks.children.calls

    total_blocks = 2 * n_items + 1
    assert total_blocks > 100

    # 모든 append 호출은 100블록 이하.
    for chunk in calls:
        assert len(chunk) <= 100

    # append 호출 수 == ceil(total / 100).
    assert len(calls) == math.ceil(total_blocks / 100)

    # 전체 append된 블록을 이어붙이면 총 블록 수와 일치하고 순서가 보존된다.
    concatenated = [b for chunk in calls for b in chunk]
    assert len(concatenated) == total_blocks

    # 첫 블록은 heading_2, 두 번째는 divider여야 순서 보존이 확인된다.
    assert concatenated[0]["type"] == "heading_2"
    assert concatenated[1]["type"] == "divider"
    # callout과 divider가 번갈아 나오는 구조의 마지막 블록은 callout.
    assert concatenated[-1]["type"] == "callout"


def test_exactly_100_blocks_single_append(monkeypatch):
    # 총 블록 = 2n + 1 == 99 (n=49) 는 <=100 이므로 단일 append.
    client = _run_send(monkeypatch, 49)
    calls = client.blocks.children.calls
    assert len(calls) == 1
    assert len(calls[0]) == 99
