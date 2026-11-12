"""gsai_notify.yml의 안전장치가 빠지지 않도록 텍스트 기반으로 검사합니다.

워크플로우는 이 저장소에서 실행해 볼 수 없으므로, 중복/누락을 막는 설정이
실수로 지워지면 바로 테스트가 실패하도록 합니다. (PyYAML은 dev 의존성에 없어
텍스트로 검사합니다.)
"""

from __future__ import annotations

import re
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "gsai_notify.yml"


def _text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _step_block(name: str) -> str:
    lines = _text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip() == f"- name: {name}")
    indent = len(lines[start]) - len(lines[start].lstrip())
    block = [lines[start]]
    for ln in lines[start + 1 :]:
        if ln.strip() and len(ln) - len(ln.lstrip()) <= indent:
            break
        block.append(ln)
    return "\n".join(block)


def test_commit_step_runs_even_if_notifier_failed():
    """알림 단계가 실패해도 커밋해야 그 실행에서 보낸 글이 다시 가지 않습니다."""
    block = _step_block("Commit state.json (if changed)")
    m = re.search(r"^\s+if:\s*(.+)$", block, re.MULTILINE)
    assert m, "commit 단계에 if: 조건이 없습니다"
    assert "always()" in m.group(1) or "!cancelled()" in m.group(1)


def test_commit_step_detects_untracked_state_file():
    block = _step_block("Commit state.json (if changed)")
    assert "git add state.json" in block
    assert "git diff --cached --quiet -- state.json" in block
    assert block.index("git add state.json") < block.index("git diff --cached")


def test_push_is_retried_with_rebase():
    block = _step_block("Commit state.json (if changed)")
    assert "git push origin HEAD:main" in block
    assert "git pull --rebase origin main" in block


def test_runs_are_serialized():
    text = _text()
    assert re.search(r"^concurrency:\s*$", text, re.MULTILINE)
    assert re.search(r"^\s+cancel-in-progress:\s*false\s*$", text, re.MULTILINE)


def test_checkout_reads_latest_main():
    """대기하던 실행이나 Re-run이 오래된 state.json으로 돌지 않도록 합니다."""
    block = _step_block("Checkout")
    assert re.search(r"^\s+ref:\s*main\s*$", block, re.MULTILINE)


def test_job_has_timeout():
    assert re.search(r"^\s+timeout-minutes:\s*\d+\s*$", _text(), re.MULTILINE)
