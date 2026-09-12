"""Pytest configuration for the gsai_notifier test suite.

목표: 이 저장소를 어떤 위치에서 pytest로 실행하더라도 `import gsai_notifier`가
성공하도록 보장합니다. FEAT-002 이후 feedparser/requests/dotenv/notion_client가
없어도 모듈을 import할 수 있으므로, 아래 stub 주입은 방어적 fallback으로만 둡니다.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

# 저장소 루트를 sys.path에 올려 `import gsai_notifier`가 해석되도록 합니다.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _install_stub_modules() -> None:
    """third-party 런타임 의존성의 최소 stub을 sys.modules에 주입합니다.

    순수 로직 테스트는 IO 함수를 호출하지 않으므로 import만 만족시키면 됩니다.
    """
    if "feedparser" not in sys.modules:
        feedparser = types.ModuleType("feedparser")
        feedparser.FeedParserDict = dict  # type: ignore[attr-defined]

        def _parse(*_args, **_kwargs):  # pragma: no cover - placeholder
            raise RuntimeError("feedparser stub: parse는 테스트에서 호출되지 않습니다.")

        feedparser.parse = _parse  # type: ignore[attr-defined]
        sys.modules["feedparser"] = feedparser

    if "requests" not in sys.modules:
        requests = types.ModuleType("requests")

        def _get(*_args, **_kwargs):  # pragma: no cover - placeholder
            raise RuntimeError("requests stub: get은 테스트에서 호출되지 않습니다.")

        def _post(*_args, **_kwargs):  # pragma: no cover - placeholder
            raise RuntimeError("requests stub: post는 테스트에서 호출되지 않습니다.")

        requests.get = _get  # type: ignore[attr-defined]
        requests.post = _post  # type: ignore[attr-defined]
        sys.modules["requests"] = requests

    if "dotenv" not in sys.modules:
        dotenv = types.ModuleType("dotenv")

        def _load_dotenv(*_args, **_kwargs):  # pragma: no cover - placeholder
            return False

        dotenv.load_dotenv = _load_dotenv  # type: ignore[attr-defined]
        sys.modules["dotenv"] = dotenv

    if "notion_client" not in sys.modules:
        notion_client = types.ModuleType("notion_client")

        class _Client:  # pragma: no cover - placeholder
            def __init__(self, *_args, **_kwargs):
                raise RuntimeError("notion_client stub: 테스트에서 호출되지 않습니다.")

        notion_client.Client = _Client  # type: ignore[attr-defined]
        sys.modules["notion_client"] = notion_client


try:  # 직접 import가 가능하면 그대로 사용합니다.
    import gsai_notifier  # noqa: F401
except ImportError:  # pragma: no cover - 방어적 fallback
    _install_stub_modules()
    import gsai_notifier  # noqa: F401
