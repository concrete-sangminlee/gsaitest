"""load_config 및 _parse_bool/_parse_int 순수 로직 테스트."""

from __future__ import annotations

import pytest

import gsai_notifier as gn

# load_config가 참조하는 모든 환경변수. 각 테스트에서 깨끗한 상태로 시작하기 위해 정리합니다.
_CONFIG_ENV_VARS = (
    "SLACK_WEBHOOK_URL",
    "FEED_URLS",
    "STATE_FILE",
    "INITIAL_NOTIFY_COUNT",
    "MAX_ITEMS_PER_MESSAGE",
    "ON_STATE_MISS",
    "VERIFY_SSL",
    "DRY_RUN",
    "NOTION_TOKEN",
    "NOTION_PAGE_ID",
    "ENV_FILE",
)


@pytest.fixture
def clean_env(monkeypatch):
    """load_config 관련 환경변수를 모두 제거한 상태를 제공합니다."""
    for name in _CONFIG_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    # dotenv 로드가 실제 .env를 읽지 않도록 무력화합니다.
    monkeypatch.setattr(gn, "load_dotenv", None)
    return monkeypatch


def test_default_feed_url(clean_env):
    clean_env.setenv("DRY_RUN", "true")
    cfg = gn.load_config()
    assert cfg.feed_urls == ["https://gsai.snu.ac.kr/feed/"]


def test_feed_urls_split_and_trim(clean_env):
    clean_env.setenv("DRY_RUN", "true")
    clean_env.setenv("FEED_URLS", " https://a/feed ,https://b/feed , , https://c/feed ")
    cfg = gn.load_config()
    assert cfg.feed_urls == ["https://a/feed", "https://b/feed", "https://c/feed"]


def test_dry_run_allows_missing_webhook(clean_env):
    clean_env.setenv("DRY_RUN", "true")
    cfg = gn.load_config()
    assert cfg.dry_run is True
    assert cfg.slack_webhook_url is None


def test_missing_webhook_raises_when_not_dry_run(clean_env):
    clean_env.setenv("DRY_RUN", "false")
    with pytest.raises(ValueError):
        gn.load_config()


def test_webhook_present_when_not_dry_run(clean_env):
    clean_env.setenv("DRY_RUN", "false")
    clean_env.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/x")
    cfg = gn.load_config()
    assert cfg.slack_webhook_url == "https://hooks.slack.com/x"
    assert cfg.dry_run is False


def test_invalid_on_state_miss_raises(clean_env):
    clean_env.setenv("DRY_RUN", "true")
    clean_env.setenv("ON_STATE_MISS", "explode")
    with pytest.raises(ValueError):
        gn.load_config()


@pytest.mark.parametrize("value", ["skip", "send", "SEND", "Skip"])
def test_valid_on_state_miss(clean_env, value):
    clean_env.setenv("DRY_RUN", "true")
    clean_env.setenv("ON_STATE_MISS", value)
    cfg = gn.load_config()
    assert cfg.on_state_miss == value.lower()


def test_initial_notify_count_clamped_to_zero(clean_env):
    clean_env.setenv("DRY_RUN", "true")
    clean_env.setenv("INITIAL_NOTIFY_COUNT", "-5")
    cfg = gn.load_config()
    assert cfg.initial_notify_count == 0


def test_max_items_per_message_clamped_to_one(clean_env):
    clean_env.setenv("DRY_RUN", "true")
    clean_env.setenv("MAX_ITEMS_PER_MESSAGE", "0")
    cfg = gn.load_config()
    assert cfg.max_items_per_message == 1


def test_numeric_config_values_passthrough(clean_env):
    clean_env.setenv("DRY_RUN", "true")
    clean_env.setenv("INITIAL_NOTIFY_COUNT", "3")
    clean_env.setenv("MAX_ITEMS_PER_MESSAGE", "7")
    cfg = gn.load_config()
    assert cfg.initial_notify_count == 3
    assert cfg.max_items_per_message == 7


def test_verify_ssl_default_true(clean_env):
    clean_env.setenv("DRY_RUN", "true")
    cfg = gn.load_config()
    assert cfg.verify_ssl is True


def test_verify_ssl_parsed_false(clean_env):
    clean_env.setenv("DRY_RUN", "true")
    clean_env.setenv("VERIFY_SSL", "no")
    cfg = gn.load_config()
    assert cfg.verify_ssl is False


def test_notion_config_passthrough(clean_env):
    clean_env.setenv("DRY_RUN", "true")
    clean_env.setenv("NOTION_TOKEN", "secret_tok")
    clean_env.setenv("NOTION_PAGE_ID", "page123")
    cfg = gn.load_config()
    assert cfg.notion_token == "secret_tok"
    assert cfg.notion_page_id == "page123"


# --- _parse_bool truth table ---


@pytest.mark.parametrize("value", ["1", "true", "t", "yes", "y", "on", "TRUE", " On "])
def test_parse_bool_truthy(value):
    assert gn._parse_bool(value, default=False) is True


@pytest.mark.parametrize("value", ["0", "false", "f", "no", "n", "off", "FALSE", " Off "])
def test_parse_bool_falsy(value):
    assert gn._parse_bool(value, default=True) is False


def test_parse_bool_none_returns_default():
    assert gn._parse_bool(None, default=True) is True
    assert gn._parse_bool(None, default=False) is False


@pytest.mark.parametrize("default", [True, False])
def test_parse_bool_bad_input_returns_default(default):
    assert gn._parse_bool("maybe", default=default) is default


# --- _parse_int ---


def test_parse_int_valid():
    assert gn._parse_int("42", default=0) == 42
    assert gn._parse_int("  -7 ", default=0) == -7


def test_parse_int_none_returns_default():
    assert gn._parse_int(None, default=9) == 9


@pytest.mark.parametrize("value", ["abc", "1.5", "", "  "])
def test_parse_int_bad_input_returns_default(value):
    assert gn._parse_int(value, default=13) == 13
