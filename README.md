# GSAI 새 글 Slack 알림 봇 (RSS 기반)

## 목표

이 프로젝트는 `gsai.snu.ac.kr`에 **새 글이 올라오면 Slack 채널로 자동 알림**을 보내는 봇입니다.  
WordPress 사이트가 제공하는 **RSS 피드**를 주기적으로 확인하고, “마지막으로 본 글”을 **상태 파일**에 저장해 **중복 없이** 새 글만 전송합니다.

## 동작 방식(아키텍처)

아래처럼 단순한 “폴링(polling)” 구조입니다.

```text
┌──────────────┐     (1) RSS 조회      ┌──────────────┐
│ Scheduler    │ ────────────────────► │ gsai.snu.ac.kr│
│ (cron/GHA)   │                      │ /feed/        │
└──────┬───────┘                      └──────┬───────┘
       │ (2) 파싱/비교                       │
       │     last_id 기준                    │
       ▼                                     │
┌──────────────┐     (3) 새 글만 전송  ┌──────▼───────┐
│ state.json   │ ◄───────────────────► │ Slack 채널    │
│ (책갈피)     │     (4) last_id 갱신  │ (Webhook)     │
└──────────────┘                      └──────────────┘
```

## 준비물

- Python **3.10+** (권장: 3.11)
- Slack 워크스페이스 권한(앱/웹훅 생성 가능)

## 1) Slack Incoming Webhook 만들기

목표는 “Slack 채널에 HTTP POST로 메시지를 보낼 수 있는 URL”을 얻는 것입니다.

1. Slack에서 **Incoming Webhooks**를 활성화합니다.
2. 알림을 받을 채널을 선택해 **Webhook URL**을 생성합니다.
3. 생성된 URL을 복사해 둡니다. (Webhook URL은 시크릿이므로 레포/로그/채팅에 붙여 넣지 마세요.)

> 대안(코드 없이): 워크스페이스에 따라 Slack의 RSS 앱을 바로 쓸 수도 있습니다.  
> 다만 필터링/상태 관리/포맷 커스터마이징이 필요하면 이 프로젝트 방식이 더 유연합니다.

## 2) 로컬에서 실행하기(가장 쉬움)

### 2-1. 설치

```bash
cd /path/to/gsai_slack_bot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2-2. 환경변수 설정

이 레포에는 예시 파일로 `env.example`가 있습니다.  
로컬에서는 아래 중 하나를 선택하세요.

- **옵션 A(권장)**: `.env` 파일을 직접 만들고 `env.example` 내용을 복사해 값만 채우기
- **옵션 B**: 쉘에서 `export`로 환경변수 설정

최소로 필요한 값:

- `SLACK_WEBHOOK_URL`
- `FEED_URLS` (기본값이 이미 `https://gsai.snu.ac.kr/feed/`)

### 2-3. 테스트(DRY_RUN)

Slack으로 보내기 전에, 콘솔 출력으로 먼저 확인할 수 있습니다.

```bash
DRY_RUN=true INITIAL_NOTIFY_COUNT=2 python gsai_notifier.py
```

### 2-4. 실전 실행

```bash
python gsai_notifier.py
```

## 3) 서버에서 주기 실행(cron)

목표는 5~10분마다 한 번씩 실행되도록 만드는 것입니다.

예시(10분마다):

```bash
crontab -e
```

```cron
*/10 * * * * cd /path/to/gsai_slack_bot && /path/to/gsai_slack_bot/.venv/bin/python gsai_notifier.py >> /path/to/gsai_slack_bot/cron.log 2>&1
```

## 4) GitHub Actions로 주기 실행(선택)

장점:
- 서버 없이도 돌아감

주의(중요):
- 스케줄 실행마다 `state.json`이 업데이트되면 **커밋이 자동으로 쌓입니다**(상태 유지 목적).

설정 방법:

1. GitHub 레포의 **Settings → Secrets and variables → Actions**에서  
   `SLACK_WEBHOOK_URL` 시크릿을 추가합니다.
2. `.github/workflows/gsai_notify.yml`을 사용합니다.

## 설정 변수(요약)

- `FEED_URLS`: 감시할 피드 URL들(쉼표 구분)
- `STATE_FILE`: 상태 파일 경로
- `INITIAL_NOTIFY_COUNT`: 첫 실행에서 최신 N개를 보내고 싶을 때
- `MAX_ITEMS_PER_MESSAGE`: Slack 메시지 1개당 최대 항목 수(많으면 여러 메시지로 분할)
- `ON_STATE_MISS`: 상태가 꼬였을 때 동작(`skip` 권장)
- `VERIFY_SSL`: SSL 검증(기본 true)
- `DRY_RUN`: true면 Slack 전송 대신 콘솔 출력

## 자주 겪는 함정(“왜 알림이 안 오지?”)

- Slack Webhook URL이 비었거나 잘못됨
- 첫 실행에서는 기본적으로 알림을 보내지 않음(`INITIAL_NOTIFY_COUNT=0`)
- 실행 주기가 너무 김(피드에 표시되는 항목 수보다 더 오래 쉬면 상태가 꼬일 수 있음)
- GitHub Actions 스케줄은 실제 실행이 몇 분 늦어질 수 있음(플랫폼 특성)


