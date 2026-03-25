# 혐오 표현 필터링 + LLM 채팅 프로그램

독립 실행 가능한 혐오 표현 필터링 + LLM 채팅 프로그램입니다.

## 특징

- ✅ **독립 실행**: 별도 디렉토리에 모든 파일 포함
- ✅ **간편한 설정**: 처음 실행 시 Groq API 키만 입력
- ✅ **자동 저장**: API 키가 설정 파일에 자동 저장
- ✅ **다음 실행부터 자동 로드**: 재실행 시 설정 자동 로드
- ✅ **3단계 필터링**: 입력 검사 → 응답 검사 → 재요청 응답 검사

## 설치

### 1. 필수 패키지 설치

```bash
cd filtering_chat_app
pip install -r requirements.txt
```

### 2. 필터링 모델 확인

프로그램은 상위 디렉토리의 `scripts` 디렉토리에서 필터링 모델을 로드합니다. 
모델이 올바르게 설정되어 있는지 확인하세요.

## 사용 방법

### 1. 첫 실행

```bash
python main.py
```

프로그램이 실행되면 Groq API 키를 입력하라는 메시지가 나타납니다.

**API 키 발급 방법:**
1. https://console.groq.com 접속
2. 회원가입 후 API Keys 메뉴에서 키 생성
3. 생성된 API 키를 복사하여 입력

API 키는 `config.json` 파일에 자동으로 저장됩니다.

### 2. 이후 실행

```bash
python main.py
```

API 키가 자동으로 로드되어 바로 채팅을 시작할 수 있습니다.

### 3. 실행 옵션

```bash
# 재요청 횟수 조정 (기본값: 1)
python main.py --max-retries 10
```

### 4. 채팅 사용법

프로그램 실행 후:

- **메시지 입력**: 메시지를 입력하고 Enter를 누르세요
- **종료**: `quit`, `exit`, 또는 `종료`를 입력하세요
- **도움말**: `help` 또는 `도움말`을 입력하세요

## 아키텍처

```
사용자 입력 
    ↓
[1차 필터링: 입력 검사]
    ↓ (통과)
LLM 응답 생성
    ↓
[2차 필터링: 응답 검사]
    ↓ (통과)          ↓ (실패)
사용자에게 응답    재요청 (1차 필터링 건너뛰기)
                        ↓
                    LLM 응답 생성
                        ↓
                    [2차 필터링: 응답 검사]
                        ↓ (통과)
                    사용자에게 응답
```

### 필터링 프로세스

1. **1차 필터링**: 사용자 입력을 검사하여 혐오 표현이 있는지 확인
   - 혐오 표현이 감지되면 경고 메시지 표시 및 처리 중단
   - 통과하면 LLM에 전달

2. **2차 필터링**: LLM이 생성한 응답을 검사
   - 혐오 표현이 감지되면 재요청 (최대 재시도 횟수까지)
   - 통과하면 사용자에게 응답 표시

3. **재요청 최적화**: 재요청 시에는 같은 입력을 사용하므로 1차 필터링을 건너뛰고 바로 LLM 요청

## 설정 파일

설정은 `config.json` 파일에 저장됩니다:

```json
{
  "groq_api_key": "your-api-key-here",
  "groq_default_model": "llama-3.1-8b-instant",
  "filter_threshold": 0.5,
  "max_retries": 1,
  "retry_delay": 1.0
}
```

### 설정 항목 설명

- `groq_api_key`: Groq API 키 (필수)
- `groq_default_model`: 기본 사용 모델 (기본값: llama-3.1-8b-instant)
- `filter_threshold`: 필터링 임계값 (0.0 ~ 1.0, 기본값: 0.5)
- `max_retries`: 최대 재요청 횟수 (기본값: 1)
- `retry_delay`: 재요청 지연 시간 초 (기본값: 1.0)

## 파일 구조

```
filtering_chat_app/
├── main.py                 # 메인 프로그램
├── config.py               # 설정 관리
├── llm_client.py           # LLM API 클라이언트
├── filtering_service.py    # 필터링 서비스
├── config.json             # 설정 파일 (자동 생성)
├── requirements.txt        # 필수 패키지 목록
└── README.md              # 이 문서
```
