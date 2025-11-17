# Korean Sovereign Alignment Guard Model

## 1. 프로젝트 개요

본 프로젝트는 대한민국 주권(영토, 역사, 문화)과 관련된 유해성 프롬프트(harmful prompt)를 탐지하고, 이를 중립적이고 사실에 기반한 질문으로 교정하는 가드(Guard) 모델 및 파이프라인입니다.

주요 기능은 다음과 같습니다.

- **유해성 탐지**: 사용자의 프롬프트가 대한민국 주권과 관련된 왜곡된 주장을 담고 있는지 판별합니다.
- **질문 교정**: 유해성으로 판별된 프롬프트를 기반으로, 사용자의 원래 의도는 유지하되 중립적인 질문으로 재생성합니다.
- **3단계 분류 모델**:
  1.  **도메인 필터**: 프롬프트가 프로젝트에서 다루는 핵심 도메인(영토, 고대사, 전쟁, 일제 인권)에 속하는지 확인합니다.
  2.  **의도 분류기 (Intent Classifier)**: 프롬프트가 단순 정보 질문인지, 왜곡된 주장을 사실처럼 단정하는지 의도를 분류합니다.
  3.  **NLI 분류기 (Natural Language Inference)**: 프롬프트와 '대한민국 입장 팩트' 사이의 관계가 모순(Contradiction)인지 판별합니다.

> **최종 유해성 판정**: `도메인 내 발화` & `ASSERTION(단정) 의도` & `팩트와 모순` 세 조건을 모두 만족할 때 **유해(Harmful)**로 판단합니다.

## 2. 프로젝트 구조

리팩토링을 통해 프로젝트 구조를 기능별로 명확하게 분리했습니다.

```
.
├── Dataset/                  # 원본 및 생성된 데이터셋
│   ├── augmented_dataset.json
│   ├── k_Japan_war_harmful_prompts.json
│   ├── K-Distortion_Red_Team_Dataset.json
│   └── K-Territory_Red_Team_Dataset.json
│
├── moderator_guard/          # 메인 소스 코드
│   ├── classifier/           # 분류 모델 관련 코드
│   │   ├── core.py           # 핵심 분류 로직 (도메인, 의도, NLI)
│   │   ├── utils.py          # 데이터 로딩 및 전처리 유틸리티
│   │   └── ... (모델 및 캐시 파일)
│   │
│   ├── data/                 # 데이터 생성 및 증강
│   │   └── build_augmented_dataset.py
│   │
│   ├── llm/                  # 언어 모델 클라이언트
│   │   └── client.py
│   │
│   ├── training/             # 모델 학습 스크립트
│   │   └── train_classifier.py
│   │
│   ├── config.py             # 프로젝트 설정 (경로, 모델명 등)
│   ├── prompt_rewriter_app.py # 최종 End-to-End 질문 교정 애플리케이션
│   └── test_pipeline.py      # 분류 모델 테스트 스크립트
│
└── README.md
```

## 3. 설치 및 설정

1.  **필요 라이브러리 설치**

    ```bash
    pip install -r moderator_guard/requirements.txt
    ```

2.  **API 키 설정**
    - **OpenAI API 키**: `moderator_guard/llm/client.py` 파일 내의 `client = OpenAI()`가 환경변수 `OPENAI_API_KEY`를 사용하도록 설정되어 있습니다. 시스템 환경변수에 키를 등록하세요.
    - **Hugging Face 토큰**: `moderator_guard/prompt_rewriter_app.py` 파일 상단의 `HF_TOKEN` 변수에 본인의 Hugging Face 토큰을 입력해야 합니다. (Llama 모델 다운로드 시 필요)

3.  **필수 데이터 파일 확인**
    - `Dataset/korea_fact_positions.json` 파일이 필요합니다. 이 파일은 '대한민국 입장 팩트' 데이터셋으로, NLI 모델의 기준이 됩니다. 현재 이 파일이 누락되어 있으니, 실행 전 반드시 해당 경로에 추가해야 합니다.

## 4. 사용 방법

### 4.1. 데이터 증강 (최초 1회)

`harmful_prompt`만 있는 원본 데이터셋(`Dataset/*.json`)을 기반으로, LLM을 사용하여 `benign_prompt`와 `ideal_answer`를 생성합니다.

```bash
python moderator_guard/data/build_augmented_dataset.py
```

- 실행 시 `Dataset` 폴더 내의 `k_*.json` 파일들을 읽어 `augmented_dataset.json` 파일을 생성합니다.

### 4.2. NLI 분류기 학습 (최초 1회)

`augmented_dataset.json` 데이터셋을 사용하여 '모순' 관계를 학습하는 NLI 분류기를 학습합니다.

1.  **팩트 인덱스 생성**: `korea_fact_positions.json` 파일로부터 임베딩에 사용할 팩트 텍스트와 임베딩 파일을 생성합니다.
    ```bash
    # moderator_guard/classifier/utils.py의 build_fact_index_from_positions 함수를 직접 실행하거나,
    # 아래 학습 과정에 포함시킬 수 있습니다.
    ```
    *(현재 `train_classifier.py`에는 이 과정이 포함되어 있지 않으므로, 별도 실행 필요)*

2.  **NLI 분류기 학습**:
    ```bash
    python moderator_guard/training/train_classifier.py
    ```
    - 실행 시 `augmented_dataset.json`을 읽어 NLI 학습용 페어를 만들고, 임베딩을 계산하여 캐시(`nli_*.npy`)를 생성합니다.
    - 최종적으로 학습된 모델(`nli_contradict_classifier.joblib`)이 `moderator_guard/classifier/` 폴더에 저장됩니다.

### 4.3. 질문 교정 애플리케이션 실행

학습된 모델을 사용하여 실시간으로 프롬프트를 입력받아 분류하고 교정합니다.

```bash
python moderator_guard/prompt_rewriter_app.py
```

- 실행 후 터미널에 프롬프트를 입력하면, `[분류 결과]`와 `=== 교정된 질문 ===`이 차례로 출력됩니다.

## 5. 주요 모듈 설명

- **`config.py`**: 모든 파일 경로, 모델 이름, 키워드 등 주요 설정을 관리합니다.
- **`llm/client.py`**: OpenAI 클라이언트를 초기화하고 임베딩을 생성하는 함수를 제공합니다.
- **`classifier/core.py`**: `guard_classify` 함수를 통해 3단계 분류 로직을 수행하는 핵심 모듈입니다.
- **`training/train_classifier.py`**: NLI 분류기 모델을 학습하고 저장하는 스크립트입니다.
- **`prompt_rewriter_app.py`**: 분류기와 생성 모델(Llama)을 결합하여 전체 파이프라인을 실행하는 메인 애플리케이션입니다.
