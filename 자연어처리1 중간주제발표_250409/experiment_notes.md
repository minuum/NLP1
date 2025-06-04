# Abstract-to-Title Generation: Base T5 vs. Enhanced DCG T5 실험 노트

## 1. 프로젝트 개요 및 목표

본 프로젝트는 논문 초록(abstract)을 입력으로 받아 적절한 제목(title)을 생성하는 모델을 개발하고 평가하는 것을 목표로 합니다.

*   **Base Model**: 표준 T5 모델 (t5-small)을 사용하여 기본적인 성능을 측정합니다.
*   **Enhanced DCG Model**: T5 모델에 Enhanced Dynamic Context Gate (DCG) 메커니즘을 추가하여 성능 향상을 시도합니다.
*   **평가 지표**: ROUGE, BERTScore, BLEU 등 정량적 지표와 함께, 생성된 제목의 정성적 품질을 분석합니다. DCG 모델의 경우, 게이트 활성화, 어텐션 패턴 등 내부 동작 분석도 포함합니다.

## 2. 코드 및 디렉토리 구조

*   **`NLP1/`** (프로젝트 루트)
    *   **`base_model/`**: Base T5 모델 관련 파일
        *   `base_train.py`: Base T5 모델 학습 스크립트
        *   `base_evaluate.py`: Base T5 모델 평가 스크립트
        *   `data/`: 학습 데이터 및 전처리된 데이터 (`processed_data.json`) 저장
        *   `model/`: 학습된 Base T5 모델 및 체크포인트 저장
        *   `results/`: Base T5 모델 평가 결과 (JSON, CSV, PNG) 저장
    *   **`dcg_model/`**: Enhanced DCG T5 모델 관련 파일
        *   `enhanced_dcg_train.py`: Enhanced DCG T5 모델 학습 스크립트
        *   `enhanced_dcg_eval.py`: Enhanced DCG T5 모델 평가 스크립트
        *   `config_dcg.py` (사용자 제공 폴더에 있었으나, `enhanced_dcg_train.py` 내부에 CONFIG로 통합된 것으로 보임)
        *   `enhanced_dcg_model/`: 학습된 Enhanced DCG T5 모델 및 설정 파일 저장
        *   `enhanced_results/`: Enhanced DCG T5 모델 평가 결과 (JSON, CSV, PNG) 저장
    *   `experiment_notes.md`: 본 실험 노트 파일

## 3. 모델 설명

### 3.1. Base T5 Model

*   Hugging Face Transformers 라이브러리의 표준 `T5ForConditionalGeneration` 모델 (`t5-small`)을 사용합니다.
*   Kaggle의 arXiv 데이터셋으로 fine-tuning하여 초록 요약 (제목 생성) 태스크를 수행합니다.
*   `base_train.py`로 학습하고, `base_evaluate.py`로 평가합니다.

### 3.2. Enhanced Dynamic Context Gate (DCG) T5 Model

*   Base T5 모델의 인코더 출력에 `EnhancedDynamicContextGate` 모듈을 적용하여 컨텍스트 이해도를 높이고자 합니다.
*   **주요 커스텀 모듈 (`enhanced_dcg_train.py`):**
    *   `MultiHeadContextAttention`: 컨텍스트 선택을 위한 멀티헤드 어텐션.
    *   `ContextMemoryModule`: 컨텍스트 패턴을 저장하고 검색하기 위한 메모리 모듈.
    *   `AdaptiveGatingMechanism`: 다양한 게이팅 전략을 결합하는 적응형 게이트.
    *   `EnhancedDynamicContextGate`: 위의 모듈들을 통합하여 동적 컨텍스트를 조절.
    *   `T5ForConditionalGenerationWithEnhancedDCG`: `EnhancedDynamicContextGate`를 T5 모델에 통합한 커스텀 모델 클래스.
*   `enhanced_dcg_train.py`로 학습하고, `enhanced_dcg_eval.py`로 평가합니다. 평가 시 DCG 내부 동작(어텐션, 메모리, 게이트)에 대한 분석도 수행합니다.

## 4. 실험 실행 단계

**사전 준비:**
*   필요한 라이브러리가 설치되어 있는지 확인합니다 (`transformers`, `torch`, `datasets`, `evaluate`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `kagglehub`, `rouge_score`).
*   Kaggle API 설정이 되어있어야 `kagglehub`를 통해 데이터셋 다운로드가 가능합니다 (최초 실행 시). `base_model/data/processed_data.json` 파일이 이미 있다면 다운로드를 건너뛸 수 있습니다.

### 4.1. Base T5 모델 학습 및 평가

1.  **작업 디렉토리 변경:**
    ```bash
    cd /Users/minu/Desktop/4-1/NLP1/base_model
    ```
2.  **학습 실행:**
    ```bash
    python base_train.py
    ```
    *   학습된 모델은 `NLP1/base_model/model/`에 저장됩니다.
    *   학습 로그 및 결과는 `NLP1/base_model/model/train_results.json` 및 `eval_results.json`에 저장됩니다.
3.  **평가 실행:**
    ```bash
    python base_evaluate.py
    ```
    *   평가 결과(지표, 플롯)는 `NLP1/base_model/results/` 디렉토리에 저장됩니다. (`base_t5_evaluation_summary.json`, `base_t5_detailed_results.csv`, `base_t5_comprehensive_analysis.png`)

### 4.2. Enhanced DCG T5 모델 학습 및 평가

1.  **작업 디렉토리 변경:**
    ```bash
    cd /Users/minu/Desktop/4-1/NLP1/dcg_model
    ```
2.  **학습 실행:**
    ```bash
    python enhanced_dcg_train.py
    ```
    *   학습된 모델은 `NLP1/dcg_model/enhanced_dcg_model/`에 저장됩니다.
    *   `dcg_config.json`도 함께 저장됩니다.
    *   학습 로그 및 결과는 `NLP1/dcg_model/enhanced_dcg_model/enhanced_train_results.json` 및 `enhanced_eval_results.json`에 저장됩니다.
3.  **평가 실행:**
    ```bash
    python enhanced_dcg_eval.py
    ```
    *   평가 결과(지표, 플롯, DCG 분석)는 `NLP1/dcg_model/enhanced_results/` 디렉토리에 저장됩니다. (`enhanced_dcg_evaluation_summary.json`, `enhanced_dcg_detailed_results.csv`, `enhanced_dcg_comprehensive_analysis.png`)

## 5. 결과 분석 및 비교

실험 실행 후 다음 항목들을 비교 분석합니다:

1.  **주요 성능 지표 비교:**
    *   ROUGE (ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum)
    *   BERTScore (F1, Precision, Recall) - 평균 및 분포
    *   BLEU Score
    *   각 모델의 `results` 및 `enhanced_results` 폴더에 저장된 `*_evaluation_summary.json` 파일과 `*_comprehensive_analysis.png` 플롯을 주로 참조합니다.

2.  **생성된 제목의 정성적 평가:**
    *   `*_detailed_results.csv` 파일에서 실제 생성된 제목과 참조 제목을 비교합니다.
    *   의미론적 유사성, 문법적 정확성, 관련성 등을 평가합니다.
    *   Base T5 모델과 Enhanced DCG 모델 간의 생성 품질 차이를 분석합니다.

3.  **Enhanced DCG 모델 내부 동작 분석 (해당하는 경우):**
    *   `enhanced_dcg_eval.py` 실행 시 생성된 `enhanced_dcg_comprehensive_analysis.png` 내 DCG Component Analysis 플롯 참조.
    *   `enhanced_dcg_evaluation_summary.json` 내 `dcg_analysis` 섹션 참조.
    *   Attention entropy, active memory slots, gate values/sparsity 등의 지표가 모델 성능과 어떤 연관이 있는지 살펴봅니다.

4.  **학습 과정 분석:**
    *   Training/Validation loss 변화 양상 비교.
    *   각 모델의 `model/eval_results.json` 및 `enhanced_dcg_model/enhanced_eval_results.json` 참조.

## 6. 논의 및 다음 단계

*   **현재 상황:** 초기 실행 결과, Base T5 모델이 Enhanced DCG 모델보다 우수한 성능을 보일 수 있다는 관찰이 있었습니다.
*   **DCG 모델 성능 부진 시 고려 사항:**
    *   **구현 오류 점검**: `EnhancedDynamicContextGate` 및 관련 모듈 (Attention, Memory, Gating) 로직 재검토.
    *   **하이퍼파라미터 튜닝**: DCG 관련 하이퍼파라미터 (`num_heads`, `context_dim`, `memory_size`, `dropout`, `gate_activation` 등) 조정.
    *   **통합 방식 점검**: T5 인코더/디코더에 DCG 모듈이 효과적으로 통합되었는지, 정보 흐름이 의도대로 작동하는지 확인.
    *   **과적합/과소적합**: 학습 데이터 크기, epoch 수, learning rate 등 일반적인 학습 파라미터도 DCG 모델에 맞게 조정 필요 가능성.
    *   **단순화된 DCG**: 현재 `EnhancedDynamicContextGate`가 여러 복잡한 메커니즘을 포함하고 있는데, 이를 단순화한 버전부터 시작하여 점진적으로 개선하는 방안도 고려.
*   **결론 도출:**
    *   실험 결과를 바탕으로 Enhanced DCG 접근 방식의 유효성을 평가합니다.
    *   만약 DCG 모델이 지속적으로 Base T5보다 성능이 낮거나 개선이 어렵다고 판단되면, DCG 접근 방식을 포기하고 Base T5 모델을 개선하거나 다른 아키텍처를 탐색하는 것을 고려할 수 있습니다.

## 7. 실행 로그 및 결과 기록

(이 섹션은 사용자가 실험을 진행하면서 로그나 주요 결과 수치를 기록하는 공간입니다.)

### Base T5 Model Log:

```
(여기에 base_train.py 및 base_evaluate.py 실행 시 주요 로그 및 결과 요약)
```

### Enhanced DCG T5 Model Log:

```
(여기에 enhanced_dcg_train.py 및 enhanced_dcg_eval.py 실행 시 주요 로그 및 결과 요약)
```

---
*이민우님, 위 실험 노트를 바탕으로 Base 모델과 Enhanced DCG 모델의 학습 및 평가를 진행해주시면 됩니다. 각 스크립트 실행은 해당 스크립트가 위치한 디렉토리 (`base_model` 또는 `dcg_model`) 내부에서 진행해야 합니다.* 