# Ab2Ti: T5와 Dynamic Context Gates (DCG)를 활용한 논문 초록-제목 생성 프로젝트

이 프로젝트는 T5(Text-to-Text Transfer Transformer) 모델을 확장하여 Dynamic Context Gates(DCG)를 적용한 자연어 생성 실험을 담고 있습니다. 주요 응용 사례로는 ArXiv 논문 초록으로부터 논문 제목을 생성하는 작업(Abstract to Title, Ab2Ti)을 수행합니다.

## 프로젝트 개요

### 목적 및 동기

학술 논문 작성 시 초록(Abstract)은 완성되었으나 적절한 제목을 찾기 어려운 경우가 많습니다. 본 프로젝트는 이러한 문제를 해결하기 위해 논문 초록에서 핵심 내용을 파악하고 적절한 제목을 자동으로 생성하는 모델을 개발했습니다. 이를 통해:

1. 연구자들이 논문 제목 작성에 소요되는 시간과 노력을 절약
2. 초록의 핵심 내용을 효과적으로 반영하는 제목 생성
3. 다양한 학술 분야에 범용적으로 적용 가능한 모델 개발

### 기술적 접근

기본 T5 모델의 성능을 향상시키기 위해 Dynamic Context Gates(DCG) 메커니즘을 도입했습니다. DCG는 소스 컨텍스트(인코더)와 타겟 컨텍스트(디코더) 사이의 정보 흐름을 동적으로 제어하여 자연어 생성의 품질을 향상시키는 메커니즘입니다. 기존 Transformer 모델의 자기 주의(self-attention)와 교차 주의(cross-attention) 메커니즘을 보완하는 방식으로 작동합니다.

## 데이터셋

ArXiv 논문 메타데이터를 사용하여 초록에서 제목을 생성하는 실험을 진행합니다. 데이터셋은 다음과 같이 제공됩니다:

- **원본 데이터**: [Kaggle ArXiv 데이터셋](https://www.kaggle.com/datasets/Cornell-University/arxiv/data)
- **저장 경로**: 
  - `arxiv_data/arxiv-metadata-oai-snapshot.json` (원본 데이터)
  - `data/arxiv_papers.json` (전처리된 데이터)

> **참고**: 원본 데이터셋은 크기가 4GB 이상으로, GitHub의 파일 크기 제한(100MB)을 초과합니다. 따라서 위 Kaggle 링크에서 직접 다운로드하여 사용하시기 바랍니다.

### 데이터셋 특징

- 170만 개 이상의 STEM(과학, 기술, 공학, 수학) 분야 논문 메타데이터
- 1991년부터 현재까지의 ArXiv 논문 정보 포함
- 각 논문별로 title, abstract, authors, categories 등의 정보 제공

## 프로젝트 워크플로우

1. **데이터 수집 및 전처리**:
   - ArXiv 메타데이터에서 초록과 제목 쌍을 추출
   - 텍스트 정제 및 토큰화
   - 훈련/검증/테스트 데이터셋 분할

2. **모델 개발**:
   - 기본 T5 모델 구현
   - DCG 메커니즘 설계 및 통합
   - 두 가지 버전의 DCG 구현 (향상된 버전과 단순화된 버전)

3. **모델 학습**:
   - 다양한 하이퍼파라미터 설정으로 실험
   - 게이트 정규화 및 적응적 임계값 적용
   - 점진적 DCG 활성화를 통한 안정적 학습

4. **평가 및 분석**:
   - ROUGE, BLEU 등의 메트릭을 통한 정량적 평가
   - 생성된 제목의 정성적 분석
   - 일반 T5와 DCG 적용 모델 간의 성능 비교

## 디렉토리 구조

```
NLP1/
├── __pycache__/          # 파이썬 캐시 파일
├── .vscode/              # VSCode 설정
├── arxiv_data/           # ArXiv 메타데이터
├── common/               # 공통 유틸리티 및 모델 컴포넌트
├── data/                 # 학습 및 평가 데이터셋
├── exp_code/             # 실험 코드
│   ├── dcg_train_2.py    # 향상된 DCG 모델 학습 코드
│   ├── rp_dcg_train_3.py # 단순화된 DCG 모델 학습 코드
│   └── ...               # 기타 모델 평가 및 학습 코드
├── notebooks/            # 주차별 실습 노트북
├── pdf/                  # 수업 자료 및 발표 자료
├── source/               # 소스 코드
├── test_markdown/        # 마크다운 테스트 파일
├── text-to-text-transfer-transformer/ # T5 기본 코드베이스
└── 자연어처리1 중간주제발표_250409/ # 발표 자료
```

## 핵심 모델: DCG(Dynamic Context Gates)

### 1. DCG의 기본 개념

DCG(Dynamic Context Gates)는 소스 컨텍스트(인코더)와 타겟 컨텍스트(디코더) 사이의 정보 흐름을 동적으로 제어하는 메커니즘입니다. 원래 Tu et al.(2017)이 RNN 기반 기계 번역에서 제안했던 개념을 Transformer 기반 T5 모델에 적용했습니다.

#### 핵심 아이디어:
- 소스와 타겟 컨텍스트 균형 조절
- 게이팅 메커니즘을 통한 컨텍스트 선택적 활용
- 내용어(content words) 중심의 정보 흐름 강화

### 2. DCG 구현 방식

#### 2.1 동적 게이트 계산
```
gate = sigmoid([c; ht] · W + b)
```
- c: 인코더 컨텍스트 벡터
- ht: 디코더 현재 히든 스테이트
- W, b: 학습 가능한 파라미터

#### 2.2 컨텍스트 융합
```
c't = gate * c + (1 - gate) * ht
```
- c't: 융합된 컨텍스트 (다음 레이어 입력)

#### 2.3 구현 접근법
프로젝트에는 두 가지 주요 DCG 구현이 포함되어 있습니다:

##### A. 향상된 DCG (dcg_train_2.py)
- **다중 디코더 레이어에 DCG 적용** (전략적 위치: 레이어 0, 2, 4)
- **멀티헤드 어텐션 기반 게이팅**
- **내용어 편향 메커니즘**
- **위치 인식 게이팅**
- **적응형 임계값 학습**

##### B. 단순화된 DCG (rp_dcg_train_3.py)
- **단일 디코더 레이어에만 DCG 적용** (마지막 레이어)
- **단순화된 게이트 메커니즘**
- **점진적 DCG 활성화로 안정적인 학습**
- **경량화된 아키텍처**

## 모델 구성

### 기본 설정
- **모델**: t5-small
- **배치 크기**: 8
- **학습률**: 3e-5
- **에폭**: 5
- **입력 길이**: 512 (초록)
- **출력 길이**: 128 (제목)
- **훈련 데이터**: 9000개 샘플
- **검증/테스트 데이터**: 각 1000개 샘플

## 실행 방법

### 데이터 준비
```bash
# Kaggle에서 ArXiv 데이터셋 다운로드
kaggle datasets download -d Cornell-University/arxiv
# 다운로드한 파일을 arxiv_data 디렉토리에 압축 해제
mkdir -p arxiv_data
unzip arxiv.zip -d arxiv_data
```

### 모델 학습
향상된 DCG 모델 학습:
```bash
python exp_code/dcg_train_2.py
```

단순화된 DCG 모델 학습:
```bash
python exp_code/rp_dcg_train_3.py
```

### 모델 평가
```bash
python exp_code/dcg_eval_2.py
python exp_code/rp_dcg_eval_3.py
```

## 실험 결과

### 성능 비교

| 모델 | ROUGE-1 | ROUGE-2 | ROUGE-L | BERT Score F1 |
|------|---------|---------|---------|--------------|
| Base T5 | 0.383 | 0.213 | 0.348 | 0.843 |
| AdvancedDCG T5 | 0.242 | 0.094 | 0.204 | 0.787 |
| SimpleDCG T5 | 0.389 | 0.214 | 0.353 | 0.844 |

### 결과 분석

- **SimpleDCG T5**가 모든 평가 지표에서 최고 성능을 보였습니다.
- 단순화된 DCG 모델이 기본 T5보다 ROUGE-1, ROUGE-2, ROUGE-L, BERT Score 모두에서 소폭 향상된 결과를 보였습니다.
- 향상된 DCG 모델(AdvancedDCG)은 오히려 성능이 저하되었는데, 이는 복잡한 게이트 메커니즘이 오히려 학습을 방해했을 가능성이 있습니다.
- 길이 분석 결과, Base T5는 평균 제목 길이가 8.1 단어, SimpleDCG T5는 8.4 단어로 실제 논문 제목과 유사한 길이를 생성했습니다.

### 핵심 결론

- RNN 기반 모델에서만 적용되던 DCG를 Transformer 기반 T5 모델에 성공적으로 적용
- 단순화된 DCG 구현이 가장 좋은 성능을 보임 (BLEU 점수: 0.087, ROUGE-1: 0.389)
- 본 연구는 논문 초록에서 제목 생성이라는 특정 태스크에서 DCG 메커니즘의 효과를 입증
- DCG는 특히 '내용 충실성(content faithfulness)'을 향상시키는 데 효과적임

## 향후 개선 방향

### 알고리즘 측면
1. **게이트 융합 메커니즘 개선**: 쉬운 데이터 부터 학습하는 curriculum learning 알고리즘 적용
2. **커리큘럼 러닝 적용**: 복잡도가 낮은 초록-제목 쌍부터 점진적으로 학습

### 평가 메트릭 추가
1. **학습 중 평가 메트릭 보완**: BERT-Score, METEOR Score 등 추가 평가 지표 도입
2. **정성적 평가 확대**: 인간 평가자를 통한 제목의 품질 평가

## 참고 자료

- 실험 결과와 추가 분석은 `exp_code/` 디렉토리에서 확인할 수 있습니다.
- 발표 자료는 `pdf/` 안의 `자연어처리1 중간주제발표_250514.pdf/` 입니다.
- [T5 원본 논문](https://arxiv.org/abs/1910.10683)
- [Kaggle ArXiv 데이터셋](https://www.kaggle.com/datasets/Cornell-University/arxiv/data)
- [Context Gates for Neural Machine Translation](https://arxiv.org/abs/1608.06043) (Tu et al., 2017)
- [BERTScore](https://arxiv.org/abs/1904.09675) (Zhang et al., ICLR 2020)
- [METEOR](https://aclanthology.org/W05-0909/) (Banerjee & Lavie, ACL 2005)

## GitHub 저장소

https://github.com/minuum/NLP1