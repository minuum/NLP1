# NLP1 프로젝트: T5와 Dynamic Context Gates (DCG)

이 프로젝트는 T5(Text-to-Text Transfer Transformer) 모델을 확장하여 Dynamic Context Gates(DCG)를 적용한 자연어 생성 실험을 담고 있습니다. 주요 응용 사례로는 ArXiv 초록으로부터 논문 제목을 생성하는 작업을 수행합니다.

## 프로젝트 개요

Dynamic Context Gates는 소스 컨텍스트(인코더)와 타겟 컨텍스트(디코더) 사이의 정보 흐름을 동적으로 제어하여 자연어 생성의 품질을 향상시키는 메커니즘입니다. 기존 Transformer 모델의 자기 주의(self-attention)와 교차 주의(cross-attention) 메커니즘을 보완하는 방식으로 작동합니다.

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

### 1. 일반 T5와 DCG 모델의 차이점

DCG는 다음과 같은 주요 차이점을 가집니다:

- **소스와 타겟 컨텍스트 균형 조절**: DCG는 인코더(소스)와 디코더(타겟) 사이의 정보 흐름을 동적으로 제어합니다.
- **게이팅 메커니즘**: 소스와 타겟 컨텍스트의 중요도에 따라 가중치를 동적으로 조절합니다.

### 2. DCG 구현 방식

프로젝트에는 두 가지 주요 DCG 구현이 포함되어 있습니다:

#### A. 향상된 DCG (dcg_train_2.py)
- **다중 디코더 레이어에 DCG 적용** (전략적 위치)
- **멀티헤드 어텐션 기반 게이팅**
- **내용어 편향 메커니즘**
- **위치 인식 게이팅**
- **적응형 임계값 학습**

#### B. 단순화된 DCG (rp_dcg_train_3.py)
- **단일 디코더 레이어에만 DCG 적용** (마지막 레이어)
- **단순화된 게이트 메커니즘**
- **점진적 DCG 활성화로 안정적인 학습**
- **경량화된 아키텍처**

## 데이터셋

ArXiv 논문 메타데이터를 사용하여 초록에서 제목을 생성하는 실험을 진행합니다. 데이터셋은 다음 경로에 저장되어 있습니다:
- `arxiv_data/arxiv-metadata-oai-snapshot.json`
- `data/arxiv_papers.json` (전처리된 데이터)

## 실행 방법

향상된 DCG 모델 학습:
```
python exp_code/dcg_train_2.py
```

단순화된 DCG 모델 학습:
```
python exp_code/rp_dcg_train_3.py
```

모델 평가:
```
python exp_code/dcg_eval_2.py
python exp_code/rp_dcg_eval_3.py
```

## 참고 자료

- 실험 결과와 추가 분석은 `notebooks/` 디렉토리에서 확인할 수 있습니다.
- 발표 자료는 `pdf/` 및 `자연어처리1 중간주제발표_250409/` 디렉토리에 있습니다.