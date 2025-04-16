# 밑바닥부터 시작하는 딥러닝 2 - NLP 학습 가이드 및 문제집

## 목차
1. [Ch01 - 신경망 복습](#ch01---신경망-복습)
2. [Ch02 - 자연어와 단어의 분산 표현](#ch02---자연어와-단어의-분산-표현)
3. [Ch03 - word2vec](#ch03---word2vec)
4. [Ch04 - word2vec 속도 개선](#ch04---word2vec-속도-개선)
5. [문제집](#문제집)

---

## Ch01 - 신경망 복습

### 핵심 개념
- 신경망의 기본 구조와 작동 원리
- 순전파(forward propagation)와 역전파(backward propagation)
- 신경망 학습을 위한 기본 구성요소: 계층(Layer), 손실 함수(Loss Function), 최적화(Optimization)

### 주요 구현
- `TwoLayerNet` 클래스: 2층 신경망 구현
- 계층 추상화: Affine, Sigmoid, SoftmaxWithLoss 등
- 학습 루프 구현: 미니배치 방식의 학습 과정

### 학습 포인트
- 딥러닝의 기본 구조를 코드로 이해하기
- 각 계층이 독립적인 모듈로 구현되는 방식 이해하기
- 순전파와 역전파의 계산 과정 이해하기

---

## Ch02 - 자연어와 단어의 분산 표현

### 핵심 개념
- 자연어 처리의 기본 개념
- 단어의 표현 방법: 원-핫 인코딩(one-hot encoding), 분산 표현(distributed representation)
- 통계 기반 기법: 동시 발생 행렬, PMI(Pointwise Mutual Information), SVD(Singular Value Decomposition)

### 주요 구현
- 동시 발생 행렬(co-occurrence matrix) 계산
- PPMI(Positive PMI) 계산
- 차원 감소를 통한 단어 벡터 생성

### 학습 포인트
- 말뭉치(corpus)의 통계적 성질을 이용한 단어 벡터화
- 희소 표현(sparse representation)의 한계와 밀집 표현(dense representation)의 필요성
- 단어 간 유사도 계산 방법

---

## Ch03 - word2vec

### 핵심 개념
- 추론 기반 기법: word2vec의 원리와 구조
- CBOW(Continuous Bag of Words) 모델: 주변 단어로부터 중앙 단어 예측
- Skip-gram 모델: 중앙 단어로부터 주변 단어 예측

### 주요 구현
- `SimpleCBOW` 클래스: CBOW 모델의 기본 구현
- `SimpleSkipGram` 클래스: Skip-gram 모델의 기본 구현
- 단어 벡터의 학습과 활용

### 학습 포인트
- 신경망을 이용한 단어 벡터 학습 방법
- CBOW와 Skip-gram의 차이점과 각각의 장단점
- 학습된 단어 벡터의 특성과 활용

---

## Ch04 - word2vec 속도 개선

### 핵심 개념
- 네거티브 샘플링(Negative Sampling): 효율적인 학습을 위한 근사 기법
- 계층적 소프트맥스(Hierarchical Softmax): 계산 효율성 개선
- word2vec의 실용적 구현

### 주요 구현
- `EmbeddingDot` 클래스: 임베딩 계층과 내적 계산을 효율적으로 처리
- `UnigramSampler` 클래스: 단어 빈도에 기반한 네거티브 샘플 생성
- `NegativeSamplingLoss` 클래스: 네거티브 샘플링을 활용한 손실 함수 구현

### 학습 포인트
- 대규모 말뭉치에서 효율적인 학습 방법
- 네거티브 샘플링의 원리와 구현 방법
- word2vec의 실전 활용을 위한 최적화 기법

---

## 문제집

### Ch01 - 신경망 복습

#### 문제 1: 순전파 구현하기
다음 TwoLayerNet 클래스의 predict 메서드 코드를 완성하세요.

```python
def predict(self, x):
    # 빈칸 채우기
    for layer in self.layers:
        x = layer.forward(x)
    return x
```

#### 문제 2: 역전파 구현하기
다음 TwoLayerNet 클래스의 backward 메서드 코드를 완성하세요.

```python
def backward(self, dout=1):
    dout = self.loss_layer.backward(dout)
    # 빈칸 채우기
    for layer in reversed(self.layers):
        dout = layer.backward(dout)
    return dout
```

### Ch02 - 자연어와 단어의 분산 표현

#### 문제 3: 동시 발생 행렬 구현하기
주어진 말뭉치(corpus)에서 동시 발생 행렬을 계산하는 함수를 완성하세요.

```python
def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    
    for idx, word_id in enumerate(corpus):
        # 빈칸 채우기
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i
            
            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
                
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
    
    return co_matrix
```

#### 문제 4: 코사인 유사도 계산하기
두 단어 벡터 간의 코사인 유사도를 계산하는 함수를 완성하세요.

```python
def cos_similarity(x, y, eps=1e-8):
    # 빈칸 채우기
    nx = x / (np.sqrt(np.sum(x**2)) + eps)
    ny = y / (np.sqrt(np.sum(y**2)) + eps)
    return np.dot(nx, ny)
```

### Ch03 - word2vec

#### 문제 5: CBOW 모델 구현하기
SimpleCBOW 클래스의 forward 메서드를 완성하세요.

```python
def forward(self, contexts, target):
    h0 = self.in_layer0.forward(contexts[:, 0])
    h1 = self.in_layer1.forward(contexts[:, 1])
    # 빈칸 채우기
    h = (h0 + h1) * 0.5
    score = self.out_layer.forward(h)
    loss = self.loss_layer.forward(score, target)
    return loss
```

#### 문제 6: Skip-gram 모델 구현하기
SimpleSkipGram 클래스의 forward 메서드를 완성하세요.

```python
def forward(self, target, contexts):
    batch_size = target.shape[0]
    # 빈칸 채우기
    h = self.in_layer.forward(target)
    
    loss = 0
    for i in range(contexts.shape[1]):
        score = self.out_layers[i].forward(h)
        loss += self.loss_layers[i].forward(score, contexts[:, i])
        
    return loss
```

### Ch04 - word2vec 속도 개선

#### 문제 7: EmbeddingDot 클래스 구현하기
EmbeddingDot 클래스의 forward 메서드를 완성하세요.

```python
def forward(self, h, idx):
    # 빈칸 채우기
    target_W = self.embed.forward(idx)
    out = np.sum(target_W * h, axis=1)
    
    self.cache = (h, target_W)
    return out
```

#### 문제 8: 네거티브 샘플링 손실 함수 구현하기
NegativeSamplingLoss 클래스의 forward 메서드를 완성하세요.

```python
def forward(self, h, target):
    batch_size = target.shape[0]
    negative_sample = self.sampler.get_negative_sample(target)
    
    # 긍정적 예 순전파
    score = self.embed_dot_layers[0].forward(h, target)
    correct_label = np.ones(batch_size, dtype=np.int32)
    loss = self.loss_layers[0].forward(score, correct_label)
    
    # 부정적 예 순전파
    # 빈칸 채우기
    negative_label = np.zeros(batch_size, dtype=np.int32)
    for i in range(self.sample_size):
        negative_target = negative_sample[:, i]
        score = self.embed_dot_layers[1 + i].forward(h, negative_target)
        loss += self.loss_layers[1 + i].forward(score, negative_label)
    
    return loss
```

#### 문제 9: 개념 문제 - 네거티브 샘플링의 이점
네거티브 샘플링(Negative Sampling)이 기존 word2vec에 비해 어떤 계산적 이점을 제공하는지 설명하세요. 계산 복잡도 측면에서의 차이점을 포함하여 답변하세요.

#### 문제 10: 개념 문제 - 단어 벡터의 특성
word2vec으로 학습된 단어 벡터가 가지는 중요한 특성 중 하나는 단어 간의 의미적 관계가 벡터 공간에서의 연산으로 표현될 수 있다는 것입니다. 이를 설명하는 유명한 예시('king - man + woman = queen')를 통해 이러한 특성이 어떻게 나타나는지 설명하세요.

---

추가 학습 자료와 참고 문헌은 "밑바닥부터 시작하는 딥러닝 2" 교재를 참고하세요. 