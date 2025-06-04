"""
AdvancedDCG: 향상된 Dynamic Context Gates를 T5 모델에 적용한 구현

이 모듈은 다음과 같은 특징을 가집니다:
1. 다중 디코더 레이어에 DCG 적용 (전략적 위치)
2. 멀티헤드 어텐션 기반 게이팅
3. 내용어 편향 메커니즘
4. 위치 인식 게이팅
5. 적응형 임계값 학습
"""

import os
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration,
    T5Config,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from transformers.modeling_outputs import Seq2SeqLMOutput
import kagglehub
from kagglehub import KaggleDatasetAdapter

# DCG 모델 설정
CONFIG = {
    "model_name": "t5-small",              # 기본 모델
    "output_dir": "./enhanced_dcg_model",  # 모델 저장 경로
    "data_dir": "./data",                  # 데이터 저장 경로
    "batch_size": 8,                       # 배치 크기
    "learning_rate": 2e-5,                 # 학습률
    "num_epochs": 3,                       # 에폭 수
    "max_input_length": 512,               # 최대 입력 길이
    "max_target_length": 128,              # 최대 출력 길이
    "train_size": 3000,                    # 훈련 데이터 크기
    "val_size": 300,                       # 검증 데이터 크기
    "test_size": 300,                      # 테스트 데이터 크기
    "gate_dropout": 0.2,                   # DCG 게이트 드롭아웃 비율
    "gate_regularization_weight": 0.08,    # 게이트 정규화 가중치
    "content_bias_weight": 0.15,           # 내용어 편향 가중치
    "warmup_ratio": 0.1,                   # 워밍업 비율
    "gradient_accumulation_steps": 2       # 경사 누적 단계
}

class AdvancedDynamicContextGates(nn.Module):
    """
    향상된 Dynamic Context Gates:
    1. 멀티헤드 어텐션 기반 게이팅
    2. 내용어 편향 메커니즘
    3. 위치 인식 게이팅
    4. 적응형 임계값 학습
    """
    def __init__(self, hidden_size, num_heads=4, dropout_rate=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # 컨텍스트 이해를 위한 멀티헤드 어텐션
        self.context_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # 내용어 탐지 모듈
        self.content_detector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # 위치 인식 임베딩
        self.position_embedding = nn.Embedding(128, hidden_size)
        
        # 소스 컨텍스트 게이트
        self.source_gate = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # 타겟 컨텍스트 게이트
        self.target_gate = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # 적응형 임계값 학습
        self.adaptive_threshold = nn.Parameter(torch.tensor(0.5))
        
        # 컨텍스트 융합 레이어
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, decoder_hidden, encoder_outputs, cross_attention_output, position_ids=None):
        """향상된 순전파"""
        batch_size, seq_len, hidden_size = decoder_hidden.shape
        
        # 위치 임베딩 생성
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=decoder_hidden.device).unsqueeze(0).expand(batch_size, -1)
        
        position_emb = self.position_embedding(position_ids % 128)
        
        # 인코더 출력의 평균 계산
        encoder_mean = encoder_outputs.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
        
        # 멀티헤드 어텐션으로 컨텍스트 관계 파악
        attended_context, _ = self.context_attention(
            query=decoder_hidden,
            key=encoder_outputs,
            value=encoder_outputs
        )
        
        # 내용어 탐지
        content_probability = self.content_detector(decoder_hidden)
        
        # 위치 및 내용 정보가 포함된 게이트 입력
        gate_input = torch.cat([
            decoder_hidden,
            encoder_mean,
            cross_attention_output,
            position_emb
        ], dim=-1)
        
        # 게이트 점수 계산
        source_gate_raw = self.source_gate(gate_input)
        target_gate_raw = self.target_gate(gate_input)
        
        # 내용어 편향 적용
        content_bias = CONFIG["content_bias_weight"] * content_probability
        source_gate_raw = source_gate_raw + content_bias
        target_gate_raw = target_gate_raw - content_bias
        
        # 적응형 시그모이드 적용
        source_gate_weight = torch.sigmoid(source_gate_raw - self.adaptive_threshold)
        target_gate_weight = torch.sigmoid(target_gate_raw + self.adaptive_threshold)
        
        # 온도 스케일링으로 게이트 정규화
        temperature = 2.0
        gate_sum = source_gate_weight + target_gate_weight + 1e-8
        source_gate_weight = (source_gate_weight / gate_sum) ** (1.0 / temperature)
        target_gate_weight = (target_gate_weight / gate_sum) ** (1.0 / temperature)
        
        # 온도 스케일링 후 재정규화
        gate_sum = source_gate_weight + target_gate_weight + 1e-8
        source_gate_weight = source_gate_weight / gate_sum
        target_gate_weight = target_gate_weight / gate_sum
        
        # 컨텍스트에 게이트 적용
        gated_source_context = source_gate_weight * attended_context
        gated_target_context = target_gate_weight * decoder_hidden
        
        # 컨텍스트 융합
        fused_context = torch.cat([gated_source_context, gated_target_context], dim=-1)
        fused_output = self.context_fusion(fused_context)
        
        # 다중 스케일 잔차 연결
        output = self.layer_norm(fused_output + decoder_hidden + 0.1 * cross_attention_output)
        output = self.dropout(output)
        
        return output, source_gate_weight, target_gate_weight, content_probability


class EnhancedT5WithDCG(T5ForConditionalGeneration):
    """향상된 DCG가 적용된 T5 모델"""
    
    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model
        
        # 전략적으로 선택된 디코더 레이어에 DCG 모듈 추가
        dcg_layers = [0, 2, 4] if config.num_decoder_layers >= 6 else [0, 2]
        
        self.dcg_modules = nn.ModuleDict({
            str(layer_idx): AdvancedDynamicContextGates(
                hidden_size=config.d_model,
                num_heads=4,
                dropout_rate=CONFIG["gate_dropout"]
            ) for layer_idx in dcg_layers
        })
        
        self.dcg_layer_indices = dcg_layers
        
        # 내용어 분석용 단어 부스트
        self.register_buffer('content_word_boost', torch.ones(config.vocab_size) * 0.1)
        
        # 가중치 초기화
        self.post_init()
        
    def compute_enhanced_gate_loss(self, gate_weights_source, gate_weights_target, content_probs, labels):
        """향상된 게이트 정규화 손실 계산"""
        total_loss = 0.0
        
        for src_gate, tgt_gate, content_prob in zip(gate_weights_source, gate_weights_target, content_probs):
            # 결정성 손실: 게이트가 명확한 결정을 내리도록 유도
            src_decisiveness = torch.mean(torch.abs(src_gate - 0.5))
            tgt_decisiveness = torch.mean(torch.abs(tgt_gate - 0.5))
            decisiveness_loss = -(src_decisiveness + tgt_decisiveness)
            
            # 내용 일관성 손실: 내용어는 소스 컨텍스트를 선호하도록
            if labels is not None:
                # 유효한 위치 식별 (패딩, 특수 토큰 제외)
                valid_positions = (labels != -100) & (labels != 0) & (labels != 1)
                if valid_positions.any():
                    content_consistency = torch.mean(
                        content_prob[valid_positions] * src_gate[valid_positions] +
                        (1 - content_prob[valid_positions]) * tgt_gate[valid_positions]
                    )
                    content_loss = -content_consistency
                else:
                    content_loss = 0.0
            else:
                content_loss = 0.0
            
            # 엔트로피 정규화: 게이트가 너무 균일하지 않도록
            src_entropy = -torch.mean(src_gate * torch.log(src_gate + 1e-8) + 
                                     (1 - src_gate) * torch.log(1 - src_gate + 1e-8))
            tgt_entropy = -torch.mean(tgt_gate * torch.log(tgt_gate + 1e-8) + 
                                     (1 - tgt_gate) * torch.log(1 - tgt_gate + 1e-8))
            entropy_loss = -(src_entropy + tgt_entropy)
            
            total_loss += decisiveness_loss + 0.5 * content_loss + 0.1 * entropy_loss
        
        return total_loss / len(gate_weights_source) if gate_weights_source else 0.0
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        **kwargs,
    ):
        """향상된 T5 순전파에 DCG 추가"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 인코더 실행
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        
        hidden_states = encoder_outputs[0]
        
        # 디코더 입력 준비
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(labels)
            
        # 디코더 실행
        decoder_kwargs = {
            "input_ids": decoder_input_ids,
            "attention_mask": decoder_attention_mask,
            "inputs_embeds": decoder_inputs_embeds,
            "past_key_values": past_key_values,
            "encoder_hidden_states": hidden_states,
            "encoder_attention_mask": attention_mask,
            "head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            "output_attentions": output_attentions,
            "output_hidden_states": True,
            "return_dict": return_dict,
        }
        
        # 캐시 위치 추가
        if cache_position is not None:
            decoder_kwargs["cache_position"] = cache_position
            
        decoder_outputs = self.decoder(**decoder_kwargs)
        
        sequence_output = decoder_outputs[0]
        
        # DCG 적용
        gate_weights_source = []
        gate_weights_target = []
        content_probabilities = []
        
        if decoder_outputs.hidden_states is not None:
            for layer_idx in self.dcg_layer_indices:
                if layer_idx < len(decoder_outputs.hidden_states) - 1:
                    dcg_module = self.dcg_modules[str(layer_idx)]
                    layer_hidden = decoder_outputs.hidden_states[layer_idx + 1]
                    
                    # 위치 정보 생성
                    seq_len = layer_hidden.size(1)
                    position_ids = torch.arange(seq_len, device=layer_hidden.device).unsqueeze(0).expand(layer_hidden.size(0), -1)
                    
                    enhanced_hidden, src_gate, tgt_gate, content_prob = dcg_module(
                        decoder_hidden=layer_hidden,
                        encoder_outputs=hidden_states,
                        cross_attention_output=sequence_output,
                        position_ids=position_ids
                    )
                    
                    gate_weights_source.append(src_gate)
                    gate_weights_target.append(tgt_gate)
                    content_probabilities.append(content_prob)
                    
                    # 마지막 레이어의 출력을 최종 시퀀스 출력으로 사용
                    if layer_idx == max(self.dcg_layer_indices):
                        sequence_output = enhanced_hidden
        
        # 언어 모델링 헤드
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim ** -0.5)
            
        lm_logits = self.lm_head(sequence_output)
        
        # 손실 계산
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            
            # 게이트 정규화 손실 추가
            if gate_weights_source and CONFIG["gate_regularization_weight"] > 0:
                gate_reg_loss = self.compute_enhanced_gate_loss(
                    gate_weights_source, gate_weights_target, content_probabilities, labels
                )
                loss += CONFIG["gate_regularization_weight"] * gate_reg_loss
        
        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output
        
        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


def setup_directories():
    """필요한 디렉토리 생성"""
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["data_dir"], exist_ok=True)


def load_data():
    """데이터 로드 및 캐싱"""
    cache_path = os.path.join(CONFIG["data_dir"], "processed_data.json")
    
    # 캐시된 데이터가 있으면 로드
    if os.path.exists(cache_path):
        print("캐시된 데이터를 로드합니다...")
        with open(cache_path, 'r') as f:
            data = json.load(f)
        return data["train"], data["val"], data["test"]
    
    # 없으면 Kaggle에서 다운로드
    print("Kaggle에서 데이터를 다운로드합니다...")
    try:
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "Cornell-University/arxiv",
            "arxiv-metadata-oai-snapshot.json",
            pandas_kwargs={'lines': True}
        )
    except Exception as e:
        print(f"데이터 다운로드 실패: {e}")
        return None, None, None
    
    # 데이터 전처리
    df = df[['title', 'abstract']].dropna()
    df = df[df['abstract'].str.len() > 100]  # 짧은 초록 제외
    df = df[df['title'].str.len() > 10]      # 짧은 제목 제외
    df = df[df['title'].str.len() < 150]     # 너무 긴 제목 제외
    df = df[df['abstract'].str.contains(r'[a-zA-Z]', regex=True)]  # 영어 텍스트만
    df = df[df['title'].str.contains(r'[a-zA-Z]', regex=True)]     # 영어 텍스트만
    
    # 데이터 섞기
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 데이터 크기 확인
    total_needed = CONFIG["train_size"] + CONFIG["val_size"] + CONFIG["test_size"]
    if len(df) < total_needed:
        print(f"데이터가 부족합니다. 필요: {total_needed}, 보유: {len(df)}")
        return None, None, None
    
    # 데이터 분할
    test_data = df.head(CONFIG["test_size"]).to_dict('records')
    val_data = df.iloc[CONFIG["test_size"]:CONFIG["test_size"] + CONFIG["val_size"]].to_dict('records')
    train_data = df.iloc[CONFIG["test_size"] + CONFIG["val_size"]:CONFIG["test_size"] + CONFIG["val_size"] + CONFIG["train_size"]].to_dict('records')
    
    # 캐시 저장
    cache_data = {
        "train": train_data,
        "val": val_data, 
        "test": test_data
    }
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f)
    
    print(f"데이터 준비 완료 - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    return train_data, val_data, test_data


def preprocess_data(data, tokenizer):
    """데이터 토큰화 전처리"""
    inputs = []
    targets = []
    
    for item in data:
        abstract = item["abstract"].strip()
        title = item["title"].strip()
        
        # 초록이 너무 길면 앞부분만 사용
        if len(abstract.split()) > 400:
            abstract = ' '.join(abstract.split()[:400])
        
        inputs.append(f"generate title: {abstract}")
        targets.append(title)
    
    # 입력 토큰화
    model_inputs = tokenizer(
        inputs,
        max_length=CONFIG["max_input_length"],
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    
    # 목표 토큰화
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=CONFIG["max_target_length"],
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
    
    return {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "labels": labels["input_ids"]
    }


class SimpleDataset(torch.utils.data.Dataset):
    """간단한 데이터셋 클래스"""
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}
    
    def __len__(self):
        return len(self.encodings["input_ids"])


def train_enhanced_dcg_model():
    """향상된 DCG 모델 훈련 함수"""
    setup_directories()
    
    # 데이터 로드
    train_data, val_data, test_data = load_data()
    if train_data is None:
        print("데이터 로드 실패")
        return
    
    # 토크나이저 로드
    print("토크나이저를 로드합니다...")
    tokenizer = T5Tokenizer.from_pretrained(CONFIG["model_name"])
    
    # DCG 모델 초기화
    print("DCG T5 모델을 초기화합니다...")
    config = T5Config.from_pretrained(CONFIG["model_name"])
    model = EnhancedT5WithDCG(config)
    
    # 기본 T5 가중치로 초기화
    base_model = T5ForConditionalGeneration.from_pretrained(CONFIG["model_name"])
    model.load_state_dict(base_model.state_dict(), strict=False)
    print("기본 T5 가중치로 초기화 완료")
    
    # 데이터 전처리
    print("데이터를 전처리합니다...")
    train_encodings = preprocess_data(train_data, tokenizer)
    val_encodings = preprocess_data(val_data, tokenizer)
    
    # 데이터셋 생성
    train_dataset = SimpleDataset(train_encodings)
    val_dataset = SimpleDataset(val_encodings)
    
    # 훈련 설정
    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=CONFIG["num_epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        warmup_ratio=CONFIG["warmup_ratio"],
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        fp16=True,
        report_to=None
    )
    
    # 데이터 콜레이터
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )
    
    # 트레이너 초기화
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    # 훈련 실행
    print("DCG T5 모델 훈련을 시작합니다...")
    try:
        train_result = trainer.train()
        
        # 모델 저장
        trainer.save_model()
        tokenizer.save_pretrained(CONFIG["output_dir"])
        
        # 훈련 결과 저장
        with open(os.path.join(CONFIG["output_dir"], "train_results.json"), "w") as f:
            json.dump({
                "train_loss": train_result.training_loss,
                "epochs": CONFIG["num_epochs"],
                "train_samples": len(train_data),
                "val_samples": len(val_data),
                "model_type": "T5_with_Enhanced_DCG",
                "gate_regularization_weight": CONFIG["gate_regularization_weight"],
                "content_bias_weight": CONFIG["content_bias_weight"]
            }, f, indent=2)
        
        # 최종 검증
        final_eval = trainer.evaluate()
        with open(os.path.join(CONFIG["output_dir"], "eval_results.json"), "w") as f:
            json.dump(final_eval, f, indent=2)
        
        print("DCG T5 모델 훈련 완료!")
        print(f"최종 validation loss: {final_eval['eval_loss']:.4f}")
        print(f"모델 저장 위치: {CONFIG['output_dir']}")
        
    except Exception as e:
        print(f"훈련 중 오류 발생: {e}")


if __name__ == "__main__":
    train_enhanced_dcg_model()