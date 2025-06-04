import os
import json
import pandas as pd
import torch
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import kagglehub
from kagglehub import KaggleDatasetAdapter

# 설정 (DCG 모델과 동일한 조건으로 설정)
CONFIG = {
    "model_name": "t5-small",
    "output_dir": "./model",
    "data_dir": "./data",
    "batch_size": 8,
    "learning_rate": 3e-5,  # DCG 모델과 동일하게 낮춤
    "num_epochs": 5,
    "max_input_length": 512,
    "max_target_length": 128,
    "train_size": 9000,  # DCG 모델과 동일한 데이터 크기
    "val_size": 1000,
    "test_size": 1000,
    "warmup_ratio": 0.1,  # DCG 모델과 동일
    "gradient_accumulation_steps": 2  # DCG 모델과 동일
}

def setup_directories():
    """디렉토리 생성"""
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["data_dir"], exist_ok=True)

def load_data():
    """데이터 로드 및 캐싱"""
    cache_path = os.path.join(CONFIG["data_dir"], "processed_data.json")
    
    if os.path.exists(cache_path):
        print("캐시된 데이터를 로드합니다...")
        with open(cache_path, 'r') as f:
            data = json.load(f)
        return data["train"], data["val"], data["test"]
    
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
    
    # 데이터 정리 및 품질 향상 (DCG 모델과 동일한 필터링)
    df = df[['title', 'abstract']].dropna()
    df = df[df['abstract'].str.len() > 100]  # 더 긴 초록만 사용
    df = df[df['title'].str.len() > 10]  # 더 긴 제목만 사용
    df = df[df['title'].str.len() < 150]  # 너무 긴 제목 제외
    
    # 영어 텍스트만 필터링 (간단한 휴리스틱)
    df = df[df['abstract'].str.contains(r'[a-zA-Z]', regex=True)]
    df = df[df['title'].str.contains(r'[a-zA-Z]', regex=True)]
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 충분한 데이터가 있는지 확인
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
    
    print(f"고품질 데이터 준비 완료 - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    return train_data, val_data, test_data

def preprocess_data(data, tokenizer):
    """향상된 데이터 전처리 (DCG 모델과 동일)"""
    inputs = []
    targets = []
    
    for item in data:
        # 더 구체적인 프롬프트 사용 (DCG 모델과 동일)
        abstract = item["abstract"].strip()
        title = item["title"].strip()
        
        # 초록이 너무 길면 앞부분만 사용
        if len(abstract.split()) > 400:
            abstract = ' '.join(abstract.split()[:400])
        
        inputs.append(f"generate title: {abstract}")
        targets.append(title)
    
    # 토큰화
    model_inputs = tokenizer(
        inputs,
        max_length=CONFIG["max_input_length"],
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    
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

def train_model():
    """모델 훈련"""
    setup_directories()
    
    # 데이터 로드
    train_data, val_data, test_data = load_data()
    if train_data is None:
        print("데이터 로드 실패")
        return
    
    # 모델과 토크나이저 로드
    print("모델을 로드합니다...")
    tokenizer = T5Tokenizer.from_pretrained(CONFIG["model_name"])
    model = T5ForConditionalGeneration.from_pretrained(CONFIG["model_name"])
    
    # 데이터 전처리
    print("데이터를 전처리합니다...")
    train_encodings = preprocess_data(train_data, tokenizer)
    val_encodings = preprocess_data(val_data, tokenizer)
    
    # 데이터셋 생성
    train_dataset = SimpleDataset(train_encodings)
    val_dataset = SimpleDataset(val_encodings)
    
    # 훈련 설정 (DCG 모델과 동일한 조건)
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
        fp16=True,  # Mixed precision for efficiency
        report_to=None
    )
    
    # 데이터 콜레이터
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )
    
    # 트레이너 (실무용 - compute_metrics 없음)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    # 훈련 시작
    print("훈련을 시작합니다...")
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
                "val_samples": len(val_data)
            }, f, indent=2)
        
        # 최종 검증
        final_eval = trainer.evaluate()
        with open(os.path.join(CONFIG["output_dir"], "eval_results.json"), "w") as f:
            json.dump(final_eval, f, indent=2)
        
        print("훈련 완료!")
        print(f"최종 validation loss: {final_eval['eval_loss']:.4f}")
        print(f"모델 저장 위치: {CONFIG['output_dir']}")
        
    except Exception as e:
        print(f"훈련 중 오류 발생: {e}")

if __name__ == "__main__":
    train_model()