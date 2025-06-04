import os
import json
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import T5Tokenizer, T5ForConditionalGeneration
import evaluate
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Enhanced Configuration
CONFIG = {
    "model_dir": "./model",
    "data_dir": "./data", 
    "results_dir": "./results",
    "batch_size": 8,
    "generation_params": {
        "max_new_tokens": 50,
        "min_length": 5,
        "num_beams": 4,
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 2,
        "early_stopping": True
    }
}

class TitleGenerationEvaluator:
    """핵심 제목 생성 평가 시스템"""
    
    def __init__(self, model_name="Base T5"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_directories()
        
        # 핵심 메트릭 초기화
        self.rouge = evaluate.load("rouge")
        try:
            self.bertscore = evaluate.load("bertscore")
            self.bert_available = True
        except:
            print("⚠️  BERT Score not available. Will use ROUGE only.")
            self.bert_available = False
            
        try:
            self.bleu = evaluate.load("bleu")
            self.bleu_available = True
        except:
            print("⚠️  BLEU Score not available.")
            self.bleu_available = False
    
    def setup_directories(self):
        """디렉토리 설정"""
        os.makedirs(CONFIG["results_dir"], exist_ok=True)
        plt.style.use('default')
        sns.set_palette("husl")
    
    def load_model(self, model_path=None):
        """모델 로드"""
        model_path = model_path or CONFIG["model_dir"]
        print(f"🔄 Loading model from {model_path}...")
        
        try:
            # Main model directory 시도
            tokenizer = T5Tokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = T5ForConditionalGeneration.from_pretrained(model_path)
            print("✅ Model loaded successfully")
            
        except Exception as e:
            print(f"❌ Main directory failed: {e}")
            # Checkpoint 시도
            import glob
            checkpoint_dirs = glob.glob(os.path.join(model_path, "checkpoint-*"))
            if checkpoint_dirs:
                latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))
                print(f"🔄 Using checkpoint: {latest_checkpoint}")
                tokenizer = T5Tokenizer.from_pretrained(latest_checkpoint)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                model = T5ForConditionalGeneration.from_pretrained(latest_checkpoint)
                print("✅ Checkpoint loaded successfully")
            else:
                raise Exception("No valid model found")
        
        model.to(self.device)
        model.eval()
        print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model, tokenizer
    
    def load_test_data(self):
        """테스트 데이터 로드"""
        cache_path = os.path.join(CONFIG["data_dir"], "processed_data.json")
        if not os.path.exists(cache_path):
            raise Exception("Test data not found. Please run training first.")
        
        with open(cache_path, 'r') as f:
            data = json.load(f)
        
        test_data = data["test"]
        print(f"📋 Test data loaded: {len(test_data)} samples")
        return test_data
    
    def generate_titles(self, model, tokenizer, test_data):
        """제목 생성"""
        print("🎯 Generating titles...")
        
        abstracts = [item["abstract"] for item in test_data]
        references = [item["title"] for item in test_data]
        generated = []
        
        for i in tqdm(range(0, len(abstracts), CONFIG["batch_size"]), desc="Generating"):
            batch_abstracts = abstracts[i:i + CONFIG["batch_size"]]
            
            # 입력 준비
            inputs = [f"generate title: {abstract}" for abstract in batch_abstracts]
            tokenized = tokenizer(
                inputs,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # 생성
            with torch.no_grad():
                # Decoder start token 설정
                if not hasattr(model.config, 'decoder_start_token_id') or model.config.decoder_start_token_id is None:
                    model.config.decoder_start_token_id = tokenizer.pad_token_id
                
                outputs = model.generate(
                    input_ids=tokenized["input_ids"],
                    attention_mask=tokenized["attention_mask"],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    decoder_start_token_id=model.config.decoder_start_token_id,
                    use_cache=True,
                    **CONFIG["generation_params"]
                )
            
            # 디코딩
            for output in outputs:
                try:
                    decoded = tokenizer.decode(output, skip_special_tokens=True)
                    
                    # 프롬프트 제거
                    if decoded.startswith("generate title:"):
                        decoded = decoded[15:].strip()
                    
                    # 빈 결과 처리
                    if not decoded or len(decoded.strip()) < 3:
                        decoded = "Title Generation Failed"
                    
                    # 길이 제한
                    if len(decoded.split()) > 15:
                        decoded = " ".join(decoded.split()[:15])
                    
                    generated.append(decoded)
                    
                except Exception as e:
                    print(f"⚠️  Decoding error: {e}")
                    generated.append("Decoding Error")
        
        return generated, references
    
    def calculate_core_metrics(self, generated, references):
        """핵심 메트릭 계산"""
        print("📊 Calculating metrics...")
        
        metrics = {}
        
        # ROUGE 점수
        rouge_scores = self.rouge.compute(
            predictions=generated,
            references=references,
            rouge_types=['rouge1', 'rouge2', 'rougeL']
        )
        metrics['rouge'] = rouge_scores
        
        # BERT Score
        if self.bert_available:
            try:
                bert_scores = self.bertscore.compute(
                    predictions=generated,
                    references=references,
                    lang="en",
                    model_type="distilbert-base-uncased"
                )
                metrics['bert'] = {
                    'f1_mean': np.mean(bert_scores['f1']),
                    'f1_std': np.std(bert_scores['f1']),
                    'precision_mean': np.mean(bert_scores['precision']),
                    'recall_mean': np.mean(bert_scores['recall']),
                    'f1_scores': bert_scores['f1']
                }
            except Exception as e:
                print(f"⚠️  BERT Score failed: {e}")
                metrics['bert'] = None
        else:
            metrics['bert'] = None
        
        # BLEU Score
        if self.bleu_available:
            try:
                bleu_scores = self.bleu.compute(
                    predictions=generated,
                    references=[[ref] for ref in references]
                )
                metrics['bleu'] = bleu_scores['bleu']
            except Exception as e:
                print(f"⚠️  BLEU Score failed: {e}")
                metrics['bleu'] = None
        else:
            metrics['bleu'] = None
        
        # 길이 통계
        gen_lengths = [len(gen.split()) for gen in generated]
        ref_lengths = [len(ref.split()) for ref in references]
        
        metrics['length_stats'] = {
            'generated_mean': np.mean(gen_lengths),
            'generated_std': np.std(gen_lengths),
            'reference_mean': np.mean(ref_lengths),
            'reference_std': np.std(ref_lengths),
            'length_ratio': np.mean(gen_lengths) / np.mean(ref_lengths)
        }
        
        return metrics
    
    def create_evaluation_plots(self, metrics, generated, references):
        """핵심 시각화"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{self.model_name} - Evaluation Results', fontsize=16, fontweight='bold')
        
        # 1. ROUGE 점수 바 차트
        ax1 = axes[0, 0]
        rouge_names = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
        rouge_values = [metrics['rouge']['rouge1'], metrics['rouge']['rouge2'], metrics['rouge']['rougeL']]
        
        bars = ax1.bar(rouge_names, rouge_values, color=['skyblue', 'lightgreen', 'salmon'])
        ax1.set_title('ROUGE Scores')
        ax1.set_ylabel('Score')
        ax1.set_ylim(0, max(rouge_values) * 1.2)
        
        # 값 표시
        for bar, value in zip(bars, rouge_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. BERT Score (가능한 경우)
        ax2 = axes[0, 1]
        if metrics['bert']:
            bert_names = ['F1', 'Precision', 'Recall']
            bert_values = [
                metrics['bert']['f1_mean'],
                metrics['bert']['precision_mean'],
                metrics['bert']['recall_mean']
            ]
            
            bars = ax2.bar(bert_names, bert_values, color=['darkgreen', 'darkblue', 'darkred'])
            ax2.set_title('BERT Score')
            ax2.set_ylabel('Score')
            ax2.set_ylim(0, 1)
            
            for bar, value in zip(bars, bert_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'BERT Score\nNot Available', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12, 
                    bbox=dict(boxstyle='round', facecolor='lightgray'))
            ax2.set_title('BERT Score')
        
        # 3. 길이 분포 비교
        ax3 = axes[0, 2]
        gen_lengths = [len(gen.split()) for gen in generated]
        ref_lengths = [len(ref.split()) for ref in references]
        
        ax3.hist([gen_lengths, ref_lengths], bins=20, alpha=0.7, 
                label=['Generated', 'Reference'], color=['orange', 'blue'])
        ax3.set_title('Title Length Distribution')
        ax3.set_xlabel('Number of Words')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        
        # 4. BERT F1 분포 (가능한 경우)
        ax4 = axes[1, 0]
        if metrics['bert'] and 'f1_scores' in metrics['bert']:
            f1_scores = metrics['bert']['f1_scores']
            n, bins, patches = ax4.hist(f1_scores, bins=25, alpha=0.7, color='green', edgecolor='black')
            
            mean_f1 = metrics['bert']['f1_mean']
            ax4.axvline(mean_f1, color='red', linestyle='-', linewidth=2, 
                       label=f'Mean: {mean_f1:.3f}')
            ax4.set_title('BERT F1 Distribution')
            ax4.set_xlabel('BERT F1 Score')
            ax4.set_ylabel('Frequency')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'BERT F1\nDistribution\nNot Available', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='lightgray'))
            ax4.set_title('BERT F1 Distribution')
        
        # 5. 성과 요약
        ax5 = axes[1, 1]
        ax5.axis('off')
        
        summary_text = f"""
        📊 Performance Summary
        
        🎯 Core Metrics:
        • ROUGE-1: {metrics['rouge']['rouge1']:.3f}
        • ROUGE-2: {metrics['rouge']['rouge2']:.3f}
        • ROUGE-L: {metrics['rouge']['rougeL']:.3f}
        """
        
        if metrics['bert']:
            summary_text += f"""
        • BERT F1: {metrics['bert']['f1_mean']:.3f} ± {metrics['bert']['f1_std']:.3f}
        """
        
        if metrics['bleu']:
            summary_text += f"""
        • BLEU: {metrics['bleu']:.3f}
        """
        
        summary_text += f"""
        
        📏 Length Analysis:
        • Avg Generated: {metrics['length_stats']['generated_mean']:.1f} words
        • Avg Reference: {metrics['length_stats']['reference_mean']:.1f} words
        • Length Ratio: {metrics['length_stats']['length_ratio']:.2f}
        """
        
        ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 6. 샘플 예시
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # 최고 성능 샘플 찾기 (BERT 가능한 경우)
        if metrics['bert'] and 'f1_scores' in metrics['bert']:
            best_idx = np.argmax(metrics['bert']['f1_scores'])
            worst_idx = np.argmin(metrics['bert']['f1_scores'])
            sample_text = f"""
            🥇 Best Example (BERT F1: {metrics['bert']['f1_scores'][best_idx]:.3f})
            
            Ref: {references[best_idx][:60]}...
            Gen: {generated[best_idx][:60]}...
            
            🔴 Worst Example (BERT F1: {metrics['bert']['f1_scores'][worst_idx]:.3f})
            
            Ref: {references[worst_idx][:60]}...
            Gen: {generated[worst_idx][:60]}...
            """
        else:
            # BERT 없는 경우 랜덤 샘플
            idx1, idx2 = np.random.choice(len(generated), 2, replace=False)
            sample_text = f"""
            📝 Sample Examples
            
            Example 1:
            Ref: {references[idx1][:60]}...
            Gen: {generated[idx1][:60]}...
            
            Example 2:
            Ref: {references[idx2][:60]}...
            Gen: {generated[idx2][:60]}...
            """
        
        ax6.text(0.05, 0.95, sample_text, transform=ax6.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        # 저장
        plot_path = os.path.join(CONFIG["results_dir"], f"{self.model_name.lower().replace(' ', '_')}_evaluation.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return plot_path
    
    def print_results(self, metrics, generated, references):
        """결과 출력"""
        print("\n" + "="*80)
        print(f"📊 {self.model_name.upper()} - EVALUATION RESULTS")
        print("="*80)
        
        # 핵심 메트릭
        print(f"\n🎯 Core Performance:")
        print(f"  ROUGE-1:  {metrics['rouge']['rouge1']:.4f}")
        print(f"  ROUGE-2:  {metrics['rouge']['rouge2']:.4f}")
        print(f"  ROUGE-L:  {metrics['rouge']['rougeL']:.4f}")
        
        if metrics['bert']:
            print(f"\n🤖 BERT Score:")
            print(f"  F1:        {metrics['bert']['f1_mean']:.4f} ± {metrics['bert']['f1_std']:.4f}")
            print(f"  Precision: {metrics['bert']['precision_mean']:.4f}")
            print(f"  Recall:    {metrics['bert']['recall_mean']:.4f}")
        
        if metrics['bleu']:
            print(f"\n📝 BLEU Score: {metrics['bleu']:.4f}")
        
        # 길이 통계
        length_stats = metrics['length_stats']
        print(f"\n📏 Length Statistics:")
        print(f"  Generated: {length_stats['generated_mean']:.1f} ± {length_stats['generated_std']:.1f} words")
        print(f"  Reference: {length_stats['reference_mean']:.1f} ± {length_stats['reference_std']:.1f} words")
        print(f"  Ratio:     {length_stats['length_ratio']:.2f}")
        
        # 성능 분석
        if metrics['bert'] and 'f1_scores' in metrics['bert']:
            f1_scores = metrics['bert']['f1_scores']
            high_perf = sum(1 for score in f1_scores if score > 0.85) / len(f1_scores)
            med_perf = sum(1 for score in f1_scores if 0.75 <= score <= 0.85) / len(f1_scores)
            low_perf = sum(1 for score in f1_scores if score < 0.75) / len(f1_scores)
            
            print(f"\n📈 Performance Distribution (BERT F1):")
            print(f"  High (>0.85):     {high_perf:.3f}")
            print(f"  Medium (0.75-0.85): {med_perf:.3f}")
            print(f"  Low (<0.75):      {low_perf:.3f}")
        
        # 샘플 출력
        print(f"\n📝 Sample Results:")
        indices = np.random.choice(len(generated), min(3, len(generated)), replace=False)
        for i, idx in enumerate(indices, 1):
            print(f"\n  Example {i}:")
            print(f"    Reference: {references[idx]}")
            print(f"    Generated: {generated[idx]}")
    
    def save_results(self, metrics, generated, references):
        """결과 저장"""
        # 요약 저장
        summary = {
            'model_name': self.model_name,
            'total_samples': len(generated),
            'rouge1': metrics['rouge']['rouge1'],
            'rouge2': metrics['rouge']['rouge2'],
            'rougeL': metrics['rouge']['rougeL'],
            'length_stats': metrics['length_stats']
        }
        
        if metrics['bert']:
            summary.update({
                'bert_f1_mean': metrics['bert']['f1_mean'],
                'bert_f1_std': metrics['bert']['f1_std'],
                'bert_precision_mean': metrics['bert']['precision_mean'],
                'bert_recall_mean': metrics['bert']['recall_mean']
            })
        
        if metrics['bleu']:
            summary['bleu'] = metrics['bleu']
        
        # JSON 저장
        summary_path = os.path.join(CONFIG["results_dir"], f"{self.model_name.lower().replace(' ', '_')}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # 상세 결과 저장
        detailed_results = []
        for i, (gen, ref) in enumerate(zip(generated, references)):
            result = {
                'index': i,
                'reference': ref,
                'generated': gen,
                'ref_length': len(ref.split()),
                'gen_length': len(gen.split())
            }
            
            if metrics['bert'] and 'f1_scores' in metrics['bert']:
                result.update({
                    'bert_f1': metrics['bert']['f1_scores'][i]
                })
            
            detailed_results.append(result)
        
        df = pd.DataFrame(detailed_results)
        csv_path = os.path.join(CONFIG["results_dir"], f"{self.model_name.lower().replace(' ', '_')}_detailed.csv")
        df.to_csv(csv_path, index=False)
        
        print(f"\n✅ Results saved:")
        print(f"   Summary: {summary_path}")
        print(f"   Detailed: {csv_path}")
        
        return summary_path, csv_path
    
    def evaluate(self, model_path=None):
        """전체 평가 수행"""
        try:
            # 데이터 로드
            test_data = self.load_test_data()
            
            # 모델 로드
            model, tokenizer = self.load_model(model_path)
            
            # 제목 생성
            generated, references = self.generate_titles(model, tokenizer, test_data)
            
            # 메트릭 계산
            metrics = self.calculate_core_metrics(generated, references)
            
            # 결과 출력
            self.print_results(metrics, generated, references)
            
            # 시각화
            plot_path = self.create_evaluation_plots(metrics, generated, references)
            
            # 결과 저장
            summary_path, csv_path = self.save_results(metrics, generated, references)
            
            return {
                'metrics': metrics,
                'generated': generated,
                'references': references,
                'paths': {
                    'plot': plot_path,
                    'summary': summary_path,
                    'detailed': csv_path
                }
            }
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            raise e

def evaluate_base_model():
    """Base T5 모델 평가"""
    evaluator = TitleGenerationEvaluator("Base T5")
    return evaluator.evaluate()

if __name__ == "__main__":
    results = evaluate_base_model()