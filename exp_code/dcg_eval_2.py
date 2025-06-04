import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    T5Config,
    T5Model
)
from transformers.modeling_outputs import Seq2SeqLMOutput
from safetensors.torch import load_file
import evaluate
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Enhanced DCG Model Classes (from training)
class AdvancedDynamicContextGates(nn.Module):
    """Enhanced Dynamic Context Gates - same as training"""
    def __init__(self, hidden_size, num_heads=4, dropout_rate=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Multi-head attention for context understanding
        self.context_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Content word detection module
        self.content_detector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Position-aware gating
        self.position_embedding = nn.Embedding(128, hidden_size)
        
        # Source context gate with content bias
        self.source_gate = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Target context gate
        self.target_gate = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Adaptive threshold learning
        self.adaptive_threshold = nn.Parameter(torch.tensor(0.5))
        
        # Context fusion with residual connections
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, decoder_hidden, encoder_outputs, cross_attention_output, position_ids=None):
        """Enhanced forward pass with gate analysis"""
        batch_size, seq_len, hidden_size = decoder_hidden.shape
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=decoder_hidden.device).unsqueeze(0).expand(batch_size, -1)
        
        position_emb = self.position_embedding(position_ids % 128)
        encoder_mean = encoder_outputs.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
        
        # Multi-head attention
        attended_context, attention_weights = self.context_attention(
            query=decoder_hidden,
            key=encoder_outputs,
            value=encoder_outputs
        )
        
        # Content word detection
        content_probability = self.content_detector(decoder_hidden)
        
        # Gate input
        gate_input = torch.cat([
            decoder_hidden,
            encoder_mean,
            cross_attention_output,
            position_emb
        ], dim=-1)
        
        # Gate computation with content bias
        source_gate_raw = self.source_gate(gate_input)
        target_gate_raw = self.target_gate(gate_input)
        
        # Content bias application
        content_bias_weight = 0.15  # From CONFIG
        content_bias = content_bias_weight * content_probability
        source_gate_raw = source_gate_raw + content_bias
        target_gate_raw = target_gate_raw - content_bias
        
        # Adaptive sigmoid
        source_gate_weight = torch.sigmoid(source_gate_raw - self.adaptive_threshold)
        target_gate_weight = torch.sigmoid(target_gate_raw + self.adaptive_threshold)
        
        # Normalize gates
        temperature = 2.0
        gate_sum = source_gate_weight + target_gate_weight + 1e-8
        source_gate_weight = (source_gate_weight / gate_sum) ** (1.0 / temperature)
        target_gate_weight = (target_gate_weight / gate_sum) ** (1.0 / temperature)
        
        gate_sum = source_gate_weight + target_gate_weight + 1e-8
        source_gate_weight = source_gate_weight / gate_sum
        target_gate_weight = target_gate_weight / gate_sum
        
        # Apply gates
        gated_source_context = source_gate_weight * attended_context
        gated_target_context = target_gate_weight * decoder_hidden
        
        # Context fusion
        fused_context = torch.cat([gated_source_context, gated_target_context], dim=-1)
        fused_output = self.context_fusion(fused_context)
        
        output = self.layer_norm(fused_output + decoder_hidden + 0.1 * cross_attention_output)
        output = self.dropout(output)
        
        return output, source_gate_weight, target_gate_weight, content_probability, attention_weights

class EnhancedT5WithDCG(T5ForConditionalGeneration):
    """Enhanced T5 Model with Advanced DCG - same as training"""
    
    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model
        
        # DCG modules for strategic layers
        dcg_layers = [0, 2, 4] if config.num_decoder_layers >= 6 else [0, 2]
        
        self.dcg_modules = nn.ModuleDict({
            str(layer_idx): AdvancedDynamicContextGates(
                hidden_size=config.d_model,
                num_heads=4,
                dropout_rate=0.2
            ) for layer_idx in dcg_layers
        })
        
        self.dcg_layer_indices = dcg_layers
        self.register_buffer('content_word_boost', torch.ones(config.vocab_size) * 0.1)
        self.post_init()

# Configuration for DCG evaluation
DCG_CONFIG = {
    "model_dir": "./enhanced_dcg_model",
    "data_dir": "./data", 
    "results_dir": "./results",
    "batch_size": 4,  # Smaller batch for detailed analysis
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

class DCGEvaluator:
    """Enhanced DCG 모델 평가 시스템"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_directories()
        
        # 메트릭 초기화
        self.rouge = evaluate.load("rouge")
        try:
            self.bertscore = evaluate.load("bertscore")
            self.bert_available = True
        except:
            print("⚠️  BERT Score not available.")
            self.bert_available = False
            
        try:
            self.bleu = evaluate.load("bleu")
            self.bleu_available = True
        except:
            self.bleu_available = False
    
    def setup_directories(self):
        os.makedirs(DCG_CONFIG["results_dir"], exist_ok=True)
        plt.style.use('default')
        sns.set_palette("husl")
    
    def load_dcg_model(self):
        """DCG 모델 로드"""
        print(f"🔄 Loading Enhanced DCG model from {DCG_CONFIG['model_dir']}...")
        
        try:
            # 토크나이저 로드
            tokenizer = T5Tokenizer.from_pretrained(DCG_CONFIG["model_dir"])
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Config 로드
            config = T5Config.from_pretrained(DCG_CONFIG["model_dir"])
            
            # DCG 모델 생성 및 가중치 로드
            model = EnhancedT5WithDCG(config)
            # Try to load from safetensors first, then fallback to pytorch_model.bin
            safetensors_path = os.path.join(DCG_CONFIG["model_dir"], "model.safetensors")
            pytorch_path = os.path.join(DCG_CONFIG["model_dir"], "pytorch_model.bin")
            
            if os.path.exists(safetensors_path):
                model.load_state_dict(load_file(safetensors_path, device=str(self.device)), strict=False)
            elif os.path.exists(pytorch_path):
                model.load_state_dict(torch.load(pytorch_path, map_location=self.device), strict=False)
            else:
                raise FileNotFoundError(f"Neither {safetensors_path} nor {pytorch_path} found")
            
            print("✅ Enhanced DCG model loaded successfully")
            
        except Exception as e:
            print(f"❌ Main directory failed: {e}")
            # 체크포인트 시도
            import glob
            checkpoint_dirs = glob.glob(os.path.join(DCG_CONFIG["model_dir"], "checkpoint-*"))
            if checkpoint_dirs:
                latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))
                print(f"🔄 Using checkpoint: {latest_checkpoint}")
                
                tokenizer = T5Tokenizer.from_pretrained(latest_checkpoint)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                config = T5Config.from_pretrained(latest_checkpoint)
                model = EnhancedT5WithDCG(config)
                # Try to load from safetensors first, then fallback to pytorch_model.bin
                safetensors_path = os.path.join(latest_checkpoint, "model.safetensors")
                pytorch_path = os.path.join(latest_checkpoint, "pytorch_model.bin")
                
                if os.path.exists(safetensors_path):
                    model.load_state_dict(load_file(safetensors_path, device=str(self.device)), strict=False)
                elif os.path.exists(pytorch_path):
                    model.load_state_dict(torch.load(pytorch_path, map_location=self.device), strict=False)
                else:
                    raise FileNotFoundError(f"Neither {safetensors_path} nor {pytorch_path} found in {latest_checkpoint}")
                print("✅ DCG checkpoint loaded successfully")
            else:
                raise Exception("No valid DCG model found")
        
        model.to(self.device)
        model.eval()
        print(f"📊 DCG Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model, tokenizer
    
    def load_test_data(self):
        """테스트 데이터 로드"""
        cache_path = os.path.join(DCG_CONFIG["data_dir"], "processed_data.json")
        if not os.path.exists(cache_path):
            raise Exception("Test data not found. Please run training first.")
        
        with open(cache_path, 'r') as f:
            data = json.load(f)
        
        test_data = data["test"]
        print(f"📋 Test data loaded: {len(test_data)} samples")
        return test_data
    
    def generate_with_gate_analysis(self, model, tokenizer, test_data):
        """DCG 분석을 포함한 제목 생성"""
        print("🎯 Generating titles with DCG analysis...")
        
        abstracts = [item["abstract"] for item in test_data]
        references = [item["title"] for item in test_data]
        generated = []
        gate_analysis = {
            'source_gate_weights': [],
            'target_gate_weights': [],
            'content_probabilities': [],
            'attention_weights': [],
            'adaptive_thresholds': []
        }
        
        for i in tqdm(range(0, len(abstracts), DCG_CONFIG["batch_size"]), desc="DCG Generation"):
            batch_abstracts = abstracts[i:i + DCG_CONFIG["batch_size"]]
            
            # 입력 준비
            inputs = [f"generate title: {abstract}" for abstract in batch_abstracts]
            tokenized = tokenizer(
                inputs,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                # DCG 분석을 위한 특별한 forward pass
                outputs_with_analysis = self._generate_with_dcg_analysis(
                    model, tokenized, tokenizer
                )
                
                generated.extend(outputs_with_analysis['generated'])
                
                # Gate 분석 데이터 수집
                if outputs_with_analysis['gate_data']:
                    for key in gate_analysis.keys():
                        if key in outputs_with_analysis['gate_data']:
                            gate_analysis[key].extend(outputs_with_analysis['gate_data'][key])
        
        return generated, references, gate_analysis
    
    def _generate_with_dcg_analysis(self, model, tokenized, tokenizer):
        """DCG 분석과 함께 생성"""
        # Decoder start token 설정
        if not hasattr(model.config, 'decoder_start_token_id') or model.config.decoder_start_token_id is None:
            model.config.decoder_start_token_id = tokenizer.pad_token_id
        
        # 생성
        outputs = model.generate(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            decoder_start_token_id=model.config.decoder_start_token_id,
            use_cache=True,
            **DCG_CONFIG["generation_params"]
        )
        
        # 디코딩
        generated_texts = []
        for output in outputs:
            try:
                decoded = tokenizer.decode(output, skip_special_tokens=True)
                if decoded.startswith("generate title:"):
                    decoded = decoded[15:].strip()
                
                if not decoded or len(decoded.strip()) < 3:
                    decoded = "DCG Generation Failed"
                
                if len(decoded.split()) > 15:
                    decoded = " ".join(decoded.split()[:15])
                
                generated_texts.append(decoded)
            except:
                generated_texts.append("DCG Decoding Error")
        
        # Gate 분석을 위한 추가 forward pass (첫 번째 샘플만)
        gate_data = {}
        try:
            if len(tokenized["input_ids"]) > 0:
                # 단일 샘플로 forward pass 수행하여 gate 정보 추출
                single_input = {
                    'input_ids': tokenized["input_ids"][:1],
                    'attention_mask': tokenized["attention_mask"][:1]
                }
                
                # Simple forward pass for gate analysis
                with torch.no_grad():
                    # Create dummy decoder input for analysis
                    decoder_input_ids = torch.full(
                        (1, 10), 
                        model.config.decoder_start_token_id, 
                        device=self.device
                    )
                    
                    # Forward pass through the model
                    encoder_outputs = model.encoder(
                        input_ids=single_input['input_ids'],
                        attention_mask=single_input['attention_mask']
                    )
                    
                    # Sample gate weights from DCG modules
                    if hasattr(model, 'dcg_modules') and model.dcg_modules:
                        # 더미 데이터로 gate 활성화 패턴 추출
                        dummy_hidden = torch.randn(1, 10, model.config.d_model, device=self.device)
                        
                        for layer_idx, dcg_module in model.dcg_modules.items():
                            _, src_gate, tgt_gate, content_prob, attn_weights = dcg_module(
                                decoder_hidden=dummy_hidden,
                                encoder_outputs=encoder_outputs.last_hidden_state,
                                cross_attention_output=dummy_hidden
                            )
                            
                            gate_data.setdefault('source_gate_weights', []).append(src_gate.mean().item())
                            gate_data.setdefault('target_gate_weights', []).append(tgt_gate.mean().item())
                            gate_data.setdefault('content_probabilities', []).append(content_prob.mean().item())
                            
                            if hasattr(dcg_module, 'adaptive_threshold'):
                                gate_data.setdefault('adaptive_thresholds', []).append(
                                    dcg_module.adaptive_threshold.item()
                                )
                            
                            break  # 첫 번째 레이어만 분석
                            
        except Exception as e:
            print(f"⚠️  Gate analysis failed: {e}")
            gate_data = {}
        
        return {
            'generated': generated_texts,
            'gate_data': gate_data
        }
    
    def calculate_dcg_metrics(self, generated, references, gate_analysis):
        """DCG 특화 메트릭 계산"""
        print("📊 Calculating DCG-enhanced metrics...")
        
        # 기본 메트릭
        metrics = {}
        
        # ROUGE
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
            except:
                metrics['bert'] = None
        else:
            metrics['bert'] = None
        
        # BLEU
        if self.bleu_available:
            try:
                bleu_scores = self.bleu.compute(
                    predictions=generated,
                    references=[[ref] for ref in references]
                )
                metrics['bleu'] = bleu_scores['bleu']
            except:
                metrics['bleu'] = None
        else:
            metrics['bleu'] = None
        
        # DCG 특화 메트릭
        dcg_metrics = {}
        
        if gate_analysis['source_gate_weights']:
            src_weights = gate_analysis['source_gate_weights']
            tgt_weights = gate_analysis['target_gate_weights']
            content_probs = gate_analysis['content_probabilities']
            
            dcg_metrics = {
                'source_gate_mean': np.mean(src_weights),
                'source_gate_std': np.std(src_weights),
                'target_gate_mean': np.mean(tgt_weights),
                'target_gate_std': np.std(tgt_weights),
                'content_prob_mean': np.mean(content_probs),
                'content_prob_std': np.std(content_probs),
                'gate_balance': np.mean(src_weights) / (np.mean(src_weights) + np.mean(tgt_weights)),
                'gate_decisiveness': np.mean([abs(s - t) for s, t in zip(src_weights, tgt_weights)])
            }
            
            if gate_analysis['adaptive_thresholds']:
                dcg_metrics['adaptive_threshold_mean'] = np.mean(gate_analysis['adaptive_thresholds'])
        
        metrics['dcg'] = dcg_metrics
        
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
    
    def create_dcg_analysis_plots(self, metrics, generated, references, gate_analysis):
        """DCG 특화 시각화"""
        fig, axes = plt.subplots(3, 3, figsize=(21, 18))
        fig.suptitle('Enhanced DCG Model - Comprehensive Analysis', fontsize=16, fontweight='bold')
        
        # 1. ROUGE Scores
        ax1 = axes[0, 0]
        rouge_names = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
        rouge_values = [metrics['rouge']['rouge1'], metrics['rouge']['rouge2'], metrics['rouge']['rougeL']]
        
        bars = ax1.bar(rouge_names, rouge_values, color=['skyblue', 'lightgreen', 'salmon'])
        ax1.set_title('ROUGE Scores')
        ax1.set_ylabel('Score')
        ax1.set_ylim(0, max(rouge_values) * 1.2)
        
        for bar, value in zip(bars, rouge_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. BERT Score
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
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('BERT Score')
        
        # 3. DCG Gate Analysis
        ax3 = axes[0, 2]
        if metrics['dcg'] and 'source_gate_mean' in metrics['dcg']:
            gate_names = ['Source Gate', 'Target Gate']
            gate_values = [
                metrics['dcg']['source_gate_mean'],
                metrics['dcg']['target_gate_mean']
            ]
            gate_stds = [
                metrics['dcg']['source_gate_std'],
                metrics['dcg']['target_gate_std']
            ]
            
            bars = ax3.bar(gate_names, gate_values, yerr=gate_stds, capsize=5,
                          color=['orange', 'purple'], alpha=0.8)
            ax3.set_title('DCG Gate Weights')
            ax3.set_ylabel('Average Weight')
            ax3.set_ylim(0, 1)
            
            for bar, value in zip(bars, gate_values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'DCG Gate\nAnalysis\nNot Available', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('DCG Gate Weights')
        
        # 4. Gate Distribution
        ax4 = axes[1, 0]
        if gate_analysis['source_gate_weights']:
            src_weights = gate_analysis['source_gate_weights']
            tgt_weights = gate_analysis['target_gate_weights']
            
            ax4.hist([src_weights, tgt_weights], bins=20, alpha=0.7,
                    label=['Source Gates', 'Target Gates'], color=['orange', 'purple'])
            ax4.set_title('Gate Weight Distributions')
            ax4.set_xlabel('Gate Weight')
            ax4.set_ylabel('Frequency')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'Gate Distribution\nNot Available', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Gate Weight Distributions')
        
        # 5. Content Probability Analysis
        ax5 = axes[1, 1]
        if gate_analysis['content_probabilities']:
            content_probs = gate_analysis['content_probabilities']
            
            n, bins, patches = ax5.hist(content_probs, bins=20, alpha=0.7, color='gold', edgecolor='black')
            mean_content = np.mean(content_probs)
            ax5.axvline(mean_content, color='red', linestyle='-', linewidth=2,
                       label=f'Mean: {mean_content:.3f}')
            ax5.set_title('Content Word Detection')
            ax5.set_xlabel('Content Probability')
            ax5.set_ylabel('Frequency')
            ax5.legend()
        else:
            ax5.text(0.5, 0.5, 'Content Probability\nNot Available', ha='center', va='center',
                    transform=ax5.transAxes, fontsize=12)
            ax5.set_title('Content Word Detection')
        
        # 6. Length Analysis
        ax6 = axes[1, 2]
        gen_lengths = [len(gen.split()) for gen in generated]
        ref_lengths = [len(ref.split()) for ref in references]
        
        ax6.hist([gen_lengths, ref_lengths], bins=20, alpha=0.7,
                label=['Generated', 'Reference'], color=['lightblue', 'lightcoral'])
        ax6.set_title('Title Length Distribution')
        ax6.set_xlabel('Number of Words')
        ax6.set_ylabel('Frequency')
        ax6.legend()
        
        # 7. DCG Performance Summary
        ax7 = axes[2, 0]
        ax7.axis('off')
        
        summary_text = f"""
        📊 DCG Performance Summary
        
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
        
        if metrics['dcg'] and 'gate_balance' in metrics['dcg']:
            summary_text += f"""
        
        🔧 DCG Mechanics:
        • Gate Balance: {metrics['dcg']['gate_balance']:.3f}
        • Gate Decisiveness: {metrics['dcg']['gate_decisiveness']:.3f}
        • Content Detection: {metrics['dcg']['content_prob_mean']:.3f}
        """
        
        ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # 8. Best/Worst Examples
        ax8 = axes[2, 1]
        ax8.axis('off')
        
        if metrics['bert'] and 'f1_scores' in metrics['bert']:
            best_idx = np.argmax(metrics['bert']['f1_scores'])
            worst_idx = np.argmin(metrics['bert']['f1_scores'])
            
            example_text = f"""
            🥇 Best Example (BERT F1: {metrics['bert']['f1_scores'][best_idx]:.3f})
            
            Ref: {references[best_idx][:50]}...
            Gen: {generated[best_idx][:50]}...
            
            🔴 Worst Example (BERT F1: {metrics['bert']['f1_scores'][worst_idx]:.3f})
            
            Ref: {references[worst_idx][:50]}...
            Gen: {generated[worst_idx][:50]}...
            """
        else:
            idx1, idx2 = np.random.choice(len(generated), 2, replace=False)
            example_text = f"""
            📝 Sample Examples
            
            Example 1:
            Ref: {references[idx1][:50]}...
            Gen: {generated[idx1][:50]}...
            
            Example 2:
            Ref: {references[idx2][:50]}...
            Gen: {generated[idx2][:50]}...
            """
        
        ax8.text(0.05, 0.95, example_text, transform=ax8.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # 9. DCG Advantage Analysis
        ax9 = axes[2, 2]
        if metrics['dcg'] and 'gate_balance' in metrics['dcg']:
            # Gate balance pie chart
            gate_balance = metrics['dcg']['gate_balance']
            sizes = [gate_balance, 1 - gate_balance]
            labels = ['Source Context', 'Target Context']
            colors = ['orange', 'purple']
            
            wedges, texts, autotexts = ax9.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                              startangle=90)
            ax9.set_title('DCG Gate Usage Balance')
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            ax9.text(0.5, 0.5, 'DCG Gate\nBalance\nNot Available', ha='center', va='center',
                    transform=ax9.transAxes, fontsize=12)
            ax9.set_title('DCG Gate Usage Balance')
        
        plt.tight_layout()
        
        # 저장
        plot_path = os.path.join(DCG_CONFIG["results_dir"], "enhanced_dcg_evaluation.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return plot_path
    
    def print_dcg_results(self, metrics, generated, references):
        """DCG 결과 출력"""
        print("\n" + "="*80)
        print("📊 ENHANCED DCG MODEL - EVALUATION RESULTS")
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
        
        # DCG 특화 메트릭
        if metrics['dcg'] and 'gate_balance' in metrics['dcg']:
            print(f"\n🔧 DCG Analysis:")
            print(f"  Source Gate Weight:   {metrics['dcg']['source_gate_mean']:.4f} ± {metrics['dcg']['source_gate_std']:.4f}")
            print(f"  Target Gate Weight:   {metrics['dcg']['target_gate_mean']:.4f} ± {metrics['dcg']['target_gate_std']:.4f}")
            print(f"  Gate Balance:         {metrics['dcg']['gate_balance']:.4f}")
            print(f"  Gate Decisiveness:    {metrics['dcg']['gate_decisiveness']:.4f}")
            print(f"  Content Detection:    {metrics['dcg']['content_prob_mean']:.4f} ± {metrics['dcg']['content_prob_std']:.4f}")
            
            if 'adaptive_threshold_mean' in metrics['dcg']:
                print(f"  Adaptive Threshold:   {metrics['dcg']['adaptive_threshold_mean']:.4f}")
        
        # 길이 통계
        length_stats = metrics['length_stats']
        print(f"\n📏 Length Statistics:")
        print(f"  Generated: {length_stats['generated_mean']:.1f} ± {length_stats['generated_std']:.1f} words")
        print(f"  Reference: {length_stats['reference_mean']:.1f} ± {length_stats['reference_std']:.1f} words")
        print(f"  Ratio:     {length_stats['length_ratio']:.2f}")
        
        # 샘플 출력
        print(f"\n📝 Sample DCG Results:")
        indices = np.random.choice(len(generated), min(3, len(generated)), replace=False)
        for i, idx in enumerate(indices, 1):
            print(f"\n  Example {i}:")
            print(f"    Reference: {references[idx]}")
            print(f"    Generated: {generated[idx]}")
    
    def save_dcg_results(self, metrics, generated, references, gate_analysis):
        """DCG 결과 저장"""
        # 요약 저장
        summary = {
            'model_name': 'Enhanced_DCG_T5',
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
        
        if metrics['dcg']:
            summary['dcg_analysis'] = metrics['dcg']
        
        # JSON 저장
        summary_path = os.path.join(DCG_CONFIG["results_dir"], "enhanced_dcg_summary.json")
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
                result['bert_f1'] = metrics['bert']['f1_scores'][i]
            
            detailed_results.append(result)
        
        df = pd.DataFrame(detailed_results)
        csv_path = os.path.join(DCG_CONFIG["results_dir"], "enhanced_dcg_detailed.csv")
        df.to_csv(csv_path, index=False)
        
        # Gate 분석 저장
        gate_path = os.path.join(DCG_CONFIG["results_dir"], "dcg_gate_analysis.json")
        with open(gate_path, 'w') as f:
            json.dump(gate_analysis, f, indent=2)
        
        print(f"\n✅ DCG Results saved:")
        print(f"   Summary: {summary_path}")
        print(f"   Detailed: {csv_path}")
        print(f"   Gate Analysis: {gate_path}")
        
        return summary_path, csv_path, gate_path
    
    def evaluate_dcg_model(self):
        """전체 DCG 평가 수행"""
        try:
            # 데이터 로드
            test_data = self.load_test_data()
            
            # DCG 모델 로드
            model, tokenizer = self.load_dcg_model()
            
            # DCG 분석과 함께 제목 생성
            generated, references, gate_analysis = self.generate_with_gate_analysis(
                model, tokenizer, test_data
            )
            
            # DCG 메트릭 계산
            metrics = self.calculate_dcg_metrics(generated, references, gate_analysis)
            
            # 결과 출력
            self.print_dcg_results(metrics, generated, references)
            
            # DCG 시각화
            plot_path = self.create_dcg_analysis_plots(metrics, generated, references, gate_analysis)
            
            # 결과 저장
            summary_path, csv_path, gate_path = self.save_dcg_results(
                metrics, generated, references, gate_analysis
            )
            
            return {
                'metrics': metrics,
                'generated': generated,
                'references': references,
                'gate_analysis': gate_analysis,
                'paths': {
                    'plot': plot_path,
                    'summary': summary_path,
                    'detailed': csv_path,
                    'gate_analysis': gate_path
                }
            }
            
        except Exception as e:
            print(f"❌ DCG Evaluation failed: {e}")
            raise e

def evaluate_dcg_model():
    """Enhanced DCG 모델 평가"""
    evaluator = DCGEvaluator()
    return evaluator.evaluate_dcg_model()

if __name__ == "__main__":
    results = evaluate_dcg_model()