"""
Hugging Face 모델 학습 스크립트
"""
import os
import sys
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# 프로젝트 루트 경로 설정
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

# 설정 및 데이터 로더 import
from scripts.config import TrainingConfig
from scripts.load_data import (
    load_from_csv, 
    load_from_json, 
    load_kold_dataset,
    load_korean_hate_speech_dataset,
    load_unsmile_dataset,
    load_kmhas_dataset,
    load_humane_lab_dataset,
    load_multiple_datasets,
    validate_dataset
)

def compute_metrics(eval_pred):
    """평가 메트릭 계산 (개선된 버전)"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Weighted 평균 (클래스 불균형 고려)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    # Macro 평균 (각 클래스 동등하게 고려)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    
    # Binary 평균 (이진 분류용)
    precision_binary, recall_binary, f1_binary, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1_weighted,  # 기본 메트릭 (weighted)
        'f1_macro': f1_macro,  # Macro F1 (클래스 균형 고려)
        'f1_binary': f1_binary,  # Binary F1
        'precision': precision_weighted,
        'precision_macro': precision_macro,
        'recall': recall_weighted,
        'recall_macro': recall_macro,
    }

def preprocess_function(examples, tokenizer, max_length=512):
    """
    데이터 전처리 함수 (최적화)
    - truncation: 긴 텍스트 자르기
    - padding: 배치 내 동일 길이로 맞추기
    - max_length: 최대 길이 제한
    """
    return tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_attention_mask=True,  # 어텐션 마스크 반환 (명시적)
    )

def main():
    # 설정 로드
    config = TrainingConfig()
    
    # 설정 출력
    print("=" * 50)
    print("학습 설정")
    print("=" * 50)
    print(f"모델: {config.model_name}")
    print(f"레이블 수: {config.num_labels}")
    print(f"에포크: {config.num_epochs}")
    print(f"배치 크기: {config.batch_size}")
    print(f"학습률: {config.learning_rate}")
    print(f"최대 길이: {config.max_length}")
    print("=" * 50)
    
    # GPU 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 중인 디바이스: {device}")
    
    # 토크나이저와 모델 로드 (최적화)
    print(f"\n[1단계] 모델 로딩 중: {config.model_name}")
    print("  토크나이저 다운로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        use_fast=True  # Fast tokenizer 사용 (속도 향상)
    )
    print("  ✓ 토크나이저 로드 완료")
    
    print("  모델 다운로드 중...")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels,
        hidden_dropout_prob=config.dropout_rate,
        attention_probs_dropout_prob=config.dropout_rate
    )
    print("  ✓ 모델 로드 완료\n")
    
    # 클래스 가중치를 모델에 저장 (나중에 사용)
    if config.use_class_weights:
        model.class_weights = None  # 나중에 설정
    
    # 데이터셋 준비
    print("[2단계] 데이터셋 준비 중...")
    
    # 5개 데이터셋 모두 합쳐서 사용 (최대 성능 향상)
    from datasets import concatenate_datasets
    
    print("5개 데이터셋을 모두 합쳐서 사용합니다 (최대 성능 향상)...")
    print("(Korean Hate Speech + UnSmile + KOLD + KMHAS + HUMANE Lab - 약 274,000개)")
    
    datasets_to_merge = []
    
    # Korean Hate Speech 데이터셋 로드
    print("\n[1/5] Korean Hate Speech 데이터셋 로딩...")
    korean_hate_dataset = load_korean_hate_speech_dataset(
        train_path='datasets/korean-hate-speech-master/labeled/train.tsv',
        dev_path='datasets/korean-hate-speech-master/labeled/dev.tsv',
        label_type='hate',
        combine_train_dev=True
    )
    datasets_to_merge.append(korean_hate_dataset)
    
    # UnSmile 데이터셋 로드
    print("\n[2/5] Korean UnSmile 데이터셋 로딩...")
    try:
        unsmile_dataset = load_unsmile_dataset(
            dataset_path=None,
            use_huggingface=True,
            label_type='hate'
        )
        datasets_to_merge.append(unsmile_dataset)
    except Exception as e:
        print(f"  경고: UnSmile 데이터셋 로드 실패 ({e})")
        print("  다른 데이터셋만 사용합니다.")
    
    # KOLD 데이터셋 로드
    print("\n[3/5] KOLD 데이터셋 로딩...")
    kold_dataset = load_kold_dataset(
        'datasets/data/kold_v1.json',
        text_key='comment',
        label_key='OFF',
        label_mapping=None
    )
    datasets_to_merge.append(kold_dataset)
    
    # KMHAS 데이터셋 로드
    print("\n[4/5] KMHAS Korean Hate Speech 데이터셋 로딩...")
    try:
        import os
        kmhas_local_path = 'datasets/kmhas_korean_hate_speech'
        if not os.path.exists(kmhas_local_path):
            os.makedirs(kmhas_local_path, exist_ok=True)
        kmhas_dataset = load_kmhas_dataset(
            use_huggingface=True,  # Hugging Face Hub에서 먼저 시도
            label_type='hate',
            dataset_path=kmhas_local_path  # 실패 시 로컬 파일 사용
        )
        datasets_to_merge.append(kmhas_dataset)
    except Exception as e:
        print(f"  경고: KMHAS 데이터셋 로드 실패 ({e})")
        print("  다른 데이터셋만 사용합니다.")
    
    # HUMANE Lab 데이터셋 로드
    print("\n[5/5] HUMANE Lab 데이터셋 로딩...")
    try:
        humane_lab_dataset = load_humane_lab_dataset(
            dataset_dir='datasets/Selectstar_Tunip_HUMANE Lab_opendata',
            max_samples=None,  # 전체 사용 (학습용이므로)
            label_type='hate'
        )
        datasets_to_merge.append(humane_lab_dataset)
    except Exception as e:
        print(f"  경고: HUMANE Lab 데이터셋 로드 실패 ({e})")
        print("  다른 데이터셋만 사용합니다.")
    
    # 데이터셋 합치기
    print("\n  데이터셋 통합 중...")
    if len(datasets_to_merge) == 0:
        raise ValueError("로드된 데이터셋이 없습니다.")
    
    dataset = concatenate_datasets(datasets_to_merge)
    print(f"  ✓ 통합 완료! 총 {len(dataset)}개 데이터")
    print(f"  사용된 데이터셋 수: {len(datasets_to_merge)}개\n")
    
    # 5개 데이터셋 모두 합쳐서 사용 (최대 성능 향상)
    #     train_path='korean-hate-speech-master/labeled/train.tsv',
    #     dev_path=None,
    #     label_type='hate',
    #     combine_train_dev=False
    # )
    
    # 방법 3: KOLD만 사용
    # dataset = load_kold_dataset(
    #     'data/kold_v1.json',
    #     text_key='comment',
    #     label_key='OFF',
    #     label_mapping=None
    # )
    
    # 데이터셋 유효성 검사
    validate_dataset(dataset)
    print(f"  데이터셋 크기: {len(dataset)}개\n")
    
    # 데이터 전처리 (최적화)
    print("[3단계] 데이터 전처리 중...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, config.max_length),
        batched=True,
        batch_size=1000,  # 배치 크기 지정 (처리 속도 향상)
        remove_columns=[col for col in dataset.column_names if col != 'label'],  # label 제외하고 제거 (메모리 절약)
        desc="토큰화 중"
    )
    
    # 데이터셋 분할 (train/validation)
    if 'train' not in tokenized_dataset.column_names:
        # stratify는 ClassLabel 타입만 지원하므로 일반 분할 사용
        # (데이터셋이 이미 합쳐진 상태에서 레이블 균형은 자동으로 유지됨)
        split_dataset = tokenized_dataset.train_test_split(
            test_size=config.train_test_split,
            seed=config.seed
        )
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']
    else:
        train_dataset = tokenized_dataset['train']
        eval_dataset = tokenized_dataset['validation'] if 'validation' in tokenized_dataset else tokenized_dataset['test']
    
    print(f"  학습 데이터: {len(train_dataset)}개")
    print(f"  검증 데이터: {len(eval_dataset)}개\n")
    
    # 클래스 가중치 계산 (불균형 처리)
    class_weights = None
    if config.use_class_weights:
        labels = train_dataset['label']
        from collections import Counter
        label_counts = Counter(labels)
        total = len(labels)
        
        # 클래스 가중치: 전체 샘플 수 / (클래스 수 * 클래스별 샘플 수)
        class_weights = torch.tensor([
            total / (config.num_labels * label_counts.get(0, 1)),
            total / (config.num_labels * label_counts.get(1, 1))
        ], dtype=torch.float32)
        
        print(f"  클래스 분포: 0={label_counts.get(0, 0)}개, 1={label_counts.get(1, 0)}개")
        print(f"  클래스 가중치: {class_weights.tolist()}\n")
        
        # 모델에 클래스 가중치 저장 (커스텀 Trainer에서 사용)
        model.class_weights = class_weights.to(device) if torch.cuda.is_available() else class_weights
    
    # 타임스탬프 기반 결과 디렉토리 생성
    from datetime import datetime
    import os
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = os.path.join(config.output_dir, f"run_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    print(f"[4단계] 학습 설정 완료")
    print(f"  결과 디렉토리: {result_dir}")
    
    # 데이터 콜레이터
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 학습 인자 설정 (성능 개선 버전)
    training_args = TrainingArguments(
        output_dir=result_dir,  # 타임스탬프 기반 디렉토리 사용
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,  # 학습률 워밍업
        lr_scheduler_type=config.lr_scheduler_type,  # 학습률 스케줄러
        lr_scheduler_kwargs=config.lr_scheduler_kwargs if config.lr_scheduler_kwargs else {},  # 스케줄러 추가 파라미터
        gradient_accumulation_steps=config.gradient_accumulation_steps,  # 그래디언트 누적
        logging_dir=f'{result_dir}/logs',
        logging_steps=config.logging_steps,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps if config.eval_strategy == "steps" else None,
        save_strategy=config.eval_strategy,
        save_steps=config.save_steps if config.eval_strategy == "steps" else None,
        load_best_model_at_end=True,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=True,  # F1, accuracy는 높을수록 좋음
        save_total_limit=config.save_total_limit,
        fp16=config.use_fp16 and torch.cuda.is_available(),
        seed=config.seed,
        dataloader_num_workers=config.dataloader_num_workers,  # 설정에서 가져오기
        dataloader_pin_memory=config.dataloader_pin_memory if torch.cuda.is_available() else False,  # GPU 사용 시만 활성화
        remove_unused_columns=config.remove_unused_columns,  # 사용하지 않는 컬럼 제거
        group_by_length=config.group_by_length,  # 길이별 그룹화
        report_to="none",  # wandb/tensorboard 비활성화
        max_grad_norm=config.max_grad_norm if config.max_grad_norm > 0 else None,  # 그래디언트 클리핑
        optim="adamw_torch",  # 최적화 알고리즘 명시 (기본값이지만 명시적으로 설정)
        include_inputs_for_metrics=False,  # 메트릭 계산 시 입력 제외 (메모리 절약)
    )
    
    # Early stopping 콜백 (선택사항)
    callbacks = []
    if config.early_stopping_patience > 0:
        from transformers import EarlyStoppingCallback
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience,
                early_stopping_threshold=0.001  # 최소 개선량
            )
        )
        print(f"  Early Stopping 활성화: patience={config.early_stopping_patience}\n")
    
    # 고급 기법 import
    try:
        from scripts.advanced_techniques import FocalLoss, LabelSmoothingCrossEntropy
    except ImportError:
    from advanced_techniques import FocalLoss, LabelSmoothingCrossEntropy
    
    # 커스텀 Trainer 클래스 (고급 학습 기법 지원)
    class AdvancedTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.config_obj = config
        
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            """
            커스텀 loss 계산 (고급 기법 지원)
            kwargs는 num_items_in_batch 등을 포함할 수 있음
            """
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            
            # Loss function 선택
            if self.config_obj.use_focal_loss:
                # Focal Loss 사용
                alpha = model.class_weights if hasattr(model, 'class_weights') and model.class_weights is not None else None
                loss_fct = FocalLoss(alpha=alpha, gamma=self.config_obj.focal_loss_gamma)
                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            elif self.config_obj.label_smoothing > 0:
                # Label Smoothing 사용
                weight = model.class_weights if hasattr(model, 'class_weights') and model.class_weights is not None else None
                loss_fct = LabelSmoothingCrossEntropy(smoothing=self.config_obj.label_smoothing, weight=weight)
                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            else:
                # 기본 Cross Entropy Loss
                if hasattr(model, 'class_weights') and model.class_weights is not None:
                    loss_fct = torch.nn.CrossEntropyLoss(weight=model.class_weights)
                else:
                    loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            
            return (loss, outputs) if return_outputs else loss
        
    
    # 트레이너 초기화 (고급 기법 사용 여부에 따라 선택)
    use_advanced = (config.use_class_weights or config.use_focal_loss or 
                   config.label_smoothing > 0 or config.max_grad_norm > 0)
    trainer_class = AdvancedTrainer if use_advanced else Trainer
    
    # tokenizer 대신 processing_class 사용 (FutureWarning 해결)
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  # tokenizer 대신 processing_class 사용
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks if callbacks else None,
    )
    
    # 학습 시작
    print("\n[5단계] 학습 시작...\n")
    trainer.train()
    
    # 모델 저장 (타임스탬프 기반 디렉토리에 저장)
    final_model_path = os.path.join(result_dir, "model")
    print(f"\n모델 저장 중: {final_model_path}")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"모델이 저장되었습니다: {final_model_path}")
    
    # 최종 평가
    print("\n최종 평가 중...")
    eval_results = trainer.evaluate()
    print(f"\n최종 평가 결과:")
    print("=" * 60)
    for key, value in eval_results.items():
        if isinstance(value, float):
            if 'loss' in key:
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value:.4f} ({value*100:.2f}%)")
    print("=" * 60)
    
    # 성능 요약
    print("\n성능 요약:")
    print(f"  정확도: {eval_results.get('eval_accuracy', 0)*100:.2f}%")
    print(f"  F1 점수 (Weighted): {eval_results.get('eval_f1', 0)*100:.2f}%")
    print(f"  F1 점수 (Macro): {eval_results.get('eval_f1_macro', 0)*100:.2f}%")
    print(f"  Precision: {eval_results.get('eval_precision', 0)*100:.2f}%")
    print(f"  Recall: {eval_results.get('eval_recall', 0)*100:.2f}%")
    
    # 로그 파일로 저장 (같은 타임스탬프 디렉토리에 저장)
    import json
    # timestamp와 result_dir는 이미 위에서 생성됨
    
    log_file_path = os.path.join(result_dir, 'training_results.json')
    log_summary_path = os.path.join(result_dir, 'training_summary.txt')
    
    # JSON 형식으로 상세 결과 저장 (모든 설정 포함)
    results_to_save = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'result_directory': result_dir,  # 결과 디렉토리 경로 추가
        'model_path': final_model_path,  # 모델 경로 추가
        'model_config': {
            'model_name': config.model_name,
            'num_labels': config.num_labels,
        },
        'data_config': {
            'max_length': config.max_length,
            'train_test_split': config.train_test_split,
            'train_samples': len(train_dataset),
            'eval_samples': len(eval_dataset),
        },
        'training_config': {
            'num_epochs': config.num_epochs,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'weight_decay': config.weight_decay,
            'warmup_ratio': config.warmup_ratio,
            'lr_scheduler_type': config.lr_scheduler_type,
            'gradient_accumulation_steps': config.gradient_accumulation_steps,
        },
        'optimization_config': {
            'use_fp16': config.use_fp16,
            'use_class_weights': config.use_class_weights,
            'dropout_rate': config.dropout_rate,
            'use_focal_loss': config.use_focal_loss,
            'focal_loss_gamma': config.focal_loss_gamma,
            'label_smoothing': config.label_smoothing,
            'max_grad_norm': config.max_grad_norm,
        },
        'evaluation_config': {
            'eval_strategy': config.eval_strategy,
            'metric_for_best_model': config.metric_for_best_model,
            'early_stopping_patience': config.early_stopping_patience,
        },
        'final_evaluation': eval_results,
        'performance_summary': {
            'accuracy': eval_results.get('eval_accuracy', 0) * 100,
            'f1_weighted': eval_results.get('eval_f1', 0) * 100,
            'f1_macro': eval_results.get('eval_f1_macro', 0) * 100,
            'f1_binary': eval_results.get('eval_f1_binary', 0) * 100,
            'precision': eval_results.get('eval_precision', 0) * 100,
            'precision_macro': eval_results.get('eval_precision_macro', 0) * 100,
            'recall': eval_results.get('eval_recall', 0) * 100,
            'recall_macro': eval_results.get('eval_recall_macro', 0) * 100,
            'loss': eval_results.get('eval_loss', 0)
        }
    }
    
    with open(log_file_path, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, indent=2, ensure_ascii=False)
    
    # 텍스트 형식으로 요약 저장 (모든 설정 포함)
    with open(log_summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("학습 결과 요약\n")
        f.write("=" * 70 + "\n")
        f.write(f"학습 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"결과 디렉토리: {result_dir}\n")
        f.write(f"모델 경로: {final_model_path}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("모델 설정\n")
        f.write("-" * 70 + "\n")
        f.write(f"모델: {config.model_name}\n")
        f.write(f"레이블 수: {config.num_labels}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("데이터 설정\n")
        f.write("-" * 70 + "\n")
        f.write(f"최대 길이: {config.max_length}\n")
        f.write(f"검증 데이터 비율: {config.train_test_split}\n")
        f.write(f"학습 데이터: {len(train_dataset):,}개\n")
        f.write(f"검증 데이터: {len(eval_dataset):,}개\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("학습 하이퍼파라미터\n")
        f.write("-" * 70 + "\n")
        f.write(f"에포크: {config.num_epochs}\n")
        f.write(f"배치 크기: {config.batch_size}\n")
        f.write(f"학습률: {config.learning_rate}\n")
        f.write(f"Weight Decay: {config.weight_decay}\n")
        f.write(f"워밍업 비율: {config.warmup_ratio}\n")
        f.write(f"학습률 스케줄러: {config.lr_scheduler_type}\n")
        f.write(f"그래디언트 누적: {config.gradient_accumulation_steps}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("최적화 설정\n")
        f.write("-" * 70 + "\n")
        f.write(f"FP16 사용: {config.use_fp16}\n")
        f.write(f"클래스 가중치: {config.use_class_weights}\n")
        f.write(f"드롭아웃: {config.dropout_rate}\n")
        f.write(f"Focal Loss: {config.use_focal_loss}\n")
        if config.use_focal_loss:
            f.write(f"  - Gamma: {config.focal_loss_gamma}\n")
        f.write(f"Label Smoothing: {config.label_smoothing}\n")
        f.write(f"그래디언트 클리핑: {config.max_grad_norm}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("평가 설정\n")
        f.write("-" * 70 + "\n")
        f.write(f"평가 전략: {config.eval_strategy}\n")
        f.write(f"최적 모델 메트릭: {config.metric_for_best_model}\n")
        f.write(f"Early Stopping Patience: {config.early_stopping_patience}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("최종 성능 지표\n")
        f.write("-" * 70 + "\n")
        f.write(f"정확도 (Accuracy): {eval_results.get('eval_accuracy', 0)*100:.2f}%\n")
        f.write(f"F1 점수 (Weighted): {eval_results.get('eval_f1', 0)*100:.2f}%\n")
        f.write(f"F1 점수 (Macro): {eval_results.get('eval_f1_macro', 0)*100:.2f}%\n")
        f.write(f"F1 점수 (Binary): {eval_results.get('eval_f1_binary', 0)*100:.2f}%\n")
        f.write(f"Precision (Weighted): {eval_results.get('eval_precision', 0)*100:.2f}%\n")
        f.write(f"Precision (Macro): {eval_results.get('eval_precision_macro', 0)*100:.2f}%\n")
        f.write(f"Recall (Weighted): {eval_results.get('eval_recall', 0)*100:.2f}%\n")
        f.write(f"Recall (Macro): {eval_results.get('eval_recall_macro', 0)*100:.2f}%\n")
        f.write(f"Loss: {eval_results.get('eval_loss', 0):.4f}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("상세 평가 결과\n")
        f.write("-" * 70 + "\n")
        for key, value in eval_results.items():
            if isinstance(value, float):
                if 'loss' in key:
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value:.4f} ({value*100:.2f}%)\n")
        f.write("=" * 70 + "\n")
    
    print(f"\n✓ 학습 결과가 저장되었습니다:")
    print(f"  - 결과 디렉토리: {result_dir}")
    print(f"  - 모델 경로: {final_model_path}")
    print(f"  - 상세 결과 (JSON): {log_file_path}")
    print(f"  - 요약 (TXT): {log_summary_path}")
    
    print("\n" + "=" * 70)
    print("학습 완료!")
    print("=" * 70)
    print(f"모든 결과는 다음 디렉토리에 저장되었습니다:")
    print(f"  {result_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()

