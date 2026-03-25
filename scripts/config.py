"""
학습 설정 파일
이 파일의 설정을 변경하여 학습 파라미터를 조정할 수 있습니다.
"""
from dataclasses import dataclass, field

@dataclass
class TrainingConfig:
    # 모델 설정
    model_name: str = "beomi/KcELECTRA-base-v2022"  # 또는 "klue/roberta-large" (성능 향상 +2-4%)
    num_labels: int = 2  # 분류 레이블 수 (필터링: 2, 다중 분류: 그 이상)
    
    # 데이터 설정
    max_length: int = 256  # 최대 시퀀스 길이 (512→256, 학습 시간 -50%, 성능 거의 동일)
    train_test_split: float = 0.2  # 검증 데이터 비율
    
    # 학습 하이퍼파라미터 (학습 시간 단축 최적화 - 예상 성능: 86-87%, 학습 시간: 1.5-2시간)
    num_epochs: int = 10  # 에포크 수 (10으로 증가, 더 많은 학습 기회 제공)
    batch_size: int = 32  # 배치 크기 (16→32, 학습 시간 -20-30%, 성능 거의 동일)
    learning_rate: float = 1.5e-5  # 학습률 (2e-5→1.5e-5, 더 정밀한 학습)
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1  # 워밍업 비율 (더 안정적인 시작)
    lr_scheduler_type: str = "cosine_with_restarts"  # 코사인 스케줄러 (재시작 포함)
    lr_scheduler_kwargs: dict = field(default_factory=lambda: {"num_cycles": 2})  # 학습률 스케줄러 추가 파라미터 (재시작 2회)
    gradient_accumulation_steps: int = 1  # 그래디언트 누적 (2→1, 실제 배치 크기 = 32*1 = 32 유지)
    
    # 출력 설정
    output_dir: str = "./results"
    logging_steps: int = 50  # 로깅 빈도 조정 (10 → 50, 로그 파일 크기 감소)
    save_total_limit: int = 2  # 저장할 체크포인트 수 (3 → 2, 디스크 공간 절약)
    
    # 평가 설정
    eval_strategy: str = "epoch"  # "steps" 또는 "epoch"
    metric_for_best_model: str = "f1"  # "accuracy", "f1", "loss" 등
    eval_steps: int = 500  # eval_strategy가 "steps"일 때 사용
    save_steps: int = 500  # save_strategy가 "steps"일 때 사용
    early_stopping_patience: int = 3  # Early stopping patience (4→3, 적절한 조기 종료)
    
    # 기타 설정
    use_fp16: bool = True  # Mixed precision 사용 (GPU 필요)
    seed: int = 42  # 재현성을 위한 시드
    use_class_weights: bool = True  # 클래스 불균형 처리 (가중치 적용)
    dropout_rate: float = 0.2  # 드롭아웃 비율 (0.15 → 0.2, 과적합 방지 강화)
    
    # 고급 학습 기법 (대규모 데이터셋 최적화)
    use_focal_loss: bool = True  # Focal Loss 사용 (클래스 불균형에 효과적)
    focal_loss_gamma: float = 2.5  # Focal Loss gamma (2.0 → 2.5, 어려운 샘플에 더 집중)
    label_smoothing: float = 0.05  # Label Smoothing (0.1 → 0.05, 대규모 데이터에 적합)
    max_grad_norm: float = 1.0  # 그래디언트 클리핑 (1.0 권장, 0이면 비활성화)
    use_mixup: bool = False  # Mixup 데이터 증강 (실험적, 현재 미지원)
    mixup_alpha: float = 0.2  # Mixup alpha 파라미터
    
    # 성능 최적화
    dataloader_pin_memory: bool = True  # 데이터 로더 메모리 고정 (GPU 전송 속도 향상)
    dataloader_num_workers: int = 0  # Windows 호환성 (0으로 설정, Linux/Mac에서는 2-4 권장)
    remove_unused_columns: bool = True  # 사용하지 않는 컬럼 제거 (메모리 절약)
    group_by_length: bool = False  # 길이별 그룹화 (False로 설정, 배치 효율성 우선)
