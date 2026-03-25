"""
앙상블 모델 설정 파일
"""
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class EnsembleConfig:
    """앙상블 모델 설정"""
    
    # 모델 경로 설정 (프로젝트 루트 기준 상대 경로)
    latest_model_path: Optional[str] = None  # 최신 학습 모델 경로 (None이면 자동 탐지 - 가장 최근 run_* 디렉토리 사용)
    
    # 사용할 모델 선택
    use_latest_model: bool = True  # 최신 학습 모델 사용 (필수)
    use_sentiment: bool = False  # Sentiment 모델 사용 (성능이 낮아 비활성화)
    use_kor_unsmile: bool = True  # kor_unsmile 모델 사용 (10가지 카테고리 분류)
    use_ko_sroberta: bool = False  # ko-sroberta-multitask 임베딩 모델 사용 (성능이 낮아 비활성화)
    
    # 각 모델의 가중치 (합이 1.0이 되도록 자동 정규화됨)
    weights: Dict[str, float] = None
    
    def __post_init__(self):
        """가중치 초기화"""
        if self.weights is None:
            # 사용 가능한 모델에 따라 가중치 자동 조정
            available_models = []
            if self.use_latest_model:
                available_models.append('latest')
            if self.use_sentiment:
                available_models.append('sentiment')
            if self.use_kor_unsmile:
                available_models.append('kor_unsmile')
            if self.use_ko_sroberta:
                available_models.append('ko_sroberta')
            
            # 기본 가중치 설정 (성능 기반 최적화)
            if 'latest' in available_models and 'kor_unsmile' in available_models:
                # Latest + kor_unsmile 이원 앙상블 (권장)
                # Latest가 더 우수하므로 더 높은 가중치
                self.weights = {
                    'latest': 0.7,      # 최신 학습 모델 (91.55% 정확도)
                    'kor_unsmile': 0.3, # 카테고리 분류 모델 (76.43% 정확도)
                }
            elif len(available_models) == 1 and 'latest' in available_models:
                self.weights = {
                    'latest': 1.0,      # 최신 모델만 사용 (가장 높은 성능)
                }
            else:
                # 기타 조합 시 기본 가중치
                self.weights = {
                    'latest': 0.7,
                    'kor_unsmile': 0.3,
                }
            
            # 사용하지 않는 모델 제거
            self.weights = {k: v for k, v in self.weights.items() if k in available_models}

