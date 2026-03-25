"""
혐오 표현 필터링 서비스
"""
import os
import sys
from typing import Optional
import logging

# 프로젝트 루트 경로 설정
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

# 앙상블 모델 import
try:
    from scripts.ensemble_predict import EnsembleModel
    from scripts.ensemble_config import EnsembleConfig
    FILTERING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"필터링 모델을 로드할 수 없습니다: {e}")
    FILTERING_AVAILABLE = False
    EnsembleModel = None
    EnsembleConfig = None

from api.models import FilterResult
from api.config import config

logger = logging.getLogger(__name__)


class FilteringService:
    """혐오 표현 필터링 서비스 클래스"""
    
    def __init__(self):
        """필터링 서비스 초기화"""
        self.ensemble_model: Optional[EnsembleModel] = None
        self.is_loaded = False
        
        if FILTERING_AVAILABLE:
            try:
                ensemble_config = EnsembleConfig()
                self.ensemble_model = EnsembleModel(config=ensemble_config, silent=True)
                
                if len(self.ensemble_model.models) > 0 or len(self.ensemble_model.pipelines) > 0:
                    self.is_loaded = True
                    logger.info("필터링 모델 로드 완료")
                else:
                    logger.warning("사용 가능한 필터링 모델이 없습니다")
            except Exception as e:
                logger.error(f"필터링 모델 초기화 실패: {e}", exc_info=True)
        else:
            logger.error("필터링 모델을 사용할 수 없습니다")
    
    def filter(self, text: str):
        """
        텍스트 필터링 수행
        
        Args:
            text: 필터링할 텍스트
            
        Returns:
            FilterResponse: 필터링 결과
        """
        if not self.is_loaded or self.ensemble_model is None:
            logger.warning("필터링 모델이 로드되지 않았습니다. 모든 텍스트를 안전으로 처리합니다.")
            from types import SimpleNamespace
            return SimpleNamespace(
                is_safe=True,
                confidence=0.0,
                label=0,
                filter_result=FilterResult.SAFE
            )
        
        try:
            # 앙상블 모델로 예측
            label, confidence, _ = self.ensemble_model.predict_ensemble(text)
            
            if label is None:
                # 예측 실패 시 안전으로 처리
                logger.warning(f"필터링 예측 실패. 텍스트를 안전으로 처리합니다: {text[:50]}...")
                from types import SimpleNamespace
                return SimpleNamespace(
                    is_safe=True,
                    confidence=0.0,
                    label=0,
                    filter_result=FilterResult.SAFE
                )
            
            # 레이블 1 (혐오)이고 신뢰도가 임계값 이상이면 차단
            is_safe = not (label == 1 and confidence >= config.FILTER_THRESHOLD)
            
            filter_result = FilterResult.SAFE if is_safe else FilterResult.UNSAFE
            
            logger.debug(
                f"필터링 결과 - 텍스트: {text[:50]}..., "
                f"레이블: {label}, 신뢰도: {confidence:.4f}, 안전: {is_safe}"
            )
            
            from types import SimpleNamespace
            return SimpleNamespace(
                is_safe=is_safe,
                confidence=float(confidence),
                label=label,
                filter_result=filter_result
            )
            
        except Exception as e:
            logger.error(f"필터링 중 오류 발생: {e}", exc_info=True)
            # 오류 발생 시 안전으로 처리 (서비스 중단 방지)
            from types import SimpleNamespace
            return SimpleNamespace(
                is_safe=True,
                confidence=0.0,
                label=0,
                filter_result=FilterResult.SAFE
            )
    
    def is_model_loaded(self) -> bool:
        """필터링 모델 로드 여부 확인"""
        return self.is_loaded


# 전역 필터링 서비스 인스턴스
filtering_service = FilteringService()



