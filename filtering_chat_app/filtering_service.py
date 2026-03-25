"""
혐오 표현 필터링 서비스
원본 프로젝트의 앙상블 모델을 사용
"""
import os
import sys
from typing import Optional
import logging

# 원본 프로젝트 경로 설정
# .exe 파일로 실행될 때와 일반 Python 스크립트로 실행될 때 경로 처리
if getattr(sys, 'frozen', False):
    # .exe 파일로 실행될 때
    base_path = os.path.dirname(sys.executable)
    
    # scripts 디렉토리를 찾기 위해 여러 경로 시도
    project_root = None
    
    # 1. 실행 파일과 같은 디렉토리
    if os.path.exists(os.path.join(base_path, 'scripts')):
        project_root = base_path
    
    # 2. 상위 디렉토리
    if project_root is None:
        parent_dir = os.path.dirname(base_path)
        if os.path.exists(os.path.join(parent_dir, 'scripts')):
            project_root = parent_dir
    
    # 3. 상위의 상위 디렉토리 (dist 폴더를 거쳐서)
    if project_root is None:
        parent_parent = os.path.dirname(os.path.dirname(base_path))
        if os.path.exists(os.path.join(parent_parent, 'scripts')):
            project_root = parent_parent
    
    # 4. 현재 위치에서 상위로 올라가며 찾기 (최대 5단계)
    if project_root is None:
        current = base_path
        for _ in range(5):
            if os.path.exists(os.path.join(current, 'scripts')):
                project_root = current
                break
            parent = os.path.dirname(current)
            if parent == current:  # 루트에 도달
                break
            current = parent
else:
    # 일반 Python 스크립트로 실행될 때
    # filtering_chat_app의 상위 디렉토리가 프로젝트 루트
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if project_root and project_root not in sys.path:
    sys.path.insert(0, project_root)

# 작업 디렉토리 설정
if project_root and os.path.exists(project_root):
    os.chdir(project_root)

# 앙상블 모델 import
FILTERING_AVAILABLE = False
EnsembleModel = None
EnsembleConfig = None

try:
    from scripts.ensemble_predict import EnsembleModel
    from scripts.ensemble_config import EnsembleConfig
    FILTERING_AVAILABLE = True
except ImportError as e:
    error_msg = str(e)
    logging.warning(f"필터링 모델을 로드할 수 없습니다: {error_msg}")
    
    # .exe 실행 시 더 자세한 안내
    if getattr(sys, 'frozen', False):
        print("\n" + "=" * 60)
        print("⚠️  필터링 모델 로드 실패")
        print("=" * 60)
        print(f"오류: {error_msg}")
        print(f"\n현재 작업 디렉토리: {os.getcwd()}")
        print(f"프로젝트 루트: {project_root}")
        scripts_path = os.path.join(project_root, 'scripts')
        print(f"scripts 경로: {scripts_path}")
        print(f"scripts 존재 여부: {os.path.exists(scripts_path)}")
        print("\n💡 해결 방법:")
        print("  1. FilteringChat.exe를 원본 프로젝트 루트에 복사하세요")
        print("     예: C:\\Users\\rhtjd\\Desktop\\수호당\\filtering AI model\\FilteringChat.exe")
        print("  2. 또는 scripts와 results 디렉토리를 실행 파일과 같은 폴더에 복사")
        print("=" * 60)
        print("\n⚠️  필터링 없이 계속 진행합니다. (모든 입력이 안전으로 처리됩니다)")
        print("=" * 60 + "\n")
    
    FILTERING_AVAILABLE = False

from enum import Enum

class FilterResult(str, Enum):
    """필터링 결과"""
    SAFE = "safe"  # 안전
    UNSAFE = "unsafe"  # 혐오 표현 감지


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
                    logging.info("필터링 모델 로드 완료")
                else:
                    logging.warning("사용 가능한 필터링 모델이 없습니다")
            except Exception as e:
                logging.error(f"필터링 모델 초기화 실패: {e}", exc_info=True)
        else:
            logging.error("필터링 모델을 사용할 수 없습니다")
    
    def filter(self, text: str):
        """
        텍스트 필터링 수행
        
        Args:
            text: 필터링할 텍스트
            
        Returns:
            필터링 결과 객체
        """
        if not self.is_loaded or self.ensemble_model is None:
            logging.warning("필터링 모델이 로드되지 않았습니다. 모든 텍스트를 안전으로 처리합니다.")
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
                logging.warning(f"필터링 예측 실패. 텍스트를 안전으로 처리합니다: {text[:50]}...")
                from types import SimpleNamespace
                return SimpleNamespace(
                    is_safe=True,
                    confidence=0.0,
                    label=0,
                    filter_result=FilterResult.SAFE
                )
            
            # 레이블 1 (혐오)이고 신뢰도가 임계값 이상이면 차단
            from config import config
            is_safe = not (label == 1 and confidence >= config.filter_threshold)
            
            filter_result = FilterResult.SAFE if is_safe else FilterResult.UNSAFE
            
            logging.debug(
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
            logging.error(f"필터링 중 오류 발생: {e}", exc_info=True)
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
