"""
API 서버 설정
"""
import os
from typing import Optional


class APIConfig:
    """API 서버 설정 클래스"""
    
    # 필터링 모델 설정
    ENSEMBLE_CONFIG_PATH: Optional[str] = os.getenv(
        "ENSEMBLE_CONFIG_PATH", 
        "scripts/ensemble_config.py"
    )
    
    # LLM 서비스 설정 (Groq만 사용)
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY", None)
    GROQ_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    GROQ_DEFAULT_MODEL: str = os.getenv("GROQ_DEFAULT_MODEL", "llama-3.1-8b-instant")
    
    # 필터링 임계값
    FILTER_THRESHOLD: float = float(os.getenv("FILTER_THRESHOLD", "0.5"))  # 신뢰도 0.5 이상이면 차단
    
    # 재요청 설정
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "1"))  # 최대 재요청 횟수
    RETRY_DELAY: float = float(os.getenv("RETRY_DELAY", "1.0"))  # 재요청 지연 시간 (초)
    
    # 서버 설정
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"


# 전역 설정 인스턴스
config = APIConfig()

