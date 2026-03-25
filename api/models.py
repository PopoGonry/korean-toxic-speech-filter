"""
API 요청/응답 모델 정의
"""
from enum import Enum


class FilterResult(str, Enum):
    """필터링 결과"""
    SAFE = "safe"  # 안전
    UNSAFE = "unsafe"  # 혐오 표현 감지


