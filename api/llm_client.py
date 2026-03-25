"""
LLM API 클라이언트 (Groq만 지원)
"""
import httpx
import logging
from typing import Optional
from api.config import config

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM API 클라이언트 (Groq)"""
    
    def __init__(self):
        """LLM 클라이언트 초기화"""
        self.groq_api_key = config.GROQ_API_KEY
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> Optional[str]:
        """
        Groq LLM으로 텍스트 생성
        
        Args:
            prompt: 사용자 프롬프트
            model: 사용할 모델명 (None이면 기본값 사용)
            max_tokens: 최대 토큰 수
            temperature: 생성 온도
            system_prompt: 시스템 프롬프트
            
        Returns:
            생성된 텍스트 또는 None (실패 시)
        """
        if not self.groq_api_key:
            logger.error("Groq API 키가 설정되지 않았습니다. GROQ_API_KEY 환경변수를 설정하세요.")
            return None
        
        url = f"{config.GROQ_BASE_URL}/chat/completions"
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model or config.GROQ_DEFAULT_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                
                result = response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", None)
                
        except httpx.TimeoutException:
            logger.error(f"Groq API 타임아웃: {model}")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"Groq API 오류 (HTTP {e.response.status_code}): {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"Groq API 요청 실패: {e}", exc_info=True)
            return None
    
    def is_available(self) -> bool:
        """Groq 사용 가능 여부 확인"""
        return self.groq_api_key is not None


# 전역 LLM 클라이언트 인스턴스
llm_client = LLMClient()



