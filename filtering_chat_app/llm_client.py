"""
LLM API 클라이언트 (Groq만 지원)
"""
import httpx
import logging
from typing import Optional
from config import config

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM API 클라이언트 (Groq)"""
    
    def __init__(self):
        """LLM 클라이언트 초기화"""
        pass
    
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
        """
        # config에서 직접 API 키 가져오기
        api_key = config.groq_api_key
        
        if not api_key or len(api_key.strip()) == 0:
            logger.error("Groq API 키가 설정되지 않았습니다.")
            return None
        
        url = f"{config.groq_base_url}/chat/completions"
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model or config.groq_default_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        headers = {
            "Authorization": f"Bearer {api_key}",
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
        return config.groq_api_key is not None and len(config.groq_api_key.strip()) > 0


llm_client = LLMClient()
