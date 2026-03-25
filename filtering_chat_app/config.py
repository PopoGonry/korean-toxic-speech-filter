"""
설정 관리 (API 키 저장 및 로드)
"""
import os
import json
from typing import Optional


class Config:
    """설정 클래스"""
    
    def __init__(self):
        """설정 초기화"""
        config_dir = os.path.dirname(os.path.abspath(__file__))
        self.CONFIG_FILE = os.path.join(config_dir, "config.json")
        
        self.groq_api_key: Optional[str] = None
        self.groq_base_url: str = "https://api.groq.com/openai/v1"
        self.groq_default_model: str = "llama-3.1-8b-instant"
        self.filter_threshold: float = 0.5
        self.max_retries: int = 1
        self.retry_delay: float = 1.0
        
        self.load_config()
    
    def load_config(self):
        """설정 파일에서 로드"""
        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.groq_api_key = data.get('groq_api_key')
                    self.groq_default_model = data.get('groq_default_model', self.groq_default_model)
                    self.filter_threshold = data.get('filter_threshold', self.filter_threshold)
                    self.max_retries = data.get('max_retries', self.max_retries)
                    self.retry_delay = data.get('retry_delay', self.retry_delay)
            except Exception as e:
                print(f"설정 파일 로드 실패: {e}")
    
    def save_config(self):
        """설정 파일에 저장"""
        try:
            data = {
                'groq_api_key': self.groq_api_key,
                'groq_default_model': self.groq_default_model,
                'filter_threshold': self.filter_threshold,
                'max_retries': self.max_retries,
                'retry_delay': self.retry_delay
            }
            with open(self.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"설정 파일 저장 실패: {e}")
    
    def is_groq_configured(self) -> bool:
        """Groq API 키가 설정되어 있는지 확인"""
        return self.groq_api_key is not None and len(self.groq_api_key.strip()) > 0


config = Config()
