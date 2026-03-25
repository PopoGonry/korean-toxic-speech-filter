"""
혐오 표현 필터링 + LLM 채팅 프로그램
처음 실행 시 Groq API 키를 입력받고, 이후 CLI로 실행됩니다.
"""
import os
import sys
import asyncio
import logging
from typing import Optional
import getpass

# 현재 디렉토리를 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
os.chdir(current_dir)

from config import config
from llm_client import llm_client
from filtering_service import filtering_service, FilterResult

# 로깅 설정
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class FilteringChatCLI:
    """혐오 표현 필터링 + LLM 채팅 CLI 클래스"""
    
    def __init__(self):
        """CLI 초기화"""
        self.llm_client = llm_client
        self.filtering_service = filtering_service
        
    def setup_groq_api_key(self):
        """Groq API 키 설정"""
        if config.is_groq_configured():
            return True
        
        print("\n" + "=" * 60)
        print("Groq API 키 설정")
        print("=" * 60)
        print("처음 실행하시는군요! Groq API 키가 필요합니다.")
        print("\n💡 API 키 발급 방법:")
        print("  1. https://console.groq.com 접속")
        print("  2. 회원가입 후 API Keys 메뉴에서 키 생성")
        print("  3. 생성된 API 키를 복사하세요")
        print("-" * 60)
        
        try:
            api_key = input("Groq API 키를 입력하세요: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n❌ 입력이 취소되었습니다.")
            return False
        
        if not api_key:
            print("❌ API 키가 입력되지 않았습니다.")
            return False
        
        # API 키 설정
        config.groq_api_key = api_key
        
        # 파일에 저장
        config.save_config()
        
        print("✅ API 키가 저장되었습니다!")
        return True
    
    async def process_message(self, user_input: str, model: str, max_retries: int, is_retry: bool = False) -> Optional[str]:
        """
        메시지 처리 (1차 필터링 -> LLM -> 2차 필터링 -> 재시도)
        
        Args:
            user_input: 사용자 입력
            model: 사용할 LLM 모델
            max_retries: 최대 재시도 횟수
            is_retry: 재요청 여부 (재요청 시 1차 필터링 건너뛰기)
            
        Returns:
            LLM 응답 또는 None
        """
        # 1차 필터링: 입력 검사 (재요청이 아닐 때만)
        if not is_retry:
            print("🔍 1차 필터링 중 (입력 검사)...")
            filter_result = self.filtering_service.filter(user_input)
            
            if filter_result.filter_result == FilterResult.UNSAFE:
                print("❌ 1차 필터링 실패: 입력에 혐오 표현이 감지되었습니다.")
                print(f"   신뢰도: {filter_result.confidence:.2%}")
                return None
            
            print("✅ 1차 필터링 통과")
        
        # LLM 응답 생성
        print(f"🤖 LLM 응답 생성 중 (Groq)...")
        response = await self.llm_client.generate(
            prompt=user_input,
            model=model
        )
        
        if not response:
            print("❌ LLM 응답 생성 실패 (Groq)")
            return None
        
        # 2차 필터링: 응답 검사
        print("🔍 2차 필터링 중 (응답 검사)...")
        response_filter = self.filtering_service.filter(response)
        
        if response_filter.filter_result == FilterResult.SAFE:
            print("✅ 2차 필터링 통과")
            return response
        
        # 응답에 혐오 표현이 감지된 경우 재시도
        print(f"⚠️  2차 필터링 실패: 응답에 혐오 표현이 감지되었습니다.")
        print(f"   신뢰도: {response_filter.confidence:.2%}")
        print(f"   디버그: LLM이 부적절한 응답을 생성했습니다. 재요청합니다...")
        
        if max_retries > 0:
            print(f"🔄 재요청 중... (남은 횟수: {max_retries})")
            retry_response = await self.process_message(
                user_input, model, max_retries - 1, is_retry=True
            )
            
            # 재요청된 응답은 이미 process_message 내부에서 2차 필터링을 통과했으므로
            # 바로 반환 (추가 필터링 불필요)
            if retry_response:
                return retry_response
            
            return None
        else:
            print("❌ 최대 재시도 횟수에 도달했습니다.")
            return None
    
    async def run(self, model: Optional[str] = None, max_retries: int = 1):
        """채팅 루프 실행"""
        # API 키 설정 확인
        if not self.setup_groq_api_key():
            return
        
        # 모델 설정
        model = model or config.groq_default_model
        
        # 필터링 모델 로드 확인
        if not self.filtering_service.is_model_loaded():
            print("⚠️  필터링 모델이 로드되지 않았습니다. 모든 입력이 안전으로 처리됩니다.")
        
        print("\n" + "=" * 60)
        print("혐오 표현 필터링 + LLM 채팅")
        print("=" * 60)
        print(f"LLM 제공자: Groq")
        print(f"모델: {model}")
        print(f"필터링 모델: {'✅ 로드됨' if self.filtering_service.is_model_loaded() else '❌ 로드 안 됨'}")
        print("=" * 60)
        print("\n💡 사용 방법:")
        print("  - 메시지를 입력하고 Enter를 누르세요")
        print("  - 종료하려면 'quit', 'exit', 또는 '종료'를 입력하세요")
        print("  - 'help' 또는 '도움말'을 입력하면 도움말을 볼 수 있습니다")
        print("=" * 60)
        
        while True:
            try:
                print("\n💬 메시지 입력: ", end="")
                user_input = input().strip()
                
                if not user_input:
                    continue
                
                # 종료 명령
                if user_input.lower() in ['quit', 'exit', '종료']:
                    print("\n👋 프로그램을 종료합니다.")
                    break
                
                # 도움말
                if user_input.lower() in ['help', '도움말']:
                    print("\n" + "=" * 60)
                    print("도움말")
                    print("=" * 60)
                    print("이 프로그램은 다음과 같이 작동합니다:")
                    print("1. 입력 메시지를 1차 필터링으로 검사합니다")
                    print("2. 안전하면 LLM에 전달하여 응답을 생성합니다")
                    print("3. LLM 응답을 2차 필터링으로 검사합니다")
                    print("4. 응답에 문제가 있으면 재요청합니다 (최대 재시도 횟수까지)")
                    print("=" * 60)
                    continue
                
                print("\n" + "-" * 60)
                print(f"입력: {user_input}")
                print("-" * 60)
                
                # 메시지 처리
                response = await self.process_message(
                    user_input, model, max_retries
                )
                
                if response:
                    print("\n" + "=" * 60)
                    print("응답:")
                    print("=" * 60)
                    print(response)
                    print("=" * 60)
                else:
                    print("\n⚠️  응답을 생성할 수 없습니다.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 프로그램을 종료합니다.")
                break
            except Exception as e:
                logger.error(f"오류 발생: {e}", exc_info=True)
                print(f"\n❌ 오류가 발생했습니다: {e}")


async def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="혐오 표현 필터링 + LLM 채팅")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="사용할 LLM 모델 (기본값: llama-3.1-8b-instant)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="최대 재시도 횟수 (기본값: 1)"
    )
    
    args = parser.parse_args()
    
    cli = FilteringChatCLI()
    await cli.run(model=args.model, max_retries=args.max_retries)


if __name__ == "__main__":
    asyncio.run(main())
