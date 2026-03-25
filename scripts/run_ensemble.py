"""
앙상블 모델 실행 스크립트
텍스트를 입력받아 앙상블 모델로 혐오 표현을 감지합니다.
"""
import os
import sys
import argparse

# 프로젝트 루트 경로 설정
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

from scripts.ensemble_predict import EnsembleModel
from scripts.ensemble_config import EnsembleConfig


def predict_text(text: str, ensemble_model: EnsembleModel):
    """
    텍스트에 대한 예측 수행
    
    Args:
        text: 예측할 텍스트
        ensemble_model: 앙상블 모델 인스턴스
        
    Returns:
        예측 결과 (label, confidence, scores)
    """
    try:
        label, confidence, scores = ensemble_model.predict_ensemble(text)
        return label, confidence, scores
    except Exception as e:
        print(f"❌ 예측 중 오류 발생: {e}")
        return None, None, None


def format_result(label: int, confidence: float, scores: dict = None):
    """
    예측 결과를 포맷팅하여 출력
    
    Args:
        label: 예측된 레이블 (0: 정상, 1: 혐오)
        confidence: 신뢰도
        scores: 각 모델의 점수 (선택사항)
    """
    if label is None:
        print("❌ 예측 실패")
        return
    
    result_text = "✅ 정상" if label == 0 else "⚠️  혐오 표현 감지"
    print("\n" + "=" * 60)
    print("예측 결과")
    print("=" * 60)
    print(f"결과: {result_text}")
    print(f"레이블: {label} ({'정상' if label == 0 else '혐오'})")
    print(f"신뢰도: {confidence:.4f} ({confidence*100:.2f}%)")
    
    if scores:
        print("\n각 모델 점수:")
        for model_name, (model_label, model_conf) in scores.items():
            label_text = "정상" if model_label == 0 else "혐오"
            print(f"  - {model_name}: {label_text} (신뢰도: {model_conf:.4f})")
    
    print("=" * 60)


def interactive_mode(ensemble_model: EnsembleModel):
    """대화형 모드 실행"""
    print("\n" + "=" * 60)
    print("앙상블 모델 대화형 모드")
    print("=" * 60)
    print("💡 사용 방법:")
    print("  - 텍스트를 입력하고 Enter를 누르세요")
    print("  - 종료하려면 'quit', 'exit', 또는 '종료'를 입력하세요")
    print("  - 'help' 또는 '도움말'을 입력하면 도움말을 볼 수 있습니다")
    print("=" * 60)
    
    while True:
        try:
            print("\n💬 텍스트 입력: ", end="")
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
                print("이 프로그램은 앙상블 모델을 사용하여 텍스트의 혐오 표현을 감지합니다.")
                print("\n입력된 텍스트는 다음 모델들의 예측을 결합하여 분석됩니다:")
                print("  - 최신 학습 모델")
                print("  - KOR_UNSMILE 모델")
                print("\n결과:")
                print("  - 레이블 0: 정상 텍스트")
                print("  - 레이블 1: 혐오 표현이 포함된 텍스트")
                print("=" * 60)
                continue
            
            # 예측 수행
            print("\n🔍 분석 중...")
            label, confidence, scores = predict_text(user_input, ensemble_model)
            
            if label is not None:
                format_result(label, confidence, scores)
            else:
                print("❌ 예측을 수행할 수 없습니다.")
                
        except KeyboardInterrupt:
            print("\n\n👋 프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류가 발생했습니다: {e}")


def single_text_mode(text: str, ensemble_model: EnsembleModel):
    """단일 텍스트 모드 실행"""
    print(f"\n입력 텍스트: {text}")
    print("🔍 분석 중...")
    
    label, confidence, scores = predict_text(text, ensemble_model)
    
    if label is not None:
        format_result(label, confidence, scores)
    else:
        print("❌ 예측을 수행할 수 없습니다.")
        sys.exit(1)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="앙상블 모델로 혐오 표현 감지",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 대화형 모드
  python scripts/run_ensemble.py
  
  # 단일 텍스트 분석
  python scripts/run_ensemble.py --text "안녕하세요"
  
  # 파일에서 텍스트 읽기
  python scripts/run_ensemble.py --file input.txt
        """
    )
    
    parser.add_argument(
        '--text',
        type=str,
        help='분석할 텍스트 (단일 텍스트 모드)'
    )
    
    parser.add_argument(
        '--file',
        type=str,
        help='분석할 텍스트가 포함된 파일 경로'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='사용할 학습 모델 경로 (기본값: 자동 탐지)'
    )
    
    parser.add_argument(
        '--silent',
        action='store_true',
        help='모델 로딩 메시지 숨기기'
    )
    
    args = parser.parse_args()
    
    # 앙상블 모델 초기화
    print("=" * 60)
    print("앙상블 모델 로딩 중...")
    print("=" * 60)
    
    try:
        config = EnsembleConfig()
        if args.model_path:
            config.latest_model_path = args.model_path
        
        ensemble_model = EnsembleModel(config=config, silent=args.silent)
        
        if len(ensemble_model.models) == 0 and len(ensemble_model.pipelines) == 0:
            print("❌ 사용 가능한 모델이 없습니다.")
            sys.exit(1)
        
        print("✅ 앙상블 모델 로드 완료!")
        print(f"사용 가능한 모델: {list(ensemble_model.models.keys()) + list(ensemble_model.pipelines.keys())}")
        print(f"가중치: {ensemble_model.weights}")
        
    except Exception as e:
        print(f"❌ 앙상블 모델 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 실행 모드 선택
    if args.file:
        # 파일 모드
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            print(f"\n파일에서 {len(lines)}개 텍스트를 읽었습니다.\n")
            
            for idx, line in enumerate(lines, 1):
                text = line.strip()
                if not text:
                    continue
                
                print(f"\n[{idx}/{len(lines)}]")
                single_text_mode(text, ensemble_model)
                
        except FileNotFoundError:
            print(f"❌ 파일을 찾을 수 없습니다: {args.file}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ 파일 읽기 오류: {e}")
            sys.exit(1)
            
    elif args.text:
        # 단일 텍스트 모드
        single_text_mode(args.text, ensemble_model)
    else:
        # 대화형 모드
        interactive_mode(ensemble_model)


if __name__ == "__main__":
    main()

