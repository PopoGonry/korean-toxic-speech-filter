"""
이종 앙상블 모델 예측 스크립트
최신 학습 모델과 Hugging Face 모델들을 결합하여 예측
"""
import os
import sys
import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
    BertForSequenceClassification
)
from typing import List, Tuple, Dict, Optional

# 프로젝트 루트 경로 설정 (scripts 디렉토리의 상위 디렉토리)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)  # 작업 디렉토리를 프로젝트 루트로 변경

# 설정 파일 import
try:
    from scripts.ensemble_config import EnsembleConfig
except ImportError:
    try:
        from ensemble_config import EnsembleConfig
    except ImportError:
        print("경고: ensemble_config.py를 찾을 수 없습니다. 기본 설정을 사용합니다.")
        EnsembleConfig = None


# Sentence Transformers (ko-sroberta-multitask용)
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"경고: sentence-transformers를 로드할 수 없습니다: {e}")
    print("  필요한 패키지: sentence-transformers")
    print("  설치 명령: pip install sentence-transformers")
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
    util = None
except Exception as e:
    print(f"경고: sentence-transformers 로드 중 오류 발생: {e}")
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None
    util = None


class EnsembleModel:
    """이종 앙상블 모델 클래스"""
    
    def __init__(self, 
                 latest_model_path: str = None,
                 use_sentiment: bool = True,
                 use_kor_unsmile: bool = True,
                 use_ko_sroberta: bool = True,
                 weights: Dict[str, float] = None,
                 config: EnsembleConfig = None,
                 bad_sentences_db: Optional[List[str]] = None,
                 silent: bool = False):
        """
        앙상블 모델 초기화
        
        Args:
            latest_model_path: 최신 학습 모델 경로 (None이면 설정 파일 또는 자동 탐지)
            use_sentiment: Sentiment 모델 사용 여부
            use_kor_unsmile: kor_unsmile 모델 사용 여부
            use_ko_sroberta: ko-sroberta-multitask 모델 사용 여부
            weights: 각 모델의 가중치 (None이면 설정 파일 또는 기본값)
            config: EnsembleConfig 객체 (None이면 기본 설정 사용)
            bad_sentences_db: 욕설 데이터베이스 (ko-sroberta 유사도 측정용)
            silent: True면 초기화 메시지 출력 안 함
        """
        # 설정 파일에서 값 가져오기
        if config is None and EnsembleConfig is not None:
            config = EnsembleConfig()
        
        if config:
            latest_model_path = latest_model_path or config.latest_model_path
            # config가 있으면 config의 값을 우선 사용 (명시적으로 False로 설정된 경우 반영)
            use_sentiment = config.use_sentiment
            use_kor_unsmile = config.use_kor_unsmile
            use_ko_sroberta = config.use_ko_sroberta
            weights = weights or config.weights
        
        # 욕설 데이터베이스 초기화 (ko-sroberta용)
        if bad_sentences_db is None:
            # 기본 욕설 예시 (실제로는 더 많은 데이터가 필요)
            bad_sentences_db = [
                "꺼져라", "죽어라", "노답이네", "멍청한 놈", "바보 같은",
                "미친놈", "개새끼", "병신", "좆같은", "시발", "씨발",
                "개소리", "헛소리", "쓰레기", "인간 쓰레기", "찌질이"
            ]
        self.bad_sentences_db = bad_sentences_db
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}  # TextClassificationPipeline용
        self.embedders = {}  # SentenceTransformer용
        self.weights = weights or {}
        
        # 사용할 모델 목록 확인
        models_to_load = []
        if config and config.use_latest_model:
            models_to_load.append('latest')
        if use_sentiment:
            models_to_load.append('sentiment')
        if use_kor_unsmile:
            models_to_load.append('kor_unsmile')
        if use_ko_sroberta and SENTENCE_TRANSFORMERS_AVAILABLE and SentenceTransformer is not None:
            models_to_load.append('ko_sroberta')
        
        if models_to_load and not silent:
            print("=" * 60)
            print("앙상블 모델 초기화 중...")
            print("=" * 60)
        
        model_count = 0
        total_models = len(models_to_load)
        
        # 1. 최신 학습 모델 로드 (use_latest_model이 True일 때만)
        if 'latest' in models_to_load:
            model_count += 1
            if latest_model_path is None:
                latest_model_path = self._find_latest_model()
            elif latest_model_path:
                # 상대 경로를 절대 경로로 변환
                if not os.path.isabs(latest_model_path):
                    latest_model_path = os.path.join(project_root, latest_model_path)
                
                if not os.path.exists(latest_model_path):
                    print(f"경고: 지정된 모델 경로가 없습니다: {latest_model_path}")
                    print("자동으로 최신 모델을 찾습니다...")
                    latest_model_path = self._find_latest_model()
            
            if latest_model_path and os.path.exists(latest_model_path):
                if not silent:
                    print(f"\n[{model_count}/{total_models}] 최신 학습 모델 로딩: {latest_model_path}")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(latest_model_path)
                    model = AutoModelForSequenceClassification.from_pretrained(latest_model_path)
                    model.eval()
                    model.to(self.device)
                    self.models['latest'] = model
                    self.tokenizers['latest'] = tokenizer
                    self.weights['latest'] = self.weights.get('latest', 0.5)  # 기본 가중치
                    if not silent:
                        print("  ✓ 최신 학습 모델 로드 완료")
                except Exception as e:
                    if not silent:
                        print(f"  ✗ 최신 학습 모델 로드 실패: {e}")
            else:
                if not silent:
                    print(f"\n[{model_count}/{total_models}] 최신 학습 모델을 찾을 수 없습니다: {latest_model_path}")
        
        # 2. Sentiment 모델 로드 (비활성화됨 - use_sentiment=False)
        if 'sentiment' in models_to_load:
            model_count += 1
            # Note: sentiment 모델은 현재 사용하지 않음 (use_sentiment=False)
            sentiment_path = os.path.join(project_root, 'ensembleModel', 'multilingual-sentiment-analysis')
            if os.path.exists(sentiment_path):
                if not silent:
                    print(f"\n[{model_count}/{total_models}] Sentiment 모델 로딩: {sentiment_path}")
                try:
                    tokenizer = AutoTokenizer.from_pretrained(sentiment_path)
                    model = AutoModelForSequenceClassification.from_pretrained(sentiment_path)
                    model.eval()
                    model.to(self.device)
                    self.models['sentiment'] = model
                    self.tokenizers['sentiment'] = tokenizer
                    self.weights['sentiment'] = self.weights.get('sentiment', 0.2)  # 기본 가중치
                    if not silent:
                        print("  ✓ Sentiment 모델 로드 완료")
                except Exception as e:
                    if not silent:
                        print(f"  ✗ Sentiment 모델 로드 실패: {e}")
            else:
                if not silent:
                    print(f"\n[{model_count}/{total_models}] Sentiment 모델 경로를 찾을 수 없습니다: {sentiment_path}")
        
        # 3. kor_unsmile 모델 로드
        if 'kor_unsmile' in models_to_load:
            model_count += 1
            if not silent:
                print(f"\n[{model_count}/{total_models}] kor_unsmile 모델 로딩...")
            try:
                model_name = 'smilegate-ai/kor_unsmile'
                model = BertForSequenceClassification.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model.eval()
                model.to(self.device)
                
                pipe = TextClassificationPipeline(
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if torch.cuda.is_available() else -1,
                    top_k=None,  # 모든 점수 반환 (return_all_scores 대신 사용)
                    function_to_apply='sigmoid',
                    max_length=300,  # kor_unsmile 모델의 최대 시퀀스 길이
                    truncation=True  # 길이 초과 시 자르기
                )
                self.pipelines['kor_unsmile'] = pipe
                self.weights['kor_unsmile'] = self.weights.get('kor_unsmile', 0.3)  # 기본 가중치
                if not silent:
                    print("  ✓ kor_unsmile 모델 로드 완료")
            except Exception as e:
                if not silent:
                    print(f"  ✗ kor_unsmile 모델 로드 실패: {e}")
                    import traceback
                    traceback.print_exc()
        
        # 4. ko-sroberta-multitask 모델 로드
        if 'ko_sroberta' in models_to_load:
            model_count += 1
            if not silent:
                print(f"\n[{model_count}/{total_models}] ko-sroberta-multitask 모델 로딩...")
            try:
                embedder = SentenceTransformer('jhgan/ko-sroberta-multitask')
                self.embedders['ko_sroberta'] = embedder
                
                # 욕설 데이터베이스 임베딩 미리 계산
                if self.bad_sentences_db:
                    if not silent:
                        print(f"  욕설 데이터베이스 임베딩 계산 중... ({len(self.bad_sentences_db)}개)")
                    self.bad_embeddings = embedder.encode(self.bad_sentences_db, show_progress_bar=False)
                    if not silent:
                        print("  ✓ 욕설 데이터베이스 임베딩 완료")
                
                self.weights['ko_sroberta'] = self.weights.get('ko_sroberta', 0.2)  # 기본 가중치
                if not silent:
                    print("  ✓ ko-sroberta-multitask 모델 로드 완료")
            except Exception as e:
                if not silent:
                    print(f"  ✗ ko-sroberta-multitask 모델 로드 실패: {e}")
                    import traceback
                    traceback.print_exc()
        
        # 실제 로드된 모델 목록 확인
        available_models = list(self.models.keys())
        if self.pipelines:
            available_models.extend(list(self.pipelines.keys()))
        if self.embedders:
            available_models.extend(list(self.embedders.keys()))
        
        # 가중치를 실제 로드된 모델에 맞게 조정
        if self.weights:
            # 실제 로드된 모델에 없는 가중치 제거
            self.weights = {k: v for k, v in self.weights.items() if k in available_models}
            # 가중치 정규화
            total_weight = sum(self.weights.values())
            if total_weight > 0:
                self.weights = {k: v / total_weight for k, v in self.weights.items()}
            else:
                # 가중치가 없으면 균등 분배
                if available_models:
                    self.weights = {model: 1.0 / len(available_models) for model in available_models}
        else:
            # 가중치가 없으면 균등 분배
            if available_models:
                self.weights = {model: 1.0 / len(available_models) for model in available_models}
        
        if not silent:
            print("\n" + "=" * 60)
            print(f"앙상블 모델 초기화 완료!")
            print(f"사용 가능한 모델: {available_models}")
            print(f"가중치: {self.weights}")
            print("=" * 60)
    
    def _find_latest_model(self) -> str:
        """가장 최근 학습 모델 경로 찾기"""
        results_dir = os.path.join(project_root, 'results')
        if not os.path.exists(results_dir):
            return None
        
        # run_* 디렉토리 찾기
        run_dirs = [d for d in os.listdir(results_dir) if d.startswith('run_') and os.path.isdir(os.path.join(results_dir, d))]
        if not run_dirs:
            return None
        
        # 가장 최근 디렉토리 찾기 (타임스탬프 기준)
        latest_dir = sorted(run_dirs, reverse=True)[0]
        model_path = os.path.join(results_dir, latest_dir, 'model')
        
        if os.path.exists(model_path):
            return model_path
        return None
    
    def predict_latest_model(self, text: str) -> Tuple[int, float, np.ndarray]:
        """최신 학습 모델로 예측"""
        if 'latest' not in self.models:
            return None, None, None
        
        tokenizer = self.tokenizers['latest']
        model = self.models['latest']
        
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            probs = probs.cpu().numpy()[0]
        
        label = int(np.argmax(probs))
        confidence = float(probs[label])
        
        return label, confidence, probs
    
    def predict_sentiment(self, text: str) -> Tuple[int, float, np.ndarray]:
        """Sentiment 모델로 예측 (Negative 감정을 혐오 표현으로 변환)"""
        if 'sentiment' not in self.models:
            return None, None, None
        
        tokenizer = self.tokenizers['sentiment']
        model = self.models['sentiment']
        
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            probs = probs.cpu().numpy()[0]
        
        # 감정 분석 결과를 혐오 표현 필터링으로 변환
        # Very Negative(0) + Negative(1) = 혐오 표현(1)
        # Neutral(2) + Positive(3) + Very Positive(4) = 정상(0)
        prob_hate = float(probs[0] + probs[1])  # Very Negative + Negative
        prob_normal = float(probs[2] + probs[3] + probs[4])  # Neutral + Positive + Very Positive
        
        converted_probs = np.array([prob_normal, prob_hate])
        label = int(np.argmax(converted_probs))
        confidence = float(converted_probs[label])
        
        return label, confidence, converted_probs
    
    def predict_kor_unsmile(self, text: str) -> Tuple[int, float, np.ndarray]:
        """kor_unsmile 모델로 예측 (10가지 카테고리별 확률)"""
        if 'kor_unsmile' not in self.pipelines:
            return None, None, None
        
        try:
            pipe = self.pipelines['kor_unsmile']
            results = pipe(text)
            
            # 결과 형식 확인 및 정규화
            # return_all_scores=True일 때 결과가 중첩 리스트일 수 있음
            if isinstance(results, list) and len(results) > 0:
                # 첫 번째 요소가 리스트인 경우 (중첩 리스트)
                if isinstance(results[0], list):
                    results = results[0]
                # 첫 번째 요소가 딕셔너리가 아닌 경우 처리
                elif not isinstance(results[0], dict):
                    # 예상치 못한 형식
                    print(f"경고: kor_unsmile 결과 형식이 예상과 다릅니다: {type(results[0])}")
                    return None, None, None
            
            # 결과는 [{'label': '여성/가족', 'score': 0.01}, ...] 형태
            # kor_unsmile은 sigmoid 기반 다중 레이블 분류 모델
            # clean 점수가 낮고 다른 혐오 카테고리 점수가 높으면 혐오로 간주
            
            clean_score = 0.0
            max_hate_score = 0.0
            total_hate_score = 0.0
            hate_count = 0
            
            # 혐오 관련 카테고리들 (kor_unsmile의 10가지 카테고리)
            hate_categories = [
                '여성/가족', '남성', '성소수자', '인종/국적', '연령',
                '지역', '종교', '기타 혐오', '악플/욕설'
            ]
            
            for item in results:
                if not isinstance(item, dict):
                    continue
                    
                label = item.get('label', '')
                score = item.get('score', 0.0)
                
                if label == 'clean':
                    clean_score = score
                elif label in hate_categories:
                    max_hate_score = max(max_hate_score, score)
                    total_hate_score += score
                    hate_count += 1
            
            # 혐오 확률 계산 (개선된 로직)
            # kor_unsmile은 sigmoid 기반 다중 레이블 분류 모델
            # clean 점수가 낮을수록 혐오일 가능성이 높음
            # 방법 1: clean 점수 기반 (더 단순하고 명확)
            prob_hate = 1.0 - clean_score
            
            # 방법 2: 혐오 카테고리 최대값과 clean 점수 비교 (대안)
            # 혐오 카테고리 중 하나라도 높은 점수를 가지면 혐오로 간주
            if max_hate_score > 0.5:
                # 혐오 카테고리가 높으면 clean 점수와 상관없이 혐오 가능성 증가
                prob_hate = max(prob_hate, max_hate_score)
            
            # 확률 범위 제한
            prob_hate = max(0.0, min(1.0, prob_hate))
            
            prob_normal = 1.0 - prob_hate
            
            probs = np.array([prob_normal, prob_hate])
            label = int(np.argmax(probs))
            confidence = float(probs[label])
            
            return label, confidence, probs
        except Exception as e:
            print(f"kor_unsmile 예측 오류: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def predict_ko_sroberta(self, text: str) -> Tuple[int, float, np.ndarray]:
        """ko-sroberta-multitask로 유사도 기반 예측"""
        if 'ko_sroberta' not in self.embedders or not hasattr(self, 'bad_embeddings'):
            return None, None, None
        
        try:
            embedder = self.embedders['ko_sroberta']
            input_embedding = embedder.encode(text, show_progress_bar=False)
            
            # 코사인 유사도 계산
            if util is not None:
                scores = util.cos_sim(input_embedding, self.bad_embeddings)
                max_similarity = float(scores.max().item())
            else:
                # util이 없는 경우 직접 계산
                from sklearn.metrics.pairwise import cosine_similarity
                scores = cosine_similarity([input_embedding], self.bad_embeddings)
                max_similarity = float(scores.max())
            
            # 유사도가 높으면 혐오 표현일 가능성이 높음
            # 임계값 조정: 0.5 이상이면 혐오로 간주
            threshold = 0.5
            
            # 유사도를 확률로 변환
            # 유사도가 임계값 이상이면 혐오 확률을 높게 설정
            if max_similarity >= threshold:
                # 임계값 이상: 유사도에 따라 확률 증가 (0.5 ~ 1.0)
                # 유사도 0.5 → 확률 0.5, 유사도 1.0 → 확률 1.0
                prob_hate = 0.5 + (max_similarity - threshold) * (0.5 / (1.0 - threshold))
                prob_hate = min(1.0, prob_hate)
            else:
                # 임계값 미만: 유사도에 따라 낮은 확률 (0.0 ~ 0.5)
                # 유사도 0.0 → 확률 0.0, 유사도 0.5 → 확률 0.5
                prob_hate = max_similarity * (0.5 / threshold)
            
            prob_normal = 1.0 - prob_hate
            probs = np.array([prob_normal, prob_hate])
            label = int(np.argmax(probs))
            confidence = float(probs[label])
            
            return label, confidence, probs
        except Exception as e:
            print(f"ko-sroberta 예측 오류: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def predict_ensemble(self, text: str) -> Tuple[int, float, Dict[str, Tuple[int, float]]]:
        """
        앙상블 예측 수행
        
        Returns:
            (최종 레이블, 최종 신뢰도, 각 모델별 예측 결과)
        """
        predictions = {}
        weighted_probs = np.zeros(2)  # [정상, 혐오]
        
        # 각 모델로 예측
        if 'latest' in self.models:
            label, conf, probs = self.predict_latest_model(text)
            if probs is not None:
                predictions['latest'] = (label, conf)
                weighted_probs += probs * self.weights['latest']
        
        if 'sentiment' in self.models:
            label, conf, probs = self.predict_sentiment(text)
            if probs is not None:
                predictions['sentiment'] = (label, conf)
                weighted_probs += probs * self.weights['sentiment']
        
        if 'kor_unsmile' in self.pipelines:
            label, conf, probs = self.predict_kor_unsmile(text)
            if probs is not None:
                predictions['kor_unsmile'] = (label, conf)
                weighted_probs += probs * self.weights['kor_unsmile']
        
        if 'ko_sroberta' in self.embedders:
            label, conf, probs = self.predict_ko_sroberta(text)
            if probs is not None:
                predictions['ko_sroberta'] = (label, conf)
                weighted_probs += probs * self.weights['ko_sroberta']
        
        # 가중 평균으로 최종 예측
        if len(predictions) == 0:
            return None, None, {}
        
        # 확률 정규화 (0으로 나누기 방지)
        prob_sum = weighted_probs.sum()
        if prob_sum > 0:
            weighted_probs = weighted_probs / prob_sum
        else:
            # 모든 확률이 0인 경우 균등 분포로 설정
            weighted_probs = np.array([0.5, 0.5])
        
        final_label = int(np.argmax(weighted_probs))
        final_confidence = float(weighted_probs[final_label])
        
        return final_label, final_confidence, predictions


def main():
    """메인 함수"""
    print("=" * 60)
    print("이종 앙상블 모델 예측 시스템")
    print("=" * 60)
    
    # 설정 파일 로드
    if EnsembleConfig:
        config = EnsembleConfig()
        print(f"\n설정 파일에서 모델 경로 로드: {config.latest_model_path}")
    else:
        config = None
        print("\n기본 설정 사용")
    
    # 앙상블 모델 초기화
    ensemble = EnsembleModel(config=config)
    
    if len(ensemble.models) == 0:
        print("\n경고: 사용 가능한 모델이 없습니다.")
        return
    
    # 디바이스 정보
    print(f"\n사용 중인 디바이스: {ensemble.device}")
    
    # 테스트 텍스트
    test_texts = [
        "안전한 콘텐츠입니다.",
        "부적절한 내용이 포함되어 있습니다.",
        "정상적인 메시지입니다.",
        "이건 정말 나쁜 말입니다."
    ]
    
    label_names = {0: "허용", 1: "차단"}
    
    print("\n" + "=" * 60)
    print("테스트 예측 결과")
    print("=" * 60)
    
    for text in test_texts:
        print(f"\n텍스트: {text}")
        print("-" * 60)
        
        final_label, final_conf, predictions = ensemble.predict_ensemble(text)
        
        if final_label is not None:
            print(f"최종 예측: {label_names[final_label]} (신뢰도: {final_conf:.4f})")
            print(f"\n각 모델별 예측:")
            for model_name, (label, conf) in predictions.items():
                print(f"  - {model_name}: {label_names[label]} (신뢰도: {conf:.4f})")
        else:
            print("예측 실패")
    
    print("\n" + "=" * 60)
    
    # 대화형 모드
    print("\n대화형 모드 (종료하려면 'quit' 또는 'exit' 입력)")
    while True:
        try:
            user_input = input("\n텍스트 입력: ").strip()
            if user_input.lower() in ['quit', 'exit', '종료']:
                break
            if not user_input:
                continue
            
            final_label, final_conf, predictions = ensemble.predict_ensemble(user_input)
            
            if final_label is not None:
                print(f"\n최종 예측: {label_names[final_label]} (신뢰도: {final_conf:.4f})")
                print(f"각 모델별 예측:")
                for model_name, (label, conf) in predictions.items():
                    print(f"  - {model_name}: {label_names[label]} (신뢰도: {conf:.4f})")
            else:
                print("예측 실패")
        except KeyboardInterrupt:
            print("\n\n종료합니다.")
            break
        except Exception as e:
            print(f"오류 발생: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

