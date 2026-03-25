"""
앙상블 모델 데이터셋 평가 스크립트
앙상블 모델로 데이터셋의 정확도, F1, Precision, Recall 등을 계산
"""
import os
import sys
import json
import time
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

# 프로젝트 루트 경로 설정
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

# 앙상블 모델 import
try:
    from scripts.ensemble_predict import EnsembleModel
    from scripts.ensemble_config import EnsembleConfig
except ImportError:
    # scripts 디렉토리에서 직접 실행하는 경우
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ensemble_predict import EnsembleModel
    from ensemble_config import EnsembleConfig

# 데이터 로더 import
try:
    from scripts.load_data import load_from_csv
    from datasets import Dataset
    import pandas as pd
except ImportError:
    # scripts 디렉토리에서 직접 실행하는 경우
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from load_data import load_from_csv
    from datasets import Dataset
    import pandas as pd


def load_tsv_dataset(file_path: str, text_column: str = 'text', label_column: str = 'label'):
    """
    TSV 파일에서 데이터셋 로드
    
    Args:
        file_path: TSV 파일 경로
        text_column: 텍스트 컬럼 이름
        label_column: 레이블 컬럼 이름
    
    Returns:
        Dataset 객체
    """
    print(f"TSV 데이터셋 로딩 중...")
    print(f"  파일 경로: {file_path}")
    
    # TSV 파일 읽기
    df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
    print(f"  총 데이터: {len(df)}개")
    
    # 컬럼 확인
    if text_column not in df.columns:
        raise ValueError(f"텍스트 컬럼 '{text_column}'을 찾을 수 없습니다. 사용 가능한 컬럼: {list(df.columns)}")
    if label_column not in df.columns:
        raise ValueError(f"레이블 컬럼 '{label_column}'을 찾을 수 없습니다. 사용 가능한 컬럼: {list(df.columns)}")
    
    # 데이터셋 생성
    dataset = Dataset.from_pandas(df[[text_column, label_column]].rename(columns={text_column: 'text', label_column: 'label'}))
    
    print(f"  ✓ 로드 완료: {len(dataset)}개")
    return dataset


def evaluate_single_model(
    ensemble: EnsembleModel,
    model_name: str,
    dataset,
    dataset_name: str = "Unknown",
    batch_size: int = 32,
    max_samples: Optional[int] = None
) -> Dict:
    """
    개별 모델로 데이터셋 평가
    
    Args:
        ensemble: EnsembleModel 객체
        model_name: 평가할 모델 이름 ('latest', 'best_previous', 'kor_unsmile')
        dataset: 평가할 데이터셋 (text, label 컬럼 필요)
        dataset_name: 데이터셋 이름
        batch_size: 배치 크기
        max_samples: 최대 평가 샘플 수 (None이면 전체)
    
    Returns:
        평가 결과 딕셔너리
    """
    # 데이터 준비
    texts = list(dataset['text'])
    true_labels = list(dataset['label'])
    
    total_samples_original = len(texts)
    
    # 샘플 수 제한 (평가하기 적절한 균형 비율로 샘플링)
    if max_samples and len(texts) > max_samples:
        # 레이블별로 인덱스 분리
        label_0_indices = [i for i, label in enumerate(true_labels) if label == 0]
        label_1_indices = [i for i, label in enumerate(true_labels) if label == 1]
        
        # 평가하기 적절한 균형 비율로 샘플링 (50:50)
        balanced_ratio = 0.5  # 50:50 균형
        label_0_samples = int(max_samples * balanced_ratio)
        label_1_samples = max_samples - label_0_samples  # 나머지는 label_1
        
        # 각 레이블에서 사용 가능한 샘플 수 확인
        available_label_0 = len(label_0_indices)
        available_label_1 = len(label_1_indices)
        
        # 사용 가능한 샘플 수가 부족한 경우 조정
        if available_label_0 < label_0_samples:
            label_0_samples = available_label_0
            label_1_samples = min(max_samples - label_0_samples, available_label_1)
        elif available_label_1 < label_1_samples:
            label_1_samples = available_label_1
            label_0_samples = min(max_samples - label_1_samples, available_label_0)
        
        # 각 레이블에서 랜덤 샘플링
        import random
        random.seed(42)  # 재현성을 위한 시드 설정
        
        if available_label_0 > label_0_samples:
            sampled_label_0_indices = random.sample(label_0_indices, label_0_samples)
        else:
            sampled_label_0_indices = label_0_indices
        
        if available_label_1 > label_1_samples:
            sampled_label_1_indices = random.sample(label_1_indices, label_1_samples)
        else:
            sampled_label_1_indices = label_1_indices
        
        # 인덱스 합치고 섞기
        sampled_indices = sampled_label_0_indices + sampled_label_1_indices
        random.shuffle(sampled_indices)
        
        # 샘플링된 데이터 추출
        texts = [texts[i] for i in sampled_indices]
        true_labels = [true_labels[i] for i in sampled_indices]
    
    total_samples = len(texts)
    
    # 예측 수행
    predicted_labels = []
    
    for i in range(0, total_samples, batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_labels = []
        
        for text in batch_texts:
            try:
                if model_name == 'latest' or model_name == 'best_previous':
                    label, _, _ = ensemble.predict_latest_model(str(text))
                elif model_name == 'kor_unsmile':
                    label, _, _ = ensemble.predict_kor_unsmile(str(text))
                else:
                    label = 0
                
                if label is not None:
                    batch_labels.append(label)
                else:
                    batch_labels.append(0)
            except Exception:
                batch_labels.append(0)
        
        predicted_labels.extend(batch_labels)
    
    # 메트릭 계산
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1_weighted = f1_score(true_labels, predicted_labels, average='weighted')
    f1_macro = f1_score(true_labels, predicted_labels, average='macro')
    f1_binary = f1_score(true_labels, predicted_labels, average='binary', pos_label=1)
    precision_weighted = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    precision_macro = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
    recall_weighted = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    recall_macro = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
    
    # 클래스별 메트릭 계산
    from sklearn.metrics import precision_recall_fscore_support
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=[0, 1], zero_division=0
    )
    
    # Confusion Matrix
    try:
        cm = confusion_matrix(true_labels, predicted_labels)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
    except Exception:
        tn, fp, fn, tp = 0, 0, 0, 0
    
    return {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'total_samples': total_samples,
        'accuracy': float(accuracy),
        'f1_weighted': float(f1_weighted),
        'f1_macro': float(f1_macro),
        'f1_binary': float(f1_binary),
        'precision_weighted': float(precision_weighted),
        'precision_macro': float(precision_macro),
        'recall_weighted': float(recall_weighted),
        'recall_macro': float(recall_macro),
        'precision_per_class': {
            '0': float(precision_per_class[0]),
            '1': float(precision_per_class[1])
        },
        'recall_per_class': {
            '0': float(recall_per_class[0]),
            '1': float(recall_per_class[1])
        },
        'f1_per_class': {
            '0': float(f1_per_class[0]),
            '1': float(f1_per_class[1])
        },
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        }
    }


def evaluate_ensemble_on_dataset(
    ensemble: EnsembleModel,
    dataset,
    dataset_name: str = "Unknown",
    batch_size: int = 32,
    max_samples: Optional[int] = None
) -> Dict:
    """
    앙상블 모델로 데이터셋 평가
    
    Args:
        ensemble: EnsembleModel 객체
        dataset: 평가할 데이터셋 (text, label 컬럼 필요)
        dataset_name: 데이터셋 이름
        batch_size: 배치 크기
        max_samples: 최대 평가 샘플 수 (None이면 전체)
    
    Returns:
        평가 결과 딕셔너리
    """
    print(f"\n{'='*60}")
    print(f"데이터셋 평가: {dataset_name}")
    print(f"{'='*60}")
    
    # 데이터 준비 (Column 객체를 list로 변환)
    texts = list(dataset['text'])
    true_labels = list(dataset['label'])
    
    total_samples_original = len(texts)
    
    # 샘플 수 제한 (평가하기 적절한 균형 비율로 샘플링)
    if max_samples and total_samples_original > max_samples:
        print(f"  샘플 수 제한: {total_samples_original:,} → {max_samples:,}")
        
        # 레이블별로 인덱스 분리
        label_0_indices = [i for i, label in enumerate(true_labels) if label == 0]
        label_1_indices = [i for i, label in enumerate(true_labels) if label == 1]
        
        # 원본 비율 계산 (정보용)
        original_label_0_ratio = len(label_0_indices) / total_samples_original
        original_label_1_ratio = len(label_1_indices) / total_samples_original
        
        print(f"  원본 레이블 분포: 클래스 0={original_label_0_ratio*100:.1f}%, 클래스 1={original_label_1_ratio*100:.1f}%")
        
        # 평가하기 적절한 균형 비율로 샘플링 (50:50)
        balanced_ratio = 0.5  # 50:50 균형
        label_0_samples = int(max_samples * balanced_ratio)
        label_1_samples = max_samples - label_0_samples  # 나머지는 label_1
        
        # 각 레이블에서 사용 가능한 샘플 수 확인
        available_label_0 = len(label_0_indices)
        available_label_1 = len(label_1_indices)
        
        # 사용 가능한 샘플 수가 부족한 경우 조정
        if available_label_0 < label_0_samples:
            label_0_samples = available_label_0
            label_1_samples = min(max_samples - label_0_samples, available_label_1)
        elif available_label_1 < label_1_samples:
            label_1_samples = available_label_1
            label_0_samples = min(max_samples - label_1_samples, available_label_0)
        
        print(f"  균형 샘플링 (50:50):")
        print(f"    - 클래스 0: {label_0_samples:,}개 (50.0%)")
        print(f"    - 클래스 1: {label_1_samples:,}개 (50.0%)")
        
        # 각 레이블에서 랜덤 샘플링
        import random
        random.seed(42)  # 재현성을 위한 시드 설정
        
        if available_label_0 > label_0_samples:
            sampled_label_0_indices = random.sample(label_0_indices, label_0_samples)
        else:
            sampled_label_0_indices = label_0_indices
        
        if available_label_1 > label_1_samples:
            sampled_label_1_indices = random.sample(label_1_indices, label_1_samples)
        else:
            sampled_label_1_indices = label_1_indices
        
        # 인덱스 합치고 섞기
        sampled_indices = sampled_label_0_indices + sampled_label_1_indices
        random.shuffle(sampled_indices)
        
        # 샘플링된 데이터 추출
        texts = [texts[i] for i in sampled_indices]
        true_labels = [true_labels[i] for i in sampled_indices]
        
        actual_label_0 = true_labels.count(0)
        actual_label_1 = true_labels.count(1)
        actual_total = len(true_labels)
        print(f"  샘플링 완료: 클래스 0={actual_label_0:,}개 ({actual_label_0/actual_total*100:.1f}%), 클래스 1={actual_label_1:,}개 ({actual_label_1/actual_total*100:.1f}%)")
    else:
        if max_samples:
            print(f"  전체 샘플 사용: {total_samples_original:,}개 (제한: {max_samples:,}개)")
        else:
            print(f"  전체 샘플 사용: {total_samples_original:,}개")
    
    total_samples = len(texts)
    label_0_count = true_labels.count(0)
    label_1_count = true_labels.count(1)
    label_0_ratio = label_0_count / total_samples * 100
    label_1_ratio = label_1_count / total_samples * 100
    
    print(f"  총 샘플 수: {total_samples:,}개")
    print(f"  레이블 분포: 0={label_0_count:,}개 ({label_0_ratio:.1f}%), 1={label_1_count:,}개 ({label_1_ratio:.1f}%)")
    
    # 예측 수행
    print(f"\n  예측 진행 중...")
    start_time = time.time()
    
    predicted_labels = []
    predicted_probs = []
    
    for i in range(0, total_samples, batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_labels = []
        batch_probs = []
        
        for text in batch_texts:
            try:
                label, confidence, _ = ensemble.predict_ensemble(str(text))  # str로 변환
                if label is not None:
                    batch_labels.append(label)
                    batch_probs.append(confidence)
                else:
                    # 예측 실패 시 기본값 (정상으로 분류)
                    batch_labels.append(0)
                    batch_probs.append(0.5)
            except Exception as e:
                # 예측 중 에러 발생 시 기본값 사용
                import traceback
                error_msg = str(e)
                if len(error_msg) > 100:
                    error_msg = error_msg[:100] + "..."
                print(f"    경고: 예측 실패 (텍스트 인덱스 {len(predicted_labels)}): {error_msg}")
                batch_labels.append(0)
                batch_probs.append(0.5)
        
        predicted_labels.extend(batch_labels)
        predicted_probs.extend(batch_probs)
        
        # 진행 상황 출력
        if (i + batch_size) % (batch_size * 10) == 0 or (i + batch_size) >= total_samples:
            progress = min(i + batch_size, total_samples)
            print(f"    진행: {progress:,}/{total_samples:,} ({progress/total_samples*100:.1f}%)")
    
    elapsed_time = time.time() - start_time
    
    # 메트릭 계산
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1_weighted = f1_score(true_labels, predicted_labels, average='weighted')
    f1_macro = f1_score(true_labels, predicted_labels, average='macro')
    f1_binary = f1_score(true_labels, predicted_labels, average='binary', pos_label=1)
    precision_weighted = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    precision_macro = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
    recall_weighted = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    recall_macro = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
    
    # Confusion Matrix
    try:
        cm = confusion_matrix(true_labels, predicted_labels)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            # 이진 분류가 아닌 경우
            tn, fp, fn, tp = 0, 0, 0, 0
    except Exception as e:
        print(f"  경고: Confusion Matrix 계산 실패: {e}")
        tn, fp, fn, tp = 0, 0, 0, 0
    
    # 클래스별 Precision, Recall 계산
    from sklearn.metrics import precision_recall_fscore_support
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=[0, 1], zero_division=0
    )
    
    # 결과 출력
    print(f"\n  평가 완료! (소요 시간: {elapsed_time:.2f}초)")
    print(f"  처리 속도: {total_samples/elapsed_time:.1f} 샘플/초")
    print(f"\n  {'='*60}")
    print(f"  평가 결과")
    print(f"  {'='*60}")
    
    # 불균형 데이터셋에서는 Accuracy보다 다른 메트릭이 중요
    label_0_ratio = true_labels.count(0) / total_samples * 100
    if abs(label_0_ratio - 50) > 30:
        print(f"  ⚠️ 불균형 데이터셋: Accuracy보다 F1-Macro/Binary를 우선 확인하세요!")
        print(f"")
    
    print(f"  [전체 메트릭]")
    print(f"    정확도 (Accuracy): {accuracy*100:.2f}%")
    print(f"    F1 점수 (Weighted): {f1_weighted*100:.2f}%")
    print(f"    F1 점수 (Macro): {f1_macro*100:.2f}% ⭐ (불균형 데이터에 중요)")
    print(f"    F1 점수 (Binary): {f1_binary*100:.2f}% ⭐ (불균형 데이터에 중요)")
    print(f"    Precision (Weighted): {precision_weighted*100:.2f}%")
    print(f"    Precision (Macro): {precision_macro*100:.2f}%")
    print(f"    Recall (Weighted): {recall_weighted*100:.2f}%")
    print(f"    Recall (Macro): {recall_macro*100:.2f}%")
    
    print(f"\n  [클래스별 메트릭]")
    print(f"    클래스 0 (정상):")
    print(f"      Precision: {precision_per_class[0]*100:.2f}%")
    print(f"      Recall: {recall_per_class[0]*100:.2f}%")
    print(f"      F1: {f1_per_class[0]*100:.2f}%")
    print(f"      Support: {support_per_class[0]:,}개")
    print(f"    클래스 1 (혐오):")
    print(f"      Precision: {precision_per_class[1]*100:.2f}% ⭐")
    print(f"      Recall: {recall_per_class[1]*100:.2f}% ⭐")
    print(f"      F1: {f1_per_class[1]*100:.2f}% ⭐")
    print(f"      Support: {support_per_class[1]:,}개")
    
    print(f"\n  [Confusion Matrix]")
    print(f"    True Negative (TN): {tn:,}  (정상을 정상으로 예측)")
    print(f"    False Positive (FP): {fp:,}  (정상을 혐오로 오분류)")
    print(f"    False Negative (FN): {fn:,}  (혐오를 정상으로 오분류) ⚠️")
    print(f"    True Positive (TP): {tp:,}  (혐오를 혐오로 예측) ⭐")
    
    # 혐오 탐지율 계산
    if (fn + tp) > 0:
        hate_detection_rate = tp / (fn + tp) * 100
        print(f"\n  [혐오 탐지율]")
        print(f"    혐오 탐지율 (Recall for Hate): {hate_detection_rate:.2f}%")
        print(f"    → 실제 혐오 중 {hate_detection_rate:.2f}%를 올바르게 탐지")
    
    print(f"  {'='*60}")
    
    # 결과 딕셔너리
    results = {
        'dataset_name': dataset_name,
        'total_samples': total_samples,
        'elapsed_time': elapsed_time,
        'samples_per_second': total_samples / elapsed_time,
        'accuracy': float(accuracy),
        'f1_weighted': float(f1_weighted),
        'f1_macro': float(f1_macro),
        'f1_binary': float(f1_binary),
        'precision_weighted': float(precision_weighted),
        'precision_macro': float(precision_macro),
        'recall_weighted': float(recall_weighted),
        'recall_macro': float(recall_macro),
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        },
        'label_distribution': {
            '0': int(true_labels.count(0)),
            '1': int(true_labels.count(1))
        }
    }
    
    return results




def main():
    """메인 함수"""
    print("=" * 60)
    print("앙상블 모델 데이터셋 평가")
    print("=" * 60)
    
    # 모델 경로 설정
    best_previous_model_path = "results/run_20251120_104711/model"  # 이전 최고 성능 모델 (87.42%)
    
    # 필요한 모델들을 시작 시 미리 로드
    print(f"\n{'='*60}")
    print("필요한 모델 미리 로딩")
    print(f"{'='*60}")
    
    # 1. KOR_UNSMILE 모델 로드
    print(f"\n[1/3] KOR_UNSMILE 모델 로딩...")
    config_kor = EnsembleConfig()
    config_kor.use_latest_model = False
    config_kor.use_kor_unsmile = True
    config_kor.use_sentiment = False
    config_kor.use_ko_sroberta = False
    ensemble_kor = EnsembleModel(config=config_kor, silent=True)
    print("  ✓ KOR_UNSMILE 모델 로드 완료")
    
    # 2. 최신 학습 모델 로드
    print(f"\n[2/3] 최신 학습 모델 로딩...")
    config_latest = EnsembleConfig()
    config_latest.latest_model_path = None  # 자동 탐지
    config_latest.use_latest_model = True
    config_latest.use_kor_unsmile = False
    config_latest.use_sentiment = False
    config_latest.use_ko_sroberta = False
    ensemble_latest = EnsembleModel(config=config_latest, silent=True)
    print("  ✓ 최신 학습 모델 로드 완료")
    
    # 3. 이전 최고 성능 모델 로드
    print(f"\n[3/3] 이전 최고 성능 모델 로딩...")
    config_best = EnsembleConfig()
    config_best.latest_model_path = best_previous_model_path
    config_best.use_latest_model = True
    config_best.use_kor_unsmile = False
    config_best.use_sentiment = False
    config_best.use_ko_sroberta = False
    ensemble_best = EnsembleModel(config=config_best, silent=True)
    print("  ✓ 이전 최고 성능 모델 로드 완료")
    
    print(f"\n{'='*60}")
    print("모델 로딩 완료!")
    print(f"{'='*60}")
    
    # 데이터셋 로드 (data.tsv만 사용)
    print(f"\n{'='*60}")
    print("데이터셋 로딩")
    print(f"{'='*60}")
    dataset_path = 'datasets/data.tsv'
    
    try:
        dataset = load_tsv_dataset(dataset_path, text_column='message_content', label_column='label')
        dataset_name = "data.tsv"
    except Exception as e:
        print(f"\n경고: 데이터셋 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 테스트 데이터 2만개로 제한
    total_size = len(dataset)
    max_samples = 20000
    
    print(f"\n{'='*60}")
    print(f"데이터셋 평가: {dataset_name}")
    print(f"  전체 크기: {total_size:,}개")
    print(f"  평가 샘플: {max_samples:,}개")
    print(f"{'='*60}")
    
    # 평가 결과 저장
    all_results = []  # 앙상블 결과
    all_individual_results = []  # 개별 모델 결과
    
    # 앙상블 1: KOR_UNSMILE + 최신 학습 모델 (이미 로드된 모델 조합)
    print(f"\n{'='*60}")
    print("앙상블 모델 평가")
    print(f"{'='*60}")
    
    try:
        print(f"\n[앙상블 1] KOR_UNSMILE + 최신 학습 모델 평가 중...")
        # 이미 로드된 모델들을 조합
        config_ensemble_1 = EnsembleConfig(
            use_latest_model=True,
            use_kor_unsmile=True,
            use_sentiment=False,
            use_ko_sroberta=False
        )
        ensemble_1 = EnsembleModel(config=config_ensemble_1, silent=True)
        # 이미 로드된 모델들을 복사
        ensemble_1.models = {**ensemble_latest.models, **ensemble_kor.models}
        ensemble_1.tokenizers = {**ensemble_latest.tokenizers, **ensemble_kor.tokenizers}
        ensemble_1.pipelines = {**ensemble_kor.pipelines}
        # EnsembleConfig에서 설정된 가중치 사용
        ensemble_1.weights = config_ensemble_1.weights.copy()
        # 가중치 정규화
        total_weight = sum(ensemble_1.weights.values())
        ensemble_1.weights = {k: v / total_weight for k, v in ensemble_1.weights.items()}
        print(f"  사용 모델: {list(ensemble_1.models.keys()) + list(ensemble_1.pipelines.keys())}")
        print(f"  가중치: {ensemble_1.weights}")
        
        ensemble_results_1 = evaluate_ensemble_on_dataset(
            ensemble=ensemble_1,
            dataset=dataset,
            dataset_name=dataset_name,
            batch_size=32,
            max_samples=max_samples
        )
        ensemble_results_1['model_type'] = 'ensemble'
        ensemble_results_1['ensemble_name'] = 'KOR_UNSMILE + 최신 학습 모델'
        all_results.append(ensemble_results_1)
        print(f"  ✓ 앙상블 1 평가 완료 (정확도: {ensemble_results_1['accuracy']*100:.2f}%)")
    except Exception as e:
        print(f"  ✗ 앙상블 1 평가 실패: {e}")
        import traceback
        traceback.print_exc()
    
    # 앙상블 2: KOR_UNSMILE + 이전 최고 성능 모델 (이미 로드된 모델 조합)
    try:
        print(f"\n[앙상블 2] KOR_UNSMILE + 이전 최고 성능 모델 평가 중...")
        # 이미 로드된 모델들을 조합
        config_ensemble_2 = EnsembleConfig(
            use_latest_model=True,
            use_kor_unsmile=True,
            use_sentiment=False,
            use_ko_sroberta=False
        )
        ensemble_2 = EnsembleModel(config=config_ensemble_2, silent=True)
        # 이미 로드된 모델들을 복사
        ensemble_2.models = {**ensemble_best.models, **ensemble_kor.models}
        ensemble_2.tokenizers = {**ensemble_best.tokenizers, **ensemble_kor.tokenizers}
        ensemble_2.pipelines = {**ensemble_kor.pipelines}
        # EnsembleConfig에서 설정된 가중치 사용 (best 모델도 'latest' 키 사용)
        ensemble_2.weights = config_ensemble_2.weights.copy()
        # 가중치 정규화
        total_weight = sum(ensemble_2.weights.values())
        ensemble_2.weights = {k: v / total_weight for k, v in ensemble_2.weights.items()}
        print(f"  사용 모델: {list(ensemble_2.models.keys()) + list(ensemble_2.pipelines.keys())}")
        print(f"  가중치: {ensemble_2.weights}")
        
        ensemble_results_2 = evaluate_ensemble_on_dataset(
            ensemble=ensemble_2,
            dataset=dataset,
            dataset_name=dataset_name,
            batch_size=32,
            max_samples=max_samples
        )
        ensemble_results_2['model_type'] = 'ensemble'
        ensemble_results_2['ensemble_name'] = 'KOR_UNSMILE + 이전 최고 성능 모델'
        all_results.append(ensemble_results_2)
        print(f"  ✓ 앙상블 2 평가 완료 (정확도: {ensemble_results_2['accuracy']*100:.2f}%)")
    except Exception as e:
        print(f"  ✗ 앙상블 2 평가 실패: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. 개별 모델 평가 (이미 로드된 모델 사용)
    print(f"\n{'='*60}")
    print("개별 모델 평가")
    print(f"{'='*60}")
    
    # KOR_UNSMILE
    try:
        print(f"\n[개별 모델 1] KOR_UNSMILE 평가 중...")
        kor_results = evaluate_single_model(
            ensemble=ensemble_kor,
            model_name='kor_unsmile',
            dataset=dataset,
            dataset_name=dataset_name,
            batch_size=32,
            max_samples=max_samples
        )
        kor_results['model_type'] = 'individual'
        all_individual_results.append(kor_results)
        print(f"  ✓ KOR_UNSMILE 평가 완료 (정확도: {kor_results['accuracy']*100:.2f}%)")
    except Exception as e:
        print(f"  ✗ KOR_UNSMILE 평가 실패: {e}")
        import traceback
        traceback.print_exc()
    
    # 최신 학습 모델
    try:
        print(f"\n[개별 모델 2] 최신 학습 모델 평가 중...")
        latest_results = evaluate_single_model(
            ensemble=ensemble_latest,
            model_name='latest',
            dataset=dataset,
            dataset_name=dataset_name,
            batch_size=32,
            max_samples=max_samples
        )
        latest_results['model_type'] = 'individual'
        all_individual_results.append(latest_results)
        print(f"  ✓ 최신 학습 모델 평가 완료 (정확도: {latest_results['accuracy']*100:.2f}%)")
    except Exception as e:
        print(f"  ✗ 최신 학습 모델 평가 실패: {e}")
        import traceback
        traceback.print_exc()
    
    # 이전 최고 성능 모델
    try:
        print(f"\n[개별 모델 3] 이전 최고 성능 모델 평가 중...")
        best_results = evaluate_single_model(
            ensemble=ensemble_best,
            model_name='best_previous',
            dataset=dataset,
            dataset_name=dataset_name,
            batch_size=32,
            max_samples=max_samples
        )
        best_results['model_type'] = 'individual'
        all_individual_results.append(best_results)
        print(f"  ✓ 이전 최고 성능 모델 평가 완료 (정확도: {best_results['accuracy']*100:.2f}%)")
    except Exception as e:
        print(f"  ✗ 이전 최고 성능 모델 평가 실패: {e}")
        import traceback
        traceback.print_exc()
    
    # 전체 결과 요약 및 비교
    if len(all_results) > 0 or len(all_individual_results) > 0:
        print(f"\n{'='*80}")
        print("전체 평가 결과 비교")
        print(f"{'='*80}")
        
        # 데이터셋별로 그룹화
        dataset_names = set()
        for r in all_results:
            dataset_names.add(r['dataset_name'])
        for r in all_individual_results:
            dataset_names.add(r['dataset_name'])
        
        dataset_names = sorted(list(dataset_names))
        
        # 각 데이터셋별로 비교 테이블 출력
        for dataset_name in dataset_names:
            print(f"\n{'='*80}")
            print(f"데이터셋: {dataset_name}")
            print(f"{'='*80}")
            print(f"{'모델':<40} {'정확도':<10} {'F1(Weighted)':<12} {'F1(Macro)':<12} {'Precision':<12} {'Recall':<12}")
            print("-" * 100)
            
            # 앙상블 결과 (2개)
            ensemble_results = [r for r in all_results if r['dataset_name'] == dataset_name]
            for r in ensemble_results:
                ensemble_name = r.get('ensemble_name', '앙상블')
                print(f"{ensemble_name:<40} {r['accuracy']*100:>8.2f}% {r['f1_weighted']*100:>10.2f}% {r['f1_macro']*100:>10.2f}% {r['precision_weighted']*100:>10.2f}% {r['recall_weighted']*100:>10.2f}%")
            
            # 개별 모델 결과
            individual_results = [r for r in all_individual_results if r['dataset_name'] == dataset_name]
            for r in sorted(individual_results, key=lambda x: x['model_name']):
                if r['model_name'] == 'latest':
                    model_name = '최신 학습 모델'
                elif r['model_name'] == 'best_previous':
                    model_name = '이전 최고 성능 모델'
                elif r['model_name'] == 'kor_unsmile':
                    model_name = 'KOR_UNSMILE'
                else:
                    model_name = r['model_name'].upper()
                print(f"{model_name:<40} {r['accuracy']*100:>8.2f}% {r['f1_weighted']*100:>10.2f}% {r['f1_macro']*100:>10.2f}% {r['precision_weighted']*100:>10.2f}% {r['recall_weighted']*100:>10.2f}%")
        
        print(f"{'='*80}")
        
        # 결과 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = os.path.join(project_root, 'results', 'Ensemble')
        os.makedirs(results_dir, exist_ok=True)
        
        # 파일명 생성
        base_filename = f'ensemble_evaluation_{timestamp}'
        json_file = os.path.join(results_dir, f'{base_filename}.json')
        summary_file = os.path.join(results_dir, f'{base_filename}_summary.txt')
        
        # 상세 설정 정보 수집
        ensemble_config_detail = {
            'ensemble_1': {
                'name': 'KOR_UNSMILE + 최신 학습 모델',
                'models': ['kor_unsmile', 'latest'],
                'weights': {'latest': 0.7, 'kor_unsmile': 0.3}
            },
            'ensemble_2': {
                'name': 'KOR_UNSMILE + 이전 최고 성능 모델',
                'models': ['kor_unsmile', 'best_previous'],
                'weights': {'latest': 0.7, 'kor_unsmile': 0.3},
                'best_previous_model_path': best_previous_model_path
            },
            'individual_models': ['kor_unsmile', 'latest', 'best_previous']
        }
        
        # 전체 요약 통계 계산
        total_samples = sum(r['total_samples'] for r in all_results) if all_results else 0
        ensemble_avg = {}
        individual_avg = {}
        
        if all_results:
            ensemble_avg = {
                'accuracy': float(sum(r['accuracy'] * r['total_samples'] for r in all_results) / total_samples),
                'f1_weighted': float(sum(r['f1_weighted'] * r['total_samples'] for r in all_results) / total_samples),
                'f1_macro': float(sum(r['f1_macro'] * r['total_samples'] for r in all_results) / total_samples),
                'precision_weighted': float(sum(r['precision_weighted'] * r['total_samples'] for r in all_results) / total_samples),
                'recall_weighted': float(sum(r['recall_weighted'] * r['total_samples'] for r in all_results) / total_samples)
            }
        
        if all_individual_results:
            model_names = set(r['model_name'] for r in all_individual_results)
            for model_name in model_names:
                model_results = [r for r in all_individual_results if r['model_name'] == model_name]
                model_total_samples = sum(r['total_samples'] for r in model_results)
                individual_avg[model_name] = {
                    'accuracy': float(sum(r['accuracy'] * r['total_samples'] for r in model_results) / model_total_samples),
                    'f1_weighted': float(sum(r['f1_weighted'] * r['total_samples'] for r in model_results) / model_total_samples),
                    'f1_macro': float(sum(r['f1_macro'] * r['total_samples'] for r in model_results) / model_total_samples),
                    'precision_weighted': float(sum(r['precision_weighted'] * r['total_samples'] for r in model_results) / model_total_samples),
                    'recall_weighted': float(sum(r['recall_weighted'] * r['total_samples'] for r in model_results) / model_total_samples)
                }
        
        # JSON 파일 저장
        output_data = {
            'timestamp': timestamp,
            'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'ensemble_config': ensemble_config_detail,
            'evaluation_settings': {
                'dataset_path': dataset_path,
                'max_samples': 20000,
                'batch_size': 32,
                'total_datasets': len(dataset_names),
                'datasets_evaluated': dataset_names
            },
            'ensemble_results': all_results,
            'individual_results': all_individual_results,
            'summary': {
                'total_datasets': len(dataset_names),
                'total_samples': total_samples,
                'ensemble_average': ensemble_avg,
                'individual_averages': individual_avg
            }
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        # 요약 텍스트 파일 저장
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("앙상블 모델 평가 결과 요약\n")
            f.write("=" * 80 + "\n")
            f.write(f"평가 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"타임스탬프: {timestamp}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("앙상블 설정\n")
            f.write("-" * 80 + "\n")
            f.write("앙상블 1: KOR_UNSMILE + 최신 학습 모델\n")
            f.write("  가중치: latest=0.7, kor_unsmile=0.3\n")
            f.write("앙상블 2: KOR_UNSMILE + 이전 최고 성능 모델\n")
            f.write(f"  모델 경로: {best_previous_model_path}\n")
            f.write("  가중치: latest=0.7, kor_unsmile=0.3\n")
            f.write("\n")
            
            f.write("-" * 80 + "\n")
            f.write("평가 설정\n")
            f.write("-" * 80 + "\n")
            f.write(f"데이터셋: {dataset_path}\n")
            f.write(f"최대 샘플 수: 20,000개\n")
            f.write(f"배치 크기: 32\n")
            f.write(f"평가 데이터셋 수: {len(dataset_names)}개\n")
            f.write(f"총 평가 샘플 수: {total_samples:,}개\n")
            f.write("\n")
            
            f.write("\n")
            f.write("-" * 80 + "\n")
            f.write("데이터셋별 상세 결과\n")
            f.write("-" * 80 + "\n")
            
            for dataset_name in dataset_names:
                f.write(f"\n데이터셋: {dataset_name}\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'모델':<40} {'정확도':<12} {'F1(Weighted)':<15} {'F1(Macro)':<15} {'Precision':<15} {'Recall':<15}\n")
                f.write("-" * 100 + "\n")
                
                # 앙상블 결과 (2개)
                ensemble_results = [r for r in all_results if r['dataset_name'] == dataset_name]
                for r in ensemble_results:
                    ensemble_name = r.get('ensemble_name', '앙상블')
                    f.write(f"{ensemble_name:<40} {r['accuracy']*100:>10.2f}% {r['f1_weighted']*100:>13.2f}% {r['f1_macro']*100:>13.2f}% {r['precision_weighted']*100:>13.2f}% {r['recall_weighted']*100:>13.2f}%\n")
                
                # 개별 모델 결과
                individual_results = [r for r in all_individual_results if r['dataset_name'] == dataset_name]
                for r in sorted(individual_results, key=lambda x: x['model_name']):
                    if r['model_name'] == 'latest':
                        model_name = '최신 학습 모델'
                    elif r['model_name'] == 'best_previous':
                        model_name = '이전 최고 성능 모델'
                    elif r['model_name'] == 'kor_unsmile':
                        model_name = 'KOR_UNSMILE'
                    else:
                        model_name = r['model_name'].upper()
                    f.write(f"{model_name:<40} {r['accuracy']*100:>10.2f}% {r['f1_weighted']*100:>13.2f}% {r['f1_macro']*100:>13.2f}% {r['precision_weighted']*100:>13.2f}% {r['recall_weighted']*100:>13.2f}%\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"\n결과가 저장되었습니다:")
        print(f"  JSON 파일: {json_file}")
        print(f"  요약 파일: {summary_file}")
    
    print("\n평가 완료!")


if __name__ == "__main__":
    main()

