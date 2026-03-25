"""
데이터 로드 유틸리티
다양한 형식의 데이터를 로드하는 헬퍼 함수들
"""
from datasets import Dataset
import pandas as pd
import json
from pathlib import Path

def load_from_csv(file_path, text_column='text', label_column='label'):
    """
    CSV 파일에서 데이터셋 로드
    
    Args:
        file_path: CSV 파일 경로
        text_column: 텍스트 컬럼 이름
        label_column: 레이블 컬럼 이름
    
    Returns:
        Dataset 객체
    """
    df = pd.read_csv(file_path)
    return Dataset.from_pandas(df)

def load_korean_hate_speech_dataset(train_path, dev_path=None, label_type='hate', combine_train_dev=False):
    """
    Korean Hate Speech 데이터셋 로드 (TSV 형식)
    
    Args:
        train_path: train.tsv 파일 경로
        dev_path: dev.tsv 파일 경로 (선택, None이면 train에서 분할)
        label_type: 사용할 레이블 타입
                   - 'hate': hate 컬럼 사용 (hate, offensive, none)
                   - 'bias': bias 컬럼 사용 (gender, others, none)
                   - 'gender_bias': contain_gender_bias 컬럼 사용 (True/False)
        combine_train_dev: True면 train과 dev를 합쳐서 반환 (기본: False, train만 반환)
    
    Returns:
        Dataset 객체 (combine_train_dev=True면 train+dev, False면 train만)
    """
    print(f"Korean Hate Speech 데이터셋 로딩 중...")
    print(f"  학습 파일: {train_path}")
    
    # TSV 파일 읽기
    train_df = pd.read_csv(train_path, sep='\t', encoding='utf-8')
    print(f"  학습 데이터: {len(train_df)}개")
    
    # 텍스트와 레이블 추출
    texts = train_df['comments'].tolist()
    
    # 레이블 타입에 따라 선택
    # 데이터셋에는 3가지 레이블이 있음:
    # 1. hate: 'hate', 'offensive', 'none' (혐오 표현 여부)
    # 2. bias: 'gender', 'others', 'none' (사회적 편향 여부)
    # 3. contain_gender_bias: True/False (성별 편향 포함 여부)
    # 
    # 현재는 'hate' 컬럼만 사용 (혐오 탐지 태스크에 적합)
    if label_type == 'hate':
        labels_raw = train_df['hate'].tolist()
        # hate, offensive → 1 (혐오/부적절), none → 0 (정상)
        # 참고: 'hate'는 명시적 혐오, 'offensive'는 부적절한 표현
        labels = [1 if label in ['hate', 'offensive'] else 0 for label in labels_raw]
        print(f"  레이블 타입: hate (hate/offensive=1, none=0)")
        print(f"  매핑 규칙: 'hate'와 'offensive'를 모두 혐오(1)로 분류")
    elif label_type == 'bias':
        labels_raw = train_df['bias'].tolist()
        # gender, others → 1, none → 0
        labels = [1 if label in ['gender', 'others'] else 0 for label in labels_raw]
        print(f"  레이블 타입: bias (gender/others=1, none=0)")
    elif label_type == 'gender_bias':
        labels_raw = train_df['contain_gender_bias'].tolist()
        # True → 1, False → 0
        labels = [1 if label else 0 for label in labels_raw]
        print(f"  레이블 타입: gender_bias (True=1, False=0)")
    else:
        raise ValueError(f"지원하지 않는 label_type: {label_type}")
    
    # dev 데이터가 있고 합치기 옵션이 켜져 있으면 합치기
    if dev_path and combine_train_dev:
        print(f"  검증 파일: {dev_path}")
        dev_df = pd.read_csv(dev_path, sep='\t', encoding='utf-8')
        print(f"  검증 데이터: {len(dev_df)}개")
        
        dev_texts = dev_df['comments'].tolist()
        
        if label_type == 'hate':
            dev_labels_raw = dev_df['hate'].tolist()
            dev_labels = [1 if label in ['hate', 'offensive'] else 0 for label in dev_labels_raw]
        elif label_type == 'bias':
            dev_labels_raw = dev_df['bias'].tolist()
            dev_labels = [1 if label in ['gender', 'others'] else 0 for label in dev_labels_raw]
        elif label_type == 'gender_bias':
            dev_labels_raw = dev_df['contain_gender_bias'].tolist()
            dev_labels = [1 if label else 0 for label in dev_labels_raw]
        
        texts.extend(dev_texts)
        labels.extend(dev_labels)
        print(f"  총 데이터: {len(texts)}개 (train + dev)")
    else:
        print(f"  총 데이터: {len(texts)}개")
    
    print(f"  레이블 분포: 0={labels.count(0)}개, 1={labels.count(1)}개")
    
    return Dataset.from_dict({'text': texts, 'label': labels})

def load_unsmile_dataset(dataset_path=None, use_huggingface=True, label_type='hate'):
    """
    Korean UnSmile 데이터셋 로드
    
    Args:
        dataset_path: 로컬 데이터셋 경로 (None이면 Hugging Face에서 로드)
        use_huggingface: True면 Hugging Face Hub에서 로드
        label_type: 사용할 레이블 타입
                   - 'hate': 혐오 표현 여부 (True/False)
                   - 'multi': 다중 레이블 (여러 혐오 유형)
    
    Returns:
        Dataset 객체
    """
    if use_huggingface:
        print("Korean UnSmile 데이터셋 로딩 중 (Hugging Face Hub)...")
        try:
            from datasets import load_dataset
            dataset = load_dataset('smilegate-ai/kor_unsmile', split='train')
            print(f"  총 데이터: {len(dataset)}개")
            
            # 레이블 추출 및 변환
            # Column 객체를 리스트로 변환
            if '문장' in dataset.column_names:
                texts = list(dataset['문장'])
            elif 'text' in dataset.column_names:
                texts = list(dataset['text'])
            else:
                # 첫 번째 문자열 컬럼 찾기
                for col in dataset.column_names:
                    if isinstance(dataset[0][col], str):
                        texts = list(dataset[col])
                        break
                else:
                    raise ValueError("텍스트 필드를 찾을 수 없습니다.")
            
            if label_type == 'hate':
                # 혐오 표현 여부 (하나라도 True면 1)
                hate_labels = []
                for item in dataset:
                    # 혐오 관련 레이블들 확인
                    is_hate = False
                    hate_columns = ['여성/가족', '남성', '성소수자', '인종/국적', '연령', '지역', '종교', '기타 혐오', '악플/욕설', 'clean']
                    
                    for col in hate_columns:
                        if col in item and item[col] == 1:
                            if col != 'clean':  # clean이 1이면 혐오 아님
                                is_hate = True
                                break
                            else:
                                is_hate = False
                                break
                    
                    hate_labels.append(1 if is_hate else 0)
                
                print(f"  레이블 타입: hate (혐오=1, 정상=0)")
                print(f"  레이블 분포: 0={hate_labels.count(0)}개, 1={hate_labels.count(1)}개")
                
                return Dataset.from_dict({'text': texts, 'label': hate_labels})
            else:
                # 다중 레이블은 일단 기본 레이블 사용
                if 'clean' in dataset.column_names:
                    labels = list(dataset['clean'])
                    labels = [1 - label for label in labels]  # clean=0이면 혐오=1
                else:
                    labels = [0] * len(texts)
                return Dataset.from_dict({'text': texts, 'label': labels})
                
        except Exception as e:
            print(f"  Hugging Face 로드 실패: {e}")
            print("  로컬 파일을 사용하거나 수동으로 다운로드하세요.")
            raise
    
    # 로컬 파일 로드 (추후 구현)
    else:
        raise NotImplementedError("로컬 파일 로드는 아직 구현되지 않았습니다.")

def load_kmhas_dataset(use_huggingface=True, label_type='hate', dataset_path=None):
    """
    KMHAS Korean Hate Speech 데이터셋 로드 (jeanlee/kmhas_korean_hate_speech)
    
    Args:
        use_huggingface: True면 Hugging Face Hub에서 로드 시도
        label_type: 사용할 레이블 타입
                   - 'hate': 혐오 표현 여부 (하나라도 혐오면 1, 모두 not_hate_speech면 0)
        dataset_path: 로컬 데이터셋 경로 (디렉토리 또는 파일)
                     - 디렉토리면 train.txt, valid.txt, test.txt 파일 찾음
                     - 파일이면 해당 파일만 로드
    
    Returns:
        Dataset 객체
    """
    # 로컬 파일이 있으면 우선 사용
    if dataset_path:
        print("KMHAS Korean Hate Speech 데이터셋 로딩 중 (로컬 파일)...")
        import os
        import glob
        import urllib.request
        
        texts = []
        labels = []
        
        # 디렉토리인 경우
        if os.path.isdir(dataset_path):
            # train, valid, test 파일 찾기
            train_files = glob.glob(os.path.join(dataset_path, "*train*.txt"))
            valid_files = glob.glob(os.path.join(dataset_path, "*valid*.txt"))
            test_files = glob.glob(os.path.join(dataset_path, "*test*.txt"))
            
            files_to_load = train_files + valid_files + test_files
            
            if not files_to_load:
                # 디렉토리 내 모든 txt 파일 찾기
                files_to_load = glob.glob(os.path.join(dataset_path, "*.txt"))
            
            # 파일이 없으면 자동 다운로드 시도
            if not files_to_load:
                print(f"  데이터 파일을 찾을 수 없습니다. 자동 다운로드를 시도합니다...")
                download_urls = {
                    'train': 'https://raw.githubusercontent.com/adlnlp/K-MHaS/main/data/kmhas_train.txt',
                    'valid': 'https://raw.githubusercontent.com/adlnlp/K-MHaS/main/data/kmhas_valid.txt',
                    'test': 'https://raw.githubusercontent.com/adlnlp/K-MHaS/main/data/kmhas_test.txt'
                }
                
                files_to_load = []
                for split_name, url in download_urls.items():
                    file_path = os.path.join(dataset_path, f'kmhas_{split_name}.txt')
                    if not os.path.exists(file_path):
                        try:
                            print(f"  다운로드 중: {split_name} ({url})")
                            urllib.request.urlretrieve(url, file_path)
                            print(f"  ✓ 다운로드 완료: {os.path.basename(file_path)}")
                        except Exception as e:
                            print(f"  경고: {split_name} 다운로드 실패 ({e})")
                            continue
                    if os.path.exists(file_path):
                        files_to_load.append(file_path)
                
                if not files_to_load:
                    raise ValueError(f"데이터 파일을 찾을 수 없고 다운로드도 실패했습니다: {dataset_path}")
            
            print(f"  로드할 파일: {len(files_to_load)}개")
            for file_path in files_to_load:
                print(f"    - {os.path.basename(file_path)}")
        
        # 단일 파일인 경우
        elif os.path.isfile(dataset_path):
            files_to_load = [dataset_path]
            print(f"  로드할 파일: {os.path.basename(dataset_path)}")
        else:
            raise ValueError(f"파일 또는 디렉토리를 찾을 수 없습니다: {dataset_path}")
        
        # 파일 읽기
        for file_path in files_to_load:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # 첫 줄이 헤더일 수 있으므로 확인
                start_idx = 0
                if len(lines) > 0:
                    first_line = lines[0].strip()
                    # 헤더인지 확인 (텍스트나 레이블이 아닌 경우)
                    if first_line.startswith('text') or first_line.startswith('sentence') or '\t' not in first_line:
                        start_idx = 1
                
                for line in lines[start_idx:]:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # KMHAS 형식: 텍스트\t레이블1,레이블2,... (레이블은 숫자 또는 문자열)
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        text = parts[0]
                        label_str = parts[1]
                        
                        # 다중 레이블 처리 (숫자 또는 문자열)
                        label_list = [l.strip() for l in label_str.split(',')]
                        
                        # 레이블 매핑: 0-8 (8이 not_hate_speech)
                        # 0: origin, 1: physical, 2: politics, 3: profanity, 4: age, 
                        # 5: gender, 6: race, 7: religion, 8: not_hate_speech
                        if label_type == 'hate':
                            # 숫자 레이블인 경우
                            try:
                                label_nums = [int(l) for l in label_list]
                                # 8(not_hate_speech)이 없고 다른 레이블이 있으면 혐오(1)
                                if 8 not in label_nums and len(label_nums) > 0:
                                    labels.append(1)
                                else:
                                    labels.append(0)
                            except ValueError:
                                # 문자열 레이블인 경우
                                if 'not_hate_speech' not in label_list and len(label_list) > 0:
                                    labels.append(1)
                                else:
                                    labels.append(0)
                        else:
                            # 기본: not_hate_speech가 아니면 혐오
                            try:
                                label_nums = [int(l) for l in label_list]
                                labels.append(0 if 8 in label_nums else 1)
                            except ValueError:
                                labels.append(0 if 'not_hate_speech' in label_list else 1)
                        
                        texts.append(text)
        
        print(f"  총 데이터: {len(texts)}개")
        print(f"  레이블 타입: hate (혐오=1, 정상=0)")
        print(f"  레이블 분포: 0={labels.count(0)}개, 1={labels.count(1)}개")
        
        return Dataset.from_dict({'text': texts, 'label': labels})
    
    # Hugging Face에서 로드 시도
    if use_huggingface:
        print("KMHAS Korean Hate Speech 데이터셋 로딩 중 (Hugging Face Hub)...")
        try:
            from datasets import load_dataset
            # 최신 datasets 라이브러리에서는 스크립트 기반 데이터셋을 지원하지 않으므로
            # trust_remote_code 옵션을 사용하거나 다른 방법 시도
            try:
                dataset = load_dataset("jeanlee/kmhas_korean_hate_speech", split='train', trust_remote_code=True)
            except Exception as e1:
                # trust_remote_code로도 안 되면 다른 방법 시도
                print(f"  trust_remote_code로 로드 실패, 다른 방법 시도...")
                # 데이터셋이 parquet 형식으로 제공되는지 확인
                try:
                    dataset = load_dataset("jeanlee/kmhas_korean_hate_speech", split='train', verification_mode='no_checks')
                except:
                    # 마지막 시도: 직접 URL에서 로드
                    raise e1
            print(f"  총 데이터: {len(dataset)}개")
            
            # 텍스트와 레이블 추출
            # 데이터셋 구조 확인
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"  데이터셋 컬럼: {dataset.column_names}")
                
                # 텍스트 필드 찾기
                text_key = None
                for key in ['text', 'comment', 'sentence', '문장', 'content']:
                    if key in sample:
                        text_key = key
                        break
                
                if text_key is None:
                    # 첫 번째 문자열 필드 사용
                    for key in sample.keys():
                        if isinstance(sample[key], str):
                            text_key = key
                            break
                
                if text_key is None:
                    raise ValueError("텍스트 필드를 찾을 수 없습니다.")
                
                texts = list(dataset[text_key])  # Column 객체를 리스트로 변환
                
                # 레이블 필드 찾기
                label_key = None
                for key in ['label', 'hate', 'offensive', 'is_hate', 'is_offensive', 'OFF']:
                    if key in sample:
                        label_key = key
                        break
                
                if label_key is None:
                    # 첫 번째 숫자 필드 사용
                    for key in sample.keys():
                        if key != text_key and isinstance(sample[key], (int, bool)):
                            label_key = key
                            break
                
                if label_key is None:
                    raise ValueError("레이블 필드를 찾을 수 없습니다.")
                
                labels_raw = list(dataset[label_key])  # Column 객체를 리스트로 변환
                
                # 레이블 정규화
                labels = []
                for label in labels_raw:
                    if isinstance(label, bool):
                        labels.append(1 if label else 0)
                    elif isinstance(label, str):
                        labels.append(1 if label.upper() in ['TRUE', '1', 'YES', 'OFF', 'HATE'] else 0)
                    elif isinstance(label, int):
                        labels.append(1 if label == 1 else 0)
                    else:
                        labels.append(1 if label else 0)
                
                print(f"  텍스트 필드: {text_key}")
                print(f"  레이블 필드: {label_key}")
                print(f"  레이블 타입: hate (혐오=1, 정상=0)")
                print(f"  레이블 분포: 0={labels.count(0)}개, 1={labels.count(1)}개")
                
                return Dataset.from_dict({'text': texts, 'label': labels})
            else:
                raise ValueError("데이터셋이 비어있습니다.")
                
        except Exception as e:
            print(f"  Hugging Face 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            raise
    else:
        raise NotImplementedError("로컬 파일 로드는 아직 구현되지 않았습니다.")

def load_from_json(file_path, text_key='text', label_key='label'):
    """
    JSON 파일에서 데이터셋 로드
    
    Args:
        file_path: JSON 파일 경로
        text_key: 텍스트 키 이름
        label_key: 레이블 키 이름
    
    Returns:
        Dataset 객체
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # JSON 형식에 따라 처리
    if isinstance(data, list):
        # 리스트 형식: [{"text": "...", "label": 0}, ...]
        texts = [item[text_key] for item in data]
        labels = [item[label_key] for item in data]
    else:
        # 딕셔너리 형식: {"text": [...], "label": [...]}
        texts = data[text_key]
        labels = data[label_key]
    
    return Dataset.from_dict({'text': texts, 'label': labels})

def load_humane_lab_dataset(dataset_dir, max_samples=None, label_type='hate'):
    """
    Selectstar_Tunip_HUMANE Lab_opendata 데이터셋 로드
    
    Args:
        dataset_dir: 데이터셋 디렉토리 경로
        max_samples: 최대 샘플 수 (None이면 전체 사용)
        label_type: 사용할 레이블 타입 ('hate': 혐오 클래스 사용)
    
    Returns:
        Dataset 객체
    """
    import glob
    import os
    
    print(f"Selectstar_Tunip_HUMANE Lab_opendata 데이터셋 로딩 중...")
    print(f"  디렉토리: {dataset_dir}")
    
    # JSON 파일 목록 가져오기
    json_files = glob.glob(os.path.join(dataset_dir, '*.json'))
    print(f"  총 JSON 파일 수: {len(json_files):,}개")
    
    texts = []
    labels = []
    error_count = 0
    skipped_count = 0
    no_text_field_count = 0
    empty_text_count = 0
    
    # 처음 몇 개 파일을 상세히 확인
    debug_first_n = 10
    
    for idx, json_file in enumerate(json_files):
        try:
            # 여러 인코딩 시도
            data = None
            for encoding in ['utf-8', 'utf-8-sig', 'cp949', 'euc-kr']:
                try:
                    with open(json_file, 'r', encoding=encoding) as f:
                        data = json.load(f)
                    break
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
            
            if data is None:
                error_count += 1
                if error_count <= 5:
                    print(f"  경고: 파일을 읽을 수 없음: {os.path.basename(json_file)}")
                continue
            
            # 텍스트 추출
            text = None
            if '문장' in data:
                text = data['문장']
            elif 'text' in data:
                text = data['text']
            else:
                no_text_field_count += 1
                if idx < debug_first_n:  # 처음 몇 개만 디버깅 출력
                    print(f"  [{idx+1}] 텍스트 필드를 찾을 수 없음: {os.path.basename(json_file)}")
                    print(f"    사용 가능한 키: {list(data.keys())}")
                continue
            
            if not text or len(str(text).strip()) == 0:
                empty_text_count += 1
                if idx < debug_first_n:
                    print(f"  [{idx+1}] 빈 텍스트: {os.path.basename(json_file)}")
                continue
            
            # 레이블 추출
            if label_type == 'hate':
                # '혐오 클래스' 필드 사용: "Y" → 1, "N" → 0
                if '혐오 클래스' in data:
                    hate_class = data['혐오 클래스']
                    label = 1 if hate_class == 'Y' else 0
                else:
                    # 혐오 클래스가 없으면 다른 필드로 판단
                    # 모욕, 욕설, 외설, 성혐오 등이 하나라도 1 이상이면 혐오
                    hate_fields = ['모욕', '욕설', '외설', '성혐오', '폭력위협/범죄조장']
                    has_hate = any(data.get(field, 0) > 0 for field in hate_fields)
                    label = 1 if has_hate else 0
            else:
                label = 0
            
            texts.append(str(text).strip())
            labels.append(label)
            
            # max_samples 제한
            if max_samples and len(texts) >= max_samples:
                print(f"  max_samples({max_samples})에 도달하여 로딩 중단")
                break
                    
        except json.JSONDecodeError as e:
            error_count += 1
            if error_count <= 5:  # 처음 5개 에러만 출력
                print(f"  경고: JSON 파싱 실패 ({os.path.basename(json_file)}): {e}")
        except UnicodeDecodeError as e:
            error_count += 1
            if error_count <= 5:  # 처음 5개 에러만 출력
                print(f"  경고: 인코딩 오류 ({os.path.basename(json_file)}): {e}")
        except Exception as e:
            error_count += 1
            if error_count <= 5:  # 처음 5개 에러만 출력
                print(f"  경고: 파일 로드 실패 ({os.path.basename(json_file)}): {type(e).__name__}: {e}")
            continue
    
    # 통계 출력
    print(f"  처리 통계:")
    print(f"    - 성공: {len(texts):,}개")
    print(f"    - 파일 로드 실패: {error_count:,}개")
    print(f"    - 텍스트 필드 없음: {no_text_field_count:,}개")
    print(f"    - 빈 텍스트: {empty_text_count:,}개")
    
    if len(texts) > 0:
        print(f"  로드된 샘플 수: {len(texts):,}개")
        print(f"  레이블 분포: 0={labels.count(0):,}개, 1={labels.count(1):,}개")
    else:
        print(f"  경고: 로드된 샘플이 없습니다!")
        if no_text_field_count > 0:
            print(f"    → {no_text_field_count}개 파일에서 텍스트 필드를 찾을 수 없습니다.")
        if empty_text_count > 0:
            print(f"    → {empty_text_count}개 파일에서 빈 텍스트가 발견되었습니다.")
        if error_count > 0:
            print(f"    → {error_count}개 파일 로드에 실패했습니다.")
    
    if len(texts) == 0:
        raise ValueError("데이터셋이 비어있습니다.")
    
    return Dataset.from_dict({'text': texts, 'label': labels})

def load_from_txt(file_path, label=0):
    """
    텍스트 파일에서 데이터셋 로드 (각 줄이 하나의 샘플)
    
    Args:
        file_path: 텍스트 파일 경로
        label: 모든 샘플에 적용할 레이블
    
    Returns:
        Dataset 객체
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    labels = [label] * len(texts)
    return Dataset.from_dict({'text': texts, 'label': labels})

def load_kold_dataset(file_path, text_key='comment', label_key='OFF', 
                      label_mapping=None):
    """
    KOLD (Korean Offensive Language Dataset) v1 데이터셋 로드
    
    Args:
        file_path: KOLD JSON 파일 경로
        text_key: 텍스트 필드 이름 (기본값: 'comment')
        label_key: 레이블 필드 이름 (기본값: 'OFF')
        label_mapping: 레이블 매핑 딕셔너리 (None이면 자동 매핑)
                      예: {'NOT_OFF': 0, 'OFF': 1} 또는 {False: 0, True: 1}
    
    Returns:
        Dataset 객체
    """
    print(f"KOLD 데이터셋 로딩 중: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 데이터 형식에 따라 처리
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        # 딕셔너리인 경우 첫 번째 키의 값이 리스트일 수 있음
        first_key = list(data.keys())[0]
        if isinstance(data[first_key], list):
            items = data[first_key]
        else:
            # 딕셔너리의 값들이 개별 항목인 경우
            items = list(data.values())
    else:
        raise ValueError(f"지원하지 않는 데이터 형식: {type(data)}")
    
    print(f"총 {len(items)}개의 항목 발견")
    
    # 텍스트와 레이블 추출
    texts = []
    labels = []
    
    for item in items:
        # 텍스트 필드 찾기 (여러 가능한 키 시도)
        text = None
        for key in [text_key, 'comment', 'text', 'sentence', 'content']:
            if key in item:
                text = item[key]
                break
        
        if text is None:
            print(f"경고: 텍스트 필드를 찾을 수 없습니다. 사용 가능한 키: {list(item.keys())}")
            continue
        
        # 레이블 필드 찾기
        label = None
        for key in [label_key, 'OFF', 'label', 'offensive', 'is_offensive']:
            if key in item:
                label = item[key]
                break
        
        if label is None:
            print(f"경고: 레이블 필드를 찾을 수 없습니다. 사용 가능한 키: {list(item.keys())}")
            continue
        
        texts.append(str(text))
        
        # 레이블 매핑
        if label_mapping is not None:
            if label in label_mapping:
                labels.append(label_mapping[label])
            else:
                # 매핑에 없으면 원본 값 사용 (정수로 변환 시도)
                try:
                    labels.append(int(label))
                except:
                    labels.append(1 if label else 0)
        else:
            # 자동 매핑
            if isinstance(label, bool):
                labels.append(1 if label else 0)
            elif isinstance(label, str):
                # 문자열 레이블 처리
                if label.upper() in ['OFF', 'TRUE', '1', 'YES']:
                    labels.append(1)
                elif label.upper() in ['NOT_OFF', 'FALSE', '0', 'NO']:
                    labels.append(0)
                else:
                    # 숫자 문자열인 경우
                    try:
                        labels.append(int(label))
                    except:
                        labels.append(1)  # 기본값
            elif isinstance(label, int):
                labels.append(label)
            else:
                # 기타 타입은 1로 매핑
                labels.append(1)
    
    print(f"성공적으로 로드된 항목: {len(texts)}개")
    print(f"레이블 분포: 0={labels.count(0)}개, 1={labels.count(1)}개")
    
    return Dataset.from_dict({'text': texts, 'label': labels})

def load_from_huggingface(dataset_name, split='train'):
    """
    Hugging Face Hub에서 데이터셋 로드
    
    Args:
        dataset_name: Hugging Face 데이터셋 이름
        split: 데이터셋 분할 ('train', 'test', 'validation' 등)
    
    Returns:
        Dataset 객체
    """
    from datasets import load_dataset
    return load_dataset(dataset_name, split=split)

def normalize_label_values(labels, label_mapping=None, auto_detect=True):
    """
    레이블을 통일된 형식(0, 1)으로 정규화
    
    Args:
        labels: 레이블 리스트 (다양한 형식 가능)
        label_mapping: 수동 매핑 딕셔너리 (None이면 자동 감지)
                      예: {'OFF': 1, 'NOT_OFF': 0} 또는 {'spam': 1, 'ham': 0}
        auto_detect: True면 자동으로 레이블 형식 감지 및 매핑
    
    Returns:
        정규화된 레이블 리스트 (0 또는 1)
    """
    if not labels:
        return []
    
    # 고유 레이블 값 확인
    unique_labels = list(set(labels))
    print(f"  발견된 레이블 값: {unique_labels}")
    
    # 수동 매핑이 제공된 경우
    if label_mapping is not None:
        print(f"  수동 매핑 사용: {label_mapping}")
        normalized = []
        for label in labels:
            if label in label_mapping:
                normalized.append(label_mapping[label])
            else:
                # 매핑에 없으면 자동 변환 시도
                if isinstance(label, bool):
                    normalized.append(1 if label else 0)
                elif isinstance(label, str):
                    normalized.append(1 if label.upper() in ['TRUE', '1', 'YES', 'OFF', 'SPAM'] else 0)
                else:
                    normalized.append(int(label) if int(label) in [0, 1] else 1)
        return normalized
    
    # 자동 감지
    if auto_detect:
        # 불린 타입
        if all(isinstance(l, bool) for l in unique_labels):
            print("  자동 감지: 불린 타입 → False=0, True=1")
            return [1 if l else 0 for l in labels]
        
        # 정수 타입
        if all(isinstance(l, int) for l in unique_labels):
            if set(unique_labels).issubset({0, 1}):
                print("  자동 감지: 이미 정규화된 정수 (0, 1)")
                return labels
            elif len(unique_labels) == 2:
                # 2개의 정수 값이 있으면 작은 값=0, 큰 값=1로 매핑
                sorted_labels = sorted(unique_labels)
                mapping = {sorted_labels[0]: 0, sorted_labels[1]: 1}
                print(f"  자동 감지: 정수 타입 → {mapping}")
                return [mapping[l] for l in labels]
        
        # 문자열 타입
        if all(isinstance(l, str) for l in unique_labels):
            # 일반적인 패턴 감지
            upper_labels = [l.upper() for l in unique_labels]
            
            # OFF/NOT_OFF 패턴
            if 'OFF' in upper_labels and 'NOT_OFF' in upper_labels:
                print("  자동 감지: OFF/NOT_OFF → OFF=1, NOT_OFF=0")
                return [1 if l.upper() == 'OFF' else 0 for l in labels]
            
            # True/False 문자열
            if 'TRUE' in upper_labels and 'FALSE' in upper_labels:
                print("  자동 감지: TRUE/FALSE → TRUE=1, FALSE=0")
                return [1 if l.upper() == 'TRUE' else 0 for l in labels]
            
            # Yes/No
            if 'YES' in upper_labels and 'NO' in upper_labels:
                print("  자동 감지: YES/NO → YES=1, NO=0")
                return [1 if l.upper() == 'YES' else 0 for l in labels]
            
            # Spam/Ham
            if 'SPAM' in upper_labels and 'HAM' in upper_labels:
                print("  자동 감지: SPAM/HAM → SPAM=1, HAM=0")
                return [1 if l.upper() == 'SPAM' else 0 for l in labels]
            
            # 2개의 고유 값이 있으면 첫 번째=0, 두 번째=1로 매핑
            if len(unique_labels) == 2:
                mapping = {unique_labels[0]: 0, unique_labels[1]: 1}
                print(f"  자동 감지: 문자열 타입 → {mapping}")
                return [mapping[l] for l in labels]
            
            # 숫자 문자열
            try:
                int_labels = [int(l) for l in labels]
                if set(int_labels).issubset({0, 1}):
                    print("  자동 감지: 숫자 문자열 → 정수로 변환")
                    return int_labels
            except:
                pass
        
        # 혼합 타입 또는 기타
        print("  경고: 레이블 형식을 자동으로 감지할 수 없습니다. 기본 매핑 사용")
        # 첫 번째 고유 값=0, 나머지=1로 매핑
        mapping = {unique_labels[0]: 0}
        for label in unique_labels[1:]:
            mapping[label] = 1
        print(f"  기본 매핑: {mapping}")
        return [mapping.get(l, 1) for l in labels]
    
    # 자동 감지 비활성화 시 원본 반환 (정수로 변환 시도)
    return [int(l) if isinstance(l, (int, str)) and str(l).isdigit() else (1 if l else 0) for l in labels]

def load_multiple_datasets(dataset_configs, normalize_labels=True):
    """
    여러 데이터셋을 로드하고 레이블을 통일하여 합침
    
    Args:
        dataset_configs: 데이터셋 설정 리스트
                        각 설정은 딕셔너리:
                        {
                            'type': 'csv' | 'json' | 'kold' | 'txt',
                            'path': '파일 경로',
                            'text_key': '텍스트 필드명' (선택),
                            'label_key': '레이블 필드명' (선택),
                            'label_mapping': {'원본값': 0/1} (선택, None이면 자동 감지)
                        }
        normalize_labels: True면 모든 레이블을 0, 1로 정규화
    
    Returns:
        합쳐진 Dataset 객체
    """
    from datasets import concatenate_datasets
    
    print("=" * 70)
    print("여러 데이터셋 로드 및 통합")
    print("=" * 70)
    
    datasets = []
    
    for i, config in enumerate(dataset_configs, 1):
        print(f"\n[{i}/{len(dataset_configs)}] 데이터셋 로딩 중...")
        print(f"  타입: {config.get('type', 'json')}")
        print(f"  경로: {config['path']}")
        
        try:
            # 데이터셋 타입에 따라 로드
            dataset_type = config.get('type', 'json').lower()
            
            if dataset_type == 'csv':
                dataset = load_from_csv(
                    config['path'],
                    text_column=config.get('text_key', 'text'),
                    label_column=config.get('label_key', 'label')
                )
            elif dataset_type == 'json':
                dataset = load_from_json(
                    config['path'],
                    text_key=config.get('text_key', 'text'),
                    label_key=config.get('label_key', 'label')
                )
            elif dataset_type == 'kold':
                dataset = load_kold_dataset(
                    config['path'],
                    text_key=config.get('text_key', 'comment'),
                    label_key=config.get('label_key', 'OFF'),
                    label_mapping=config.get('label_mapping', None)
                )
            elif dataset_type == 'txt':
                dataset = load_from_txt(
                    config['path'],
                    label=config.get('label', 0)
                )
            else:
                raise ValueError(f"지원하지 않는 데이터셋 타입: {dataset_type}")
            
                # 레이블 정규화
            if normalize_labels:
                print(f"  레이블 정규화 중...")
                original_labels = dataset['label']
                normalized_labels_list = normalize_label_values(
                    original_labels,
                    label_mapping=config.get('label_mapping', None)
                )
                
                # 레이블 업데이트
                dataset = dataset.map(lambda x, idx: {'label': normalized_labels_list[idx]}, 
                                    with_indices=True)
            
            print(f"  ✓ 로드 완료: {len(dataset)}개 항목")
            datasets.append(dataset)
            
        except Exception as e:
            print(f"  ✗ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not datasets:
        raise ValueError("로드된 데이터셋이 없습니다.")
    
    # 데이터셋 합치기
    print(f"\n데이터셋 통합 중...")
    combined_dataset = concatenate_datasets(datasets)
    
    print(f"\n✓ 통합 완료!")
    print(f"  총 데이터셋 수: {len(datasets)}개")
    print(f"  총 항목 수: {len(combined_dataset)}개")
    
    if normalize_labels:
        labels = combined_dataset['label']
        print(f"  레이블 분포: 0={labels.count(0)}개, 1={labels.count(1)}개")
    
    print("=" * 70)
    
    return combined_dataset

def validate_dataset(dataset):
    """
    데이터셋 유효성 검사
    
    Args:
        dataset: Dataset 객체
    
    Returns:
        bool: 유효하면 True
    """
    required_columns = ['text', 'label']
    
    for col in required_columns:
        if col not in dataset.column_names:
            raise ValueError(f"데이터셋에 '{col}' 컬럼이 없습니다.")
    
    if len(dataset) == 0:
        raise ValueError("데이터셋이 비어있습니다.")
    
    # 레이블 타입 확인
    labels = dataset['label']
    if not all(isinstance(label, int) for label in labels):
        print("경고: 일부 레이블이 정수가 아닙니다. 정수로 변환합니다.")
        dataset = dataset.map(lambda x: {'label': int(x['label'])})
    
    return True

# 사용 예시
if __name__ == "__main__":
    # 예시 1: CSV 파일 로드
    # dataset = load_from_csv('data/train.csv')
    
    # 예시 2: JSON 파일 로드
    # dataset = load_from_json('data/train.json')
    
    # 예시 3: 텍스트 파일 로드
    # dataset = load_from_txt('data/positive.txt', label=0)
    
    # 예시 4: Hugging Face Hub에서 로드
    # dataset = load_from_huggingface('imdb', split='train')
    
    print("데이터 로드 유틸리티가 준비되었습니다.")
    print("train.py에서 이 함수들을 import하여 사용하세요.")

