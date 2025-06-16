import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from collections import Counter
import gc
import json

def load_number_data():
    """데이터를 청크 단위로 로드하여 메모리 사용량 최적화"""
    conn = sqlite3.connect('pattern_analysis_v2.db')
    chunk_size = 10000  # 한 번에 처리할 레코드 수
    
    # 전체 레코드 수 확인
    total_records = pd.read_sql_query(
        "SELECT COUNT(*) as count FROM pattern_records WHERE pattern1_number IS NOT NULL AND pattern2_number IS NOT NULL AND result1_number IS NOT NULL AND result2_number IS NOT NULL",
        conn
    ).iloc[0]['count']
    
    # 데이터를 청크 단위로 로드
    query = '''
        SELECT 
            timestamp,
            pattern1_number, 
            pattern2_number, 
            result1_number, 
            result2_number
        FROM pattern_records
        WHERE pattern1_number IS NOT NULL 
          AND pattern2_number IS NOT NULL
          AND result1_number IS NOT NULL 
          AND result2_number IS NOT NULL
        ORDER BY timestamp ASC
    '''
    
    chunks = []
    for chunk in pd.read_sql_query(query, conn, chunksize=chunk_size):
        chunks.append(chunk)
        gc.collect()  # 메모리 정리
    
    conn.close()
    return pd.concat(chunks, ignore_index=True)

def extract_features(df, progress_bar=None):
    """Feature 추출 함수 최적화"""
    def get_digit_frequencies(number):
        digits = list(str(number))
        freq = Counter(digits)
        return [freq.get(str(i), 0) for i in range(10)]

    # 시간대 정보 추출
    df['hour'] = pd.to_datetime(df['timestamp'], format='%y%m%d%H%M').dt.hour
    
    # Feature 생성
    features1 = []
    features2 = []
    total_rows = len(df)
    
    for idx, row in df.iterrows():
        if progress_bar and idx % 1000 == 0:
            progress_bar.progress((idx + 1) / total_rows)
        
        # Pattern1 features
        p1_digits = get_digit_frequencies(row['pattern1_number'])
        p1_features = p1_digits + [row['hour']]
        features1.append(p1_features)

        # Pattern2 features
        p2_digits = get_digit_frequencies(row['pattern2_number'])
        p2_features = p2_digits + [row['hour']]
        features2.append(p2_features)

    return np.array(features1), np.array(features2)

def train_number_model():
    """모델 학습 함수 최적화"""
    try:
        # 데이터 로드
        with st.spinner("데이터 로딩 중..."):
            df = load_number_data()
            if df.empty:
                st.error("학습할 데이터가 없습니다.")
                return None, None

        # 고유한 결과값 개수 확인
        unique_results1 = df['result1_number'].nunique()
        unique_results2 = df['result2_number'].nunique()
        st.info(f"학습 데이터의 고유한 결과값 개수:\nPattern1 → Result1: {unique_results1}개\nPattern2 → Result2: {unique_results2}개")

        # Feature 추출
        progress_bar = st.progress(0)
        st.text("Feature 추출 중...")
        X1, X2 = extract_features(df, progress_bar)
        y1 = df['result1_number'].values
        y2 = df['result2_number'].values
        
        # 메모리 정리
        del df
        gc.collect()

        # 모델 학습
        st.text("Pattern1 모델 학습 중...")
        model1 = RandomForestClassifier(
            n_estimators=100,  # 트리 개수 감소
            max_depth=8,       # 깊이 제한
            min_samples_split=10,  # 분할 최소 샘플 수 증가
            n_jobs=-1,         # 모든 CPU 코어 사용
            random_state=42
        )
        model1.fit(X1, y1)
        
        st.text("Pattern2 모델 학습 중...")
        model2 = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=10,
            n_jobs=-1,
            random_state=42
        )
        model2.fit(X2, y2)

        # 모델 저장
        model_data = {
            'model1': model1,
            'model2': model2,
            'n_features1': X1.shape[1],
            'n_features2': X2.shape[1],
            'unique_results1': unique_results1,
            'unique_results2': unique_results2
        }
        joblib.dump(model_data, 'number_prediction_model.joblib')
        
        # 메모리 정리
        del X1, X2, y1, y2, model1, model2
        gc.collect()
        
        st.success("모델 학습이 완료되었습니다.")
        return True

    except Exception as e:
        st.error(f"모델 학습 중 오류 발생: {str(e)}")
        return False

def load_model_data():
    if not os.path.exists('number_prediction_model.joblib'):
        return None
    try:
        return joblib.load('number_prediction_model.joblib')
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def extract_prediction_features(pattern_number, hour=None):
    """예측용 feature 추출 함수"""
    if hour is None:
        hour = pd.Timestamp.now().hour
    
    # 숫자별 빈도 계산
    digits = list(str(pattern_number))
    freq = Counter(digits)
    digit_features = [freq.get(str(i), 0) for i in range(10)]
    
    # 시간 정보
    time_features = [hour]
    
    return np.array(digit_features + time_features).reshape(1, -1)

def predict_result1(pattern1_number):
    model_data = load_model_data()
    if model_data is None:
        st.warning("모델이 없습니다. 먼저 모델을 학습해주세요.")
        return None

    try:
        X_pred = extract_prediction_features(pattern1_number)
        if X_pred.shape[1] != model_data['n_features1']:
            st.error("모델 feature 불일치. 모델을 다시 학습해주세요.")
            return None
        
        # 확률 예측
        proba = model_data['model1'].predict_proba(X_pred)[0]
        # 모든 예측 결과와 확률
        predictions = []
        for idx, (number, prob) in enumerate(zip(model_data['model1'].classes_, proba)):
            predictions.append({
                'number': number,
                'probability': prob
            })
        # 확률 기준으로 정렬
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        return predictions
    except Exception as e:
        st.error(f"예측 중 오류 발생: {str(e)}")
        return None

def predict_result2(pattern2_number):
    model_data = load_model_data()
    if model_data is None:
        st.warning("모델이 없습니다. 먼저 모델을 학습해주세요.")
        return None

    try:
        X_pred = extract_prediction_features(pattern2_number)
        if X_pred.shape[1] != model_data['n_features2']:
            st.error("모델 feature 불일치. 모델을 다시 학습해주세요.")
            return None
        
        # 확률 예측
        proba = model_data['model2'].predict_proba(X_pred)[0]
        # 모든 예측 결과와 확률
        predictions = []
        for idx, (number, prob) in enumerate(zip(model_data['model2'].classes_, proba)):
            predictions.append({
                'number': number,
                'probability': prob
            })
        # 확률 기준으로 정렬
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        return predictions
    except Exception as e:
        st.error(f"예측 중 오류 발생: {str(e)}")
        return None

def load_pattern_data():
    """Load pattern data from pattern.json"""
    try:
        with open('pattern.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading pattern data: {str(e)}")
        return None

def search_patterns(pattern_data, search_query):
    """Search patterns based on query with sequence start matching"""
    results = []
    
    # Normalize search query: remove spaces and convert to lowercase
    normalized_query = ''.join(search_query.lower().split())
    
    for group_name in ['groupA', 'groupB']:
        patterns = pattern_data['patterns'][group_name]
        for pattern in patterns:
            # 패턴 번호로 검색
            if pattern.get('pattern_number', '').startswith(normalized_query):
                results.append({
                    'group': group_name[5],  # 'A' or 'B'
                    'sequence': pattern.get('sequence', []),
                    'group_value': pattern.get('group', group_name[5].lower()),
                    'pattern_number': pattern.get('pattern_number', 'N/A')
                })
    
    return results

def filter_predictions_by_pattern(pattern_number, predictions, pattern_type='pattern1'):
    """Filter predictions based on pattern search results (for pattern1 or pattern2)"""
    if not predictions:
        return predictions

    # Load pattern data
    pattern_data = load_pattern_data()
    if not pattern_data:
        return predictions

    # Get first 2 digits of pattern_number
    search_prefix = str(pattern_number)[:2]
    st.write(f"검색 접두사: {search_prefix}")
    
    # Search patterns with the prefix
    initial_patterns = search_patterns(pattern_data, search_prefix)
    st.write(f"초기 패턴 검색 결과 수: {len(initial_patterns)}")
    
    if not initial_patterns:
        st.write("초기 패턴을 찾을 수 없습니다.")
        return predictions[:3]

    # Get 4th, 5th, 6th items from sequences
    target_sequences = []
    for pattern in initial_patterns:
        sequence = pattern['sequence']
        if len(sequence) >= 6:
            target_seq = ''.join(sequence[3:6]).lower()
            target_sequences.append(target_seq)
            st.write(f"추출된 시퀀스: {target_seq} (패턴 번호: {pattern['pattern_number']})")

    if not target_sequences:
        st.write("시퀀스를 추출할 수 없습니다.")
        return predictions[:3]

    # Search patterns with target sequences
    related_patterns = []
    for seq in target_sequences:
        # 시퀀스로 패턴 검색 (시작 부분 매칭)
        for group_name in ['groupA', 'groupB']:
            patterns = pattern_data['patterns'][group_name]
            for pattern in patterns:
                pattern_seq = ''.join(pattern.get('sequence', [])).lower()
                if pattern_seq.startswith(seq):  # 시작 부분 매칭으로 변경
                    related_patterns.append({
                        'group': group_name[5],
                        'sequence': pattern.get('sequence', []),
                        'group_value': pattern.get('group', group_name[5].lower()),
                        'pattern_number': pattern.get('pattern_number', 'N/A')
                    })
        st.write(f"시퀀스 '{seq}'로 시작하는 패턴 수: {len(related_patterns)}")

    # Get unique pattern numbers and their group values from related patterns
    related_patterns_dict = {}
    for pattern in related_patterns:
        if pattern['pattern_number'] != 'N/A':
            related_patterns_dict[pattern['pattern_number']] = {
                'group': pattern['group'],
                'group_value': pattern['group_value'],
                'sequence': pattern['sequence']
            }
    
    st.write(f"관련 패턴 번호 수: {len(related_patterns_dict)}")
    st.write(f"관련 패턴 번호: {sorted(list(related_patterns_dict.keys()))}")

    # Filter predictions and add group information
    filtered_predictions = []
    for pred in predictions:
        pred_number = str(pred['number'])
        if pred_number in related_patterns_dict:
            pred_info = related_patterns_dict[pred_number]
            filtered_predictions.append({
                'number': pred['number'],
                'probability': pred['probability'],
                'group': pred_info['group'],
                'group_value': pred_info['group_value'],
                'sequence': pred_info['sequence']
            })
            seq_str = ' '.join(pred_info['sequence'])
            st.write(f"필터링된 예측: {pred_number} (확률: {pred['probability']:.2%}, 그룹: {pred_info['group']}, 값: {pred_info['group_value']})\n시퀀스: {seq_str}")

    # If no predictions match the filter, return top 3 original predictions
    if not filtered_predictions:
        st.write("필터링된 예측이 없습니다. 상위 3개 예측을 반환합니다.")
        return predictions[:3]

    # Sort by probability and return top 3
    filtered_predictions.sort(key=lambda x: x['probability'], reverse=True)
    st.write(f"최종 필터링된 예측 수: {len(filtered_predictions)}")
    return filtered_predictions[:3]

def filter_predictions_by_pattern2(pattern2_number, predictions):
    """Pattern2 전용: 입력값의 3,4번째 문자로 패턴 검색 및 필터링"""
    if not predictions:
        return predictions

    # Load pattern data
    pattern_data = load_pattern_data()
    if not pattern_data:
        return predictions

    # Get 3rd and 4th digits of pattern2_number
    pattern2_str = str(pattern2_number)
    if len(pattern2_str) < 4:
        st.write("pattern2_number가 너무 짧습니다. (최소 4자리 필요)")
        return predictions[:3]
    search_prefix = pattern2_str[2:4]
    st.write(f"Pattern2 검색 접두사(3,4번째): {search_prefix}")

    # Search patterns with the prefix
    initial_patterns = search_patterns(pattern_data, search_prefix)
    st.write(f"초기 패턴 검색 결과 수: {len(initial_patterns)}")
    
    if not initial_patterns:
        st.write("초기 패턴을 찾을 수 없습니다.")
        return predictions[:3]

    # Get 4th, 5th, 6th items from sequences
    target_sequences = []
    for pattern in initial_patterns:
        sequence = pattern['sequence']
        if len(sequence) >= 6:
            target_seq = ''.join(sequence[3:6]).lower()
            target_sequences.append(target_seq)
            st.write(f"추출된 시퀀스: {target_seq} (패턴 번호: {pattern['pattern_number']})")

    if not target_sequences:
        st.write("시퀀스를 추출할 수 없습니다.")
        return predictions[:3]

    # Search patterns with target sequences
    related_patterns = []
    for seq in target_sequences:
        # 시퀀스로 패턴 검색 (시작 부분 매칭)
        for group_name in ['groupA', 'groupB']:
            patterns = pattern_data['patterns'][group_name]
            for pattern in patterns:
                pattern_seq = ''.join(pattern.get('sequence', [])).lower()
                if pattern_seq.startswith(seq):
                    related_patterns.append({
                        'group': group_name[5],
                        'sequence': pattern.get('sequence', []),
                        'group_value': pattern.get('group', group_name[5].lower()),
                        'pattern_number': pattern.get('pattern_number', 'N/A')
                    })
        st.write(f"시퀀스 '{seq}'로 시작하는 패턴 수: {len(related_patterns)}")

    # Get unique pattern numbers and their group values from related patterns
    related_patterns_dict = {}
    for pattern in related_patterns:
        if pattern['pattern_number'] != 'N/A':
            related_patterns_dict[pattern['pattern_number']] = {
                'group': pattern['group'],
                'group_value': pattern['group_value'],
                'sequence': pattern['sequence']
            }
    
    st.write(f"관련 패턴 번호 수: {len(related_patterns_dict)}")
    st.write(f"관련 패턴 번호: {sorted(list(related_patterns_dict.keys()))}")

    # Filter predictions and add group information
    filtered_predictions = []
    for pred in predictions:
        pred_number = str(pred['number'])
        if pred_number in related_patterns_dict:
            pred_info = related_patterns_dict[pred_number]
            filtered_predictions.append({
                'number': pred['number'],
                'probability': pred['probability'],
                'group': pred_info['group'],
                'group_value': pred_info['group_value'],
                'sequence': pred_info['sequence']
            })
            seq_str = ' '.join(pred_info['sequence'])
            st.write(f"필터링된 예측: {pred_number} (확률: {pred['probability']:.2%}, 그룹: {pred_info['group']}, 값: {pred_info['group_value']})\n시퀀스: {seq_str}")

    # If no predictions match the filter, return top 3 original predictions
    if not filtered_predictions:
        st.write("필터링된 예측이 없습니다. 상위 3개 예측을 반환합니다.")
        return predictions[:3]

    # Sort by probability and return top 3
    filtered_predictions.sort(key=lambda x: x['probability'], reverse=True)
    st.write(f"최종 필터링된 예측 수: {len(filtered_predictions)}")
    return filtered_predictions[:3]

def main():
    st.title("Number Prediction App (개별 예측)")
    st.markdown("pattern1_number, pattern2_number를 각각 입력하면 result1_number, result2_number를 개별로 예측합니다.")

    # Initialize session state for results
    if 'result1' not in st.session_state:
        st.session_state.result1 = None
    if 'result2' not in st.session_state:
        st.session_state.result2 = None

    if st.button("모델 학습/재학습"):
        if os.path.exists('number_prediction_model.joblib'):
            os.remove('number_prediction_model.joblib')
        train_number_model()

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Pattern1 → Result1 예측")
        pattern1_number = st.text_input("pattern1_number 입력", key="p1_input")
        if st.button("Pattern1 예측", key="p1_btn"):
            if pattern1_number:
                with st.spinner("예측 중..."):
                    predictions = predict_result1(pattern1_number)
                    if predictions:
                        st.write("모든 예측 결과:")
                        for i, pred in enumerate(predictions[:5], 1):
                            st.write(f"{i}위: {pred['number']} (확률: {pred['probability']:.2%})")
                        
                        st.write("---")
                        st.write("패턴 기반 필터링 결과:")
                        filtered_predictions = filter_predictions_by_pattern(pattern1_number, predictions, pattern_type='pattern1')
                        st.session_state.result1 = filtered_predictions
                        st.success("최종 예측 결과:")
                        for i, pred in enumerate(filtered_predictions, 1):
                            group_info = f" (그룹: {pred.get('group', 'N/A')}, 값: {pred.get('group_value', 'N/A')})" if 'group' in pred else ""
                            seq_str = ' '.join(pred.get('sequence', []))
                            st.write(f"{i}위: {pred['number']} (확률: {pred['probability']:.2%}){group_info}\n시퀀스: {seq_str}")
        elif st.session_state.result1:
            st.success("이전 예측 결과:")
            for i, pred in enumerate(st.session_state.result1, 1):
                group_info = f" (그룹: {pred.get('group', 'N/A')}, 값: {pred.get('group_value', 'N/A')})" if 'group' in pred else ""
                seq_str = ' '.join(pred.get('sequence', []))
                st.write(f"{i}위: {pred['number']} (확률: {pred['probability']:.2%}){group_info}\n시퀀스: {seq_str}")

    with col2:
        st.subheader("Pattern2 → Result2 예측")
        pattern2_number = st.text_input("pattern2_number 입력", key="p2_input")
        if st.button("Pattern2 예측", key="p2_btn"):
            if pattern2_number:
                with st.spinner("예측 중..."):
                    predictions = predict_result2(pattern2_number)
                    if predictions:
                        st.write("모든 예측 결과:")
                        for i, pred in enumerate(predictions[:5], 1):
                            st.write(f"{i}위: {pred['number']} (확률: {pred['probability']:.2%})")
                        
                        st.write("---")
                        st.write("패턴 기반 필터링 결과:")
                        filtered_predictions = filter_predictions_by_pattern2(pattern2_number, predictions)
                        st.session_state.result2 = filtered_predictions
                        st.success("최종 예측 결과:")
                        for i, pred in enumerate(filtered_predictions, 1):
                            group_info = f" (그룹: {pred.get('group', 'N/A')}, 값: {pred.get('group_value', 'N/A')})" if 'group' in pred else ""
                            seq_str = ' '.join(pred.get('sequence', []))
                            st.write(f"{i}위: {pred['number']} (확률: {pred['probability']:.2%}){group_info}\n시퀀스: {seq_str}")
        elif st.session_state.result2:
            st.success("이전 예측 결과:")
            for i, pred in enumerate(st.session_state.result2, 1):
                group_info = f" (그룹: {pred.get('group', 'N/A')}, 값: {pred.get('group_value', 'N/A')})" if 'group' in pred else ""
                seq_str = ' '.join(pred.get('sequence', []))
                st.write(f"{i}위: {pred['number']} (확률: {pred['probability']:.2%}){group_info}\n시퀀스: {seq_str}")

if __name__ == "__main__":
    main() 