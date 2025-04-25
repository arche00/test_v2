import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
import requests
import json
from typing import Optional, Dict, Any
import time
from database import (
    get_db_connection,
    get_pattern_transitions as db_get_pattern_transitions,
    insert_pattern_record,
    cleanup_old_records
)
from psycopg2.extras import DictCursor

# config.json 파일에서 API 토큰 로드
def load_api_token():
    try:
        # 먼저 Streamlit secrets에서 토큰 확인
        if 'huggingface_api_token' in st.secrets:
            return st.secrets['huggingface_api_token']
        
        # 환경 변수에서 토큰 확인
        token = os.environ.get('HUGGINGFACE_API_TOKEN')
        if token:
            return token
            
        # 로컬 config.json에서 토큰 확인 (개발 환경용)
        if os.path.exists('config.json'):
            with open('config.json', 'r') as f:
                config = json.load(f)
                return config.get('huggingface_api_token')
                
        return None
    except Exception as e:
        print(f"토큰 로드 중 오류 발생: {str(e)}")
        return None

# API 토큰 초기화
if 'hf_api_token' not in st.session_state:
    api_token = load_api_token()
    if api_token:
        st.session_state.hf_api_token = api_token
        print(f"API 토큰이 성공적으로 로드되었습니다.")
    else:
        print("API 토큰을 찾을 수 없습니다.")

# 페이지 설정
st.set_page_config(
    page_title="패턴 분석 시스템",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일 설정
st.markdown("""
    <style>
    .stText {
        writing-mode: horizontal-tb;
        font-size: 24px;
    }
    .prediction-text {
        font-size: 28px;
        font-weight: bold;
        color: #FF4B4B;
    }
    </style>
""", unsafe_allow_html=True)

def get_pattern_transitions():
    """
    DB에서 패턴 전이 데이터를 가져옵니다.
    최근 150개와 최근 3시간의 데이터 중 더 많은 것을 사용합니다.
    """
    conn = get_db_connection()
    if conn is None:
        st.warning("데이터베이스 연결에 실패했습니다.")
        return None

    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            # 최근 3시간 데이터 수 확인
            cur.execute("""
                SELECT COUNT(*) 
                FROM pattern_records 
                WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '3 hours'
            """)
            recent_count = cur.fetchone()[0]
            
            # 최근 3시간과 150개 중 더 큰 값으로 조회
            if recent_count > 150:
                # 최근 3시간 데이터 사용
                cur.execute("""
                    SELECT pattern, next_pattern, timestamp
                    FROM pattern_records
                    WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '3 hours'
                    ORDER BY timestamp DESC
                """)
                st.info(f"최근 3시간 데이터 {recent_count}개를 사용합니다.")
            else:
                # 최근 150개 데이터 사용
                cur.execute("""
                    SELECT pattern, next_pattern, timestamp
                    FROM pattern_records
                    ORDER BY timestamp DESC
                    LIMIT 150
                """)
                st.info(f"최근 150개 데이터를 사용합니다.")
            
            records = [dict(row) for row in cur.fetchall()]
            
            if not records:
                st.warning("데이터베이스에 데이터가 없습니다.")
                return None

            df = pd.DataFrame(records)
            df = df.sort_values(by='timestamp', ascending=True)
            
            return df
            
    except Exception as e:
        st.error(f"데이터 조회 중 오류 발생: {str(e)}")
        return None
    finally:
        conn.close()

def predict_next_pattern(df: pd.DataFrame, current_pattern: str) -> Optional[Dict[str, Any]]:
    """
    현재 패턴을 기반으로 다음 패턴을 예측합니다.
    3시간 데이터 사용 시 시간대별로 다른 가중치를 부여합니다.
    """
    if df is None or df.empty or not current_pattern:
        return None
    
    # 현재 패턴으로 시작하는 전이 패턴 찾기
    pattern_data = df[df['pattern'] == current_pattern].copy()
    
    if pattern_data.empty:
        return None
    
    # 데이터가 3시간 데이터인 경우
    pattern_data['timestamp'] = pd.to_datetime(pattern_data['timestamp'])
    time_range = (pattern_data['timestamp'].max() - pattern_data['timestamp'].min()).total_seconds() / 3600
    
    if time_range <= 3:
        # 최근 시간일수록 높은 가중치 부여
        now = pattern_data['timestamp'].max()
        pattern_data['time_diff'] = (now - pattern_data['timestamp']).dt.total_seconds() / 60  # 분 단위
        
        # 시간대별 가중치 부여
        def get_weight(minutes):
            if minutes <= 30:  # 최근 30분
                return 1.0
            elif minutes <= 60:  # 30분~1시간
                return 0.8
            elif minutes <= 120:  # 1~2시간
                return 0.6
            else:  # 2~3시간
                return 0.4
        
        pattern_data['weight'] = pattern_data['time_diff'].apply(get_weight)
        
        # 가중치를 적용한 다음 패턴 집계
        next_patterns = pd.Series({
            pattern: weights.sum() 
            for pattern, weights in pattern_data.groupby('next_pattern')['weight']
        })
        
        total_weight = pattern_data['weight'].sum()
        confidence = next_patterns.max() / total_weight
        
        # 시간대별 데이터 수 계산 (디버그용)
        time_distribution = {
            '0-30분': len(pattern_data[pattern_data['time_diff'] <= 30]),
            '30-60분': len(pattern_data[(pattern_data['time_diff'] > 30) & (pattern_data['time_diff'] <= 60)]),
            '1-2시간': len(pattern_data[(pattern_data['time_diff'] > 60) & (pattern_data['time_diff'] <= 120)]),
            '2-3시간': len(pattern_data[pattern_data['time_diff'] > 120])
        }
    else:
        # 3시간 이상의 데이터인 경우 기존 방식대로 처리
        next_patterns = pattern_data['next_pattern'].value_counts()
        confidence = next_patterns.iloc[0] / len(pattern_data)
        time_distribution = None
    
    best_next = next_patterns.index[0]
    
    # 신뢰도가 50% 미만이면 반대 패턴이 더 높은 확률
    if confidence < 0.5:
        best_next = 'b' if best_next == 'a' else 'a'
        confidence = 1 - confidence
    
    # 디버그 정보 추가
    debug_info = {
        'pattern': current_pattern,
        'total_matches': len(pattern_data),
        'next_patterns': next_patterns.to_dict(),
        'confidence_adjusted': confidence >= 0.5,
        'using_weighted_calc': time_range <= 3,
        'time_distribution': time_distribution
    }
    
    return {
        'next_pattern': best_next,
        'confidence': confidence,
        'method': '시간대별 가중치' if time_range <= 3 else '빈도 기반',
        'debug_info': debug_info
    }

def predict_next_pattern2(df, current_pattern1, current_pattern2):
    """
    현재 패턴1과 패턴2를 기반으로 다음 패턴을 예측합니다.
    """
    if df is None or df.empty or not current_pattern1 or not current_pattern2:
        return None
    
    # 현재 패턴1과 패턴2로 시작하는 전이 패턴 찾기
    transitions = df[(df['prev_pattern1'] == current_pattern1) & 
                    (df['prev_pattern2'] == current_pattern2)]
    if transitions.empty:
        return None
    
    # 가장 빈번한 다음 패턴 찾기
    next_patterns = transitions['pattern1'].value_counts()
    if next_patterns.empty:
        return None
    
    # 예측 결과 반환
    return {
        'next_pattern': next_patterns.index[0],
        'confidence': next_patterns.iloc[0] / len(transitions),
        'total_occurrences': len(transitions)
    }

def prepare_training_data():
    """
    ML 모델 학습을 위한 데이터를 준비합니다.
    """
    try:
        conn = get_db_connection()
        
        # 전체 데이터 조회
        query = '''
            SELECT pattern, next_pattern, timestamp
            FROM pattern_records
            ORDER BY timestamp
        '''
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            st.warning("학습할 데이터가 없습니다.")
            return None, None, None
        
        # 패턴 특성 계산
        df['banker_count'] = df['pattern'].apply(lambda x: x.count('a'))
        df['player_count'] = df['pattern'].apply(lambda x: x.count('b'))
        df['transitions'] = df['pattern'].apply(lambda x: sum(1 for i in range(len(x)-1) if x[i] != x[i+1]))
        
        # 특성(X)과 레이블(y) 준비
        features = ['banker_count', 'player_count', 'transitions']
        
        X = df[features].values
        y = df['next_pattern'].values  # 다음 패턴 예측
        
        return X, y, features
    except Exception as e:
        st.error(f"학습 데이터 준비 중 오류 발생: {str(e)}")
        return None, None, None

def train_ml_model():
    """
    RandomForest 모델을 학습시킵니다.
    """
    X, y, features = prepare_training_data()
    if X is None or y is None:
        return None
    
    # 레이블 인코딩
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 모델 학습
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)
    
    # 모델 저장
    model_path = 'pattern_prediction_model.joblib'
    joblib.dump((model, le, features), model_path)
    
    return model, le, features

def get_ml_predictions(patterns):
    """ML 모델을 사용하여 여러 패턴에 대한 예측을 수행합니다."""
    ml_data = []
    for pattern in patterns:
        prediction = predict_with_ml(pattern)
        if prediction:
            ml_data.append({
                "패턴": pattern,
                "예측값": prediction['next_pattern'],
                "신뢰도": prediction['confidence']
            })
    return ml_data

def display_ml_prediction_table(df):
    """
    ML 모델의 예측 결과를 테이블 형식으로 표시합니다.
    """
    st.markdown("## 머신러닝 모델 예측 결과")
    
    # 패턴1 (4개: aa, ab, ba, bb)
    pattern1_list = ['aa', 'ab', 'ba', 'bb']
    
    # 패턴2 (8개: aaa, aab, aba, abb, baa, bab, bba, bbb)
    pattern2_list = ['aaa', 'aab', 'aba', 'abb', 'baa', 'bab', 'bba', 'bbb']
    
    # 패턴1과 패턴2 예측을 수평으로 배치
    col1, col2 = st.columns([3, 3])
    
    # 패턴1 예측
    with col1:
        st.markdown("### 패턴1 ML 예측 (aa, ab, ba, bb)")
        ml_data1 = get_ml_predictions(pattern1_list)
        if ml_data1:
            df1 = pd.DataFrame(ml_data1)
            df1['신뢰도'] = df1['신뢰도'].apply(lambda x: f"{x:.1%}")
            st.table(df1)
        else:
            st.info("ML 예측 데이터가 없습니다.")
    
    # 패턴2 예측
    with col2:
        st.markdown("### 패턴2 ML 예측 (aaa ~ bbb)")
        ml_data2 = get_ml_predictions(pattern2_list)
        if ml_data2:
            df2 = pd.DataFrame(ml_data2)
            df2['신뢰도'] = df2['신뢰도'].apply(lambda x: f"{x:.1%}")
            st.table(df2)
        else:
            st.info("ML 예측 데이터가 없습니다.")

def predict_with_ml(current_pattern1, current_pattern2=None):
    """
    ML 모델을 사용하여 다음 패턴을 예측합니다.
    """
    try:
        # 모델 로드
        model_path = 'pattern_prediction_model.joblib'
        if not os.path.exists(model_path):
            model, le, features = train_ml_model()
        else:
            model, le, features = joblib.load(model_path)
        
        # 현재 패턴의 특성 계산
        pattern_features = {
            'banker_count': current_pattern1.count('a'),
            'player_count': current_pattern1.count('b'),
            'transitions': sum(1 for i in range(len(current_pattern1)-1) if current_pattern1[i] != current_pattern1[i+1])
        }
        
        # 예측
        X_pred = np.array([pattern_features[f] for f in features]).reshape(1, -1)
        y_pred = model.predict_proba(X_pred)
        
        # 결과 변환
        predicted_class = le.inverse_transform([np.argmax(y_pred)])[0]
        confidence = np.max(y_pred)
        
        return {
            'next_pattern': predicted_class,
            'confidence': confidence,
            'method': 'ML Model (RandomForest)'
        }
        
    except Exception as e:
        st.error(f"ML 예측 중 오류 발생: {str(e)}")
        return None

def get_huggingface_prediction(pattern: str) -> Optional[Dict[str, Any]]:
    """
    Hugging Face API를 사용하여 패턴 예측을 수행합니다.
    """
    API_TOKEN = st.session_state.get("hf_api_token")
    if not API_TOKEN:
        return None

    # 패턴 분석용 모델 선택 (text-classification 모델 사용)
    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    
    try:
        # 입력 데이터 구성
        payload = {
            "inputs": pattern,
            "parameters": {
                "candidate_labels": ["next_a", "next_b"]
            }
        }
        
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            # API 응답 검증
            if isinstance(result, dict) and 'scores' in result and 'labels' in result:
                # 최고 확률의 예측값 선택
                max_score_idx = result['scores'].index(max(result['scores']))
                predicted_value = 'a' if result['labels'][max_score_idx] == 'next_a' else 'b'
                confidence = result['scores'][max_score_idx]
                
                # 신뢰도가 50% 미만이면 반대 패턴 선택
                if confidence < 0.5:
                    predicted_value = 'b' if predicted_value == 'a' else 'a'
                    confidence = 1 - confidence
                
                return {
                    'next_pattern': predicted_value,
                    'confidence': confidence,
                    'method': 'Hugging Face API',
                    'model_name': 'BART Large MNLI',
                    'raw_predictions': {
                        'next_a': result['scores'][result['labels'].index('next_a')],
                        'next_b': result['scores'][result['labels'].index('next_b')]
                    }
                }
            else:
                st.error(f"예상치 못한 API 응답 형식: {result}")
                return None
        else:
            st.error(f"API 오류: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"API 호출 중 오류 발생: {str(e)}")
        return None

def find_similar_patterns(df: pd.DataFrame, pattern: str) -> Optional[Dict[str, Any]]:
    """
    유사한 패턴을 찾아 예측을 수행합니다.
    """
    if df is None or df.empty:
        return None
        
    # 패턴 길이와 구성이 비슷한 패턴들을 찾음
    similar_patterns = df[
        (df['pattern1'].str.len() == len(pattern)) &
        (df['pattern1'].str.count('a') == pattern.count('a'))
    ]
    
    if similar_patterns.empty:
        return None
        
    # 가장 빈번한 다음 패턴 찾기
    next_patterns = similar_patterns['result1'].value_counts()
    
    return {
        'next_pattern': next_patterns.index[0],
        'confidence': next_patterns.iloc[0] / len(similar_patterns),
        'method': '유사 패턴 기반'
    }

def clear_database():
    """
    데이터베이스를 초기화합니다.
    """
    return cleanup_old_records()

def update_database():
    """
    데이터베이스를 최신 데이터로 업데이트합니다.
    """
    try:
        if cleanup_old_records():
            st.success("데이터베이스가 성공적으로 업데이트되었습니다.")
            st.experimental_rerun()  # 페이지 새로고침
        else:
            st.error("데이터베이스 업데이트 중 오류가 발생했습니다.")
    except Exception as e:
        st.error(f"데이터베이스 업데이트 중 오류 발생: {str(e)}")

def analyze_pattern_combination(df: pd.DataFrame, pattern1: str, pattern2: str) -> Optional[Dict]:
    """
    두 패턴의 조합에 대한 통계를 분석합니다.
    """
    if df is None or df.empty:
        return None
        
    # 패턴 조합 찾기
    combined = df[
        (df['pattern'] == pattern1) & (df['next_pattern'] == pattern2)
    ]
    
    if combined.empty:
        return None
    
    # 통계 계산
    total_occurrences = len(combined)
    
    # 전체 패턴 중 현재 조합의 비율 계산
    total_patterns = len(df)
    sequential_prob = total_occurrences / total_patterns if total_patterns > 0 else 0
    
    # 평균 전환 횟수 (현재는 항상 1, 나중에 필요하면 수정)
    avg_transitions = 1
    
    return {
        'total_occurrences': total_occurrences,
        'sequential_probability': sequential_prob,
        'avg_transitions': avg_transitions
    }

def create_comparison_data(local_data, api_data):
    """
    로컬과 API 예측을 비교하여 일치하는 항목과 차이가 있는 항목을 분리하여 반환합니다.
    """
    matching_predictions = []
    differing_predictions = []
    
    if not api_data:
        return matching_predictions, differing_predictions
        
    local_dict = {item["패턴"]: item for item in local_data}
    api_dict = {item["패턴"]: item for item in api_data}
    
    for pattern in local_dict.keys():
        if pattern in api_dict:
            local_pred = local_dict[pattern]
            api_pred = api_dict[pattern]
            comparison_item = {
                "패턴": pattern,
                "로컬 예측": local_pred["예측값"],
                "로컬 신뢰도": f"{local_pred['신뢰도']:.1%}",
                "API 예측": api_pred["예측값"],
                "API 신뢰도": f"{api_pred['신뢰도']:.1%}"
            }
            if local_pred["예측값"] == api_pred["예측값"]:
                matching_predictions.append(comparison_item)
            else:
                differing_predictions.append(comparison_item)
                
    return matching_predictions, differing_predictions

def get_local_predictions(patterns, df):
    """로컬 예측을 수행합니다."""
    local_data = []
    for pattern in patterns:
        prediction = predict_next_pattern(df, pattern)
        if prediction:
            local_data.append({
                "패턴": pattern,
                "예측값": prediction['next_pattern'],
                "신뢰도": prediction['confidence']
            })
    return local_data

def get_api_predictions(patterns):
    """API 예측을 수행하고 오류 발생 시 적절한 메시지를 반환합니다."""
    api_data = []
    api_error = None
    if "hf_api_token" in st.session_state:
        try:
            for pattern in patterns:
                prediction = get_huggingface_prediction(pattern)
                if prediction:
                    api_data.append({
                        "패턴": pattern,
                        "예측값": prediction['next_pattern'],
                        "신뢰도": prediction['confidence']
                    })
        except Exception as e:
            api_error = f"API 호출 중 오류 발생: {str(e)}"
    return api_data, api_error

def display_pattern_prediction_table(df):
    """
    패턴1(4개)과 패턴2(8개)의 예측값을 테이블 형식으로 수평 배치하여 표시합니다.
    """
    st.markdown("## 패턴 예측 비교 테이블")
    
    # 패턴1 (4개: aa, ab, ba, bb)
    pattern1_list = ['aa', 'ab', 'ba', 'bb']
    
    # 패턴2 (8개: aaa, aab, aba, abb, baa, bab, bba, bbb)
    pattern2_list = ['aaa', 'aab', 'aba', 'abb', 'baa', 'bab', 'bba', 'bbb']
    
    # 패턴1과 패턴2 예측을 수평으로 배치
    col1, col2 = st.columns([3, 3])
    
    # 패턴1 예측
    with col1:
        st.markdown("### 패턴1 예측 (aa, ab, ba, bb)")
        
        # 로컬과 API 예측을 나란히 표시
        subcol1, subcol2 = st.columns(2)
        
        # 로컬 예측 (항상 표시)
        with subcol1:
            st.markdown("#### 로컬 기반 예측")
            local_data1 = get_local_predictions(pattern1_list, df)
            if local_data1:
                st.table(pd.DataFrame(local_data1))
            else:
                st.info("예측 데이터가 없습니다.")
        
        # API 예측
        with subcol2:
            st.markdown("#### API 기반 예측")
            api_data1, api_error1 = get_api_predictions(pattern1_list)
            if api_error1:
                st.error(api_error1)
            elif api_data1:
                st.table(pd.DataFrame(api_data1))
            else:
                st.info("API 토큰을 설정하세요.")
        
        # 예측 비교 분석 (일치/차이 분리)
        matching1, differing1 = create_comparison_data(local_data1, api_data1)
        
        if matching1:
            st.markdown("#### 예측 일치 (로컬 == API)")
            st.table(pd.DataFrame(matching1))
        
        if differing1:
            st.markdown("#### 예측 차이 (로컬 != API)")
            st.table(pd.DataFrame(differing1))
    
    # 패턴2 예측
    with col2:
        st.markdown("### 패턴2 예측 (aaa ~ bbb)")
        
        # 로컬과 API 예측을 나란히 표시
        subcol1, subcol2 = st.columns(2)
        
        # 로컬 예측 (항상 표시)
        with subcol1:
            st.markdown("#### 로컬 기반 예측")
            local_data2 = get_local_predictions(pattern2_list, df)
            if local_data2:
                st.table(pd.DataFrame(local_data2))
            else:
                st.info("예측 데이터가 없습니다.")
        
        # API 예측
        with subcol2:
            st.markdown("#### API 기반 예측")
            api_data2, api_error2 = get_api_predictions(pattern2_list)
            if api_error2:
                st.error(api_error2)
            elif api_data2:
                st.table(pd.DataFrame(api_data2))
            else:
                st.info("API 토큰을 설정하세요.")
        
        # 예측 비교 분석 (일치/차이 분리)
        matching2, differing2 = create_comparison_data(local_data2, api_data2)
        
        if matching2:
            st.markdown("#### 예측 일치 (로컬 == API)")
            st.table(pd.DataFrame(matching2))
        
        if differing2:
            st.markdown("#### 예측 차이 (로컬 != API)")
            st.table(pd.DataFrame(differing2))

def main():
    st.title("패턴 분석 시스템")
    
    # 데이터 로드
    df = get_pattern_transitions()
    if df is None:
        return
    
    # DB 관리 버튼들
    col1, col2 = st.columns(2)
    with col1:
        if st.button("DB 업데이트"):
            if update_database():
                st.success("데이터베이스가 업데이트되었습니다.")
                st.experimental_rerun()
    with col2:
        if st.button("ML 모델 재학습"):
            with st.spinner("모델 학습 중..."):
                model, le, features = train_ml_model()
                if model is not None:
                    st.success("모델 학습이 완료되었습니다!")
    
    # 예측값 테이블 표시
    display_pattern_prediction_table(df)
    
    st.markdown("---")
    
    # ML 예측 결과 테이블 표시
    display_ml_prediction_table(df)

if __name__ == "__main__":
    main() 