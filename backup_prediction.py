import streamlit as st
import sqlite3
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
    가장 최근 150개의 데이터를 조회합니다.
    """
    try:
        conn = sqlite3.connect('pattern_analysis_v2.db')
        c = conn.cursor()
        
        # 가장 최근 150개의 패턴 전이 데이터 조회
        transitions = c.execute('''
            SELECT 
                pattern1, result1, pattern2, result2,
                prev_pattern1, prev_pattern2, transition_type,
                transition_count,
                pattern1_banker_count, pattern1_player_count,
                pattern2_banker_count, pattern2_player_count,
                pattern1_transitions, pattern2_transitions,
                timestamp
            FROM pattern_records
            ORDER BY timestamp DESC
            LIMIT 150
        ''').fetchall()
        
        # 데이터를 원래 순서(오래된 것 -> 최신 것)로 뒤집기 (선택사항, 예측 로직에 따라 필요할 수 있음)
        # transitions.reverse()
        
        # DataFrame으로 변환
        df = pd.DataFrame(transitions, columns=[
            'pattern1', 'result1', 'pattern2', 'result2',
            'prev_pattern1', 'prev_pattern2', 'transition_type',
            'transition_count',
            'pattern1_banker_count', 'pattern1_player_count',
            'pattern2_banker_count', 'pattern2_player_count',
            'pattern1_transitions', 'pattern2_transitions',
            'timestamp'
        ])
        
        # timestamp 기준으로 오름차순 정렬 (예측 함수들이 시간 순서를 가정할 수 있으므로)
        df = df.sort_values(by='timestamp', ascending=True)
        
        if df.empty:
            st.warning("데이터베이스에 데이터가 없습니다.")
        elif len(df) < 150:
            st.warning(f"데이터가 150개 미만입니다 ({len(df)}개). 사용 가능한 모든 최신 데이터를 사용합니다.")
        else:
            st.info(f"가장 최신 데이터 {len(df)}개를 사용합니다.")
            
        conn.close()
        return df
    except Exception as e:
        st.error(f"데이터 조회 중 오류 발생: {str(e)}")
        return None

def predict_next_pattern(df: pd.DataFrame, current_pattern: str) -> Optional[Dict[str, Any]]:
    """
    현재 패턴을 기반으로 다음 패턴을 예측합니다.
    """
    if df is None or df.empty or not current_pattern:
        return None
    
    # 현재 패턴으로 시작하는 전이 패턴 찾기
    pattern_data = df[
        (df['pattern1'] == current_pattern) | 
        (df['pattern2'] == current_pattern)
    ]
    
    if pattern_data.empty:
        return None
    
    # 패턴1인 경우
    pattern1_next = pattern_data[pattern_data['pattern1'] == current_pattern]['result1'].value_counts()
    # 패턴2인 경우
    pattern2_next = pattern_data[pattern_data['pattern2'] == current_pattern]['result2'].value_counts()
    
    # 두 결과 합치기
    next_patterns = pd.concat([pattern1_next, pattern2_next]).groupby(level=0).sum()
    
    if next_patterns.empty:
        return None
    
    total_occurrences = len(pattern_data)
    best_next = next_patterns.index[0]
    confidence = next_patterns.iloc[0] / total_occurrences
    
    # 신뢰도가 50% 미만이면 반대 패턴이 더 높은 확률
    if confidence < 0.5:
        best_next = 'b' if best_next == 'a' else 'a'
        confidence = 1 - confidence
    
    # 디버그 정보 추가
    debug_info = {
        'pattern': current_pattern,
        'total_matches': total_occurrences,
        'pattern1_matches': len(pattern1_next),
        'pattern2_matches': len(pattern2_next),
        'next_patterns': next_patterns.to_dict(),
        'confidence_adjusted': confidence >= 0.5
    }
    
    return {
        'next_pattern': best_next,
        'confidence': confidence,
        'method': '빈도 기반',
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
        conn = sqlite3.connect('pattern_analysis_v2.db')
        
        # 전체 데이터 조회
        query = '''
            SELECT 
                pattern1, result1, pattern2, result2,
                pattern1_banker_count, pattern1_player_count,
                pattern2_banker_count, pattern2_player_count,
                pattern1_transitions, pattern2_transitions,
                timestamp
            FROM pattern_records
            ORDER BY timestamp
        '''
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # 특성(X)과 레이블(y) 준비
        features = ['pattern1_banker_count', 'pattern1_player_count',
                   'pattern1_transitions', 'pattern2_banker_count',
                   'pattern2_player_count', 'pattern2_transitions']
        
        X = df[features].values
        y = df['result1'].values  # 다음 결과 예측
        
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
        
        # 현재 패턴의 특성 추출
        conn = sqlite3.connect('pattern_analysis_v2.db')
        c = conn.cursor()
        
        query = '''
            SELECT 
                pattern1_banker_count, pattern1_player_count,
                pattern1_transitions, pattern2_banker_count,
                pattern2_player_count, pattern2_transitions
            FROM pattern_records
            WHERE pattern1 = ? AND pattern2 = ?
            ORDER BY timestamp DESC
            LIMIT 1
        '''
        
        row = c.execute(query, (current_pattern1, current_pattern2 or current_pattern1)).fetchone()
        conn.close()
        
        if row is None:
            return None
        
        # 예측
        X_pred = np.array(row).reshape(1, -1)
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
    try:
        conn = sqlite3.connect('pattern_analysis_v2.db')
        c = conn.cursor()
        
        # 테이블 데이터 삭제
        c.execute('DELETE FROM pattern_records')
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"DB 초기화 중 오류 발생: {str(e)}")
        return False

def update_database():
    """
    데이터베이스를 최신 데이터로 업데이트합니다.
    """
    try:
        conn = sqlite3.connect('pattern_analysis_v2.db')
        c = conn.cursor()
        
        # 오래된 데이터 삭제 (예: 30일 이상)
        c.execute('''
            DELETE FROM pattern_records 
            WHERE timestamp < datetime('now', '-30 days')
        ''')
        
        # 통계 업데이트
        c.execute('''
            UPDATE pattern_records 
            SET transition_count = (
                SELECT COUNT(*) 
                FROM pattern_records pr2 
                WHERE pr2.pattern1 = pattern_records.pattern1
            )
        ''')
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"DB 업데이트 중 오류 발생: {str(e)}")
        return False

def analyze_pattern_combination(df: pd.DataFrame, pattern1: str, pattern2: str) -> Optional[Dict]:
    """
    두 패턴의 조합에 대한 통계를 분석합니다.
    """
    if df is None or df.empty:
        return None
        
    # 패턴 조합 찾기
    combined = df[
        ((df['pattern1'] == pattern1) & (df['pattern2'] == pattern2)) |
        ((df['prev_pattern1'] == pattern1) & (df['pattern1'] == pattern2))
    ]
    
    if combined.empty:
        return None
    
    # 통계 계산
    total_occurrences = len(combined)
    sequential = combined[
        ((df['pattern1'] == pattern1) & (df['pattern2'] == pattern2))
    ]
    sequential_prob = len(sequential) / total_occurrences if total_occurrences > 0 else 0
    
    avg_transitions = combined['transition_count'].mean() if 'transition_count' in combined.columns else 0
    
    return {
        'total_occurrences': total_occurrences,
        'sequential_probability': sequential_prob,
        'avg_transitions': avg_transitions
    }

def create_comparison_data(local_data, api_data):
    """로컬과 API 예측을 비교하여 일치하는 항목과 차이가 있는 항목을 분리하여 반환합니다."""
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

def display_pattern_prediction_table(df):
    """
    패턴1(4개)과 패턴2(8개)의 예측값을 테이블 형식으로 수평 배치하여 표시합니다.
    """
    st.markdown("## 패턴 예측 비교 테이블")
    
    # 패턴1 (4개: aa, ab, ba, bb)
    pattern1_list = ['aa', 'ab', 'ba', 'bb']
    
    # 패턴2 (8개: aaa, aab, aba, abb, baa, bab, bba, bbb)
    pattern2_list = ['aaa', 'aab', 'aba', 'abb', 'baa', 'bab', 'bba', 'bbb']
    
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
                st.table(pd.DataFrame([{
                    "패턴": d["패턴"],
                    "예측값": d["예측값"],
                    "신뢰도": f"{d['신뢰도']:.1%}"
                } for d in local_data1]))
            else:
                st.info("예측 데이터가 없습니다.")
        
        # API 예측
        with subcol2:
            st.markdown("#### API 기반 예측")
            api_data1, api_error1 = get_api_predictions(pattern1_list)
            if api_error1:
                st.error(api_error1)
            elif api_data1:
                st.table(pd.DataFrame([{
                    "패턴": d["패턴"],
                    "예측값": d["예측값"],
                    "신뢰도": f"{d['신뢰도']:.1%}"
                } for d in api_data1]))
            else:
                st.info("API 토큰을 설정하세요.")
        
        # 예측 비교 분석 (일치/차이 분리)
        matching1, differing1 = create_comparison_data(local_data1, api_data1)
        
        if matching1:
            st.markdown("#### 예측 일치 분석 (로컬 == API)")
            st.table(pd.DataFrame(matching1))
        
        if differing1:
            st.markdown("#### 예측 차이 분석 (로컬 != API)")
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
                st.table(pd.DataFrame([{
                    "패턴": d["패턴"],
                    "예측값": d["예측값"],
                    "신뢰도": f"{d['신뢰도']:.1%}"
                } for d in local_data2]))
            else:
                st.info("예측 데이터가 없습니다.")
        
        # API 예측
        with subcol2:
            st.markdown("#### API 기반 예측")
            api_data2, api_error2 = get_api_predictions(pattern2_list)
            if api_error2:
                st.error(api_error2)
            elif api_data2:
                st.table(pd.DataFrame([{
                    "패턴": d["패턴"],
                    "예측값": d["예측값"],
                    "신뢰도": f"{d['신뢰도']:.1%}"
                } for d in api_data2]))
            else:
                st.info("API 토큰을 설정하세요.")
        
        # 예측 비교 분석 (일치/차이 분리)
        matching2, differing2 = create_comparison_data(local_data2, api_data2)
        
        if matching2:
            st.markdown("#### 예측 일치 분석 (로컬 == API)")
            st.table(pd.DataFrame(matching2))
            
        if differing2:
            st.markdown("#### 예측 차이 분석 (로컬 != API)")
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
                st.experimental_rerun()  # 앱을 새로고침하여 변경사항 반영
    with col2:
        if st.button("ML 모델 재학습"):
            with st.spinner("모델 학습 중..."):
                model, le, features = train_ml_model()
                if model is not None:
                    st.success("모델 학습이 완료되었습니다!")
    
    # 예측값 테이블 표시
    display_pattern_prediction_table(df)
    
    st.markdown("---")
    
    # 분석 영역 구분
    left_col, right_col = st.columns(2)
    
    # 왼쪽 컬럼: 로컬 분석
    with left_col:
        st.markdown("## 로컬 기반 패턴 분석")
        st.info("📊 빠른 응답이 필요한 경우 사용하세요. 로컬 데이터베이스를 기반으로 분석합니다.")
        
        # 패턴1 입력
        pattern1 = st.text_input("패턴1 입력 (예: aa, ab, ba, bb)", key="pattern1_input_local")
        if pattern1:
            prediction1 = predict_next_pattern(df, pattern1)
            if prediction1:
                confidence_note = "직접 예측" if prediction1['debug_info']['confidence_adjusted'] else "반대 패턴 예측"
                st.markdown(f"""
                    <div class="prediction-text">
                    <span style="color: #1f77b4;">패턴1 예측 결과:</span><br>
                    예측 방법: {prediction1.get('method', '빈도 기반')} ({confidence_note})<br>
                    다음 패턴 예측: {prediction1['next_pattern']}<br>
                    신뢰도: {prediction1['confidence']:.1%}<br>
                    </div>
                """, unsafe_allow_html=True)
                
                with st.expander("패턴1 상세 정보"):
                    st.json(prediction1['debug_info'])
            else:
                st.warning("패턴1에 대한 예측 데이터가 없습니다.")
        
        # 패턴2 입력
        pattern2 = st.text_input("패턴2 입력 (예: aaa, aab, aba, abb)", key="pattern2_input_local")
        if pattern2:
            prediction2 = predict_next_pattern(df, pattern2)
            if prediction2:
                st.markdown(f"""
                    <div class="prediction-text">
                    <span style="color: #2ca02c;">패턴2 예측 결과:</span><br>
                    예측 방법: {prediction2.get('method', '빈도 기반')}<br>
                    다음 패턴 예측: {prediction2['next_pattern']}<br>
                    신뢰도: {prediction2['confidence']:.1%}<br>
                    </div>
                """, unsafe_allow_html=True)
                
                with st.expander("패턴2 상세 정보"):
                    st.json(prediction2['debug_info'])
            else:
                st.warning("패턴2에 대한 예측 데이터가 없습니다.")
        
        # 패턴 조합 분석
        if pattern1 and pattern2:
            st.markdown("### 패턴 조합 분석")
            combined_stats = analyze_pattern_combination(df, pattern1, pattern2)
            if combined_stats:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("전체 발생 횟수", combined_stats['total_occurrences'])
                with col2:
                    st.metric("연속 발생 확률", f"{combined_stats['sequential_probability']:.1%}")
                with col3:
                    st.metric("평균 전환 횟수", f"{combined_stats['avg_transitions']:.1f}")
    
    # 오른쪽 컬럼: API 기반 분석
    with right_col:
        st.markdown("## API 기반 고급 분석")
        st.info("🤖 더 정확한 예측이 필요한 경우 사용하세요. 응답 시간이 다소 걸릴 수 있습니다.")
        
        # API 설정
        with st.expander("API 설정"):
            st.markdown("""
            ### Hugging Face API 토큰 설정 가이드
            1. [Hugging Face Settings](https://huggingface.co/settings/tokens)에 접속
            2. "New token" 버튼 클릭 후 Access Token 생성
            3. 생성된 토큰을 아래에 입력
            """)
            
            if "hf_api_token" in st.session_state:
                st.success("API 토큰이 설정되어 있습니다.")
                if st.button("API 토큰 재설정", key="reset_token"):
                    del st.session_state.hf_api_token
                    st.experimental_rerun()
            else:
                api_token = st.text_input(
                    "Hugging Face API 토큰을 입력하세요:",
                    type="password",
                    help="API 토큰은 huggingface.co에서 발급받을 수 있습니다."
                )
                if api_token:
                    st.session_state.hf_api_token = api_token
                    st.success("API 토큰이 설정되었습니다!")
        
        # API 기반 패턴 분석
        pattern_api = st.text_input("패턴 입력 (예: aa, ab, ba, bb)", key="pattern_input_api")
        if pattern_api and "hf_api_token" in st.session_state:
            with st.spinner("AI 모델이 분석 중입니다..."):
                prediction = get_huggingface_prediction(pattern_api)
                if prediction:
                    st.markdown(f"""
                        <div class="prediction-text">
                        예측 방법: {prediction['method']}<br>
                        다음 패턴 예측: {prediction['next_pattern']}<br>
                        신뢰도: {prediction['confidence']:.1%}<br>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("상세 분석 정보"):
                        st.write("패턴 특성 분석:")
                        st.json({
                            "패턴 길이": len(pattern_api),
                            "반복성": len(set(pattern_api)) == 1,
                            "전환 횟수": sum(1 for i in range(len(pattern_api)-1) if pattern_api[i] != pattern_api[i+1])
                        })
                else:
                    st.error("API 호출 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")
        elif pattern_api:
            st.warning("API 토큰을 먼저 설정해주세요.")
    
    # 하단에 통계 데이터 표시
    st.markdown("---")
    st.subheader("최근 패턴 전이 데이터")
    if not df.empty:
        st.dataframe(df[['pattern1', 'result1', 'pattern2', 'result2', 'transition_type', 'transition_count']])

if __name__ == "__main__":
    main() 