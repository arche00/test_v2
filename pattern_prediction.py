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
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset
import evaluate
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    page_title="Pattern Analysis System",
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
    표본수는 (1) 최신 200개, (2) 최근 3시간 내 저장된 표본 수 중 큰 값으로 결정합니다.
    """
    try:
        conn = sqlite3.connect('pattern_analysis_v2.db')
        c = conn.cursor()
        # 현재 시간 기준 최근 3시간 타임스탬프 계산 (YYMMDDHH 형식)
        current_time = datetime.now()
        three_hours_ago = current_time - timedelta(hours=3)
        recent_timestamp = three_hours_ago.strftime("%y%m%d%H")
        # (1) 최신 200개
        latest_200 = c.execute('''
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
            LIMIT 200
        ''').fetchall()
        # (2) 최근 3시간 표본
        recent_3h = c.execute('''
            SELECT 
                pattern1, result1, pattern2, result2,
                prev_pattern1, prev_pattern2, transition_type,
                transition_count,
                pattern1_banker_count, pattern1_player_count,
                pattern2_banker_count, pattern2_player_count,
                pattern1_transitions, pattern2_transitions,
                timestamp
            FROM pattern_records
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        ''', (recent_timestamp,)).fetchall()
        # 더 큰 표본 사용
        if len(recent_3h) > len(latest_200):
            transitions = recent_3h
            st.info(f"최근 3시간 표본({len(recent_3h)}개) 사용")
        else:
            transitions = latest_200
            st.info(f"최신 200개 표본({len(latest_200)}개) 사용")
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
        df = df.sort_values(by='timestamp', ascending=True)
        if df.empty:
            st.warning("데이터베이스에 데이터가 없습니다.")
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
    pattern.json의 패턴 정보와 DB의 새로운 필드를 포함합니다.
    """
    try:
        # 1. DB 데이터 로드
        conn = sqlite3.connect('pattern_analysis_v2.db')
        query = '''
            SELECT 
                pattern1, result1, pattern2, result2,
                pattern1_banker_count, pattern1_player_count,
                pattern2_banker_count, pattern2_player_count,
                pattern1_transitions, pattern2_transitions,
                pattern1_number, result1_number,
                pattern2_number, result2_number,
                timestamp
            FROM pattern_records
            ORDER BY timestamp
        '''
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # 2. pattern.json 데이터 로드
        try:
            with open('pattern.json', 'r') as f:
                pattern_data = json.load(f)
            
            # pattern.json의 패턴 정보를 딕셔너리로 변환
            pattern_info = {}
            for group_name in ['groupA', 'groupB']:
                for pattern in pattern_data['patterns'][group_name]:
                    pattern_number = pattern.get('pattern_number')
                    if pattern_number:
                        pattern_info[pattern_number] = {
                            'group': group_name[-1].lower(),  # 'a' 또는 'b'
                            'sequence': pattern.get('sequence', [])
                        }
        except Exception as e:
            st.warning(f"pattern.json 로드 중 오류 발생: {str(e)}")
            pattern_info = {}

        # 3. 특성(X)과 레이블(y) 준비
        features = [
            # 기존 특성
            'pattern1_banker_count', 'pattern1_player_count',
                   'pattern1_transitions', 'pattern2_banker_count',
            'pattern2_player_count', 'pattern2_transitions',
            
            # 패턴 번호 관련 특성
            'pattern1_number_exists', 'result1_number_exists',
            'pattern2_number_exists', 'result2_number_exists',
            
            # 패턴 그룹 정보
            'pattern1_in_groupA', 'pattern1_in_groupB',
            'pattern2_in_groupA', 'pattern2_in_groupB'
        ]

        # 기본 특성 추출
        X_basic = df[['pattern1_banker_count', 'pattern1_player_count',
                     'pattern1_transitions', 'pattern2_banker_count',
                     'pattern2_player_count', 'pattern2_transitions']].values

        # 패턴 번호 존재 여부 (1 또는 0)
        X_numbers = np.column_stack([
            df['pattern1_number'].notna().astype(int),
            df['result1_number'].notna().astype(int),
            df['pattern2_number'].notna().astype(int),
            df['result2_number'].notna().astype(int)
        ])

        # 패턴 그룹 정보
        X_groups = np.zeros((len(df), 4))
        for i, row in df.iterrows():
            # pattern1 그룹 확인
            if row['pattern1_number'] in pattern_info:
                group = pattern_info[row['pattern1_number']]['group']
                X_groups[i, 0] = 1 if group == 'a' else 0  # groupA
                X_groups[i, 1] = 1 if group == 'b' else 0  # groupB
            
            # pattern2 그룹 확인
            if row['pattern2_number'] in pattern_info:
                group = pattern_info[row['pattern2_number']]['group']
                X_groups[i, 2] = 1 if group == 'a' else 0  # groupA
                X_groups[i, 3] = 1 if group == 'b' else 0  # groupB

        # 모든 특성 결합
        X = np.column_stack([X_basic, X_numbers, X_groups])
        y = df['result1'].values
        
        return X, y, features
    except Exception as e:
        st.error(f"학습 데이터 준비 중 오류 발생: {str(e)}")
        import traceback
        st.error(f"상세 에러: {traceback.format_exc()}")
        return None, None, None

def train_ml_model():
    """
    RandomForest 모델을 학습시킵니다.
    """
    X, y, features = prepare_training_data()
    if X is None or y is None:
        return None
    
    try:
    # 레이블 인코딩
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
        # 모델 학습 (하이퍼파라미터 개선)
        model = RandomForestClassifier(
            n_estimators=200,  # 트리 개수 증가
            max_depth=10,      # 트리 깊이 제한
            min_samples_split=5,  # 분할을 위한 최소 샘플 수
            min_samples_leaf=2,   # 리프 노드의 최소 샘플 수
            random_state=42
        )
    model.fit(X, y_encoded)
        
        # 모델 성능 평가
        y_pred = model.predict(X)
        accuracy = np.mean(y_pred == y_encoded)
        st.info(f"모델 학습 완료 (정확도: {accuracy:.2%})")
        
        # 특성 중요도 저장
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # 세션 상태에 특성 중요도 저장
        st.session_state.feature_importance = feature_importance
    
    # 모델 저장
    model_path = 'pattern_prediction_model.joblib'
    joblib.dump((model, le, features), model_path)
    
    return model, le, features
        
    except Exception as e:
        st.error(f"모델 학습 중 오류 발생: {str(e)}")
        import traceback
        st.error(f"상세 에러: {traceback.format_exc()}")
        return None

def predict_with_ml(current_pattern1, current_pattern2=None):
    """
    ML 모델을 사용하여 다음 패턴을 예측합니다.
    """
    try:
        # 모델 로드
        model_path = 'pattern_prediction_model.joblib'
        if not os.path.exists(model_path):
            st.info("RandomForest 모델을 학습합니다...")
            model, le, features = train_ml_model()
            if model is None:
                st.warning("학습 데이터가 부족하여 모델을 학습할 수 없습니다.")
                return None
        else:
            model, le, features = joblib.load(model_path)
        
        # 현재 패턴의 특성 계산
        pattern1 = current_pattern1
        pattern2 = current_pattern2 or current_pattern1

        # 1. 기본 특성 계산
        pattern1_banker_count = pattern1.count('a')
        pattern1_player_count = pattern1.count('b')
        pattern1_transitions = sum(1 for i in range(len(pattern1)-1) if pattern1[i] != pattern1[i+1])
        
        pattern2_banker_count = pattern2.count('a')
        pattern2_player_count = pattern2.count('b')
        pattern2_transitions = sum(1 for i in range(len(pattern2)-1) if pattern2[i] != pattern2[i+1])

        # 2. pattern.json 데이터 로드
        try:
            with open('pattern.json', 'r') as f:
                pattern_data = json.load(f)
            
            # pattern.json의 패턴 정보를 딕셔너리로 변환
            pattern_info = {}
            for group_name in ['groupA', 'groupB']:
                for pattern in pattern_data['patterns'][group_name]:
                    pattern_number = pattern.get('pattern_number')
                    if pattern_number:
                        pattern_info[pattern_number] = {
                            'group': group_name[-1].lower(),
                            'sequence': pattern.get('sequence', [])
                        }
        except Exception as e:
            st.warning(f"pattern.json 로드 중 오류 발생: {str(e)}")
            pattern_info = {}

        # 3. 패턴 번호 존재 여부 확인 (DB 조회)
        conn = sqlite3.connect('pattern_analysis_v2.db')
        c = conn.cursor()
        
        query = '''
            SELECT pattern1_number, result1_number, pattern2_number, result2_number
            FROM pattern_records
            WHERE pattern1 = ? AND pattern2 = ?
            ORDER BY timestamp DESC
            LIMIT 1
        '''
        
        row = c.execute(query, (pattern1, pattern2)).fetchone()
        conn.close()
        
        # 패턴 번호 존재 여부 (1 또는 0)
        pattern1_number_exists = 1 if row and row[0] else 0
        result1_number_exists = 1 if row and row[1] else 0
        pattern2_number_exists = 1 if row and row[2] else 0
        result2_number_exists = 1 if row and row[3] else 0

        # 4. 패턴 그룹 정보
        pattern1_in_groupA = 0
        pattern1_in_groupB = 0
        pattern2_in_groupA = 0
        pattern2_in_groupB = 0

        if row and row[0] in pattern_info:
            group = pattern_info[row[0]]['group']
            pattern1_in_groupA = 1 if group == 'a' else 0
            pattern1_in_groupB = 1 if group == 'b' else 0
        
        if row and row[2] in pattern_info:
            group = pattern_info[row[2]]['group']
            pattern2_in_groupA = 1 if group == 'a' else 0
            pattern2_in_groupB = 1 if group == 'b' else 0

        # 5. 모든 특성 결합
        X_pred = np.array([
            pattern1_banker_count, pattern1_player_count, pattern1_transitions,
            pattern2_banker_count, pattern2_player_count, pattern2_transitions,
            pattern1_number_exists, result1_number_exists,
            pattern2_number_exists, result2_number_exists,
            pattern1_in_groupA, pattern1_in_groupB,
            pattern2_in_groupA, pattern2_in_groupB
        ]).reshape(1, -1)
        
        # 예측
        y_pred = model.predict_proba(X_pred)
        
        # 결과 변환
        predicted_class = le.inverse_transform([np.argmax(y_pred)])[0]
        confidence = np.max(y_pred)
        
        # 디버그 정보
        debug_info = {
            'pattern1': pattern1,
            'pattern2': pattern2,
            'features': dict(zip(features, X_pred[0])),
            'probabilities': dict(zip(le.classes_, y_pred[0]))
        }
        
        st.session_state.last_rf_debug = debug_info
                
                return {
            'next_pattern': predicted_class,
                    'confidence': confidence,
            'method': 'ML Model (RandomForest)',
            'debug_info': debug_info
        }
            
    except Exception as e:
        st.error(f"ML 예측 중 오류 발생: {str(e)}")
        import traceback
        st.error(f"상세 에러: {traceback.format_exc()}")
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

def save_prediction_records(compare_df, prediction_type):
    """
    예측 테이블의 데이터를 prediction_records 테이블에 저장합니다.
    
    Args:
        compare_df (pd.DataFrame): 예측 비교 데이터프레임
        prediction_type (str): 예측 유형 ('pattern1' 또는 'pattern2')
    """
    try:
        conn = sqlite3.connect('pattern_analysis_v2.db')
        c = conn.cursor()
        
        # 현재 시간을 YYMMDDHHMM 형식으로 저장
        current_time = datetime.now().strftime("%y%m%d%H%M")
        
        for _, row in compare_df.iterrows():
            # 다수 예측값과 평균 신뢰도 계산
            preds = [row['로컬 예측'], row['BART 예측'], row['DistilBERT 예측']]
            confs = [row['로컬 신뢰도'], row['BART 신뢰도'], row['DistilBERT 신뢰도']]
            
            # 다수 예측값 찾기
            majority_pred = None
            avg_conf = None
            is_local_sst2_same = False
            
            for v in set(preds):
                idxs = [i for i, p in enumerate(preds) if p == v]
                if len(idxs) == 2:
                    majority_pred = v
                    avg_conf = sum([confs[i] for i in idxs if confs[i] is not None]) / 2
                    is_local_sst2_same = (row['로컬 예측'] == row['DistilBERT 예측'])
                    break
            
            # 데이터 저장
            c.execute('''
                INSERT INTO prediction_records 
                (timestamp, pattern, local_prediction, local_confidence,
                 bart_prediction, bart_confidence, distilbert_prediction,
                 distilbert_confidence, majority_prediction, average_confidence,
                 is_local_sst2_same, prediction_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                current_time, row['패턴'],
                row['로컬 예측'], row['로컬 신뢰도'],
                row['BART 예측'], row['BART 신뢰도'],
                row['DistilBERT 예측'], row['DistilBERT 신뢰도'],
                majority_pred, avg_conf,
                is_local_sst2_same, prediction_type
            ))
        
        conn.commit()
        conn.close()
        return True
            except Exception as e:
        st.error(f"예측 데이터 저장 중 오류 발생: {str(e)}")
        return False
    
def get_all_predictions(patterns, df):
        local_data = []
    bart_data = []
    transformer_data = []
    local_conf = []
    bart_conf = []
    transformer_conf = []
        for pattern in patterns:
        # Local
        local_pred = predict_next_pattern(df, pattern)
        local_data.append(local_pred['next_pattern'] if local_pred else None)
        local_conf.append(local_pred['confidence'] if local_pred else None)
        
        # RandomForest
        bart_pred = predict_with_ml(pattern)
        bart_data.append(bart_pred['next_pattern'] if bart_pred else None)
        bart_conf.append(bart_pred['confidence'] if bart_pred else None)
        
        # Local Transformers
        if "local_clf" in st.session_state:
            try:
                result = st.session_state.local_clf(pattern)
                if result and isinstance(result, list) and len(result) > 0:
                    transformer_data.append(result[0]["label"])
                    transformer_conf.append(result[0]["score"])
                else:
                    transformer_data.append(None)
                    transformer_conf.append(None)
            except Exception as e:
                transformer_data.append(None)
                transformer_conf.append(None)
        else:
            transformer_data.append(None)
            transformer_conf.append(None)
            
    return local_data, bart_data, transformer_data, local_conf, bart_conf, transformer_conf

def display_pattern_prediction_table(df):
    """
    Pattern prediction comparison table display
    """
    st.markdown("# Pattern Prediction Comparison Table")
    
    # Pattern lists
    pattern1_list = ['aa', 'ab', 'ba', 'bb']
    pattern2_list = ['aaa', 'aab', 'aba', 'abb', 'baa', 'bab', 'bba', 'bbb']
    
    # Pattern1 predictions
    local_preds1, bart_preds1, transformer_preds1, local_confs1, bart_confs1, transformer_confs1 = get_all_predictions(pattern1_list, df)
    compare_df1 = pd.DataFrame({
        "Pattern": pattern1_list,
        "Local Prediction": local_preds1,
        "Local Confidence": local_confs1,
        "RandomForest Prediction": bart_preds1,
        "RandomForest Confidence": bart_confs1,
        "Local Transformers Prediction": transformer_preds1,
        "Local Transformers Confidence": transformer_confs1
    })

    # Pattern2 predictions
    local_preds2, bart_preds2, transformer_preds2, local_confs2, bart_confs2, transformer_confs2 = get_all_predictions(pattern2_list, df)
    compare_df2 = pd.DataFrame({
        "Pattern": pattern2_list,
        "Local Prediction": local_preds2,
        "Local Confidence": local_confs2,
        "RandomForest Prediction": bart_preds2,
        "RandomForest Confidence": bart_confs2,
        "Local Transformers Prediction": transformer_preds2,
        "Local Transformers Confidence": transformer_confs2
    })

    # Combine results
    compare_df = pd.concat([compare_df1, compare_df2], ignore_index=True)

    # Save button
    if st.button("Save Prediction Table"):
        if save_prediction_records(compare_df1, 'pattern1'):
            st.success("Pattern1 prediction data saved")
        if save_prediction_records(compare_df2, 'pattern2'):
            st.success("Pattern2 prediction data saved")

    # 1. All models predict the same
    all_same = compare_df[
        (compare_df['Local Prediction'] == compare_df['RandomForest Prediction']) & 
        (compare_df['Local Prediction'] == compare_df['Local Transformers Prediction'])
    ]
    if not all_same.empty:
        st.markdown("### [Comparison] All three models predict the same")
        st.dataframe(all_same.reset_index(drop=True), use_container_width=True, hide_index=True)

    # 2. Two models predict the same
    def two_same_row(row):
        preds = [row['Local Prediction'], row['RandomForest Prediction'], row['Local Transformers Prediction']]
        return len(set(preds)) == 2
    
    two_same = compare_df[compare_df.apply(two_same_row, axis=1)].copy()
    
    def get_majority_and_avg_conf(row):
        preds = [row['Local Prediction'], row['RandomForest Prediction'], row['Local Transformers Prediction']]
        confs = [row['Local Confidence'], row['RandomForest Confidence'], row['Local Transformers Confidence']]
        for v in set(preds):
            idxs = [i for i, p in enumerate(preds) if p == v]
            if len(idxs) == 2:
                avg_conf = sum([confs[i] for i in idxs if confs[i] is not None]) / 2
                is_local_transformer_same = (row['Local Prediction'] == row['Local Transformers Prediction'])
                return pd.Series({"Majority Prediction": v, "Average Confidence": avg_conf, "Local_Transformer_Same": is_local_transformer_same})
        return pd.Series({"Majority Prediction": None, "Average Confidence": None, "Local_Transformer_Same": False})
    
    if not two_same.empty:
        two_same = two_same.join(two_same.apply(get_majority_and_avg_conf, axis=1))
        two_same = two_same.sort_values(["Local_Transformer_Same", "Average Confidence"], ascending=[False, False])
        col_order = ["Majority Prediction", "Pattern", "Local Prediction", "Local Confidence", 
                    "RandomForest Prediction", "RandomForest Confidence", 
                    "Local Transformers Prediction", "Local Transformers Confidence", "Average Confidence"]
        two_same = two_same[[c for c in col_order if c in two_same.columns]]
        st.markdown("### [Comparison] Two out of three models predict the same (Local=Transformer priority, average confidence order)")
        st.dataframe(two_same.reset_index(drop=True), use_container_width=True, hide_index=True)

    # Pattern1/2 prediction tables
    st.markdown("## Pattern1 Prediction (aa, ab, ba, bb)")
    df1 = compare_df1.copy()
    for col in ['Local Confidence', 'RandomForest Confidence', 'Local Transformers Confidence']:
        df1[col] = df1[col].apply(lambda x: f"{x:.1%}" if x is not None else None)
    st.dataframe(df1, use_container_width=True, hide_index=True)

    st.markdown("## Pattern2 Prediction (aaa ~ bbb)")
    df2 = compare_df2.copy()
    for col in ['Local Confidence', 'RandomForest Confidence', 'Local Transformers Confidence']:
        df2[col] = df2[col].apply(lambda x: f"{x:.1%}" if x is not None else None)
    st.dataframe(df2, use_container_width=True, hide_index=True)

def get_prediction_history():
    """
    예측 기록을 가져와서 DataFrame으로 반환합니다.
    """
    try:
        conn = sqlite3.connect('pattern_analysis_v2.db')
        query = '''
            SELECT 
                timestamp,
                pattern,
                local_prediction,
                local_confidence,
                bart_prediction,
                bart_confidence,
                distilbert_prediction,
                distilbert_confidence,
                majority_prediction,
                average_confidence,
                prediction_type
            FROM prediction_records
            ORDER BY timestamp
        '''
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"예측 기록 조회 중 오류 발생: {str(e)}")
        return None

def display_prediction_trends():
    """
    예측값의 변화를 시각화하여 표시합니다.
    """
    df = get_prediction_history()
    if df is None or df.empty:
        st.warning("저장된 예측 기록이 없습니다.")
        return
    
    # 타임스탬프를 datetime으로 변환
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%y%m%d%H%M')
    
    # 패턴별로 데이터 분리
    pattern1_df = df[df['prediction_type'] == 'pattern1']
    pattern2_df = df[df['prediction_type'] == 'pattern2']
    
    # Pattern1 예측 변화 히트맵
    st.markdown("### Pattern1 예측 변화")
    pattern1_pivot = pattern1_df.pivot_table(
        index='timestamp',
        columns='pattern',
        values='majority_prediction',
        aggfunc='first'
    )
    
    fig1 = go.Figure(data=go.Heatmap(
        z=pattern1_pivot.values,
        x=pattern1_pivot.columns,
        y=pattern1_pivot.index,
        colorscale='RdYlBu',
        text=pattern1_pivot.values,
        texttemplate='%{text}',
        textfont={"size": 14}
    ))
    
    fig1.update_layout(
        title="Pattern1 예측값 변화 히트맵",
        xaxis_title="패턴",
        yaxis_title="시간"
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # Pattern2 예측 변화 히트맵
    st.markdown("### Pattern2 예측 변화")
    pattern2_pivot = pattern2_df.pivot_table(
        index='timestamp',
        columns='pattern',
        values='majority_prediction',
        aggfunc='first'
    )
    
    fig2 = go.Figure(data=go.Heatmap(
        z=pattern2_pivot.values,
        x=pattern2_pivot.columns,
        y=pattern2_pivot.index,
        colorscale='RdYlBu',
        text=pattern2_pivot.values,
        texttemplate='%{text}',
        textfont={"size": 14}
    ))
    
    fig2.update_layout(
        title="Pattern2 예측값 변화 히트맵",
        xaxis_title="패턴",
        yaxis_title="시간"
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # 모델별 예측 분포 스택 바 차트
    st.markdown("### 모델별 예측 분포")
    
    # 시간별 모델 예측 분포 계산
    model_distribution = df.groupby(['timestamp', 'local_prediction']).size().unstack(fill_value=0)
    
    fig3 = go.Figure()
    for prediction in model_distribution.columns:
        fig3.add_trace(go.Bar(
            x=model_distribution.index,
            y=model_distribution[prediction],
            name=f"예측값: {prediction}",
            text=model_distribution[prediction],
            textposition='auto'
        ))
    
    fig3.update_layout(
        title="시간별 예측값 분포",
        xaxis_title="시간",
        yaxis_title="예측 횟수",
        barmode='stack'
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    # 신뢰도 분포 박스 플롯
    st.markdown("### 예측 신뢰도 분포")
    fig4 = go.Figure()
    
    for model in ['local_confidence', 'bart_confidence', 'distilbert_confidence']:
        fig4.add_trace(go.Box(
            y=df[model],
            name=model.replace('_confidence', ''),
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8
        ))
    
    fig4.update_layout(
        title="모델별 예측 신뢰도 분포",
        yaxis_title="신뢰도",
        showlegend=True
    )
    st.plotly_chart(fig4, use_container_width=True)

def train_local_transformer_model_from_db(db_path, model_name="distilbert-base-uncased", force_retrain=False):
    """
    DB에서 Local Transformers 모델을 학습시킵니다.
    force_retrain: True인 경우 기존 모델이 있어도 재학습을 진행합니다.
    """
    logger.info("Starting model training process...")
    
    model_dir = "local_transformer_model"
    os.makedirs(model_dir, exist_ok=True)
    
    # 기존 모델 체크
    if not force_retrain and os.path.exists(model_dir) and os.path.exists(os.path.join(model_dir, "config.json")):
        try:
            logger.info("Loading existing model...")
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            clf = pipeline("text-classification", model=model, tokenizer=tokenizer)
            return clf
        except Exception as e:
            logger.warning(f"Failed to load existing model: {str(e)}. Training new model...")
    
    # 데이터 로드 및 전처리
    logger.info("Loading data from database...")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT pattern1, result1 FROM pattern_records", conn)
    conn.close()
    
    if df.empty:
        raise ValueError("No training data found in the database")
    
    logger.info(f"Initial data shape: {df.shape}")
    df = df.dropna()
    df = df[df['result1'].isin(['a', 'b'])]
    logger.info(f"Processed data shape: {df.shape}")
    
    if len(df) < 10:
        raise ValueError("Insufficient training data (minimum 10 samples required)")
    
    # 레이블 인코딩
    df['label'] = df['result1'].map({'a': 0, 'b': 1})
    dataset = Dataset.from_pandas(df[['pattern1', 'label']].rename(columns={'pattern1': 'text'}))
    
    # 토크나이저 설정
    logger.info("Setting up tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=128)
    
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 모델 설정
    logger.info("Setting up model...")
    id2label = {0: "a", 1: "b"}
    label2id = {"a": 0, "b": 1}
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=id2label,
        label2id=label2id
    )
    
    # 평가 메트릭스
    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)
    
    # 학습 설정
    logger.info("Configuring training arguments...")
    training_args = TrainingArguments(
        output_dir=model_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_dir=os.path.join(model_dir, "logs"),
        logging_steps=10,
        push_to_hub=False,
    )
    
    # Trainer 초기화
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # 학습 실행
    logger.info("Starting training...")
    try:
        train_result = trainer.train()
        logger.info(f"Training completed. Metrics: {train_result.metrics}")
        st.write("Training metrics:", train_result.metrics)
        
        # 모델 저장
        logger.info("Saving model...")
        trainer.save_model(model_dir)
        tokenizer.save_pretrained(model_dir)
        
        # pipeline 생성
        logger.info("Creating pipeline...")
        clf = pipeline("text-classification", model=model, tokenizer=tokenizer)
        return clf
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise e

def main():
    st.title("Pattern Analysis System")
    
    # 데이터 로드
    df = get_pattern_transitions()
    if df is None:
        st.error("Failed to load data")
        return
    
    # 기존 DB 업데이트 및 머신러닝 재학습 버튼 (Local Transformers Model 위에 위치)
    st.markdown("---")
    st.header("DB 관리 및 머신러닝 모델")
    col_db, col_ml = st.columns(2)
    with col_db:
        if st.button("DB 업데이트"):
            update_database()
            st.success("DB가 성공적으로 업데이트되었습니다.")
    with col_ml:
        if st.button("머신러닝 재학습"):
            train_ml_model()
            st.success("머신러닝 모델이 재학습되었습니다.")

    # [NEW] Local Transformers Model 섹션
    st.markdown("---")
    st.header("[NEW] Local Transformers Model (DB Upload Based)")

    # DB 파일 업로드 섹션
    uploaded_db = st.file_uploader("Upload Pattern Analysis DB (SQLite .db)", type=["db"])
    
    # 학습 상태를 저장할 session_state 변수들
    if 'training_status' not in st.session_state:
        st.session_state.training_status = None
    if 'db_uploaded' not in st.session_state:
        st.session_state.db_uploaded = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False

    # DB 업로드 처리
    if uploaded_db:
        with open("uploaded_pattern.db", "wb") as f:
            f.write(uploaded_db.read())
        st.session_state.db_uploaded = True
        st.success("DB upload completed! You can now train the model.")
    
    # 학습/재학습 버튼 컬럼
    col1, col2 = st.columns(2)
    
    with col1:
        # 초기 학습 버튼 (DB가 업로드된 경우에만 활성화)
        if st.button("Train Local Transformers Model", disabled=not st.session_state.db_uploaded):
            if not st.session_state.db_uploaded:
                st.warning("Please upload DB file first!")
            else:
                try:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    st.session_state.training_status = 'running'
                    status_text.text("Loading data...")
                    progress_bar.progress(10)
                    
                    with st.spinner("Training model..."):
                        status_text.text("Preparing tokenizer...")
                        progress_bar.progress(30)
                        
                        status_text.text("Training in progress...")
                        progress_bar.progress(50)
                        
                        local_clf = train_local_transformer_model_from_db("uploaded_pattern.db", force_retrain=True)
                        st.session_state.local_clf = local_clf
                        
                        status_text.text("Saving model...")
                        progress_bar.progress(90)
                        
                        progress_bar.progress(100)
                        status_text.text("Training completed!")
                        st.session_state.training_status = 'completed'
                        st.session_state.model_trained = True
                        st.success("Local Transformers Model training completed!")
                        
                        # 학습 완료 후 테이블 자동 업데이트
                        st.rerun()
                        
                except Exception as e:
                    st.session_state.training_status = 'failed'
                    st.error(f"Training error: {str(e)}")
    
                with col2:
        # 재학습 버튼 (모델이 이미 학습된 경우에만 활성화)
        if st.button("Retrain Model", disabled=not st.session_state.model_trained):
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                st.session_state.training_status = 'running'
                status_text.text("Retraining model...")
                progress_bar.progress(30)
                
                with st.spinner("Retraining in progress..."):
                    local_clf = train_local_transformer_model_from_db("uploaded_pattern.db", force_retrain=True)
                    st.session_state.local_clf = local_clf
                    
                    progress_bar.progress(100)
                    status_text.text("Retraining completed!")
                    st.session_state.training_status = 'completed'
                    st.success("Model retraining completed!")
                    
                    # 재학습 완료 후 테이블 자동 업데이트
                    st.rerun()
                    
            except Exception as e:
                st.session_state.training_status = 'failed'
                st.error(f"Retraining error: {str(e)}")

    # 학습 상태 표시
    if st.session_state.training_status == 'running':
        st.warning("Model training in progress. Please wait...")
    elif st.session_state.training_status == 'completed':
        st.success("Model is ready for predictions!")
    elif st.session_state.training_status == 'failed':
        st.error("Model training failed. Please try again.")

    # 예측 입력 필드 (모델이 준비된 경우에만 표시)
    if "local_clf" in st.session_state:
        st.info("Local Transformers Model is ready. Enter a pattern below for prediction.")
        user_pattern = st.text_input("Enter pattern to predict (e.g., aa, ab, ba, bb)", key="local_transformer_predict")
        if user_pattern:
            result = st.session_state.local_clf(user_pattern)
            st.markdown(f"**Local Transformers Prediction Result:** {result[0]['label']} (Confidence: {result[0]['score']:.1%})")
    
    # 예측 테이블 표시
    display_pattern_prediction_table(df)

if __name__ == "__main__":
    main() 