# import streamlit as st
import pandas as pd
import sqlite3
from collections import defaultdict
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

class PatternAnalyzer:
    def __init__(self, source_db_path, result_db_path):
        self.source_db_path = source_db_path
        self.result_db_path = result_db_path

    def analyze_sequences(self):
        # 결과 DB 초기화 (여기로 이동)
        with sqlite3.connect(self.result_db_path) as conn:
            conn.execute("DROP TABLE IF EXISTS pattern_analysis_results")
            conn.execute("DROP TABLE IF EXISTS analysis_metadata")
            conn.execute("""
                CREATE TABLE pattern_analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern TEXT NOT NULL,
                    pattern_length INTEGER NOT NULL,
                    frequency INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(pattern, pattern_length)
                )
            """)
            conn.execute("""
                CREATE TABLE analysis_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_date TIMESTAMP NOT NULL,
                    total_sessions INTEGER NOT NULL,
                    total_patterns INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
        # 소스 DB에서 데이터 읽기 및 결과 저장 기존 코드 유지
        with sqlite3.connect(self.source_db_path) as source_conn:
            df = pd.read_sql_query("""
                SELECT session_id, prediction_results 
                FROM session_prediction_results
                WHERE prediction_results IS NOT NULL
            """, source_conn)

        # 모든 세션의 패턴 빈도 누적
        all_patterns = defaultdict(int)
        for _, row in df.iterrows():
            sequence = row['prediction_results']
            patterns = self.extract_patterns(sequence)
            for pattern, frequency in patterns.items():
                all_patterns[(pattern, len(pattern))] += frequency

        # 결과 DB에 저장
        with sqlite3.connect(self.result_db_path) as result_conn:
            for (pattern, length), frequency in all_patterns.items():
                result_conn.execute("""
                    INSERT OR REPLACE INTO pattern_analysis_results 
                    (pattern, pattern_length, frequency)
                    VALUES (?, ?, ?)
                """, (pattern, length, frequency))
            # 메타데이터 저장
            result_conn.execute("""
                INSERT INTO analysis_metadata 
                (analysis_date, total_sessions, total_patterns)
                VALUES (?, ?, ?)
            """, (
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                len(df),
                len(all_patterns)
            ))
            result_conn.commit()

    def extract_patterns(self, sequence, min_length=1, max_length=10):
        """시퀀스에서 패턴을 추출"""
        patterns = defaultdict(int)
        n = len(sequence)
        
        # 각 길이에 대해
        for length in range(min_length, min(max_length + 1, n + 1)):
            # 각 시작 위치에 대해
            for start in range(n - length + 1):
                pattern = sequence[start:start + length]
                patterns[pattern] += 1
                
        return patterns

def load_pattern_data():
    """패턴 데이터 로드"""
    with sqlite3.connect('pattern_analysis_results.db') as conn:
        df = pd.read_sql_query("""
            SELECT pattern, pattern_length, frequency 
            FROM pattern_analysis_results 
            ORDER BY pattern_length, frequency DESC
        """, conn)
    return df

def load_metadata():
    """메타데이터 로드"""
    with sqlite3.connect('pattern_analysis_results.db') as conn:
        df = pd.read_sql_query("""
            SELECT * FROM analysis_metadata 
            ORDER BY analysis_date DESC
        """, conn)
    return df

def sort_group_id(df):
    """group_id를 앞 숫자 기준으로 정렬하는 함수"""
    return df.sort_values(by='group_id', key=lambda x: x.str.split('-').str[0].astype(int))

def load_group_prediction_data():
    """모든 group_id별 예측 결과 w/l 빈도 집계"""
    with sqlite3.connect('pattern_analysis_v2.db') as conn:
        # 전체 데이터 로드
        df = pd.read_sql_query("""
            SELECT 
                group_id,
                pattern12_prediction_result,
                pattern123_prediction_result
            FROM pattern_analysis
        """, conn)
        
        # 최근 1일 데이터 로드
        recent_df = pd.read_sql_query("""
            SELECT 
                group_id,
                pattern12_prediction_result,
                pattern123_prediction_result
            FROM pattern_analysis
            WHERE created_at >= datetime('now', '-1 day')
        """, conn)
        
        # group_id별로 집계
        result = []
        for group_id, group_df in df.groupby('group_id'):
            # 전체 데이터 집계
            pattern12_counts = group_df['pattern12_prediction_result'].value_counts().to_dict()
            pattern123_counts = group_df['pattern123_prediction_result'].value_counts().to_dict()
            
            # 패턴12 빈도 차이 계산
            pattern12_w = pattern12_counts.get('w', 0)
            pattern12_l = pattern12_counts.get('l', 0)
            pattern12_higher = 'W' if pattern12_w > pattern12_l else 'L' if pattern12_w < pattern12_l else '동일'
            
            # 패턴123 빈도 차이 계산
            pattern123_w = pattern123_counts.get('w', 0)
            pattern123_l = pattern123_counts.get('l', 0)
            pattern123_higher = 'W' if pattern123_w > pattern123_l else 'L' if pattern123_w < pattern123_l else '동일'
            
            # 최근 1일 데이터 집계
            recent_group_df = recent_df[recent_df['group_id'] == group_id]
            recent_pattern12_counts = recent_group_df['pattern12_prediction_result'].value_counts().to_dict()
            recent_pattern123_counts = recent_group_df['pattern123_prediction_result'].value_counts().to_dict()
            
            # 최근 1일 패턴12 빈도
            recent_pattern12_w = recent_pattern12_counts.get('w', 0)
            recent_pattern12_l = recent_pattern12_counts.get('l', 0)
            recent_pattern12_higher = 'W' if recent_pattern12_w > recent_pattern12_l else 'L' if recent_pattern12_w < recent_pattern12_l else '동일'
            
            # 최근 1일 패턴123 빈도
            recent_pattern123_w = recent_pattern123_counts.get('w', 0)
            recent_pattern123_l = recent_pattern123_counts.get('l', 0)
            recent_pattern123_higher = 'W' if recent_pattern123_w > recent_pattern123_l else 'L' if recent_pattern123_w < recent_pattern123_l else '동일'
            
            result.append({
                'group_id': group_id,
                '패턴 1-2 w': pattern12_w,
                '패턴 1-2 l': pattern12_l,
                '패턴 1-2 높은빈도': pattern12_higher,
                '패턴 1-2-3 w': pattern123_w,
                '패턴 1-2-3 l': pattern123_l,
                '패턴 1-2-3 높은빈도': pattern123_higher,
                '최근3일 1-2 w': recent_pattern12_w,
                '최근3일 1-2 l': recent_pattern12_l,
                '최근3일 1-2 높은빈도': recent_pattern12_higher,
                '최근3일 1-2-3 w': recent_pattern123_w,
                '최근3일 1-2-3 l': recent_pattern123_l,
                '최근3일 1-2-3 높은빈도': recent_pattern123_higher,
            })
        result_df = pd.DataFrame(result)
        result_df = sort_group_id(result_df)
        return result_df

def create_pattern_analysis_db():
    """패턴 분석 결과를 저장할 DB 생성"""
    with sqlite3.connect('pattern_analysis_results.db') as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_pattern_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                group_id TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                pattern_length INTEGER NOT NULL,
                pattern TEXT NOT NULL,
                frequency INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(session_id, group_id, pattern_type, pattern_length, pattern)
            )
        """)
        
        # 3자리 패턴 테이블 생성
        conn.execute("""
            CREATE TABLE IF NOT EXISTS three_char_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern TEXT NOT NULL,
                frequency INTEGER NOT NULL,
                ratio REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(pattern)
            )
        """)
        
        # 4자리 패턴 테이블 생성
        conn.execute("""
            CREATE TABLE IF NOT EXISTS four_char_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern TEXT NOT NULL,
                frequency INTEGER NOT NULL,
                ratio REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(pattern)
            )
        """)
        
        # 패턴 검증 통합 결과 테이블 생성
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pattern_validation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                validation_id TEXT NOT NULL,
                round_number INTEGER NOT NULL,
                search_pattern TEXT NOT NULL,
                predicted_pattern TEXT NOT NULL,
                predicted_char TEXT NOT NULL,
                actual_char TEXT NOT NULL,
                result TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                validation_result TEXT NOT NULL,
                game_result TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(validation_id, round_number, pattern_type)
            )
        """)
        
        # 검증 세션 시퀀스 테이블 생성
        conn.execute("""
            CREATE TABLE IF NOT EXISTS validation_sequences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                validation_id TEXT NOT NULL,
                sequence TEXT NOT NULL,
                win_count INTEGER NOT NULL,
                loss_count INTEGER NOT NULL,
                total_count INTEGER NOT NULL,
                win_rate REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(validation_id)
            )
        """)
        
        conn.commit()

def analyze_session_patterns():
    """
    session_id별 group_start 순서대로 예측 결과를 이어붙이고 길이별로 분석하는 함수
    
    처리 과정:
    1. session_pattern_analysis 테이블 생성
    2. pattern_analysis_v2.db에서 데이터 로드
    3. session_id별로 그룹화하여 처리
    4. 각 session의 pattern12와 pattern123 결과를 이어붙임
    5. 1~10 길이의 패턴을 추출하고 빈도수 계산
    6. 결과를 session_pattern_analysis 테이블에 저장
    
    저장되는 데이터:
    - session_id: 세션 식별자
    - group_id: 그룹 식별자
    - pattern_type: pattern12 또는 pattern123
    - pattern_length: 패턴 길이 (1~10)
    - pattern: 추출된 패턴
    - frequency: 패턴의 빈도수
    """
    # 1. 테이블 생성
    create_pattern_analysis_db()
    
    with sqlite3.connect('pattern_analysis_v2.db') as conn:
        # 2. 데이터 로드
        # session_id와 group_start 순서대로 정렬하여 로드
        df = pd.read_sql_query("""
            SELECT 
                session_id,
                group_id,
                group_start,
                pattern12_prediction_result,
                pattern123_prediction_result
            FROM pattern_analysis
            ORDER BY session_id, group_start
        """, conn)
        
        # 3. session_id별로 그룹화하여 처리
        session_patterns = {}
        for session_id, session_df in df.groupby('session_id'):
            # group_start 순서대로 정렬
            session_df = session_df.sort_values('group_start')
            
            # 4. pattern12와 pattern123 결과를 이어붙임 (대문자로 변환)
            # 예: ['W', 'L', 'W'] -> 'WLW'
            pattern12_sequence = ''.join(session_df['pattern12_prediction_result'].dropna()).upper()
            pattern123_sequence = ''.join(session_df['pattern123_prediction_result'].dropna()).upper()
            
            # 5. 각 길이별로 패턴 추출 및 빈도수 계산
            patterns = {}
            for length in range(1, 11):  # 1~10 길이의 패턴
                # pattern12 패턴 추출 및 빈도수 계산
                pattern12_patterns = {}
                for i in range(len(pattern12_sequence)-length+1):
                    pattern = pattern12_sequence[i:i+length]
                    pattern12_patterns[pattern] = pattern12_patterns.get(pattern, 0) + 1
                
                # pattern123 패턴 추출 및 빈도수 계산
                pattern123_patterns = {}
                for i in range(len(pattern123_sequence)-length+1):
                    pattern = pattern123_sequence[i:i+length]
                    pattern123_patterns[pattern] = pattern123_patterns.get(pattern, 0) + 1
                
                # 각 패턴 타입별로 결과 저장
                patterns[f'pattern12_{length}'] = pattern12_patterns
                patterns[f'pattern123_{length}'] = pattern123_patterns
            
            session_patterns[session_id] = patterns
        
        # 6. 결과를 DB에 저장
        with sqlite3.connect('pattern_analysis_results.db') as result_conn:
            for session_id, patterns in session_patterns.items():
                session_df = df[df['session_id'] == session_id]
                for _, row in session_df.iterrows():
                    group_id = row['group_id']
                    
                    # 각 길이별 패턴 빈도 계산 및 저장
                    for pattern_type in patterns:
                        length = int(pattern_type.split('_')[1])
                        pattern_dict = patterns[pattern_type]
                        
                        # 각 패턴과 그 빈도수를 저장
                        for pattern, frequency in pattern_dict.items():
                            result_conn.execute("""
                                INSERT OR REPLACE INTO session_pattern_analysis 
                                (session_id, group_id, pattern_type, pattern_length, pattern, frequency)
                                VALUES (?, ?, ?, ?, ?, ?)
                            """, (session_id, group_id, pattern_type, length, pattern, frequency))
            
            result_conn.commit()

def load_session_pattern_data():
    """저장된 패턴 분석 결과 로드"""
    with sqlite3.connect('pattern_analysis_results.db') as conn:
        df = pd.read_sql_query("""
            SELECT 
                group_id,
                pattern_type,
                pattern_length,
                pattern,
                frequency
            FROM session_pattern_analysis
            ORDER BY group_id, pattern_type, pattern_length, frequency DESC
        """, conn)
    return df

def add_game_result(validation_results):
    """3자리 패턴 검증 결과에 game_result 필드를 추가하는 함수"""
    for i, result in enumerate(validation_results):
        if i < 2:  # 첫 번째와 두 번째 라운드는 비교 대상이 없음
            result['game_result'] = '-'
        else:
            # 현재 라운드의 validation_result와 2번째 이전 라운드의 validation_result 비교
            current_result = result['validation_result']
            previous_result = validation_results[i-2]['validation_result']
            
            # validation_result가 '-'인 경우 game_result도 '-'로 처리
            if current_result == '-' or previous_result == '-':
                result['game_result'] = '-'
            elif current_result == previous_result:
                result['game_result'] = 'P'
            else:
                result['game_result'] = 'F'
    
    return validation_results

def validate_pattern_3char(actual_pattern, three_char_df):
    """3자리 패턴 검증 함수"""
    validation_results = []
    
    # 각 라운드별 검증 수행
    for i in range(len(actual_pattern) - 2):
        # 현재 검색할 2자리 패턴
        search_pattern = actual_pattern[i:i+2]
        
        # 최근 1일 데이터에서 검색 패턴으로 시작하는 3자리 패턴 찾기
        with sqlite3.connect('pattern_analysis_v2.db') as conn:
            recent_df = pd.read_sql_query("""
                SELECT session_id, prediction_results 
                FROM session_prediction_results 
                WHERE prediction_results IS NOT NULL
                AND created_at >= datetime('now', '-1 day')
            """, conn)
        
        # 패턴 추출 및 분석
        recent_patterns = defaultdict(int)
        for _, row in recent_df.iterrows():
            sequence = row['prediction_results']
            if sequence:
                # 각 길이에 대해
                for length in range(1, 4):  # 1~3 길이의 패턴
                    # 각 시작 위치에 대해
                    for start in range(len(sequence) - length + 1):
                        pattern = sequence[start:start + length]
                        if pattern.startswith(search_pattern) and len(pattern) == 3:
                            recent_patterns[pattern] += 1
        
        # 빈도순으로 정렬된 패턴 리스트 생성
        matching_patterns = pd.DataFrame({
            'pattern': list(recent_patterns.keys()),
            'frequency': list(recent_patterns.values())
        }).sort_values('frequency', ascending=False)
        
        if not matching_patterns.empty:
            # 가장 높은 빈도의 패턴 선택
            best_pattern = matching_patterns.iloc[0]['pattern']
            predicted_char = best_pattern[2]
            actual_char = actual_pattern[i+2]
            
            # 빈도 차이 계산
            if len(matching_patterns) >= 2:
                top_freq = matching_patterns.iloc[0]['frequency']
                second_freq = matching_patterns.iloc[1]['frequency']
                freq_diff = top_freq - second_freq
                
                # 빈도 차이가 10 미만이면 예측 문자를 'P'로 변경
                if freq_diff < 10:
                    predicted_char = 'P'
            
            # 결과 비교
            if predicted_char == 'P':
                result = 'P'
            else:
                result = 'W' if predicted_char.upper() == actual_char else 'L'
            
            # 상위 3개 패턴의 빈도 정보 수집
            top_patterns_info = []
            for idx, row in matching_patterns.head(3).iterrows():
                top_patterns_info.append(f"{row['pattern']}({row['frequency']})")
            
            validation_results.append({
                '라운드': i + 1,
                '검색패턴': search_pattern,
                '예측패턴': best_pattern,
                '예측문자': predicted_char,
                '실제문자': actual_char,
                '결과': result,
                '빈도정보': ' | '.join(top_patterns_info)  # 상위 3개 패턴의 빈도 정보
            })
    
    # 마지막 라운드에 대한 예측 패턴 추가
    if len(actual_pattern) >= 2:
        last_search = actual_pattern[-2:]
        # 최근 1일 데이터에서 검색 패턴으로 시작하는 3자리 패턴 찾기
        with sqlite3.connect('pattern_analysis_v2.db') as conn:
            recent_df = pd.read_sql_query("""
                SELECT session_id, prediction_results 
                FROM session_prediction_results 
                WHERE prediction_results IS NOT NULL
                AND created_at >= datetime('now', '-1 day')
            """, conn)
        
        # 패턴 추출 및 분석
        recent_patterns = defaultdict(int)
        for _, row in recent_df.iterrows():
            sequence = row['prediction_results']
            if sequence:
                # 각 길이에 대해
                for length in range(1, 4):  # 1~3 길이의 패턴
                    # 각 시작 위치에 대해
                    for start in range(len(sequence) - length + 1):
                        pattern = sequence[start:start + length]
                        if pattern.startswith(last_search) and len(pattern) == 3:
                            recent_patterns[pattern] += 1
        
        # 빈도순으로 정렬된 패턴 리스트 생성
        matching_patterns = pd.DataFrame({
            'pattern': list(recent_patterns.keys()),
            'frequency': list(recent_patterns.values())
        }).sort_values('frequency', ascending=False)
        
        if not matching_patterns.empty:
            best_pattern = matching_patterns.iloc[0]['pattern']
            predicted_char = best_pattern[2]
            
            # 빈도 차이 계산
            if len(matching_patterns) >= 2:
                top_freq = matching_patterns.iloc[0]['frequency']
                second_freq = matching_patterns.iloc[1]['frequency']
                freq_diff = top_freq - second_freq
                
                # 빈도 차이가 10 미만이면 예측 문자를 'P'로 변경
                if freq_diff < 10:
                    predicted_char = 'P'
            
            # 상위 3개 패턴의 빈도 정보 수집
            top_patterns_info = []
            for idx, row in matching_patterns.head(3).iterrows():
                top_patterns_info.append(f"{row['pattern']}({row['frequency']})")
            
            validation_results.append({
                '라운드': len(actual_pattern) - 1,
                '검색패턴': last_search,
                '예측패턴': best_pattern,
                '예측문자': predicted_char,
                '실제문자': '-',
                '결과': '-',
                '빈도정보': ' | '.join(top_patterns_info)  # 상위 3개 패턴의 빈도 정보
            })
    
    # validation_result 필드 추가
    validation_results = add_validation_result(validation_results)
    
    # game_result 필드 추가
    validation_results = add_game_result(validation_results)
    
    return validation_results

def validate_pattern_4char(actual_pattern, four_char_df):
    """4자리 패턴 검증 함수"""
    validation_results = []
    
    # 각 라운드별 검증 수행
    for i in range(len(actual_pattern) - 3):
        # 현재 검색할 3자리 패턴
        search_pattern = actual_pattern[i:i+3]
        
        # 최근 1일 데이터에서 검색 패턴으로 시작하는 4자리 패턴 찾기
        with sqlite3.connect('pattern_analysis_v2.db') as conn:
            recent_df = pd.read_sql_query("""
                SELECT session_id, prediction_results 
                FROM session_prediction_results 
                WHERE prediction_results IS NOT NULL
                AND created_at >= datetime('now', '-1 day')
            """, conn)
        
        # 패턴 추출 및 분석
        recent_patterns = defaultdict(int)
        for _, row in recent_df.iterrows():
            sequence = row['prediction_results']
            if sequence:
                # 각 길이에 대해
                for length in range(1, 5):  # 1~4 길이의 패턴
                    # 각 시작 위치에 대해
                    for start in range(len(sequence) - length + 1):
                        pattern = sequence[start:start + length]
                        if pattern.startswith(search_pattern) and len(pattern) == 4:
                            recent_patterns[pattern] += 1
        
        # 빈도순으로 정렬된 패턴 리스트 생성
        matching_patterns = pd.DataFrame({
            'pattern': list(recent_patterns.keys()),
            'frequency': list(recent_patterns.values())
        }).sort_values('frequency', ascending=False)
        
        if not matching_patterns.empty:
            # 가장 높은 빈도의 패턴 선택
            best_pattern = matching_patterns.iloc[0]['pattern']
            predicted_char = best_pattern[3]
            actual_char = actual_pattern[i+3]
            
            # 빈도 차이 계산
            if len(matching_patterns) >= 2:
                top_freq = matching_patterns.iloc[0]['frequency']
                second_freq = matching_patterns.iloc[1]['frequency']
                freq_diff = top_freq - second_freq
                
                # 빈도 차이가 10 미만이면 예측 문자를 'P'로 변경
                if freq_diff < 10:
                    predicted_char = 'P'
            
            # 결과 비교
            if predicted_char == 'P':
                result = 'P'
            else:
                result = 'W' if predicted_char.upper() == actual_char else 'L'
            
            # 상위 3개 패턴의 빈도 정보 수집
            top_patterns_info = []
            for idx, row in matching_patterns.head(3).iterrows():
                top_patterns_info.append(f"{row['pattern']}({row['frequency']})")
            
            validation_results.append({
                '라운드': i + 2,
                '검색패턴': search_pattern,
                '예측패턴': best_pattern,
                '예측문자': predicted_char,
                '실제문자': actual_char,
                '결과': result,
                '빈도정보': ' | '.join(top_patterns_info)  # 상위 3개 패턴의 빈도 정보
            })
    
    # 마지막 라운드에 대한 예측 패턴 추가
    if len(actual_pattern) >= 3:
        last_search = actual_pattern[-3:]
        # 최근 1일 데이터에서 검색 패턴으로 시작하는 4자리 패턴 찾기
        with sqlite3.connect('pattern_analysis_v2.db') as conn:
            recent_df = pd.read_sql_query("""
                SELECT session_id, prediction_results 
                FROM session_prediction_results 
                WHERE prediction_results IS NOT NULL
                AND created_at >= datetime('now', '-1 day')
            """, conn)
        
        # 패턴 추출 및 분석
        recent_patterns = defaultdict(int)
        for _, row in recent_df.iterrows():
            sequence = row['prediction_results']
            if sequence:
                # 각 길이에 대해
                for length in range(1, 5):  # 1~4 길이의 패턴
                    # 각 시작 위치에 대해
                    for start in range(len(sequence) - length + 1):
                        pattern = sequence[start:start + length]
                        if pattern.startswith(last_search) and len(pattern) == 4:
                            recent_patterns[pattern] += 1
        
        # 빈도순으로 정렬된 패턴 리스트 생성
        matching_patterns = pd.DataFrame({
            'pattern': list(recent_patterns.keys()),
            'frequency': list(recent_patterns.values())
        }).sort_values('frequency', ascending=False)
        
        if not matching_patterns.empty:
            best_pattern = matching_patterns.iloc[0]['pattern']
            predicted_char = best_pattern[3]
            
            # 빈도 차이 계산
            if len(matching_patterns) >= 2:
                top_freq = matching_patterns.iloc[0]['frequency']
                second_freq = matching_patterns.iloc[1]['frequency']
                freq_diff = top_freq - second_freq
                
                # 빈도 차이가 10 미만이면 예측 문자를 'P'로 변경
                if freq_diff < 10:
                    predicted_char = 'P'
            
            # 상위 3개 패턴의 빈도 정보 수집
            top_patterns_info = []
            for idx, row in matching_patterns.head(3).iterrows():
                top_patterns_info.append(f"{row['pattern']}({row['frequency']})")
            
            validation_results.append({
                '라운드': len(actual_pattern) - 1,
                '검색패턴': last_search,
                '예측패턴': best_pattern,
                '예측문자': predicted_char,
                '실제문자': '-',
                '결과': '-',
                '빈도정보': ' | '.join(top_patterns_info)  # 상위 3개 패턴의 빈도 정보
            })
    
    # validation_result 필드 추가
    validation_results = add_validation_result(validation_results)
    
    # game_result 필드 추가
    validation_results = add_game_result(validation_results)
    
    return validation_results

def display_validation_results(validation_results, title):
    """검증 결과 표시 함수"""
    if validation_results:
        results_df = pd.DataFrame(validation_results)
        results_df = results_df.sort_values('라운드', ascending=False)
        st.write(title)
        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "라운드": st.column_config.NumberColumn("라운드", format="%d"),
                "검색패턴": st.column_config.TextColumn("검색패턴"),
                "예측패턴": st.column_config.TextColumn("예측패턴"),
                "예측문자": st.column_config.TextColumn("예측문자"),
                "실제문자": st.column_config.TextColumn("실제문자"),
                "결과": st.column_config.TextColumn("결과"),
                "빈도정보": st.column_config.TextColumn("빈도정보")  # 빈도 정보 컬럼 추가
            }
        )
        
        # 승률 계산 (마지막 라운드 제외)
        valid_results = [r for r in validation_results if r['결과'] != '-']
        wins = sum(1 for r in valid_results if r['결과'] == 'P')
        total = len(valid_results)
        win_rate = (wins / total * 100) if total > 0 else 0
        st.write(f"승률: {win_rate:.2f}% ({wins}/{total})")
        
        return results_df

def add_validation_result(validation_results):
    """3자리 패턴 검증 결과에 validation_result 필드를 추가하는 함수"""
    for result in validation_results:
        if result['결과'] != '-':  # 마지막 라운드가 아닌 경우만 처리
            # 예측 패턴의 마지막 문자와 실제 문자 비교
            predicted_char = result['예측패턴'][-1] if len(result['예측패턴']) >= 3 else result['예측문자']
            validation_result = 'W' if predicted_char.upper() == result['실제문자'] else 'L'
            result['validation_result'] = validation_result
        else:
            result['validation_result'] = '-'
    return validation_results

def analyze_start_patterns(results_df):
    """시작 패턴 분석 함수"""
    import streamlit as st
    
    # 시작 문자별 빈도 분석
    start_char_counts = results_df['시작문자'].value_counts()
    total_sequences = len(results_df)
    
    # 시작 문자 비율 계산
    start_char_ratios = (start_char_counts / total_sequences * 100).round(2)
    
    # 시작 문자와 연속수 조합 분석
    pattern_combinations = results_df.groupby(['시작문자', '시작연속수']).size().reset_index(name='빈도')
    pattern_combinations['비율'] = (pattern_combinations['빈도'] / total_sequences * 100).round(2)
    pattern_combinations = pattern_combinations.sort_values('빈도', ascending=False)
    
    # 결과 표시
    st.subheader("시작 패턴 분석")
    
    # 시작 문자별 빈도와 비율 표시
    st.write("시작 문자별 빈도")
    char_stats = pd.DataFrame({
        '시작문자': start_char_counts.index,
        '빈도': start_char_counts.values,
        '비율(%)': start_char_ratios.values
    })
    st.dataframe(
        char_stats,
        use_container_width=True,
        hide_index=True,
        column_config={
            "시작문자": st.column_config.TextColumn("시작문자"),
            "빈도": st.column_config.NumberColumn("빈도", format="%d"),
            "비율(%)": st.column_config.NumberColumn("비율(%)", format="%.2f")
        }
    )
    
    # 시작 문자와 연속수 조합 표시
    st.write("시작 문자와 연속수 조합 분석")
    st.dataframe(
        pattern_combinations,
        use_container_width=True,
        hide_index=True,
        column_config={
            "시작문자": st.column_config.TextColumn("시작문자"),
            "시작연속수": st.column_config.NumberColumn("시작연속수", format="%d"),
            "빈도": st.column_config.NumberColumn("빈도", format="%d"),
            "비율(%)": st.column_config.NumberColumn("비율(%)", format="%.2f")
        }
    )

def analyze_consecutive_patterns():
    """연속되는 F와 P 패턴을 분석하는 함수"""
    import streamlit as st
    
    with sqlite3.connect('pattern_analysis_results.db') as conn:
        # game_result_sequences 테이블에서 sequence 데이터 로드
        df = pd.read_sql_query("""
            SELECT validation_id, sequence 
            FROM game_result_sequences 
            WHERE sequence IS NOT NULL
        """, conn)
        
        if df.empty:
            st.warning("분석할 데이터가 없습니다.")
            return
        
        # 각 시퀀스별로 연속 패턴 분석
        results = []
        for _, row in df.iterrows():
            sequence = row['sequence']
            validation_id = row['validation_id']
            
            # 연속된 F와 P 패턴 찾기
            current_char = None
            count = 0
            patterns = []
            max_f_count = 0
            max_p_count = 0
            
            # 시작 문자 분석 (- 문자 제외)
            start_char = None
            start_count = 0
            for char in sequence:
                if char != '-':
                    start_char = char
                    break
            
            if start_char:
                for char in sequence:
                    if char == start_char:
                        start_count += 1
                    elif char != '-':  # -가 아닌 다른 문자가 나오면 중단
                        break
            
            for char in sequence:
                if char == current_char:
                    count += 1
                else:
                    if count > 0 and current_char != '-':
                        if current_char == 'F':
                            max_f_count = max(max_f_count, count)
                        elif current_char == 'P':
                            max_p_count = max(max_p_count, count)
                    current_char = char
                    count = 1
            
            # 마지막 패턴 처리
            if count > 0 and current_char != '-':
                if current_char == 'F':
                    max_f_count = max(max_f_count, count)
                elif current_char == 'P':
                    max_p_count = max(max_p_count, count)
            
            # 결과 저장
            results.append({
                'validation_id': validation_id,
                'sequence': sequence,
                '최대연속F': max_f_count,
                '최대연속P': max_p_count,
                '시작문자': start_char if start_char else '-',
                '시작연속수': start_count
            })
        
        # 결과 표시
        if results:
            st.subheader("연속 패턴 분석 결과")
            
            # 전체 레코드 수 표시
            st.write(f"총 레코드 수: {len(results)}개")
            
            # 최대값 계산
            results_df = pd.DataFrame(results)
            max_consecutive_f = results_df['최대연속F'].max()
            max_consecutive_p = results_df['최대연속P'].max()
            
            # 최대값 표시
            st.write(f"최대 연속 F: {max_consecutive_f}회, 최대 연속 P: {max_consecutive_p}회")
            
            # 데이터프레임 표시
            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=False,  # 인덱스 표시
                column_config={
                    "validation_id": st.column_config.TextColumn("검증 ID"),
                    "sequence": st.column_config.TextColumn("시퀀스"),
                    "최대연속F": st.column_config.NumberColumn("최대연속F", format="%d"),
                    "최대연속P": st.column_config.NumberColumn("최대연속P", format="%d"),
                    "시작문자": st.column_config.TextColumn("시작문자"),
                    "시작연속수": st.column_config.NumberColumn("시작연속수", format="%d")
                }
            )
            
            # 시작 패턴 분석 실행
            analyze_start_patterns(results_df)

def main():
    import streamlit as st
    st.title("패턴 분석 결과 뷰어")
    
    # Initialize validation_id
    validation_id = None
    
    def display_validation_results(validation_results, title):
        """검증 결과 표시 함수"""
        if validation_results:
            results_df = pd.DataFrame(validation_results)
            results_df = results_df.sort_values('라운드', ascending=False)
            st.write(title)
            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "라운드": st.column_config.NumberColumn("라운드", format="%d"),
                    "검색패턴": st.column_config.TextColumn("검색패턴"),
                    "예측패턴": st.column_config.TextColumn("예측패턴"),
                    "예측문자": st.column_config.TextColumn("예측문자"),
                    "실제문자": st.column_config.TextColumn("실제문자"),
                    "결과": st.column_config.TextColumn("결과"),
                    "빈도정보": st.column_config.TextColumn("빈도정보")  # 빈도 정보 컬럼 추가
                }
            )
            
            # 승률 계산 (마지막 라운드 제외)
            valid_results = [r for r in validation_results if r['결과'] != '-']
            wins = sum(1 for r in valid_results if r['결과'] == 'P')
            total = len(valid_results)
            win_rate = (wins / total * 100) if total > 0 else 0
            st.write(f"승률: {win_rate:.2f}% ({wins}/{total})")
            
            return results_df
    
    def display_combined_results(validation_results_3char, validation_results_4char):
        """통합 검증 결과 표시 함수"""
        if validation_results_3char or validation_results_4char:
            # 3자리 패턴 검증 결과에서 홀수 라운드만 추출
            odd_rounds_3char = [r for r in validation_results_3char if r['라운드'] % 2 == 1]
            
            # 4자리 패턴 검증 결과에서 짝수 라운드만 추출
            even_rounds_4char = [r for r in validation_results_4char if r['라운드'] % 2 == 0]
            
            # 결과 통합
            combined_results = odd_rounds_3char + even_rounds_4char
            
            # 라운드 기준으로 정렬 (표시용)
            display_results = sorted(combined_results, key=lambda x: x['라운드'], reverse=True)
            
            # 데이터프레임 생성 및 표시
            results_df = pd.DataFrame(display_results)
            st.write("패턴 검증 통합 결과")
            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "라운드": st.column_config.NumberColumn("라운드", format="%d"),
                    "검색패턴": st.column_config.TextColumn("검색패턴"),
                    "예측패턴": st.column_config.TextColumn("예측패턴"),
                    "예측문자": st.column_config.TextColumn("예측문자"),
                    "실제문자": st.column_config.TextColumn("실제문자"),
                    "결과": st.column_config.TextColumn("결과")
                }
            )
            
            # 승률 계산 (마지막 라운드 제외)
            valid_results = [r for r in combined_results if r['결과'] != '-']
            wins = sum(1 for r in valid_results if r['결과'] == 'P')
            total = len(valid_results)
            win_rate = (wins / total * 100) if total > 0 else 0
            st.write(f"승률: {win_rate:.2f}% ({wins}/{total})")
            
            # 저장 버튼 추가
            if st.button("검증 결과 저장"):
                validation_id = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # 결과 시퀀스 생성 (라운드 순서대로)
                sequence = ''.join([r['결과'] for r in sorted(valid_results, key=lambda x: x['라운드'])])
                
                # game_result 시퀀스 생성 (라운드 순서대로)
                game_result_sequence = ''.join([r['game_result'] for r in sorted(combined_results, key=lambda x: x['라운드'])])
                
                # 3자리 패턴 결과 저장
                for result in odd_rounds_3char:
                    with sqlite3.connect('pattern_analysis_results.db') as conn:
                        conn.execute("""
                            INSERT OR REPLACE INTO pattern_validation_results 
                            (validation_id, round_number, search_pattern, predicted_pattern, 
                             predicted_char, actual_char, result, pattern_type, validation_result, game_result)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            validation_id,
                            result['라운드'],
                            result['검색패턴'],
                            result['예측패턴'],
                            result['예측문자'],
                            result['실제문자'],
                            result['결과'],
                            '3char',
                            result['validation_result'],
                            result['game_result']
                        ))
                
                # 4자리 패턴 결과 저장
                for result in even_rounds_4char:
                    with sqlite3.connect('pattern_analysis_results.db') as conn:
                        conn.execute("""
                            INSERT OR REPLACE INTO pattern_validation_results 
                            (validation_id, round_number, search_pattern, predicted_pattern, 
                             predicted_char, actual_char, result, pattern_type, validation_result, game_result)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            validation_id,
                            result['라운드'],
                            result['검색패턴'],
                            result['예측패턴'],
                            result['예측문자'],
                            result['실제문자'],
                            result['결과'],
                            '4char',
                            result['validation_result'],
                            result['game_result']
                        ))
                
                # 검증 시퀀스 저장
                with sqlite3.connect('pattern_analysis_results.db') as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO validation_sequences 
                        (validation_id, sequence, win_count, loss_count, total_count, win_rate)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        validation_id,
                        sequence,
                        wins,
                        total - wins,
                        total,
                        win_rate
                    ))
                    
                    # game_result 시퀀스 저장
                    conn.execute("""
                        INSERT OR REPLACE INTO game_result_sequences 
                        (validation_id, sequence)
                        VALUES (?, ?)
                    """, (
                        validation_id,
                        game_result_sequence
                    ))
                
                st.success("검증 결과가 저장되었습니다!")
    
    # 분석기 초기화
    analyzer = PatternAnalyzer('pattern_analysis_v2.db', 'pattern_analysis_results.db')
    
    # 분석 새로고침 버튼
    if st.button("분석 새로고침"):
        analyzer.analyze_sequences()
        analyze_session_patterns()
        st.success("패턴 분석이 완료되었습니다!")
    
    # 패턴 데이터 로드
    df = load_pattern_data()
    
    # 메타데이터 표시 (상단으로 이동)
    st.subheader("분석 메타데이터")
    metadata = load_metadata()
    if not metadata.empty:
        st.dataframe(metadata, use_container_width=True)
    
    # 패턴 검증 도구
    st.subheader("패턴 검증 도구")
    
    # 실제 패턴 결과 입력
    actual_pattern = st.text_input("실제 패턴 결과를 입력하세요 (예: WWLWW)", "")
    
    if actual_pattern:
        # 대소문자 구분 없이 처리
        actual_pattern = actual_pattern.upper()
        
        # 최근 24시간 3자리 패턴 데이터 로드
        with sqlite3.connect('pattern_analysis_v2.db') as conn:
            recent_df = pd.read_sql_query("""
                SELECT prediction_results 
                FROM session_prediction_results 
                WHERE prediction_results IS NOT NULL
                AND created_at >= datetime('now', '-1 day')
            """, conn)
            
            if not recent_df.empty:
                # 앞3자리 추출
                recent_df['first_three'] = recent_df['prediction_results'].str[:3]
                # null 값 제거
                recent_df = recent_df.dropna(subset=['first_three'])
                # 3자리가 아닌 케이스 제거
                recent_df = recent_df[recent_df['first_three'].str.len() == 3]
                # 집계
                counts = recent_df['first_three'].value_counts()
                total = len(recent_df)
                # 비율 계산
                ratios = (counts / total * 100).round(2)
                # 결과 데이터프레임 생성
                three_char_df = pd.DataFrame({
                    'pattern': counts.index,
                    'frequency': counts.values,
                    'ratio': ratios.values
                })
                
                # 앞4자리 추출
                recent_df['first_four'] = recent_df['prediction_results'].str[:4]
                # null 값 제거
                recent_df = recent_df.dropna(subset=['first_four'])
                # 4자리가 아닌 케이스 제거
                recent_df = recent_df[recent_df['first_four'].str.len() == 4]
                # 집계
                counts = recent_df['first_four'].value_counts()
                total = len(recent_df)
                # 비율 계산
                ratios = (counts / total * 100).round(2)
                # 결과 데이터프레임 생성
                four_char_df = pd.DataFrame({
                    'pattern': counts.index,
                    'frequency': counts.values,
                    'ratio': ratios.values
                })
        
        # 3자리 패턴 검증
        validation_results_3char = validate_pattern_3char(actual_pattern, three_char_df)
        
        # 4자리 패턴 검증
        validation_results_4char = validate_pattern_4char(actual_pattern, four_char_df)
        
        # 통합 결과 표시
        display_combined_results(validation_results_3char, validation_results_4char)
        
        # 3자리와 4자리 결과를 수평으로 배치
        col1, col2 = st.columns(2)
        with col1:
            display_validation_results(validation_results_3char, "3자리 패턴 검증 결과")
        with col2:
            display_validation_results(validation_results_4char, "4자리 패턴 검증 결과")

    st.subheader("패턴 검색 및 분석")
    
    # 패턴 검색 기능
    search_pattern = st.text_input("검색할 패턴을 입력하세요", "")
    
    if search_pattern and len(search_pattern) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("전체 데이터 검색 결과")
            # 전체 데이터 검색
            pattern_length = len(search_pattern)
            mask = df['pattern'].str.lower().str.startswith(search_pattern.lower(), na=False)
            next_patterns = df[mask & (df['pattern_length'] == pattern_length + 1)]
            
            if not next_patterns.empty:
                next_patterns = next_patterns.sort_values('frequency', ascending=False)
                st.dataframe(next_patterns, use_container_width=True)
                
                # 추천 패턴
                recommended = next_patterns.iloc[0]
                st.success(f"추천 패턴: {recommended['pattern']} (빈도: {recommended['frequency']})")
            else:
                st.warning(f"'{search_pattern}'로 시작하는 다음 패턴이 없습니다.")
        
        with col2:
            st.write("최근 1일 데이터 검색 결과")
            # 최근 1일 데이터 로드
            with sqlite3.connect('pattern_analysis_v2.db') as conn:
                recent_df = pd.read_sql_query("""
                    SELECT session_id, prediction_results 
                    FROM session_prediction_results 
                    WHERE prediction_results IS NOT NULL
                    AND created_at >= datetime('now', '-1 day')
                """, conn)
            
            # 패턴 추출 및 분석
            recent_patterns = defaultdict(int)
            for _, row in recent_df.iterrows():
                sequence = row['prediction_results']
                patterns = PatternAnalyzer.extract_patterns(None, sequence)
                for pattern, frequency in patterns.items():
                    recent_patterns[(pattern, len(pattern))] += frequency
            
            # 검색 패턴으로 시작하는 다음 패턴 찾기
            pattern_length = len(search_pattern)
            recent_next_patterns = []
            for (pattern, length), frequency in recent_patterns.items():
                if pattern.lower().startswith(search_pattern.lower()) and length == pattern_length + 1:
                    recent_next_patterns.append({
                        'pattern': pattern,
                        'pattern_length': length,
                        'frequency': frequency
                    })
            
            if recent_next_patterns:
                recent_df = pd.DataFrame(recent_next_patterns)
                recent_df = recent_df.sort_values('frequency', ascending=False)
                st.dataframe(recent_df, use_container_width=True)
                
                # 추천 패턴
                recent_recommended = recent_df.iloc[0]
                st.success(f"최근 1일 추천 패턴: {recent_recommended['pattern']} (빈도: {recent_recommended['frequency']})")
            else:
                st.warning(f"최근 1일 데이터에서 '{search_pattern}'로 시작하는 다음 패턴이 없습니다.")
    
    # group_id별 예측 결과 빈도 표시 (하단에 추가)
    st.subheader("group_id별 예측 결과(w/l) 빈도")
    freq_df = load_group_prediction_data()
    if not freq_df.empty:
        # 컬럼 그룹화를 위한 설정
        col_groups = {
            '전체 데이터': ['패턴 1-2 w', '패턴 1-2 l', '패턴 1-2 높은빈도', '패턴 1-2-3 w', '패턴 1-2-3 l', '패턴 1-2-3 높은빈도'],
            '최근 1일': ['최근3일 1-2 w', '최근3일 1-2 l', '최근3일 1-2 높은빈도', '최근3일 1-2-3 w', '최근3일 1-2-3 l', '최근3일 1-2-3 높은빈도']
        }
        
        # 컬럼 순서 재정렬
        ordered_columns = ['group_id'] + [col for group in col_groups.values() for col in group]
        freq_df = freq_df[ordered_columns]
        
        # 컬럼 이름 변경
        column_mapping = {
            '패턴 1-2 w': '전체 1-2 W',
            '패턴 1-2 l': '전체 1-2 L',
            '패턴 1-2 높은빈도': '전체 1-2 높은빈도',
            '패턴 1-2-3 w': '전체 1-2-3 W',
            '패턴 1-2-3 l': '전체 1-2-3 L',
            '패턴 1-2-3 높은빈도': '전체 1-2-3 높은빈도',
            '최근 1-2 W': '최근 1-2 W',
            '최근 1-2 L': '최근 1-2 L',
            '최근 1-2 높은빈도': '최근 1-2 높은빈도',
            '최근 1-2-3 W': '최근 1-2-3 W',
            '최근 1-2-3 L': '최근 1-2-3 L',
            '최근 1-2-3 높은빈도': '최근 1-2-3 높은빈도'
        }
        freq_df = freq_df.rename(columns=column_mapping)
        
        # 컬럼 너비를 40~50px로 더 좁게 설정
        column_config = {
            'group_id': st.column_config.TextColumn('Group ID', width=50),
            '전체 1-2 W': st.column_config.NumberColumn('전체 1-2 W', width=40),
            '전체 1-2 L': st.column_config.NumberColumn('전체 1-2 L', width=40),
            '전체 1-2 높은빈도': st.column_config.TextColumn('전체 1-2 높은빈도', width=50),
            '전체 1-2-3 W': st.column_config.NumberColumn('전체 1-2-3 W', width=40),
            '전체 1-2-3 L': st.column_config.NumberColumn('전체 1-2-3 L', width=40),
            '전체 1-2-3 높은빈도': st.column_config.TextColumn('전체 1-2-3 높은빈도', width=50),
            '최근 1-2 W': st.column_config.NumberColumn('최근 1-2 W', width=40),
            '최근 1-2 L': st.column_config.NumberColumn('최근 1-2 L', width=40),
            '최근 1-2 높은빈도': st.column_config.TextColumn('최근 1-2 높은빈도', width=50),
            '최근 1-2-3 W': st.column_config.NumberColumn('최근 1-2-3 W', width=40),
            '최근 1-2-3 L': st.column_config.NumberColumn('최근 1-2-3 L', width=40),
            '최근 1-2-3 높은빈도': st.column_config.TextColumn('최근 1-2-3 높은빈도', width=50)
        }
        
        # 데이터프레임 표시
        st.dataframe(
            freq_df,
            use_container_width=True,
            column_config=column_config,
            hide_index=True
        )
    else:
        st.warning("예측 결과 데이터가 없습니다.")

    # ----------------------
    # 통합 패턴 예측 테이블
    # ----------------------
    st.subheader("패턴 예측 테이블")
    
    # 업데이트 버튼 추가
    update_button = st.button("테이블 업데이트", key="update_table")
    
    if update_button:
        # 3자리 패턴 목록 (2자리 검색 패턴)
        three_char_patterns = ['WW', 'WL', 'LL', 'LW']
        
        # 4자리 패턴 목록 (3자리 검색 패턴)
        four_char_patterns = ['WWW', 'WWL', 'WLL', 'WLW', 'LLL', 'LLW', 'LWL', 'LWW']
        
        # 최근 1일 데이터에서 패턴 분석
        with sqlite3.connect('pattern_analysis_v2.db') as conn:
            recent_df = pd.read_sql_query("""
                SELECT prediction_results 
                FROM session_prediction_results 
                WHERE prediction_results IS NOT NULL
                AND created_at >= datetime('now', '-1 day')
            """, conn)
        
        # 결과 저장 리스트
        all_results = []
        
        # 3자리 패턴 분석
        for pattern in three_char_patterns:
            next_chars = defaultdict(int)
            for _, row in recent_df.iterrows():
                sequence = row['prediction_results']
                if sequence:
                    for i in range(len(sequence) - 2):
                        if sequence[i:i+2] == pattern:
                            next_char = sequence[i+2]
                            next_chars[next_char] += 1
            
            if next_chars:
                sorted_chars = sorted(next_chars.items(), key=lambda x: x[1], reverse=True)
                top_char = sorted_chars[0]
                second_char = sorted_chars[1] if len(sorted_chars) > 1 else (None, 0)
                
                all_results.append({
                    '검색패턴': pattern,
                    '예측문자': top_char[0],
                    '빈도': top_char[1],
                    '차이': top_char[1] - second_char[1] if second_char[0] else top_char[1],
                    '패턴유형': '3자리'
                })
        
        # 4자리 패턴 분석
        for pattern in four_char_patterns:
            next_chars = defaultdict(int)
            for _, row in recent_df.iterrows():
                sequence = row['prediction_results']
                if sequence:
                    for i in range(len(sequence) - 3):
                        if sequence[i:i+3] == pattern:
                            next_char = sequence[i+3]
                            next_chars[next_char] += 1
            
            if next_chars:
                sorted_chars = sorted(next_chars.items(), key=lambda x: x[1], reverse=True)
                top_char = sorted_chars[0]
                second_char = sorted_chars[1] if len(sorted_chars) > 1 else (None, 0)
                
                all_results.append({
                    '검색패턴': pattern,
                    '예측문자': top_char[0],
                    '빈도': top_char[1],
                    '차이': top_char[1] - second_char[1] if second_char[0] else top_char[1],
                    '패턴유형': '4자리'
                })
        
        # 결과 표시
        if all_results:
            st.dataframe(
                pd.DataFrame(all_results),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "검색패턴": st.column_config.TextColumn("검색패턴"),
                    "예측문자": st.column_config.TextColumn("예측문자"),
                    "빈도": st.column_config.NumberColumn("빈도", format="%d"),
                    "차이": st.column_config.NumberColumn("차이", format="%d"),
                    "패턴유형": st.column_config.TextColumn("패턴유형")
                }
            )

    # 연속 패턴 분석 추가
    analyze_consecutive_patterns()

    return validation_id

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "analyze_sequences":
        analyzer = PatternAnalyzer('pattern_analysis_v2.db', 'pattern_analysis_results.db')
        analyzer.analyze_sequences()
        print("analyze_sequences finished")
    else:
        main() 