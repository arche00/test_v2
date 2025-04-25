import os
import psycopg2
from psycopg2.extras import DictCursor
from datetime import datetime, timedelta
import streamlit as st

class DatabaseManager:
    def __init__(self):
        self.conn = None
        self.connect()

    def connect(self):
        """데이터베이스 연결을 설정합니다."""
        try:
            # Streamlit Cloud에서는 st.secrets에서, 로컬에서는 환경 변수에서 설정을 가져옵니다
            if 'postgres' in st.secrets:
                db_config = st.secrets.postgres
            else:
                db_config = {
                    'host': os.getenv('POSTGRES_HOST', 'localhost'),
                    'database': os.getenv('POSTGRES_DB', 'pattern_analysis'),
                    'user': os.getenv('POSTGRES_USER', 'postgres'),
                    'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
                    'port': os.getenv('POSTGRES_PORT', '5432')
                }

            self.conn = psycopg2.connect(**db_config)
            self.create_tables()
        except Exception as e:
            st.error(f"데이터베이스 연결 중 오류 발생: {str(e)}")

    def create_tables(self):
        """필요한 테이블들을 생성합니다."""
        try:
            with self.conn.cursor() as cur:
                # pattern_records 테이블 생성
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS pattern_records (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        group_range VARCHAR(50),
                        pattern1 VARCHAR(10),
                        result1 CHAR(1),
                        pattern2 VARCHAR(10),
                        result2 CHAR(1),
                        prev_pattern1 VARCHAR(10),
                        prev_pattern2 VARCHAR(10),
                        transition_type VARCHAR(20),
                        transition_count INTEGER DEFAULT 1,
                        pattern1_banker_count INTEGER,
                        pattern1_player_count INTEGER,
                        pattern2_banker_count INTEGER,
                        pattern2_player_count INTEGER,
                        pattern1_transitions INTEGER,
                        pattern2_transitions INTEGER
                    )
                ''')

                # group_sequences 테이블 생성
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS group_sequences (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        tot VARCHAR(100)
                    )
                ''')

                self.conn.commit()
        except Exception as e:
            st.error(f"테이블 생성 중 오류 발생: {str(e)}")

    def save_pattern_record(self, group_range, pattern_123, pattern_1234):
        """패턴 분석 결과를 저장합니다."""
        try:
            with self.conn.cursor() as cur:
                # 이전 패턴 조회
                cur.execute('''
                    SELECT pattern1, pattern2 
                    FROM pattern_records 
                    ORDER BY id DESC LIMIT 1
                ''')
                prev_record = cur.fetchone()
                prev_pattern1 = prev_record[0] if prev_record else None
                prev_pattern2 = prev_record[1] if prev_record else None

                # 전이 타입 계산
                transition_type = None
                transition_count = 1
                if prev_pattern1 and pattern_123:
                    transition_type = f"{prev_pattern1}->{pattern_123[:2]}"
                    cur.execute('''
                        SELECT transition_count 
                        FROM pattern_records 
                        WHERE transition_type = %s 
                        ORDER BY id DESC LIMIT 1
                    ''', (transition_type,))
                    prev_transition = cur.fetchone()
                    if prev_transition:
                        transition_count = prev_transition[0] + 1

                # 패턴 특성 계산
                def calculate_pattern_stats(pattern):
                    if not pattern:
                        return 0, 0, 0
                    banker_count = pattern.count('a')
                    player_count = pattern.count('b')
                    transitions = sum(1 for i in range(len(pattern)-1) if pattern[i] != pattern[i+1])
                    return banker_count, player_count, transitions

                pattern1 = pattern_123[:2] if pattern_123 else ''
                pattern1_stats = calculate_pattern_stats(pattern1)
                pattern2 = pattern_1234[:3] if pattern_1234 else ''
                pattern2_stats = calculate_pattern_stats(pattern2)

                # 데이터 삽입
                cur.execute('''
                    INSERT INTO pattern_records 
                    (group_range, pattern1, result1, pattern2, result2,
                     prev_pattern1, prev_pattern2, transition_type, transition_count,
                     pattern1_banker_count, pattern1_player_count,
                     pattern2_banker_count, pattern2_player_count,
                     pattern1_transitions, pattern2_transitions)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ''', (
                    group_range,
                    pattern1, pattern_123[2] if len(pattern_123) >= 3 else '',
                    pattern2, pattern_1234[3] if len(pattern_1234) >= 4 else '',
                    prev_pattern1, prev_pattern2, transition_type, transition_count,
                    pattern1_stats[0], pattern1_stats[1],
                    pattern2_stats[0], pattern2_stats[1],
                    pattern1_stats[2], pattern2_stats[2]
                ))

                self.conn.commit()
        except Exception as e:
            st.error(f"패턴 기록 저장 중 오류 발생: {str(e)}")

    def save_group_sequence(self, tot_value):
        """그룹 시퀀스를 저장합니다."""
        try:
            with self.conn.cursor() as cur:
                cur.execute('''
                    INSERT INTO group_sequences (tot)
                    VALUES (%s)
                ''', (tot_value,))
                self.conn.commit()
        except Exception as e:
            st.error(f"그룹 시퀀스 저장 중 오류 발생: {str(e)}")

    def get_pattern_transitions(self, limit=150):
        """최근 패턴 전이 데이터를 조회합니다."""
        try:
            with self.conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute('''
                    SELECT *
                    FROM pattern_records
                    ORDER BY timestamp DESC
                    LIMIT %s
                ''', (limit,))
                records = cur.fetchall()
                return [dict(record) for record in records]
        except Exception as e:
            st.error(f"패턴 전이 데이터 조회 중 오류 발생: {str(e)}")
            return None

    def update_database(self):
        """오래된 데이터를 삭제하고 통계를 업데이트합니다."""
        try:
            with self.conn.cursor() as cur:
                # 30일 이상 된 데이터 삭제
                cur.execute('''
                    DELETE FROM pattern_records 
                    WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL '30 days'
                ''')

                # 전이 횟수 업데이트
                cur.execute('''
                    UPDATE pattern_records pr1
                    SET transition_count = (
                        SELECT COUNT(*)
                        FROM pattern_records pr2
                        WHERE pr2.pattern1 = pr1.pattern1
                    )
                ''')

                self.conn.commit()
                return True
        except Exception as e:
            st.error(f"데이터베이스 업데이트 중 오류 발생: {str(e)}")
            return False

    def clear_database(self):
        """데이터베이스의 모든 데이터를 삭제합니다."""
        try:
            with self.conn.cursor() as cur:
                cur.execute('TRUNCATE TABLE pattern_records, group_sequences')
                self.conn.commit()
                return True
        except Exception as e:
            st.error(f"데이터베이스 초기화 중 오류 발생: {str(e)}")
            return False

    def close(self):
        """데이터베이스 연결을 종료합니다."""
        if self.conn:
            self.conn.close()

# 전역 데이터베이스 매니저 인스턴스
db = DatabaseManager() 