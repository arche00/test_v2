import os
import psycopg2
from psycopg2.extras import DictCursor
import streamlit as st

def get_db_connection():
    """
    Streamlit Cloud 또는 로컬 환경에서 데이터베이스 연결을 생성합니다.
    """
    try:
        if 'DATABASE_URL' in st.secrets:
            # Streamlit Cloud 환경
            conn = psycopg2.connect(st.secrets['DATABASE_URL'])
        else:
            # 로컬 개발 환경
            conn = psycopg2.connect(
                dbname=os.getenv('DB_NAME', 'pattern_analysis'),
                user=os.getenv('DB_USER', 'postgres'),
                password=os.getenv('DB_PASSWORD', 'postgres'),
                host=os.getenv('DB_HOST', 'localhost'),
                port=os.getenv('DB_PORT', '5432')
            )
        return conn
    except Exception as e:
        st.error(f"데이터베이스 연결 오류: {str(e)}")
        return None

def init_database():
    """
    이 함수는 비활성화되었습니다.
    기존 데이터베이스를 사용하기 위해 테이블 초기화를 건너뜁니다.
    """
    return True  # 항상 성공을 반환하여 테이블 초기화를 건너뜁니다

def insert_pattern_record(pattern, next_pattern):
    """
    패턴 기록을 데이터베이스에 저장합니다.
    """
    conn = get_db_connection()
    if conn is None:
        return False
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO pattern_records (pattern, next_pattern)
                VALUES (%s, %s)
            """, (pattern, next_pattern))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"패턴 기록 저장 오류: {str(e)}")
        return False
    finally:
        conn.close()

def get_pattern_transitions(limit=150):
    """
    최근 패턴 전이 데이터를 지정된 개수만큼 조회합니다.
    """
    conn = get_db_connection()
    if conn is None:
        return []
    
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute("""
                SELECT pattern, next_pattern, timestamp
                FROM pattern_records
                ORDER BY timestamp DESC
                LIMIT %s
            """, (limit,))
            return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        st.error(f"패턴 전이 데이터 조회 오류: {str(e)}")
        return []
    finally:
        conn.close()

def cleanup_old_records(days=90):
    """
    이 함수는 더 이상 레코드를 삭제하지 않습니다.
    모든 레코드를 보존하기 위해 비활성화되었습니다.
    """
    return True  # 레코드 삭제를 비활성화하고 성공을 반환

def insert_group_sequence(sequence):
    """
    그룹 시퀀스를 데이터베이스에 저장합니다.
    """
    conn = get_db_connection()
    if conn is None:
        return False
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO group_sequences (sequence)
                VALUES (%s)
            """, (sequence,))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"그룹 시퀀스 저장 오류: {str(e)}")
        return False
    finally:
        conn.close()

def get_recent_sequences(limit=100):
    """
    최근 그룹 시퀀스를 조회합니다.
    """
    conn = get_db_connection()
    if conn is None:
        return []
    
    try:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute("""
                SELECT sequence, timestamp
                FROM group_sequences
                ORDER BY timestamp DESC
                LIMIT %s
            """, (limit,))
            return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        st.error(f"최근 시퀀스 조회 오류: {str(e)}")
        return []
    finally:
        conn.close() 