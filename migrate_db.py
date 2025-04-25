import sqlite3
import psycopg2
from database import get_db_connection
from datetime import datetime
import os

def convert_timestamp(timestamp_str):
    """SQLite 타임스탬프를 PostgreSQL 형식으로 변환"""
    try:
        # 입력된 타임스탬프가 'YYMMDDHHMMSS' 형식인 경우
        if len(timestamp_str) == 8:  # YYMMDDHH 형식
            year = int('20' + timestamp_str[:2])  # 20YY 형식으로 변환
            month = int(timestamp_str[2:4])
            day = int(timestamp_str[4:6])
            hour = int(timestamp_str[6:8])
            return datetime(year, month, day, hour).isoformat()
        elif len(timestamp_str) == 12:  # YYMMDDHHMMSS 형식
            year = int('20' + timestamp_str[:2])
            month = int(timestamp_str[2:4])
            day = int(timestamp_str[4:6])
            hour = int(timestamp_str[6:8])
            minute = int(timestamp_str[8:10])
            second = int(timestamp_str[10:12])
            return datetime(year, month, day, hour, minute, second).isoformat()
    except (ValueError, TypeError):
        pass
    
    # 기본값으로 현재 시간 반환
    return datetime.now().isoformat()

def migrate_data():
    # SQLite DB 연결
    sqlite_conn = sqlite3.connect('pattern_analysis.db')
    sqlite_cur = sqlite_conn.cursor()
    
    # PostgreSQL DB 연결
    pg_conn = get_db_connection()
    if pg_conn is None:
        print("PostgreSQL 연결 실패")
        return False
    
    pg_cur = pg_conn.cursor()
    
    try:
        # pattern_records 테이블 마이그레이션
        print("패턴 기록 마이그레이션 시작...")
        sqlite_cur.execute("""
            SELECT pattern1, result1, pattern2, result2, timestamp, group_range 
            FROM pattern_records
        """)
        records = sqlite_cur.fetchall()
        
        processed_count = 0
        # pattern1과 result1을 처리
        for record in records:
            pattern1, result1, pattern2, result2, timestamp, group_range = record
            converted_timestamp = convert_timestamp(timestamp)
            
            if pattern1 and result1:
                pg_cur.execute("""
                    INSERT INTO pattern_records (pattern, next_pattern, timestamp)
                    VALUES (%s, %s, %s)
                """, (pattern1, result1, converted_timestamp))
                processed_count += 1
            
            # pattern2와 result2가 있는 경우 추가 처리
            if pattern2 and result2:
                pg_cur.execute("""
                    INSERT INTO pattern_records (pattern, next_pattern, timestamp)
                    VALUES (%s, %s, %s)
                """, (pattern2, result2, converted_timestamp))
                processed_count += 1
        
        print(f"총 {processed_count}개의 패턴 기록이 마이그레이션되었습니다.")
        
        # 변경사항 커밋
        pg_conn.commit()
        print("\n마이그레이션이 성공적으로 완료되었습니다.")
        return True
        
    except Exception as e:
        print(f"마이그레이션 중 오류 발생: {str(e)}")
        pg_conn.rollback()
        return False
        
    finally:
        sqlite_cur.close()
        sqlite_conn.close()
        pg_cur.close()
        pg_conn.close()

if __name__ == "__main__":
    print("데이터베이스 마이그레이션을 시작합니다...")
    migrate_data() 