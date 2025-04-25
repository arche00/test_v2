import sqlite3
import pandas as pd
from datetime import datetime
import streamlit as st
import psycopg2
from database import get_db_connection

def convert_timestamp(timestamp_str):
    """SQLite 타임스탬프를 PostgreSQL 형식으로 변환합니다."""
    try:
        # DDMMYYHHMM 형식을 datetime 객체로 변환
        dt = datetime.strptime(str(timestamp_str), '%d%m%y%H%M')
        # PostgreSQL 형식으로 변환 (YYYY-MM-DD HH:MM:SS)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        print(f"타임스탬프 변환 중 오류 발생: {timestamp_str}, {str(e)}")
        return None

def migrate_data():
    """SQLite에서 PostgreSQL로 데이터를 마이그레이션합니다."""
    try:
        # SQLite에서 데이터 읽기
        sqlite_conn = sqlite3.connect('pattern_analysis_v2.db')
        sqlite_cursor = sqlite_conn.cursor()
        
        # PostgreSQL 연결
        pg_conn = get_db_connection()
        pg_cursor = pg_conn.cursor()
        
        # SQLite에서 모든 레코드 가져오기
        sqlite_cursor.execute("SELECT pattern1, result1, timestamp FROM pattern_records ORDER BY timestamp")
        records = sqlite_cursor.fetchall()
        
        print(f"총 {len(records)}개의 레코드를 마이그레이션합니다...")
        
        # PostgreSQL에 데이터 삽입
        success_count = 0
        for record in records:
            pattern, next_pattern, timestamp = record
            converted_timestamp = convert_timestamp(timestamp)
            if converted_timestamp:
                try:
                    pg_cursor.execute(
                        "INSERT INTO pattern_records (pattern, next_pattern, timestamp) VALUES (%s, %s, %s)",
                        (pattern, next_pattern, converted_timestamp)
                    )
                    success_count += 1
                except Exception as e:
                    print(f"레코드 삽입 중 오류 발생: {record}, {str(e)}")
        
        pg_conn.commit()
        print(f"마이그레이션이 완료되었습니다! {success_count}/{len(records)} 레코드가 성공적으로 이전되었습니다.")
        
    except Exception as e:
        print(f"마이그레이션 중 오류 발생: {str(e)}")
        if 'pg_conn' in locals():
            pg_conn.rollback()
    
    finally:
        if 'sqlite_cursor' in locals():
            sqlite_cursor.close()
        if 'sqlite_conn' in locals():
            sqlite_conn.close()
        if 'pg_cursor' in locals():
            pg_cursor.close()
        if 'pg_conn' in locals():
            pg_conn.close()

if __name__ == "__main__":
    print("데이터 마이그레이션을 시작합니다...")
    migrate_data() 