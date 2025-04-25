from database import init_database
import streamlit as st

if __name__ == "__main__":
    print("데이터베이스 테이블 초기화를 시작합니다...")
    if init_database():
        print("데이터베이스 테이블이 성공적으로 초기화되었습니다.")
    else:
        print("데이터베이스 테이블 초기화 중 오류가 발생했습니다.") 