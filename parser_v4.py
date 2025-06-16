import streamlit as st
import copy

# 페이지 설정을 가장 먼저 실행
st.set_page_config(
    page_title="Pattern Analysis System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import os
import json
import time
import logging
from typing import Optional, Dict, Any
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from collections import Counter
import gc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Windows 환경 호환: 파일 경로를 항상 현재 작업 디렉토리 기준으로 절대경로로 변환
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'pattern_analysis_v2.db')
MODEL_PATH = os.path.join(BASE_DIR, 'parser_v3_model.joblib')
PATTERN_JSON_PATH = os.path.join(BASE_DIR, 'pattern.json')

# CSS 스타일 설정 - 간격 최소화
st.markdown("""
    <style>
    /* 상단 여백 최소화 */
    .stApp {
        margin-top: -2rem;
    }
    /* 제목 간격 최소화 */
    h1, h2, h3, h4, h5, h6 {
        margin: 0 !important;
        padding: 0.1rem 0 !important;
        line-height: 1 !important;
    }
    /* 섹션 간격 최소화 */
    .stMarkdown {
        margin: 0 !important;
        padding: 0.1rem 0 !important;
        line-height: 1 !important;
    }
    /* 그리드 컨테이너 간격 최소화 */
    .grid-container {
        margin: 0 !important;
        padding: 0.1rem 0 !important;
    }
    /* 테이블 간격 최소화 */
    .dataframe {
        margin: 0 !important;
        padding: 0.1rem 0 !important;
    }
    /* 텍스트 간격 최소화 */
    p, div {
        margin: 0 !important;
        padding: 0.1rem 0 !important;
        line-height: 1 !important;
    }
    /* 버튼 간격 최소화 */
    .stButton {
        margin: 0 !important;
        padding: 0.1rem 0 !important;
    }
    /* 입력 필드 간격 최소화 */
    .stTextInput, .stTextArea {
        margin: 0 !important;
        padding: 0.1rem 0 !important;
    }
    /* 구분선 간격 최소화 */
    hr {
        margin: 0.1rem 0 !important;
        padding: 0 !important;
    }
    /* 셀 간격 최소화 */
    .bead-road-cell {
        padding: 0.1rem !important;
        line-height: 1 !important;
    }
    /* 테이블 셀 간격 최소화 */
    .dataframe td, .dataframe th {
        padding: 0.1rem !important;
        line-height: 1 !important;
    }
    .stText {
        writing-mode: horizontal-tb;
        font-size: 24px;
    }
    .prediction-text {
        font-size: 28px;
        font-weight: bold;
        color: #FF4B4B;
    }
    /* 상단 여백 최소화 및 라디오 버튼 영역 개선 */
    div[data-baseweb="radio"] label {margin-right: 2.5rem !important; padding: 0.3rem 1.2rem 0.3rem 0.7rem !important; border-radius: 0.5rem; transition: background 0.2s;}
    div[data-baseweb="radio"] label:hover {background: #f0f2f6;}
    div[data-baseweb="radio"] {margin-bottom: 0.2rem !important;}
    </style>
""", unsafe_allow_html=True)

# Database setup
def init_db():
    try:
        # Get absolute path to database file
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pattern_analysis_v2.db')
        # st.info(f"데이터베이스 경로: {db_path}")  # 제거
        
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # Check if tables exist
        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = c.fetchall()
        # st.info(f"현재 테이블 목록: {[table[0] for table in tables]}")  # 제거
        
        # Check record count
        c.execute("SELECT COUNT(*) FROM pattern_records")
        record_count = c.fetchone()[0]
        # st.info(f"현재 레코드 수: {record_count}")  # 제거
        
        conn.close()
    except Exception as e:
        pass  # st.error(f"데이터베이스 초기화 중 오류 발생: {str(e)}")

# Initialize database when the app starts
init_db()

# Function to save pattern analysis results
def save_pattern_record(group_range, pattern_123, pattern_1234, pattern1_number, result1_number, pattern2_number, result2_number):
    try:
        # Get absolute path to database file
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pattern_analysis_v2.db')
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # Format timestamp as YYMMDDHHMM
        current_time = datetime.now()
        timestamp = current_time.strftime("%y%m%d%H%M")
        
        prev_record = c.execute('''
            SELECT pattern1, pattern2 
            FROM pattern_records 
            ORDER BY round DESC LIMIT 1
        ''').fetchone()
        
        prev_pattern1 = prev_record[0] if prev_record else None
        prev_pattern2 = prev_record[1] if prev_record else None
        transition_type = None
        transition_count = 1
        
        if prev_pattern1 and pattern_123:
            transition_type = f"{prev_pattern1}->{pattern_123[:2]}"
            prev_transition = c.execute('''
                SELECT transition_count 
                FROM pattern_records 
                WHERE transition_type = ? 
                ORDER BY round DESC LIMIT 1
            ''', (transition_type,)).fetchone()
            if prev_transition:
                transition_count = prev_transition[0] + 1
        
        def calculate_pattern_stats(pattern):
            if not pattern:
                return 0, 0, 0
            banker_count = pattern.count('a')
            player_count = pattern.count('b')
            transitions = sum(1 for i in range(len(pattern)-1) if pattern[i] != pattern[i+1])
            return banker_count, player_count, transitions
        
        pattern1 = pattern_123[:2] if pattern_123 else ''
        pattern1_banker_count, pattern1_player_count, pattern1_transitions = calculate_pattern_stats(pattern1)
        pattern2 = pattern_1234[:3] if pattern_1234 else ''
        pattern2_banker_count, pattern2_player_count, pattern2_transitions = calculate_pattern_stats(pattern2)
        
        c.execute('''
            INSERT INTO pattern_records 
            (timestamp, group_range, pattern1, result1, pattern2, result2,
             prev_pattern1, prev_pattern2, transition_type, transition_count,
             pattern1_banker_count, pattern1_player_count, pattern2_banker_count, pattern2_player_count,
             pattern1_transitions, pattern2_transitions,
             pattern1_number, result1_number, pattern2_number, result2_number)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, group_range, 
              pattern1, pattern_123[2] if len(pattern_123) >= 3 else '',
              pattern2, pattern_1234[3] if len(pattern_1234) >= 4 else '',
              prev_pattern1, prev_pattern2, transition_type, transition_count,
              pattern1_banker_count, pattern1_player_count, pattern2_banker_count, pattern2_player_count,
              pattern1_transitions, pattern2_transitions,
              pattern1_number, result1_number, pattern2_number, result2_number))
        
        conn.commit()
        st.success("패턴 데이터가 저장되었습니다.")
        
        # Show current record count
        c.execute("SELECT COUNT(*) FROM pattern_records")
        record_count = c.fetchone()[0]
        st.info(f"현재 총 레코드 수: {record_count}")
        
    except Exception as e:
        st.error(f"데이터 저장 중 오류 발생: {str(e)}")
    finally:
        conn.close()

# Function to display pattern records
def display_pattern_records():
    try:
        conn = sqlite3.connect('pattern_analysis_v2.db')
        c = conn.cursor()
        # 라운드 내림차순으로 정렬 (최신순)
        records = c.execute('SELECT * FROM pattern_records ORDER BY round DESC LIMIT 10').fetchall()
        conn.close()
        
        st.markdown("### Pattern Analysis Records")
        if records:
            # Create a table header
            st.markdown("""
            | Round | Time | Group Range | Pattern1 | Result1 | Pattern2 | Result2 |
            |--------|------|-----------|--------|--------|--------|--------|""")
            
            # Add each record to the table
            for record in records:
                st.markdown(f"| {record[0]} | {record[1]} | {record[2]} | {record[3]} | {record[4]} | {record[5]} | {record[6]} |")
        else:
            st.info("저장된 기록이 없습니다.")
    except Exception as e:
        st.error(f"데이터 조회 중 오류 발생: {str(e)}")

# Table dimensions
TABLE_WIDTH = 15
TABLE_HEIGHT = 6

# Cell types
CELL_BANKER = 'b'
CELL_PLAYER = 'p'
CELL_TIE = 't'
CELL_EMPTY = ''

# Pattern definitions
PATTERN_WIDTH = 2
PATTERN_TOP_ROWS = [0,1,2]
PATTERN_BOTTOM_ROWS = [3,4,5]

def parse_bead_road_svg(svg_code):
    soup = BeautifulSoup(svg_code, 'html.parser')
    grid = [['' for _ in range(TABLE_HEIGHT)] for _ in range(TABLE_WIDTH)]
    
    coordinates = soup.find_all('svg', attrs={'data-type': 'coordinates'})
    for coord in coordinates:
        x = int(float(coord.get('data-x', 0)))
        y = int(float(coord.get('data-y', 0)))
        text_elem = coord.find('text')
        if text_elem and text_elem.string:
            result = text_elem.string.strip()
            if 0 <= x < TABLE_WIDTH and 0 <= y < TABLE_HEIGHT:
                grid[x][y] = result.lower()
    
    return grid

def display_grid_with_title(grid, title):
    html = '''
    <style>
    .grid-container { display: table; border-collapse: collapse; margin: 0 auto 20px auto; width: 80%; margin-top: 0 !important; }
    .grid-row { display: table-row; }
    .bead-road-cell { width: 22px; height: 22px; border: 1px solid black; display: table-cell; 
                     text-align: center; vertical-align: middle; font-family: monospace; font-size: 0.95rem; padding: 0; }
    .banker { color: red; font-weight: bold; }
    .player { color: blue; font-weight: bold; }
    .tie { color: green; font-weight: bold; }
    .grid-title { font-size:1.05rem; font-weight:600; margin-bottom:0 !important; padding-bottom:0 !important; display:block; }
    </style>
    '''
    html += f'<span class="grid-title">{title}</span>'
    html += '<div class="grid-container">'
    for y in range(TABLE_HEIGHT):
        html += '<div class="grid-row">'
        for x in range(TABLE_WIDTH):
            cell = grid[x][y]
            css_class = 'banker' if cell == 'b' else 'player' if cell == 'p' else 'tie' if cell == 't' else ''
            html += f'<div class="bead-road-cell {css_class}">{cell.upper() if cell else "&nbsp;"}</div>'
        html += '</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def get_pattern_positions():
    patterns = []
    pattern_number = 1
    
    for start_col in range(TABLE_WIDTH - PATTERN_WIDTH + 1):
        cols = (start_col, start_col + 1)
        
        top_pattern = {
            'pattern_number': pattern_number,
            'columns': cols,
            'rows': PATTERN_TOP_ROWS,
            'coordinates': [(cols[0], y) for y in PATTERN_TOP_ROWS] + [(cols[1], y) for y in PATTERN_TOP_ROWS]
        }
        patterns.append(top_pattern)
        pattern_number += 1
        
        bottom_pattern = {
            'pattern_number': pattern_number,
            'columns': cols,
            'rows': PATTERN_BOTTOM_ROWS,
            'coordinates': [(cols[0], y) for y in PATTERN_BOTTOM_ROWS] + [(cols[1], y) for y in PATTERN_BOTTOM_ROWS]
        }
        patterns.append(bottom_pattern)
        pattern_number += 1
    
    return patterns

def display_zones(zones):
    if not zones:
        st.info("표시할 그룹이 없습니다.")
        return
    
    first_zone = zones[0]
    st.markdown(f"### First Group (Column {first_zone['start_x'] + 1} ~ {first_zone['end_x'] + 1})")
    
    st.markdown("""
        <style>
        .zone-container { display: table; border-collapse: collapse; margin: 10px 0; }
        .zone-row { display: table-row; }
        .zone-cell { width: 40px; height: 40px; border: 1px solid black; display: table-cell; 
                    text-align: center; vertical-align: middle; font-family: monospace; }
        .banker { color: red; font-weight: bold; }
        .player { color: blue; font-weight: bold; }
        .tie { color: green; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)
    
    html_zone = ['<div class="zone-container">']
    for y in range(6):
        html_zone.append('<div class="zone-row">')
        for x in range(len(first_zone['zone_data'])):
            cell = first_zone['zone_data'][x][y]
            css_class = 'banker' if cell == 'b' else 'player' if cell == 'p' else 'tie' if cell == 't' else ''
            html_zone.append(f'<div class="zone-cell {css_class}">{cell.upper() if cell else "&nbsp;"}</div>')
        html_zone.append('</div>')
    html_zone.append('</div>')
    
    st.markdown(''.join(html_zone), unsafe_allow_html=True)
    
    st.markdown("### First Group's Pattern")
    patterns = get_pattern_positions()
    first_group_patterns = [p for p in patterns if p['columns'][0] >= first_zone['start_x'] and p['columns'][1] <= first_zone['end_x']]
    
    for pattern in first_group_patterns:
        st.markdown(f"#### Pattern {pattern['pattern_number']}")
        pattern_html = ['<div class="zone-container">']
        for y in pattern['rows']:
            pattern_html.append('<div class="zone-row">')
            for x in pattern['columns']:
                cell = first_zone['zone_data'][x - first_zone['start_x']][y]
                css_class = 'banker' if cell == 'b' else 'player' if cell == 'p' else 'tie' if cell == 't' else ''
                pattern_html.append(f'<div class="zone-cell {css_class}">{cell.upper() if cell else "&nbsp;"}</div>')
            pattern_html.append('</div>')
        pattern_html.append('</div>')
        st.markdown(''.join(pattern_html), unsafe_allow_html=True)

def divide_grid_into_overlapping_zones(grid, zone_width=3):
    zones = []
    for start_x in range(15 - zone_width + 1):
        end_x = start_x + zone_width
        zone_data = [[grid[x][y] for y in range(6)] for x in range(start_x, end_x)]
        if any(cell in {'b', 't', 'p'} for column in zone_data for cell in column):
            zones.append({
                'zone_data': zone_data,
                'start_x': start_x,
                'end_x': end_x - 1
            })
    return zones

def find_pattern_group(pattern_values):
    """
    Find group value of a pattern from pattern.json.
    
    Args:
        pattern_values (list): List of pattern characters (e.g., ['B', 'P', 'B', 'B', 'P', 'B'])
    
    Returns:
        str or None: Group value ('a' or 'b'), None if not found
    """
    try:
        with open('pattern.json', 'r') as f:
            pattern_data = json.load(f)
        
        # Convert input pattern to lowercase
        pattern_values = [v.lower() for v in pattern_values if v]  # Exclude empty strings
        
        # Check all patterns in groupA and groupB
        for group_name in ['groupA', 'groupB']:
            patterns = pattern_data['patterns'][group_name]
            for pattern in patterns:
                if pattern.get('sequence') == pattern_values:
                    return pattern.get('group', group_name[5].lower())  # Return group value, default to 'a' or 'b' if not found
        
        return None
    except Exception as e:
        st.error(f"패턴 그룹 검색 중 오류 발생: {str(e)}")
        return None

def get_pattern_values(grid, pattern_positions):
    """
    Extract values from grid using pattern coordinates.
    
    Args:
        grid (list): Full grid data
        pattern_positions (list): List of pattern coordinates [(x1,y1), (x2,y2), ...]
    
    Returns:
        list: List of pattern values (e.g., ['B', 'P', 'B', 'B', 'P', 'B'])
    """
    values = []
    for x, y in pattern_positions:
        value = grid[x][y]
        if value:
            values.append(value.upper())
        else:
            values.append('')
    return values

def get_all_groups_first_two_values(zones):
    """
    Extract first 2 values from all groups and concatenate them into a single string.
    
    Args:
        zones (list): List of zone data to analyze
        
    Returns:
        str: Concatenated string of first 2 values from all groups (e.g., 'bababp')
    """
    result = ''
    for zone in zones:
        first_two = get_first_two_group_values(zone)
        if first_two:  # Only add if not empty string
            result += first_two
    return result

def save_group_sequence(tot_value):
    """
    Save group sequence.
    
    Args:
        tot_value (str): Concatenated string of first 2 values from all groups
    """
    try:
        # Format timestamp as YYMMDDHHMM
        current_time = datetime.now()
        timestamp = current_time.strftime("%y%m%d%H%M")
        
        conn = sqlite3.connect('pattern_analysis_v2.db')
        c = conn.cursor()
        
        c.execute('''
            INSERT INTO group_sequences 
            (timestamp, tot)
            VALUES (?, ?)
        ''', (timestamp, tot_value))
        
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"그룹 시퀀스 저장 중 오류 발생: {str(e)}")

def find_pattern_number_only(pattern_values):
    """
    pattern.json에서 입력된 시퀀스와 완전히 일치하는 패턴의 넘버만 반환합니다.
    Args:
        pattern_values (list): 예시 ['b', 'b', 'b']
    Returns:
        str or None: 패턴 넘버(예: '144047'), 없으면 None
    """
    try:
        with open('pattern.json', 'r') as f:
            pattern_data = json.load(f)
        pattern_values = [v.lower() for v in pattern_values if v]
        for group_name in ['groupA', 'groupB']:
            patterns = pattern_data['patterns'][group_name]
            for pattern in patterns:
                if pattern.get('sequence') == pattern_values:
                    return pattern.get('pattern_number')
        return None
    except Exception as e:
        st.error(f"패턴 넘버 검색 중 오류 발생: {str(e)}")
        return None

def display_pattern_groups(zones):
    """
    Display pattern group analysis results in a separate section.
    """
    if not zones:
        return
    
    # Place title and save button side by side
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown("### Pattern Group Analysis")
    
    # Display all groups' first 2 values concatenated
    all_first_two = get_all_groups_first_two_values(zones)
    if all_first_two:
        st.text(f"All groups' first 2 values: {all_first_two}")
        st.markdown("---")
    
    # List to store analysis results
    analysis_results = []
    pattern_numbers_list = []
    
    # Reverse the zones list to display in reverse order
    for zone in reversed(zones):
        patterns = get_pattern_positions()
        group_patterns = [p for p in patterns if p['columns'][0] >= zone['start_x'] and p['columns'][1] <= zone['end_x']]
        
        if len(group_patterns) < 4:  # Minimum 4 patterns are needed
            continue
            
        # Extract pattern values
        pattern_values = []
        for pattern in group_patterns[:4]:  # Use only first 4 patterns
            values = []
            for x, y in pattern['coordinates']:
                relative_x = x - zone['start_x']
                value = zone['zone_data'][relative_x][y]
                if value:
                    values.append(value.upper())
            pattern_values.append(values)
            
        # 각 패턴별 넘버 리스트
        pattern_numbers = []
        for v in pattern_values[:4]:
            pattern_number = find_pattern_number_only([x.lower() for x in v]) if v else None
            pattern_numbers.append(pattern_number if pattern_number is not None else '-')
        # 넘버 가공
        numbers_dict = process_pattern_numbers(pattern_numbers)
        pattern_numbers_list.append(numbers_dict)
        
        # Search for pattern 123 and 1234 groups
        groups_123 = []
        groups_1234 = []
        pattern_123_valid = True
        if len(pattern_values) >= 3:
            for i in range(3):
                if not pattern_values[i]:
                    pattern_123_valid = False
                    break
                group = find_pattern_group(pattern_values[i])
                if group is None:
                    pattern_123_valid = False
                    break
                groups_123.append(group)
        
        # Search for pattern 1234
        pattern_1234_valid = True
        if len(pattern_values) >= 4:
            for i in range(4):
                if not pattern_values[i]:
                    pattern_1234_valid = False
                    break
                group = find_pattern_group(pattern_values[i])
                if group is None:
                    pattern_1234_valid = False
                    break
                groups_1234.append(group)
        
        # Generate result string
        pattern_123_text = ''.join(groups_123) if pattern_123_valid and len(groups_123) == 3 else ''
        pattern_1234_text = ''.join(groups_1234) if pattern_1234_valid and len(groups_1234) == 4 else ''
        
        # Extract first 2 values
        first_two = get_first_two_group_values(zone)
        
        # Group range text
        group_range = f"{zone['start_x'] + 1}-{zone['end_x'] + 1}"
        
        # Store analysis result
        if pattern_123_text or pattern_1234_text:
            analysis_results.append({
                'group_range': group_range,
                'pattern_123': pattern_123_text,
                'pattern_1234': pattern_1234_text
            })
        
        # Display result only if there are valid patterns
        if any([pattern_123_text, pattern_1234_text, first_two]):
            st.markdown(f"#### Group {group_range}")
            for idx, v in enumerate(pattern_values[:4]):
                pattern_number = pattern_numbers[idx]
                st.text(f"Pattern {idx+1} Number: {pattern_number if pattern_number is not None else '-'}")
            
            # Add combined pattern numbers display
            if len(pattern_numbers) >= 2:
                pattern1_2 = pattern_numbers[0] + pattern_numbers[1] if pattern_numbers[0] != '-' and pattern_numbers[1] != '-' else '-'
                st.text(f"Pattern 1,2: {pattern1_2}")
            
            if len(pattern_numbers) >= 3:
                pattern1_2_3 = pattern_numbers[0] + pattern_numbers[1] + pattern_numbers[2] if all(p != '-' for p in pattern_numbers[:3]) else '-'
                st.text(f"Pattern 1,2,3: {pattern1_2_3}")
            
            # Add pattern 3,4 combined display
            if len(pattern_numbers) >= 4:
                pattern3_4 = pattern_numbers[2] + pattern_numbers[3] if pattern_numbers[2] != '-' and pattern_numbers[3] != '-' else '-'
                st.text(f"Pattern 3,4: {pattern3_4}")
            
            # Pattern 1,2,3
            st.text(f"Pattern 1,2,3 Group: {pattern_123_text}")
            # Pattern 1,2,3,4
            st.text(f"Pattern 1,2,3,4 Group: {pattern_1234_text}")
            st.text(f"First 2 values: {first_two}")
            st.markdown("---")
    
    # Place save button
    with col2:
        if st.button("Save Pattern"):
            for idx, result in enumerate(analysis_results):
                numbers_dict = pattern_numbers_list[idx] if idx < len(pattern_numbers_list) else {'pattern1_number':'','result1_number':'','pattern2_number':'','result2_number':''}
                save_pattern_record(
                    result['group_range'],
                    result['pattern_123'],
                    result['pattern_1234'],
                    numbers_dict['pattern1_number'],
                    numbers_dict['result1_number'],
                    numbers_dict['pattern2_number'],
                    numbers_dict['result2_number']
                )
            if all_first_two:
                save_group_sequence(all_first_two)
            st.success("Pattern saved successfully!")

def get_pattern_statistics():
    """
    Calculate pattern statistics from DB.
    """
    try:
        conn = sqlite3.connect('pattern_analysis_v2.db')
        c = conn.cursor()
        
        # Check total record count
        total_records = c.execute('SELECT COUNT(*) FROM pattern_records').fetchone()[0]
        
        # Calculate current time's recent 3 hours timestamp (YYMMDDHH format)
        current_time = datetime.now()
        three_hours_ago = current_time - timedelta(hours=3)
        recent_timestamp = three_hours_ago.strftime("%y%m%d%H")
        
        if total_records <= 100:
            # Statistics for all records
            r1_stats = c.execute('''
                SELECT pattern1 || result1 as p1r1, COUNT(*) as count
                FROM pattern_records
                WHERE result1 != ''
                GROUP BY p1r1
                ORDER BY count DESC
            ''').fetchall()
            
            r2_stats = c.execute('''
                SELECT pattern2 || result2 as p2r2, COUNT(*) as count
                FROM pattern_records
                WHERE result2 != ''
                GROUP BY p2r2
                ORDER BY count DESC
            ''').fetchall()
            
            sample_size = total_records
            
        else:
            # Check recent record count
            recent_count = c.execute(
                'SELECT COUNT(*) FROM pattern_records WHERE timestamp >= ?',
                (recent_timestamp,)
            ).fetchone()[0]
            
            # Decide larger number between recent 3 hours and recent 100
            if recent_count > 100:
                # Use recent 3 hours data
                r1_stats = c.execute('''
                    SELECT pattern1 || result1 as p1r1, COUNT(*) as count
                    FROM pattern_records
                    WHERE result1 != '' AND timestamp >= ?
                    GROUP BY p1r1
                    ORDER BY count DESC
                ''', (recent_timestamp,)).fetchall()
                
                r2_stats = c.execute('''
                    SELECT pattern2 || result2 as p2r2, COUNT(*) as count
                    FROM pattern_records
                    WHERE result2 != '' AND timestamp >= ?
                    GROUP BY p2r2
                    ORDER BY count DESC
                ''', (recent_timestamp,)).fetchall()
                
                sample_size = recent_count
                
            else:
                # Use recent 100 data
                r1_stats = c.execute('''
                    SELECT pattern1 || result1 as p1r1, COUNT(*) as count
                    FROM pattern_records
                    WHERE result1 != ''
                    GROUP BY p1r1
                    ORDER BY count DESC
                    LIMIT 100
                ''').fetchall()
                
                r2_stats = c.execute('''
                    SELECT pattern2 || result2 as p2r2, COUNT(*) as count
                    FROM pattern_records
                    WHERE result2 != ''
                    GROUP BY p2r2
                    ORDER BY count DESC
                    LIMIT 100
                ''').fetchall()
                
                sample_size = 100

        # Group P1 patterns and sort
        p1_groups = {
            'aa': [], 'ab': [], 'ba': [], 'bb': []
        }
        
        # Group P2 patterns and sort
        p2_groups = {
            'bba': [], 'baa': [], 'abb': [], 'aab': [],
            'aba': [], 'aaa': [], 'bbb': [], 'bab': []
        }
        
        # Assign each P1 pattern to corresponding group
        for pattern, count in r1_stats:
            if len(pattern) >= 2:
                group_key = pattern[:2].lower()
                if group_key in p1_groups:
                    p1_groups[group_key].append((pattern, count))

        # Assign each P2 pattern to corresponding group
        for pattern, count in r2_stats:
            if len(pattern) >= 3:
                group_key = pattern[:3].lower()
                if group_key in p2_groups:
                    p2_groups[group_key].append((pattern, count))

        # Calculate max count for each P1 group
        p1_group_max_counts = {}
        for group, patterns in p1_groups.items():
            if patterns:
                p1_group_max_counts[group] = max(count for _, count in patterns)
            else:
                p1_group_max_counts[group] = 0

        # Calculate max count for each P2 group
        p2_group_max_counts = {}
        for group, patterns in p2_groups.items():
            if patterns:
                p2_group_max_counts[group] = max(count for _, count in patterns)
            else:
                p2_group_max_counts[group] = 0

        # Sort groups by max count
        sorted_p1_groups = sorted(p1_groups.items(), 
                                key=lambda x: p1_group_max_counts[x[0]], 
                                reverse=True)
        
        sorted_p2_groups = sorted(p2_groups.items(),
                                key=lambda x: p2_group_max_counts[x[0]],
                                reverse=True)

        conn.close()
        return {
            'total_records': total_records,
            'sample_size': sample_size,
            'p1_groups': sorted_p1_groups,
            'p2_groups': sorted_p2_groups
        }
        
    except Exception as e:
        st.error(f"통계 데이터 조회 중 오류 발생: {str(e)}")
        return None

def display_pattern_statistics(stats):
    """
    Display pattern statistics.
    """
    if not stats:
        return

    # Basic style settings
    st.markdown("""
        <style>
        .stMarkdown {
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
        }
        div[data-testid="stHorizontalBlock"] {
            gap: 2em;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header information
    st.write("### Pattern Statistics")
    st.write(f"Total records: {stats['total_records']} | Sample size: {stats['sample_size']}")
    
    # Create columns
    col1, col2 = st.columns(2)
    
    # P1 Statistics
    with col1:
        st.write("#### P1 Pattern Statistics")
        for group_name, patterns in stats['p1_groups']:
            st.write(f"**{group_name.upper()} Group**")
            sorted_patterns = sorted(patterns, key=lambda x: x[1], reverse=True)
            group_total = sum(count for _, count in sorted_patterns)
            
            if group_total == 0:
                st.write("No records (0%)")
            else:
                for pattern, count in sorted_patterns:
                    group_percentage = (count / group_total) * 100
                    st.write(f"{pattern}: {count} times ({group_percentage:.1f}%)")
            st.write("---")
    
    # P2 Statistics
    with col2:
        st.write("#### P2 Pattern Statistics")
        for group_name, patterns in stats['p2_groups']:
            st.write(f"**{group_name.upper()} Group**")
            sorted_patterns = sorted(patterns, key=lambda x: x[1], reverse=True)
            group_total = sum(count for _, count in sorted_patterns)
            
            if group_total == 0:
                st.write("No records (0%)")
            else:
                for pattern, count in sorted_patterns:
                    group_percentage = (count / group_total) * 100
                    st.write(f"{pattern}: {count} times ({group_percentage:.1f}%)")
            st.write("---")

def convert_tie_values(grid):
    """
    Convert T values according to rules.
    """
    converted_grid = [row[:] for row in grid]  # Copy grid
    
    # Apply 1st column rule
    for y in range(6):
        if converted_grid[0][y] == 't':
            if y == 0:  # 1st row 1st column
                converted_grid[0][y] = converted_grid[0][1]  # Convert to 2nd row 1st column value
            else:  # 1st column other rows
                converted_grid[0][y] = converted_grid[0][y-1]  # Convert to previous row value
    
    # Apply other columns rule
    for x in range(1, 15):
        for y in range(6):
            if converted_grid[x][y] == 't':
                if y == 0:  # 1st row of each column
                    converted_grid[x][y] = converted_grid[x-1][y]  # Convert to previous column 1st value
                else:  # Other rows
                    # Count left, left up, up values
                    values = {
                        converted_grid[x-1][y]: 1,  # Left
                        converted_grid[x-1][y-1]: 1,  # Left up
                        converted_grid[x][y-1]: 1  # Up
                    }
                    # Convert to most frequent value
                    max_value = max(values.items(), key=lambda x: x[1])[0]
                    converted_grid[x][y] = max_value
    
    return converted_grid

def get_group_ratio_trend():
    """
    Calculate group ratio trend over time.
    """
    try:
        conn = sqlite3.connect('pattern_analysis_v2.db')
        c = conn.cursor()
        
        # Check group count by time
        c.execute('''
            SELECT 
                substr(timestamp, 1, 8) as date_hour,
                pattern1,
                COUNT(*) as count
            FROM pattern_records
            WHERE pattern1 != ''
            GROUP BY date_hour, pattern1
            ORDER BY date_hour
        ''')
        
        # Organize results by time
        time_series = {}
        for date_hour, pattern, count in c.fetchall():
            if date_hour not in time_series:
                time_series[date_hour] = {'aa': 0, 'ab': 0, 'ba': 0, 'bb': 0}
            time_series[date_hour][pattern] = count
        
        # Calculate each time ratio
        ratio_series = []
        for date_hour, counts in time_series.items():
            total = sum(counts.values())
            if total > 0:
                ratios = {
                    'time': f"{date_hour[:2]}-{date_hour[2:4]}-{date_hour[4:6]} {date_hour[6:8]}",
                    'aa': (counts['aa'] / total) * 100,
                    'ab': (counts['ab'] / total) * 100,
                    'ba': (counts['ba'] / total) * 100,
                    'bb': (counts['bb'] / total) * 100
                }
                ratio_series.append(ratios)
        
        conn.close()
        return ratio_series
        
    except Exception as e:
        st.error(f"그룹 비율 추이 계산 중 오류 발생: {str(e)}")
        return None

def display_group_ratio_trend():
    """
    Display group ratio trend.
    """
    ratio_series = get_group_ratio_trend()
    if not ratio_series:
        return
    
    st.write("### Group Ratio Trend")
    
    # Display time series data
    for data in ratio_series:
        st.write(f"**{data['time']}**")
        st.write(f"AA: {data['aa']:.1f}% | AB: {data['ab']:.1f}% | BA: {data['ba']:.1f}% | BB: {data['bb']:.1f}%")
        st.write("---")

def get_first_two_group_values(zone):
    """
    Extract first 2 values from a pattern group.
    
    Args:
        zone (dict): Zone data to analyze
        
    Returns:
        str: First 2 characters of pattern 123 group (e.g., 'ba')
    """
    patterns = get_pattern_positions()
    group_patterns = [p for p in patterns if p['columns'][0] >= zone['start_x'] and p['columns'][1] <= zone['end_x']]
    
    if len(group_patterns) < 4:  # Minimum 4 patterns are needed
        return ''
        
    # Extract pattern values
    pattern_values = []
    for pattern in group_patterns[:4]:  # Use only first 4 patterns
        values = []
        for x, y in pattern['coordinates']:
            relative_x = x - zone['start_x']
            value = zone['zone_data'][relative_x][y]
            if value:
                values.append(value.upper())
        pattern_values.append(values)
        
    # Search for pattern 123 group
    groups_123 = []
    pattern_123_valid = True
    
    if len(pattern_values) >= 3:
        for i in range(3):
            if not pattern_values[i]:
                pattern_123_valid = False
                break
            group = find_pattern_group(pattern_values[i])
            if group is None:
                pattern_123_valid = False
                break
            groups_123.append(group)
    
    # Generate and return result string and first 2 characters
    pattern_123_text = ''.join(groups_123) if pattern_123_valid and len(groups_123) == 3 else ''
    return pattern_123_text[:2] if len(pattern_123_text) >= 2 else ''

def display_recent_records():
    """
    Display recent 3 records in table format.
    """
    try:
        conn = sqlite3.connect('pattern_analysis_v2.db')
        c = conn.cursor()
        
        # pattern_records table's recent 3 records
        pattern_records = c.execute('''
            SELECT round, timestamp, pattern1 || result1 as P1, pattern2 || result2 as P2 
            FROM pattern_records 
            ORDER BY round DESC LIMIT 3
        ''').fetchall()
        
        # group_sequences table's recent 3 records
        group_sequences = c.execute('''
            SELECT round, timestamp, tot 
            FROM group_sequences 
            ORDER BY round DESC LIMIT 3
        ''').fetchall()
        
        conn.close()
        
        st.markdown("### Recent Records")
        
        # Pattern Records display
        st.markdown("#### Pattern Records")
        if pattern_records:
            df_pattern = pd.DataFrame(pattern_records, columns=['Round', 'Time', 'P1', 'P2'])
            st.dataframe(df_pattern.set_index('Round'), use_container_width=True)  # Adjust table width
        else:
            st.info("저장된 패턴 기록이 없습니다.")
        
        # Group Sequences display
        st.markdown("#### Group Sequences")
        if group_sequences:
            df_sequence = pd.DataFrame(group_sequences, columns=['Round', 'Time', 'Total Value'])
            st.dataframe(df_sequence.set_index('Round'), use_container_width=True) # Adjust table width
        else:
            st.info("저장된 그룹 시퀀스가 없습니다.")
            
    except Exception as e:
        st.error(f"기록 조회 중 오류 발생: {str(e)}")

def process_pattern_numbers(pattern_numbers):
    """
    그룹 내 패턴별 넘버 리스트를 받아 아래와 같이 가공하여 반환합니다.
    Args:
        pattern_numbers (list): [패턴1넘버, 패턴2넘버, 패턴3넘버, 패턴4넘버]
    Returns:
        dict: pattern1_number, result1_number, pattern2_number, result2_number
    """
    # None 또는 '-' 처리
    n1 = pattern_numbers[0] if len(pattern_numbers) > 0 and pattern_numbers[0] not in [None, '-'] else ''
    n2 = pattern_numbers[1] if len(pattern_numbers) > 1 and pattern_numbers[1] not in [None, '-'] else ''
    n3 = pattern_numbers[2] if len(pattern_numbers) > 2 and pattern_numbers[2] not in [None, '-'] else ''
    n4 = pattern_numbers[3] if len(pattern_numbers) > 3 and pattern_numbers[3] not in [None, '-'] else ''
    return {
        'pattern1_number': n1 + n2,
        'result1_number': n3,
        'pattern2_number': n1 + n2 + n3,
        'result2_number': n4
    }

def load_number_data():
    """데이터를 청크 단위로 로드하여 메모리 사용량 최적화"""
    conn = sqlite3.connect(DB_PATH)
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
        joblib.dump(model_data, MODEL_PATH)
        
        # 메모리 정리
        del X1, X2, y1, y2, model1, model2
        gc.collect()
        
        st.success("모델 학습이 완료되었습니다.")
        return True

    except Exception as e:
        st.error(f"모델 학습 중 오류 발생: {str(e)}")
        return False

def load_model_data():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
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
        with open(PATTERN_JSON_PATH, 'r') as f:
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

# 중앙 영역 전용 예측 함수
def predict_middle_result1(pattern1_number):
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

def predict_middle_result2(pattern2_number):
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

# 중앙 영역 전용 필터링 함수
def filter_middle_predictions_by_pattern(pattern_number, predictions):
    """Filter predictions based on pattern search results (for pattern1 or pattern2)"""
    if not predictions:
        return predictions

    # Load pattern data
    pattern_data = load_pattern_data()
    if not pattern_data:
        return predictions

    # Get first 2 digits of pattern_number
    search_prefix = str(pattern_number)[:2]
    
    # Search patterns with the prefix
    initial_patterns = search_patterns(pattern_data, search_prefix)
    
    if not initial_patterns:
        return predictions[:3]

    # Get 4th, 5th, 6th items from sequences
    target_sequences = []
    for pattern in initial_patterns:
        sequence = pattern['sequence']
        if len(sequence) >= 6:
            target_seq = ''.join(sequence[3:6]).lower()
            target_sequences.append(target_seq)

    if not target_sequences:
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

    # Get unique pattern numbers and their group values from related patterns
    related_patterns_dict = {}
    for pattern in related_patterns:
        if pattern['pattern_number'] != 'N/A':
            related_patterns_dict[pattern['pattern_number']] = {
                'group': pattern['group'],
                'group_value': pattern['group_value'],
                'sequence': pattern['sequence']
            }

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

    # If no predictions match the filter, return top 3 original predictions
    if not filtered_predictions:
        return predictions[:3]

    # Sort by probability and return top 3
    filtered_predictions.sort(key=lambda x: x['probability'], reverse=True)
    return filtered_predictions[:3]

def filter_middle_predictions_by_pattern2(pattern2_number, predictions):
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
        return predictions[:3]
    search_prefix = pattern2_str[2:4]

    # Search patterns with the prefix
    initial_patterns = search_patterns(pattern_data, search_prefix)
    
    if not initial_patterns:
        return predictions[:3]

    # Get 4th, 5th, 6th items from sequences
    target_sequences = []
    for pattern in initial_patterns:
        sequence = pattern['sequence']
        if len(sequence) >= 6:
            target_seq = ''.join(sequence[3:6]).lower()
            target_sequences.append(target_seq)

    if not target_sequences:
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

    # Get unique pattern numbers and their group values from related patterns
    related_patterns_dict = {}
    for pattern in related_patterns:
        if pattern['pattern_number'] != 'N/A':
            related_patterns_dict[pattern['pattern_number']] = {
                'group': pattern['group'],
                'group_value': pattern['group_value'],
                'sequence': pattern['sequence']
            }

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

    # If no predictions match the filter, return top 3 original predictions
    if not filtered_predictions:
        return predictions[:3]

    # Sort by probability and return top 3
    filtered_predictions.sort(key=lambda x: x['probability'], reverse=True)
    return filtered_predictions[:3]

# 오른쪽 영역 전용 예측 함수
def predict_right_result1(pattern1_number):
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

def predict_right_result2(pattern2_number):
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

# 오른쪽 영역 전용 필터링 함수
def filter_right_predictions_by_pattern(pattern_number, predictions):
    """Filter predictions based on pattern search results (for pattern1 or pattern2)"""
    if not predictions:
        return predictions

    # Load pattern data
    pattern_data = load_pattern_data()
    if not pattern_data:
        return predictions

    # Get first 2 digits of pattern_number
    search_prefix = str(pattern_number)[:2]
    
    # Search patterns with the prefix
    initial_patterns = search_patterns(pattern_data, search_prefix)
    
    if not initial_patterns:
        return predictions[:3]

    # Get 4th, 5th, 6th items from sequences
    target_sequences = []
    for pattern in initial_patterns:
        sequence = pattern['sequence']
        if len(sequence) >= 6:
            target_seq = ''.join(sequence[3:6]).lower()
            target_sequences.append(target_seq)

    if not target_sequences:
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

    # Get unique pattern numbers and their group values from related patterns
    related_patterns_dict = {}
    for pattern in related_patterns:
        if pattern['pattern_number'] != 'N/A':
            related_patterns_dict[pattern['pattern_number']] = {
                'group': pattern['group'],
                'group_value': pattern['group_value'],
                'sequence': pattern['sequence']
            }

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

    # If no predictions match the filter, return top 3 original predictions
    if not filtered_predictions:
        return predictions[:3]

    # Sort by probability and return top 3
    filtered_predictions.sort(key=lambda x: x['probability'], reverse=True)
    return filtered_predictions[:3]

def filter_right_predictions_by_pattern2(pattern2_number, predictions):
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
        return predictions[:3]
    search_prefix = pattern2_str[2:4]

    # Search patterns with the prefix
    initial_patterns = search_patterns(pattern_data, search_prefix)
    
    if not initial_patterns:
        return predictions[:3]

    # Get 4th, 5th, 6th items from sequences
    target_sequences = []
    for pattern in initial_patterns:
        sequence = pattern['sequence']
        if len(sequence) >= 6:
            target_seq = ''.join(sequence[3:6]).lower()
            target_sequences.append(target_seq)

    if not target_sequences:
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

    # Get unique pattern numbers and their group values from related patterns
    related_patterns_dict = {}
    for pattern in related_patterns:
        if pattern['pattern_number'] != 'N/A':
            related_patterns_dict[pattern['pattern_number']] = {
                'group': pattern['group'],
                'group_value': pattern['group_value'],
                'sequence': pattern['sequence']
            }

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

    # If no predictions match the filter, return top 3 original predictions
    if not filtered_predictions:
        return predictions[:3]

    # Sort by probability and return top 3
    filtered_predictions.sort(key=lambda x: x['probability'], reverse=True)
    return filtered_predictions[:3]

def main():
    # Set full page width
    st.title("Bead Road Parser v4")
    
    # Initialize session state for left column (common)
    if 'text_key' not in st.session_state:
        st.session_state.text_key = 0
    if 'grid' not in st.session_state:
        st.session_state.grid = None
    if 'show_grid' not in st.session_state:
        st.session_state.show_grid = False
    if 'converted_grid' not in st.session_state:
        st.session_state.converted_grid = None
    if 'selected_cell' not in st.session_state:
        st.session_state.selected_cell = None
    if 'converted_grid_history' not in st.session_state:
        st.session_state.converted_grid_history = []
    
    # Initialize session state for middle column (Number Prediction)
    if 'middle_pattern1_number' not in st.session_state:
        st.session_state.middle_pattern1_number = ''
    if 'middle_pattern2_number' not in st.session_state:
        st.session_state.middle_pattern2_number = ''
    if 'middle_result1' not in st.session_state:
        st.session_state.middle_result1 = None
    if 'middle_result2' not in st.session_state:
        st.session_state.middle_result2 = None
    
    # Initialize session state for right column (Number Prediction)
    if 'right_pattern1_number' not in st.session_state:
        st.session_state.right_pattern1_number = ''
    if 'right_pattern2_number' not in st.session_state:
        st.session_state.right_pattern2_number = ''
    if 'right_result1' not in st.session_state:
        st.session_state.right_result1 = None
    if 'right_result2' not in st.session_state:
        st.session_state.right_result2 = None
    
    # Split screen into three equal columns
    left_col, middle_col, right_col = st.columns([1, 1, 1])
    
    # Left column: SVG input and analysis results
    with left_col:
        svg_code = st.text_area("Paste SVG code here", height=68, key=f"svg_input_{st.session_state.text_key}")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Reset"):
                st.session_state.text_key += 1
                st.session_state.grid = None
                st.session_state.show_grid = False
                st.session_state.converted_grid = None
                st.session_state.selected_cell = None
                st.rerun()
        
        with col2:
            if st.button("Parse SVG"):
                if svg_code:
                    try:
                        grid = parse_bead_road_svg(svg_code)
                        st.session_state.grid = grid
                        st.session_state.show_grid = True
                        st.session_state.converted_grid = convert_tie_values(grid)
                        st.session_state.selected_cell = None
                        st.success("Successfully parsed the SVG code!")
                    except Exception as e:
                        st.error(f"Error parsing SVG: {str(e)}")
                else:
                    st.warning("Please paste SVG code first")
        
        # Display Full Grid if available
        if st.session_state.show_grid and st.session_state.grid is not None:
            display_grid_with_title(st.session_state.grid, "Full Grid")
            
            # Apply T conversion rule
            if st.session_state.converted_grid is None:
                st.session_state.converted_grid = convert_tie_values(st.session_state.grid)
            display_grid_with_title(st.session_state.converted_grid, "Converted Grid")
            
            # Manual input for empty cells below the table
            with st.expander("수동 입력 (Converted Grid)", expanded=True):
                empty_cells = [(x+1, y+1) for x in range(TABLE_WIDTH) for y in range(TABLE_HEIGHT) if not st.session_state.converted_grid[x][y]]
                if empty_cells:
                    selected = st.selectbox("비어있는 셀 좌표를 선택하세요 (X, Y)", empty_cells, key="empty_cell_select")
                    st.session_state.selected_cell = selected
                    x, y = selected[0]-1, selected[1]-1
                    st.info(f"선택된 셀: X={x+1}, Y={y+1}")

                    # B/P 버튼 선택 UI
                    if 'bp_btn_value' not in st.session_state:
                        st.session_state.bp_btn_value = 'B'
                    colb, colp = st.columns([1,1], gap="large")
                    with colb:
                        if st.button('B', key='bp_btn_b', help='B 선택', use_container_width=True):
                            st.session_state.bp_btn_value = 'B'
                    with colp:
                        if st.button('P', key='bp_btn_p', help='P 선택', use_container_width=True):
                            st.session_state.bp_btn_value = 'P'
                    st.markdown(f'<div style="margin-top:0.5rem;font-size:1.2rem;font-weight:bold;">현재 선택: <span style="color:#1e40af">{st.session_state.bp_btn_value}</span></div>', unsafe_allow_html=True)

                    col_apply, col_undo = st.columns([1,1])
                    with col_apply:
                        if st.button("적용", key="apply_manual2"):
                            st.session_state.converted_grid_history.append(copy.deepcopy(st.session_state.converted_grid))
                            st.session_state.converted_grid[x][y] = st.session_state.bp_btn_value.lower()
                            st.success(f"({x+1}, {y+1}) 셀을 {st.session_state.bp_btn_value}로 변경했습니다.")
                            st.session_state.selected_cell = None
                            st.rerun()
                    with col_undo:
                        if st.button("되돌리기", key="undo_manual2", disabled=len(st.session_state.converted_grid_history) == 0):
                            if st.session_state.converted_grid_history:
                                st.session_state.converted_grid = st.session_state.converted_grid_history.pop()
                                st.success("이전 상태로 되돌렸습니다.")
                                st.session_state.selected_cell = None
                                st.rerun()
                else:
                    st.info("비어있는 셀이 없습니다.")
            
            # Process zones and display pattern analysis
            zones = divide_grid_into_overlapping_zones(st.session_state.converted_grid)
            if zones:
                display_pattern_groups(zones)
            else:
                st.info("No zones with relevant data to display.")
    
    # Middle column: Number Prediction UI - 완전히 분리된 함수 사용
    with middle_col:
        st.subheader("Number Prediction")
        if st.button("모델 학습/재학습", key="middle_train_model_btn"):
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            train_number_model()
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Pattern1 → Result1 예측")
            middle_pattern1_input = st.text_input(
                "pattern1_number 입력",
                value=st.session_state.middle_pattern1_number,
                key="middle_p1_input"
            )
            
            middle_predict1_clicked = st.button("Pattern1 예측", key="middle_p1_btn")
            middle_p1_reset_clicked = st.button("초기화", key="middle_p1_reset_btn")
            
            if middle_predict1_clicked and middle_pattern1_input:
                # 중앙 영역 전용 함수 사용
                st.session_state.middle_pattern1_number = middle_pattern1_input
                with st.spinner("예측 중..."):
                    predictions = predict_middle_result1(middle_pattern1_input)
                    if predictions:
                        filtered_predictions = filter_middle_predictions_by_pattern(middle_pattern1_input, predictions)
                        st.session_state.middle_result1 = filtered_predictions
                
            if middle_p1_reset_clicked:
                st.session_state.middle_pattern1_number = ''
                st.session_state.middle_result1 = None
            
            if st.session_state.middle_result1:
                st.success("예측 결과:")
                for i, pred in enumerate(st.session_state.middle_result1, 1):
                    group_info = f" (그룹: {pred.get('group', 'N/A')})" if 'group' in pred else ""
                    st.write(f"{i}위: {pred['number']}{group_info} (확률: {pred['probability']:.2%})")
        
        with col2:
            st.markdown("#### Pattern2 → Result2 예측")
            middle_pattern2_input = st.text_input(
                "pattern2_number 입력",
                value=st.session_state.middle_pattern2_number,
                key="middle_p2_input"
            )
            
            middle_predict2_clicked = st.button("Pattern2 예측", key="middle_p2_btn")
            middle_p2_reset_clicked = st.button("초기화", key="middle_p2_reset_btn")
            
            if middle_predict2_clicked and middle_pattern2_input:
                # 중앙 영역 전용 함수 사용
                st.session_state.middle_pattern2_number = middle_pattern2_input
                with st.spinner("예측 중..."):
                    predictions = predict_middle_result2(middle_pattern2_input)
                    if predictions:
                        filtered_predictions = filter_middle_predictions_by_pattern2(middle_pattern2_input, predictions)
                        st.session_state.middle_result2 = filtered_predictions
                
            if middle_p2_reset_clicked:
                st.session_state.middle_pattern2_number = ''
                st.session_state.middle_result2 = None
            
            if st.session_state.middle_result2:
                st.success("예측 결과:")
                for i, pred in enumerate(st.session_state.middle_result2, 1):
                    group_info = f" (그룹: {pred.get('group', 'N/A')})" if 'group' in pred else ""
                    st.write(f"{i}위: {pred['number']}{group_info} (확률: {pred['probability']:.2%})")
    
    # Right column: Number Prediction UI (복제) - 완전히 분리된 함수 사용
    with right_col:
        st.subheader("Number Prediction (Right)")
        if st.button("모델 학습/재학습", key="right_train_model_btn"):
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            train_number_model()
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Pattern1 → Result1 예측")
            right_pattern1_input = st.text_input(
                "pattern1_number 입력",
                value=st.session_state.right_pattern1_number,
                key="right_p1_input"
            )
            
            right_predict1_clicked = st.button("Pattern1 예측", key="right_p1_btn")
            right_p1_reset_clicked = st.button("초기화", key="right_p1_reset_btn")
            
            if right_predict1_clicked and right_pattern1_input:
                # 오른쪽 영역 전용 함수 사용
                st.session_state.right_pattern1_number = right_pattern1_input
                with st.spinner("예측 중..."):
                    predictions = predict_right_result1(right_pattern1_input)
                    if predictions:
                        filtered_predictions = filter_right_predictions_by_pattern(right_pattern1_input, predictions)
                        st.session_state.right_result1 = filtered_predictions
                
            if right_p1_reset_clicked:
                st.session_state.right_pattern1_number = ''
                st.session_state.right_result1 = None
            
            if st.session_state.right_result1:
                st.success("예측 결과:")
                for i, pred in enumerate(st.session_state.right_result1, 1):
                    group_info = f" (그룹: {pred.get('group', 'N/A')})" if 'group' in pred else ""
                    st.write(f"{i}위: {pred['number']}{group_info} (확률: {pred['probability']:.2%})")
        
        with col2:
            st.markdown("#### Pattern2 → Result2 예측")
            right_pattern2_input = st.text_input(
                "pattern2_number 입력",
                value=st.session_state.right_pattern2_number,
                key="right_p2_input"
            )
            
            right_predict2_clicked = st.button("Pattern2 예측", key="right_p2_btn")
            right_p2_reset_clicked = st.button("초기화", key="right_p2_reset_btn")
            
            if right_predict2_clicked and right_pattern2_input:
                # 오른쪽 영역 전용 함수 사용
                st.session_state.right_pattern2_number = right_pattern2_input
                with st.spinner("예측 중..."):
                    predictions = predict_right_result2(right_pattern2_input)
                    if predictions:
                        filtered_predictions = filter_right_predictions_by_pattern2(right_pattern2_input, predictions)
                        st.session_state.right_result2 = filtered_predictions
                
            if right_p2_reset_clicked:
                st.session_state.right_pattern2_number = ''
                st.session_state.right_result2 = None
            
            if st.session_state.right_result2:
                st.success("예측 결과:")
                for i, pred in enumerate(st.session_state.right_result2, 1):
                    group_info = f" (그룹: {pred.get('group', 'N/A')})" if 'group' in pred else ""
                    st.write(f"{i}위: {pred['number']}{group_info} (확률: {pred['probability']:.2%})")

if __name__ == "__main__":
    main()