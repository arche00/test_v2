import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
from bs4 import BeautifulSoup
import json
import sqlite3
import uuid
import copy  # Add copy module for deep copy

# 페이지 설정을 가장 먼저 실행
st.set_page_config(
    page_title="Pattern Analysis System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

def convert_tie_values(grid):
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
    try:
        with open('pattern.json', 'r') as f:
            pattern_data = json.load(f)
        
        pattern_values = [v.lower() for v in pattern_values if v]
        
        for group_name in ['groupA', 'groupB']:
            patterns = pattern_data['patterns'][group_name]
            for pattern in patterns:
                if pattern.get('sequence') == pattern_values:
                    return pattern.get('group', group_name[5].lower())
        
        return None
    except Exception as e:
        st.error(f"패턴 그룹 검색 중 오류 발생: {str(e)}")
        return None

def get_first_two_group_values(zone):
    patterns = get_pattern_positions()
    group_patterns = [p for p in patterns if p['columns'][0] >= zone['start_x'] and p['columns'][1] <= zone['end_x']]
    
    if len(group_patterns) < 4:
        return ''
        
    pattern_values = []
    for pattern in group_patterns[:4]:
        values = []
        for x, y in pattern['coordinates']:
            relative_x = x - zone['start_x']
            value = zone['zone_data'][relative_x][y]
            if value:
                values.append(value.upper())
        pattern_values.append(values)
        
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
    
    pattern_123_text = ''.join(groups_123) if pattern_123_valid and len(groups_123) == 3 else ''
    return pattern_123_text[:2] if len(pattern_123_text) >= 2 else ''

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

def display_pattern_groups(zones):
    if not zones:
        return
    
    st.markdown("### Pattern Group Analysis")
    
    # Display all groups' first 2 values concatenated
    all_first_two = ''
    for zone in zones:
        first_two = get_first_two_group_values(zone)
        if first_two:
            all_first_two += first_two
    
    if all_first_two:
        st.text(f"All groups' first 2 values: {all_first_two}")
        st.markdown("---")
    
    # Sort zones by start_x to display in order
    sorted_zones = sorted(zones, key=lambda x: x['start_x'])
    
    for zone in sorted_zones:
        patterns = get_pattern_positions()
        group_patterns = [p for p in patterns if p['columns'][0] >= zone['start_x'] and p['columns'][1] <= zone['end_x']]
        
        if len(group_patterns) < 4:
            continue
            
        pattern_values = []
        for pattern in group_patterns[:4]:
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
        
        pattern_123_text = ''.join(groups_123) if pattern_123_valid and len(groups_123) == 3 else ''
        pattern_1234_text = ''.join(groups_1234) if pattern_1234_valid and len(groups_1234) == 4 else ''
        
        first_two = get_first_two_group_values(zone)
        group_range = f"{zone['start_x'] + 1}-{zone['end_x'] + 1}"
        
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
            
            st.text(f"Pattern 1,2,3 Group: {pattern_123_text}")
            st.text(f"Pattern 1,2,3,4 Group: {pattern_1234_text}")
            st.text(f"First 2 values: {first_two}")
            st.markdown("---")

def get_pattern_sequence_type(zone):
    """Get pattern sequence type from zone data for Pattern 1,2"""
    try:
        # Get 1st row 3rd column value (index 0,2)
        value = zone['zone_data'][2][0] if len(zone['zone_data']) > 2 and len(zone['zone_data'][2]) > 0 else ''
        return 'P_Sequence' if value.upper() == 'P' else 'B_Sequence' if value.upper() == 'B' else ''
    except Exception as e:
        st.error(f"Error in get_pattern_sequence_type: {str(e)}")  # Error log
        return ''

def get_pattern123_sequence_type(zone):
    """Get pattern sequence type from zone data for Pattern 1,2,3"""
    try:
        # Get 4th row 3rd column value (index 3,2)
        value = zone['zone_data'][2][3] if len(zone['zone_data']) > 2 and len(zone['zone_data'][2]) > 3 else ''
        return 'P_Sequence' if value.upper() == 'P' else 'B_Sequence' if value.upper() == 'B' else ''
    except Exception as e:
        st.error(f"Error in get_pattern123_sequence_type: {str(e)}")  # Error log
        return ''

def get_pattern_from_zone(zone):
    """Extract pattern from zone data"""
    try:
        # Get Pattern 1,2 result (2nd row 3rd column)
        pattern = zone['zone_data'][2][1] if len(zone['zone_data']) > 2 and len(zone['zone_data'][2]) > 1 else ''
        return pattern.upper() if pattern else ''
    except Exception as e:
        return ''

def get_pattern_prediction(pattern, sequence_type):
    """Get prediction from CSV file based on pattern and sequence type"""
    try:
        df = pd.read_csv('/Users/tj/test_v3/pattern1_result_v1.csv')
        if sequence_type and pattern:
            # Map sequence type to prediction column
            prediction_col = 'P_Prediction' if sequence_type == 'P_Sequence' else 'B_Prediction'
            # Remove leading zero from 4-digit pattern numbers
            pattern_str = str(pattern).strip()
            if len(pattern_str) == 4 and pattern_str.startswith('0'):
                pattern_str = pattern_str[1:]
            # Find matching pattern in Pattern_Number column
            filtered_df = df[df['Pattern_Number'].astype(str).str.strip() == pattern_str]
            if not filtered_df.empty:
                return filtered_df[prediction_col].iloc[0], True
        return '', False
    except Exception as e:
        st.error(f"Error in get_pattern_prediction: {str(e)}")
        return '', False

def get_pattern123_prediction(pattern, sequence_type):
    """Get prediction from CSV file based on pattern 1,2,3 and sequence type"""
    try:
        df = pd.read_csv('/Users/tj/test_v3/pattern2_result_v1.csv')
        if sequence_type and pattern:
            # Map sequence type to prediction column
            prediction_col = 'P_Prediction' if sequence_type == 'P_Sequence' else 'B_Prediction'
            # Remove leading zero from 6-digit pattern numbers
            pattern_str = str(pattern).strip()
            if len(pattern_str) == 6 and pattern_str.startswith('0'):
                pattern_str = pattern_str[1:]
            # Find matching pattern in Pattern_Number column
            filtered_df = df[df['Pattern_Number'].astype(str).str.strip() == pattern_str]
            if not filtered_df.empty:
                return filtered_df[prediction_col].iloc[0], True
        return '', False
    except Exception as e:
        st.error(f"Error in get_pattern123_prediction: {str(e)}")
        return '', False

def compare_pattern_prediction(pattern, prediction):
    """Compare pattern result with prediction"""
    if pattern and prediction:
        return 'w' if pattern.upper() == prediction.upper() else 'l'
    return ''

def get_pattern_results(zone):
    """Extract pattern results and predictions from zone data"""
    try:
        # Get Pattern 1,2 result (2nd row 3rd column)
        pattern1_2 = zone['zone_data'][2][1] if len(zone['zone_data']) > 2 and len(zone['zone_data'][2]) > 1 else ''
        # Get Pattern 1,2,3 result (5th row 3rd column)
        pattern1_2_3 = zone['zone_data'][2][4] if len(zone['zone_data']) > 2 and len(zone['zone_data'][2]) > 4 else ''
        
        # Get patterns from zone for pattern number combination
        patterns = get_pattern_positions()
        group_patterns = [p for p in patterns if p['columns'][0] >= zone['start_x'] and p['columns'][1] <= zone['end_x']]
        
        if len(group_patterns) >= 4:
            pattern_values = []
            for pattern in group_patterns[:4]:
                values = []
                for x, y in pattern['coordinates']:
                    relative_x = x - zone['start_x']
                    value = zone['zone_data'][relative_x][y]
                    if value:
                        values.append(value.upper())
                pattern_values.append(values)
                
            # Get pattern numbers
            pattern_numbers = []
            for v in pattern_values[:4]:
                pattern_number = find_pattern_number_only([x.lower() for x in v]) if v else None
                pattern_numbers.append(pattern_number if pattern_number is not None else '-')
            
            # Get Pattern 1,2 combination
            pattern1_2_combined = pattern_numbers[0] + pattern_numbers[1] if pattern_numbers[0] != '-' and pattern_numbers[1] != '-' else '-'
            # Get Pattern 1,2,3 combination
            pattern1_2_3_combined = pattern_numbers[0] + pattern_numbers[1] + pattern_numbers[2] if all(p != '-' for p in pattern_numbers[:3]) else '-'
            # Get Pattern 3,4 combination
            pattern3_4_combined = pattern_numbers[2] + pattern_numbers[3] if pattern_numbers[2] != '-' and pattern_numbers[3] != '-' else '-'
        else:
            pattern1_2_combined = '-'
            pattern1_2_3_combined = '-'
            pattern3_4_combined = '-'
        
        # Get sequence types for each pattern
        sequence_type_12 = get_pattern_sequence_type(zone)
        sequence_type_123 = get_pattern123_sequence_type(zone)
        
        # Get predictions using respective sequence types
        prediction1_2, found1_2 = get_pattern_prediction(pattern1_2_combined, sequence_type_12)
        prediction1_2_3, found1_2_3 = get_pattern123_prediction(pattern1_2_3_combined, sequence_type_123)
        
        # Compare and get results
        comparison1_2 = compare_pattern_prediction(pattern1_2, prediction1_2)
        comparison1_2_3 = compare_pattern_prediction(pattern1_2_3, prediction1_2_3)
        
        return pattern1_2, pattern1_2_3, prediction1_2, prediction1_2_3, comparison1_2, comparison1_2_3, sequence_type_12, pattern1_2_combined, pattern1_2_3_combined, pattern3_4_combined
    except Exception as e:
        return '', '', '', '', '', '', '', '', '', ''

def save_pattern_analysis(zones, session_id):
    """Save pattern analysis results to database"""
    try:
        # Get absolute path to database file
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pattern_analysis_v2.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get current date and total groups
        current_date = datetime.now().date()
        total_groups = len(zones)
        
        # Prepare data for insertion
        for idx, zone in enumerate(zones, 1):
            group_range = f"{zone['start_x'] + 1}-{zone['end_x'] + 1}"
            
            # Get pattern results
            pattern1_2, pattern1_2_3, prediction1_2, prediction1_2_3, comparison1_2, comparison1_2_3, sequence_type, pattern1_2_combined, pattern1_2_3_combined, pattern3_4_combined = get_pattern_results(zone)
            
            # Calculate prediction accuracy
            total_predictions = 0
            correct_predictions = 0
            if comparison1_2:
                total_predictions += 1
                if comparison1_2 == 'W':
                    correct_predictions += 1
            if comparison1_2_3:
                total_predictions += 1
                if comparison1_2_3 == 'W':
                    correct_predictions += 1
            prediction_accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
            
            # Insert data
            cursor.execute('''
                INSERT INTO pattern_analysis (
                    session_id, session_date, total_groups_in_session,
                    group_id, group_start, group_end, group_sequence,
                    pattern12_result, pattern12_combined, pattern12_prediction, pattern12_prediction_result,
                    pattern123_result, pattern123_combined, pattern123_prediction, pattern123_prediction_result,
                    sequence_type, prediction_accuracy
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                session_id, current_date, total_groups,
                group_range, zone['start_x'] + 1, zone['end_x'] + 1, idx,
                pattern1_2, pattern1_2_combined, prediction1_2, comparison1_2,
                pattern1_2_3, pattern1_2_3_combined, prediction1_2_3, comparison1_2_3,
                sequence_type, prediction_accuracy
            ))
        
        conn.commit()
        return True
            
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()

def init_db():
    """Initialize database and create tables if they don't exist"""
    try:
        # Get absolute path to database file
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pattern_analysis_v2.db')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create pattern_analysis table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pattern_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                session_date TEXT NOT NULL,
                total_groups_in_session INTEGER NOT NULL,
                group_id TEXT NOT NULL,
                group_start INTEGER NOT NULL,
                group_end INTEGER NOT NULL,
                group_sequence INTEGER NOT NULL,
                pattern12_result TEXT,
                pattern12_combined TEXT,
                pattern12_prediction TEXT,
                pattern12_prediction_result TEXT,
                pattern123_result TEXT,
                pattern123_combined TEXT,
                pattern123_prediction TEXT,
                pattern123_prediction_result TEXT,
                sequence_type TEXT,
                prediction_accuracy REAL,
                created_at TIMESTAMP DEFAULT (strftime('%Y-%m-%d %H:%M:%S', datetime('now', '+9 hours'))),
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        st.success(f"Database initialized at: {db_path}")
    except Exception as e:
        st.error(f"Database initialization error: {str(e)}")
    finally:
        if conn:
            conn.close()

# Initialize database when the app starts
init_db()

def main():
    # Set full page width
    st.markdown("""
        <style>
        .stApp {margin-top: -2.5rem;}
        div[data-testid="stExpander"],
        div[data-testid="stExpander"] *,
        div[data-testid="stVerticalBlock"],
        div[data-testid="stVerticalBlock"] *,
        div[data-testid="stElementContainer"],
        div[data-testid="stElementContainer"] *,
        div[data-testid="stHorizontalBlock"],
        div[data-testid="stHorizontalBlock"] *,
        div[data-testid="stColumn"],
        div[data-testid="stColumn"] * {
            margin: 0 !important;
            padding: 0 !important;
            box-shadow: none !important;
            background: none !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Bead Road Parser V2")
    
    # Initialize session state
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
    if 'converted_grid_history' not in st.session_state:  # Add history state
        st.session_state.converted_grid_history = []
    
    # Split screen into left and right columns (1:1 ratio)
    left_col, right_col = st.columns([1, 1])
    
    # Left column: SVG input and analysis results
    with left_col:
        svg_code = st.text_area("Paste SVG code here", height=68, key=f"svg_input_{st.session_state.text_key}")
        
        col1, col2, col3 = st.columns([1, 3, 1])
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
        
        with col3:
            if st.button("Save Pattern"):
                if st.session_state.show_grid and st.session_state.converted_grid is not None:
                    zones = divide_grid_into_overlapping_zones(st.session_state.converted_grid)
                    if zones:
                        # Generate unique session ID
                        session_id = str(uuid.uuid4())
                        if save_pattern_analysis(zones, session_id):
                            try:
                                # Get all groups' first 2 values concatenated
                                all_first_two = ''
                                for zone in zones:
                                    first_two = get_first_two_group_values(zone)
                                    if first_two:
                                        all_first_two += first_two
                                
                                if all_first_two:  # Only save if we have data
                                    # Save to pattern_analysis_v3.db
                                    db_path = '/Users/tj/test_v3/pattern_analysis_v3.db'
                                    conn = sqlite3.connect(db_path)
                                    cursor = conn.cursor()
                                    
                                    # Create group_sequences table if not exists
                                    cursor.execute('''
                                        CREATE TABLE IF NOT EXISTS group_sequences (
                                            round INTEGER PRIMARY KEY AUTOINCREMENT,
                                            timestamp TEXT,
                                            tot TEXT
                                        )
                                    ''')
                                    
                                    # Create pattern_records table if not exists
                                    cursor.execute('''
                                        CREATE TABLE IF NOT EXISTS pattern_records (
                                            round INTEGER PRIMARY KEY AUTOINCREMENT,
                                            timestamp TEXT,
                                            group_range TEXT,
                                            pattern1 TEXT,
                                            result1 TEXT,
                                            pattern2 TEXT,
                                            result2 TEXT,
                                            prev_pattern1 TEXT,
                                            prev_pattern2 TEXT,
                                            transition_type TEXT,
                                            transition_count INTEGER DEFAULT 1,
                                            pattern1_banker_count INTEGER,
                                            pattern1_player_count INTEGER,
                                            pattern2_banker_count INTEGER,
                                            pattern2_player_count INTEGER,
                                            pattern1_transitions INTEGER,
                                            pattern2_transitions INTEGER,
                                            pattern1_number TEXT,
                                            result1_number TEXT,
                                            pattern2_number TEXT,
                                            result2_number TEXT
                                        )
                                    ''')
                                    
                                    # Format timestamp as YYMMDDHHMM
                                    current_time = datetime.now()
                                    timestamp = current_time.strftime("%y%m%d%H%M")
                                    
                                    # Insert group sequence
                                    cursor.execute('''
                                        INSERT INTO group_sequences (timestamp, tot)
                                        VALUES (?, ?)
                                    ''', (timestamp, all_first_two))
                                    
                                    # Save pattern records for each zone
                                    for zone in zones:
                                        patterns = get_pattern_positions()
                                        group_patterns = [p for p in patterns if p['columns'][0] >= zone['start_x'] and p['columns'][1] <= zone['end_x']]
                                        
                                        if len(group_patterns) < 4:
                                            continue
                                            
                                        # Extract pattern values
                                        pattern_values = []
                                        for pattern in group_patterns[:4]:
                                            values = []
                                            for x, y in pattern['coordinates']:
                                                relative_x = x - zone['start_x']
                                                value = zone['zone_data'][relative_x][y]
                                                if value:
                                                    values.append(value.upper())
                                            pattern_values.append(values)
                                        
                                        # Get pattern numbers
                                        pattern_numbers = []
                                        for v in pattern_values[:4]:
                                            pattern_number = find_pattern_number_only([x.lower() for x in v]) if v else None
                                            pattern_numbers.append(pattern_number if pattern_number is not None else '-')
                                        
                                        # Process pattern numbers
                                        numbers_dict = process_pattern_numbers(pattern_numbers)
                                        
                                        # Get pattern groups
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
                                        
                                        pattern_123_text = ''.join(groups_123) if pattern_123_valid and len(groups_123) == 3 else ''
                                        pattern_1234_text = ''.join(groups_1234) if pattern_1234_valid and len(groups_1234) == 4 else ''
                                        
                                        # Get group range
                                        group_range = f"{zone['start_x'] + 1}-{zone['end_x'] + 1}"
                                        
                                        # Calculate pattern stats
                                        def calculate_pattern_stats(pattern):
                                            if not pattern:
                                                return 0, 0, 0
                                            banker_count = pattern.count('a')
                                            player_count = pattern.count('b')
                                            transitions = sum(1 for i in range(len(pattern)-1) if pattern[i] != pattern[i+1])
                                            return banker_count, player_count, transitions
                                        
                                        pattern1 = pattern_123_text[:2] if pattern_123_text else ''
                                        pattern1_banker_count, pattern1_player_count, pattern1_transitions = calculate_pattern_stats(pattern1)
                                        pattern2 = pattern_1234_text[:3] if pattern_1234_text else ''
                                        pattern2_banker_count, pattern2_player_count, pattern2_transitions = calculate_pattern_stats(pattern2)
                                        
                                        # Get previous patterns
                                        prev_record = cursor.execute('''
                                            SELECT pattern1, pattern2 
                                            FROM pattern_records 
                                            ORDER BY round DESC LIMIT 1
                                        ''').fetchone()
                                        prev_pattern1 = prev_record[0] if prev_record else None
                                        prev_pattern2 = prev_record[1] if prev_record else None
                                        
                                        # Calculate transition
                                        transition_type = None
                                        transition_count = 1
                                        if prev_pattern1 and pattern1:
                                            transition_type = f"{prev_pattern1}->{pattern1}"
                                            prev_transition = cursor.execute('''
                                                SELECT transition_count 
                                                FROM pattern_records 
                                                WHERE transition_type = ? 
                                                ORDER BY round DESC LIMIT 1
                                            ''', (transition_type,)).fetchone()
                                            if prev_transition:
                                                transition_count = prev_transition[0] + 1
                                        
                                        # Insert pattern record
                                        cursor.execute('''
                                            INSERT INTO pattern_records 
                                            (timestamp, group_range, pattern1, result1, pattern2, result2,
                                             prev_pattern1, prev_pattern2, transition_type, transition_count,
                                             pattern1_banker_count, pattern1_player_count, pattern2_banker_count, pattern2_player_count,
                                             pattern1_transitions, pattern2_transitions,
                                             pattern1_number, result1_number, pattern2_number, result2_number)
                                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                        ''', (timestamp, group_range, 
                                              pattern1, pattern_123_text[2] if len(pattern_123_text) >= 3 else '',
                                              pattern2, pattern_1234_text[3] if len(pattern_1234_text) >= 4 else '',
                                              prev_pattern1, prev_pattern2, transition_type, transition_count,
                                              pattern1_banker_count, pattern1_player_count, pattern2_banker_count, pattern2_player_count,
                                              pattern1_transitions, pattern2_transitions,
                                              numbers_dict['pattern1_number'], numbers_dict['result1_number'],
                                              numbers_dict['pattern2_number'], numbers_dict['result2_number']))
                                    
                                    conn.commit()
                                    st.success("패턴이 성공적으로 저장되었습니다!")
                                else:
                                    st.warning("저장할 패턴 데이터가 없습니다.")
                            except Exception as e:
                                st.error(f"패턴 저장 중 오류 발생: {str(e)}")
                            finally:
                                if conn:
                                    conn.close()
                        else:
                            st.error("패턴 저장에 실패했습니다.")
                    else:
                        st.warning("저장할 패턴이 없습니다.")
                else:
                    st.warning("먼저 SVG 코드를 파싱해주세요.")
        
        # Display Full Grid if available
        if st.session_state.show_grid and st.session_state.grid is not None:
            display_grid_with_title(st.session_state.grid, "Full Grid")
            
            # Apply T conversion rule
            if st.session_state.converted_grid is None:
                st.session_state.converted_grid = convert_tie_values(st.session_state.grid)
            display_grid_with_title(st.session_state.converted_grid, "Converted Grid")
            
            # Manual input for empty cells below the table
            with st.expander("수동 입력 (Converted Grid)", expanded=True):
                # Find all empty cells
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
                            # Save current grid to history for undo
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
    
    # Right column: Group Result
    with right_col:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Group Result")
        with col2:
            if st.button("Save Analysis", key="save_analysis"):
                if st.session_state.show_grid and st.session_state.converted_grid is not None:
                    zones = divide_grid_into_overlapping_zones(st.session_state.converted_grid)
                    if zones:
                        # Generate unique session ID
                        session_id = str(uuid.uuid4())
                        if save_pattern_analysis(zones, session_id):
                            # Save prediction results to database
                            try:
                                # Collect all prediction results
                                all_prediction_results = []
                                sorted_zones_results = sorted(zones, key=lambda x: x['start_x'])  # Left to right order
                                for zone in sorted_zones_results:
                                    pattern1_2, pattern1_2_3, prediction1_2, prediction1_2_3, comparison1_2, comparison1_2_3, sequence_type, pattern1_2_combined, pattern1_2_3_combined, pattern3_4_combined = get_pattern_results(zone)
                                    if comparison1_2:
                                        all_prediction_results.append(comparison1_2.upper())
                                    if comparison1_2_3:
                                        all_prediction_results.append(comparison1_2_3.upper())
                                
                                if all_prediction_results:
                                    combined_results = ''.join(all_prediction_results)
                                    
                                    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pattern_analysis_v2.db')
                                    conn = sqlite3.connect(db_path)
                                    cursor = conn.cursor()
                                    
                                    # Create table if it doesn't exist
                                    cursor.execute('''
                                        CREATE TABLE IF NOT EXISTS session_prediction_results (
                                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                                            session_id TEXT NOT NULL,
                                            prediction_results TEXT NOT NULL,
                                            created_at TIMESTAMP DEFAULT (datetime('now', '+9 hours'))
                                        )
                                    ''')
                                    
                                    # Insert combined results
                                    cursor.execute('''
                                        INSERT INTO session_prediction_results (session_id, prediction_results)
                                        VALUES (?, ?)
                                    ''', (session_id, combined_results))
                                    
                                    conn.commit()
                            except Exception as e:
                                st.error(f"Error saving prediction results: {str(e)}")
                            finally:
                                if conn:
                                    conn.close()
                            
                            st.success("Analysis saved successfully!")
                        else:
                            st.error("Failed to save analysis")
                    else:
                        st.warning("No zones to save")
                else:
                    st.warning("Please parse SVG code first")
        
        if st.session_state.show_grid and st.session_state.converted_grid is not None:
            zones = divide_grid_into_overlapping_zones(st.session_state.converted_grid)
            if zones:
                # Session Prediction Results: left to right
                sorted_zones_results = sorted(zones, key=lambda x: x['start_x'])
                # Group info display: right to left
                sorted_zones_groups = sorted(zones, key=lambda x: x['start_x'], reverse=True)
                
                # Collect all prediction results (Session Prediction Results)
                all_prediction_results = []
                for zone in sorted_zones_results:
                    pattern1_2, pattern1_2_3, prediction1_2, prediction1_2_3, comparison1_2, comparison1_2_3, sequence_type, pattern1_2_combined, pattern1_2_3_combined, pattern3_4_combined = get_pattern_results(zone)
                    if comparison1_2:
                        all_prediction_results.append(comparison1_2.upper())
                    if comparison1_2_3:
                        all_prediction_results.append(comparison1_2_3.upper())
                
                # Display combined prediction results
                if all_prediction_results:
                    combined_results = ''.join(all_prediction_results)
                    st.markdown("### Session Prediction Results")
                    st.markdown(f"**{combined_results}**")
                    st.markdown("---")
                
                # Insert search boxes here
                pattern12_prediction_search_box()
                pattern123_prediction_search_box()
                st.markdown("---")
                
                # Display individual group results (right to left)
                for zone in sorted_zones_groups:
                    group_range = f"{zone['start_x'] + 1}-{zone['end_x'] + 1}"
                    pattern1_2, pattern1_2_3, prediction1_2, prediction1_2_3, comparison1_2, comparison1_2_3, sequence_type, pattern1_2_combined, pattern1_2_3_combined, pattern3_4_combined = get_pattern_results(zone)

                    # Check if there is anything to display for this group
                    has_content = any([
                        pattern1_2, pattern1_2_3, prediction1_2, prediction1_2_3, comparison1_2, comparison1_2_3
                    ])
                    if not has_content:
                        continue  # Skip this group if nothing to display

                    st.markdown(f"#### Group {group_range}")
                    # Pattern 1,2 results
                    st.text(f"Pattern 1,2 combined: {pattern1_2_combined}")
                    if pattern1_2:
                        st.text(f"Pattern 1,2 result: {pattern1_2.upper()}")
                        if prediction1_2:
                            st.text(f"Pattern 1,2 Prediction: {prediction1_2.upper()}")
                            st.text(f"Pattern 1,2 Prediction Result: {comparison1_2.upper()}")
                        else:
                            st.text("No Pattern 1,2 prediction found in CSV")
                    # Pattern 1,2,3 results
                    st.text(f"Pattern 1,2,3 combined: {pattern1_2_3_combined}")
                    if pattern1_2_3:
                        st.text(f"Pattern 1,2,3 result: {pattern1_2_3.upper()}")
                        if prediction1_2_3:
                            st.text(f"Pattern 1,2,3 Prediction: {prediction1_2_3.upper()}")
                            st.text(f"Pattern 1,2,3 Prediction Result: {comparison1_2_3.upper()}")
                        else:
                            st.text("No Pattern 1,2,3 prediction found in CSV")
                    # Pattern 3,4 combined always at the end
                    st.text(f"Pattern 3,4 combined: {pattern3_4_combined}")
                    st.markdown("---")

def pattern12_prediction_search_box():
    st.markdown("#### Pattern 1,2 Prediction 검색")
    pattern_input = st.text_input("Pattern 1,2 번호 입력", key="pattern12_search_input")
    if st.button("검색", key="pattern12_search_btn"):
        if pattern_input:
            pred_p, found_p = get_pattern_prediction(pattern_input, "P_Sequence")
            pred_b, found_b = get_pattern_prediction(pattern_input, "B_Sequence")
            st.markdown("**P_Sequence 결과:**")
            if found_p:
                st.success(f"Prediction: {pred_p}")
            else:
                st.warning("No prediction found for P_Sequence.")
            st.markdown("**B_Sequence 결과:**")
            if found_b:
                st.success(f"Prediction: {pred_b}")
            else:
                st.warning("No prediction found for B_Sequence.")
        else:
            st.info("패턴 번호를 입력하세요.")

def pattern123_prediction_search_box():
    st.markdown("#### Pattern 1,2,3 Prediction 검색")
    pattern_input = st.text_input("Pattern 1,2,3 번호 입력", key="pattern123_search_input")
    if st.button("검색", key="pattern123_search_btn"):
        if pattern_input:
            pred_p, found_p = get_pattern123_prediction(pattern_input, "P_Sequence")
            pred_b, found_b = get_pattern123_prediction(pattern_input, "B_Sequence")
            st.markdown("**P_Sequence 결과:**")
            if found_p:
                st.success(f"Prediction: {pred_p}")
            else:
                st.warning("No prediction found for P_Sequence.")
            st.markdown("**B_Sequence 결과:**")
            if found_b:
                st.success(f"Prediction: {pred_b}")
            else:
                st.warning("No prediction found for B_Sequence.")
        else:
            st.info("패턴 번호를 입력하세요.")

if __name__ == "__main__":
    main() 