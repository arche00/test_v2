import streamlit as st
from bs4 import BeautifulSoup
import sqlite3
from datetime import datetime, timedelta
import json
import pandas as pd

# Database setup
def init_db():
    conn = sqlite3.connect('pattern_analysis_v3.db')
    c = conn.cursor()
    
    # pattern_records 테이블 생성 (이미 존재하면 생성하지 않음)
    c.execute('''
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
    
    # group_sequences 테이블 생성 (이미 존재하면 생성하지 않음)
    c.execute('''
        CREATE TABLE IF NOT EXISTS group_sequences (
            round INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            tot TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database when the app starts
init_db()

# Function to save pattern analysis results
def save_pattern_record(group_range, pattern_123, pattern_1234, pattern1_number, result1_number, pattern2_number, result2_number):
    """
    Save pattern analysis results.
    Args:
        group_range (str): Group range (e.g., "1-3")
        pattern_123 (str): Group value of pattern 123 (e.g., "aab")
        pattern_1234 (str): Group value of pattern 1234 (e.g., "aabb")
        pattern1_number (str): 패턴1+2 넘버
        result1_number (str): 패턴3 넘버
        pattern2_number (str): 패턴1+2+3 넘버
        result2_number (str): 패턴4 넘버
    """
    # Format timestamp as YYMMDDHHMM
    current_time = datetime.now()
    timestamp = current_time.strftime("%y%m%d%H%M")
    try:
        conn = sqlite3.connect('pattern_analysis_v3.db')
        c = conn.cursor()
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
    except Exception as e:
        st.error(f"데이터 저장 중 오류 발생: {str(e)}")
    finally:
        conn.close()

# Function to display pattern records
def display_pattern_records():
    try:
        conn = sqlite3.connect('pattern_analysis_v3.db')
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

def display_grid(grid):
    st.markdown("""
        <style>
        .grid-container { display: table; border-collapse: collapse; margin: 20px 0; }
        .grid-row { display: table-row; }
        .bead-road-cell { width: 40px; height: 40px; border: 1px solid black; display: table-cell; 
                         text-align: center; vertical-align: middle; font-family: monospace; }
        .banker { color: red; font-weight: bold; }
        .player { color: blue; font-weight: bold; }
        .tie { color: green; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)
    
    html_table = ['<div class="grid-container">']
    for y in range(6):
        html_table.append('<div class="grid-row">')
        for x in range(15):
            cell = grid[x][y]
            css_class = 'banker' if cell == 'b' else 'player' if cell == 'p' else 'tie' if cell == 't' else ''
            html_table.append(f'<div class="bead-road-cell {css_class}">{cell.upper() if cell else "&nbsp;"}</div>')
        html_table.append('</div>')
    html_table.append('</div>')
    
    st.markdown(''.join(html_table), unsafe_allow_html=True)

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
        
        conn = sqlite3.connect('pattern_analysis_v3.db')
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
    
    # Existing code
    for zone in zones:
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
        
        # Display result
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
        conn = sqlite3.connect('pattern_analysis_v3.db')
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
                    # Get values from left, left up, and up
                    left = converted_grid[x-1][y]
                    left_up = converted_grid[x-1][y-1]
                    up = converted_grid[x][y-1]
                    
                    # If left-up and up are the same, use that value
                    if left_up == up:
                        converted_grid[x][y] = up
                    # If left-up and left are the same, use that value
                    elif left_up == left:
                        converted_grid[x][y] = left
                    # Otherwise use the up value
                    else:
                        converted_grid[x][y] = up
    
    return converted_grid

def get_group_ratio_trend():
    """
    Calculate group ratio trend over time.
    """
    try:
        conn = sqlite3.connect('pattern_analysis_v3.db')
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
        conn = sqlite3.connect('pattern_analysis_v3.db')
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

def main():
    # Set full page width
    st.set_page_config(layout="wide")
    
    st.title("Bead Road Parser")
    
    # Initialize session state
    if 'text_key' not in st.session_state:
        st.session_state.text_key = 0
    if 'grid' not in st.session_state:
        st.session_state.grid = None
    if 'show_grid' not in st.session_state:
        st.session_state.show_grid = False
    
    # Split screen into left and right columns (adjust ratio)
    left_col, right_col = st.columns([7, 3])
    
    # Left column: SVG input and analysis results
    with left_col:
        svg_code = st.text_area("Paste SVG code here", height=200, key=f"svg_input_{st.session_state.text_key}")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Reset"):
                st.session_state.text_key += 1
                st.session_state.grid = None
                st.session_state.show_grid = False
                st.experimental_rerun()
        
        with col2:
            if st.button("Parse SVG"):
                if svg_code:
                    try:
                        grid = parse_bead_road_svg(svg_code)
                        st.session_state.grid = grid
                        st.session_state.show_grid = True
                        st.success("Successfully parsed the SVG code!")
                    except Exception as e:
                        st.error(f"Error parsing SVG: {str(e)}")
                else:
                    st.warning("Please paste SVG code first")
        
        # Display Full Grid if available
        if st.session_state.show_grid and st.session_state.grid is not None:
            st.subheader("Full Grid")
            display_grid(st.session_state.grid)
            
            # Apply T conversion rule
            converted_grid = convert_tie_values(st.session_state.grid)
            st.subheader("Converted Grid")
            display_grid(converted_grid)
            
            # Process zones and display pattern analysis
            zones = divide_grid_into_overlapping_zones(converted_grid)  # Use converted grid
            if zones:
                display_pattern_groups(zones)
            else:
                st.info("No zones with relevant data to display.")
    
    # Right column: Pattern analysis records
    with right_col:
        # Display pattern statistics
        stats = get_pattern_statistics()
        if stats:
            display_pattern_statistics(stats)
        
        # Display recent records
        display_recent_records()

if __name__ == "__main__":
    main()