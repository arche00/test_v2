import streamlit as st
import json
import pandas as pd
from datetime import datetime
import sqlite3
import os

# Constants
PATTERN_WIDTH = 2
PATTERN_HEIGHT = 3
GRID_WIDTH = 15
GRID_HEIGHT = 6

def search_pattern_in_db(pattern_number):
    """Search pattern in database by pattern number"""
    try:
        db_path = 'pattern_analysis_v2.db'
        conn = sqlite3.connect(db_path)
        query = '''
            SELECT 
                timestamp,
                group_range,
                pattern1,
                result1,
                pattern2,
                result2,
                pattern1_number,
                result1_number,
                pattern2_number,
                result2_number
            FROM pattern_records
            WHERE pattern1_number LIKE ?
            ORDER BY pattern1 ASC, timestamp DESC
        '''
        search_pattern = f"{pattern_number}%"
        df = pd.read_sql_query(query, conn, params=(search_pattern,))
        conn.close()
        if not df.empty:
            # Format timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%y%m%d%H%M').dt.strftime('%Y-%m-%d %H:%M')
            df = df.sort_values(by=['pattern1', 'timestamp'], ascending=[True, False])
            return df
        return None
    except Exception as e:
        st.error(f"Database search error: {str(e)}")
        return None

def load_pattern_data():
    """Load pattern data from pattern.json"""
    try:
        with open('pattern.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading pattern data: {str(e)}")
        return None

def display_pattern_grid(pattern_values, title="Pattern Grid"):
    """Display a pattern grid with given values"""
    st.markdown(f"#### {title}")
    
    # Create grid HTML with inline styles
    html_table = ['<div style="display: table; border-collapse: collapse; margin: 5px 0;">']
    
    # First column (0, 3, 6)
    for y in range(3):
        html_table.append('<div style="display: table-row;">')
        # First column
        value1 = pattern_values[y] if y < len(pattern_values) else ''
        bg_color1 = '#E6F3FF' if value1 == 'B' else '#FFE6E6' if value1 == 'P' else '#FFFFFF'
        html_table.append(f'<div style="display: table-cell; width: 25px; height: 25px; border: 1px solid black; text-align: center; vertical-align: middle; font-family: monospace; font-size: 12px; background-color: {bg_color1}; font-weight: bold;">{value1}</div>')
        
        # Second column
        value2 = pattern_values[y + 3] if y + 3 < len(pattern_values) else ''
        bg_color2 = '#E6F3FF' if value2 == 'B' else '#FFE6E6' if value2 == 'P' else '#FFFFFF'
        html_table.append(f'<div style="display: table-cell; width: 25px; height: 25px; border: 1px solid black; text-align: center; vertical-align: middle; font-family: monospace; font-size: 12px; background-color: {bg_color2}; font-weight: bold;">{value2}</div>')
        
        html_table.append('</div>')
    
    html_table.append('</div>')
    
    st.markdown(''.join(html_table), unsafe_allow_html=True)

def search_patterns(pattern_data, search_query):
    """Search patterns based on query with sequence start matching"""
    results = []
    
    # Normalize search query: remove spaces and convert to lowercase
    normalized_query = ''.join(search_query.lower().split())
    
    for group_name in ['groupA', 'groupB']:
        patterns = pattern_data['patterns'][group_name]
        for pattern in patterns:
            sequence = pattern.get('sequence', [])
            # Normalize sequence: join all values and convert to lowercase
            normalized_sequence = ''.join([s.lower() for s in sequence])
            
            # Check if sequence starts with the normalized query
            if normalized_sequence.startswith(normalized_query):
                results.append({
                    'group': group_name[5],  # 'A' or 'B'
                    'sequence': sequence,
                    'group_value': pattern.get('group', group_name[5].lower()),
                    'pattern_number': pattern.get('pattern_number', 'N/A')
                })
    
    return results

def calculate_pattern_statistics(df):
    """Calculate statistics for pattern results"""
    stats = {
        'pattern1': {
            'B': len(df[df['result1'] == 'B']),
            'P': len(df[df['result1'] == 'P'])
        },
        'pattern2': {
            'B': len(df[df['result2'] == 'B']),
            'P': len(df[df['result2'] == 'P'])
        }
    }
    return stats

def main():
    st.set_page_config(layout="wide")
    
    # Apply global CSS styles
    st.markdown("""
        <style>
        /* Global styles */
        .stMarkdown {
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
        }
        .stMarkdown p {
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
            font-size: 12px !important;
        }
        .stMarkdown h3 {
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
            font-size: 14px !important;
        }
        .stMarkdown h4 {
            margin-bottom: 0 !important;
            padding-bottom: 0 !important;
            font-size: 12px !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Pattern Search")
    
    # DB 검색 2개 영역
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Database Search 1")
        pattern_number1 = st.text_input(
            "Enter pattern number to search in database (pattern1_number only)",
            help="Enter the pattern number to search. The search will match from the beginning of the pattern number.",
            key="db_search1"
        )
        df1 = None
        found_info1 = None
        if pattern_number1:
            df1 = search_pattern_in_db(pattern_number1)
            pattern_data = load_pattern_data()
            if pattern_data:
                for group_name in ['groupA', 'groupB']:
                    for pattern in pattern_data['patterns'][group_name]:
                        if pattern.get('pattern_number') == pattern_number1:
                            found_info1 = pattern
                            break
                    if found_info1:
                        break
        info_col, card_col = st.columns([5, 1])
        with info_col:
            if pattern_number1:
                if df1 is not None:
                    st.write(f"Found {len(df1)} records")
                    st.write(f"Unique patterns: {df1['pattern1'].nunique()}")
                    stats1 = calculate_pattern_statistics(df1)
                    st.markdown("#### Pattern Statistics")
                    st.markdown(f"""
                        <div style=\"background-color:#f0f2f6;padding:10px;border-radius:5px;margin:5px 0;\">
                            <h4>Pattern 1 Results</h4>
                            <p>Banker: {stats1['pattern1']['B']}</p>
                            <p>Player: {stats1['pattern1']['P']}</p>
                        </div>
                        <div style=\"background-color:#f0f2f6;padding:10px;border-radius:5px;margin:5px 0;\">
                            <h4>Pattern 2 Results</h4>
                            <p>Banker: {stats1['pattern2']['B']}</p>
                            <p>Player: {stats1['pattern2']['P']}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.dataframe(df1, use_container_width=True)
                else:
                    st.info("No matching records found in database")
        with card_col:
            if pattern_number1 and found_info1:
                display_pattern_grid(found_info1['sequence'], f"Pattern #{found_info1['pattern_number']}")

    with col2:
        st.markdown("### Database Search 2")
        pattern_number2 = st.text_input(
            "Enter pattern number to search in database (pattern1_number only)",
            help="Enter the pattern number to search. The search will match from the beginning of the pattern number.",
            key="db_search2"
        )
        df2 = None
        found_info2 = None
        if pattern_number2:
            df2 = search_pattern_in_db(pattern_number2)
            pattern_data = load_pattern_data()
            if pattern_data:
                for group_name in ['groupA', 'groupB']:
                    for pattern in pattern_data['patterns'][group_name]:
                        if pattern.get('pattern_number') == pattern_number2:
                            found_info2 = pattern
                            break
                    if found_info2:
                        break
        info_col2, card_col2 = st.columns([5, 1])
        with info_col2:
            if pattern_number2:
                if df2 is not None:
                    st.write(f"Found {len(df2)} records")
                    st.write(f"Unique patterns: {df2['pattern1'].nunique()}")
                    stats2 = calculate_pattern_statistics(df2)
                    st.markdown("#### Pattern Statistics")
                    st.markdown(f"""
                        <div style=\"background-color:#f0f2f6;padding:10px;border-radius:5px;margin:5px 0;\">
                            <h4>Pattern 1 Results</h4>
                            <p>Banker: {stats2['pattern1']['B']}</p>
                            <p>Player: {stats2['pattern1']['P']}</p>
                        </div>
                        <div style=\"background-color:#f0f2f6;padding:10px;border-radius:5px;margin:5px 0;\">
                            <h4>Pattern 2 Results</h4>
                            <p>Banker: {stats2['pattern2']['B']}</p>
                            <p>Player: {stats2['pattern2']['P']}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.dataframe(df2, use_container_width=True)
                else:
                    st.info("No matching records found in database")
        with card_col2:
            if pattern_number2 and found_info2:
                display_pattern_grid(found_info2['sequence'], f"Pattern #{found_info2['pattern_number']}")

    st.markdown("---")

    # 패턴 JSON 기반 검색 (기존 기능)
    pattern_data = load_pattern_data()
    if not pattern_data:
        st.error("Failed to load pattern data")
        return
    
    # Search interface
    st.sidebar.header("Search Options")
    search_query = st.sidebar.text_input(
        "Enter pattern to search (e.g., 'bbbbb' for patterns starting with 'bbbbb')",
        help="Enter the starting sequence of the pattern you want to find. Use 'b' for banker and 'p' for player. Case doesn't matter."
    )
    
    # Group filter
    group_filter = st.sidebar.selectbox(
        "Filter by Group",
        ["All", "A", "B"]
    )
    
    if search_query:
        # Search patterns
        results = search_patterns(pattern_data, search_query)
        
        # Filter results by group if needed
        if group_filter != "All":
            results = [r for r in results if r['group'] == group_filter]
        
        # Display results in a grid
        if results:
            st.write(f"Found {len(results)} patterns")
            
            # Group results by group
            grouped_results = {}
            for result in results:
                group = result['group']
                if group not in grouped_results:
                    grouped_results[group] = []
                grouped_results[group].append(result)
            
            # Display each group's results horizontally
            for group, group_results in grouped_results.items():
                st.markdown(f"### Group {group}")
                cols = st.columns(len(group_results))
                for idx, result in enumerate(group_results):
                    with cols[idx]:
                        display_pattern_grid(result['sequence'], f"Pattern #{result.get('pattern_number', 'N/A')}")
                        st.write(f"Value: {result['group_value']}")
                st.markdown("---")
        else:
            st.info("No patterns found matching your search criteria")
    
    # Display pattern statistics
    st.sidebar.markdown("---")
    st.sidebar.header("Pattern Statistics")
    
    # Count patterns by group
    group_counts = {
        'A': len(pattern_data['patterns']['groupA']),
        'B': len(pattern_data['patterns']['groupB'])
    }
    
    st.sidebar.write("Patterns by Group:")
    st.sidebar.write(f"Group A: {group_counts['A']}")
    st.sidebar.write(f"Group B: {group_counts['B']}")
    st.sidebar.write(f"Total: {sum(group_counts.values())}")

if __name__ == "__main__":
    main() 