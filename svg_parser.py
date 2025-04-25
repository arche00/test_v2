import streamlit as st
from bs4 import BeautifulSoup
import json

"""
Original table input processing code is commented out for future reference
def parse_table_html(html_content):
    # ... existing code ...
    pass
"""

def parse_bead_road_svg(svg_code):
    soup = BeautifulSoup(svg_code, 'html.parser')
    
    # Initialize 6x15 grid with empty strings
    grid = [['' for _ in range(15)] for _ in range(6)]
    
    # Find all coordinate elements
    coordinates = soup.find_all('svg', attrs={'data-type': 'coordinates'})
    
    for coord in coordinates:
        # Get x, y coordinates
        x = int(float(coord.get('data-x', 0)))
        y = int(float(coord.get('data-y', 0)))
        
        # Find the result text (B, P, or T)
        text_elem = coord.find('text')
        if text_elem and text_elem.string:
            result = text_elem.string.strip()
            if 0 <= y < 6 and 0 <= x < 15:  # Ensure within grid bounds
                grid[y][x] = result.lower()
    
    return grid

def display_grid(grid):
    # CSS for the grid
    st.markdown("""
        <style>
        .grid-container {
            display: table;
            border-collapse: collapse;
            margin: 20px 0;
            overflow-x: auto;
        }
        .grid-row {
            display: table-row;
        }
        .bead-road-cell {
            width: 40px;
            height: 40px;
            border: 1px solid black;
            display: table-cell;
            text-align: center;
            vertical-align: middle;
            font-family: monospace;
            font-size: 16px;
            position: relative;
        }
        .cell-index {
            position: absolute;
            top: 1px;
            left: 1px;
            font-size: 8px;
            color: #666;
        }
        .banker { color: red; font-weight: bold; }
        .player { color: blue; font-weight: bold; }
        .tie { color: green; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)
    
    # Display the grid with container
    html_table = ['<div class="grid-container" style="max-width: 100%; overflow-x: auto;">']
    for y, row in enumerate(grid):  # Still iterate by rows for display
        html_table.append('<div class="grid-row">')
        for x, cell in enumerate(row[:15]):  # Show up to 15 columns
            css_class = ''
            if cell == 'b':
                css_class = 'banker'
            elif cell == 'p':
                css_class = 'player'
            elif cell == 't':
                css_class = 'tie'
            # Calculate cell index (1-based) using the new index system
            cell_index = y + 1 + (x * 6)
            # Add index to each cell
            html_table.append(
                f'<div class="bead-road-cell {css_class}" data-index="{cell_index}">'
                f'<span class="cell-index">{cell_index}</span>'
                f'{cell.upper() if cell else "&nbsp;"}'
                f'</div>'
            )
        html_table.append('</div>')
    html_table.append('</div>')
    
    st.markdown(''.join(html_table), unsafe_allow_html=True)
    
    # Add grid data to session state for potential further processing
    if 'grid_data' not in st.session_state:
        st.session_state.grid_data = grid

"""
Original table grid view and export code is commented out for future reference
# ... existing code ...
"""

def load_pattern_classifications():
    """Load pattern classifications from JSON file"""
    try:
        with open('pattern-classification-2025-04-20 (2).json', 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading pattern classifications: {str(e)}")
        return None

def sequence_to_string(sequence):
    """Convert sequence list to string"""
    return ''.join(sequence)

def get_table_index(x, y):
    """
    Convert grid coordinates to table index
    New index system:
    1  7  13 19 25 31 ...
    2  8  14 20 26 32 ...
    3  9  15 21 27 33 ...
    4  10 16 22 28 34 ...
    5  11 17 23 29 35 ...
    6  12 18 24 30 36 ...
    """
    return y + 1 + (x * 6)

def analyze_zones(grid):
    """
    Analyze the grid by overlapping zones
    Each zone contains 4 patterns of 3x2 blocks
    """
    # Load pattern classifications
    classifications = load_pattern_classifications()
    if not classifications:
        return [], []
    
    # Create pattern lookup dictionaries
    group_a_patterns = {sequence_to_string(p['sequence']): p['pattern_number'] 
                       for p in classifications['patterns']['groupA']}
    group_b_patterns = {sequence_to_string(p['sequence']): p['pattern_number'] 
                       for p in classifications['patterns']['groupB']}
    
    zones = []
    zone_stats = []
    
    # Define zone ranges with corrected pattern indices
    zone_ranges = [
        {"name": "1-18", "patterns": [
            # First pattern: [1,2,3,7,8,9]
            {"indices": [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]},
            # Second pattern: [4,5,6,10,11,12]
            {"indices": [(0,3), (0,4), (0,5), (1,3), (1,4), (1,5)]},
            # Third pattern: [7,8,9,13,14,15]
            {"indices": [(2,0), (2,1), (2,2), (3,0), (3,1), (3,2)]},
            # Fourth pattern: [10,11,12,16,17,18]
            {"indices": [(4,0), (4,1), (4,2), (5,0), (5,1), (5,2)]}
        ]}
    ]
    
    # Analyze patterns
    for zone in zone_ranges:
        found_patterns = []
        
        for pattern_def in zone["patterns"]:
            sequence = []
            indices = pattern_def["indices"]
            
            # Get values for this pattern using the coordinates directly
            for y, x in indices:
                if y < len(grid) and x < len(grid[0]):
                    cell = grid[y][x].upper() if grid[y][x] else ' '
                    sequence.append(cell)
                else:
                    sequence.append(' ')
            
            pattern_str = ''.join(sequence)
            
            if pattern_str.strip():  # Only analyze non-empty patterns
                # Convert coordinates to indices for display
                first_three = [y * 6 + x + 1 for y, x in indices[:3]]
                second_three = [y * 6 + x + 1 for y, x in indices[3:]]
                position_desc = f"({','.join(map(str, first_three))})+({','.join(map(str, second_three))})"
                
                pattern_info = {
                    'sequence': sequence,
                    'position': position_desc,
                    'group': None,
                    'classification_number': None
                }
                
                # Check if pattern matches any classification
                if pattern_str in group_a_patterns:
                    pattern_info['group'] = 'A'
                    pattern_info['classification_number'] = group_a_patterns[pattern_str]
                elif pattern_str in group_b_patterns:
                    pattern_info['group'] = 'B'
                    pattern_info['classification_number'] = group_b_patterns[pattern_str]
                
                found_patterns.append(pattern_info)
        
        # Calculate statistics
        total_patterns = len(found_patterns)
        group_a_count = sum(1 for p in found_patterns if p['group'] == 'A')
        group_b_count = sum(1 for p in found_patterns if p['group'] == 'B')
        unclassified = sum(1 for p in found_patterns if p['group'] is None)
        
        zone_stats.append({
            'zone_range': zone["name"],
            'total_patterns': total_patterns,
            'group_a_count': group_a_count,
            'group_b_count': group_b_count,
            'unclassified': unclassified,
            'patterns': found_patterns
        })
    
    return zones, zone_stats

def main():
    st.title("Bead Road Parser")
    
    # Initialize session state for text area
    if 'reset_clicked' not in st.session_state:
        st.session_state.reset_clicked = False
    
    # Text area for SVG input
    if st.session_state.reset_clicked:
        svg_code = st.text_area("Paste SVG code here", value="", height=200, key='svg_input')
        st.session_state.reset_clicked = False
    else:
        svg_code = st.text_area("Paste SVG code here", height=200, key='svg_input')
    
    # Create columns for buttons
    col1, col2 = st.columns([1, 4])
    
    # Buttons in columns
    with col1:
        if st.button("Reset"):
            st.session_state.reset_clicked = True
            st.experimental_rerun()
    
    with col2:
        if st.button("Parse SVG"):
            if svg_code:
                try:
                    grid = parse_bead_road_svg(svg_code)
                    st.success("Successfully parsed the SVG code!")
                    display_grid(grid)
                    
                    # Perform and display zone analysis
                    zones, zone_stats = analyze_zones(grid)
                    
                except Exception as e:
                    st.error(f"Error parsing SVG: {str(e)}")
            else:
                st.warning("Please paste SVG code first")

if __name__ == "__main__":
    main() 