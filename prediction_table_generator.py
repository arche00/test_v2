import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
import os
import json
import joblib
from collections import Counter
import gc
import itertools

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'pattern_analysis_v2.db')
MODEL_PATH = os.path.join(BASE_DIR, 'parser_v3_model.joblib')
PATTERN_JSON_PATH = os.path.join(BASE_DIR, 'pattern.json')

# Page configuration
st.set_page_config(
    page_title="Prediction Table Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set pandas option for large datasets
pd.set_option("styler.render.max_elements", 1572864)

def load_model_data():
    """Load the trained model data"""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def load_pattern_data():
    """Load pattern data from pattern.json"""
    try:
        with open(PATTERN_JSON_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading pattern data: {str(e)}")
        return None

def extract_prediction_features(pattern_number, hour=None):
    """Extract features for prediction"""
    if hour is None:
        hour = pd.Timestamp.now().hour
    
    digits = list(str(pattern_number))
    freq = Counter(digits)
    digit_features = [freq.get(str(i), 0) for i in range(10)]
    time_features = [hour]
    
    return np.array(digit_features + time_features).reshape(1, -1)

def generate_pattern1_combinations():
    """Generate all possible combinations of two pattern numbers from 01 to 64"""
    patterns = [str(i).zfill(2) for i in range(1, 65)]
    return [p1 + p2 for p1, p2 in itertools.product(patterns, patterns)]

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

def predict_single_pattern(pattern_type, pattern_number, hour=None):
    """Generate prediction for a single pattern number"""
    model_data = load_model_data()
    if model_data is None:
        st.warning("모델이 없습니다. 먼저 모델을 학습해주세요.")
        return None

    try:
        # Extract features and get predictions
        X_pred = extract_prediction_features(pattern_number, hour)
        if pattern_type == 'pattern1':
            if X_pred.shape[1] != model_data['n_features1']:
                return None
            proba = model_data['model1'].predict_proba(X_pred)[0]
            classes = model_data['model1'].classes_
        else:  # pattern2
            if X_pred.shape[1] != model_data['n_features2']:
                return None
            proba = model_data['model2'].predict_proba(X_pred)[0]
            classes = model_data['model2'].classes_

        # Get pattern information
        pattern_data = load_pattern_data()
        results = []
        
        # Get first 2 digits of pattern_number for pattern1 or 3rd,4th digits for pattern2
        if pattern_type == 'pattern1':
            search_prefix = str(pattern_number)[:2]
        else:
            pattern2_str = str(pattern_number)
            if len(pattern2_str) < 4:
                st.warning("pattern2_number가 너무 짧습니다. (최소 4자리 필요)")
                return None
            search_prefix = pattern2_str[2:4]

        # Search patterns with the prefix
        initial_patterns = search_patterns(pattern_data, search_prefix)
        if not initial_patterns:
            st.warning("초기 패턴을 찾을 수 없습니다.")
            return None

        # Get 4th, 5th, 6th items from sequences
        target_sequences = []
        for pattern in initial_patterns:
            sequence = pattern['sequence']
            if len(sequence) >= 6:
                target_seq = ''.join(sequence[3:6]).lower()
                target_sequences.append(target_seq)

        if not target_sequences:
            st.warning("시퀀스를 추출할 수 없습니다.")
            return None

        # Search patterns with target sequences
        related_patterns = []
        for seq in target_sequences:
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

        # Get unique pattern numbers and their group values
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
        for pred_number, prob in zip(classes, proba):
            pred_number_str = str(pred_number)
            if pred_number_str in related_patterns_dict:
                pred_info = related_patterns_dict[pred_number_str]
                filtered_predictions.append({
                    'Pattern Number': pattern_number,
                    'Prediction': pred_number,
                    'Probability': prob,
                    'Group': pred_info['group'],
                    'Group Value': pred_info['group_value'],
                    'Sequence': ' '.join(pred_info['sequence'])
                })

        # If no predictions match the filter, return top 8 original predictions
        if not filtered_predictions:
            st.warning("필터링된 예측이 없습니다. 상위 8개 예측을 반환합니다.")
            for pred_number, prob in zip(classes, proba):
                filtered_predictions.append({
                    'Pattern Number': pattern_number,
                    'Prediction': pred_number,
                    'Probability': prob,
                    'Group': 'N/A',
                    'Group Value': 'N/A',
                    'Sequence': 'N/A'
                })

        # Sort by probability and return top 8
        filtered_predictions.sort(key=lambda x: x['Probability'], reverse=True)
        return pd.DataFrame(filtered_predictions[:8])

    except Exception as e:
        st.error(f"예측 중 오류 발생: {str(e)}")
        return None

def process_pattern_chunk(chunk_patterns, pattern_type, model_data, pattern_data, hour):
    """Process a chunk of patterns and return filtered predictions"""
    chunk_results = []
    
    for pattern_number in chunk_patterns:
        # Extract features and get predictions
        X_pred = extract_prediction_features(pattern_number, hour)
        if pattern_type == 'pattern1':
            if X_pred.shape[1] != model_data['n_features1']:
                continue
            proba = model_data['model1'].predict_proba(X_pred)[0]
            classes = model_data['model1'].classes_
        else:  # pattern2
            if X_pred.shape[1] != model_data['n_features2']:
                continue
            proba = model_data['model2'].predict_proba(X_pred)[0]
            classes = model_data['model2'].classes_

        # Get first 2 digits of pattern_number for pattern1 or 3rd,4th digits for pattern2
        if pattern_type == 'pattern1':
            search_prefix = str(pattern_number)[:2]
        else:
            pattern2_str = str(pattern_number)
            if len(pattern2_str) < 4:
                continue
            search_prefix = pattern2_str[2:4]

        # Search patterns with the prefix
        initial_patterns = search_patterns(pattern_data, search_prefix)
        if not initial_patterns:
            continue

        # Get 4th, 5th, 6th items from sequences
        target_sequences = []
        for pattern in initial_patterns:
            sequence = pattern['sequence']
            if len(sequence) >= 6:
                target_seq = ''.join(sequence[3:6]).lower()
                target_sequences.append(target_seq)

        if not target_sequences:
            continue

        # Search patterns with target sequences
        related_patterns = []
        for seq in target_sequences:
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

        # Get unique pattern numbers and their group values
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
        for pred_number, prob in zip(classes, proba):
            pred_number_str = str(pred_number)
            if pred_number_str in related_patterns_dict:
                pred_info = related_patterns_dict[pred_number_str]
                filtered_predictions.append({
                    'Pattern Number': pattern_number,
                    'Prediction': pred_number,
                    'Probability': prob,
                    'Group': pred_info['group'],
                    'Group Value': pred_info['group_value'],
                    'Sequence': ' '.join(pred_info['sequence'])
                })

        # If no predictions match the filter, use top 8 original predictions
        if not filtered_predictions:
            for pred_number, prob in zip(classes, proba):
                filtered_predictions.append({
                    'Pattern Number': pattern_number,
                    'Prediction': pred_number,
                    'Probability': prob,
                    'Group': 'N/A',
                    'Group Value': 'N/A',
                    'Sequence': 'N/A'
                })

        # Sort by probability and take top 8
        filtered_predictions.sort(key=lambda x: x['Probability'], reverse=True)
        chunk_results.extend(filtered_predictions[:8])
    
    return chunk_results

def generate_prediction_table(pattern_type, start_number, end_number, hour=None, chunk_size=100):
    """Generate prediction table for a range of pattern numbers"""
    model_data = load_model_data()
    if model_data is None:
        st.warning("모델이 없습니다. 먼저 모델을 학습해주세요.")
        return None

    try:
        # Generate pattern numbers based on type
        if pattern_type == 'pattern1':
            pattern_numbers_list = generate_pattern1_combinations()
        else:
            pattern_numbers_list = [str(num).zfill(6) for num in range(start_number, end_number + 1)]

        # Load pattern data once
        pattern_data = load_pattern_data()
        if not pattern_data:
            st.error("패턴 데이터를 로드할 수 없습니다.")
            return None

        # Process in chunks with progress bar
        all_results = []
        total_patterns = len(pattern_numbers_list)
        progress_bar = st.progress(0)
        
        # Use multiprocessing for faster processing
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        
        # Calculate optimal chunk size based on CPU cores
        num_cores = multiprocessing.cpu_count()
        optimal_chunk_size = max(100, total_patterns // (num_cores * 4))
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = []
            for i in range(0, total_patterns, optimal_chunk_size):
                chunk = pattern_numbers_list[i:i + optimal_chunk_size]
                futures.append(executor.submit(
                    process_pattern_chunk,
                    chunk,
                    pattern_type,
                    model_data,
                    pattern_data,
                    hour
                ))
            
            # Collect results as they complete
            for i, future in enumerate(futures):
                chunk_results = future.result()
                all_results.extend(chunk_results)
                progress = min(1.0, (i + 1) * optimal_chunk_size / total_patterns)
                progress_bar.progress(progress)

        # Create DataFrame
        df = pd.DataFrame(all_results)
        
        # Sort by pattern number and probability
        df = df.sort_values(['Pattern Number', 'Probability'], ascending=[True, False])
        
        return df

    except Exception as e:
        st.error(f"예측 테이블 생성 중 오류 발생: {str(e)}")
        return None

def test_pattern2_combinations_performance():
    """Test function to check pattern2 combinations performance"""
    st.title("Pattern2 Combinations Performance Test")
    
    # Test parameters
    test_size = 4096 * 8  # Total combinations to test
    chunk_size = 1000     # Process in chunks of 1000
    
    # Initialize progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Track memory usage
    import psutil
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    try:
        # Generate test data
        test_patterns = [str(num).zfill(6) for num in range(test_size)]
        
        # Process in chunks
        results = []
        for i in range(0, test_size, chunk_size):
            chunk = test_patterns[i:i + chunk_size]
            
            # Update progress
            progress = min(1.0, (i + chunk_size) / test_size)
            progress_bar.progress(progress)
            status_text.text(f"Processing {i + len(chunk)}/{test_size} patterns...")
            
            # Simulate processing (replace with actual processing logic)
            chunk_results = [{"Pattern Number": p, "Status": "Processed"} for p in chunk]
            results.extend(chunk_results)
            
            # Force garbage collection
            gc.collect()
        
        # Calculate final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory
        
        # Display results
        st.success(f"Test completed successfully!")
        st.write(f"Total patterns processed: {len(results)}")
        st.write(f"Memory usage: {memory_used:.2f} MB")
        st.write(f"Average memory per pattern: {memory_used/len(results):.4f} MB")
        
        # Display sample results
        st.write("Sample results (first 5):")
        st.dataframe(pd.DataFrame(results[:5]))
        
    except Exception as e:
        st.error(f"Error during performance test: {str(e)}")
    finally:
        # Cleanup
        del results
        gc.collect()

def generate_block_combinations():
    """Generate combinations based on block dependencies and save to file"""
    st.title("Block Combinations Generator")
    
    # Initialize progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Generate Block 1 combinations (64 possibilities)
        block1_combinations = []
        for i in range(64):
            # Convert to 6-digit binary and replace 0/1 with p/b
            binary = format(i, '06b')
            # Convert to 3x2 format
            row1 = binary[0:2].replace('0', 'p').replace('1', 'b')
            row2 = binary[2:4].replace('0', 'p').replace('1', 'b')
            row3 = binary[4:6].replace('0', 'p').replace('1', 'b')
            # Create sequence in specified format
            sequence = f"{row1[0]}{row2[0]}{row3[0]}{row1[1]}{row2[1]}{row3[1]}"
            combination = f"{row1}|{row2}|{row3}"  # Keep original format for display
            block1_combinations.append((combination, sequence))
        
        # Generate Block 3 combinations (64 possibilities)
        block3_combinations = []
        for i in range(64):
            binary = format(i, '06b')
            row1 = binary[0:2].replace('0', 'p').replace('1', 'b')
            row2 = binary[2:4].replace('0', 'p').replace('1', 'b')
            row3 = binary[4:6].replace('0', 'p').replace('1', 'b')
            # Create sequence in specified format
            sequence = f"{row1[0]}{row2[0]}{row3[0]}{row1[1]}{row2[1]}{row3[1]}"
            combination = f"{row1}|{row2}|{row3}"  # Keep original format for display
            block3_combinations.append((combination, sequence))
        
        # Generate Block 2 combinations (8 possibilities)
        block2_combinations = []
        for i in range(8):
            binary = format(i, '03b')
            # For Block 2, we only need 3 new cells
            new_cells = binary.replace('0', 'p').replace('1', 'b')
            # Create complete 3x2 structure with dependencies
            row1 = f"x{new_cells[0]}"  # First column depends on Block1
            row2 = f"x{new_cells[1]}"  # First column depends on Block1
            row3 = f"x{new_cells[2]}"  # First column depends on Block1
            combination = f"{row1}|{row2}|{row3}"
            block2_combinations.append(combination)
        
        # Calculate total combinations
        total_combinations = len(block1_combinations) * len(block3_combinations) * len(block2_combinations)
        
        # Generate all combinations
        all_combinations = []
        count = 0
        
        for b1, b1_seq in block1_combinations:
            # Get Block1's second column values
            b1_rows = b1.split('|')
            b1_col2 = [row[1] for row in b1_rows]  # Get second column values from Block1
            
            for b3, b3_seq in block3_combinations:
                for b2 in block2_combinations:
                    # Replace 'x' in Block2 with actual values from Block1
                    b2_rows = b2.split('|')
                    b2_complete = []
                    for i, row in enumerate(b2_rows):
                        # Replace 'x' with corresponding value from Block1
                        complete_row = f"{b1_col2[i]}{row[1]}"
                        b2_complete.append(complete_row)
                    b2_complete = '|'.join(b2_complete)
                    
                    # Create Block2 sequence
                    b2_seq = f"{b1_col2[0]}{b1_col2[1]}{b1_col2[2]}{b2_rows[0][1]}{b2_rows[1][1]}{b2_rows[2][1]}"
                    
                    # Create full combination
                    combination = {
                        'Block1': b1,
                        'Block3': b3,
                        'Block2': b2_complete,
                        'Block1_Sequence': b1_seq,
                        'Block2_Sequence': b2_seq,
                        'Block3_Sequence': b3_seq,
                        'Combination_ID': f"{count:06d}"
                    }
                    all_combinations.append(combination)
                    
                    # Update progress
                    count += 1
                    progress = count / total_combinations
                    progress_bar.progress(progress)
                    status_text.text(f"Generating combinations: {count}/{total_combinations}")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_combinations)
        
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'block_combinations_{timestamp}.csv'
        df.to_csv(filename, index=False)
        
        # Display results
        st.success(f"Combinations generated and saved to {filename}")
        st.write(f"Total combinations: {len(df)}")
        st.write("Sample combinations (first 5):")
        st.dataframe(df.head())
        
        # Display example of Block structure
        st.write("### Block Structure Example")
        st.write("Block 1 format (3x2):")
        example_block1 = block1_combinations[0][0]
        rows1 = example_block1.split('|')
        st.write(f"Row 1: {rows1[0]}")
        st.write(f"Row 2: {rows1[1]}")
        st.write(f"Row 3: {rows1[2]}")
        st.write(f"Sequence: {block1_combinations[0][1]}")
        
        st.write("\nBlock 2 format (3x2) with actual values:")
        example_block2 = all_combinations[0]['Block2']
        rows2 = example_block2.split('|')
        st.write(f"Row 1: {rows2[0]} (First column from Block1)")
        st.write(f"Row 2: {rows2[1]} (First column from Block1)")
        st.write(f"Row 3: {rows2[2]} (First column from Block1)")
        st.write(f"Sequence: {all_combinations[0]['Block2_Sequence']}")
        
        # Display statistics
        st.write("### Combination Statistics")
        st.write(f"Block 1 combinations: {len(block1_combinations)}")
        st.write(f"Block 3 combinations: {len(block3_combinations)}")
        st.write(f"Block 2 combinations: {len(block2_combinations)}")
        st.write(f"Total unique combinations: {total_combinations}")
        
    except Exception as e:
        st.error(f"Error generating combinations: {str(e)}")
    finally:
        # Cleanup
        del all_combinations
        gc.collect()

def process_combinations_with_patterns():
    """Process combinations CSV and create new table with pattern numbers"""
    st.title("Process Combinations with Pattern Numbers")
    
    try:
        # Load pattern data
        pattern_data = load_pattern_data()
        if not pattern_data:
            st.error("Failed to load pattern data")
            return
        
        # Create pattern sequence to number mapping
        pattern_map = {}
        for group in ['groupA', 'groupB']:
            for pattern in pattern_data['patterns'][group]:
                # Convert sequence list to string
                sequence = ''.join(pattern['sequence'])
                pattern_map[sequence] = pattern['pattern_number']
        
        # Read combinations CSV
        try:
            # Use BASE_DIR for file path
            input_file = os.path.join(BASE_DIR, 'block_combinations_20250516_113059.csv')
            if not os.path.exists(input_file):
                st.error(f"Combinations CSV file not found at: {input_file}")
                return
            df = pd.read_csv(input_file)
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            return
        
        # Create new DataFrame with pattern numbers
        new_data = []
        
        # Process each row
        for _, row in df.iterrows():
            # Get sequences
            block1_seq = row['Block1_Sequence']
            block2_seq = row['Block2_Sequence']
            block3_seq = row['Block3_Sequence']
            
            # Find pattern numbers
            block1_num = pattern_map.get(block1_seq, 'N/A')
            block2_num = pattern_map.get(block2_seq, 'N/A')
            block3_num = pattern_map.get(block3_seq, 'N/A')
            
            new_data.append({
                'Block1_Sequence': block1_seq,
                'Block2_Sequence': block2_seq,
                'Block3_Sequence': block3_seq,
                'Block1_Number': block1_num,
                'Block2_Number': block2_num,
                'Block3_Number': block3_num
            })
        
        # Create new DataFrame
        new_df = pd.DataFrame(new_data)
        
        # Save to new CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f'pattern_combinations_{timestamp}.csv'
        new_df.to_csv(output_filename, index=False)
        
        # Display results
        st.success(f"New table saved to {output_filename}")
        st.write("Sample data (first 5 rows):")
        st.dataframe(new_df.head())
        
        # Display statistics
        st.write("### Statistics")
        st.write(f"Total combinations processed: {len(new_df)}")
        st.write(f"Patterns found for Block1: {new_df['Block1_Number'].ne('N/A').sum()}")
        st.write(f"Patterns found for Block2: {new_df['Block2_Number'].ne('N/A').sum()}")
        st.write(f"Patterns found for Block3: {new_df['Block3_Number'].ne('N/A').sum()}")
        
    except Exception as e:
        st.error(f"Error processing combinations: {str(e)}")

def generate_pattern2_combinations():
    """Generate combinations from pattern2.csv file"""
    try:
        # Read pattern2.csv file
        pattern2_file = os.path.join(BASE_DIR, 'pattern2.csv')
        if not os.path.exists(pattern2_file):
            st.error(f"pattern2.csv file not found at: {pattern2_file}")
            return None
            
        df = pd.read_csv(pattern2_file)
        if 'Pattern Number' not in df.columns:
            st.error("Pattern Number column not found in pattern2.csv")
            return None
            
        # Extract pattern numbers and ensure they are 6 digits
        pattern_numbers = df['Pattern Number'].astype(str).str.zfill(6).tolist()
        return pattern_numbers
        
    except Exception as e:
        st.error(f"Error reading pattern2.csv: {str(e)}")
        return None

def generate_prediction_table_pattern2(hour=None, chunk_size=100):
    """Generate prediction table for pattern2 combinations"""
    st.title("Pattern2 Prediction Table Generator")
    
    # Initialize progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Load model data
        model_data = load_model_data()
        if model_data is None:
            st.warning("모델이 없습니다. 먼저 모델을 학습해주세요.")
            return None
            
        # Load pattern data
        pattern_data = load_pattern_data()
        if not pattern_data:
            st.error("패턴 데이터를 로드할 수 없습니다.")
            return None
            
        # Get pattern2 combinations
        pattern_numbers_list = generate_pattern2_combinations()
        if not pattern_numbers_list:
            return None
            
        # Calculate total combinations
        total_patterns = len(pattern_numbers_list)
        
        # Process in chunks
        all_results = []
        count = 0
        
        # Use multiprocessing for faster processing
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        
        # Calculate optimal chunk size based on CPU cores
        num_cores = multiprocessing.cpu_count()
        optimal_chunk_size = max(100, total_patterns // (num_cores * 4))
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = []
            for i in range(0, total_patterns, optimal_chunk_size):
                chunk = pattern_numbers_list[i:i + optimal_chunk_size]
                futures.append(executor.submit(
                    process_pattern_chunk,
                    chunk,
                    'pattern2',
                    model_data,
                    pattern_data,
                    hour
                ))
            
            # Collect results as they complete
            for i, future in enumerate(futures):
                chunk_results = future.result()
                all_results.extend(chunk_results)
                progress = min(1.0, (i + 1) * optimal_chunk_size / total_patterns)
                progress_bar.progress(progress)
                status_text.text(f"Processing patterns: {count + len(chunk_results)}/{total_patterns}")
                count += len(chunk_results)
        
        # Create DataFrame
        df = pd.DataFrame(all_results)
        
        # Sort by pattern number and probability
        df = df.sort_values(['Pattern Number', 'Probability'], ascending=[True, False])
        
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'pattern2_predictions_{timestamp}.csv'
        df.to_csv(filename, index=False)
        
        # Display results
        st.success(f"Predictions saved to {filename}")
        st.write(f"Total predictions: {len(df)}")
        st.write("Sample predictions (first 5):")
        st.dataframe(df.head())
        
        # Display statistics
        st.write("### Statistics")
        st.write(f"Total patterns processed: {len(pattern_numbers_list)}")
        st.write(f"Total predictions generated: {len(df)}")
        st.write(f"Average predictions per pattern: {len(df)/len(pattern_numbers_list):.2f}")
        
        return df
        
    except Exception as e:
        st.error(f"Error generating prediction table: {str(e)}")
        return None
    finally:
        # Cleanup
        del all_results
        gc.collect()

def main():
    st.title("Prediction Table Generator")
    
    # Sidebar controls
    st.sidebar.header("Settings")
    
    prediction_mode = st.sidebar.radio(
        "Prediction Mode",
        ['Single Pattern', 'Pattern Combinations'],
        help="Choose between single pattern prediction or pattern combinations"
    )
    
    pattern_type = st.sidebar.radio(
        "Select Pattern Type",
        ['pattern1', 'pattern2'],
        help="Choose between Pattern1→Result1 or Pattern2→Result2 prediction"
    )
    
    hour = st.sidebar.slider(
        "Hour of Day",
        min_value=0,
        max_value=23,
        value=datetime.now().hour,
        help="Select hour for prediction (0-23)"
    )
    
    if prediction_mode == 'Single Pattern':
        st.sidebar.header("Single Pattern Prediction")
        pattern_number = st.sidebar.text_input(
            "Enter Pattern Number",
            help="For Pattern1: Enter 4 digits (e.g., '0102')\nFor Pattern2: Enter 6 digits"
        )
        
        if st.sidebar.button("Predict"):
            if pattern_number:
                with st.spinner("Generating prediction..."):
                    df = predict_single_pattern(pattern_type, pattern_number, hour)
                    if df is not None:
                        st.write("### Prediction Results")
                        st.dataframe(
                            df.style.format({
                                'Probability': '{:.2%}'
                            }),
                            use_container_width=True
                        )
                        
                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "Download CSV",
                            csv,
                            f"single_prediction_{pattern_type}_{pattern_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv",
                            key='download-csv'
                        )
            else:
                st.warning("Please enter a pattern number")
    
    else:  # Pattern Combinations
        st.sidebar.header("Pattern Combinations")
        if pattern_type == 'pattern1':
            st.sidebar.info("Pattern1: 01-64 범위의 패턴 번호 2개를 조합한 모든 결과를 생성합니다.")
            st.sidebar.warning("주의: 64x64=4096개의 조합이 생성되며, 각 조합당 8개의 예측 결과가 생성됩니다.")
        else:
            col1, col2 = st.sidebar.columns(2)
            with col1:
                start_number = st.number_input(
                    "Start Number",
                    min_value=0,
                    max_value=999999,
                    value=0,
                    step=1000
                )
            with col2:
                end_number = st.number_input(
                    "End Number",
                    min_value=0,
                    max_value=999999,
                    value=9999,
                    step=1000
                )
        
        if st.sidebar.button("Generate Table"):
            with st.spinner("Generating prediction table..."):
                df = generate_prediction_table(
                    pattern_type,
                    start_number if pattern_type == 'pattern2' else 0,
                    end_number if pattern_type == 'pattern2' else 0,
                    hour
                )
                
                if df is not None:
                    # Display statistics
                    st.write(f"Total predictions: {len(df)}")
                    st.write(f"Unique pattern numbers: {df['Pattern Number'].nunique()}")
                    
                    # Display table with pagination
                    st.dataframe(
                        df.style.format({
                            'Probability': '{:.2%}'
                        }),
                        use_container_width=True,
                        height=400
                    )
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download CSV",
                        csv,
                        f"prediction_table_{pattern_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        key='download-csv'
                    )

    # Add test function to main menu
    st.sidebar.header("Performance Test")
    if st.sidebar.button("Run Pattern2 Performance Test"):
        test_pattern2_combinations_performance()

    # Add block combinations generator to main menu
    st.sidebar.header("Block Combinations")
    if st.sidebar.button("Generate Block Combinations"):
        generate_block_combinations()

    # Add pattern processing to main menu
    st.sidebar.header("Pattern Processing")
    if st.sidebar.button("Process Combinations with Patterns"):
        process_combinations_with_patterns()

    # Add pattern2 processing to main menu
    st.sidebar.header("Pattern2 Processing")
    if st.sidebar.button("Generate Pattern2 Predictions"):
        generate_prediction_table_pattern2()

if __name__ == "__main__":
    main() 