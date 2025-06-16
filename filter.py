import pandas as pd

def processAndWriteCSVData(df):
    """
    Process CSV data to find maximum probability sequences for 'p' and 'b' patterns.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing the data
        
    Returns:
        pd.DataFrame: Processed results with Pattern Number, P Sequence, P Probability, B Sequence, B Probability
    """
    # Get unique pattern numbers
    pattern_numbers = df['Pattern Number'].unique()
    
    # Initialize results list
    results = []
    
    # Process each pattern number
    for pattern in pattern_numbers:
        # Filter data for current pattern
        filtered_df = df[df['Pattern Number'] == pattern]
        
        # Find max probability for 'p' sequence
        p_data = filtered_df[filtered_df['Sequence'].str.split().str[3] == 'p']
        p_max_prob_row = p_data.loc[p_data['Probability'].idxmax()] if not p_data.empty else None
        
        # Find max probability for 'b' sequence
        b_data = filtered_df[filtered_df['Sequence'].str.split().str[3] == 'b']
        b_max_prob_row = b_data.loc[b_data['Probability'].idxmax()] if not b_data.empty else None
        
        # Add results to list
        results.append([
            pattern,
            p_max_prob_row['Sequence'] if p_max_prob_row is not None else "",
            p_max_prob_row['Probability'] if p_max_prob_row is not None else "",
            b_max_prob_row['Sequence'] if b_max_prob_row is not None else "",
            b_max_prob_row['Probability'] if b_max_prob_row is not None else ""
        ])
    
    # Create DataFrame from results
    result_df = pd.DataFrame(results, columns=[
        'Pattern Number', 'P Sequence', 'P Probability', 
        'B Sequence', 'B Probability'
    ])
    
    return result_df

# CSV 파일 읽기
df = pd.read_csv('/Users/tj/test_v3/pattern2_predictions_20250516_120943.csv')

# 모든 패턴에 대해 처리
result_df = processAndWriteCSVData(df)

# 결과 출력
print("\n처리된 결과:")
print(result_df)

# 결과를 CSV 파일로 저장
output_filename = 'processed_results.csv'
result_df.to_csv(output_filename, index=False)
print(f"\n결과가 {output_filename}에 저장되었습니다.")