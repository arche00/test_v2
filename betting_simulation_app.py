import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to connect to the database and fetch session prediction results
def fetch_session_prediction_results():
    db_path = '/Users/tj/test_v3/pattern_analysis_v2.db'
    conn = sqlite3.connect(db_path)
    query = 'SELECT session_id, prediction_results FROM session_prediction_results'
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Function to simulate betting strategy
def simulate_betting_strategy(df):
    # Example strategy: Bet on 'W' results
    df['bet_result'] = df['prediction_results'].apply(lambda x: 'W' in x)
    return df

# Function to visualize results
def visualize_results(df):
    # Calculate and display statistics
    total_bets = len(df)
    win_rate = (df['bet_result'].sum() / total_bets) * 100
    st.write(f'Total Bets: {total_bets}')
    st.write(f'Win Rate: {win_rate:.2f}%')
    
    # Display a portion of the session_prediction_results table
    st.write('Sample of Session Prediction Results:')
    st.dataframe(df.head())
    
    # Sequence Analysis
    st.write('Sequence Analysis:')
    sequence_counts = df['prediction_results'].value_counts().head(10)
    st.write('Top 10 Prediction Result Sequences:')
    st.dataframe(sequence_counts)
    
    # Statistical Indicators
    st.write('Statistical Indicators:')
    mean_result = df['bet_result'].mean()
    variance_result = df['bet_result'].var()
    st.write(f'Mean Result: {mean_result:.2f}')
    st.write(f'Variance Result: {variance_result:.2f}')
    
    # Visualizations
    st.write('Visualizations:')
    
    # Pie Chart for Win/Loss Ratio
    fig, ax = plt.subplots()
    df['bet_result'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
    ax.set_title('Win/Loss Ratio')
    st.pyplot(fig)
    
    # Bar Graph for Win/Loss Count by Session
    fig, ax = plt.subplots()
    df.groupby('session_id')['bet_result'].sum().plot(kind='bar', ax=ax)
    ax.set_title('Win/Loss Count by Session')
    st.pyplot(fig)
    
    # Line Chart for Win Rate Trend by Session
    fig, ax = plt.subplots()
    df.groupby('session_id')['bet_result'].mean().plot(kind='line', ax=ax)
    ax.set_title('Win Rate Trend by Session')
    st.pyplot(fig)
    
    # Detailed Statistics
    st.write('Detailed Statistics:')
    st.write('Consecutive Wins/Losses:')
    consecutive_wins = df['bet_result'].astype(int).groupby((df['bet_result'].astype(int).diff() != 0).cumsum()).cumsum().max()
    consecutive_losses = df['bet_result'].astype(int).groupby((df['bet_result'].astype(int).diff() != 0).cumsum()).cumsum().min()
    st.write(f'Max Consecutive Wins: {consecutive_wins}')
    st.write(f'Max Consecutive Losses: {consecutive_losses}')
    
    # Conditional Win Rate
    st.write('Conditional Win Rate:')
    win_after_win = df[df['bet_result'].shift(1) == True]['bet_result'].mean() * 100
    win_after_loss = df[df['bet_result'].shift(1) == False]['bet_result'].mean() * 100
    st.write(f'Win Rate After Win: {win_after_win:.2f}%')
    st.write(f'Win Rate After Loss: {win_after_loss:.2f}%')
    
    # Session-wise Detailed Statistics
    st.write('Session-wise Detailed Statistics:')
    session_stats = df.groupby('session_id').agg({
        'bet_result': ['count', 'sum', 'mean']
    }).rename(columns={'count': 'Total Bets', 'sum': 'Wins', 'mean': 'Win Rate'})
    st.dataframe(session_stats)
    
    # Data-driven Insights
    st.write('Data-driven Insights:')
    if win_rate > 50:
        st.write('The overall win rate is above 50%, indicating a potentially profitable strategy.')
    else:
        st.write('The overall win rate is below 50%, suggesting a need for strategy adjustment.')
    
    # Session Sequence Analysis
    st.write('Session Sequence Analysis:')
    df['sequence_length'] = df['prediction_results'].str.len()
    sequence_length_stats = df.groupby('session_id')['sequence_length'].agg(['mean', 'min', 'max'])
    st.write('Sequence Length Statistics by Session:')
    st.dataframe(sequence_length_stats)
    
    # Pattern Analysis
    st.write('Pattern Analysis:')
    df['pattern'] = df['prediction_results'].apply(lambda x: ''.join(['W' if 'W' in x else 'L' for x in x.split()]))
    pattern_counts = df['pattern'].value_counts().head(10)
    st.write('Top 10 Patterns:')
    st.dataframe(pattern_counts)

# Main function to run the Streamlit app
def main():
    st.title('Betting Strategy Simulation')
    df = fetch_session_prediction_results()
    df = simulate_betting_strategy(df)
    visualize_results(df)

if __name__ == '__main__':
    main() 