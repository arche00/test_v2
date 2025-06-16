import sqlite3
import json
from collections import Counter

def analyze_prediction_results():
    conn = sqlite3.connect('pattern_analysis_v2.db')
    cursor = conn.cursor()
    
    # Get prediction results
    cursor.execute("SELECT prediction_results FROM session_prediction_results")
    results = cursor.fetchall()
    
    # Analyze results
    total_games = 0
    wins = 0
    losses = 0
    win_streaks = []
    loss_streaks = []
    current_win_streak = 0
    current_loss_streak = 0
    
    for result in results:
        if result[0]:  # Check if result is not None
            # Handle string format like "LLLWLLLWWWLWW"
            predictions = result[0]
            for pred in predictions:
                total_games += 1
                if pred == 'W':
                    wins += 1
                    current_win_streak += 1
                    if current_loss_streak > 0:
                        loss_streaks.append(current_loss_streak)
                        current_loss_streak = 0
                elif pred == 'L':
                    losses += 1
                    current_loss_streak += 1
                    if current_win_streak > 0:
                        win_streaks.append(current_win_streak)
                        current_win_streak = 0
    
    # Add final streaks
    if current_win_streak > 0:
        win_streaks.append(current_win_streak)
    if current_loss_streak > 0:
        loss_streaks.append(current_loss_streak)
    
    # Calculate statistics
    win_rate = (wins / total_games * 100) if total_games > 0 else 0
    avg_win_streak = sum(win_streaks) / len(win_streaks) if win_streaks else 0
    max_win_streak = max(win_streaks) if win_streaks else 0
    avg_loss_streak = sum(loss_streaks) / len(loss_streaks) if loss_streaks else 0
    max_loss_streak = max(loss_streaks) if loss_streaks else 0
    
    print(f"Total Games: {total_games}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"\nWin Streaks:")
    print(f"Average Win Streak: {avg_win_streak:.2f}")
    print(f"Maximum Win Streak: {max_win_streak}")
    print(f"Win Streak Distribution: {sorted(Counter(win_streaks).items())}")
    print(f"\nLoss Streaks:")
    print(f"Average Loss Streak: {avg_loss_streak:.2f}")
    print(f"Maximum Loss Streak: {max_loss_streak}")
    print(f"Loss Streak Distribution: {sorted(Counter(loss_streaks).items())}")
    
    # Analyze consecutive patterns
    cursor.execute("""
        SELECT pattern12_result, pattern12_prediction_result 
        FROM pattern_analysis 
        WHERE pattern12_result IS NOT NULL
    """)
    pattern_results = cursor.fetchall()
    
    pattern_accuracy = Counter()
    for result, prediction in pattern_results:
        if result and prediction:
            pattern_accuracy[(result, prediction)] += 1
    
    print("\nPattern Analysis:")
    for (result, prediction), count in pattern_accuracy.most_common():
        print(f"Pattern: {result} -> {prediction}: {count} times")
    
    conn.close()

if __name__ == "__main__":
    analyze_prediction_results() 