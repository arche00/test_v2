import sqlite3
from collections import Counter, defaultdict

def analyze_sequences():
    conn = sqlite3.connect('pattern_analysis_v2.db')
    cursor = conn.cursor()
    
    # Get all prediction results
    cursor.execute("SELECT prediction_results FROM session_prediction_results")
    results = cursor.fetchall()
    
    # Analyze sequences
    loss_streaks = []
    win_streaks = []
    current_loss_streak = 0
    current_win_streak = 0
    sequence_patterns = defaultdict(int)
    
    for result in results:
        if result[0]:
            sequence = result[0]
            # Count streaks
            for outcome in sequence:
                if outcome == 'W':
                    if current_loss_streak > 0:
                        loss_streaks.append(current_loss_streak)
                        current_loss_streak = 0
                    current_win_streak += 1
                else:  # L
                    if current_win_streak > 0:
                        win_streaks.append(current_win_streak)
                        current_win_streak = 0
                    current_loss_streak += 1
            
            # Add final streaks
            if current_loss_streak > 0:
                loss_streaks.append(current_loss_streak)
                current_loss_streak = 0
            if current_win_streak > 0:
                win_streaks.append(current_win_streak)
                current_win_streak = 0
            
            # Analyze sequence patterns (3-game sequences)
            for i in range(len(sequence) - 2):
                pattern = sequence[i:i+3]
                sequence_patterns[pattern] += 1
    
    # Print analysis results
    print("Loss Streak Analysis:")
    loss_streak_counter = Counter(loss_streaks)
    for length, count in sorted(loss_streak_counter.items()):
        print(f"{length} consecutive losses: {count} times")
    
    print("\nWin Streak Analysis:")
    win_streak_counter = Counter(win_streaks)
    for length, count in sorted(win_streak_counter.items()):
        print(f"{length} consecutive wins: {count} times")
    
    print("\nMost Common 3-Game Sequences:")
    for pattern, count in sorted(sequence_patterns.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"Pattern {pattern}: {count} times")
    
    conn.close()

if __name__ == "__main__":
    analyze_sequences() 