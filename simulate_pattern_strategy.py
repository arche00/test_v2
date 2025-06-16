import sqlite3

class PatternBasedStrategy:
    def __init__(self):
        self.base_bet = 1
        self.current_bet = self.base_bet
        self.max_bet = 5
        self.total_wins = 0
        self.total_losses = 0
        self.balance = 0
        self.consecutive_losses = 0
        self.consecutive_wins = 0

    def get_bet_amount(self, pattern):
        if pattern == 'p->w':
            return min(3, self.max_bet)
        elif pattern == 'b->w':
            return min(2, self.max_bet)
        elif pattern == 'b->l':
            return 1
        else:  # p->l
            return 0

    def bet(self, result, pattern):
        if pattern:
            self.current_bet = self.get_bet_amount(pattern)
        
        if result == 'W':
            self.balance += self.current_bet
            self.total_wins += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            
            if self.consecutive_wins >= 3:
                self.current_bet = self.base_bet
                self.consecutive_wins = 0
        else:  # L
            self.balance -= self.current_bet
            self.total_losses += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            
            if self.consecutive_losses >= 2:
                self.current_bet = 0
                self.consecutive_losses = 0

    def report(self):
        return self.total_wins, self.total_losses, self.balance

def simulate_strategy():
    conn = sqlite3.connect('pattern_analysis_v2.db')
    cursor = conn.cursor()
    
    # Get prediction results and patterns
    cursor.execute("""
        SELECT pr.prediction_results, pa.pattern12_result, pa.pattern12_prediction_result
        FROM session_prediction_results pr
        LEFT JOIN pattern_analysis pa ON pr.session_id = pa.session_id
    """)
    results = cursor.fetchall()
    
    strategy = PatternBasedStrategy()
    
    for result in results:
        if result[0]:  # prediction_results
            predictions = result[0]
            pattern = f"{result[1]}->{result[2]}" if result[1] and result[2] else None
            
            for outcome in predictions:
                strategy.bet(outcome, pattern)
    
    wins, losses, balance = strategy.report()
    print(f"Total Wins: {wins}")
    print(f"Total Losses: {losses}")
    print(f"Final Balance: {balance}")
    print(f"Win Rate: {(wins/(wins+losses)*100):.2f}%")
    
    conn.close()

if __name__ == "__main__":
    simulate_strategy() 