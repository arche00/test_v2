import sqlite3

class SequenceBasedStrategy:
    def __init__(self):
        self.base_bet = 1
        self.current_bet = self.base_bet
        self.max_bet = 8
        self.total_wins = 0
        self.total_losses = 0
        self.balance = 0
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.is_betting = True

    def bet(self, result):
        if not self.is_betting:
            if result == 'W':
                self.is_betting = True
                self.current_bet = self.base_bet
            return

        if result == 'W':
            self.balance += self.current_bet
            self.total_wins += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.current_bet = self.base_bet
            
            if self.consecutive_wins >= 2:
                self.current_bet = self.base_bet
                self.consecutive_wins = 0
        else:  # L
            self.balance -= self.current_bet
            self.total_losses += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            
            if self.consecutive_losses >= 3:
                self.is_betting = False
                self.consecutive_losses = 0
            else:
                self.current_bet = min(self.current_bet * 2, self.max_bet)

    def report(self):
        return self.total_wins, self.total_losses, self.balance

def simulate_strategy():
    conn = sqlite3.connect('pattern_analysis_v2.db')
    cursor = conn.cursor()
    
    # Get all prediction results
    cursor.execute("SELECT prediction_results FROM session_prediction_results")
    results = cursor.fetchall()
    
    strategy = SequenceBasedStrategy()
    
    for result in results:
        if result[0]:
            for outcome in result[0]:
                strategy.bet(outcome)
    
    wins, losses, balance = strategy.report()
    print(f"Total Wins: {wins}")
    print(f"Total Losses: {losses}")
    print(f"Final Balance: {balance}")
    print(f"Win Rate: {(wins/(wins+losses)*100):.2f}%")
    
    conn.close()

if __name__ == "__main__":
    simulate_strategy() 