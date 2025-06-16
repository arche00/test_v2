import sqlite3

class RecoveryStrategy:
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
        self.recovery_mode = False
        self.total_loss = 0
        self.recovery_target = 0
        self.daily_loss_limit = 30  # 일일 손실 한도 추가 감소
        self.last_three_results = []
        self.win_streak_threshold = 3  # 연속 승리 기준

    def round_bet(self, value):
        # 0.5 단위로 반올림
        return max(self.base_bet, round(value * 2) / 2)

    def calculate_recovery_bet(self):
        # 손실의 1/4만 회복 시도
        if self.total_loss <= 0:
            return self.base_bet
        recovery_bet = min(self.total_loss / 4, self.max_bet)
        return self.round_bet(recovery_bet)

    def should_bet(self):
        if len(self.last_three_results) < 3:
            return True
        
        # 연속 3패배 후에는 베팅하지 않음
        if self.last_three_results[-3:] == ['L', 'L', 'L']:
            return False
        
        # 연속 2패배 후에는 베팅하지 않음
        if len(self.last_three_results) >= 2 and self.last_three_results[-2:] == ['L', 'L']:
            return False
        
        # 연속 2승 후에는 베팅 단위 증가
        if self.last_three_results[-2:] == ['W', 'W']:
            return True
        
        return True

    def bet(self, result):
        # 최근 결과 업데이트
        self.last_three_results.append(result)
        if len(self.last_three_results) > 3:
            self.last_three_results.pop(0)

        if not self.is_betting:
            if result == 'W' and self.should_bet():
                self.is_betting = True
                self.recovery_mode = True
                self.current_bet = self.calculate_recovery_bet()
            return

        if not self.should_bet():
            self.is_betting = False
            return

        if result == 'W':
            self.balance += self.current_bet
            self.total_wins += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            
            if self.recovery_mode:
                self.total_loss -= self.current_bet
                if self.total_loss <= 0:
                    self.recovery_mode = False
                    self.current_bet = self.base_bet
                else:
                    self.current_bet = self.calculate_recovery_bet()
            else:
                # 연속 승리 시 베팅 단위 증가 (1.1배로 감소)
                if self.consecutive_wins >= 2:
                    self.current_bet = self.round_bet(min(self.current_bet * 1.1, self.max_bet))
                else:
                    self.current_bet = self.base_bet
            
            if self.consecutive_wins >= self.win_streak_threshold:
                self.consecutive_wins = 0
        else:  # L
            self.balance -= self.current_bet
            self.total_losses += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            
            if self.recovery_mode:
                self.total_loss += self.current_bet
                # 회복 모드에서도 연속 패배 시 즉시 중단
                if self.consecutive_losses >= 1:  # 1회 패배 시 즉시 중단
                    self.is_betting = False
                    self.consecutive_losses = 0
                else:
                    self.current_bet = self.calculate_recovery_bet()
            else:
                self.total_loss += self.current_bet
                if self.consecutive_losses >= 1:  # 1회 패배 시 즉시 중단
                    self.is_betting = False
                    self.consecutive_losses = 0
                else:
                    self.current_bet = self.round_bet(min(self.current_bet * 1.2, self.max_bet))

            # 일일 손실 한도 체크
            if abs(self.balance) >= self.daily_loss_limit:
                self.is_betting = False

    def report(self):
        return self.total_wins, self.total_losses, self.balance, self.total_loss

def simulate_strategy():
    conn = sqlite3.connect('pattern_analysis_v2.db')
    cursor = conn.cursor()
    
    # Get all prediction results
    cursor.execute("SELECT prediction_results FROM session_prediction_results")
    results = cursor.fetchall()
    
    strategy = RecoveryStrategy()
    
    for result in results:
        if result[0]:
            for outcome in result[0]:
                strategy.bet(outcome)
    
    wins, losses, balance, total_loss = strategy.report()
    print(f"\n=== 전체 결과 ===")
    print(f"총 승리: {wins}")
    print(f"총 패배: {losses}")
    print(f"최종 잔액: {balance:.2f}")
    print(f"남은 회복 손실: {total_loss:.2f}")
    print(f"승률: {(wins/(wins+losses)*100):.2f}%")
    
    conn.close()

if __name__ == "__main__":
    simulate_strategy() 