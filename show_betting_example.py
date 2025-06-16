import sqlite3

# -------------------------------------------------------------
# Betting Amount Logic (베팅 금액 로직)
#
# - base_bet: 최소/초기 베팅 단위 (기본값 1)
# - bet_unit: 베팅 증감 단위 (기본값 0.5)
#
# [Normal Mode]
#   - 첫 베팅 및 패배 시: current_bet = base_bet (항상 초기 금액으로 리셋)
#   - 연속 승리 2회 이상: current_bet += bet_unit (최대 max_bet까지 증가)
#   - 승리 후 연속승 카운트가 win_streak_threshold(3) 이상이면 카운트 리셋
#
# [Recovery Mode]
#   - 손실(total_loss)의 1/4을 0.5 단위로 반올림하여 current_bet으로 설정
#   - 승리 시: total_loss에서 current_bet만큼 차감, 손실이 0 이하가 되면 Normal Mode로 복귀
#   - 패배 시: total_loss에 current_bet만큼 추가, current_bet 재계산
#
# [공통]
#   - 베팅 기록은 betting_history에 저장 (bet_amount, result, mode 등)
#   - 연속 패배/승리 카운트는 각 결과에 따라 누적/리셋
# -------------------------------------------------------------

class RecoveryStrategy:
    def __init__(self):
        self.base_bet = 1
        self.current_bet = self.base_bet
        self.max_bet = 8
        self.min_bet = 1
        self.bet_unit = 0.5  # 베팅 증감 단위
        self.total_wins = 0
        self.total_losses = 0
        self.balance = 0
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.is_betting = True
        self.recovery_mode = False
        self.total_loss = 0
        self.recovery_target = 0
        self.daily_loss_limit = 30
        self.last_three_results = []
        self.win_streak_threshold = 3
        self.betting_history = []  # 베팅 기록 저장

    def calculate_recovery_bet(self):
        if self.total_loss <= 0:
            return self.base_bet
        recovery_bet = min(self.total_loss / 4, self.max_bet)
        # 0.5 단위로 반올림
        recovery_bet = round(recovery_bet * 2) / 2
        final_bet = max(self.min_bet, min(recovery_bet, self.max_bet))
        print(f"\nRecovery 베팅 금액 계산:")
        print(f"총 손실: {self.total_loss}")
        print(f"손실의 1/4: {self.total_loss/4}")
        print(f"최대베팅금액: {self.max_bet}")
        print(f"계산된 베팅금액: {recovery_bet}")
        print(f"최종 베팅금액: {final_bet}")
        return final_bet

    def adjust_bet(self, increase=True):
        if increase:
            self.current_bet = min(self.current_bet + self.bet_unit, self.max_bet)
        else:
            self.current_bet = max(self.current_bet - self.bet_unit, self.min_bet)
        # 0.5 단위로 반올림
        self.current_bet = round(self.current_bet * 2) / 2

    def should_bet(self):
        if len(self.last_three_results) < 3:
            return True
        if self.last_three_results[-3:] == ['L', 'L', 'L']:
            return False
        if len(self.last_three_results) >= 2 and self.last_three_results[-2:] == ['L', 'L']:
            return False
        if self.last_three_results[-2:] == ['W', 'W']:
            return True
        return True

    def bet(self, result):
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

        bet_record = {
            'bet_amount': self.current_bet,
            'result': result,
            'mode': 'Recovery' if self.recovery_mode else 'Normal',
            'balance_before': self.balance,
            'total_loss': self.total_loss
        }

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
                if self.consecutive_wins >= 2:
                    self.adjust_bet(increase=True)  # 0.5 단위로 증가
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
                if self.consecutive_losses >= 1:
                    self.is_betting = False
                    self.consecutive_losses = 0
                else:
                    self.current_bet = self.calculate_recovery_bet()
            else:
                self.total_loss += self.current_bet
                if self.consecutive_losses >= 1:
                    self.is_betting = False
                    self.consecutive_losses = 0
                else:
                    self.adjust_bet(increase=True)  # 0.5 단위로 증가

            if abs(self.balance) >= self.daily_loss_limit:
                self.is_betting = False

        bet_record['balance_after'] = self.balance
        self.betting_history.append(bet_record)

    def report(self):
        return self.total_wins, self.total_losses, self.balance, self.total_loss

    def show_recovery_bets(self, num_examples=10):
        print("\n=== Recovery 모드 베팅 예시 (최대 10건) ===")
        count = 0
        for i, record in enumerate(self.betting_history):
            if record['mode'] == 'Recovery':
                print(f"\n베팅 #{i+1} (Recovery)")
                print(f"베팅 금액: {record['bet_amount']}")
                print(f"총 손실(total_loss) 직전: {record['total_loss']}")
                print(f"설계상 베팅금(1/4, 0.5단위, 최소1): {max(1, round((record['total_loss']/4)*2)/2)}")
                print(f"결과: {'승리' if record['result'] == 'W' else '패배'}")
                print(f"베팅 전 잔액: {record['balance_before']:.2f}")
                print(f"베팅 후 잔액: {record['balance_after']:.2f}")
                count += 1
                if count >= num_examples:
                    break

    def show_normal_mode_losses(self, num_examples=5):
        print("\n=== 일반모드 베팅 시나리오 ===")
        count = 0
        for i, record in enumerate(self.betting_history):
            if record['mode'] == 'Normal':
                print(f"\n베팅 #{i+1} (Normal)")
                print(f"베팅 금액: {record['bet_amount']}")
                print(f"결과: {'승리' if record['result'] == 'W' else '패배'}")
                print(f"베팅 전 잔액: {record['balance_before']:.2f}")
                print(f"베팅 후 잔액: {record['balance_after']:.2f}")
                count += 1
                if count >= num_examples:
                    break

def simulate_strategy():
    conn = sqlite3.connect('pattern_analysis_v2.db')
    cursor = conn.cursor()
    
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
    
    # Recovery 모드 베팅 예시 보여주기
    strategy.show_recovery_bets()
    
    # 일반모드 연속 패배 시나리오 보여주기
    strategy.show_normal_mode_losses()
    
    conn.close()

if __name__ == "__main__":
    simulate_strategy() 