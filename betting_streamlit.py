import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
import os

# -------------------------------------------------------------
# Betting Amount Logic (베팅 금액 로직)
#
# - base_bet: Minimum/initial bet unit (default 1)
# - bet_unit: Bet increment unit (default 0.5)
#
# [Normal Mode]
#   - On first bet and after any loss: current_bet = base_bet (always reset to initial amount)
#   - After 2 or more consecutive wins: current_bet += bet_unit (up to max_bet)
#   - After win_streak_threshold(3) consecutive wins: win count resets
#
# [Recovery Mode]
#   - current_bet is set to 1/4 of total_loss, rounded to the nearest 0.5
#   - On win: total_loss decreases by current_bet, if total_loss <= 0 return to Normal Mode
#   - On loss: total_loss increases by current_bet, recalculate current_bet
#
# [Common]
#   - All bet records are saved in betting_history (bet_amount, result, mode, etc.)
#   - Consecutive win/loss counters are updated per result
# -------------------------------------------------------------

# --- RecoveryStrategy from show_betting_example.py ---
class RecoveryStrategy:
    def __init__(self, min_bet=1, initial_balance=0, initial_bet=None, bet_unit=0.5):
        self.base_bet = min_bet
        self.current_bet = initial_bet if initial_bet is not None else self.base_bet
        self.max_bet = 8
        self.min_bet = min_bet
        self.bet_unit = bet_unit  # 베팅 증감 단위
        self.total_wins = 0
        self.total_losses = 0
        self.balance = initial_balance
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

        bet_record = {
            'Time': datetime.now().strftime('%H:%M:%S'),
            'bet_amount': self.current_bet,
            'result': result,
            'mode': 'Recovery' if self.recovery_mode else 'Normal',
            'balance_before': self.balance,
            'total_loss': self.total_loss
        }

        if self.recovery_mode:
            if result == 'W':
                self.balance += self.current_bet
                self.total_wins += 1
                self.consecutive_wins += 1
                self.consecutive_losses = 0
                self.total_loss -= self.current_bet
                if self.total_loss <= 0:
                    self.recovery_mode = False
                    self.current_bet = self.base_bet
                else:
                    self.current_bet = self.calculate_recovery_bet()
            else:  # L
                self.balance -= self.current_bet
                self.total_losses += 1
                self.consecutive_losses += 1
                self.consecutive_wins = 0
                self.total_loss += self.current_bet
                self.current_bet = self.calculate_recovery_bet()
        else:
            if result == 'W':
                self.balance += self.current_bet
                self.total_wins += 1
                self.consecutive_wins += 1
                self.consecutive_losses = 0
                if self.consecutive_wins >= 2:
                    self.adjust_bet(increase=True)
                else:
                    self.current_bet = self.base_bet
                if self.consecutive_wins >= self.win_streak_threshold:
                    self.consecutive_wins = 0
            else:  # L
                self.balance -= self.current_bet
                self.total_losses += 1
                self.consecutive_losses += 1
                self.consecutive_wins = 0
                self.total_loss += self.current_bet
                if self.total_loss > 0:
                    self.recovery_mode = True
                    self.current_bet = self.calculate_recovery_bet()
                else:
                    self.current_bet = self.base_bet

        bet_record['balance_after'] = self.balance
        bet_record['consecutive_losses'] = self.consecutive_losses
        self.betting_history.append(bet_record)

    def get_state(self):
        return {
            'Mode': 'Recovery' if self.recovery_mode else 'Normal',
            'Current Bet': self.current_bet,
            'Total Wins': self.total_wins,
            'Total Losses': self.total_losses,
            'Balance': self.balance,
            'Total Loss': self.total_loss,
            'Consecutive Wins': self.consecutive_wins,
            'Consecutive Losses': self.consecutive_losses
        }

# --- DB Simulation Function ---
def load_prediction_results(db_path):
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT prediction_results FROM session_prediction_results")
        results = cursor.fetchall()
        sequence = []
        for row in results:
            if row[0]:
                sequence.extend(list(row[0]))
        return sequence
    except Exception as e:
        return None
    finally:
        conn.close()

def simulate_with_db_data(sequence, min_bet=1, initial_balance=0, initial_bet=None, bet_unit=0.5):
    strategy = RecoveryStrategy(min_bet=min_bet, initial_balance=initial_balance, initial_bet=initial_bet, bet_unit=bet_unit)
    for outcome in sequence:
        if outcome in ['W', 'L']:
            strategy.bet(outcome)
    state = strategy.get_state()
    win_rate = (state['Total Wins'] / (state['Total Wins'] + state['Total Losses'])) if (state['Total Wins'] + state['Total Losses']) > 0 else 0
    df = pd.DataFrame(strategy.betting_history)
    return state, df, win_rate

# --- Simulation Function (Random) ---
def simulate_to_target(target_profit, min_bet, win_rate=0.48, n_sim=1000, max_rounds=10000, initial_balance=0, initial_bet=None, bet_unit=0.5):
    rounds_needed = []
    for _ in range(n_sim):
        strategy = RecoveryStrategy(min_bet=min_bet, initial_balance=initial_balance, initial_bet=initial_bet, bet_unit=bet_unit)
        rounds = 0
        while strategy.balance < target_profit and rounds < max_rounds:
            result = 'W' if np.random.rand() < win_rate else 'L'
            strategy.bet(result)
            rounds += 1
        if strategy.balance >= target_profit:
            rounds_needed.append(rounds)
    if rounds_needed:
        return np.mean(rounds_needed), np.std(rounds_needed), len(rounds_needed)
    else:
        return None, None, 0

# --- Streamlit App ---
def get_strategy(min_bet=1, initial_balance=0, initial_bet=None, bet_unit=0.5):
    if (
        'strategy' not in st.session_state or
        st.session_state.get('min_bet', 1) != min_bet or
        st.session_state.get('initial_balance', 0) != initial_balance or
        st.session_state.get('initial_bet', None) != initial_bet or
        st.session_state.get('bet_unit', 0.5) != bet_unit
    ):
        st.session_state['strategy'] = RecoveryStrategy(min_bet=min_bet, initial_balance=initial_balance, initial_bet=initial_bet, bet_unit=bet_unit)
        st.session_state['min_bet'] = min_bet
        st.session_state['initial_balance'] = initial_balance
        st.session_state['initial_bet'] = initial_bet
        st.session_state['bet_unit'] = bet_unit
    return st.session_state['strategy']

def reset_strategy(min_bet=1, initial_balance=0, initial_bet=None, bet_unit=0.5):
    st.session_state['strategy'] = RecoveryStrategy(min_bet=min_bet, initial_balance=initial_balance, initial_bet=initial_bet, bet_unit=bet_unit)
    st.session_state['min_bet'] = min_bet
    st.session_state['initial_balance'] = initial_balance
    st.session_state['initial_bet'] = initial_bet
    st.session_state['bet_unit'] = bet_unit

db_path = 'pattern_analysis_v2.db'

st.set_page_config(page_title="Betting Strategy Simulator", layout="wide")
st.title("Betting Strategy Simulator")

# --- Sidebar Controls ---
st.sidebar.header("Simulation Settings")
target_profit = st.sidebar.number_input("Target Profit", min_value=1.0, value=10.0, step=0.5)
min_bet = st.sidebar.number_input("Minimum Bet Unit", min_value=0.5, value=1.0, step=0.5)
initial_balance = st.sidebar.number_input("Initial Amount", min_value=0.0, value=0.0, step=1.0)
initial_bet = st.sidebar.number_input("Initial Bet Amount", min_value=min_bet, value=min_bet, step=bet_unit if 'bet_unit' in locals() else 0.5)
bet_unit = st.sidebar.number_input("Bet Increment Unit", min_value=0.1, value=0.5, step=0.1)
run_sim = st.sidebar.button("Simulate to Target Profit")

st.sidebar.header("DB Data")
if not os.path.exists(db_path):
    st.sidebar.warning(f"DB file not found: {db_path}")
    db_sequence = None
else:
    if 'db_sequence' not in st.session_state:
        st.session_state['db_sequence'] = load_prediction_results(db_path)
    if st.sidebar.button("Reload DB Data"):
        st.session_state['db_sequence'] = load_prediction_results(db_path)
    db_sequence = st.session_state['db_sequence']
    st.sidebar.info(f"Loaded {len(db_sequence) if db_sequence else 0} outcomes from DB")
run_db_sim = st.sidebar.button("Simulate with DB Data")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Current State")
    strategy = get_strategy(min_bet=min_bet, initial_balance=initial_balance, initial_bet=initial_bet, bet_unit=bet_unit)
    state = strategy.get_state()
    next_bet = state['Current Bet']  # Always show the strategy-calculated bet amount
    for k, v in state.items():
        if k == 'Balance':
            color = 'green' if v >= 0 else 'red'
            st.markdown(f"**{k}: <span style='color:{color}'>{v}</span>**", unsafe_allow_html=True)
        else:
            st.markdown(f"**{k}: {v}**")
    st.markdown(f"**Next Bet Amount: <span style='color:blue'>{next_bet}</span>**", unsafe_allow_html=True)
    st.write("")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if st.button("Win", key="win_btn"):
            strategy.bet('W')
            st.experimental_rerun()
    with c2:
        if st.button("Loss", key="loss_btn"):
            strategy.bet('L')
            st.experimental_rerun()
    with c3:
        if st.button("Reset History", key="reset_btn"):
            reset_strategy(min_bet=min_bet, initial_balance=initial_balance, initial_bet=initial_bet, bet_unit=bet_unit)
            st.experimental_rerun()

with col2:
    st.subheader("Betting History")
    if strategy.betting_history:
        df = pd.DataFrame(strategy.betting_history)
        st.dataframe(df[::-1], use_container_width=True)
    else:
        st.info("No betting history yet.")

# --- Simulation Result (Random) ---
if run_sim:
    with st.spinner("Simulating..."):
        avg_rounds, std_rounds, n_success = simulate_to_target(target_profit, min_bet, initial_balance=initial_balance, initial_bet=initial_bet, bet_unit=bet_unit)
    if n_success > 0:
        st.success(f"Average rounds to reach target profit ({target_profit}): {avg_rounds:.1f} ± {std_rounds:.1f} (from {n_success} runs)")
    else:
        st.error("Target profit was not reached in any simulation run. Try lowering the target or increasing the win rate.")

# --- Simulation Result (DB Data) ---
if run_db_sim:
    if db_sequence is None:
        st.error("No DB data loaded.")
    elif not db_sequence:
        st.warning("DB data is empty.")
    else:
        with st.spinner("Simulating with DB data..."):
            state, df, win_rate = simulate_with_db_data(db_sequence, min_bet=min_bet, initial_balance=initial_balance, initial_bet=initial_bet, bet_unit=bet_unit)
        st.subheader("DB Data Simulation Result (show_betting_example.py logic)")
        st.write(f"Final Balance: {state['Balance']}")
        st.write(f"Total Wins: {state['Total Wins']}, Total Losses: {state['Total Losses']}")
        st.write(f"Win Rate: {win_rate*100:.2f}%")
        st.dataframe(df[::-1], use_container_width=True) 