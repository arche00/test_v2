import sqlite3
import re

def find_longest_repeat(pattern, sequences):
    max_len = 0
    # 패턴의 각 문자를 선택적으로 매칭하도록 수정
    regex = f'({pattern[0]}.*?{pattern[1]})'
    for seq in sequences:
        seq = seq.lower()
        for match in re.finditer(regex, seq):
            repeat = match.group(1)
            if len(repeat) > max_len:
                max_len = len(repeat)
    return max_len

def find_longest_single(char, sequences):
    max_len = 0
    regex = f'({char}+)'  # 연속된 char
    for seq in sequences:
        seq = seq.lower()
        for match in re.finditer(regex, seq):
            repeat = match.group(1)
            if len(repeat) > max_len:
                max_len = len(repeat)
    return max_len

def main():
    conn = sqlite3.connect('pattern_analysis_v2.db')
    cur = conn.cursor()
    cur.execute("SELECT prediction_results FROM session_prediction_results WHERE prediction_results IS NOT NULL")
    sequences = [row[0] for row in cur.fetchall()]
    conn.close()

    wl_max = find_longest_repeat('wl', sequences)
    lw_max = find_longest_repeat('lw', sequences)
    w_max = find_longest_single('w', sequences)
    l_max = find_longest_single('l', sequences)

    print(f"가장 긴 'wl' 반복 패턴 길이: {wl_max}")
    print(f"가장 긴 'lw' 반복 패턴 길이: {lw_max}")
    print(f"가장 긴 'w' 연속 길이: {w_max}")
    print(f"가장 긴 'l' 연속 길이: {l_max}")

if __name__ == "__main__":
    main() 