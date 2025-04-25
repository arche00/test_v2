import json

def calculate_characteristics(sequence):
    """시퀀스에서 특성을 계산합니다."""
    banker_count = sequence.count('b')
    player_count = sequence.count('p')
    
    # 시퀀스 타입 결정
    if banker_count == len(sequence):
        sequence_type = "continuous"
    elif player_count == len(sequence):
        sequence_type = "continuous"
    else:
        sequence_type = "mixed"
    
    # 전환 횟수 계산
    transitions = 0
    for i in range(len(sequence) - 1):
        if sequence[i] != sequence[i + 1]:
            transitions += 1
    
    return {
        "banker_count": banker_count,
        "player_count": player_count,
        "sequence_type": sequence_type,
        "start_with": sequence[0],
        "end_with": sequence[-1],
        "transitions": transitions
    }

# pattern.json 파일 읽기
with open('pattern.json', 'r') as f:
    data = json.load(f)

# groupA와 groupB의 모든 패턴 업데이트
for group in ['groupA', 'groupB']:
    for pattern in data['patterns'][group]:
        sequence = pattern['sequence']
        pattern['characteristics'] = calculate_characteristics(sequence)

# 업데이트된 데이터 저장
with open('pattern.json', 'w') as f:
    json.dump(data, f, indent=2) 