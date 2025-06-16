# Pattern Prediction 추가 필터링 로직 정리 v4

이 문서는 Pattern1, Pattern2 예측 결과에 대해 추가 필터링을 적용하는 로직을 요약합니다. 추후 재사용 및 유지보수를 위해 참고용으로 활용할 수 있습니다.

---

## 1. Pattern1 추가 필터링 (`filter_predictions_by_pattern1`)

### 주요 단계
1. **패턴 데이터 로드**
   - `pattern.json`에서 패턴 정보를 불러옴
2. **검색 접두사 추출**
   - 입력값의 앞 2자리(`pattern1_number[:2]`)를 접두사로 사용
3. **초기 패턴 검색**
   - 접두사로 시작하는 패턴을 검색
4. **타겟 시퀀스 추출**
   - 검색된 패턴의 시퀀스에서 4~6번째 값을 이어붙여 타겟 시퀀스 생성
5. **관련 패턴 재검색**
   - 타겟 시퀀스로 시작하는 패턴을 다시 검색
6. **예측 후보 필터링**
   - 예측 결과 중 관련 패턴 번호와 일치하는 것만 남김
7. **B/P 분류 및 최고 확률 선택**
   - 시퀀스 4번째 값이 'b'인 후보 중 최고 확률 1개, 'p'인 후보 중 최고 확률 1개를 각각 선택
   - 둘 다 없으면 상위 3개 확률 반환

### 반환
- B/P 최고 확률 1개씩(있으면), 없으면 상위 3개

---

## 2. Pattern2 추가 필터링 (`filter_predictions_by_pattern2`)

### 주요 단계
1. **패턴 데이터 로드**
   - `pattern.json`에서 패턴 정보를 불러옴
2. **검색 접두사 추출**
   - 입력값의 3,4번째(`pattern2_number[2:4]`)를 접두사로 사용
3. **초기 패턴 검색**
   - 접두사로 시작하는 패턴을 검색
4. **타겟 시퀀스 추출**
   - 검색된 패턴의 시퀀스에서 4~6번째 값을 이어붙여 타겟 시퀀스 생성
5. **관련 패턴 재검색**
   - 타겟 시퀀스로 시작하는 패턴을 다시 검색
6. **예측 후보 필터링**
   - 예측 결과 중 관련 패턴 번호와 일치하는 것만 남김
7. **B/P 분류 및 최고 확률 선택**
   - 시퀀스 4번째 값이 'b'인 후보 중 최고 확률 1개, 'p'인 후보 중 최고 확률 1개를 각각 선택
   - 둘 다 없으면 상위 3개 확률 반환

### 반환
- B/P 최고 확률 1개씩(있으면), 없으면 상위 3개

---

## 3. 공통 참고사항
- 관련 패턴 번호는 `pattern.json`의 `pattern_number`와 일치
- 시퀀스는 `sequence` 필드의 4번째 값(`sequence[3]`)으로 B/P 분류
- 예측 결과는 확률 기준 내림차순 정렬

---

## 4. 재사용 시 참고
- 위 로직은 함수로 분리하여 재사용 가능
- 패턴 데이터 구조(`pattern.json`)가 바뀌면 로직도 함께 수정 필요
- B/P 외 다른 분류가 필요할 경우, 시퀀스 인덱스만 조정하면 됨 

---

## 5. 시행착오 및 실제 작성 코드

### 시행착오 요약
- **문제점:**
  - Pattern1/Pattern2 예측 UI와 세션 상태를 분리했음에도, 한쪽 예측/초기화 시 다른 쪽이 사라지거나 값이 초기화되는 현상 발생
  - Streamlit rerun 및 session_state 동기화 문제로, 입력 필드의 key와 value 관리가 꼬임
  - 입력 필드의 key를 카운터로 관리했으나, 버튼 클릭 시 key 증가 타이밍/위치가 미묘하게 달라서 완전한 분리가 안 됨
  - parser_v3와 달리, parser_v4에서는 입력 필드의 key와 session_state 동기화 방식이 달라서 영향 발생
- **시도한 방법:**
  - 입력 필드의 value 인자 제거, key만 사용
  - 버튼 클릭 시 input_key(카운터) 증가
  - session_state를 명확히 분리
- **결과:**
  - UI상 분리는 되었으나, Streamlit rerun 및 위젯 재생성 타이밍 문제로 완전한 분리는 어려웠음
  - parser_v3의 방식(버튼 클릭 시 key 증가 및 value 동기화)을 참고하는 것이 가장 효과적임을 확인

### 실제 작성한 추가 필터링 함수 예시

#### filter_predictions_by_pattern1
```python
def filter_predictions_by_pattern1(pattern1_number, predictions):
    if not predictions:
        return predictions
    pattern_data = load_pattern_data()
    if not pattern_data:
        return predictions
    search_prefix = str(pattern1_number)[:2]
    initial_patterns = search_patterns(pattern_data, search_prefix)
    if not initial_patterns:
        return predictions[:3]
    target_sequences = []
    for pattern in initial_patterns:
        sequence = pattern['sequence']
        if len(sequence) >= 6:
            target_seq = ''.join(sequence[3:6]).lower()
            target_sequences.append(target_seq)
    related_patterns = []
    for seq in target_sequences:
        for group_name in ['groupA', 'groupB']:
            patterns = pattern_data['patterns'][group_name]
            for pattern in patterns:
                pattern_seq = ''.join(pattern.get('sequence', [])).lower()
                if pattern_seq.startswith(seq):
                    related_patterns.append({
                        'group': group_name[5],
                        'sequence': pattern.get('sequence', []),
                        'group_value': pattern.get('group', group_name[5].lower()),
                        'pattern_number': pattern.get('pattern_number', 'N/A')
                    })
    related_patterns_dict = {p['pattern_number']: p for p in related_patterns if p['pattern_number'] != 'N/A'}
    filtered_predictions = [
        {
            **pred,
            **related_patterns_dict[str(pred['number'])]
        }
        for pred in predictions if str(pred['number']) in related_patterns_dict
    ]
    b_candidates = [pred for pred in filtered_predictions if len(pred['sequence']) > 3 and str(pred['sequence'][3]).lower() == 'b']
    p_candidates = [pred for pred in filtered_predictions if len(pred['sequence']) > 3 and str(pred['sequence'][3]).lower() == 'p']
    best_b = max(b_candidates, key=lambda x: x['probability']) if b_candidates else None
    best_p = max(p_candidates, key=lambda x: x['probability']) if p_candidates else None
    result_list = [pred for pred in [best_b, best_p] if pred is not None]
    if not result_list:
        return filtered_predictions[:3]
    result_list.sort(key=lambda x: x['probability'], reverse=True)
    return result_list
```

#### filter_predictions_by_pattern2
```python
def filter_predictions_by_pattern2(pattern2_number, predictions):
    if not predictions:
        return predictions
    pattern_data = load_pattern_data()
    if not pattern_data:
        return predictions
    pattern2_str = str(pattern2_number)
    if len(pattern2_str) < 4:
        return predictions[:3]
    search_prefix = pattern2_str[2:4]
    initial_patterns = search_patterns(pattern_data, search_prefix)
    if not initial_patterns:
        return predictions[:3]
    target_sequences = []
    for pattern in initial_patterns:
        sequence = pattern['sequence']
        if len(sequence) >= 6:
            target_seq = ''.join(sequence[3:6]).lower()
            target_sequences.append(target_seq)
    related_patterns = []
    for seq in target_sequences:
        for group_name in ['groupA', 'groupB']:
            patterns = pattern_data['patterns'][group_name]
            for pattern in patterns:
                pattern_seq = ''.join(pattern.get('sequence', [])).lower()
                if pattern_seq.startswith(seq):
                    related_patterns.append({
                        'group': group_name[5],
                        'sequence': pattern.get('sequence', []),
                        'group_value': pattern.get('group', group_name[5].lower()),
                        'pattern_number': pattern.get('pattern_number', 'N/A')
                    })
    related_patterns_dict = {p['pattern_number']: p for p in related_patterns if p['pattern_number'] != 'N/A'}
    filtered_predictions = [
        {
            **pred,
            **related_patterns_dict[str(pred['number'])]
        }
        for pred in predictions if str(pred['number']) in related_patterns_dict
    ]
    b_candidates = [pred for pred in filtered_predictions if len(pred['sequence']) > 3 and str(pred['sequence'][3]).lower() == 'b']
    p_candidates = [pred for pred in filtered_predictions if len(pred['sequence']) > 3 and str(pred['sequence'][3]).lower() == 'p']
    best_b = max(b_candidates, key=lambda x: x['probability']) if b_candidates else None
    best_p = max(p_candidates, key=lambda x: x['probability']) if p_candidates else None
    final_predictions = [pred for pred in [best_b, best_p] if pred is not None]
    if not final_predictions:
        return filtered_predictions[:3]
    final_predictions.sort(key=lambda x: x['probability'], reverse=True)
    return final_predictions
``` 