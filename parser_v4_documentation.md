# Bead Road Parser v4 Documentation

## Overview
Bead Road Parser v4 is a Streamlit-based application for analyzing pattern sequences in gambling games. It provides tools for parsing SVG-based bead road diagrams, analyzing patterns, and predicting outcomes based on historical data.

## Key Features
- SVG parsing of bead road diagrams
- Pattern analysis and grouping
- Machine learning-based prediction system
- Manual grid editing capabilities
- Statistical analysis of pattern occurrences

## Architecture
The application is divided into three main columns with a 1:1:1 ratio:
1. **Left Column**: SVG input and grid display
2. **Middle Column**: Pattern analysis and group detection
3. **Right Column**: Prediction system

## Components

### SVG Parsing and Grid Display
- Parses SVG code to extract bead road grid data
- Displays the original and converted grid (with tie values converted)
- Allows manual editing of grid cells

### Pattern Analysis
- Divides the grid into overlapping zones
- Identifies patterns within each zone
- Matches patterns to predefined pattern groups (A/B)
- Displays pattern statistics and historical data

### Prediction System
- Two independent prediction systems:
  - Middle area: Pattern1 → Result1 prediction
  - Right area: Pattern2 → Result2 prediction
- Each system maintains its own state variables and user interface
- Pattern filtering based on sequence matching

## State Management
- Each section (middle and right) maintains completely separate state variables to prevent interference
- Independent functions handle prediction and filtering for each area
- Distinct UI components with unique keys prevent state sharing

## Technical Implementation
- Pattern data stored in pattern.json
- Historical records stored in SQLite database
- Machine learning model using RandomForestClassifier
- Memory optimization through chunked data loading

## Known Limitations
- High memory usage during model training
- Performance can degrade with large datasets
- UI spacing can be inconsistent on some displays

## Future Enhancements
- Improved prediction algorithms
- Enhanced visualization options
- Export/import functionality for patterns
- Real-time data integration

---

# 비드 로드 파서 v4 문서

## 개요
비드 로드 파서 v4는 도박 게임에서의 패턴 시퀀스를 분석하기 위한 Streamlit 기반 애플리케이션입니다. SVG 기반 비드로드 다이어그램을 파싱하고, 패턴을 분석하며, 과거 데이터를 기반으로 결과를 예측하는 도구를 제공합니다.

## 주요 기능
- SVG 비드 로드 다이어그램 파싱
- 패턴 분석 및 그룹화
- 머신러닝 기반 예측 시스템
- 수동 그리드 편집 기능
- 패턴 발생 통계 분석

## 아키텍처
애플리케이션은 1:1:1 비율의 세 개의 주요 컬럼으로 나뉩니다:
1. **왼쪽 컬럼**: SVG 입력 및 그리드 디스플레이
2. **중앙 컬럼**: 패턴 분석 및 그룹 감지
3. **오른쪽 컬럼**: 예측 시스템

## 구성 요소

### SVG 파싱 및 그리드 디스플레이
- SVG 코드를 파싱하여 비드 로드 그리드 데이터 추출
- 원본 및 변환된 그리드(타이 값 변환) 표시
- 그리드 셀 수동 편집 가능

### 패턴 분석
- 그리드를 겹치는 영역으로 분할
- 각 영역 내의 패턴 식별
- 패턴을 사전 정의된 패턴 그룹(A/B)과 매칭
- 패턴 통계 및 과거 데이터 표시

### 예측 시스템
- 두 개의 독립적인 예측 시스템:
  - 중앙 영역: Pattern1 → Result1 예측
  - 오른쪽 영역: Pattern2 → Result2 예측
- 각 시스템은 자체 상태 변수와 사용자 인터페이스 유지
- 시퀀스 매칭 기반 패턴 필터링

## 상태 관리
- 각 섹션(중앙 및 오른쪽)은 간섭을 방지하기 위해 완전히 별도의 상태 변수 유지
- 각 영역에 대한 독립적인 함수가 예측 및 필터링 처리
- 고유 키를 가진 별개의 UI 구성 요소로 상태 공유 방지

## 기술적 구현
- pattern.json에 패턴 데이터 저장
- SQLite 데이터베이스에 과거 기록 저장
- RandomForestClassifier를 사용한 머신러닝 모델
- 청크 데이터 로딩을 통한 메모리 최적화

## 알려진 제한사항
- 모델 학습 중 높은 메모리 사용량
- 대용량 데이터셋에서 성능 저하 가능성
- 일부 디스플레이에서 UI 간격이 일관되지 않을 수 있음

## 향후 개선사항
- 개선된 예측 알고리즘
- 향상된 시각화 옵션
- 패턴의 내보내기/가져오기 기능
- 실시간 데이터 통합 