-- pattern_records 테이블 생성
CREATE TABLE IF NOT EXISTS pattern_records (
    id SERIAL PRIMARY KEY,
    pattern VARCHAR(255) NOT NULL,
    next_pattern VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- group_sequences 테이블 생성
CREATE TABLE IF NOT EXISTS group_sequences (
    id SERIAL PRIMARY KEY,
    sequence TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_pattern ON pattern_records(pattern);
CREATE INDEX IF NOT EXISTS idx_next_pattern ON pattern_records(next_pattern);
CREATE INDEX IF NOT EXISTS idx_timestamp ON pattern_records(timestamp); 