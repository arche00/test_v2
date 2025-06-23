CREATE TABLE pattern_analysis (
    -- Primary key
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    
    -- Session information
    session_id VARCHAR(36) NOT NULL,
    session_date DATE NOT NULL,
    total_groups_in_session INT NOT NULL,
    
    -- Group information
    group_id VARCHAR(10) NOT NULL,          -- Format: '1-3', '2-4', etc.
    group_start INT NOT NULL,               -- Starting position of the group
    group_end INT NOT NULL,                 -- Ending position of the group
    group_sequence INT NOT NULL,            -- Order within session
    
    -- Pattern 1,2 information
    pattern12_result CHAR(1),               -- B or P
    pattern12_combined INT,                 -- Combined pattern number (4 digits)
    pattern12_prediction CHAR(1),           -- B or P
    pattern12_prediction_result CHAR(1),    -- W or L
    
    -- Pattern 1,2,3 information
    pattern123_result CHAR(1),              -- B or P
    pattern123_combined INT,                -- Combined pattern number (6 digits)
    pattern123_prediction CHAR(1),          -- B or P
    pattern123_prediction_result CHAR(1),   -- W or L
    
    -- Sequence type
    sequence_type VARCHAR(20),              -- P_Sequence or B_Sequence
    
    -- Prediction accuracy
    prediction_accuracy DECIMAL(5,2),       -- Prediction accuracy percentage
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Indexes
    INDEX idx_session_id (session_id),
    INDEX idx_session_date (session_date),
    INDEX idx_group_id (group_id),
    INDEX idx_group_range (group_start, group_end),
    INDEX idx_pattern12_combined (pattern12_combined),
    INDEX idx_pattern123_combined (pattern123_combined),
    
    -- Constraints
    CONSTRAINT unique_session_group UNIQUE (session_id, group_id),
    CONSTRAINT valid_pattern12_result CHECK (pattern12_result IN ('B', 'P')),
    CONSTRAINT valid_pattern123_result CHECK (pattern123_result IN ('B', 'P')),
    CONSTRAINT valid_pattern12_prediction CHECK (pattern12_prediction IN ('B', 'P')),
    CONSTRAINT valid_pattern123_prediction CHECK (pattern123_prediction IN ('B', 'P')),
    CONSTRAINT valid_prediction_result CHECK (pattern12_prediction_result IN ('W', 'L') AND pattern123_prediction_result IN ('W', 'L')),
    CONSTRAINT valid_sequence_type CHECK (sequence_type IN ('P_Sequence', 'B_Sequence')),
    CONSTRAINT valid_prediction_accuracy CHECK (prediction_accuracy >= 0 AND prediction_accuracy <= 100)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci; 