-- =========================================
-- Table: file_info
-- =========================================
-- Drop the table if it already exists
DROP TABLE IF EXISTS file_info CASCADE;

CREATE TABLE IF NOT EXISTS file_info (
    id SERIAL PRIMARY KEY,
    file_id VARCHAR(255) UNIQUE NOT NULL,
    filename VARCHAR(255) NOT NULL,
    data_count INTEGER NOT NULL,
    date_uploaded TIMESTAMPTZ DEFAULT NOW(),
    remarks TEXT
);

-- =========================================
-- Table: trained_data
-- =========================================
-- Drop the table if it already exists
DROP TABLE IF EXISTS trained_data CASCADE;

CREATE TABLE IF NOT EXISTS trained_data (
    id SERIAL PRIMARY KEY,
    file_id VARCHAR(255) NOT NULL,
    comment TEXT NOT NULL,
    label INTEGER NOT NULL,
    remarks TEXT,

    CONSTRAINT fk_trained_data_file_info
        FOREIGN KEY (file_id)
        REFERENCES file_info(file_id)
        ON DELETE CASCADE
);


-- =========================================
-- Table: TrainedModelTbl
-- =========================================
-- Drop the table if it already exists
DROP TABLE IF EXISTS trained_model CASCADE;

CREATE TABLE IF NOT EXISTS trained_model (
    id SERIAL PRIMARY KEY,
    school_year VARCHAR(20) NOT NULL,
    semester VARCHAR(20) NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    classifier_name VARCHAR(255) NOT NULL,
    metrics JSON DEFAULT '{}'::json,
    data_count INTEGER DEFAULT 0,
    date_trained TIMESTAMPTZ DEFAULT NOW(),
    remarks TEXT
);

-- =========================================
-- Table: trained_model_result
-- =========================================
-- Drop the table if it already exists
DROP TABLE IF EXISTS trained_model_result CASCADE;

CREATE TABLE IF NOT EXISTS trained_model_result (
    id SERIAL PRIMARY KEY,
    trained_model_id INTEGER NOT NULL,
    comment TEXT NOT NULL,
    actual_label VARCHAR(50) NOT NULL,
    predicted_label VARCHAR(50) NOT NULL,
    confidence REAL,
    is_matched BOOLEAN DEFAULT FALSE,

    CONSTRAINT fk_trained_model_result_trained_model
        FOREIGN KEY (trained_model_id)
        REFERENCES trained_model(id)
        ON DELETE CASCADE
);