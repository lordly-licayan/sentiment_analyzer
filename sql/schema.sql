-- =========================================
-- Table: FileInfoTbl
-- =========================================
-- Drop the table if it already exists
DROP TABLE IF EXISTS FileInfoTbl CASCADE;

CREATE TABLE IF NOT EXISTS FileInfoTbl (
    id SERIAL PRIMARY KEY,
    file_id VARCHAR(255) UNIQUE NOT NULL,
    filename VARCHAR(255) NOT NULL,
    no_of_data INTEGER NOT NULL,
    date_uploaded TIMESTAMPTZ DEFAULT NOW(),
    remarks TEXT
);


-- =========================================
-- Table: CommentsTbl
-- =========================================
-- Drop the table if it already exists
DROP TABLE IF EXISTS CommentsTbl CASCADE;

CREATE TABLE IF NOT EXISTS CommentsTbl (
    id SERIAL PRIMARY KEY,
    file_id VARCHAR(255) NOT NULL,
    comment TEXT NOT NULL,
    label INTEGER NOT NULL,
    remarks TEXT,
    FOREIGN KEY (file_id) REFERENCES FileInfoTbl(file_id) ON DELETE CASCADE
);


-- =========================================
-- Table: TrainedModelTbl
-- =========================================
-- Drop the table if it already exists
DROP TABLE IF EXISTS TrainedModelTbl CASCADE;

CREATE TABLE IF NOT EXISTS TrainedModelTbl (
    id SERIAL PRIMARY KEY,
    sy VARCHAR(20) NOT NULL,
    semester VARCHAR(20) NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    classifier VARCHAR(255) NOT NULL,
    metrics JSON DEFAULT '{}'::json,
    no_of_data INTEGER DEFAULT 0,
    date_trained TIMESTAMPTZ DEFAULT NOW(),
    remarks TEXT
);