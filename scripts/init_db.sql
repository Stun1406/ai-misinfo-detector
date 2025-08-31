-- AI Misinformation Detector Database Initialization
-- This script will be run automatically when PostgreSQL container starts

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Set timezone
SET timezone = 'UTC';

-- The tables will be created by SQLAlchemy models
-- This script can include any additional indexes or constraints if needed

-- Create indexes for better performance
-- These will be created after tables are initialized by the application

-- Sample data can be inserted here for testing
-- INSERT INTO fact_sources (title, content, source_name, source_url, topic) VALUES 
-- ('Sample Fact Check', 'This is a sample verified fact...', 'Test Source', 'https://example.com', 'health');
