-- Create Database
CREATE DATABASE IF NOT EXISTS soil_health_db;
USE soil_health_db;

-- Table for Users
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    role ENUM('admin', 'farmer') DEFAULT 'farmer',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for Soil Test Results
CREATE TABLE IF NOT EXISTS soil_tests (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    nitrogen FLOAT NOT NULL,
    phosphorous FLOAT NOT NULL,
    potassium FLOAT NOT NULL,
    temperature FLOAT NOT NULL,
    humidity FLOAT NOT NULL,
    moisture FLOAT NOT NULL,
    soil_type VARCHAR(50) NOT NULL,
    crop_type VARCHAR(50) NOT NULL,
    soil_health VARCHAR(50) NOT NULL,
    fertilizer VARCHAR(100) NOT NULL,
    score FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Insert Default Admin
-- Password 'admin' will be hashed in Python generally, 
-- but for simple WAMP setup we often start with direct inserts if safe.
-- I will provide a registration route, but here is a placeholder.
INSERT IGNORE INTO users (name, email, password, role) 
VALUES ('Admin User', 'admin@example.com', 'admin', 'admin');
