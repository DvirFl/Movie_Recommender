-- Creates a separate database for Optuna study storage,
-- avoiding write contention with MLflow on the same DB.
CREATE DATABASE optuna_studies;
GRANT ALL PRIVILEGES ON DATABASE optuna_studies TO current_user;