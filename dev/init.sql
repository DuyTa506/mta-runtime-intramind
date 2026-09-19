-- Disposable local development only. Application and test ledgers are separate.
CREATE USER runtime_test WITH PASSWORD 'runtime_test_only';
CREATE DATABASE runtime_test OWNER runtime_test;
CREATE DATABASE temporal OWNER runtime_dev;
CREATE DATABASE temporal_visibility OWNER runtime_dev;
REVOKE CONNECT ON DATABASE runtime_test FROM PUBLIC;
GRANT CONNECT ON DATABASE runtime_test TO runtime_test;
