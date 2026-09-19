#!/bin/sh
set -eu
export SQL_PLUGIN=postgres12
export SQL_HOST=postgres
export SQL_PORT=5432
export SQL_USER=runtime_dev
export SQL_PASSWORD=runtime_dev_only
export SQL_DATABASE=temporal
temporal-sql-tool setup-schema -v 0.0
temporal-sql-tool update-schema -d /etc/temporal/schema/postgresql/v12/temporal/versioned
export SQL_DATABASE=temporal_visibility
temporal-sql-tool setup-schema -v 0.0
temporal-sql-tool update-schema -d /etc/temporal/schema/postgresql/v12/visibility/versioned
