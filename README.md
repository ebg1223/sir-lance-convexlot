# Data Loader

Data Loader loads Convex document deltas and schemas into Lance datasets, with S3-backed state for cursors, locks, schema cache, table config state, audit records, and maintenance metadata.

## What it does

- Reads Convex deltas from `/api/document_deltas`
- Reads Convex JSON schemas from `/api/json_schemas`
- Writes and reconciles Lance datasets on S3
- Tracks incremental cursor and lock state in S3
- Supports generated columns and scalar indexes
- Supports table-specific TOML index config
- Includes maintenance commands for index optimization, compaction, and cleanup
- Includes migration/repair utilities for existing S3 Tables and Lance data

## Install

Requires Python 3.12+.

```bash
uv sync
```

## CLI

```bash
uv run sirlance --help
```

Common commands:

```bash
uv run sirlance incremental-once
uv run sirlance incremental-loop
uv run sirlance migrate-table
uv run sirlance create-lance-indexes
```

## Core environment variables

```bash
CONVEX_URL=...
CONVEX_DEPLOY_KEY=...
LANCE_BUCKET=...
LANCE_PREFIX=tables
LANCE_INCREMENTAL_STATE_BUCKET=...
LANCE_INCREMENTAL_STATE_PREFIX=lance-incremental
AWS_REGION=us-east-1
```

Optional table config:

```bash
LANCE_TABLE_CONFIG_DIR=config/lance/tables
LANCE_TABLE_CONFIG_BUCKET=...
LANCE_TABLE_CONFIG_PREFIX=config/lance/tables
```

## Docker

```bash
docker build -t sirlance .
docker run --rm sirlance --help
```

## Tests

```bash
uv run python -m unittest discover -s tests -v
```
