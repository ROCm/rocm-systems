# Coverage Workflow for rocprofiler-compute in Monorepo

## Overview

This document describes the coverage generation and CDash upload workflow for `rocprofiler-compute` within the super-repo structure.

## Directory Structure

```
monorepo/
├── projects/
│   └── rocprofiler-compute/
│       ├── CMakeLists.txt
│       ├── coverage/
│       │   └── coverage-latest.xml  # committed coverage file
│       ├── tools/
│       │   ├── update_coverage.sh  # coverage generation/update script
│       │   └── run-ci.py             # CDash upload script
│       └── ...
└── .github/
    └── workflows/
        └── rocprofiler-compute-cdash.yml  # gitHub Action for auto-upload
```

## Workflows

### 1. Manual Coverage Generation (Periodic Update)

Run this periodically to update the coverage baseline:

```bash
# From monorepo root
cd projects/rocprofiler-compute
./tools/update_coverage.sh

# This will:
# - Build with coverage enabled
# - Run all tests
# - Generate coverage.xml
# - Copy/overwrite to coverage/coverage-latest.xml
# - Show instructions for committing
```

### 2. Commit Coverage to Repository

```bash
# From monorepo root
git add projects/rocprofiler-compute/coverage/coverage-latest.xml
git commit -m "rocprofiler-compute: Update coverage to XX.X% line coverage"
git push origin develop
```

### 3. Automatic CDash Upload

When `coverage/coverage-latest.xml` is pushed to `develop` branch:
- GitHub Action triggers automatically
- Uploads coverage to CDash
- Posts summary to GitHub Actions

## CDash Project

View results at: https://my.cdash.org/index.php?project=rocprofiler-compute

## Maintenance

## Troubleshooting

### Coverage Generation Fails

```bash
#check Python dependencies
pip install coverage pytest pytest-cov

#verify tests can run
cd projects/rocprofiler-compute/build
ctest --verbose
```
