# RCCL Test Suite

Testing infrastructure for ROCm Communication Collectives Library (RCCL).

## Table of Contents
- [Overview](#overview)
- [Testing Frameworks](#testing-frameworks)

---

## Overview

The RCCL test suite provides following frameworks along with the existing rccl-UnitTests TestBed framework:

## Testing Frameworks

Following is a new testing framework for running single node & single process test in isolation:

### 1. Process Isolated Test Framework
Run tests in isolated processes with clean environment settings.

📄 **[Full Documentation](common/ProcessIsolatedTestFramework.md)**

### 2. MPI Test Framework
Base class for multi-process distributed tests using MPI. Logging: environment-driven **per-rank log files** (`RCCL_MPI_LOG_ALL_RANKS`), **`TEST_*` macros** with `NCCL_DEBUG`, and scoped **`MPIHelpers::TestLogAssertionContext`** for asserting NCCL lines (see the summary tables in the doc).

📄 **[Full Documentation](common/MPITestFramework.md)**

