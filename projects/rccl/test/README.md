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

### 1. Process Isolated Test Runner
Run tests in isolated processes with clean environment settings.

📄 **[Full Documentation](common/ProcessIsolatedTestRunner.md)**

### 2. MPI Test Runner
Base class for multi-process distributed tests using MPI.

📄 **[Full Documentation](common/MPITestRunner.md)**

