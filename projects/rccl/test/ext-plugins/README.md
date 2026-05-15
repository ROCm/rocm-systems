# RCCL Plugin Tests

## Description

This directory contains automated tests for RCCL (ROCm Communication Collectives Library) plugins:

1. **CSV Tuner Plugin**: Validates the functionality of the CSV-based tuning plugin across different collective operations (AllReduce, Broadcast, Reduce, AllGather, and ReduceScatter) and various configuration scenarios.

2. **Profiler Plugin**: Validates the profiler plugin that captures detailed runtime events for collective and P2P operations, including Group, Collective, P2P, ProxyOp, ProxyStep, and GPU kernel events.

The tests are written in Python using the pytest framework, making it easy to run, maintain, and extend the test coverage.

## Directory Structure

```
ext-plugins/
├── README.md                    # This file - documentation for the test suite
├── requirements.txt             # Python dependencies required for testing
├── pytest.ini                   # Pytest configuration and test markers
├── .gitignore                   # Git ignore rules for Python/pytest artifacts
├── venv/                        # Python virtual environment (created after setup)
├── logs/                        # Test execution logs and output files
├── profiler_dumps/              # Profiler plugin output (JSON trace files)
├── assets/                      # Test configuration files and assets
│   └── csv_confs/               # CSV configuration files for testing
│       ├── incorrect_values_config.conf
│       ├── multinode_config.conf
│       ├── no_matching_config.conf
│       ├── singlenode_config.conf
│       ├── unsupported_algo_proto_config.conf
│       ├── valid_config_with_wildcards.conf
│       └── valid_config_without_wildcards.conf
└── tests/                       # Test suite directory
    ├── conftest.py              # Pytest fixtures and shared test configuration
    ├── ext-tuner/               # CSV Tuner Plugin specific tests
    │   ├── test_allreduce.py
    │   ├── test_broadcast.py
    │   ├── test_reduce.py
    │   ├── test_allgather.py
    │   └── test_reducescatter.py
    └── ext-profiler/            # Profiler Plugin specific tests
        ├── test_allreduce.py
        ├── test_broadcast.py
        ├── test_reduce.py
        ├── test_allgather.py
        ├── test_alltoall.py
        ├── test_reducescatter.py
        └── test_sendrecv.py
```

## Installation & Setup

### Prerequisites

- Python 3.6 or higher
- RCCL library installed
- ROCm environment configured
- **Important**: Update the installation paths in `tests/conftest.py` to match your environment:

```python
RCCL_INSTALL_DIR = "path/to/rccl"
OMPI_INSTALL_DIR = "path/to/ompi"
RCCL_TESTS_DIR = "path/to/rccl-tests"
```

Replace these placeholder paths with your actual installation directories before running the tests.

### Building the RCCL Plugins

Before running the tests, you need to build the RCCL plugin libraries.

#### Building the CSV Tuner Plugin

The CSV tuner plugin is located in the `ext-tuner/example` directory.

**Step 1: Navigate to the plugin directory**

```bash
cd rccl/ext-tuner/example
```

**Step 2: Build the plugin**

```bash
make
```

This will compile the plugin and create `libnccl-tuner-example.so` in the same directory.

#### Building the Profiler Plugin

The profiler plugin is located in the `ext-plugins/example` directory.

**Step 1: Navigate to the plugin directory**

```bash
cd rccl/test/ext-plugins/example
```

**Step 2: Build the plugin**

```bash
make
```

This will compile the plugin and create `libnccl-profiler.so` in the same directory. The profiler plugin captures detailed runtime events including:
- **Group events**: High-level operation grouping
- **Collective events**: AllReduce, Broadcast, Reduce, ReduceScatter operations
- **P2P events**: Send, Recv, AllGather, AllToAll operations
- **ProxyOp events**: Network proxy operations (ScheduleSend, ScheduleRecv, ProgressSend, ProgressRecv)
- **ProxyStep events**: Detailed network steps (SendWait, RecvWait, GPU waits)
- **ProxyCtrl events**: Proxy thread control (Append, Sleep)
- **GPU events**: Kernel channel execution

### Step 1: Create Virtual Environment

Create a Python virtual environment to isolate the test dependencies:

```bash
python3 -m venv venv
```

### Step 2: Activate Virtual Environment

Activate the virtual environment using the appropriate command for your shell:

**On Linux/Mac (bash/zsh):**
```bash
source venv/bin/activate
```

Once activated, you should see `(venv)` at the beginning of your command prompt.

### Step 3: Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Running Tests

### Run All Tests

To run the entire test suite:

```bash
pytest --cache-clear
```

### Run Tests with Verbose Output

For more detailed test output:

```bash
pytest -v --cache-clear
```

### Run Tests by Marker

Tests are organized using pytest markers. You can run specific groups of tests:

**Run plugin-specific tests:**
```bash
pytest -m ext_tuner --cache-clear      # CSV Tuner Plugin tests only
pytest -m ext_profiler --cache-clear   # Profiler Plugin tests only
```

**Run tests for specific collective operations (across all plugins):**
```bash
pytest -m allreduce --cache-clear      # AllReduce tests
pytest -m broadcast --cache-clear      # Broadcast tests
pytest -m reduce --cache-clear         # Reduce tests
pytest -m allgather --cache-clear      # AllGather tests
pytest -m reducescatter --cache-clear  # ReduceScatter tests
pytest -m alltoall --cache-clear       # AllToAll tests (profiler only)
pytest -m sendrecv --cache-clear       # SendRecv tests (profiler only)
```

**Combine markers to run specific tests:**
```bash
pytest -m "ext_profiler and allreduce" --cache-clear   # Profiler AllReduce tests only
pytest -m "ext_tuner and broadcast" --cache-clear      # Tuner Broadcast tests only
```

### Run Tests with Log Output

To see live log output during test execution:

```bash
pytest -v -s --cache-clear
```

### Generate Test Report

To generate a detailed test report:

```bash
pytest --verbose --tb=short
```

## Additional Notes

- **Deactivating Virtual Environment**: When you're done testing, deactivate the virtual environment by running:
  ```bash
  deactivate
  ```

- **Log Files**: Test execution logs are stored in the `logs/` directory for later review and debugging.

- **Profiler Output**: Profiler plugin tests generate JSON trace files in the `profiler_dumps/` directory. These files contain detailed event traces that can be analyzed for debugging or performance analysis. The directory is automatically cleaned before each test session by the pytest fixture.
