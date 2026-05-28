# RCCL CSV Tuner Plugin Tests

## Description

This directory contains automated tests for the RCCL (ROCm Communication Collectives Library) CSV Tuner Plugin. The test suite validates the functionality of the CSV-based tuning plugin across different collective operations (AllReduce, Broadcast, Reduce, AllGather, and ReduceScatter) and various configuration scenarios.

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
    └── ext-tuner/               # CSV Tuner Plugin specific tests
        ├── test_allreduce.py
        ├── test_broadcast.py
        ├── test_reduce.py
        ├── test_allgather.py
        └── test_reducescatter.py
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

### Building the RCCL CSV Tuner Plugin

Before running the tests, you need to build the RCCL CSV tuner plugin library. The plugin is located in the `ext-tuner/example` directory.

#### Step 1: Navigate to the plugin directory

```bash
cd rccl/ext-tuner/example
```

#### Step 2: Build the plugin

```bash
make
```

This will compile the plugin and create `libnccl-tuner-example.so` in the same directory.

### Step 1: Create Virtual Environment

Create a Python virtual environment to isolate the test dependencies:

```bash
python3 -m venv venv
```

### Step 2: Activate Virtual Environment

Activate the virtual environment using the appropriate command for your shell:

**On Linux:**
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

**Run CSV Plugin tests:**
```bash
pytest -m mark.ext_tuner --cache-clear
```

**Run tests for specific collective operations:**
```bash
pytest -m allreduce --cache-clear      # AllReduce tests
pytest -m broadcast --cache-clear      # Broadcast tests
pytest -m reduce --cache-clear         # Reduce tests
pytest -m allgather --cache-clear      # AllGather tests
pytest -m reducescatter --cache-clear  # ReduceScatter tests
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
