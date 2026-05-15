# RCCL Test Runner

A Python-based test runner focused on RCCL unit and functional tests with hierarchical configuration support and integrated code coverage reporting. Extensible to support performance benchmarks, MPI tests, and custom test scripts.

## Overview

This test runner provides a maintainable, extensible alternative to shell-based test execution. It uses JSON configuration files with hierarchical inheritance, and integrates with LLVM code coverage tools.

## Key Features

- **Multiple Test Types**: Support for GTest, performance tests, and custom executables
- **Hierarchical Configuration**: Use `"extends"` directive to inherit and merge configurations
- **Environment Variable Management**: Global, configuration, suite, and test-specific environment variables
- **Path Variable Expansion**: Use environment variables in paths with nested default value expansion
- **Custom Library Support**: Use pre-built RCCL libraries from custom locations via environment variables
- **Configurable Build System**: Customize CMake options, environment variables, and parallel jobs via config
- **MPI Support**: Full support for multi-rank and multi-node tests
- **Flexible Test Filtering**: Run all tests, specific test suites, or individual tests
- **Build Integration**: Automated RCCL building with CMake
- **Code Coverage**: Integrated LLVM coverage report generation (HTML and text)
- **Clean Output**: Automatic filtering of MPI verbose messages (enable with --verbose)
- **Verbose Logging**: Detailed output for debugging and troubleshooting

## Quick Start

### Basic Usage

```bash
# Run with specific configuration
python test_runner.py --config my_tests.json

# Run with verbose output
python test_runner.py --config my_tests.json --verbose

# Run specific test by name
python test_runner.py --config my_tests.json --test-name SHM_ComprehensiveWorkflow
```

### Generate Coverage Report

```bash
# Build, run tests, and generate coverage report
python test_runner.py --config test_config_sample.json --coverage-report --verbose

# Use existing build and generate coverage
python test_runner.py --config test_config_sample.json --no-build --coverage-report
```

### Use Custom RCCL Library

```bash
# Use pre-built RCCL library from custom location
export RCCL_LIB_PATH=/path/to/custom/rccl/build
python test_runner.py --config test_config_sample.json

# Or use RCCL_BUILD_DIR (alternative name)
export RCCL_BUILD_DIR=/path/to/custom/rccl/build
python test_runner.py --config test_config_sample.json

# When set, build step is automatically skipped
# --no-build is not needed
```

## Environment Variables

The test runner supports the following environment variables to customize behavior:

### Library and Build Configuration

| Variable | Description | Example |
|----------|-------------|---------|
| `RCCL_LIB_PATH` | Path to pre-built RCCL library directory (contains `librccl.so` and `test/` subdirectory). When set, the build step is automatically skipped. | `/path/to/rccl/build` |
| `RCCL_BUILD_DIR` | Alternative name for `RCCL_LIB_PATH`. Either variable can be used. | `/path/to/rccl/build` |
| `RCCL_TEST_MPI_HOSTFILE` | Path to MPI hostfile for multi-node tests. | `~/.mpi_hostfile` |

### Configuration Path Variables

These can be overridden via environment variables or specified in the JSON config:

| Variable | Description | Default |
|----------|-------------|---------|
| `WORKDIR` | RCCL source and build directory | Current rccl repository root |
| `ROCM_PATH` | ROCm installation path | `/opt/rocm` |
| `MPI_PATH` | MPI installation path | System default or config-specific |

### Priority Order

When determining which RCCL library to use, the test runner follows this priority:

1. **`RCCL_LIB_PATH` or `RCCL_BUILD_DIR` environment variable** (highest priority)
   - Skips build automatically
   - Must contain `librccl.so` and `test/` subdirectory
2. **`--no-build` flag with local build**
   - Uses local `build_debug_cov_on_tests_on/` directory
   - Requires prior build
3. **Default build process** (lowest priority)
   - Builds RCCL in timestamped directory
   - Uses CMake configuration from JSON

**Example Usage:**

```bash
# Priority 1: Use custom library (build skipped automatically)
export RCCL_LIB_PATH=/path/to/prebuilt/rccl/build
python test_runner.py --config my_tests.json

# Priority 2: Use existing local build (no new build)
python test_runner.py --config my_tests.json --no-build

# Priority 3: Fresh build (default)
python test_runner.py --config my_tests.json
```

## Configuration File Format

### Basic Structure

```json
{
  "system_configurations": {
    "name": "system-name",
    "description": "System description"
  },
  "paths": {
    "workdir": "/path/to/rccl",
    "rocm_path": "/opt/rocm",
    "mpi_path": "/path/to/mpi"
  },
  "env_variables": {
    "GLOBAL_VAR": "value"
  },
  "test_configurations": {
    "config_name": {
      "env_variables": {...},
      "tests": [...]
    }
  },
  "test_suites": [
    {
      "name": "Test Suite Name",
      "config": "config_name",
      "enabled": true
    }
  ]
}
```

### Environment Variable Expansion in Paths

The `paths` section supports environment variable expansion, allowing you to avoid hardcoding paths and make configurations portable across different systems.

#### Supported Syntax

```json
{
  "paths": {
    "workdir": "${HOME}/code/rccl",
    "rocm_path": "$ROCM_PATH",
    "mpi_path": "${MPI_PATH:-/opt/mpi}"
  }
}
```

**Syntax Options:**
- `${VAR}` - Expands to the value of `VAR`, left as-is if undefined
- `$VAR` - Expands to the value of `VAR`, left as-is if undefined
- `${VAR:-default}` - Expands to the value of `VAR`, or `default` if undefined (bash-style default)

#### Examples

```json
{
  "paths": {
    "workdir": "${WORKDIR:-${HOME}/code/rti/scripts/rccl}",
    "rocm_path": "${ROCM_PATH:-/opt/rocm}",
    "mpi_path": "${MPI_PATH:-${HOME}/softwares/ompi}"
  }
}
```

**Usage:**
```bash
# Use environment variables
export WORKDIR=/custom/path/to/rccl
export ROCM_PATH=/opt/rocm-6.0
export MPI_PATH=/usr/local/mpi

python test_runner.py --config test_config_sample.json

# Or use defaults (no environment variables set)
python test_runner.py --config test_config_sample.json
```

**Benefits:**
- **Portability**: Share configurations across different systems
- **Flexibility**: Override paths without modifying config files
- **CI/CD**: Easy integration with build systems and pipelines
- **Multi-user**: Same config works for different user environments

### Test Types Supported

The test runner uses the `is_gtest` boolean flag to distinguish between test types:

- **`is_gtest: true`** (default) - GTest-based unit tests using `--gtest_filter` syntax
- **`is_gtest: false`** - Non-GTest tests (performance benchmarks, custom scripts, etc.)

This simplified approach supports all test categories while reducing configuration complexity.

#### GTest Tests (`is_gtest: true`)

Used for unit tests with GTest framework. The `test_filter` field uses GTest filter syntax.

```json
{
  "name": "AllReduce_InPlace",
  "description": "Test AllReduce collective operation with in-place buffers",
  "is_gtest": true,
  "binary": "rccl-UnitTests",
  "test_filter": "AllReduce.InPlace",
  "num_ranks": 1,
  "num_nodes": 1,
  "timeout": 60
}
```

**Command generated:**
```bash
./rccl-UnitTests --gtest_filter=AllReduce.InPlace
```

#### Performance Tests (`is_gtest: false`)

Used for performance benchmarks. Arguments are passed directly without GTest syntax.

```json
{
  "name": "Perf_Bandwidth",
  "description": "Bandwidth benchmark for AllReduce",
  "is_gtest": false,
  "binary": "all_reduce_perf",
  "command_args": "-b 8 -e 128M -f 2",
  "num_ranks": 2,
  "num_nodes": 1,
  "timeout": 300
}
```

**Command generated:**
```bash
mpirun -np 2 ./all_reduce_perf -b 8 -e 128M -f 2
```

#### Custom Scripts (`is_gtest: false`)

Used for custom validation scripts or any non-GTest executables.

```json
{
  "name": "Custom_Validation",
  "description": "Custom GPU validation script",
  "is_gtest": false,
  "binary": "validate_gpus.sh",
  "command_args": "--full-check --verbose",
  "num_ranks": 1,
  "num_nodes": 1,
  "timeout": 120
}
```

**Command generated:**
```bash
./validate_gpus.sh --full-check --verbose
```

**Key Differences:**

| Feature | `is_gtest: true` | `is_gtest: false` |
|---------|------------------|-------------------|
| Test framework | GTest (Google Test) | Any executable |
| Filter syntax | `--gtest_filter=<pattern>` | Plain arguments |
| `test_filter` field | GTest pattern (e.g., `Suite.Test*`) | Passed as plain argument |
| `command_args` field | Appended after filter | Primary argument method |
| Typical use cases | Unit tests, functional tests | Performance tests, custom scripts |

### Test Definition Fields

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `name` | Yes | string | Unique test identifier |
| `description` | Recommended | string | Human-readable test description |
| `is_gtest` | Optional | boolean | Whether test uses GTest framework (default: true). Set to false for perf or custom tests |
| `binary` | Yes | string | Test binary name (relative to build/test/) |
| `test_filter` | Optional | string | Test filter (GTest filter syntax for gtest, plain argument for non-gtest) |
| `command_args` | Optional | string | Additional command-line arguments |
| `num_ranks` | Optional | integer | Number of MPI ranks (default: 1) |
| `num_nodes` | Optional | integer | Number of nodes (default: 1) |
| `num_gpus` | Optional | integer | GPUs per node - controls rank distribution (default: 8) |
| `timeout` | Optional | integer | Timeout in seconds (0 = unlimited) |
| `env_variables` | Optional | object | Test-specific environment variables |

### Configuration Inheritance

Use the `"extends"` directive to inherit from parent configurations:

```json
{
  "test_configurations": {
    "base": {
      "env_variables": {
        "NCCL_DEBUG": "INFO"
      }
    },
    "shm_tests": {
      "extends": "base",
      "env_variables": {
        "NCCL_SHM_DISABLE": "0"
      },
      "tests": [...]
    },
    "advanced_shm": {
      "extends": ["base", "shm_tests"],
      "env_variables": {
        "NCCL_SHM_USE_CUDA_MEMCPY": "1"
      }
    }
  }
}
```

### Hierarchical Defaults

To reduce repetition, you can specify default values at multiple levels with a clear override hierarchy:

**Priority Order (highest to lowest):**
1. **Individual test** - highest priority, overrides everything
2. **Test suite level** - overrides configuration defaults
3. **Configuration level** - base defaults for all tests in that config
4. **Built-in defaults** - system fallback values

**Supported default fields:** `is_gtest`, `binary`, `num_ranks`, `num_nodes`, `num_gpus`, `timeout`

#### Example with Three-Level Hierarchy

```json
{
  "test_configurations": {
    "p2p_tests": {
      "is_gtest": true,
      "binary": "rccl-UnitTestsMPI",
      "num_ranks": 2,
      "num_nodes": 1,
      "num_gpus": 2,
      "timeout": 120,
      "env_variables": {
        "NCCL_P2P_DISABLE": "0"
      },
      "tests": [
        {
          "name": "P2P_Basic",
          "description": "Basic P2P test",
          "test_filter": "P2pMPITest.Basic"
          // Uses config defaults: is_gtest=true, binary, num_ranks=2, num_nodes=1, num_gpus=2, timeout=120
        },
        {
          "name": "P2P_LongRunning",
          "description": "Long-running P2P test",
          "test_filter": "P2pMPITest.LongRunning",
          "timeout": 300
          // Overrides timeout=300, inherits other config defaults
        }
      ]
    }
  },
  "test_suites": [
    {
      "name": "P2P_Basic_Suite",
      "config": "p2p_tests",
      "num_ranks": 4,
      "num_gpus": 4,
      "timeout": 180
      // Suite-level: overrides config's num_ranks, num_gpus, and timeout
      // Tests in this suite will use: num_ranks=4, num_gpus=4, timeout=180
    },
    {
      "name": "P2P_Stress_Suite",
      "config": "p2p_tests",
      "num_nodes": 2,
      "num_ranks": 4,
      "num_gpus": 2,
      "timeout": 600
      // Suite-level: overrides config's num_nodes, num_ranks, num_gpus, and timeout
      // Tests in this suite will use: num_nodes=2, num_ranks=4, num_gpus=2, timeout=600
    }
  ]
}
```

**Benefits:**
- **Less Repetition**: Define common values once
- **Easier Maintenance**: Update defaults in one place
- **Flexible Overrides**: Tests can still customize any field
- **Cleaner Config**: Shorter, more readable test definitions

## Command-Line Options

```
Required:
  -c, --config CONFIG       Test configuration file (JSON format)

Optional:
  -v, --verbose             Enable verbose output (shows build paths, commands, etc.)
  -o, --output DIR          Output directory for logs and reports
  --test-name NAME          Run only specific test by name
  --no-build                Skip build step and use existing build
  --skip-tests              Skip test execution (useful with --coverage-report)
  --coverage-report         Generate code coverage report (HTML + text)
  --overwrite               Overwrite previous workspace directories
  --report-suffix SUFFIX    Suffix for report directory (default: blank)
  -h, --help                Show help message and exit
```

## Code Coverage Reports

The test runner integrates with LLVM tools to generate comprehensive code coverage reports.

### Generating Coverage

```bash
# Build and test with coverage (recommended)
python test_runner.py --config test_config_sample.json --coverage-report --verbose

# Generate report from existing profraw files
python test_runner.py --config test_config_sample.json --no-build --skip-tests --coverage-report
```

### Coverage Output

When `--coverage-report` is specified, the runner generates:

1. **HTML Report**: Visual coverage report in `reports/` directory
   - View with: `firefox reports/index.html`
   - Shows line-by-line coverage with syntax highlighting

2. **Text Report**: Function-level coverage summary
   - Location: `reports/function_coverage_report.txt`
   - Includes per-function and per-file statistics

### Coverage Implementation Details

- Uses LLVM instrumentation (`-fprofile-instr-generate -fcoverage-mapping`)
- Collects `.profraw` files during test execution
- Merges profiles with `llvm-profdata`
- Generates reports with `llvm-cov show` and `llvm-cov report`
- Filters out irrelevant files (test/, gtest, external dependencies)

## Examples

### Run All Enabled Test Suites

```bash
python test_runner.py --config test_config_sample.json --verbose
```

### Run Specific Test

```bash
python test_runner.py --config test_config_sample.json --test-name P2P_AllTests
```

### Skip Build (Use Existing)

```bash
python test_runner.py --config test_config_sample.json --no-build
```

### Build and Generate Coverage

```bash
# Full workflow: build, test, coverage
python test_runner.py --config adhoc_test_config.json --coverage-report --verbose
```

### Generate Coverage from Existing Build

```bash
# Skip build, use existing profraw files
python test_runner.py --config adhoc_test_config.json --no-build --skip-tests --coverage-report
```

### Custom Output Directory

```bash
python test_runner.py --config test_config_sample.json -o /path/to/output --verbose
```

### Run with Overwrite (Clean Previous Results)

```bash
python test_runner.py --config test_config_sample.json --overwrite --coverage-report
```

## Environment Variable Merging

Environment variables are merged hierarchically (later values override earlier):

1. **Global** `env_variables` (top-level in config)
2. **Configuration** `env_variables` (test configuration level)
3. **Test Suite** `env_variables` (suite level)
4. **Test-specific** `env_variables` (individual test level)

Example:
```json
{
  "env_variables": {
    "NCCL_DEBUG": "INFO"
  },
  "test_configurations": {
    "shm_tests": {
      "env_variables": {
        "NCCL_SHM_DISABLE": "0"
      },
      "tests": [
        {
          "name": "SHM_Test",
          "env_variables": {
            "NCCL_DEBUG": "TRACE"
          }
        }
      ]
    }
  }
}
```

Result: `NCCL_DEBUG=TRACE`, `NCCL_SHM_DISABLE=0`

## Test Execution

### Single-Node Tests

- All ranks run on a single node
- Multiple ranks map to different GPUs
- Examples: SHM tests, P2P tests, unit tests

```json
{
  "name": "SHM_Test",
  "num_ranks": 2,
  "num_nodes": 1
}
```

### Multi-Node Tests

- Ranks distributed across multiple nodes via MPI
- Requires SLURM allocation or hostfile configuration
- Use `num_gpus` to control ranks per node (default: 8)
- Examples: NET transport tests, InfiniBand tests

```json
{
  "name": "NET_Test_4Nodes_2GPUs",
  "num_ranks": 8,
  "num_nodes": 4,
  "num_gpus": 2
}
```

**`num_gpus` Field:**
- Controls how many MPI ranks are placed on each node
- Overrides hostfile `slots` specification
- For multi-node tests, uses `--map-by ppr:{num_gpus}:node`
- Default value: 8 (matches typical 8-GPU nodes)

**Example: 2 nodes, 1 GPU per node**
```json
{
  "name": "NET_Test_2Nodes_1GPU",
  "num_ranks": 2,
  "num_nodes": 2,
  "num_gpus": 1
}
```
Command: `mpirun -np 2 --hostfile file --map-by ppr:1:node ...`

### Setting Up Multi-Node Tests

**Option 1: MPI Hostfile**
```bash
export RCCL_TEST_MPI_HOSTFILE=/path/to/hostfile
python test_runner.py --config net_ib_test_config.json
```

**Option 2: Default Hostfile**
Create `~/.mpi_hostfile` with node names (one per line):
```
node01 slots=8
node02 slots=8
```

## Advanced Features

### Build Configuration (New!)

Customize the RCCL build process through the `build_configuration` section in your JSON config file.

#### Basic Structure

```json
{
  "build_configuration": {
    "cmake_options": {
      "CMAKE_BUILD_TYPE": "Debug",
      "ENABLE_CODE_COVERAGE": "ON",
      "ONLY_FUNCS": "SendRecv|AllReduce"
    },
    "env_variables": {
      "HIPCC_COMPILE_FLAGS_APPEND": "-g -O1"
    },
    "parallel_jobs": 64,
    "generator": "Unix Makefiles"
  }
}
```

#### Examples

**Fast Development Build (No Coverage):**
```json
{
  "build_configuration": {
    "cmake_options": {
      "ENABLE_CODE_COVERAGE": "OFF"
    },
    "parallel_jobs": 128
  }
}
```

**Release Build:**
```json
{
  "build_configuration": {
    "cmake_options": {
      "CMAKE_BUILD_TYPE": "Release",
      "TRACE": "OFF",
      "COLLTRACE": "OFF"
    }
  }
}
```

**Test Specific Functions Only:**
```json
{
  "build_configuration": {
    "cmake_options": {
      "ONLY_FUNCS": "Broadcast|Reduce"
    }
  }
}
```

**All Options:**
- `cmake_options` - Any CMake option (user values override defaults)
- `env_variables` - Build environment variables
- `parallel_jobs` - Number of parallel build threads (default: 64)
- `generator` - CMake generator: "Unix Makefiles", "Ninja", etc.

See `BUILD_CONFIGURATION_GUIDE.md` for complete documentation.

### Enhanced Environment Variable Expansion

Environment variables in the `paths` section now support **nested expansion** in default values:

```json
{
  "paths": {
    "workdir": "${WORKDIR:-$HOME/code/rti/scripts/rccl}",
    "rocm_path": "${ROCM_PATH:-/opt/rocm}",
    "mpi_path": "${MPI_PATH:-$HOME/softwares/ompi}"
  }
}
```

**Key Feature:** If `WORKDIR` is not set, the default `$HOME/code/rti/scripts/rccl` will expand `$HOME` automatically!

### Flexible Binary Paths

Specify test binary locations in multiple ways for maximum flexibility:

#### 1. Default (Relative to build_dir/test/)

```json
{
  "binary": "all_reduce_perf"
}
```
Result: `<workdir>/build_debug_cov_on_tests_on/test/all_reduce_perf`

#### 2. Absolute Path

```json
{
  "binary": "/opt/custom_rccl_build/test/all_reduce_perf"
}
```
Result: Uses the absolute path directly

#### 3. Environment Variable in Binary Name

```json
{
  "binary": "${MY_RCCL_TESTS}/all_reduce_perf"
}
```
Result: Expands `$MY_RCCL_TESTS` environment variable

#### 4. Home Directory Expansion

```json
{
  "binary": "~/my_builds/rccl/test/all_reduce_perf"
}
```
Result: Expands `~` to home directory

#### 5. Using test_binary_dir in Paths

```json
{
  "paths": {
    "test_binary_dir": "${RCCL_TEST_BIN_DIR}"
  },
  "test_configurations": {
    "my_tests": {
      "binary": "all_reduce_perf"
    }
  }
}
```
Result: `${RCCL_TEST_BIN_DIR}/all_reduce_perf`

#### 6. Using test_binary_dir in Test Config

```json
{
  "test_configurations": {
    "my_tests": {
      "tests": [
        {
          "name": "CustomBinary",
          "test_binary_dir": "/opt/rccl/tests",
          "binary": "all_reduce_perf"
        }
      ]
    }
  }
}
```
Result: `/opt/rccl/tests/all_reduce_perf`

#### Resolution Priority Order

1. **Absolute path in binary** - Highest priority
2. **Environment variable expansion** (if results in absolute path)
3. **test_binary_dir in test config** + binary
4. **test_binary_dir in paths** + binary
5. **Default:** `build_dir/test/` + binary - Lowest priority

#### Use Cases

- **CI/CD with pre-built binaries:** Use absolute paths or `RCCL_TEST_BIN_DIR`
- **Multiple RCCL versions:** Different `test_binary_dir` per configuration
- **Custom build locations:** Environment variables for flexibility
- **Standard builds:** Use default (no configuration needed)

#### Verbose Mode

Use `--verbose` to see the resolved binary path:
```bash
python test_runner.py --config test.json --verbose
```

Output includes:
```
Binary:  all_reduce_perf
Binary path: /home/user/code/rti/scripts/rccl/build_debug_cov_on_tests_on/test/all_reduce_perf
```

### Configuration Best Practices

**Reduce Repetition:** Move common values to configuration level

```json
{
  "test_configurations": {
    "p2p_tests": {
      "timeout": 120,
      "env_variables": {
        "NCCL_P2P_USE_CUDA_MEMCPY": "1",
        "NCCL_LEGACY_CUDA_REGISTER": "1"
      },
      "tests": [
        {
          "name": "Test1"
          // Inherits timeout and env vars from config level
        },
        {
          "name": "Test2",
          "timeout": 300
          // Overrides timeout, inherits env vars
        }
      ]
    }
  }
}
```

**Benefits:**
- ✅ Single source of truth for common settings
- ✅ Easier maintenance
- ✅ Tests can still override when needed
- ✅ Cleaner, more readable configurations

## Development and Testing

### Validate Configuration

```bash
# Test JSON syntax
python3 -m json.tool test_config_sample.json

# Test configuration loading
python3 -c "from lib.test_config import TestConfigProcessor; \
            p = TestConfigProcessor('test_config_sample.json'); \
            print('Configuration valid!')"

# Dry run (validate without executing)
python test_runner.py --config test_config_sample.json --skip-tests --verbose
```

### Adding New Tests

1. Add test definition to appropriate configuration in JSON file
2. Specify `is_gtest`, `description`, and required fields
3. Test with dry run first: `--skip-tests --verbose`
4. Run actual test: `--test-name YourTest --verbose`

### Test Type Handling

The test runner uses a boolean `is_gtest` flag to distinguish between test types:

- **`is_gtest: true`** (default): Uses GTest framework with `--gtest_filter=<filter>` syntax
- **`is_gtest: false`**: Runs binary with plain arguments (for performance tests, custom scripts, etc.)

This simplified approach eliminates the need for multiple test type conditionals while supporting all test categories (gtest, perf, custom).

## Troubleshooting

### "Configuration file not found"
- Check the path to your JSON config file
- Use absolute paths or ensure you're in the correct directory
- Verify file permissions

### "MPI path not found"
- Update `paths.mpi_path` in your configuration
- Ensure MPI is installed: `which mpirun`
- Check MPI_PATH environment variable

### "Test binary not found"
- Build first: remove `--no-build` flag
- Check binary name in `build/test/` directory
- Verify CMAKE built successfully

### Multi-node tests hang
- Ensure SLURM allocation or hostfile is configured
- Check network connectivity: `ping other_node`
- Verify MPI can reach nodes: `mpirun -np 2 hostname`
- Check firewall settings

### CMake configuration fails
- Check ROCm path: `ls $ROCM_PATH`
- Verify compiler: `$ROCM_PATH/bin/amdclang++ --version`
- Check MPI path: `ls $MPI_PATH/bin/mpirun`

### Coverage report fails
- Ensure LLVM tools are available: `which llvm-profdata llvm-cov`
- Check for `.profraw` files in build directory
- Verify coverage build flags were set correctly
- Run with `--verbose` to see detailed error messages

### "LLVM_PROFILE_FILE not being used"
- Ensure `--coverage-report` flag is specified
- Check that tests are actually executing (not skipped)
- Verify environment variables with `--verbose`

---

## Appendix: Environment Variables Reference

This section provides a quick reference for all environment variables supported by the test runner.

### Library and Build Location

| Variable | Description | Example |
|----------|-------------|---------|
| `RCCL_LIB_PATH` | Path to pre-built RCCL library directory. Automatically skips build. | `export RCCL_LIB_PATH=/path/to/rccl/build` |
| `RCCL_BUILD_DIR` | Alternative name for `RCCL_LIB_PATH`. | `export RCCL_BUILD_DIR=/home/user/rccl_builds/debug` |

**Requirements**: Directory must contain `librccl.so` and `test/` subdirectory.

### Configuration Paths

These override the paths specified in the JSON configuration file:

| Variable | Description | Example |
|----------|-------------|---------|
| `WORKDIR` | RCCL source and build directory | `export WORKDIR=/home/user/code/rccl` |
| `ROCM_PATH` | ROCm installation path | `export ROCM_PATH=/opt/rocm-6.0` |
| `MPI_PATH` | MPI installation path | `export MPI_PATH=/usr/local/openmpi` |

### Test Execution

| Variable | Description | Example |
|----------|-------------|---------|
| `RCCL_TEST_MPI_HOSTFILE` | Path to MPI hostfile for multi-node tests | `export RCCL_TEST_MPI_HOSTFILE=~/.mpi_hostfile` |

**Note**: Falls back to `~/.mpi_hostfile` if not set. For SLURM environments, hostfile is auto-generated from `SLURM_NODELIST`.

### Test-Specific Variables

These can be set globally or specified in the JSON configuration per test:

| Variable | Description | Example |
|----------|-------------|---------|
| `NCCL_DEBUG` | NCCL debug level (VERSION, WARN, INFO, TRACE) | `export NCCL_DEBUG=INFO` |
| `NCCL_DEBUG_SUBSYS` | NCCL debug subsystems to enable | `export NCCL_DEBUG_SUBSYS=INIT,COLL,NET` |
| `HSA_NO_SCRATCH_RECLAIM` | Disable HIP scratch memory reclaim | `export HSA_NO_SCRATCH_RECLAIM=1` |
| `NCCL_LAUNCH_MODE` | NCCL launch mode (GROUP, PARALLEL) | `export NCCL_LAUNCH_MODE=GROUP` |

### Coverage and Profiling

| Variable | Description | Example |
|----------|-------------|---------|
| `LLVM_PROFILE_FILE` | LLVM coverage profile output pattern | `export LLVM_PROFILE_FILE=rccl_%p_%m.profraw` |

**Note**: Automatically set by test runner to prevent collisions. Manual override not recommended.

### Complete Example

```bash
#!/bin/bash
# Configure paths
export WORKDIR=/home/user/code/rccl
export ROCM_PATH=/opt/rocm-6.0
export MPI_PATH=/usr/local/openmpi

# Use pre-built library
export RCCL_LIB_PATH=/home/user/rccl_builds/instrumented

# Configure MPI
export RCCL_TEST_MPI_HOSTFILE=~/.mpi_hostfile

# Enable debug output
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL,NET

# Run tests
python test_runner.py --config my_tests.json --verbose
```

### Variable Priority

When the same configuration can be specified in multiple places, the priority is:

1. **Environment variables** (highest priority)
2. **Test-specific configuration** (in JSON)
3. **Test suite configuration** (in JSON)
4. **Test configuration defaults** (in JSON)
5. **Built-in defaults** (lowest priority)

**Example**: If `ROCM_PATH` is set as an environment variable, it overrides the `rocm_path` value in the JSON configuration file.

