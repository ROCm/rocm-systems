# MPI Ranks Filtering Test

This test validates the `--mpi-ranks` feature of rocprofv3, which allows selective profiling of specific MPI ranks.

## Test Scenarios

### 1. With MPI - Single Rank (`with-mpi-single`)
- Runs with 4 MPI processes using `mpirun -n 4`
- Uses `--mpi-ranks 0` to profile only rank 0
- **Expected behavior**: Only rank 0 should generate profiling output files
- **Validation**: Checks that exactly 1 JSON/CSV output file exists with valid profiling data

### 2. With MPI - Multiple Ranks (`with-mpi-multiple`)
- Runs with 4 MPI processes using `mpirun -n 4`
- Uses `--mpi-ranks 0-1,3` to profile ranks 0, 1, and 3
- **Expected behavior**: Ranks 0, 1, and 3 should generate output; rank 2 should not
- **Validation**: Checks that exactly 3 JSON/CSV output files exist with valid profiling data

### 3. Without MPI (`without-mpi`)
- Runs without MPI (single process)
- Uses `--mpi-ranks 0` but no MPI environment is detected
- **Expected behavior**: Since no MPI environment is detected, rocprofv3 should generate output normally
- **Validation**: Checks that at least 1 JSON/CSV output file exists with valid profiling data

## Graceful Degradation

The test ensures that:
1. **If mpirun is not available**: The MPI-based tests are skipped (not failed)
2. **If only single GPU available**: The test still runs correctly
3. **If no MPI environment detected**: The `--mpi-ranks` flag is gracefully ignored and profiling proceeds normally

## Test Structure

- `CMakeLists.txt`: Defines the test configurations and conditionally includes them based on mpirun availability
- `conftest.py`: pytest configuration providing fixtures for output directory and test mode
- `validate.py`: Main test script that validates the output based on the test mode
- `pytest.ini`: pytest configuration file

## Running the Tests

The tests are automatically run as part of the rocprofiler-sdk test suite:

```bash
# Run all tests
ctest -R rocprofv3-test-mpi-ranks

# Run specific test
ctest -R rocprofv3-test-mpi-ranks-with-mpi-single -VV

# Run without MPI test only
ctest -R rocprofv3-test-mpi-ranks-without-mpi -VV
```

## Requirements

- ROCm with HIP support
- rocprofiler-sdk installed
- MPI implementation (OpenMPI, MPICH, etc.) - optional for MPI tests
- pytest with Python 3
- At least one AMD GPU (tests work with single GPU)
