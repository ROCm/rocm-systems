# Roofline Test Programs

This directory contains test programs for the `calc_roofline_data()` function from `analysis_db.py`.

## Files

- **test_roofline.py** - Python test module that calls `calc_roofline_data()` with dummy data
- **test_roofline.cpp** - C++ program that embeds Python and calls the Python test
- **test_roofline_Makefile** - Makefile for building the C++ program
- **test_roofline_CMakeLists.txt** - CMake configuration for building the C++ program

## Python Test

### Description
The Python module creates dummy data structures and calls the `calc_roofline_data()` method to calculate roofline performance metrics.

### Usage
```bash
# From rocprofiler-compute root directory
python3 test_roofline/test_roofline.py

# Or from test_roofline directory
cd test_roofline
python3 test_roofline.py
```

### Output
Returns a dictionary mapping workload paths to pandas DataFrames containing:
- `kernel_name`: Name of the kernel
- `total_flops`: Total floating point operations (GFLOPs)
- `l1_cache_data`: Arithmetic intensity at L1 cache level (FLOPS/byte)
- `l2_cache_data`: Arithmetic intensity at L2 cache level (FLOPS/byte)
- `hbm_cache_data`: Arithmetic intensity at HBM level (FLOPS/byte)

## C++ Test

### Description
The C++ program embeds the Python interpreter, imports the `test_roofline` module, calls the `main()` function, and converts the returned Python data structures to native C++ types.

### Building

#### Option 1: Using Makefile
```bash
# From test_roofline directory
cd test_roofline

# Build the executable
make -f test_roofline_Makefile

# Build and run (runs from root directory)
make -f test_roofline_Makefile run

# Clean
make -f test_roofline_Makefile clean
```

#### Option 2: Using CMake
```bash
# From test_roofline directory
cd test_roofline

# Create build directory
mkdir build && cd build

# Configure
cmake -DCMAKE_BUILD_TYPE=Release -Stest_roofline_CMakeLists.txt ..

# Build
cmake --build .

# Run from root directory
cd ../..
./test_roofline/build/test_roofline
```

#### Option 3: Direct compilation
```bash
# From test_roofline directory
cd test_roofline
g++ -std=c++17 -I/usr/include/python3.10 -o test_roofline_cpp test_roofline.cpp -lpython3.10

# The binary works from either location:
./test_roofline_cpp              # From test_roofline/
cd .. && ./test_roofline/test_roofline_cpp  # From root
```

### C++ Data Structures

The program converts Python data to the following C++ structures:

```cpp
struct RooflineKernelData {
    std::string kernel_name;
    double total_flops;
    double l1_cache_data;
    double l2_cache_data;
    double hbm_cache_data;
};

using RooflineResult = std::map<std::string, std::vector<RooflineKernelData>>;
```

### Requirements

- Python 3.10 development headers
- C++17 compatible compiler
- Python packages: pandas, numpy (installed in the environment)

## Example Output

```
================================================================================
C++ Roofline Results
================================================================================

Workload: /tmp/dummy_workload
Number of kernels: 2

Kernel Name         Total FLOPs    AI L1          AI L2          AI HBM
--------------------------------------------------------------------------------
test_kernel_2       1000.0         33.3           50.0           100.0
test_kernel_1       1000.0         33.3           50.0           100.0

================================================================================
SUCCESS: C++ program completed successfully
================================================================================
```

## Notes

- The Python test uses dummy data with hardcoded roofline metric expressions
- The C++ program initializes the Python interpreter in-process
- Both programs must be run from the rocprofiler-compute root directory
- The `src/` directory must be in the Python import path
