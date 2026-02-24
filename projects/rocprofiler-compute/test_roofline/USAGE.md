# Quick Usage Guide

## Python Test

```bash
# From rocprofiler-compute root:
python3 test_roofline/test_roofline.py

# Or from test_roofline directory:
cd test_roofline
python3 test_roofline.py
```

## C++ Test

### Build

```bash
# From test_roofline directory:
cd test_roofline
make -f test_roofline_Makefile
```

### Run

The C++ program works from both locations:

```bash
# From test_roofline directory:
./test_roofline_cpp

# From rocprofiler-compute root:
./test_roofline/test_roofline_cpp

# Or use make run (which runs from root):
cd test_roofline
make -f test_roofline_Makefile run
```

## What it does

Both programs test the `calc_roofline_data()` function from `analysis_db.py` and return roofline performance metrics for GPU kernels:
- Total FLOPs (floating point operations)
- Arithmetic Intensity at L1, L2, and HBM cache levels

See `test_roofline_README.md` for detailed documentation.
