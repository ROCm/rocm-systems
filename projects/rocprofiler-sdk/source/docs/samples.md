# Samples

## Running Samples

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 4
```

Samples and tool can be run in order to see the profiler in action. This section covers on how to build these samples and run the tool.
Once the rocm build is installed, samples are installed under:

```bash
/opt/rocm/share/rocprofiler-sdk/samples
```

rocprofv3 tool is installed under:

```bash
/opt/rocm/bin
```

### Building Samples

From any directory, run:

```bash
cmake -B build-rocprofiler-sdk-samples /opt/rocm/share/rocprofiler-sdk/samples -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build-rocprofiler-sdk-samples --target all --parallel 8


### Running samples

To run the built samples, cd into the `build-rocprofiler-sdk-samples` directory and run:

```bash
ctest -V
