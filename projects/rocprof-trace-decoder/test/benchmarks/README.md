# Native marker API benchmark

Build the optional benchmark from the repository root:

```sh
cmake -S . -B build-marker-bench -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=ON -DBUILD_PYTHON=OFF
cmake --build build-marker-bench --target marker_benchmark -j
LD_LIBRARY_PATH="build-marker-bench/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  build-marker-bench/test/benchmarks/marker_benchmark
```

Add `-DDISABLE_COMGR=ON -DUSE_LLVM_DISASM=OFF` to benchmark a build without
disassembly. The benchmark uses only public C headers and the shared decoder
library; it never calls internal parser or marker helpers. Its encoders write
the few SQTT token layouts it needs directly.
The Linux command above selects the library from this build even if the shell's
library search path contains another installed decoder.

Each pattern contains roughly 32 MiB of encoded gfx9 or gfx12 SQTT by default
(`--arch` selects one). `points` emits point markers, and `payload130` emits
markers with 130 raw payload words. Both use 16 wave slots spread across four
SIMDs. Payload words are interleaved across every slot. A synthetic code object
supplies marker metadata. These patterns do not contain packed clocks or
instructions requiring disassembly.

Each input is parsed with an empty record filter (`disabled`), raw shaderdata
only, buffered markers, immediate markers, and buffered markers plus raw
shaderdata (`both`). The consumer reads marker IDs, code-object IDs, header
values, and every payload word into a checked checksum. Counts and checksum
must match the generated input or the benchmark fails.

To measure a captured trace instead, pass the shader-engine file and its code
objects; counts are reported but not checked:

```sh
marker_benchmark --trace run/1_2_shader_engine_0_1.att \
  --codeobj 1:run/1_gfx1201_code_object_id_1.out --codeobj 2:run/1_gfx1201_code_object_id_2.out
```

`input_MB/s` divides the actual encoded input bytes, including SQTT headers and
register tokens, by elapsed time. MB means 1,000,000 bytes. `input_bytes` is the
size of one input, and marker/payload/callback counts are totals across the
reported iterations. ELF loading, input generation, filter configuration, and
one warmup parse are excluded from timing. Repeated inputs are resident in
memory; this is a synthetic single-thread measurement rather than a prediction
for every architecture or real trace.

Increase input size or measurement duration, select individual cases, or set an
exact repetition count:

```sh
LD_LIBRARY_PATH="build-marker-bench/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  build-marker-bench/test/benchmarks/marker_benchmark --mib 128 --seconds 1
LD_LIBRARY_PATH="build-marker-bench/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  build-marker-bench/test/benchmarks/marker_benchmark \
    --pattern payload130 --delivery buffered --iterations 10
```

`--seconds` defaults to 0.25 seconds per case. `--iterations` overrides that
minimum. Run `--help` for all options. Record the compiler, build type, CPU, and
command alongside results when comparing revisions.
