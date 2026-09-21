Hacking and Debugging RocSHMEM
==============================

This documentation is mostly for core RocSHMEM developers and contributors. Power users may still find it useful.

How to debug parallel programs
------------------------------

When using Open MPI as the launch  mechanism, you can use `OMPI_MCA_mpi_abort_delay=-1` to keep parallel processes active after a crash. You can then use `ssh -t $nodename rocgdb -p $pid` to connect a `rocgdb` to the failed process. Sometimes errors are caught by UCX, `UCX_HANDLE_ERRORS=freeze` will have the same effect in such cases.

Look into `scripts/functional_tests/GDB_README` about an alternative technique that deploys multiple `xterm` to gdb into parallel processes.

How to use the address sanitizer (ASAN)
---------------------------------------

Refer to [General documentation for ASAN on AMD GPUs][1].

### Compiling with ASAN

Add `-DASAN=ON` to the `cmake` invocation, whether configuring a fresh build directory or reconfiguring an existing one.
  `cmake . <...> -DASAN=ON`

`xnack-` targets are silently invalid under ASAN (the compiler skips instrumenting device code for them without a hard error), so rocSHMEM automatically drops any `:xnack-` entries from `GPU_TARGETS` when `ASAN=ON`, and re-derives HIP's offload targets from the result on every configure. Simply toggling `-DASAN=ON`/`-DASAN=OFF` on an existing build directory is enough; no manual CMake cache surgery is required.

### Running with ASAN

You need to set environment variable `HSA_XNACK=1`. Do not forget to unset this variable when not using ASAN (it will impact performance).

You may need to add the path to `libclang_rt.asan-x86_64.so` into `LD_LIBRARY_PATH` by hand. Depending on the ROCm version, it may be in an unusual place, e.g., `$ROCM_ROOT/lib/llvm/lib/clang/19/lib/linux/libclang_rt.asan-x86_64.so`; `find /opt/rocm -name libclang_rt.asan-x86_64.so` may be required to find it.

ASAN may [crash when using Open MPI][2]. If that happens, set environment variable `OMPI_MCA_memory=^patcher`. Do not forget to unset this variable when not using ASAN (it will impact performance).

When running the program, the behavior of ASAN can be controlled with the `ASAN_OPTIONS` environment variable.

you can redude the amount of spurious leak reports by setting environment variable `LSAN_OPTIONS=suppressions=$ROCSHEM_SRC/scripts/lsan-suppressions.txt`.

### References

[1]: https://rocm.docs.amd.com/projects/llvm-project/en/docs-6.4.0/conceptual/using-gpu-sanitizer.html
[2]: https://github.com/open-mpi/ompi/issues/13069
