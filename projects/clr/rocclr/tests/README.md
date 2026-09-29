# ROCclr flag parsing tests

Build and run the host-only regression test without a ROCm installation or GPU:

```sh
cmake -S projects/clr/rocclr/tests -B build/rocclr-flags
cmake --build build/rocclr-flags
ctest --test-dir build/rocclr-flags --output-on-failure
```

The test compiles the production `utils/flags.cpp` with its real flag declarations.
It substitutes the process-ID and logging dependencies so it can exercise
`Flag::setValue` independently of runtime initialization and GPU backends.
