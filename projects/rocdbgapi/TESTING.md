<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Testing ROCdbgapi

**ROCdbgapi (amd-dbgapi) is the shared library that provides low-level
control and inspection of AMD GPU execution state** for debuggers and tools
built on top of it. It has two consumers: **ROCgdb** and **rocr-debug-agent**.

**ROCdbgapi has no test suite of its own.** There is no `test/` directory, no
`enable_testing()` / `add_test()`, and no unit, functional, or performance
tier in this project. Every meaningful validation of a ROCdbgapi change
happens indirectly, by building it and running it through one of its
consumers' test suites:

- [rocr-debug-agent's test suite](../rocr-debug-agent/TESTING.md) - 8 GPU
  scenarios exercising wave, register, memory, and code-object handling.
- [ROCgdb's test suite](https://github.com/ROCm/ROCgdb/blob/amd-staging/TESTING.md) -
  the `gdb.rocm/` GPU suite plus CPU suites, built `--with-amd-dbgapi`.

## Validating a change to ROCdbgapi

Use [TheRock](https://github.com/ROCm/TheRock) to build ROCdbgapi together
with rocr-debug-agent and ROCgdb, then run their test suites. TheRock's
[`debug-tools/README.md`](https://github.com/ROCm/TheRock/blob/main/debug-tools/README.md)
covers building and testing all three components together.

Always run both rocr-debug-agent's and ROCgdb's test suites whenever a change
touches ROCdbgapi - they're the only way ROCdbgapi's behavior gets validated.
CI does the same automatically, since there's no suite of its own to run.
