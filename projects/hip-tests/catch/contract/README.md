# HIP Semantic Contract Tests

This directory contains public HIP API semantic contract tests.

Contract tests validate externally observable HIP behavior through public HIP APIs only. They are small, deterministic, device-required tests intended to fail like semantic contract violations, not stress, performance, platform-policy, or customer-scenario tests.

The first domains are:

- `memory`: allocation and free contracts
- `transfer`: synchronous copy contracts
- `runtime`: runtime initialization, device visibility, version, and error-state contracts
- `device`: portable current-device property contracts
- `stream_event`: stream and event lifecycle, query, synchronization, and wait-event ordering contracts
- `async_transfer`: async copy visibility and invalid-kind consistency contracts
- `memset`: byte, word, dword, and async memset contracts

Run the layer with:

```bash
cmake --build <build-dir> --target contract_tests
ctest --test-dir <build-dir> -L contract --output-on-failure
```
