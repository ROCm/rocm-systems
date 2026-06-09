# Kernel Logging Plugin

Logs kernel dispatch metadata and detects MFMA instruction usage.

## What it logs

- **Kernel dispatches**: entry PC, grid dimensions, workgroup dimensions,
  register counts, and kernel name (when available from the code object).
- **MFMA detection**: reports the first MFMA or WMMA instruction seen in
  each dispatch.

## Enabling

```bash
RJ_LOG=1 LD_PRELOAD=librocjitsu_kmd.so ./my_app
```

Output goes through the plugin sink system — see the
[plugins README](../README.md) for sink configuration.
