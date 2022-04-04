# Generating a Critical Trace

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 4
```

## Overview

A critical trace is defined in omnitrace as the most time-consuming path through a parallelized code.
The steps for generating a critical trace are:

1. Enable the `OMNITRACE_CRITICAL_TRACE` setting
2. Configure any other relevant critical-trace settings, as needed
   - `omnitrace-avail --categories settings::critical-trace`
3. Execute application
4. Locate the JSON files with `call-chain` in their name
5. Provide these files to the `omnitrace-critical-trace` executable
6. Open generated perfetto file in [ui.perfetto.dev](https://ui.perfetto.dev/)

## omnitrace-critical-trace Executable

The `omnitrace-critical-trace` executable post-processes one or more `call-chain` JSON files and generates a perfetto output
for visualizing the critical trace.

**INCOMPLETE**

This executable is still under-development.
