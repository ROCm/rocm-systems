# External Correlation ID Request Sample

## Services

- Code object callback tracing for mapping kernel IDs to kernel names
- HIP Runtime API:
  - hipLaunchKernel
  - hipMemcpyAsync
  - hipMemsetAsync
  - hipMalloc
- Kernel dispatch
- Memory Copy
- External correlation ID request:
  - Kernel dispatch
  - Memory copy
- Correlation ID retirement

## Properties

- Subscribes to an external correlation ID request for all kernel dispatches and async memory copies
- Generates an external correlation ID containing all the arguments passed to the request callback
- Demonstrates that all external correlation IDs which are requested are passed back to tool in buffer callbacks
- Demonstrates that all internal correlation IDs which are provided as an input argument to request are retired
- Buffer size of 4096 bytes which is automatically flushed once >= 87.5% of buffer is filled (3584 bytes)
