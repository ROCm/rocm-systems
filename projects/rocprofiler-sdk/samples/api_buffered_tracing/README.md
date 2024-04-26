# API Buffer Tracing Sample

## Services

- Code object callback tracing for mapping kernel IDs to kernel names
- HSA API (Core, AMD Ext)
- HIP API (Runtime)
- Kernel dispatch
- Memory copy
- Page Migration
- Scratch Memory

## Properties

- Buffer size of 4096 bytes which is automatically flushed once >= 87.5% of buffer is filled (3584 bytes)
- Creation of dedicated thread for buffer callback delivery
- Push external correlation IDs once per thread (value is thread ID)
- Receives notifications for internal thread creation
