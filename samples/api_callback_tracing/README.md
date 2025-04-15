# API Callback Tracing Sample

## Services

- Captures events like API calls using callbacks.
- HSA API (Core, AMD Ext)
- HIP API (Runtime)
- Marker API (Core, Name)

## Properties

- Handles roctxProfilerPause and roctxProfilerResume operations using a control context.
- Captures API calls and logs details like thread ID, operation type, and duration.
- Provides a detailed trace of all function calls and events for debugging.
