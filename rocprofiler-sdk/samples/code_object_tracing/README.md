# CodeObject Tracing

## Services

Trace and analyze the execution of GPU code objects and kernel symbols.

## Properties

- This tool is designed to capture and log information about code object loading/unloading and kernel symbol registration/un-registration events during the execution of GPU programs.

- Whenever a relevant event occurs, such as a code object being loaded/unloaded or a kernel symbol being registered/unregistered. The function processes the event data, formats it into a human-readable string, and appends it to the call stack.
