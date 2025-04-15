# CodeObject tracing

## Services

- code object tracing.
  
## Properties

- This tool hooks into ROCProfiler's callback and buffer tracing mechanisms to:
  - Decode and analyze GPU code objects.
  - Three kernel variants are used in sample; simple transpose, in-place LDS swap and LDS no bank conflicts.
  - Trace kernel symbols and instructions.
  - Log disassembly and statistics for debugging or performance analysis.
