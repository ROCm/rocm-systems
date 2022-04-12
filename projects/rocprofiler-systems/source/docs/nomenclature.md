# Nomenclature

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 3
```

The list provided below is intended to (A) provide a basic glossary for those who are not familiar with binary instrumentation and (B) provide clarification to ambiguities when certain terms
have different contextual meanings, e.g., omnitrace's meaning of the term "module" when instrumenting Python.

- **Binary**
  - File written in the Executable and Linkable Format (ELF)
  - Standard file format for executable files, shared libraries, etc.
- **Binary Instrumentation**
  - Inserting callbacks to instrumentation into an existing binary. This can be performed statically or dynamically
- **Static Binary Instrumentation**
  - Loads an existing binary, determines instrumentation points, and generates a new binary with instrumentation directly embedded
  - Applicable to executables and libraries but limited to only the functions defined in the binary
  - Also known as: **Binary Rewrite**
- **Dynamic Binary Instrumentation**
  - Loads an existing binary into memory, inserts instrumentation, executes binary
  - Limited to executables but capable of instrumenting linked libraries
  - Also known as: **Runtime Instrumentation**
- **Sampling**
  - At periodic intervals, the application is paused and the current call-stack of the CPU is recorded alongside with various other metrics
  - Uses timers that measure either (A) real clock time or (B) the CPU time used by the current thread and the CPU time expended on behalf of the thread by the system
  - **Sampling Rate**
    - The period at which (A) or (B) are triggered (in units of `# interrupts / second`)
    - Higher values increase the number of samples
  - **Sampling Delay**
    - How long to wait before (A) and (B) begin triggering at their designated rate
- **Module**
  - With respect to binary instrumentation, a module is defined as either the filename (e.g. `foo.c`) or library name (`libfoo.so`) which contains the definition of one or more functions
  - With respect to Python instrumentation, a module is defined as the _file_ which contains the definition of one or more functions.
    - The full path to this file _typically_ contains the name of the "Python module"
- **Basic Block**
  - Straight-line code sequence with:
    - No branches in (except for the entry)
    - No branches out (except for the exit)
- **Address Range**
  - The instructions for a function in a binary start at certain address with the ELF file and end at a certain address, the range is `end - start`
  - The address range is a decent approximation for the "cost" of a function, i.e., a larger address range approx. equates to more instructions
- **Instrumentation Traps**
  - On the x86 architecture, because instructions are of variable size, the instruction at a point may be too small for Dyninst to replace it with the normal code sequence used to call instrumentation
    - Also, when instrumentation is placed at points other than subroutine entry, exit, or call points, traps may be used to ensure the instrumentation fits
  - By default, omnitrace avoids instrumentation which requires using a trap
- **Overlapping functions**
  - Due to language constructs or compiler optimizations, it may be possible for multiple functions to overlap (that is, share part of the same function body) or for a single function to have multiple entry points
  - In practice, it is impossible to determine the difference between multiple overlapping functions and a single function with multiple entry points
  - By default, omnitrace avoids instrumenting overlapping functions

## Additional Notes

The ["Data granularity in profiler types"](https://en.wikipedia.org/wiki/Profiling_(computer_programming)#Data_granularity_in_profiler_types) section of
the Wikipedia ["Profiling (computer programming)"](https://en.wikipedia.org/wiki/Profiling_(computer_programming)) page may be a useful reference in understanding
the different profiling modes and their trade-offs.
