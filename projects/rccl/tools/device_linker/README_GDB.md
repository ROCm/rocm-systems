# How to Debug the "Invalid Abbreviation" Error

## Quick Start

Run this to start an interactive gdb session:
```bash
./tools/device_linker/gdb_interactive.sh
```

Then in gdb, type these commands one at a time:

```
break DWARFUnit::tryExtractDIEsIfNeeded
run -o /tmp/test_output.o --dispatcher build/release/device_linker_output/dispatcher_device.elf --host-table build/release/hipify/gensrc/host_table.cpp --target gfx942:xnack-:sramecc+ --input-dir build/release/specialized_objs/
```

When it breaks (which should be when the error occurs), type:
```
bt          # Show stack trace
print this  # See which DWARFUnit object
info locals # See local variables
```

Then type `continue` to see if it breaks again.

## What to Look For

When it breaks at `DWARFUnit::tryExtractDIEsIfNeeded`, check:
1. **Stack trace (`bt`)**: Shows the call chain - is it reading input DWARF or output DWARF?
2. **Local variables (`info locals`)**: Look for pointers to data buffers - are they pointing to our `debug_info_out_` buffer or to input file data?
3. **The `this` pointer**: Which DWARFUnit is it? Is it from an input file or from the merged output?

## Understanding the Output

- If the data pointer points to our `debug_info_out_` buffer → We're writing wrong codes
- If the data pointer points to input file data → The error is from reading input files (less likely)
- If you see `DWARFContext::create` in the stack → It's reading back the merged DWARF we wrote

## Alternative: Break on Error Message

If you can't break at `tryExtractDIEsIfNeeded`, try breaking where the error message is printed. Since it's not `fprintf`, it might be:
- `llvm::WithColor::error`
- `llvm::errs()` stream writes
- Or search for the string in the binary

To search for the string:
```
(gdb) info proc mappings
(gdb) find 0x400000, +0x7fffffffffff, "contains invalid abbreviation"
```
