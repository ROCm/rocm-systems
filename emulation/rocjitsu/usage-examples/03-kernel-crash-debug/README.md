# Example 3: Invalid Pointer — Host Validation Required

## Objective

Learn to handle **NULL and invalid device pointers** in HIP kernels:
- Recognize a common launch bug (`d_ptr = nullptr`)
- Understand **simulator vs hardware** fault behavior
- Validate pointers on the **host** before launch
- Compare buggy vs fixed code under the same rocjitsu config

## What This Example Demonstrates

1. **NULL device pointer** — kernel launched with `d_ptr = nullptr`
2. **Emulator behavior** — rocjitsu may complete without faulting (unlike real GPU)
3. **Host-side validation** — the correct place to catch this bug today
4. **Fixed pattern** — `hipMalloc` + optional device-side null guard

**Important:** This is **not** a demo of rocjitsu catching crashes. Rocjitsu does
not report NULL dereferences. The buggy binary may exit after sync with a
**WARNING** — that means the emulator accepted the bad stores, not that the
pointer was valid.

## Key Concepts

- Always `hipMalloc` (or otherwise obtain a valid device pointer) before launch
- Check pointers on the host before `<<<>>>`
- `RJ_LOG=1` confirms kernel **dispatch**, not pointer validity
- Use `timeout` when debugging **hangs** on `hipDeviceSynchronize()` (other crash types)

## Files

- `src/crash_example.cpp` — NULL pointer launch (misleading if read as “rocjitsu caught it”)
- `src/crash_fixed.cpp` — Valid allocation + null guard
- `Makefile` — Build and run targets

## Quick Start

```bash
make
make run-crash    # buggy: WARNING + exit 1 (|| true in Makefile)
make run-fixed    # fixed: clean success
```

## The Bug and Fix

### NULL pointer (bug)

```cpp
float *d_ptr = nullptr;
unsafe_kernel<<<1, 64>>>(d_ptr, 64);  // undefined on hardware
```

### Fixed

```cpp
float *d_ptr = nullptr;
hipMalloc(&d_ptr, 64 * sizeof(float));
if (d_ptr == nullptr) { /* handle error */ }
safe_kernel<<<1, 64>>>(d_ptr, 64);  // kernel also guards ptr != nullptr
```

## Expected Output

### Buggy (`make run-crash`)

```text
Invalid Pointer Example - NULL device pointer
  d_ptr = nullptr
  Launch: unsafe_kernel<<<1, 64>>>(d_ptr, 64)

Launching kernel with NULL pointer...

WARNING: hipDeviceSynchronize() returned successfully.
  rocjitsu did NOT fault on the NULL pointer.
  The emulator wrote through VA 0 into sparse backing memory.
  On real hardware this is undefined — may fault, hang, or corrupt.
  Fix: hipMalloc before launch + host-side validation (see crash_fixed.cpp).
```

With `RJ_LOG=1`, stderr may also show kernel dispatch metadata (e.g.
`grid=[64,1,1] wg=[64,1,1]`). That confirms the kernel ran — it does **not**
mean the pointer was valid.

### Fixed (`make run-fixed`)

```text
Launching kernel with valid pointer...
Success! Kernel completed safely.
```

## Debugging Workflow

1. **Run buggy version** — note the WARNING; do not trust exit 0 under emulation alone
2. **Optional:** `RJ_LOG=1 make run-crash` — confirm dispatch appeared on stderr
3. **Run fixed version** — `make run-fixed` should succeed
4. **In your code:** validate every device pointer after `hipMalloc` / before launch
5. **On hardware:** use ROCm sanitizers / debugger for real fault behavior

## What rocjitsu Helps With (and What It Doesn't)

| Helps | Does not help (this example) |
|-------|------------------------------|
| Safe repro environment (no lab GPU reset) | Automatic NULL detection |
| Same config for buggy vs fixed comparison | HIP error on bad pointer |
| `RJ_LOG=1` dispatch visibility | “Crash caught” message |
| `timeout` for **hang** debugging | Page fault like real GPU |

## Key Takeaways

- NULL pointers are a **host validation** problem in this example
- Rocjitsu may **silently emulate** stores to VA 0 — behavior differs from hardware
- Compare with `crash_fixed.cpp` for the production pattern
- See [Example 2](../02-memory-bounds-error/) for another case where host checks catch the bug

## Next Steps

- [Example 2: Memory Bounds](../02-memory-bounds-error/)
- [Example 4: Race Detection](../04-data-race-simple/)
- [GUIDE.md](GUIDE.md) — full debugging notes
