# Shared Utilities

The `util` library provides low-level building blocks shared across
rocjitsu and simdojo. It is a header-heavy library with minimal
compiled sources.

## Headers

### Bit Manipulation

| File | Description |
|------|-------------|
| `bit.h` | Bit masks, insertion, extraction, counting, alignment. Provides `mask()`, `insert_bits()`, `bits()`, `is_power_of_2()`, `align_up()`. |
| `bitfield.h` | Typed bitfield access over integral or array storage. `Bitfield<N>` with templated `get<>()` / `set()` for arbitrary bit ranges. |

### Data Types and Conversions

| File | Description |
|------|-------------|
| `data_types.h` | Floating-point format conversions: FP16, BF16, FP8 (E4M3), BF8 (E5M2), FP4, FP6. Scalar and vectorized block converters with IEEE-754 rounding modes (truncation, round-to-nearest-even, stochastic). |
| `convert.h` | Type-erasing conversion between opaque C API handles and internal object pointers. |

### SIMD

| File | Description |
|------|-------------|
| `simd.h` | Vectorized operations via `<experimental/simd>` with scalar fallback. Float conversions, integer arithmetic (`mul_hi`, `popcount`, `clz`), rounding, and transcendental approximations (`rcp`, `rsq`, `log`, `exp`). |
| `simd_test_hooks.h` | Test-only seam for forcing scalar execution to compare against SIMD paths. |

### Memory and Concurrency

| File | Description |
|------|-------------|
| `arena_alloc.h` | Fixed-size block pool allocator with free-list. `ArenaAlloc<BlockSize, NumBlocks, BlockAlign>` provides O(1) alloc/dealloc with global-allocator fallback. |
| `spinlock.h` | TTAS spinlock optimized for sub-microsecond critical sections. Uses C++20 `atomic::wait()` with ThreadSanitizer annotations. |
| `intrusive_list.h` | Bidirectional intrusive linked list with optional parent-pointer tracking. `IntrusiveList<T>` and `IListNode<T>` with O(1) insert/erase. |

### Diagnostics

| File | Description |
|------|-------------|
| `log.h` | Compile-time and runtime configurable logging with named groups (VM, CP). Thread-safe, supports lazy evaluation via lambda forms. Use `Logger::print<GroupId>()` for group-filtered output. |
| `except.h` | Exception hierarchy: `Exception` (base), `InvalidInst`, `UnimplementedInst`, `ConfigError`. All derive from `std::exception`. |

### Miscellaneous

| File | Description |
|------|-------------|
| `dynamic_loader.h` | Runtime symbol resolution via `dlsym` (Linux) / `GetProcAddress` (Windows). `lookup_symbol<T>()` returns a typed function pointer. |
| `meta_programming.h` | Type traits and concepts: `IsIntegral`, `IsUnsigned`, `IsArithmetic`, `IsVoid`. Also provides `GetOption` for template option extraction and `always_false_v<T>` for deferred static assertions. |

## Usage

All headers are under `lib/util/include/util/` and can be included as:

```cpp
#include "util/log.h"
#include "util/bit.h"
```

The library links as `util` in CMake. Both `rocjitsu` and `simdojo`
depend on it.
