# addrlib

AMD's address library: surface addressing, tiling and swizzle math. Given a surface
description (dimensions, bits per pixel, mip levels, swizzle mode, chip family) it
returns the allocation size, alignment, pitch and per-mip offsets that the hardware
expects.

This directory is the single shared copy. It previously existed three times — in
`projects/rocr-runtime`, `projects/crux` and `runtimes/image/rocaddr3` — which let
them drift: the CRUX copy had fallen behind on gfx1170/gfx1171 support.

## Consuming it

### CMake

```cmake
add_subdirectory(<path-to>/shared/addrlib addrlib)
target_link_libraries(my_target PRIVATE rocm::addrlib)
```

`rocm::addrlib` is a static library; `addrlib-objects` is the same code as an object
library for consumers that embed it directly. Include directories, the C++ standard
and the namespace definition all propagate from the target.

### Namespace

addrlib is compiled *into the consumer's namespace*, because callers use unqualified
names from inside their own namespace:

| Consumer | Namespace |
|---|---|
| ROCr, rocaddr3 | `rocr` (default) |
| CRUX | `amd::crux::hsa` |

Set it at configure time:

```cmake
set(ADDRLIB_NAMESPACE "amd::crux::hsa" CACHE STRING "" FORCE)
```

The default lives in `inc/addrtypes.h`, so the sources also build standalone without
CMake — which is what `runtimes/image/rocaddr3` does from `build.rs`.

Both namespaces have been verified to produce byte-identical object code.

### Non-CMake builds

Compile the twelve `.cpp` files listed in `CMakeLists.txt` with the `inc`, `src`,
`src/<gfx*>` and `src/chip/<gfx*>` directories on the include path. `LITTLEENDIAN_CPU`
(or `BIGENDIAN_CPU`) **must** be defined — `src/chip/*/*_gb_reg.h` hard-errors without it.

## Modifying this tree

Don't, beyond re-syncing from upstream. It is a vendored drop, and local fixes have to
be re-applied on every sync — that is how the three copies diverged in the first place.
Compiler warnings are suppressed by the build (`-w`, `SYSTEM` includes) rather than
patched out of the source, mirroring how CRUX handles the vendored Khronos headers.

The one intentional local change is the `ADDRLIB_NAMESPACE` macro, which replaced a
hard-coded `namespace rocr`.

## A caution for testing

addrlib does not reject every invalid input. Some combinations **hang in an infinite
loop** rather than returning an error, which matters if you write a test that sweeps
inputs. Observed cases:

- `ADDR_SW_LINEAR_GENERAL` on non-power-of-two surfaces (e.g. 256x129)
- 256KB swizzle modes on chip families that do not support them (e.g. gfx9)
- `ADDR_RSRC_TEX_3D` on the families tested, for both odd and power-of-two depths
- `Addr3*` entry points called against a pre-gfx12 library

Query the valid modes first (`Addr2GetPreferredSurfaceSetting`,
`Addr3GetPossibleSwizzleModes`) instead of sweeping blindly.
