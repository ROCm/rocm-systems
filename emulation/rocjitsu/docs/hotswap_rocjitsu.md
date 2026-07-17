# Using rocjitsu for HotSwap

rocjitsu can provide the code-object rewrite used by an HSA runtime's HotSwap
path. The integration is intentionally narrow: rocjitsu supplies the small
COMGR-compatible interface required for HotSwap while the runtime and its
ordinary COMGR users continue to use their normal libraries.

The current implementation supports gfx1250 B0 code objects running on a
gfx1250 A0 target.

## Build and install

Build the adapter and selector with the rest of rocjitsu, then install them to
a private prefix:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target rocjitsu_comgr rocjitsu_hotswap_select
cmake --install build --prefix "$PWD/install" --component rocjitsu-hotswap
```

The install contains two relevant shared libraries:

- `librocjitsu_comgr.so` implements the COMGR data and HotSwap rewrite entry
  points required by the runtime.
- `librocjitsu_hotswap_select.so` redirects only the runtime's explicit
  HotSwap COMGR load to `librocjitsu_comgr.so`.

Use the directory containing those installed libraries as
`ROCJITSU_LIBDIR` below. Depending on the platform, it is commonly
`$PWD/install/lib` or `$PWD/install/lib64`.

## Run a workload

HotSwap is enabled when `HSA_HOTSWAP_DISABLE` is absent. Set the adapter to an
absolute path and preload the selector:

```sh
ROCJITSU_LIBDIR=/absolute/path/to/rocjitsu/install/lib

env -u HSA_HOTSWAP_DISABLE \
  ROCJITSU_HOTSWAP_COMGR="$ROCJITSU_LIBDIR/librocjitsu_comgr.so" \
  LD_PRELOAD="$ROCJITSU_LIBDIR/librocjitsu_hotswap_select.so" \
  HSA_HOTSWAP_VERBOSE=1 \
  HSA_HOTSWAP_CACHE_DIR="$PWD/.cache/rocjitsu-hotswap" \
  ./application
```

To restrict a test to one GPU, add the visibility settings appropriate for
the application, for example:

```sh
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0
```

Use a new or empty `HSA_HOTSWAP_CACHE_DIR` when changing the adapter build.
An existing runtime cache can satisfy a request without invoking the rewrite
again.

For a disabled baseline, start a separate process with HotSwap disabled and a
different cache directory:

```sh
env HSA_HOTSWAP_DISABLE=1 \
  HSA_HOTSWAP_CACHE_DIR="$PWD/.cache/hotswap-disabled" \
  ./application
```

Do not preload the general rocjitsu HSA hook library for this integration.
Only the selector is required.

## Confirm that rocjitsu handled the rewrite

With `HSA_HOTSWAP_VERBOSE=1`, a successful selection prints messages with the
following prefixes:

- `[rocjitsu-selector]` reports that the runtime's COMGR request was redirected
  and identifies the shared object that owns the rewrite symbol.
- `[rocjitsu-comgr]` reports whether the input was translated or was already
  A0-compatible.

Seeing only generic runtime HotSwap output is not sufficient proof that the
rocjitsu adapter was selected.

Set `ROCJITSU_HOTSWAP_DUMP_DIR` to an existing writable directory to save an
input code object when translation fails:

```sh
mkdir -p "$PWD/hotswap-failures"
export ROCJITSU_HOTSWAP_DUMP_DIR="$PWD/hotswap-failures"
```

## B0 to A0 policy

The adapter accepts valid gfx1250 HSA code objects and uses revision metadata
from the runtime request to select the action:

- B0 to A0 requests apply the gfx1250 legalization rules and return a new code
  object.
- Inputs already marked A0-compatible are returned unchanged, with a verbose
  message when logging is enabled.
- Other revision or architecture combinations are rejected.

Legalization membership follows the instruction patterns used by the
established HotSwap implementation. Implemented patterns are rewritten;
recognized patterns without a complete rocjitsu legalization fail the entire
rewrite. The adapter does not emit a partially translated code object or
silently run a required pattern unchanged.

The adapter is not a general COMGR replacement. Its exported data-object
operations exist only to support the runtime's HotSwap call sequence, which is
why selection is scoped to that runtime load instead of changing the process's
normal library search path.
