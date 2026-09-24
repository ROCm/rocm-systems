# LLD: torch_trace_collector

## Motivation

Implementation of the RecordFunction collector in `hld-torch-trace-collector.md`.

---

## Code flow

```mermaid
flowchart LR
  launch["launch.py"] --> torch["torch.py"]
  torch --> loader["torch_cpp_loader.py"]
  loader --> finder["native_tool_finder.py"]
  loader --> cpu["workload libtorch_cpu.so"]
  loader --> so["torch_trace_collector.so"]
  cpu --> so
  so --> api["plain-C entry points"]
  torch --> api
  api --> core["torch_trace_collector.cpp"]
  core --> snap["snapshot_store.cpp"]
  core --> roctx["roctxRangePushA"]
```

The wrap set lives in `torch.py` and does not change when the collector loads.

---

## Threading

How workers see Python scopes:

- Worker RecordFunction sees ATen and autograd names
  (`evaluate_function`, `AddmmBackward0`). It does not see Python wraps.
- The worker never runs those wraps, so debug info is how wrap frames
  reach it.
- Without that copy, worker ranges start at `evaluate_function` and omit
  `Tensor.backward`. Forward module and ATen names still appear from the
  snapshot.
- Python wraps store the live wrap stack in PyTorch's per-thread debug
  info.
- After forward returns, module wraps have popped, so when autograd
  queues the worker this is typically just `Tensor.backward`.
- Autograd copies that onto the worker. Whenever the worker stack is empty,
  those wrap frames are copied onto it.

```mermaid
%%{init: {"flowchart": {"htmlLabels": true, "curve": "linear", "nodeSpacing": 8, "rankSpacing": 70, "padding": 4}}}%%
flowchart TB
  subgraph top [ ]
    direction LR
    subgraph main [Main thread]
      direction TB
      D["debug info"]
      P["forward snapshot"]
    end
    AG(["autograd"])
    subgraph worker [Worker thread]
      direction TB
      WpadT["<br/>"]
      subgraph wflow [ ]
        direction LR
        S1["overlay"] --> S2["consume snapshot"]
        S2 --> S3["push leaf"]
        S3 --> FIN["ROCTX range"]
      end
      WpadB["<br/>"]
    end
    D -->|"copies debug info<br/>onto the worker task"| AG
    AG -->|"RecordFunction,<br/>empty stack"| S1
  end
  subgraph store [Process-wide snapshot store]
    direction TB
    SpadT["<br/>"]
    KEY["(seqNr, thread id)"]
    SpadB["<br/>"]
  end
  store -->|"backward op lookup"| S2
  S2 ~~~ store
  style top fill:none,stroke:none
  style wflow fill:none,stroke:none
  style WpadT fill:none,stroke:none,color:transparent
  style WpadB fill:none,stroke:none,color:transparent
  style SpadT fill:none,stroke:none,color:transparent
  style SpadB fill:none,stroke:none,color:transparent
```

| Step | From | Stack |
| --- | --- | --- |
| debug info | wrap stack on the main thread | **Tensor.backward** |
| forward snapshot | saved on the main thread during forward | **SimpleNet.forward / Linear.forward / aten::linear / aten::addmm** |
| overlay | debug info copied onto the worker | **Tensor.backward** |
| consume snapshot | process-wide store, prefix dedup | **Tensor.backward / SimpleNet.forward / Linear.forward / aten::linear / aten::addmm** |
| push leaf | RecordFunction name | **… / AddmmBackward0** |
| ROCTX range | full worker stack | **Tensor.backward / SimpleNet.forward / Linear.forward / aten::linear / aten::addmm / AddmmBackward0** |

- Debug info shares the wrap stack still live on the main thread. In this
  example that is `Tensor.backward`.
- It does not include `SimpleNet.forward` or `aten::addmm`; those frames
  have already left the stack.
- Consume appends that frozen forward nest, then the leaf is pushed.
  Save is not this path; it already ran on the forward thread.
- The ROCTX range is wrap + forward nest + leaf.

- Stack and debug-info guards are `thread_local` (`torch_trace_collector.cpp`).
- Snapshot store and install handle are process-wide.
- Python thread's `push_user_scope` publishes the **live** wrap stack into `ThreadLocalDebugInfo`.
- Autograd copies the main thread's `ThreadLocalState` onto the worker thread before `evaluate_function`.
- Overlay copies that restored `ThreadLocalDebugInfo` chain onto the worker's empty marker stack.

---

## RecordFunction start and end
####  `torch_trace_collector.cpp`

1. If the stack is empty, overlay debug info. If overlay pushed frames, the leaf is nested.
2. If the scope is `BACKWARD_FUNCTION`, `seqNr >= 0`, and `forwardThreadId() != 0`, consume `(seqNr, forwardThreadId)` and push frames that are not already a shared prefix. `forwardThreadId() == 0` means no forward identity (`evaluate_function` and other non-Node backward records).
3. Push the leaf (RecordFunction name plus default leaf context).
4. If the scope is `FUNCTION` and `seqNr >= 0`, save the stack (including the leaf) under `(seqNr, currentThreadId())`.
5. Format the stack, append `|torch`, `roctxRangePushA`.

Overlay and consumed snapshot frames are extra pushes. `end_cb` pops the
ROCTX range, the leaf, then those extras.

---

## Snapshot store
#### `snapshot_store.cpp`, `torch_trace_collector.cpp`

- **Insert:** After a forward leaf with a valid `seqNr` is pushed, the collector copies that thread's stack into this map. The matching backward often runs later on a worker, after those frames have popped.
- **Consume:** The matching backward looks up `(seqNr, forward thread id(non-zero))` and pushes frames the stack does not already have. This map is not debug info.
- **Overlay:** If this thread's stack is empty, copy the live wrap chain from debug info onto it (typically `Tensor.backward`). Autograd copied that TLS; it is not this map.
- **Entry:** Wrap frames plus nested ATen names, including the forward leaf.
- **Key:** `(seqNr, thread id)`. Save uses `currentThreadId()`; consume uses `forwardThreadId()`. Consume moves the entry out. A second save of the same key overwrites.
- **Shards:** 64 shards. Hash of the key picks the shard. Each has its own map, LRU, and lock. Keys in the same shard share that lock.
- **LRU:** Each shard keeps at most 10000 entries. A new key past that drops the oldest in that shard (`snapshots_dropped`).
- **Lifetime:** `pending()` is the sum of shard sizes. `uninstall()` / `clear()` empties every shard. Detached forwards stay until LRU evicts them.
- **Counters:** `dump_stats()`: `snapshots_saved`, `snapshots_consumed`, `snapshots_dropped`, `snapshots_overwritten`, and `pending()` as `snapshots_pending`.

---

## Wire format
#### `wire_format.h`

- Each ROCTX range name is the full stack: `marker1/.../markerN:context1/.../contextN[|backend]`.
- Marker `%` and `/` are `%25` and `%2F`. Contexts are not encoded.
- RecordFunction ranges append `|torch`. User-scope ranges append `|<backend>` when `backend` is non-empty.
- Profile post-processing moves `|backend` into a `Backend` column. Unrecognized suffixes are `unknown`.

- **Leaf context** (`leaf_context.h`). RecordFunction runs in C++ and does not carry a location; unlike wrap frames that have it. Dummy locations are set by `leaf_context.h` based on RecordFunction scope, `seqNr`, and whether the stack was empty after overlay — e.g. a backward op vs a nested ATen op.

---

## Plain-C interface

The shared library exports an ABI revision query plus operations to install and
uninstall the callback, query installation state, push and pop user scopes, and
copy collector statistics into caller-owned storage. Strings cross as UTF-8 C
strings, and statistics use a size-tagged C structure. No Python objects or
PyTorch C++ objects cross this public boundary.

The Python facade binds these functions with `ctypes` and preserves the method
surface previously provided by the extension module. Failures are reported by
status code, and exceptions are contained before returning through the C or
RecordFunction callback boundary.

---

## Build and load

- `src/lib/torch_trace_collector/CMakeLists.txt` always builds the collector as
  C++17. It requires `rocprofiler-sdk-roctx`, but no PyTorch installation,
  PyTorch headers or libraries, or Python development headers.
- Small declarations under `torch_abi/` describe only the private PyTorch C++
  surface required by the existing live-stack collector. The resulting
  artifact intentionally resolves those symbols from the workload at runtime.
- CMake produces one `torch_trace_collector.so`, without a Torch-version or
  Python-SOABI suffix. Library output is `${CMAKE_BINARY_DIR}/lib`; the install
  destination is `${CMAKE_INSTALL_LIBDIR}/rocprofiler-compute` (`lib` or
  `lib64`).
- `torch_cpp_loader.py` imports the workload's PyTorch, checks its exact
  version/build tag, Git revision, debug flag, and libstdc++ ABI mode against
  the validated allowlist, and locates the generic artifact.
- The loader reopens the workload's `libtorch_cpu.so` so its Torch/c10 symbols
  are globally visible, loads the collector locally through `ctypes`, and uses
  only its plain-C interface. The collector has no Python ABI dependency.
- Before callback registration, the native gate matches the mapped
  `libtorch_cpu.so` and `libc10.so` GNU build IDs as a validated pair. This
  build-only collector has no AOTI dependency.
- A missing artifact, unsupported identity, mismatched library pair, loader
  error, or rejected installation emits a warning and returns control to
  `TorchDispatchMode`; it does not terminate the workload.

Search is rooted at the executing Python package (checkout: `src/`; install: `<prefix>/libexec/rocprofiler-compute/`). Order, via `find_prebuilt_artifacts` in `native_tool_finder.py`:

1. `<package_root>/../../lib*/rocprofiler-compute/torch_trace_collector.so`
2. `<package_root>/../build/lib`
3. `<package_root>/lib/_build/lib`

The first unique resolved path wins. Install is scanned first, so a packaged
`.so` beats a source build **in the same process**. Run the in-tree
`rocprof-compute` (package root `src/`) to use a source build; the install glob
then does not see `/opt/rocm`.

---

## Tests

- `src/lib/torch_trace_collector/tests/test_torch_trace_collector.cpp` verifies
  snapshot join of backward to forward, overlay of wrap frames on a worker,
  dummy locations, marker encoding, and install.
- Loader unit tests verify generic-artifact discovery, exact Torch identity
  gating, native-library promotion, the plain-C call boundary, and fallback.
- ELF and packaging checks verify the generic name, exported C surface,
  unresolved Torch/c10 imports, absence of Torch/Python `DT_NEEDED` entries,
  and installation when no PyTorch is available at build time.
- `tests/integration/test_profile_torch_trace.py` verifies end-to-end
  `--torch-trace` on a sample workload.
- `tests/integration/test_torch_trace_coverage.py` compares `--torch-trace`
  operator and kernel coverage to `torch.profiler`.
