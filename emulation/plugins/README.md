# Emulation Plugins

Standalone libraries implementing core analysis and instrumentation logic
for GPU simulation/emulation. Each plugin is intended to be a self-contained library with its own
build, tests, and CI workflow.

Plugins here must not depend on any specific emulator (e.g. rocjitsu).
They define algorithms and data structures that emulators consume through
thin adapter layers. This keeps the core logic testable in isolation and
reusable across different projects.

## Plugins

| Plugin | Description |
|---|---|
| [race-detector](race-detector/) | Detects synchronization hazards (missing `s_waitcnt`, `s_barrier`) by tracking in-flight memory events at the register and LDS byte level. |

## Building a plugin standalone

Each plugin can be built and tested independently:

```bash
cmake -G Ninja -B build -S emulation/plugins/<plugin-name> -DBUILD_TESTING=ON
cmake --build build -j$(nproc)
ctest --test-dir build
```

## Adding a new plugin

1. Create a directory under `emulation/plugins/<name>/` with its own `CMakeLists.txt`.
2. Support standalone builds: gate `FetchContent` dependencies behind `if(NOT PROJECT_NAME)`.
3. Add a CI workflow at `.github/workflows/<name>.yml` triggered on changes to the plugin path.
4. Keep the plugin free of emulator-specific dependencies. Emulator integration belongs in the
   emulator's own plugin adapter layer (e.g. `emulation/rocjitsu/plugins/`).
