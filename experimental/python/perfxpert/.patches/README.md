# PerfXpert OpenCode Patches

This directory contains AMD-specific patches applied to the opencode submodule to integrate it into the PerfXpert GPU profiling analysis pipeline.

## Patch Series

The patches are organized chronologically and applied in order during the build process (Phase 8 PR 3+).

### Planned Patches (Phase 8 PR 2)

1. **001-node-env-config.patch** — Node environment configuration for rocm-systems container
2. **002-opencode-api-wrapper.patch** — OpenCode gRPC/JSON API wrapper for perfxpert integration
3. **003-perfxpert-session-handler.patch** — Interactive session state management bridge
4. **004-gpu-aware-bundling.patch** — GPU-aware code bundle generation (MI300X optimizations)
5. **005-telemetry-disable.patch** — Disable upstream telemetry; route to perfxpert logging

## Application

Patches are applied automatically during the build phase:

```bash
cd opencode
for patch in ../.patches/*.patch; do
  git apply "$patch" || { echo "Failed: $patch"; exit 1; }
done
```

## Testing

Each patch is validated against:
- Unit test suite: `npm test`
- Integration tests: perfxpert's opencode-launcher tests (Phase 8 PR 4)

## Reverting

To revert to upstream opencode (for testing or debugging):

```bash
git clean -fd
git submodule update --force
```
