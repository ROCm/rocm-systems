# PR 8706 Review Follow-up Design

## Scope

Address the seven actionable items in review 4859765852 without changing the
set or meaning of execution-plugin callbacks.

## Threading contract

Rename the opt-in policy from `requires_serial_execution()` to
`requires_serial_hot_hooks()` on both `ExecutionPlugin` and
`ExecutionPluginGroup`. The name describes the actual effect: lifecycle,
dispatch, workgroup, wavefront, and barrier hooks are always serialized, while
instruction, memory-routing, and register hooks are serialized only when a
contained plugin requests it.

The group samples each plugin's policy once in `add()`. Plugins must therefore
return a stable value from construction onward. Group construction is a
configure-before-publication phase: callers must finish `add()`, `add_sink()`,
and `set_sink_dir()` before publishing the group to simulation components, and
must not mutate it afterward.

Use one recursive callback mutex for both hook classes. This preserves mutual
exclusion between infrequent callbacks and opted-in hot callbacks while allowing
an infrequent callback to read registers and synchronously re-enter hot read
hooks on the same thread. Separate mutexes were rejected because they would let
the two hook classes overlap and could race plugin state. A non-notifying
register path was rejected because snapshot reads are observable register reads
and suppressing their callbacks would change plugin behavior.

Both dispatch helpers return immediately when the group contains no plugins.
This keeps the shared default empty group out of the process-wide callback lock.

## Same-build plugin guard

Add a fourth required C export that returns a deterministic same-build identity.
CMake computes the identity from the plugin-facing headers and toolchain
configuration and makes it available to the host loader and in-tree plugin
targets. `ROCJITSU_DEFINE_PLUGIN` emits the export automatically.

The loader resolves and compares this identity before calling
`rocjitsu_plugin_metadata`, so it never dereferences layout-dependent metadata
from a stale module. A fixture preserving the immediately previous metadata and
`ExecutionPlugin` layouts omits the new export and proves rejection happens
before plugin creation. The identity is a stale-build guard, not a stable ABI or
a compatibility promise.

## Tests and documentation

- Combine a serial-hot-hook plugin with a halt callback that reads registers and
  prove the callback returns with the snapshot populated.
- Exercise empty-group callbacks concurrently to cover the lock bypass.
- Restore the previous-layout plugin fixture and assert rejection before create.
- Give the positive overlap test a generous deadline and the two non-overlap
  tests short bounds.
- Update plugin docs for the renamed policy, separate hook categories, empty
  dispatch bypass, configure-before-publication contract, and same-build guard.
- Remove the remaining profiling references from plugin output documentation
  and the Mirage Rust loader comment.

Validation includes the focused execution-plugin and loader suites, the Release
`rocjitsu_tests` build, the branch-diff pre-commit sweep, and the relevant
sanitizer configuration when available.
