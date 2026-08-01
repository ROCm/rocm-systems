# PR 8706 Review-Fix Design

## Goal

Address every actionable review comment on PR 8706 while keeping the pull
request limited to execution-plugin policy. The final branch must exclude the
already-merged PR 8705 history, reject stale v2 runtime plugins before calling
their factory, prevent group subclasses from hiding callbacks, and explain the
conservative threading default with concrete examples.

## Branch shape

Create a backup ref for the current head, then rebase the single PR 8706 commit
from the old PR 8705 head onto current `origin/develop`. Verify that the rebased
diff contains only the execution-plugin change before adding review fixes.

## Plugin contract

Adding `ExecutionPlugin::requires_serial_execution()` changes the virtual table
layout, so increment `kPluginAbiVersion` from 2 to 3. Bundled plugins rebuild
against v3 normally.

Add a separate legacy fixture that preserves the v2 execution-plugin layout
and reports ABI version 2 without including the current plugin interface. Its
factory records a trace if called. The v3 loader test must reject the module and
prove that the factory trace is absent. Keep the existing matching-v3 fixture
to cover successful loading and lifecycle dispatch.

## Group policy boundary

Make `ExecutionPluginGroup` `final` and make all forwarding callbacks
non-virtual. Capability queries continue to derive exclusively from contained
`ExecutionPlugin` objects. Decorators such as `ProfiledExecutionPlugin` remain
ordinary contained plugins, so their hooks and serialization requirements
cannot be hidden from policy aggregation.

## Threading default

Keep `ExecutionPlugin::requires_serial_execution()` defaulting to `true`.
Existing and future plugins remain safe unless they explicitly opt into
concurrent command-processor callbacks after an audit. A default of `false`
would silently expose mutable counters, dispatch maps, output sinks, profiling
state, and callback-order assumptions to races. A pure virtual declaration
would force every plugin to choose but would add unnecessary source churn and
still require the same ABI bump.

Expand the plugin documentation with these concrete failure modes and state
that returning `false` requires the full callback path, including sink and
lifecycle behavior, to be concurrency-safe.

## Validation

Run the focused plugin-loader and execution-plugin tests, then the relevant
Rocjitsu test suite and pre-commit checks. Check `git diff --check`, compare the
rebased commit with the old PR-only commit using `git range-diff`, and verify the
live PR head and file list after a force-with-lease push.

## GitHub follow-up

Prepare concise responses for the stale-diff, ABI, group-subclassing, and
threading-default comments. Do not post replies or resolve threads unless the
user separately authorizes those GitHub write actions.
