# C++ Naming

One rule:

> Types are PascalCase.

Classes, structs, enums, and type aliases. `CountersWriter`, `SdkWrapper`,
`ProcessState`, `InstallState`.

That is the whole rule. There is no rule here about file names, namespaces,
member prefixes, or function names. Follow whatever the file you are editing
already does.

## Why only this

The tree has two type-naming styles. `pc_sampling_collector/` uses `*_t` snake
case, following rocprofiler-sdk's own style, and so do some types in
`rocprofiler_compute_tool/` that sit directly on the sdk's API. Everything
written since uses PascalCase.

Without a rule, the next author guesses. With this rule, new types are
consistent and nothing existing has to move.

New files under `pc_sampling_collector/` will look mixed for a while. That is
the expected outcome, not something to go and fix. Do not rename existing types,
and do not open a PR that inventories violations.
