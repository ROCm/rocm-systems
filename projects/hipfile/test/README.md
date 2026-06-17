# hipFile Test Directories

## `amd_detail`

AMD-specific tests. Primarily consists of tests of the internals
of hipFile.

## `common`

Header and source files that are used by multiple test programs.

## `legacy`

A couple of test programs from the early days of hipFile. They are
built but require a bit of setup to run properly and are thus not
in CI.

These will be removed when we have more extensive testing.

## `nvidia_detail`

NVIDIA-specific tests. Currently limited to API compatibility tests.

## `system`

hipFile API unit tests that require a GPU go here. These should run on
both AMD and NVIDIA.

## `unit`

Doesn't exist yet, but any hipFile unit tests that don't require a GPU would go here.
