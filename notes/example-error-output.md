# Demo output

Output produced by `build/ci/bin/diagnostic_demo`. The full ANSI-encoded
stream is in `example-error-output.txt`. Below are clean transcripts
(color escapes stripped).

The demo throws an exception from a deep call chain
(`library_initialize -> sampler_setup -> collector_setup ->
parse_metric_value`), catches it in `main`, and renders the result with
`format_exception` in two modes. Because `format_exception` captures the
trace at format-time (after the throw has unwound the call chain), the
visible frames reflect the post-catch stack: `section` and `main`.

The exception (message + stacktrace) is wrapped in a single side-bar
block. The header `┌─ error` (or `+- error`) opens the block; every
content line gets a `│ ` (or `| `) left margin; `└─` (or `+-`) closes
it.

## Section 1 - default (with color)

The `error` label is bright-red bold; the bar glyph and border are dim
gray; function names are cyan; `:line` is yellow.

```
=== side-bar wrap (default - with color) ===
┌─ error
│ Failed to parse PMC metric 'bad_metric': value out of range
│
│   #0  (anonymous namespace)::section(std::exception const&, bool, char const*)  in source/lib/common/diagnostic/tests/demo.cpp:59
│   #1  main  in source/lib/common/diagnostic/tests/demo.cpp:74
│        ... 3 frames skipped (libc, runtime) ...
└─
```

## Section 2 - no color

Identical content, ASCII bar, no ANSI escapes:

```
=== side-bar wrap (no color) ===
+- error
| Failed to parse PMC metric 'bad_metric': value out of range
|
|   #0  (anonymous namespace)::section(std::exception const&, bool, char const*)  in source/lib/common/diagnostic/tests/demo.cpp:59
|   #1  main  in source/lib/common/diagnostic/tests/demo.cpp:75
|        ... 3 frames skipped (libc, runtime) ...
+-
```

## Color palette demonstrated

The ANSI-decorated output uses:

- `error` label: bold bright red (`\033[1;91m`)
- side-bar glyph (`┌─`, `│`, `└─`): dim gray (`\033[90m`)
- `#N` frame index: dim gray (`\033[90m`)
- function names: cyan (`\033[36m`)
- `in` keyword: dim (`\033[2m`)
- file paths: dim gray (`\033[90m`)
- `:line`: yellow (`\033[33m`)
- `(module.so)` suffix / `[+0xNN]` offset / `[inlined]`: dim gray
- skipped/more trailers: dim
