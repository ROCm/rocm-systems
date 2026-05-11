# Demo output

Output produced by `build/ci/bin/diagnostic_demo` (run via
`CLICOLOR_FORCE=1` so the color variants render even off-TTY).

The full ANSI-encoded stream is in `example-error-output.txt`. Below
are clean transcripts.

## PART A - throw-site capture

`stacktrace::capture()` is invoked at the deep call site (simulating
the future custom exception's constructor). The resulting trace contains
the full call chain at the moment of capture.

### A1 - default options (color + auto-detect)

```
error: Failed to parse PMC metric 'bad_metric': value out of range

  at (anonymous namespace)::parse_metric_value(...)
     in source/lib/common/diagnostic/tests/demo.cpp:45:1
  at (anonymous namespace)::collector_setup(...)
     in source/lib/common/diagnostic/tests/demo.cpp:52:1
  at (anonymous namespace)::sampler_setup(...)
     in source/lib/common/diagnostic/tests/demo.cpp:59:1
  at (anonymous namespace)::library_initialize(...)
     in source/lib/common/diagnostic/tests/demo.cpp:66:1
  at main
     in /usr/include/c++/13/bits/basic_string.h:804:19
  ... 3 frames skipped (libc, runtime) ...
```

Top frame is the throw site (`parse_metric_value`); the deepest call
appears first per the format spec. 3 libc/runtime frames are filtered.

### A3 - color on + offsets + modules

Adds `[+0xNN]` PC offsets to each function line.

```
error: Failed to parse PMC metric 'bad_metric': value out of range

  at (anonymous namespace)::parse_metric_value(...) [+0x21]
     in source/lib/common/diagnostic/tests/demo.cpp:45:1
  at (anonymous namespace)::collector_setup(...) [+0x1d]
     in source/lib/common/diagnostic/tests/demo.cpp:52:1
  ...
```

### A4 - color on + source excerpt

Shows the source code of each frame (1 line of context above + below)
with a caret underline on the failing line:

```
  at (anonymous namespace)::parse_metric_value(...)
     in source/lib/common/diagnostic/tests/demo.cpp:45:1
            44 |     }
            45 | }
               | ^
            46 |
```

### A5 - max_frames_shown=2 (truncation)

```
error: Failed to parse PMC metric 'bad_metric': value out of range

  at (anonymous namespace)::parse_metric_value(...)
     in source/lib/common/diagnostic/tests/demo.cpp:45:1
  at (anonymous namespace)::collector_setup(...)
     in source/lib/common/diagnostic/tests/demo.cpp:52:1
  ... 3 frames skipped (libc, runtime) ...
  ... 2 more ...
```

Both the `skipped` (filter) trailer and the `more` (truncation) trailer
appear when both apply.

## PART B - catch-site capture (`format_exception`)

```
error: Failed to parse PMC metric 'bad_metric': value out of range

  ... 3 frames skipped (libc, runtime) ...
```

Phase-1 limitation: when capture happens at the catch site, the throw
site has already unwound from the stack. Only the catcher's frame
remains and after default filters apply, nothing visible may be left.
This is why the future custom `rocprofsys::exception` will capture in
its constructor (PART A's pattern) and carry the trace as a member.

## Color palette demonstrated

The ANSI-decorated output uses:

- `error:` keyword: bold bright red (`\033[1;91m`)
- function names: cyan (`\033[36m`)
- `at` / `in` keywords: dim
- file paths: dim gray (`\033[90m`)
- `:line:col`: yellow (`\033[33m`)
- `[+0xNN]` offset / `(module.so)` suffix / `[inlined]`: dim gray
- caret `^^^^^` in source-excerpt mode: bold bright green (`\033[1;92m`)
- skipped/more trailers: dim
