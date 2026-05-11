# Demo output

Output produced by `build/ci/bin/diagnostic_demo` (run via
`CLICOLOR_FORCE=1` so the `auto_detect` variant renders color even
off-TTY).

The full ANSI-encoded stream is in `example-error-output.txt`. Below
are clean transcripts (color escapes stripped).

The demo captures one trace from a deep call chain
(`library_initialize -> sampler_setup -> collector_setup ->
parse_metric_value`) and renders it three ways.

## Section 1 - default (color auto_detect)

With `CLICOLOR_FORCE=1` set during the run, auto_detect resolves to ON
and the trace carries ANSI escapes.

```
error: Failed to parse PMC metric 'bad_metric': value out of range

  #0  (anonymous namespace)::parse_metric_value(...)  in projects/.../demo.cpp:33
  #1  (anonymous namespace)::collector_setup(...)     in projects/.../demo.cpp:40
  #2  (anonymous namespace)::sampler_setup(...)       in projects/.../demo.cpp:47
  #3  (anonymous namespace)::library_initialize(...)  in projects/.../demo.cpp:54
  #4  main                                            in /usr/include/.../basic_string.h:804
       ... 3 frames skipped (libc, runtime) ...
```

(Demangled `std::__cxx11::basic_string<...>` argument types are elided
to `(...)` in the transcript above for readability. The real demo
output shows them in full and exceeds `max_function_width` (60), so the
`in` clause follows after a single space rather than a padded column -
the format spec's documented behavior for oversized names.)

## Section 2 - color OFF

Identical content, no ANSI escapes anywhere:

```
error: Failed to parse PMC metric 'bad_metric': value out of range

  #0  (anonymous namespace)::parse_metric_value(...)  in projects/.../demo.cpp:33
  #1  (anonymous namespace)::collector_setup(...)     in projects/.../demo.cpp:40
  #2  (anonymous namespace)::sampler_setup(...)       in projects/.../demo.cpp:47
  #3  (anonymous namespace)::library_initialize(...)  in projects/.../demo.cpp:54
  #4  main                                            in /usr/include/.../basic_string.h:804
       ... 3 frames skipped (libc, runtime) ...
```

## Section 3 - color FORCED ON

`color_mode::on` bypasses the TTY check. Same content, ANSI escapes
restored.

## Color palette demonstrated

The ANSI-decorated output uses:

- `error:` keyword: bold bright red (`\033[1;91m`)
- `#N` frame index: dim gray (`\033[90m`)
- function names: cyan (`\033[36m`)
- `in` keyword: dim (`\033[2m`)
- file paths: dim gray (`\033[90m`)
- `:line`: yellow (`\033[33m`)
- `(module.so)` suffix / `[+0xNN]` offset / `[inlined]`: dim gray
- skipped/more trailers: dim
