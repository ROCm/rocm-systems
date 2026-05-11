# rocprofsys diagnostic format spec (gdb-style compact)

Locked spec for the Phase-1 stacktrace + exception formatter. The format
mirrors gdb's `bt` output: one line per frame, prefixed with `#N`, with
function-name column padded to align the `in <file>:<line>` suffix.

## Header line

```
error: <exception::what()>
```

`error:` is bold bright-red when color is on. The header is single-line.
If `what()` contains a newline (the timemory `core/exception.cpp` embeds
backtrace text), strip everything from the first newline onward; the
trace is rendered separately.

A blank line follows the header before the trace.

## Frame line

```
  #N  <function-name>            in <relative-path>:<line>
```

- 2-space indent.
- `#N` frame index, where N starts at 0 (top of stack).
- Frame-index width auto: 1, 2, or 3 digits, sized to the largest
  index that will be printed.
- Two spaces after `#N`.
- Function name padded with spaces so the `in` keyword aligns across
  frames, capped at `max_function_width` (default 60).
- Two spaces, then `in <file>:<line>`.

Inlined frames render like any other frame plus a trailing `[inlined]`
tag at end of line. There is never more than one line per frame.

If line info is unavailable, the line drops the `:line` suffix:

```
  #2  <function-name>            in <relative-path>
```

If the file is unavailable too, the `in ...` clause is omitted entirely.

If the function name is unresolved, the address is rendered as a
hex literal in its place:

```
  #3  0x00007f8c4a3b1c40        in <relative-path>:<line>
```

## Long function names

When a single demangled name exceeds `max_function_width` (default 60),
that frame skips the padding rule. The `in <file>:<line>` follows after
a single space on the same line, accepting the misalignment. One frame
per line is preserved.

## Skipped frames

Skip filters apply to demangled names (substring match). When any
frames are dropped, append a single trailer line:

```
       ... 4 frames skipped (libc, runtime) ...
```

## Truncated frames

When `max_frames_shown` is exceeded, keep the top `max_frames_shown`
frames and append:

```
       ... 12 more ...
```

Both trailers can co-exist in the same trace.

## Module suffix

When `with_module=true` AND the frame's module is not the main exe,
append `(<module-basename>)` in dim gray after the function name (and
before the `in` clause):

```
  #5  __libc_start_main (libc.so.6)  in /usr/src/...:N
```

Frames in the main executable never carry a module suffix.

## Offset suffix

When `with_offset=true`, append `[+0xNN]` in dim gray after the function
name:

```
  #1  rocprofsys::pmc::sampler::setup() [+0x1ac]  in source/.../sampler.cpp:92
```

## Color palette

Element                       | Style                | ANSI
---                           | ---                  | ---
`error:`                      | bold bright red      | `\033[1;91m`
`#N` frame index              | dim gray             | `\033[90m`
function name                 | cyan                 | `\033[36m`
`in` keyword                  | dim                  | `\033[2m`
file path                     | dim gray             | `\033[90m`
`:line`                       | yellow               | `\033[33m`
`[inlined]`                   | dim gray             | `\033[90m`
`[+0xNN]`                     | dim gray             | `\033[90m`
`(module)`                    | dim gray             | `\033[90m`
`... N skipped ... `          | dim                  | `\033[2m`
`... N more ...`              | dim                  | `\033[2m`

Reset is `\033[0m`. Every styled span ends with reset; color never
crosses a frame boundary.

## Color decision rules

```
color = format_options.color
if color == on:           emit ANSI
if color == off:          plain
if color == auto_detect:
    if env NO_COLOR is set (any value): off
    elif env CLICOLOR_FORCE=1 is set:   on
    elif isatty(fd 2):                  on
    else:                               off
```

`fd 2` is the default because `print_exception()` writes to stderr.

## Env-var overrides

Var                            | Effect
---                            | ---
`NO_COLOR`                     | color off (any value)
`CLICOLOR_FORCE=1`             | color on when not a TTY
`ROCPROFSYS_TRACE_VERBOSE=1`   | with_offset=true, with_module=true, max_frames_shown=64
`ROCPROFSYS_TRACE_NO_FILTER=1` | clear default skip filters

These overrides are read once at the first call to `to_string()` per
process and cached.

## Defaults summary

```
format_options{
  color              = auto_detect,
  with_module        = true,
  with_offset        = false,
  with_file_line     = true,
  skip_substrings    = default_skip_filters(),
  max_function_width = 60,
  max_frames_shown   = 32,
}
```

## Hard rules (test-locked)

1. The `error: ...` header is on its own line.
2. Each frame is exactly one line.
3. Frame indices are contiguous (0, 1, 2, ...) and monotonically
   increasing in print order.
4. Skip and truncate trailers each appear at most once per trace.
5. Color escapes never cross frame boundaries: every styled token is
   `<style><content>\033[0m`.
6. `NO_COLOR` always wins over `CLICOLOR_FORCE`.
