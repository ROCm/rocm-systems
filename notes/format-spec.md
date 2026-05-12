# rocprofsys diagnostic format spec (framed gdb-style)

Locked spec for the Phase-1 stacktrace + exception formatter. The format
mirrors gdb's `bt` output for the trace - one line per frame, prefixed
with `#N`, function-name column padded so `in <file>:<line>` aligns -
and wraps the `error: <message>` header in a single-line box for visual
weight.

## Top-level shape

```
<box top border>
| error: <exception::what()>                                          |
<box bottom border>

  #0  <function-name>            in <relative-path>:<line>
  #1  <function-name>            in <relative-path>:<line>
  ...
       ... N frames skipped (libc, runtime) ...
       ... M more ...
```

A blank line separates the box from the first frame.

## Two color modes (bool toggle)

`format_options::with_color` is a plain `bool`, default `true`.

| Value | Box characters         | ANSI escapes |
|---    |---                     |---           |
| true  | UTF-8 box-drawing chars | emitted     |
| false | ASCII `+`, `-`, `|`     | none        |

`exception_format_options::with_color` is `std::optional<bool>`. When
set, it is the explicit override and takes precedence over everything.
When unset (the default), `format_exception` and `print_exception`
resolve the effective value as:

```
effective_color = NO_COLOR is set in env ? false : true
```

`NO_COLOR` is honored only at the top-level public entry points
(`format_exception`, `print_exception`). Once `with_color` is set
explicitly, the env var is not consulted.

There is no `auto_detect` mode and no `CLICOLOR_FORCE` support. Anyone
who needs color in a pipe sets `opt.with_color = true` explicitly.

## Header box

```
┌────────────────────────────────────────────────────────────────────┐
│ error: Failed to parse PMC metric 'bad_metric': value out of range │
└────────────────────────────────────────────────────────────────────┘
```

- Interior width = `max(56, longest line)` capped at 116 characters,
  with one space of padding on each side.
- Long messages wrap at the chosen interior width. Greedy word-wrap;
  tokens longer than the interior width are hard-broken.
- The `error:` keyword is INSIDE the box, on the first line of the
  message.
- `what()` is single-line trimmed (everything from the first newline
  on is dropped before wrapping) so that timemory's exception text
  with embedded backtraces does not leak into the box.

UTF-8 box-drawing characters used when colored:

- `┌` U+250C, `┐` U+2510, `└` U+2514, `┘` U+2518, `─` U+2500, `│` U+2502.

ASCII fallback when not colored: `+`, `-`, `|` (corners and edges all
use the same characters, gdb-style).

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

If the function name is unresolved, the address is rendered as a hex
literal in its place:

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

| Element                | Style                | ANSI         |
|---                     |---                   |---           |
| `error:`               | bold bright red      | `\033[1;91m` |
| box border             | dim gray             | `\033[90m`   |
| `#N` frame index       | dim gray             | `\033[90m`   |
| function name          | cyan                 | `\033[36m`   |
| `in` keyword           | dim                  | `\033[2m`    |
| file path              | dim gray             | `\033[90m`   |
| `:line`                | yellow               | `\033[33m`   |
| `[inlined]`            | dim gray             | `\033[90m`   |
| `[+0xNN]`              | dim gray             | `\033[90m`   |
| `(module)`             | dim gray             | `\033[90m`   |
| `... N skipped ...`    | dim                  | `\033[2m`    |
| `... N more ...`       | dim                  | `\033[2m`    |

Reset is `\033[0m`. Every styled span ends with reset; color never
crosses a frame boundary.

## Env-var overrides

| Var                            | Effect                                              |
|---                             |---                                                  |
| `NO_COLOR`                     | flips the default `with_color` to false (any value) |
| `ROCPROFSYS_TRACE_VERBOSE=1`   | with_offset=true, with_module=true, max_frames_shown=64 |
| `ROCPROFSYS_TRACE_NO_FILTER=1` | clear default skip filters                          |

`ROCPROFSYS_*` overrides are read once at the first call to
`to_string()` per process and cached.

## Removed from earlier drafts

- `color_mode` enum (replaced by `with_color : bool`).
- `auto_detect` mode (no TTY sniffing in the formatter; callers decide
  or accept the default).
- `CLICOLOR_FORCE` env var (no longer consulted).

## Defaults summary

```
trace_format_options{
  with_color         = true,
  with_module        = true,
  with_offset        = false,
  with_file_line     = true,
  skip_substrings    = default_skip_filters(),
  max_function_width = 60,
  max_frames_shown   = 32,
}

exception_format_options{
  trace_options                  = {},
  with_color                     = std::nullopt,  // resolved against NO_COLOR
  capture_trace_at_call_site     = true,
}
```

## Hard rules (test-locked)

1. The `error: ...` header sits inside a single-row box (multi-row when
   the message wraps).
2. Each trace frame is exactly one line.
3. Frame indices are contiguous (0, 1, 2, ...) and monotonically
   increasing in print order.
4. Skip and truncate trailers each appear at most once per trace.
5. Color escapes never cross frame boundaries: every styled token is
   `<style><content>\033[0m`.
6. With color on, the box uses UTF-8 box-drawing characters; with color
   off, the box uses pure ASCII (`+`, `-`, `|`).
7. `error:` is always bright-red bold when colored.
8. `NO_COLOR` env disables color when `with_color` is unset; explicit
   `with_color` always wins.
