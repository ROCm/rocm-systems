# rocprofsys diagnostic format spec

Locked spec for the Phase-1 stacktrace + exception formatter. Derived
from `error-format-research.md`. Examples below are normative (all
tests check against them).

## Header line

```
error: <exception::what()>
```

`error:` is bold bright-red when color is on. The whole header is on
one line. If `what()` contains a newline (the timemory `core/exception.cpp`
embeds backtrace text in `what()`), strip everything from the first
newline onward when the trace will be appended separately.

A blank line follows the header before the trace.

## Default frame format

```
  at <demangled function>
     in <relative path>:<line>
```

- 2-space outer indent.
- 5-space continuation indent for the location line so the path nests
  visually under the function.
- `at` and `in` keywords match the Rust `color-eyre` style.
- Function name in cyan when color is on.
- Path in dim gray, `:line` in yellow.

If line info is not resolved, fall back to:

```
  at <demangled function>
     in <relative path>
```

If file is not resolved either, only the `at` line is printed.

If the function name itself is unresolved, render the address:

```
  at 0x00007f8c4a3b1c40
     in <relative path>:<line>
```

## Inlined frames

When libdw reports an inline-frame chain for a single instruction, each
inlined frame is rendered as its own entry with a trailing `[inlined]`
tag in dim gray:

```
  at rocprofsys::pmc::collectors::gpu::collector::setup() [inlined]
     in source/lib/.../collector.hpp:88
```

Inline-frame ordering: outer-to-inner. So the function the user wrote
the call in appears above the inlined callee. (Matches what GCC's
`-rdynamic` + libdw `dwfl_module_addrname` ordering returns.)

## Skipped frames

Skip filters are applied to the demangled function name (substring
match). When any frames are dropped, append exactly one line at the end:

```
  ... 4 frames skipped (libc, runtime) ...
```

The category in parens is a constant string derived from the skip-list
group; if mixed groups, drop the parens.

## Truncated frames

When `max_frames_shown` is exceeded, keep the top `max_frames_shown`
frames and append:

```
  ... 12 more ...
```

(Java style.) The "skipped" line and the "more" line can both appear in
the same trace if both conditions hit.

## Module suffix

When `with_module=true` AND the frame's module is not the main exe,
append `(<module-basename>)` in dim gray to the function line:

```
  at __libc_start_main (libc.so.6)
```

When the frame is in the main executable, never show the module.

## Offset suffix

When `with_offset=true`, append `[+0xNN]` in dim gray to the function
line, where `NN` is the PC offset within the resolved symbol:

```
  at rocprofsys::pmc::sampler::setup() [+0x1ac]
     in source/lib/.../sampler.cpp:92
```

## Source excerpt mode

When `with_source_excerpt=true`, after the location line, show one
line of context above and below the failing line, with the failing line
underlined by carets:

```
  at rocprofsys::pmc::collectors::gpu::parse_metric_value(...)
     in source/lib/.../parse.cpp:142
        140 |     auto v = std::stoul(s);
        141 |     if(v > MAX_METRIC) {
        142 |         throw std::runtime_error("value out of range");
            |         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        143 |     }
```

Caret span = full trim()ed length of the failing line. We don't have
column-precise diagnostics; this is a best effort.

The 8-char left margin holds the line number, right-aligned, then ` | `
divider. The caret line uses a blank line-number column followed by ` | `
then the caret string. The caret line is in bold green.

If the file cannot be read (build moved, container, stripped), silently
skip the excerpt for that frame (do not error out).

## Color choices and rules

Element                       | Style                | ANSI
---                           | ---                  | ---
`error:`                      | bold bright red      | `\033[1;91m`
function name                 | cyan                 | `\033[36m`
`at` / `in` keywords          | dim                  | `\033[2m`
file path                     | dim gray             | `\033[90m`
`:line` and `:line:col`       | yellow               | `\033[33m`
`[inlined]`                   | dim gray             | `\033[90m`
`[+0xNN]`                     | dim gray             | `\033[90m`
`(module)`                    | dim gray             | `\033[90m`
`... N skipped ... `          | dim                  | `\033[2m`
`... N more ...`              | dim                  | `\033[2m`
caret `^^^^^`                 | bold bright green    | `\033[1;92m`
literal source line           | none                 | -
line-number margin / divider  | dim gray             | `\033[90m`

Reset is `\033[0m`. Always emit reset at the end of every styled span;
never let color leak across lines.

## Color decision rules

```
color = format_options.color
if color == auto_detect:
    if env NO_COLOR is set (any value): color = off
    elif env CLICOLOR_FORCE=1 is set: color = on
    elif isatty(fd 2): color = on
    else: color = off
```

`fd 2` is the default because `print_exception()` writes to stderr.
For programmatic `to_string()` use, callers pass an explicit
`color_mode::on/off`.

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
  color                = auto_detect,
  with_module          = true,
  with_offset          = false,
  with_file_line       = true,
  with_inlined         = true,
  with_source_excerpt  = false,
  skip_substrings      = default_skip_filters(),
  max_function_width   = 100,
  max_frames_shown     = 32,
}
```

## Hard rules (test-locked)

1. The `error: ...` header is on its own line. Never wrapped.
2. Each frame is exactly 2 lines (3 if inlined and that's added on the
   same `at` line; never split across more lines in default mode).
3. With `with_source_excerpt=true`, the excerpt block is at most 5 lines
   per frame (3 source + 1 caret + at most 1 padding).
4. Skip and truncate trailers each appear at most once per trace.
5. Color escapes never cross frame boundaries: every styled token is
   `<style><content>\033[0m`.
6. `NO_COLOR` always wins over `CLICOLOR_FORCE`.
