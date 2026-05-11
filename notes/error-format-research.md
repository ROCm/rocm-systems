# Error format research

Survey of best-practice stacktrace + diagnostic formats from 6 mainstream
runtimes / tools, distilled into a concrete spec for rocprofsys.

Source material is recalled from the documented behavior of each project.
URLs noted but not fetched (no network access in this session).

---

## 1. Rust panic, `color-eyre`, `anyhow`

Reference: <https://docs.rs/color-eyre/>, `RUST_BACKTRACE` doc.

Default panic without backtrace:

```
thread 'main' panicked at 'value out of range', src/main.rs:14:9
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
```

With `RUST_BACKTRACE=1`:

```
thread 'main' panicked at 'value out of range', src/main.rs:14:9
stack backtrace:
   0: std::panicking::begin_panic
   1: my_crate::parse_metric_value
             at src/parse.rs:142:9
   2: my_crate::collector::setup
             at src/collector.rs:88:9
   3: my_crate::main
             at src/main.rs:14:5
note: Some details are omitted, run with `RUST_BACKTRACE=full` for a verbose backtrace.
```

`color-eyre` adds:

- Section header `Stack backtrace:` in bold red.
- Frame index in dim gray.
- Function path in cyan (module separator `::` highlighted).
- File path in dim gray, `:line:col` in yellow.
- "frames omitted" message at top and bottom for runtime / panic-machinery.
- Optional `--span-trace` if tracing instrumented.

Key takeaways:

- Top frame is the deepest (most recent call), typed first.
- Per-frame layout is **two lines**: index + function on line 1, `at file:line:col` on line 2.
- Source excerpts are NOT shown by default; only on `RUST_LIB_BACKTRACE=full` + a special handler.
- Panic header is separated from the trace by a blank line.
- `NO_COLOR` env var disables ANSI.

---

## 2. Python tracebacks

Reference: <https://docs.python.org/3/library/traceback.html>.

Default:

```
Traceback (most recent call last):
  File "/home/u/app.py", line 14, in <module>
    main()
  File "/home/u/app.py", line 9, in main
    parse_value(s)
  File "/home/u/app.py", line 4, in parse_value
    raise ValueError("value out of range")
ValueError: value out of range
```

Notes:

- Header literally says **"most recent call last"**. The deepest frame is at
  the BOTTOM, opposite of Rust. Rationale: scanning the trace top-to-bottom
  follows the call chain, so the cause line and the exception type appear
  next to each other - the eye stops at the bottom right where the message
  is.
- Each frame is two lines: `File "...", line N, in funcname` then the
  literal source line indented two spaces.
- `SyntaxError` adds a third line with a `^^^^^` caret under the offending
  token, similar to clang.
- Python 3.11+ adds carets under the failing sub-expression on regular
  exceptions too (PEP 657).
- Chained exceptions: `During handling of the above exception, another
  exception occurred:` separator. Cause vs. context distinguished
  ("The above exception was the direct cause of the following exception:").

Key takeaway: bottom-most-frame ordering puts the error type + message at
the visual focus point. Rust does the opposite. Both have advocates - Rust's
ordering is friendlier when the trace is long and you scroll back; Python's
is friendlier when the trace is short and you read top-to-bottom.

For a CLI/profiler tool that may emit very long traces, Rust ordering wins:
the user sees the error type immediately and can stop reading once they
have enough context.

---

## 3. Clang / GCC diagnostics

Reference: clang docs - `-fdiagnostics-color`, `-fcaret-diagnostics`.

```
src/parse.cpp:142:14: error: no matching function for call to 'parse_value'
  142 |     auto v = parse_value(s);
      |              ^~~~~~~~~~~
src/parse.hpp:88:5: note: candidate function not viable: requires 2 arguments
   88 | int parse_value(std::string_view s, int radix);
      |     ^
```

Format breakdown:

- `file:line:col:` separated by colons; clickable in most terminals.
- Severity keyword (`error:`, `warning:`, `note:`) in bold color (red, magenta, cyan).
- Quoted source line numbered in a margin column.
- Caret `^` and tilde `~~~` underline the relevant token range.
- Follow-up `note:` lines are indented relative to the primary error.

ANSI palette (`-fdiagnostics-color=always`):

- `error:` bright red bold
- `warning:` bright magenta bold
- `note:` bright cyan bold
- caret `^`/`~~~` bright green bold
- file paths bold (no color), `:N:N` bold
- column margin and `|` separators dim

Key takeaways for rocprofsys:

- The `file:line:col:` prefix is well-established; users muscle-memory know it.
- Caret + colored source line is gold for the failing line - opt-in is
  reasonable since it requires file IO.
- Severity word in red bold is universally readable.

---

## 4. V8 / Node `Error.stack`

Reference: ECMA `Error.stack` non-standard property, V8 implementation.

```
ValueError: value out of range
    at parseValue (/home/u/app.js:4:15)
    at main (/home/u/app.js:9:5)
    at Object.<anonymous> (/home/u/app.js:14:1)
    at Module._compile (node:internal/modules/cjs/loader:1126:14)
```

Notes:

- Compact one-line frames: `    at <function> (<file>:line:col)`.
- Indented 4 spaces.
- Top is the throw site, bottom is the entry point.
- Anonymous frames render as `<anonymous>` or `Object.<anonymous>`.
- No color. No source excerpt. Designed for log scraping.

Key takeaway: a single-line-per-frame format is the most compact possible
and works well when frames are NOT deep. For native code with mangled
C++ names that can hit 200+ chars per name, single-line is hostile to
80-column terminals. We want 2 lines per frame.

---

## 5. Java `Throwable.printStackTrace`

Reference: JDK Throwable javadoc.

```
java.lang.RuntimeException: value out of range
    at com.example.Parser.parseValue(Parser.java:142)
    at com.example.Collector.setup(Collector.java:88)
    at com.example.Main.main(Main.java:14)
Caused by: java.lang.NumberFormatException: For input string: "abc"
    at java.base/java.lang.Integer.parseInt(Integer.java:652)
    at com.example.Parser.parseValue(Parser.java:140)
    ... 2 more
```

Notes:

- Per-frame: `at <FQN.Method>(<File.java>:<line>)`. No column.
- `Caused by:` chains the underlying cause; the inner trace truncates
  with `... N more` once it shares a tail with the outer trace.
- Top frame is deepest (Rust ordering).
- No color (terminal-agnostic).

Key takeaways:

- The `... N more` suffix to elide shared tails of nested traces is a
  great pattern - and useful even for non-nested cases when many
  frames are filtered noise.
- "Caused by:" header is a clean separator for chained errors.
  We won't have nested exceptions in Phase 1, but reserve the header for
  Phase 2 (`std::nested_exception`).

---

## 6. Go panic + `runtime.Stack`

```
panic: value out of range

goroutine 1 [running]:
main.parseValue(0xc000010210, 0x3)
	/home/u/app.go:14 +0x55
main.main()
	/home/u/app.go:9 +0x32
exit status 2
```

Notes:

- Goroutine label per stack: `goroutine N [state]:`.
- Per-frame: function name on line 1 (with arg values printed!), file:line +0xPC offset on line 2.
- `+0xNN` after the file:line is the PC offset within the function - useful for `addr2line` post-processing.
- All frames shown. No filtering.
- No color.

Key takeaways:

- Per-frame two-line layout, function on top, location on bottom: matches
  the structure we want.
- Per-thread label (we'll use `thread N`) is necessary if/when we capture
  multi-thread traces.
- The `+0xNN` PC offset is useful and cheap; we expose it as `with_offset`
  but off by default.

---

## 7. `boost::stacktrace`

Reference: <https://www.boost.org/doc/libs/release/doc/html/stacktrace.html>.

Default `to_string`:

```
 0# parseValue at /home/u/app.cpp:142
 1# main at /home/u/app.cpp:14
 2# __libc_start_main in /lib/x86_64-linux-gnu/libc.so.6
 3# _start
```

Notes:

- One frame per line, indexed.
- `<func> at <file>:<line>` if symbol+line resolved, else
  `<func> in <module>`, else address only.
- Module shown only when file:line missing.
- No color.
- No filtering of libc frames.

`std::stacktrace::to_string` (gcc 14, C++23) defaults are similar: index
prefix, function/file:line on one line, no color, no filtering.

Key takeaway: the C++ STL default is acceptable but unstyled and unfiltered.
We should beat it on both axes.

---

## 8. `backtrace_symbols` glibc default

```
./app(_Z10parseValueRKSs+0x42) [0x401234]
./app(main+0x18) [0x401190]
/lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0xf3) [0x7f...]
./app(_start+0x2e) [0x40103e]
```

Notes:

- Mangled function names (`_Z10parseValueRKSs`).
- Module name in parens before `(symbol+0xNN)`.
- Address in `[brackets]` at the end.
- No file:line ever.
- No color.
- No demangling.

Key takeaway: the bad-default baseline. Demangle, drop the address by
default, resolve file:line via libdw, and you have an order-of-magnitude
better trace.

---

## Adopted spec for rocprofsys

Synthesizing the above into the format we'll ship:

### Ordering: top-most frame first (Rust / Java)

Rationale: native traces with libdw often have 30-60 frames. The user
needs to see the throw site immediately, not after scrolling past
runtime + main + `__libc_start_main`. Putting the deepest frame at the
top mirrors what `color-eyre` and Java do, and what most modern CLIs
expect. The error message stays at the top above the trace, so it's
seen first regardless of trace ordering.

### Per-frame layout: two lines

```
  at <demangled function name>
     in <relative file path>:<line>
```

Function name on top, location on bottom (Go ordering). Using `at` /
`in` keywords (Java + Rust style). 2-space outer indent + 5-space
continuation indent for the file line, so the file path visually
nests under the function.

Trailing tags `[inlined]`, `[+0xNN]` in dim gray on the function line
when their flags are on.

### Color palette (TTY + not `NO_COLOR`)

Element                         | Style
---                             | ---
`error:` keyword                | bold bright red
function name                   | cyan
file path                       | dim gray
`:line` and `:line:col`         | yellow
`[inlined]`                     | dim gray
`[+0xNN]` offset                | dim gray
`... N frames skipped ...`      | dim
caret `^^^^^` (excerpt mode)    | bold green
literal source line             | normal

Inspired directly by clang + color-eyre.

### Default skip filters (substring on demangled name)

- `rocprofsys::diagnostic::stacktrace::capture`
- `__libc_start_main`
- `_start`
- `__GI___`
- `__cxa_throw`, `_Unwind_RaiseException`, `__cxa_rethrow`
- `__sanitizer::`, `__asan_`, `__ubsan_`, `__tsan_`
- `std::__throw_`, `std::__detail::`

When N frames are dropped, append a single line:

```
  ... 4 frames skipped (libc, runtime) ...
```

### Truncation

Default `max_frames_shown = 32`. When more frames exist, keep top-32 and
emit a Java-style trailer:

```
  ... 12 more ...
```

### Source excerpt (opt-in)

Off by default (file IO + may enlarge the message dramatically). When
on, show 1 line of context above and below the failing line, with a
6-char column margin and a `|` divider per clang. The caret `^^^^^`
underlines the failing line; we don't have token-level precision so it
spans the whole line.

### Address column

Never shown by default. Available via `with_offset` (appends
`[+0xNN]` to the function-name line) or `ROCPROFSYS_TRACE_VERBOSE=1`
env var override.

### Module column

Shown only when the module is NOT the main executable, in parens after
the function name in dim gray:

```
  at __libc_start_main (libc.so.6)
```

### Env vars

Var                              | Effect
---                              | ---
`NO_COLOR=1`                     | force color off (no-color.org standard)
`CLICOLOR_FORCE=1`               | force color on even when not a TTY
`ROCPROFSYS_TRACE_VERBOSE=1`     | enable `with_offset`, `with_module`, raise `max_frames_shown` to 64
`ROCPROFSYS_TRACE_NO_FILTER=1`   | disable default skip filters

### What we explicitly do NOT do

- No multi-thread traces (Phase 1 is current-thread only). No goroutine-
  style `thread N [state]:` header.
- No nested exception chain (`Caused by:`). Reserved for Phase 2 when we
  have the custom `rocprofsys::exception` hierarchy with `std::nested_exception`.
- No span trace / instrumentation linkage. Not applicable to native code.
- No async-await frame names. Not applicable.
