# Design Proposal: `rocprofsys::common::path` — a unified C++20 filesystem module

Proposes a single, elegant filesystem module that covers **every** path/filesystem
need catalogued in [`analysis-filesystem-usage.md`](analysis-filesystem-usage.md),
replacing `tim::filepath::*`, the syscall-based guts of `common::path`, and the
~20 files of scattered raw syscalls / ad-hoc string slicing.

Reference for the primitives used here: [`reference-filesystem-apis.md`](reference-filesystem-apis.md).

> **Note on `(issue #N)` tags.** Section notes tagged *(issue #N)* record design decisions
> that resolve items raised in the design review; the resolutions are folded into this doc,
> which is self-contained (the separate review file is no longer required to understand them).

> **Standard note.** The `programming-cpp` skill states C++17; this project's
> `CMakeLists.txt` sets `CMAKE_CXX_STANDARD 20` and the user asked for a C++20
> module. This design targets **C++20**: `std::filesystem` (a C++17 library),
> plus C++20 `std::string_view::starts_with`/`ends_with` for the classification
> helpers and *one optional* `concept` to constrain the stream template. Nothing
> here needs C++23 (`std::string::contains`, `std::ranges::to`) or
> `std::formatter<fs::path>` (which does not exist in C++20).

---

## 1. Goals

1. **One home for path logic.** Every site in the survey resolves to this module
   (or the two thin domain modules built on it — §4). No more three-way split.
2. **Elegant & minimal API.** Fold the many hand-rolled variants (5 `dirname`s,
   2 `basename`s, 2 `exists` semantics, 2 `is_text_file`s (→ one `is_elf`), the tripled `open`
   shim, dozens of `find_last_of('/')` slices) into a small, orthogonal set of
   functions. The user's headline example — "strip one or two levels" — becomes
   a single `parent_path(p, levels)`.
3. **Preserve load-bearing semantics** (the survey's traps): `realpath`
   fallback-to-input, `open` auto-mkdir + fallback. **Deliberately fix** the one
   trap that is a bug — the two `exists` semantics collapse to a single true-for-dirs
   `exists()` (§5.4), a conscious behavior change, not a preserved quirk.
4. **C++20-idiomatic, testable.** No *filesystem* error throws across the API —
   functions catch internally and return a fallback (match the timemory contract).
   Functions are **not** `noexcept` unless genuinely non-allocating, so a `bad_alloc`
   propagates rather than terminating (§5.0). `[[nodiscard]]` where ignoring is a bug,
   owning returns (kills the `basename` dangling hazards), no `PATH_MAX` buffers.

## 2. Design principles (mapped to Core Guidelines)

| Principle | Application |
|-----------|-------------|
| P.1/F.1 "express ideas directly / name operations" | `parent_path(p, 2)` instead of `dirname(dirname(p))` |
| P.11 "encapsulate messy constructs" | `PATH_MAX` buffers, `stat`+`S_IS*`, `opendir` loops, and the mkdir-then-open dance are hidden behind named functions |
| F.6/E "error strategy" | All I/O functions report **filesystem** failure by value (`bool` / empty string / `0` / input). They call the **throwing** `fs` overloads wrapped in a `catch(...)` that returns the fallback — **no `error_code` overloads** (we never inspect *why* a call failed). **No `filesystem_error` crosses the API** (critical at the `LD_PRELOAD` boundary). Functions are **not** `noexcept` unless genuinely non-allocating (they build `std::string`/`fs::path`/`std::vector`, so `bad_alloc` may propagate — see §5.0) |
| F.20/F.21 "return, don't out-param" | Return `std::string` / `std::optional` / small structs; drop the dead `realpath` `_resolved` out-param |
| I.4 "strongly typed" | One `exists()` (existence) plus orthogonal `is_directory()` / `is_regular_file()` type predicates — not three `*_exists` variants; `levels` is a named count, not a magic loop |
| C.4/single-responsibility | Core path ops separated from rocprofsys install-layout and from dl introspection (§4) |
| Per.* | Lexical ops are pure `fs::path` string surgery (no syscalls); `string_view` inputs avoid copies; `directory_entry` caches `stat` |

## 3. Foundation decision

- **Build on `std::filesystem`** for everything *except* `realpath`, **uniformly across
  all consumers including the `LD_PRELOAD`ed `rocprof-sys-dl` library** — the same bodies
  everywhere. Do **not** keep a separate raw-syscall implementation for `dl`.
  - Since GCC 9.1/Clang 9 `std::filesystem` lives in `libstdc++.so.6`, which `dl` already
    pulls in (it uses `std::string` / `std::ofstream` / `throw`). So `fs` adds no new
    dependency in the injected context. It eliminates the `PATH_MAX` buffers, the
    `opendir` boilerplate, and the dangling-`basename` hazards.
  - `path.hpp` is a header-only `INTERFACE` library, so its bodies compile into *every*
    consumer. A `dl`-only syscall variant would require `#ifdef`ed bodies — a fork of the
    very code we are consolidating. Uniform `fs` is both safer (see error handling) and
    structurally simpler.
- **Error handling — one pattern (see §5.0 for the full policy).** Every path function
  uses the **throwing** `fs` overloads wrapped in a `catch(...)` that returns the
  fallback value. **No `error_code` overloads.** Our contract collapses every filesystem
  failure into one fallback value; we never inspect *why* it failed, which is the only
  thing `error_code` adds over a `catch`. Functions are **not** `noexcept` unless
  non-allocating (§5.0), so `bad_alloc` propagates instead of terminating.
  - This keeps every *filesystem* error inside our own stack frame — no `filesystem_error`
    unwinds across the `LD_PRELOAD` boundary into target code (`dl` already throws today,
    e.g. `dl.cpp:1324`, `dl.cpp:1532`, and already allocates strings at every path call
    site, so an unavoidable `bad_alloc` is no new risk there).
  - Zero happy-path cost: table-based unwinding on Linux/GCC means a `try` that does not
    throw costs nothing; predicates (`exists` / `is_directory` / `is_regular_file` /
    `remove`) don't throw on "not found" anyway, so throws are rare.
  - Revisit only if profiling ever shows a specific wrapper throwing inside a hot loop —
    none exists today (the hot loops are `exists`/`realpath`, which don't throw-on-miss /
    stay POSIX).
- **Keep POSIX `::realpath` (with fallback-to-input) for `path::realpath`.**
  This is the one place `fs` has no byte-identical twin: `fs::canonical` throws on
  missing paths, `fs::weakly_canonical` *normalizes* the fallback (changing map-key
  / `==` identity — see survey G4). Wrapping `::realpath` preserves current behavior
  exactly. (Documented, single function; the verbatim-fallback is a **pinned test
  invariant** — §7, issue #4.) This is `dl`'s most-used path op, so
  `dl` still bottoms out on a syscall there.

### Verified (both `dl`-uniformity concerns cleared)
- **(a) Static-libstdc++ builds — SAFE.** `ROCPROFSYS_BUILD_STATIC_LIBSTDCXX` /
  `ROCPROFSYS_BUILD_STATIC_LIBGCC` are OFF by default (`BuildSettings.cmake:60,63`).
  `fs` is available in *both* configs: dynamic → `libstdc++.so.6` loaded as a `dl.so`
  dependency; static (`-static-libstdc++`) → libstdc++ (which includes `std::filesystem`
  since GCC 9.1, no `-lstdc++fs`) is baked into `dl.so`. No config leaves `fs`
  unavailable. Bonus: the `catch(...)` also neutralizes the static-libstdc++
  cross-`.so` `type_info` hazard (every filesystem throw is caught in the same `.so`;
  `catch(...)` needs no RTTI match).
- **(b) Signal / pre-init context — SAFE.** No `__attribute__((constructor))`, no signal
  handlers, no `pthread_atfork` in `dl.cpp` / `main.c`. Both `dl` path-call sites are
  on-demand in normal execution: the `indirect` ctor (`find_path`/`dirname`/`basename`)
  is a lazy Meyers singleton (`dl.cpp:480 static auto* _v = new indirect{...}` in
  `get_indirect()`, reached only from API entry points); `readlink("/proc/<pid>/exe")`
  (`dl.cpp:1246`) runs in the tooling-init / `push_trace` path. libc + libstdc++ are
  fully initialized by then, so `fs` allocation is safe.

## 4. Module structure

Split the current 445-line "god header" by responsibility. All three are
header-only (the `common-library` is an `INTERFACE` target) and timemory-free.

```
source/lib/common/
  path.hpp            // THE module: pure path + filesystem ops (this proposal)
  install_layout.hpp  // rocprofsys-specific: get_rocprofsys_root / get_internal_* (was in path.hpp)
  link_map.hpp        // dlopen/dlinfo introspection: get_link_map / get_origin (was in path.hpp; NOT a filesystem op)
```

- `path.hpp` knows nothing about rocprofsys layout or the dynamic linker.
- `install_layout.hpp` and `link_map.hpp` **consume** `path.hpp`. This removes the
  God-header smell (survey cross-cutting observations) while keeping call sites' includes small.
- Namespace: **keep `rocprofsys::common::path`** (decided — §11.1; zero churn on the ~40
  existing `common::path::` sites). `path::` throughout this doc is shorthand — use
  `common::path::` directly or a per-file `namespace path = ::rocprofsys::common::path;`
  alias.

---

## 5. Proposed API (`namespace rocprofsys::common::path`)

All functions are `inline`. Inputs are `std::string_view` (zero-copy); outputs are
owning `std::string` unless noted.

### 5.0 Exception & `noexcept` policy (issue #1)

Two separate concerns, decided independently:

1. **Filesystem errors never escape.** Every function that touches disk calls the
   **throwing** `fs` overload inside a `catch(...)` that returns the fallback value
   (`false` / `""` / input / `0`). This preserves the timemory "never throw on fs
   errors" contract and — critically — guarantees **no `filesystem_error` unwinds
   across the `LD_PRELOAD` boundary** into target code. (The `dl` shim already
   allocates strings and already `throw`s `std::runtime_error` in normal operation —
   `dl.cpp:1324`, `dl.cpp:1532` — so this boundary tolerates our own-frame throws; we
   simply never let a *filesystem* error be the thing that unwinds.)

2. **`noexcept` is dropped from anything that allocates.** Nearly every function
   builds a `std::string` / `fs::path` / `std::vector` (even `exists` constructs
   `fs::path{p}` internally), so `std::bad_alloc` can propagate. Marking such a
   function `noexcept` would turn an OOM into `std::terminate` — issue #1.
   We therefore **do not** mark allocating functions `noexcept`; a propagating
   `bad_alloc` is honest and, in this tool, effectively fatal anyway (the `dl` ctor
   allocates today with the same consequence). The internal `try/catch` stays — it is
   about the *contract* (fs errors), not the compiler *guarantee* (`noexcept`).

   **The only functions that keep `noexcept`** are the genuinely non-allocating
   lexical predicates that operate purely on `string_view`s and return `bool`:
   `is_absolute`, `is_relative`, `has_extension`, `has_any_extension` (front-char /
   suffix comparisons — no allocation, no `fs::path` construction). Every other
   signature below therefore has **no `noexcept`** (a change from earlier revisions of
   this doc, which marked I/O functions `noexcept`).

### 5.1 Path decomposition — lexical, pure, no I/O

```cpp
/// Parent directory, optionally walking up `levels` times. Strips the final
/// component `levels` times; == `levels` iterations of fs::path::parent_path().
/// Pure lexical strip — no normalization (use normalize() for that).
/// parent_path("/a/b/c")        -> "/a/b"
/// parent_path("/a/b/c", 2)     -> "/a"        (replaces dirname(dirname(x)))
/// parent_path("file", 1)       -> ""          (no separator)
[[nodiscard]] std::string parent_path(std::string_view p, unsigned levels = 1);

/// Final component. filename("/a/b.so") -> "b.so". Owning => no dangling.
[[nodiscard]] std::string filename(std::string_view p);

/// Component without final extension. stem("/a/b.tar.gz") -> "b.tar".
[[nodiscard]] std::string stem(std::string_view p);

/// Final extension, WITH dot. extension("/a/b.so") -> ".so"; ("a") -> "".
[[nodiscard]] std::string extension(std::string_view p);

/// Collapse '.'/'..'/'//': normalize("a/./b/../c") -> "a/c". No disk access.
[[nodiscard]] std::string normalize(std::string_view p);

[[nodiscard]] bool is_absolute(std::string_view p) noexcept;   // p starts with '/'
[[nodiscard]] bool is_relative(std::string_view p) noexcept;
```

> **Not provided: `join()` (issue #7).** An earlier revision proposed a variadic
> `join(...)` to fold `TIMEMORY_JOIN` / `common::join` / `fmt::format("{}/{}")`. Dropped:
> the sibling branch **`tim-rem-string-manipulation`** already removes both `TIMEMORY_JOIN`
> and `common::join` (deleting `common/join.hpp`) and standardizes path-joining on
> **`fmt::format("{}/{}", …)`** (plain textual concatenation with a slash). So there is no
> join abstraction left to consolidate, and adding `path::join` would *re-abstract* exactly
> what that branch de-abstracts. It would also reopen the semantics question raised in
> review (`fs::path::operator/` discards the left side on an absolute right operand, vs.
> textual concat) — moot once we simply use `fmt::format`. The module's own internal
> path-building uses `fmt::format("{}/{}", …)` accordingly. **Coordination:** this design
> assumes the post-`tim-rem-string-manipulation` state (no `common::join`); the two overlap
> in `path.hpp`/instrument/config, so sequence them together.

> **Not provided: `make_absolute()` / `current_dir()` (issue #3).** An earlier
> revision proposed `make_absolute(p, base = current_dir())` to fold the
> `if(p[0] != '/') p = pwd + "/" + p` idiom (config.cpp ×6, attach, instrument). Dropped,
> because: (1) `getcwd()` and `$PWD` are **not** timemory — no site here goes through
> `tim::filepath` (config.cpp uses raw `getenv`/`getcwd`; instrument's `get_cwd()` is a
> local helper), so there is no removal mandate to touch them; (2) the two consumer
> families want **opposite** cwd semantics — output-path construction needs the *logical*
> `$PWD` (config.cpp/attach, user-visible → Epic §10), while instrument's input-path
> resolution needs the *physical* `getcwd()` — so no single primitive serves both without
> the caller choosing anyway; and (3) each config.cpp site wraps the result in
> `settings::format(..., tag)`, so a generic helper would not cleanly drop in. Per
> Simplicity First these stay in place, behavior provably unchanged. (Physical-from-logical
> would be `realpath(current_dir())` if ever needed.)

**`parent_path` full edge contract (issue #5).** The headline function replaces
`dirname(dirname(realpath(exe)))`, so its behavior at/over the root is specified exactly:

| Input | Result | Rule |
|-------|--------|------|
| `parent_path("/a/b/c")` | `/a/b` | strip one component |
| `parent_path("/a/b/c", 2)` | `/a` | strip two |
| `parent_path("/a/b/c", 3)` | `/` | reached root |
| `parent_path("/a", 1)` | `/` | **absolute clamps at root** |
| `parent_path("/a", 5)` | `/` | over-walk stays at `/` |
| `parent_path("/", n)` | `/` | root's parent is `/` |
| `parent_path("a/b", 1)` | `a` | relative |
| `parent_path("a/b", 5)` | `""` | **relative bottoms out at empty** |
| `parent_path("file", 1)` | `""` | no separator |
| `parent_path("file", 3)` | `""` | over-walk stays empty |
| `parent_path(p, 0)` | `p` | identity |
| `parent_path("/a/b/")` | `/a/b` | trailing slash: empty last component stripped first |

Rules: **absolute paths clamp at `/`**; **relative paths bottom out at `""`**; `levels == 0`
is identity; **pure lexical strip, no normalization**.

**Behavior-change caveats (document in migration):**
- **Root over-walk** vs. `common::path::dirname` (`substr(0, find_last_of('/'))`):
  `dirname("/a")` → `""` today; `parent_path("/a")` → `/` (fs clamps at root). Not reached
  by any surveyed site (all operate on deep paths).
- **No-slash edge differs by which `dirname` a site migrates from:**
  - `common::path::dirname` already returns `""` → `parent_path` **matches** (no change).
  - `tim::filepath::dirname` returns the **input unchanged** (verified: `filepath.cpp:302-303`)
    → `parent_path` → `""` is a **change**. `filepath` is aliased to `::tim::filepath`
    (`common.hpp:88`), so tim-side sites like `perfetto.cpp:280` fall here. In practice these
    receive *composed* output paths that contain a directory (e.g.
    `get_perfetto_output_filename` → `"{pwd}/perfetto-trace-{pid}.proto"`), so the slash-less
    case is not normally hit; if it were, `input → ""` is benign (empty ⇒ cwd).

### 5.2 Path classification — string tests (C++20)

```cpp
/// True if the final extension equals `ext` (with or without leading dot).
/// has_extension("x.so", ".so") == has_extension("x.so", "so") == true
[[nodiscard]] bool has_extension(std::string_view p, std::string_view ext) noexcept;

/// True if the path ends with ANY of the given extensions.
[[nodiscard]] bool has_any_extension(std::string_view p,
                                     std::initializer_list<std::string_view> exts) noexcept;

/// If the path ends with one of `exts`, return it stripped; else unchanged.
/// strip_known_extension("f.json", {".txt",".json"}) -> "f"
[[nodiscard]] std::string strip_known_extension(
    std::string_view p, std::initializer_list<std::string_view> exts);
```
These fold the ~30 brittle `find(".so")==len-3` / `find("lib")==0` / trailing-`.txt`
idioms (survey G16) onto `std::string_view::ends_with`/`starts_with`. *Contains*
checks (`find(".so.")!=npos`) stay as-is (`std::string::contains` is C++23).

### 5.3 Path resolution — touches disk

```cpp
/// Canonical absolute path via POSIX ::realpath; returns `p` verbatim on failure
/// (byte-identical to tim::filepath::realpath / common::path::realpath).
[[nodiscard]] std::string realpath(std::string_view p);

/// Symlink target (one level). Returns `p` if not a symlink. = ::readlink.
[[nodiscard]] std::string read_symlink(std::string_view p);
```

### 5.4 Existence & type — throwing `fs` + `catch(...)` (not `noexcept`, see §5.0)

One `exists()` for **existence**, plus two orthogonal **type** predicates. This is *not*
a three-way split of the existence check — it is one existence question plus two `is_*`
type queries, matching `std::filesystem`. All construct `fs::path{p}` internally, so
per §5.0 they are **not** `noexcept` (fs errors caught → fallback; `bad_alloc` propagates).

```cpp
/// Existence, ANY type (dir OR regular file OR valid symlink). FALSE for broken
/// symlinks and missing paths. == std::filesystem::exists (stat-based, follows symlinks).
/// This is THE single existence check — replaces tim::filepath::exists,
/// common::path::exists, and avail::file_exists.
[[nodiscard]] bool exists(std::string_view p);

/// Type check: is a directory (following symlinks). == fs::is_directory.
[[nodiscard]] bool is_directory(std::string_view p);

/// Type check: is a regular file (following symlinks). == fs::is_regular_file.
[[nodiscard]] bool is_regular_file(std::string_view p);

[[nodiscard]] bool is_symlink(std::string_view p);   // = common::path::is_link

/// True if the file begins with the ELF magic (0x7F 'E' 'L' 'F'). FALSE on
/// non-ELF, empty, too-short, or unopenable files. Pure predicate, no side
/// effects. Replaces the two is_text_file copies (with inverted call-site sense
/// and precise ELF semantics — see below).
[[nodiscard]] bool is_elf(std::string_view p);

/// Size in bytes of a regular file. Returns 0 on missing/error AND for an empty
/// (0-byte) file — the name makes the "missing → 0" collapse explicit at call sites
/// (issue #9). Matches the sole consumer (post_processor byte-summing), whose
/// local copy is removed in favor of this. Callers needing to distinguish missing from
/// empty gate with exists()/is_regular_file() first.
[[nodiscard]] std::uintmax_t file_size_or_zero(std::string_view p);
```

**Why `is_elf` replaces `is_text_file` (issue #2):** The survey's two
`is_text_file` copies (`details.cpp` + `path.hpp`) return `bool` via a NUL-byte scan and
**collapse "binary" with "cannot open"** — both yield `false`. Their two (and only)
consumers, `instrument.hpp:169`/`:237`, use that to *reject* text files
(`if(is_text_file(x)) errprintf(...)`), so a file that can't be opened returns `false` →
instrumentation **proceeds** and fails murkily later. Resolution:
- **A precise, generic `is_elf(p)`** (first 4 bytes == `0x7F 'E' 'L' 'F'`) replaces the
  NUL-scan heuristic. dyninst only instruments ELF objects (exe + `.so`), so this answers
  the question the call sites actually care about. `is_elf` carries no instrumentation
  knowledge, so it belongs in the generic `path.hpp` layer.
- **The call sites invert:** `if(is_text_file(x)) reject` → `if(!path::is_elf(x)) reject`.
  This puts the can't-open case on the **safe** side — a file we can't read is "not ELF" →
  rejected, instead of silently proceeding. The remaining conflation (can't-open vs.
  genuinely-not-ELF) is now *harmless* (both correctly reject); a 3-state enum was
  considered and rejected as unnecessary once the outcome is safe.
- **`is_elf` is a pure, side-effect-free predicate** — the current copies' `errprintf`/log
  on open-failure (which disagree: `details.cpp` exits, `path.hpp` only logs) is dropped;
  the error/exit decision stays at the two call sites.
- **Both old copies are deleted**, not migrated.

**Why one `exists()` + two `is_*`, not three `*_exists` (the survey's #1 trap):**
- The survey called the two coexisting `exists` semantics (`false-for-dirs` vs
  `true-for-dirs`) the #1 correctness trap — source of ≥3 latent bugs. The cure is a
  **single** `exists()` with `fs::exists` semantics (true-for-dirs, false-for-broken-link).
  Verified: no call site depends on the old false-for-dirs behavior; the sites that pass
  directories to a false-for-dirs check are the latent bugs — `argparse.cpp:136`
  (`_libdir`), `impl.cpp:398` (`_config_folder`), `instrument.cpp:296` (`omni_root`) —
  which this fixes. Broken-symlink → `false` matches the current *file*-site behavior
  (`tim::filepath::exists` is stat-based) exactly, so no file site regresses.
- **Rejected:** a three-name existence split (`file_exists`/`dir_exists`/`any_exists`)
  re-creates the "which `*exists*` has which semantics?" trap; a boolean parameter
  (`exists(p, include_dirs)`) hides the semantic behind an opaque argument.
- `is_directory` / `is_regular_file` answer a *different* (type) question and are needed
  only where code branches on the kind of thing (verified by call-site check):
  - `is_directory` — `list_dir_files` (§5.5), `instrument.cpp:2788` (+ its local
    `is_directory` helper), and `makedir` pre-guards (droppable — `make_dirs` is idempotent).
  - `is_regular_file` — **only** `get_absolute_filepath` (`instrument.cpp`), replacing its
    local `is_file` helper; its three uses reduce cleanly (`is_file` implies existence):
    `exists && is_file` → `is_regular_file`, `!exists || !is_file` → `!is_regular_file`.
  - `analysis.cpp:140,150` (`exists && _satisfies_binary_filter`) is a content/scope
    filter, not a type check → plain `exists()`.

**Migration test change (only guaranteed observable change for `exists`):**
`source/lib/common/tests/test_path.cpp:113` `Exists_BrokenSymlink` → flip `EXPECT_TRUE`
to `EXPECT_FALSE` (a broken symlink no longer "exists"). Verify with the `test_path`
suite + a smoke instrumentation run (this is a behavior change, not a pure refactor).

### 5.5 Directory operations

Throwing `fs` overloads inside a `catch(...)` returning the fallback. Not `noexcept`
(§5.0): fs errors caught → fallback; `bad_alloc` propagates.

```cpp
/// mkdir -p. Idempotent. Returns true if the dir exists afterward.
/// Replaces makedir/tim::makedir; the exists/is_directory pre-guards are dropped.
[[nodiscard]] bool make_dirs(std::string_view p);

/// Ensure the PARENT directory of a file path exists. make_parent_dirs("a/b/c.txt")
/// creates "a/b". Used by the open shim and DB/output-dir sites.
[[nodiscard]] bool make_parent_dirs(std::string_view file_path);

[[nodiscard]] bool          remove(std::string_view p);      // file/empty dir; true if gone (tolerates missing)
[[nodiscard]] std::uintmax_t remove_all(std::string_view p); // recursive

/// Entries (names, '.'/'..' excluded). Returns empty on missing/not-a-dir (no throw on fs error).
/// Overload with a predicate for filtering.
[[nodiscard]] std::vector<std::string> list_directory(std::string_view dir);
template <typename Pred>
[[nodiscard]] std::vector<std::string> list_directory(std::string_view dir, Pred keep);
```
`list_directory` replaces the raw `opendir`/`readdir` loops in `discovery.cpp` and
`preset_registry.cpp`, built on `fs::directory_iterator` (whose `directory_entry` caches
`stat`). The throwing `directory_iterator` overloads inside the `catch(...)` give the
desired behavior directly — missing/not-a-dir → empty; a mid-scan error → best-effort
partial — with **no `error_code`** needed:

```cpp
std::vector<std::string> list_directory(std::string_view dir) {
    std::vector<std::string> result;
    try {
        for (const auto& e : std::filesystem::directory_iterator{std::filesystem::path{dir}})
            result.emplace_back(e.path().filename().string());
    } catch (...) {}
    return result;
}
```

**`list_dir_files` migration (throw-preserving — zero behavior change).** Keep
`list_directory` as the generic "empty on missing" primitive; the `list_dir_files`
wrapper (`discovery.cpp:28-51`) re-adds its throw with an explicit `is_directory` check,
so its contract is preserved and **no test changes** are needed:
```cpp
data::directory_files_t
list_dir_files(const std::string& path)
{
    if(path.empty()) return {};
    if(!path::is_directory(path))
        throw std::runtime_error(fmt::format("Error opening directory: {}", path));
    return path::list_directory(path);
}
```
Drops `<dirent.h>`/`<memory>`/`<cerrno>`/`<cstring>`, the `unique_ptr`+`closedir` dance,
the manual `.`/`..` filter, and the `readdir` silent-truncation footgun.
`directory_files_t` is already `std::vector<std::string>` (`data_types.hpp:92`). Caveats:
`is_directory` throws on `ENOENT`/`ENOTDIR` but not `EACCES` (not a realistic scenario for
`tmp_directory`); and the throw preserves a pre-existing latent hazard (production caller
`cache_manager.cpp:53-58` doesn't `try/catch`, so a missing `tmp_directory` yields an
uncaught `runtime_error` — address separately if desired). Same migration applies to
`preset_registry.cpp`.

### 5.6 Streams — the auto-mkdir `open` shim (one template)

Collapses the 3 copy-pasted timemory bodies into one function. Whether it creates
the parent directory is decided at compile time by the stream's category.

```cpp
/// True if `stream` can write (ofstream/fstream) — controls the auto-mkdir branch.
template <typename S>
concept OutputStream = std::derived_from<S, std::ostream>;

/// Open `stream` on `filepath`, forwarding extra open args (e.g. ios::binary).
/// For output streams: creates the parent dir tree first, and on mkdir failure
/// falls back to "./<filename>" (preserves tim::filepath::open behavior exactly).
/// For input streams: no directory creation.
/// Returns stream.is_open() && stream.good().
template <typename StreamT, typename... Args>
[[nodiscard]] bool open(StreamT& stream, std::string_view filepath, Args&&... args)
{
    if constexpr (OutputStream<StreamT>) {
        std::string target{ filepath };
        if (!make_parent_dirs(target))
            target = "./" + filename(filepath);        // trap #3 fallback
        stream.open(target, std::forward<Args>(args)...);
    } else {
        stream.open(std::string{ filepath }, std::forward<Args>(args)...);
    }
    return stream.is_open() && stream.good();
}

/// C-stream variant (the single config.cpp tmp_file site). Auto-mkdir parent.
/// Not noexcept (§5.0): builds strings + calls make_parent_dirs.
[[nodiscard]] std::FILE* fopen(std::string_view filepath, const char* mode);
```
All 15 `open` sites and the 1 `fopen` site keep their exact call shape
(`if (open(ofs, name, std::ios::binary)) ...`). The `OutputStream` concept is the
*only* concept in the design (adopt-at-most-one; Simplicity First) — it just gives a
clean diagnostic; `std::is_base_of_v<std::ostream, StreamT>` in an `if constexpr`
works identically if concepts are undesired.

### 5.7 Process / environment paths

`current_dir()` is intentionally **not** provided — see the issue-#3 note in §5.1
(`getcwd`/`$PWD` are non-timemory and site-specific; left in place).

```cpp
[[nodiscard]] std::string temp_dir();                    // $TMPDIR else /tmp
[[nodiscard]] std::string executable_path();             // realpath("/proc/self/exe")

/// First existing "dir/name" (optionally trying lib/lib64 subdirs). Generic
/// search primitive underlying find_path / the dynamic_library resolver.
[[nodiscard]] std::string find_in_dirs(std::string_view name,
                                       const std::vector<std::string>& dirs,
                                       bool try_lib_subdirs = false);
```

### 5.8 Out of scope for `path.hpp` (live in the sibling modules)

- **`install_layout.hpp`** (`namespace rocprofsys::common::install`): `root()`,
  `libpath(lib)`, `libdir()`, `script_path()` — built from `path::executable_path`
  + `parent_path` + `fmt::format("{}/{}", …)` + `exists`. (Was `get_rocprofsys_root` /
  `get_internal_*`.) Also the domain-specific `find_path` + default search paths.
- **`link_map.hpp`** (`namespace rocprofsys::common::dl`): `link_map(name)`,
  `origin(name)` — `dlopen`/`dlinfo`. **Not filesystem** (survey G19); yields paths
  that flow into `path::realpath`.

  (Sibling namespaces stay nested under `common::` — same zero-churn rationale as §11.1;
  `install::` / `dl::` below are shorthand, like `path::`.)

---

## 6. Semantics preserved (the traps, made explicit)

| Behavior | Old | New (guaranteed identical) |
|----------|-----|----------------------------|
| realpath returns input on failure | `::realpath` + fallback | `path::realpath` (same POSIX call) |
| two `exists` semantics (false- vs true-for-dirs) | `tim::filepath::exists` / `common::path::exists` | single `path::exists` (true-for-dirs, false-for-broken-link) — **behavior change**: fixes 3 latent dir bugs; one test flip (`Exists_BrokenSymlink`) |
| open auto-creates parent + `./base` fallback | `tim::filepath::open` | `path::open` (`if constexpr` output branch) |
| makedir idempotent, EEXIST-ok | `filepath::makedir` | `path::make_dirs` (drops pre-guards ⇒ fixes 3 latent bugs) |
| basename owning vs borrowed | glibc `::basename` (dangling) | `path::filename` returns `std::string` |
| dirname no-slash edge | `tim::filepath::dirname` → input; `common::path::dirname` → "" | `parent_path` → "" — matches `common::path` (no change); a **change** only for `tim::filepath` sites, and only if fed a slash-less path (composed output paths normally carry a dir — §5.1) |

## 7. Testability

- **Lexical functions (§5.1/5.2)** are pure (`same input → same output`), trivially
  unit-tested with no filesystem — the bulk of the API. In particular, `parent_path`
  gets the full §5.1 edge table (root clamp, relative bottom-out, `levels=0/1/2`/over-walk,
  trailing slash, no-slash) plus the real use case `parent_path(realpath("/proc/self/exe"), 2)`.
- **I/O functions (§5.3–5.7)** are tested exactly as `test_path.cpp` does today:
  a `mkdtemp` fixture + real files/symlinks (fast, hermetic, already 49 tests). This
  is the idiomatic seam for filesystem code; a policy/DI abstraction over syscalls
  would add complexity for little gain (Per.2/Simplicity First) — **not** proposed.
- All the trap-preserving behaviors (fallback-to-input, auto-mkdir, `./base` fallback,
  idempotent mkdir) get explicit regression tests, as does the deliberate `exists`
  change (true-for-dirs, false-for-broken-link — incl. the flipped `Exists_BrokenSymlink`).
- **`realpath` verbatim-fallback is a pinned invariant (issue #4).** A regression
  test asserts that `realpath` on a **non-existent** path returns the input **byte-for-byte**
  — not normalized, not made absolute — since consumers use `realpath` results as
  `std::map`/`std::set` keys and in `==`/prefix comparisons (survey G4; e.g.
  `analysis.cpp:209`, `instrument.cpp:1206`, `preset_registry.cpp:190`). Suggested cases:
  `realpath("/does/not/exist") == "/does/not/exist"`, `realpath("./a/../b") == "./a/../b"`
  (relative, non-existent → unchanged), and `realpath(<existing symlink>)` → resolved
  target. **Consumers are deliberately left untouched** (out of scope) — they continue to
  rely on this invariant, which is exactly why it is pinned by test rather than treated as
  an implementation detail.

## 8. Migration mapping (old → new)

| Old spelling | New |
|--------------|-----|
| `tim::filepath::dirname(p)` / `common::path::dirname(p)` / `find_last_of('/')`+`substr` | `path::parent_path(p)` |
| `dirname(dirname(p))` / `dirname(dirname(realpath(exe)))` | `path::parent_path(p, 2)` |
| `tim::filepath::basename(p)` / `::basename` / `tool_runner::basename_of` | `path::filename(p)` |
| `_iname.find_last_of('.')`+`substr` | `path::extension(p)` / `path::stem(p)` |
| `tim::filepath::realpath(p,nullptr,false)` | `path::realpath(p)` |
| `tim::filepath::readlink(p)` | `path::read_symlink(p)` |
| `tim::filepath::exists(p)` / `common::path::exists(p)` / `avail::file_exists(p)` | `path::exists(p)` (single existence check) |
| `tim::filepath::direxists(p)` / `instrument.cpp is_directory` | `path::is_directory(p)` (type check) |
| `instrument.cpp is_file` (`exists && is_file`) | `path::is_regular_file(p)` (type check) |
| `common::path::is_link(p)` / `path_type` | `path::is_symlink(p)` |
| `is_text_file(p)` (both copies) | `path::is_elf(p)` — **call sites invert**: `if(is_text_file) reject` → `if(!is_elf) reject`; ELF-magic semantics; old copies deleted |
| `tim::filepath::makedir(p)` / `tim::makedir(p)` (+ pre-guards) | `path::make_dirs(p)` |
| `tim::filepath::open(ofs,p[,flags])` / `open(ifs,p)` | `path::open(ofs,p[,flags])` (unchanged shape) |
| `tim::filepath::fopen(p,mode)` | `path::fopen(p,mode)` |
| `TIMEMORY_JOIN('/',…)` / `common::join('/',…)` | *(unchanged — `fmt::format("{}/{}",…)`; join abstractions removed by `tim-rem-string-manipulation`; issue #7)* |
| `if(p[0] != '/') p = pwd + "/" + p` | *(unchanged — left in place; issue #3)* |
| `p.find(".so")==len-3` etc. | `path::has_extension(p,".so")` |
| strip trailing `.txt/.json/.xml` | `path::strip_known_extension(p,{".txt",".json",".xml"})` |
| `opendir`/`readdir` loop | `path::list_directory(dir, pred)` |
| `post_processor.cpp` local `file_size_or_zero` (raw `stat`) | `path::file_size_or_zero(p)` (local helper deleted; issue #9) |
| `getcwd` / `$PWD` sites | *(unchanged — non-timemory, site-specific; issue #3)* |
| `$TMPDIR`/`/tmp` | `path::temp_dir()` |
| `get_rocprofsys_root` / `get_internal_*` | `install::root()` / `install::lib*()` (sibling module) |
| `get_link_map` / `get_origin` | `dl::link_map()` / `dl::origin()` (sibling module) |

## 9. Before / after

```cpp
// install root (internal_libs.cpp:411, instrument.cpp:316)
auto base = filepath::dirname(filepath::dirname(filepath::realpath("/proc/self/exe", nullptr, false)));
auto base = path::parent_path(path::executable_path(), 2);

// exists+mkdir (impl.cpp:398 — latent bug) / (database.cpp)
if (!filepath::exists(_config_folder)) filepath::makedir(_config_folder);   // bug: exists false-for-dirs
path::make_dirs(_config_folder);                                            // idempotent, correct

// library-name classification (instrument.cpp:1165)
auto is_lib = base.find(".so")==base.length()-3 || base.find(".a")==base.length()-2 || base.find("lib")==0;
auto is_lib = path::has_any_extension(base,{".so",".a"}) || std::string_view{base}.starts_with("lib");

// output open (perfetto.cpp:253) — unchanged shape, dedups the shim
if (!path::open(ofs, _filename, std::ios::out | std::ios::binary)) { ... }
```

## 10. Non-goals / deferred

- No user-visible output path/format changes (Epic constraint).
- No `fs::path`-typed public API (string-in/string-out minimizes churn across ~150
  sites); an internal `fs::path` is an implementation detail. `fs::path` overloads
  can be added later if a caller benefits.
- `is_elf` (replacing the two `is_text_file` byte-scans) is a 4-byte magic read (no `fs`
  equivalent); deduplicated into one generic predicate with precise ELF semantics.

## 11. Decisions (resolved)

These were previously "open questions"; each is now settled (issue #8) so
implementation does not re-litigate them.

1. **Namespace — keep `rocprofsys::common::path`.** Zero churn on the ~40 existing
   `common::path::` sites (`path.hpp` already lives under `common/`). `path::` throughout
   this doc is shorthand — callers use `common::path::` directly or a per-file
   `namespace path = ::rocprofsys::common::path;` alias, as suits the site. (Rejected:
   renaming to `rocprofsys::path`, which is cleaner but churns ~40 sites for no functional
   gain.)
2. **Stream dispatch — use the `OutputStream` concept** (§5.6). Nicer diagnostics; it is
   the only concept in the design. `if constexpr(std::is_base_of_v<std::ostream, StreamT>)`
   remains a drop-in fallback if concepts are ever undesired.
3. **`find_in_dirs` placement — generic in `path.hpp`; rocprofsys-specific search policy
   (lib/lib64, ldconfig) in `install_layout.hpp`** (§5.7). Keeps `path.hpp` domain-free.

## 12. Rollout / PR sequencing

- Land `path.hpp` + tests first (additive, no behavior change), then migrate call sites
  per subsystem, then delete `tim::filepath` includes/aliases and the old `common::path`
  bodies. Keeps each PR < ~400 lines and reviewable.

### 12.1 The `LD_PRELOAD`ed `dl` consumer — validate separately (issue #11)

The preloaded `rocprof-sys-dl` library (analysis layer 2) is **categorically the
highest-risk consumer**: it is injected into arbitrary target processes, so a path-code
regression can crash or corrupt *any instrumented application*, not just the tool. §3
decided to use `fs` **uniformly** in `dl` and *argues* it is safe (static-libstdc++ builds,
signal/pre-init context, the `noexcept`+`catch` boundary) — but that reasoning must be
**validated**, not assumed. So `dl` gets its own step, not a generic "migrate per
subsystem" bucket:

1. **Migrate `dl` in its own isolated PR**, so a regression is independently bisectable
   and revertable.
2. **Real `LD_PRELOAD` smoke run** — instrument an actual target binary end-to-end,
   exercising the live `dl` path sites: the `indirect` ctor (`find_path`/`dirname`) and
   `readlink("/proc/self/exe")`. Unit tests are not sufficient here.
3. **Across the build configs §3 reasoned about** — default (dynamic `libstdc++.so.6`)
   **and** `ROCPROFSYS_BUILD_STATIC_LIBSTDCXX=ON` — since that is where the cross-`.so`
   `type_info` / `fs`-availability concerns live.
4. **Force the fallback paths** — point at a missing lib / non-existent path so the
   `catch(...)`→fallback and the "no `filesystem_error` crosses the `LD_PRELOAD` boundary"
   guarantee (§5.0) are actually exercised in the injected context.
5. **Validate before deleting the old `tim::filepath` bodies**, so old vs. new can be
   compared and rolled back if the smoke run regresses.

This turns §3's "verified by reasoning" into "verified by test" for the one path where
being wrong is worst.
