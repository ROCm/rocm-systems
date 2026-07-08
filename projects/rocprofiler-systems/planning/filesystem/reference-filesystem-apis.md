# Filesystem APIs Reference: Timemory vs Raw Syscalls vs C++17 `std::filesystem`

Companion to [`analysis-filesystem-usage.md`](analysis-filesystem-usage.md). For every
filesystem *capability* used in rocprofiler-systems, this document shows the three
ways to do it — the **Timemory** function currently used, the **raw syscall / libc**
primitive, and the **C++17 `std::filesystem`** (`fs`) function — with an explanation
of what each option can and cannot do.

Project is **C++20**, so everything in `<filesystem>` (a C++17 library) is available
and unchanged in C++20 (C++20 only adds `starts_with`/`ends_with` on strings, useful
for the G16 classification checks — not for the filesystem ops themselves).

Conventions used below:
- `fs` = `std::filesystem`. `ec` = a `std::error_code` passed to the non-throwing overload.
- Every `fs` mutating/querying function has **two overloads**: one that **throws**
  `fs::filesystem_error` on error, and one taking `std::error_code& ec` that is
  **`noexcept`** and reports errors via `ec`. Timemory functions never throw
  (they return `false`/input/`0`), so faithful replacements must use the **`ec` overload**.

---

## Part 0 — Master capability map

| # | Capability (from survey) | Timemory | Raw syscall / libc | `std::filesystem` |
|---|--------------------------|----------|--------------------|-------------------|
| G1 | Parent directory | `filepath::dirname` / `tim::dirname` | `::dirname(3)` (glibc) or manual `find_last_of('/')` | `path::parent_path()` |
| G2 | Filename / basename | `filepath::basename` | `::basename(3)` (glibc) | `path::filename()` |
| G3 | Stem / extension | — | manual `find_last_of('.')` | `path::stem()` / `path::extension()` |
| G4 | Canonicalize (resolve symlinks) | `filepath::realpath` | `::realpath(3)` | `fs::canonical` / `fs::weakly_canonical` |
| G5 | Read symlink target | — (dl uses `filepath::readlink`) | `::readlink(2)` | `fs::read_symlink` |
| G6 | File exists (reg/symlink) | `filepath::exists` | `stat(2)`/`lstat(2)`+`S_ISREG`; `access(2)` | `fs::is_regular_file` (\|\| `is_symlink`) |
| G7 | Directory exists | `filepath::direxists` | `stat(2)`+`S_ISDIR` | `fs::is_directory` |
| G8 | Anything exists | `common::path::exists` | `lstat(2)` (any type); `access(F_OK)` | `fs::exists` |
| G9 | Is symlink / path type | — (`common::path::is_link`) | `lstat(2)`+`S_ISLNK` | `fs::is_symlink` / `fs::symlink_status` |
| G10 | Text vs binary file | — (home-grown) | read + scan for `\0` | (no direct equiv — read bytes) |
| G11 | Create directory tree | `filepath::makedir` / `tim::makedir` | `mkdir(2)` (loop), `mkdtemp(3)` | `fs::create_directory` / `fs::create_directories` |
| G12 | Open output + auto-mkdir | `filepath::open`/`fopen` (shim) | `mkdir` loop + `ofstream`/`fopen` | `create_directories` + `ofstream` |
| G13 | Open input stream | `filepath::open`(ifstream) | `fopen`/`open(2)` | `std::ifstream{fs::path}` |
| G14 | Direct stream (no mkdir) | — | `ofstream`/`ifstream`/`fopen` | `std::ofstream{fs::path}` |
| G15 | Join path components | `TIMEMORY_JOIN('/',…)` | manual `+ "/" +` | `path::operator/` / `append` |
| G16 | Path-string classification | — | `find`/`substr` idioms | `path::extension()`+C++20 `ends_with` |
| G17 | Search file across dirs | — (`common::path::find_path`) | loop + `access`/`stat` | loop + `fs::exists` |
| G18 | Install-layout discovery | — (`common::path::get_*`) | `readlink("/proc/self/exe")`+`dirname` | `fs::read_symlink`+`parent_path` |
| G19 | Loaded-library paths | — | `dlopen`/`dlinfo` (`RTLD_DI_LINKMAP`/`ORIGIN`) | *(none — not a filesystem op)* |
| — | Current working dir | `filepath::get_cwd` | `getcwd(3)` | `fs::current_path()` |
| — | Delete file / tree | — | `unlink(2)`/`remove(3)`/`rmdir(2)` | `fs::remove` / `fs::remove_all` |
| — | Rename / move | — | `rename(2)` | `fs::rename` |
| — | File size | — | `stat(2)` `st_size` | `fs::file_size` |
| — | Temp dir root | — | `$TMPDIR` / `/tmp` | `fs::temp_directory_path()` |
| — | Enumerate directory | — | `opendir`/`readdir`/`closedir` | `fs::directory_iterator` |

`dlopen`/`dlinfo` (G19) are **not** filesystem operations — they query the dynamic
linker's in-memory state. There is no `fs` equivalent; they stay as-is.

---

## Part 1 — Raw POSIX syscalls / libc functions (reference)

Each entry: header · signature · what it does · error behavior · gotchas.

### `stat` / `lstat` / `fstat` — file metadata
`#include <sys/stat.h>`
```c
int stat (const char* path, struct stat* buf);   // follows symlinks
int lstat(const char* path, struct stat* buf);   // does NOT follow (stats the link itself)
int fstat(int fd,           struct stat* buf);   // by open fd
```
- Fills `struct stat`: `st_mode` (type+perms), `st_size`, `st_mtim`, `st_ino`, `st_dev`, …
- Type tested with macros on `st_mode`: `S_ISREG`, `S_ISDIR`, `S_ISLNK`, `S_ISCHR`, `S_ISBLK`, `S_ISFIFO`, `S_ISSOCK`.
- Returns `0` on success, `-1` + `errno` on failure (`ENOENT` = no such path, `EACCES`, `ENOTDIR`, `ELOOP`…).
- **Gotcha:** `stat` follows symlinks so `S_ISLNK` is *never* true from `stat` — you need `lstat` to detect a link. `stat` on a valid symlink → the target's type; on a *broken* symlink → fails (`ENOENT`).
- Used in the repo for: existence, is-file, is-dir, file-size (`post_processor.cpp`, `internal_libs.cpp`, `rocprof-sys-instrument.cpp`, `path.hpp`).

### `access` / `faccessat` — permission/existence probe
`#include <unistd.h>`
```c
int access(const char* path, int mode);   // mode: F_OK | R_OK | W_OK | X_OK
```
- `F_OK` tests existence only; `R_OK/W_OK/X_OK` test read/write/execute permission for the *real* UID/GID.
- Returns `0` if all requested checks pass, else `-1`+`errno`.
- **Gotcha:** checks against real (not effective) UID → can differ from what an actual `open` would allow; classic TOCTOU footgun. Used in `test_discovery.cpp` (`F_OK`).

### `realpath` — canonicalize an existing path
`#include <stdlib.h> #include <limits.h>`
```c
char* realpath(const char* path, char* resolved_path);  // resolved_path: buffer >= PATH_MAX, or NULL to malloc
```
- Resolves **all** symlinks, collapses `.`/`..`, makes absolute. Requires **every component to exist**.
- Returns pointer to result (must `free` if `resolved_path==NULL`), or `NULL`+`errno` on any failure (`ENOENT`, `ELOOP`…).
- **Gotcha:** all-or-nothing — one missing component → `NULL`. The repo wraps it to **fall back to the input string** on failure (both `common::path::realpath` and `tim::filepath::realpath`).

### `readlink` — read a symlink's target
`#include <unistd.h>`
```c
ssize_t readlink(const char* path, char* buf, size_t bufsiz);   // NOT null-terminated!
```
- Writes the link target (a *path string*, possibly relative) into `buf`, returns byte count.
- Does **not** append `\0`; you must terminate manually. Truncates silently if `bufsiz` too small (return == `bufsiz`).
- Returns `-1`+`errno` if not a symlink (`EINVAL`) or missing (`ENOENT`).
- Used to resolve `/proc/self/exe` and `/proc/<pid>/exe` (install-root discovery; `dl.cpp`, `path.hpp`).

### `mkdir` — create one directory
`#include <sys/stat.h>`
```c
int mkdir(const char* path, mode_t mode);   // mode filtered by process umask
```
- Creates a **single** level; parent must already exist (`ENOENT` otherwise). `EEXIST` if it already exists.
- "`mkdir -p`" requires a manual component loop (which is exactly what `filepath::makedir` does).
- Used directly in tests; via `filepath::makedir` in production.

### `mkdtemp` — create a unique temp directory
`#include <stdlib.h>`
```c
char* mkdtemp(char* template);   // template must end in "XXXXXX", modified in place
```
- Atomically creates a uniquely-named dir (mode `0700`). Returns the (mutated) name, or `NULL`+`errno`.
- Used in `test_path.cpp` fixtures.

### `opendir` / `readdir` / `closedir` — enumerate a directory
`#include <dirent.h>`
```c
DIR* opendir(const char* path);
struct dirent* readdir(DIR* d);   // returns NULL at end (or on error — check errno)
int  closedir(DIR* d);
```
- `dirent::d_name` is the entry name (not a full path); `.`/`..` are included and must be skipped.
- Not thread-safe on the same `DIR*`; `readdir` returns a pointer to a shared buffer.
- Used in `discovery.cpp`, `preset_registry.cpp`, `test_discovery.cpp`.

### `unlink` / `rmdir` / `remove` — delete
`#include <unistd.h>` (`unlink`,`rmdir`) · `#include <cstdio>` (`remove`)
```c
int unlink(const char* path);   // remove a file / symlink (not a dir)
int rmdir (const char* path);   // remove an EMPTY dir
int remove(const char* path);   // unlink() for files, rmdir() for dirs
```
- All return `0`/`-1`+`errno`; `ENOENT` if missing (repo tolerates this).
- Used in `discovery.cpp` (`std::remove`), `test_discovery.cpp` (`unlink`/`rmdir`).

### `rename` — move/rename
`#include <cstdio>` → `int rename(const char* old, const char* neu);`
- Atomic within a filesystem; `EXDEV` across filesystems (needs copy+delete). Not currently used but relevant.

### `getcwd` — current working directory
`#include <unistd.h>` → `char* getcwd(char* buf, size_t size);` (glibc: `buf==NULL` mallocs).
- Used by `rocprof-sys-instrument.cpp::get_cwd` and inside `filepath::makedir`.

### `symlink` — create a symlink
`#include <unistd.h>` → `int symlink(const char* target, const char* linkpath);` — tests only.

### `basename` / `dirname` (glibc) — path components
`#include <libgen.h>` (POSIX) **or** `#include <cstring>` (GNU basename)
```c
char* basename(char* path);   // last component
char* dirname (char* path);   // everything before it
```
- **Major gotcha (lifetime + mutation):** the GNU `::basename` (from `<string.h>`) may return a pointer *into the argument buffer*; the POSIX versions (`<libgen.h>`) may **modify** the argument. Neither owns memory. This is the source of the dangling-`string_view` hazards in `link_map.cpp` / `details.cpp`.
- `basename("/a/b/")` → `"b"`, `basename("/")` → `"/"`, `basename("a")` → `"a"` (GNU semantics differ from `fs::path::filename`).

### `open` / `fopen` — open a file (streams)
`#include <fcntl.h>` (`open`, returns fd) · `#include <cstdio>` (`fopen`, returns `FILE*`).
- Used directly for perfetto temp files (`::fopen(..., "rb")`); production streams mostly go through `std::ofstream`/`ifstream`.

### `dlopen` / `dlinfo` — dynamic-linker introspection (G19, **not** filesystem)
`#include <dlfcn.h> #include <link.h>`
```c
void* dlopen(const char* file, int flags);       // RTLD_LAZY|RTLD_NOLOAD etc.
int   dlinfo(void* handle, int request, void* p);// RTLD_DI_LINKMAP, RTLD_DI_ORIGIN
```
- Walks the in-memory `struct link_map` (`l_name`, `l_next`) or returns a library's `ORIGIN` dir.
- **No `fs` or syscall replacement** — this queries loader state, not the filesystem. Keep as-is.

---

## Part 2 — `std::filesystem` (`fs`) functions (reference)

`#include <filesystem>`. All are C++17. Remember the **throwing vs `ec` overload** rule.

### 2.1 `fs::path` — lexical path manipulation (no disk access)
Construct from `std::string`/`const char*`. Pure string surgery — none of these touch disk:

| Member | Result | Notes vs GNU basename/dirname |
|--------|--------|-------------------------------|
| `parent_path()` | everything before last `/` | `"/a/b"→"/a"`; `"a"→""` (GNU dirname→"."); `"/a/b/"→"/a/b"` |
| `filename()` | last component | `"/a/b"→"b"`; `"/a/b/"→""` (trailing slash!) |
| `stem()` | filename without final extension | `"a.tar.gz"→"a.tar"` |
| `extension()` | final extension incl. dot | `"a.tar.gz"→".gz"`; `"a"→""`; dotfile `".bashrc"→""` |
| `root_path()`/`root_directory()` | `/` portion | for absolute detection |
| `is_absolute()`/`is_relative()` | bool | replaces `find('/')==0` |
| `lexically_normal()` | collapse `.`/`..` textually (no disk) | `"a/./b/../c"→"a/c"` |
| `lexically_relative(base)` | relative form | — |
| `operator/` , `append` | join with correct separator | `path("a")/"b"→"a/b"` |
| `operator+=` , `concat` | concatenate without separator | — |
| `string()` | `std::string` (native encoding) | use this, **not** `u8string()` (returns `std::u8string` in C++20) |
| `c_str()` | `const char*` | for legacy APIs |
| `replace_extension(e)` | change extension | — |
| `begin()/end()` | iterate components | — |

**Gotchas:** trailing-slash and dotfile edge cases differ from the GNU `basename`/`dirname`
the repo currently relies on (documented per-site in the survey). A no-slash input gives
`parent_path()==""` (GNU `dirname`→ the input). Not hit by current concrete-file inputs, but
note it when swapping.

### 2.2 Existence & type queries

```cpp
bool fs::exists(const path&, error_code& ec) noexcept;          // any type (file/dir/…)
bool fs::is_regular_file(const path&, error_code&) noexcept;    // follows symlinks
bool fs::is_directory   (const path&, error_code&) noexcept;    // follows symlinks
bool fs::is_symlink     (const path&, error_code&) noexcept;    // does NOT follow (uses symlink_status)
bool fs::is_other/is_fifo/is_socket/is_block_file/is_character_file(...);
```
- `fs::status(p)` follows symlinks; `fs::symlink_status(p)` does not — both return `fs::file_status` (`.type()`, `.permissions()`).
- `fs::file_type` enum: `regular`, `directory`, `symlink`, `not_found`, `block`, `character`, `fifo`, `socket`, `unknown`.
- **Mapping the two repo `exists` semantics:**
  - `tim::filepath::exists` / `avail::file_exists` (reg||symlink, **false for dirs**) → `is_regular_file(p,ec) || is_symlink(p,ec)`. **Do NOT use `fs::exists`** (true for dirs).
  - `common::path::exists` (dir||reg||symlink) → `fs::exists(p,ec)` (equivalent; note `fs::exists` follows symlinks so a *broken* symlink → `false`, whereas the current `lstat`-based one → `true`; minor divergence for broken links).
  - `direxists`/`is_directory` → `fs::is_directory(p,ec)`.
  - `is_link` → `fs::is_symlink(p,ec)`.

### 2.3 Canonicalization & resolution
```cpp
path fs::canonical        (const path&, error_code&);   // resolve symlinks+.. ; REQUIRES existence (errors if missing)
path fs::weakly_canonical (const path&, error_code&);   // resolve existing prefix, normalize the rest ; OK if missing
path fs::absolute         (const path&, error_code&);   // make absolute (prepend current_path); does NOT resolve symlinks
path fs::relative         (const path&, const path& base, error_code&);
path fs::read_symlink     (const path&, error_code&);   // the symlink's target (one level; = ::readlink)
bool fs::equivalent       (const path&, const path&, error_code&);  // same file (dev+inode)
```
- **`realpath` mapping:** POSIX `::realpath` = "resolve all symlinks, require existence, else fail." The repo's fallback-to-input wrapper has **no exact `fs` twin**:
  - `fs::canonical` matches `::realpath`'s "require existence" but **errors** on missing paths (vs `::realpath` returning `NULL`).
  - `fs::weakly_canonical` doesn't error on missing paths but **normalizes** the tail (changes the returned string vs `::realpath`'s verbatim input — matters for map-key/`==` equality; see survey G4 note).
  - ⇒ To preserve exact current behavior, either keep the `::realpath`+fallback wrapper, or use `weakly_canonical` and accept normalized output. `fs::read_symlink` = `::readlink` (one level, no fallback).

### 2.4 Directory creation
```cpp
bool fs::create_directory (const path&, error_code&) noexcept;   // one level; parent must exist; false if already there
bool fs::create_directories(const path&, error_code&) noexcept;  // "mkdir -p": all missing parents; false if already there
```
- Default mode `0777` (minus umask), matching `mkdir`. Idempotent — safe to call unconditionally (no need for the `direxists` pre-guard, which removes a TOCTOU race).
- **Mapping:** `filepath::makedir`/`tim::makedir` → `fs::create_directories(p, ec)`.

### 2.5 Removal / move / copy
```cpp
bool     fs::remove    (const path&, error_code&) noexcept;   // file or EMPTY dir; false (no error) if missing
uintmax_t fs::remove_all(const path&, error_code&);           // recursive; returns count removed
void     fs::rename    (const path& from, const path& to, error_code&) noexcept;
void     fs::copy      (const path& from, const path& to, copy_options, error_code&);
bool     fs::copy_file (const path& from, const path& to, copy_options, error_code&);
```
- `fs::remove` returning `false` for a missing path (via `ec`) matches the repo's `errno==ENOENT` tolerance.
- **Mapping:** `std::remove`/`unlink`+`rmdir` → `fs::remove`; recursive test teardown → `fs::remove_all` (already used in `test_path.cpp`).

### 2.6 Queries & navigation
```cpp
uintmax_t     fs::file_size       (const path&, error_code&) noexcept;   // regular files
path          fs::current_path    (error_code&);                        // getcwd; setter overload too
path          fs::temp_directory_path(error_code&);                     // $TMPDIR/TMP/... or /tmp
file_time_type fs::last_write_time (const path&, error_code&) noexcept;
space_info    fs::space           (const path&, error_code&) noexcept;  // capacity/free/available
void          fs::permissions     (const path&, perms, perm_options, error_code&);
```
- **Mapping:** `stat.st_size` → `fs::file_size`; `getcwd` → `fs::current_path`; `$TMPDIR||/tmp` → `fs::temp_directory_path`.

### 2.7 Directory iteration
```cpp
fs::directory_iterator          (const path&, error_code&);   // one level
fs::recursive_directory_iterator(const path&, error_code&);   // descends
// each *it is a fs::directory_entry with .path(), .is_regular_file(), .is_directory(), ...
```
- Auto-skips `.`/`..`. `directory_entry` **caches** stat info (fewer syscalls than manual `readdir`+`stat`).
- **Mapping:** `opendir`/`readdir` loops (`discovery.cpp`, `preset_registry.cpp`) → `fs::directory_iterator`.

---

## Part 3 — Timemory `filepath` functions (what they do & their quirks)

Source: `external/timemory/source/timemory/utility/filepath.{hpp,cpp}`. All are
effectively `noexcept` (return `false`/input/`0` on error). These are what we remove.

| Timemory fn | Behavior | Quirk to preserve / beware |
|-------------|----------|-----------------------------|
| `realpath(p, resolved=nullptr, warn=true)` | `::realpath` into `PATH_MAX` buf, **returns input on failure** | 20/21 call sites pass `warn=false`; `_resolved` out-param path is dead code |
| `exists(p)` | `stat` + `S_ISREG||S_ISLNK` → **false for directories** | The recurring semantic trap; drives ≥3 latent bugs |
| `direxists(p)` | `stat` + `S_ISDIR` | — |
| `dirname(p)` | slice before last `/` | no-slash input → returns input unchanged (differs from `parent_path()`→"") |
| `basename(sv)` | glibc `::basename(sv.data())` → `const char*` | **borrows/aliases memory** → dangling hazards |
| `makedir(dir, umask=0777)` | recursive `mkdir -p`, EEXIST-tolerant | discards nothing important; return int usually ignored |
| `open(ofstream&, p, args…)` | split dir/base, `makedir(dir)`, **fallback `./base`** on mkdir fail, then `open` | auto-mkdir is load-bearing; variadic forwards ios flags |
| `open(ifstream&, p, args…)` | split, open (does **not** mkdir) | input variant |
| `fopen(p, mode)` | like ofstream `open` (auto-mkdir + fallback) then `::fopen` | single site (`config.cpp`) |
| `canonical(p)`/`osrepr(p)` | separator normalization (no `::realpath`) | on Linux ≈ identity; internal helper |
| `get_cwd()` | `getcwd` wrapper | not `tim::filepath` at the instrument call site (local fn) |

Unused under `source/` (do **not** need replacement): `replace`, `os`, `inverse`, `is_link`, `canonicalize`.

---

## Part 4 — Side-by-side per capability

### G1 Parent directory
```cpp
// Timemory / manual
std::string d = tim::filepath::dirname(p);          // slice at last '/'
// Raw
char buf[PATH_MAX]; strcpy(buf,p.c_str()); char* d = ::dirname(buf);  // mutates buf!
// fs (C++17)
std::string d = fs::path{p}.parent_path().string(); // no disk access, no mutation
```

### G2 Filename / basename
```cpp
const char* b = tim::filepath::basename(p);         // borrows memory (dangling risk)
char* b = ::basename(buf);                            // mutates/aliases
std::string b = fs::path{p}.filename().string();    // owning, safe
```

### G3 Extension / stem
```cpp
auto pos = name.find_last_of('.'); ext = name.substr(pos+1);   // manual
std::string ext  = fs::path{name}.extension().string();  // ".json" (incl. dot)
std::string stem = fs::path{name}.stem().string();       // name without ext
```

### G4 Canonicalize
```cpp
std::string r = tim::filepath::realpath(p, nullptr, false);       // ::realpath, input on fail
char buf[PATH_MAX]; const char* r = ::realpath(p.c_str(), buf) ? buf : p.c_str();
std::error_code ec; auto r = fs::weakly_canonical(fs::path{p}, ec); // normalizes tail; NEVER fs::canonical for maybe-missing paths
```

### G5 Read symlink
```cpp
std::string t = tim::filepath::readlink(p);                       // ::readlink + fallback
ssize_t n = ::readlink(p.c_str(), buf, sizeof buf); buf[n]='\0';  // manual NUL-terminate
std::error_code ec; std::string t = fs::read_symlink(fs::path{p}, ec).string();
```

### G6 / G7 / G8 Existence & type
```cpp
// file-exists (false for dirs):
bool f = tim::filepath::exists(p);                       // stat + S_ISREG||S_ISLNK
bool f = (stat(p,&st)==0 && (S_ISREG(st.st_mode)||S_ISLNK(st.st_mode)));
bool f = fs::is_regular_file(p,ec) || fs::is_symlink(p,ec);   // NOT fs::exists
// dir-exists:
bool d = tim::filepath::direxists(p);                    // stat + S_ISDIR
bool d = fs::is_directory(p,ec);
// anything-exists:
bool e = common::path::exists(p);                        // lstat any type
bool e = fs::exists(p,ec);
```

### G9 Is symlink
```cpp
bool l = common::path::is_link(p);                       // lstat + S_ISLNK
bool l = (lstat(p,&st)==0 && S_ISLNK(st.st_mode));
bool l = fs::is_symlink(p,ec);
```

### G11 / G12 Create dir (+ open with mkdir)
```cpp
tim::filepath::makedir(dir);                             // mkdir -p, EEXIST ok
// manual: loop over components calling mkdir(...) tolerating EEXIST
std::error_code ec; fs::create_directories(dir, ec);    // idempotent, one call
// open-with-mkdir shim → fs::create_directories(path.parent_path(), ec); ofs.open(path);
```

### G15 Join
```cpp
auto p = TIMEMORY_JOIN('/', a, b, c);                   // or fmt::format("{}/{}", a, b)
auto p = a + "/" + b;                                    // manual (double-slash risk)
fs::path p = fs::path{a} / b / c;                        // separator-correct
```

### Directory enumeration / delete / cwd / size / tempdir
```cpp
// enumerate
for(auto& e : fs::directory_iterator{dir, ec}) use(e.path());     // vs opendir/readdir
// delete (tolerate missing)
fs::remove(p, ec);            fs::remove_all(dir, ec);            // vs unlink/rmdir/std::remove
// cwd / size / tempdir
auto cwd  = fs::current_path(ec);                                 // vs getcwd
auto size = fs::file_size(p, ec);                                 // vs stat.st_size
auto tmp  = fs::temp_directory_path(ec);                          // vs $TMPDIR || /tmp
```

---

## Part 5 — Behavioral notes that matter when choosing

1. **Exceptions:** always pass `ec` to keep the timemory "never throws" contract.
   The codebase does use exceptions elsewhere, so a stray throw isn't catastrophic,
   but the *filepath call sites* expect `false`/input on error, not a throw.
2. **Symlink following:** `stat`/`fs::status`/`is_regular_file`/`is_directory` **follow**
   symlinks; `lstat`/`fs::symlink_status`/`is_symlink` **do not**. Match the original
   call's choice (most repo sites want "follows").
3. **Broken symlinks:** `fs::exists` → `false` (follows, target missing); the repo's
   `lstat`-based `common::path::exists` → `true`. `is_symlink` → `true` for both.
   (There is an explicit `Exists_BrokenSymlink` test expecting `true`.)
4. **`realpath` vs `weakly_canonical`:** only `weakly_canonical` tolerates missing paths,
   but it **normalizes** (`./a/../b`→`b`) whereas `::realpath` returns the **verbatim input**
   on failure. Since results are used as `std::map`/`std::set` keys and in `==`, prefer
   keeping the `::realpath`+fallback wrapper for byte-identical behavior.
5. **`PATH_MAX` buffers:** all raw wrappers hard-code `PATH_MAX` (fragile on long paths);
   `fs` handles arbitrary lengths internally — a real robustness win.
6. **`basename`/`dirname` memory:** the libc versions mutate/alias the argument; `fs::path`
   members return owning `path` objects — eliminates the dangling-`string_view` bugs.
7. **`create_directories` is idempotent:** drop the `exists`/`direxists` pre-guards
   (removes TOCTOU races at `impl.cpp:399`, `database.cpp`, `config.cpp:3013`).
8. **`fs::directory_entry` caches stat:** iterating with `fs` and calling `.is_regular_file()`
   is fewer syscalls than `readdir`+separate `stat`.
9. **No env-var / `~` expansion** in any layer (timemory, `common::path`, or `fs`) —
   callers expand env vars themselves; unchanged by the choice.
10. **C++20 `std::formatter<fs::path>` does NOT exist** — don't `std::format`/`fmt` a
    `fs::path` directly; call `.string()`. `std::string::contains` is C++23 (keep
    `find(x)!=npos`); `starts_with`/`ends_with` are C++20 (usable for G16).
11. **Linking:** since GCC 9.1 / Clang 9, `std::filesystem` lives in `libstdc++.so.6`
    (no separate `-lstdc++fs`); under the project's C++20 toolchain no extra link is needed,
    and the preloaded `dl` library already loads `libstdc++.so.6`.
