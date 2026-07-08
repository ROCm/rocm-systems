# Filesystem-Operations Usage Survey (rocprofiler-systems)

Research/analysis document that collects **every filesystem operation** in the
codebase, grouped by *meaning* (what the code is actually trying to do), with a
reasonable amount of surrounding context and a plain-English explanation for
each site. This is the raw material for designing a single, clean filesystem
module (removal of `tim::filepath` + consolidation of `common::path` + the
scattered raw syscalls).

Scope: `source/` (production + the filesystem-relevant tests). Excludes
argparse `parser.exists("flag-name")` (CLI-arg presence checks, not filesystem).

---

## How the operations are currently provided (3 overlapping layers)

1. **`tim::filepath::*`** (timemory, `external/timemory/.../utility/filepath.hpp`)
   — `realpath`, `exists`, `direxists`, `dirname`, `basename`, `makedir`,
   `open` (ofstream/ifstream), `fopen`. Plus the `tim::dirname` / `tim::makedir`
   façade. Accessed directly (`tim::filepath::`) or via 6 `namespace filepath =
   ::tim::filepath;` aliases (notably the project-wide one in
   `source/lib/core/common.hpp:88`).
2. **`rocprofsys::common::path::*`** (`source/lib/common/path.hpp`, header-only
   `INTERFACE` library, **syscall-based, timemory-free**) — `exists`, `dirname`,
   `realpath`, `readlink`, `is_link`, `is_text_file`, `path_type`, `find_path`,
   `get_default_lib_search_paths`, `get_origin`, `get_link_map`,
   `get_rocprofsys_root`, `get_internal_libpath/script_path/libdir`. Consumed by
   the **`LD_PRELOAD`ed `rocprof-sys-dl`** library.
3. **Raw syscalls / libc / STL** scattered across ~20 files — `stat`/`lstat`,
   `mkdir`/`mkdtemp`, `::readlink`, `::realpath`, `::basename`, `opendir`/
   `readdir`, `std::ofstream`/`std::ifstream`, `std::fopen`, `std::remove`,
   `access`, `symlink`, and `test_common::fs::` (`std::filesystem` in tests only).

**Semantic landmine (recurring below):** two different `exists` semantics coexist —
`common::path::exists` returns **true for directories** (`S_ISDIR||S_ISREG||S_ISLNK`),
while `tim::filepath::exists` and `rocprof-sys-avail/common.cpp:file_exists`
return **false for directories** (`S_ISREG||S_ISLNK`).

---

## Category legend & counts

| Code | Meaning | Approx. sites |
|------|---------|--------------:|
| G1  | Get parent directory (`dirname` / `parent_path`) | ~19 |
| G2  | Get filename / basename component | ~19 |
| G3  | Get stem / strip extension | 1 |
| G4  | Resolve real/absolute path (`realpath`) | ~30 |
| G5  | Read symlink target (`readlink`) | 3 |
| G6  | Check **file** exists (regular/symlink; false-for-dirs) | ~13 |
| G7  | Check **directory** exists / `is_directory` | ~8 |
| G8  | Generic exists (dir **or** file: `common::path::exists`) | ~12 |
| G9  | Check is symlink (`is_link`) / path-type classify | ~4 |
| G10 | Check is text/binary file | ~5 |
| G11 | Create directory tree (`makedir` / `mkdir`) | ~7 |
| G12 | Open OUTPUT stream / `fopen` **with auto-mkdir** (shim) | ~12 |
| G13 | Open INPUT stream (shim or direct, incl. procfs/sysfs) | ~11 |
| G14 | Direct `std::ofstream`/`ifstream`/`fopen` (no shim) | ~14 |
| G15 | Build / join path components | ~25 |
| G16 | Path-string classification (extension/prefix checks) | ~30 |
| G17 | Search for file across directories (`find_path`) | ~13 |
| G18 | Install-layout discovery (`get_rocprofsys_root` / `get_internal_*`) | ~13 |
| G19 | Dynamic-linker introspection yielding paths (`link_map`/`get_origin`) | ~13 |

---

# G1 — Get parent directory (`dirname`)

**`source/lib/core/trace_cache/discovery.cpp:144-146`**
```cpp
auto _filename      = config::get_perfetto_output_filename();
auto _output_folder = tim::filepath::dirname(_filename);
auto _script_path   = std::string{ "rocprof-sys-merge-output.sh" };
```
What: Parent directory of the perfetto output filename, used as the merge-script working folder.

**`source/lib/core/perfetto.cpp:280-286`**
```cpp
auto _output_folder = filepath::dirname(_filename);
auto _script_path   = std::string{ "rocprof-sys-merge-output.sh" };
auto _script_dir    = get_env(env_vars::SCRIPT_PATH, std::string{});
```
What: Same operation as above (perfetto path → parent dir) in the library-side perfetto output code.

**`source/lib/core/config.cpp:3065-3070`**
```cpp
const auto source =
    (get_use_rocpd() && !get_caching_perfetto())
        ? get_database_absolute_path("rocpd", std::to_string(process::get_id()))
        : get_perfetto_output_filename();
return tim::filepath::dirname(source);
```
What: Parent directory of the active backend output (rocpd db or perfetto file) so unified-memory output can be co-located.

**`source/lib/core/rocpd/data_storage/database.cpp:59-67`**
```cpp
create_directory_for_database_file(const std::string& db_file)
{
    auto _db_dirname = tim::filepath::dirname(db_file);
    if(!tim::filepath::direxists(_db_dirname))
        tim::filepath::makedir(_db_dirname);
}
```
What: Parent directory of the database file (then mkdir'd — see G7/G11).

**`source/lib/core/argparse.cpp:133-138`**
```cpp
auto _libdir = filepath::dirname(_data.env.dl_libpath);
if(filepath::exists(_libdir))
    update_env(_data, "LD_LIBRARY_PATH", _libdir, update_mode::APPEND);
```
What: Directory containing the dl library, to append to `LD_LIBRARY_PATH`. (Note: guarded by the buggy false-for-dirs `exists` — see G6.)

**`source/lib/rocprof-sys-dl/dl.cpp:217-220`**
```cpp
auto _search_paths = common::join(':', common::path::dirname(_omnilib),
                                  common::path::dirname(_dllib));
```
What: Derives search paths from the parent dirs of the two resolved rocprof-sys libraries.

**`source/lib/common/path.hpp:239-245`** *(implementation)*
```cpp
std::string
dirname(const std::string& _fname)
{
    if(_fname.find('/') != std::string::npos)
        return _fname.substr(0, _fname.find_last_of('/'));
    return std::string{};
}
```
What: The in-tree `dirname` — string-slice at last `/`; returns `""` for a slash-less input.

**`source/python/libpyrocprofsys.cpp:564-568`**
```cpp
auto _file =
    py::module::import("rocprofsys").attr("__file__").cast<std::string>();
if(_file.find('/') != std::string::npos)
    _file = _file.substr(0, _file.find_last_of('/'));
get_config().base_module_path = _file;
```
What: Directory of the Python module's `__file__` (manual dirname) used as the internal-path prefix.

**`source/bin/rocprof-sys-instrument/internal_libs.cpp:410-411`**
```cpp
auto _rocprofsys_base_path = filepath::dirname(
    filepath::dirname(filepath::realpath("/proc/self/exe", nullptr, false)));
```
What: `dirname(dirname(exe))` walk-up to the install root.

**`source/bin/rocprof-sys-instrument/rocprof-sys-instrument.cpp:318`**
```cpp
bin_search_paths.emplace_back(filepath::dirname(_omni_exe_path));
```
What: Parent dir of the instrument exe → a binary search path.

**`source/bin/rocprof-sys-instrument/rocprof-sys-instrument.cpp:320-321`**
```cpp
auto _omni_lib_path =
    JOIN('/', filepath::dirname(filepath::dirname(_omni_exe_path)), "lib");
```
What: `dirname(dirname(exe))` + `lib` → install lib dir.

**`source/bin/rocprof-sys-instrument/rocprof-sys-instrument.cpp:332-334`**
```cpp
auto _omni_internal_libexec_path =
    JOIN('/', filepath::dirname(filepath::dirname(_omni_exe_path)), "libexec",
         "rocprofiler-systems");
```
What: `dirname(dirname(exe))` + `libexec/...` → internal libexec (script) dir.

**`source/bin/rocprof-sys-instrument/rocprof-sys-instrument.cpp:354-355`**
```cpp
if(!find(filepath::dirname(itr), lib_search_paths))
    lib_search_paths.emplace_back(filepath::dirname(itr));
```
What: Parent dir of each matched loaded-library path → added to lib search paths.

**`source/bin/rocprof-sys-instrument/rocprof-sys-instrument.cpp:1170`**
```cpp
auto _rocprofsys_exe_path = tim::dirname(path::realpath("/proc/self/exe"));
```
What: Parent dir of the resolved exe path via the `tim::dirname` façade.

**`source/bin/rocprof-sys-instrument/rocprof-sys-instrument.cpp:1338-1342`**
```cpp
for(const auto& itr : _dyn_api_rt_paths)
{
    lib_search_paths.emplace_back(itr);
    lib_search_paths.emplace_back(filepath::dirname(itr));
}
```
What: Parent dir of each dyninstAPI_RT candidate → lib search path.

**`source/bin/rocprof-sys-instrument/rocprof-sys-instrument.cpp:2504-2507`**
```cpp
if(outf.find('/') != string_t::npos)
{
    auto outdir = outf.substr(0, outf.find_last_of('/'));
    tim::makedir(outdir);
}
```
What: Manual dirname (string-truncate at last `/`) of the output file before mkdir.

**`source/bin/rocprof-sys-instrument/rocprof-sys-instrument.cpp:2799`**
```cpp
if(!is_directory(itr) || is_file(itr)) itr = filepath::dirname(itr);
```
What: Reduce a non-directory search-path entry to its parent dir.

**`source/bin/rocprof-sys-instrument/rocprof-sys-instrument.cpp:2941-2947`**
```cpp
rocprofsys::set_env<string_t>("DYNINST_REWRITER_PATHS",
                              join::join(join::array_config{ ":", "", "" },
                                         dirname(_dyn_api_rt_abs),
                                         lib_search_paths),
                              1);
```
What: Directory containing the dyninst RT library (via `tim::dirname` façade) exported as `DYNINST_REWRITER_PATHS`.

---

# G2 — Get filename / basename component

**`source/lib/binary/link_map.cpp:120-124`**
```cpp
auto _name = (!_lib) ? config::get_exe_realpath() : std::string{ _lib };
for(const auto& itr : _fini_chain)
    LOG_DEBUG("[linkmap][{}]: {}", filepath::basename(_name), itr.real());
```
What: Basename for a debug log label.

**`source/lib/binary/link_map.cpp:149-153`**
```cpp
std::string_view
link_file::base() const
{
    return std::string_view{ filepath::basename(name) };
}
```
What: `link_file::base()` returns the basename — **`string_view` aliasing the `basename` result (lifetime hazard).**

**`source/lib/binary/analysis.cpp:206-213`**
```cpp
auto _base_v = std::string_view{ filepath::basename(_v) };
auto _real_v = filepath::realpath(_v);
for(const auto& mitr : _maps)
    if(std::string_view{ filepath::basename(mitr.pathname) } == _base_v ||
       _real_v == _v)
```
What: Basename of the library path, compared against each map entry's basename.

**`source/lib/binary/analysis.cpp:210-217`**
```cpp
if(std::string_view{ filepath::basename(mitr.pathname) } == _base_v ||
   _real_v == _v)
{
    _exclude_range_v.emplace(
        address_range{ mitr.load_address, mitr.last_address });
}
```
What: Basename of each memory-map entry's pathname (matched against target lib basename).

**`source/bin/common/tool_runner.cpp:83-88`** *(implementation)*
```cpp
std::string_view
basename_of(std::string_view path)
{
    const auto slash = path.rfind('/');
    return (slash == std::string_view::npos) ? path : path.substr(slash + 1);
}
```
What: A local, string-view basename helper (independent reimplementation).

**`source/bin/common/tool_runner.cpp:363-372`**
```cpp
if(!injected &&
   basename_of(arg).find(data.out.launcher) != std::string_view::npos)
```
What: basename of each command arg, matched against the launcher name.

**`source/lib/rocprof-sys/library/causal/experiment.cpp:356-358`**
```cpp
_ss << "[" << filepath::basename(selection.symbol.file) << ":"
    << selection.symbol.line << "]";
```
What: Filename-only rendering of a symbol's source file.

**`source/lib/rocprof-sys-dl/dl.cpp:208-214`**
```cpp
fprintf(stderr, "[rocprof-sys][dl][pid=%i] %s resolved to '%s'\n", getpid(),
        ::basename(_omnilib.c_str()), m_omnilib.c_str());
```
What: libc `::basename` of each library for a log line.

**`source/lib/rocprof-sys-dl/dl.cpp:1563`**
```cpp
rocprofsys_pop_trace(basename(argv[0]));
```
What: libc `basename` of `argv[0]` to name the popped trace region.

**`source/bin/rocprof-sys-causal/impl.cpp:303`**
```cpp
auto parser = parser_t{ basename(argv[0]), _desc };
```
What: libc `basename` of `argv[0]` for the argparse program name.

**`source/lib/core/config.cpp:1371-1375`**
```cpp
auto _exe          = (_cmd.empty()) ? "exe" : _cmd.front();
get_exe_realpath() = filepath::realpath(_exe, nullptr, false);
auto _pos          = _exe.find_last_of('/');
if(_pos < _exe.length() - 1) _exe = _exe.substr(_pos + 1);
get_exe_name() = _exe;
```
What: Manual basename of the exe (string-slice at last `/`).

**`source/bin/rocprof-sys-instrument/details.cpp:883-885`**
```cpp
auto _module_name = std::string{ get_name(mod) };
auto _module_base = std::string{ tim::filepath::basename(_module_name) };
auto _module_real = tim::filepath::realpath(_module_name, nullptr, false);
```
What: Basename of a module path for internal-libs matching.

**`source/bin/rocprof-sys-instrument/details.cpp:898-901`**
```cpp
for(const auto& [lib_path, sub_map] : _internal_libs)
{
    auto _lib_base = std::string{ tim::filepath::basename(lib_path) };
    if(_module_base == _lib_base || _module_real == lib_path ||
```
What: Basename of each internal-lib path, compared to the module basename.

**`source/bin/rocprof-sys-instrument/details.cpp:1089-1094`**
```cpp
for(auto* itr : symtab_data.modules)
{
    const auto* _base_name = tim::filepath::basename(itr->fullName());
    auto        _real_name = tim::filepath::realpath(itr->fullName(), nullptr, false);
    if(!_base_name) continue;
```
What: Basename of each Symtab module — **binds `const char*` to the `fullName()` temporary (dangling-read hazard).**

**`source/bin/rocprof-sys-instrument/module_function.cpp:417-419`**
```cpp
auto _basename = [](std::string_view _v) {
    return std::string{ tim::filepath::basename(_v) };
};
```
What: Local basename helper wrapping `tim::filepath::basename`.

**`source/bin/rocprof-sys-instrument/module_function.cpp:432-433 / 458-464`**
```cpp
auto _module_base = _basename(module_name);
...
if(_module_base == _basename(litr.first) || ...)
```
What: Basename of the module and of each internal lib, for exclusion matching.

**`source/bin/rocprof-sys-instrument/rocprof-sys-instrument.cpp:1175-1181`**
```cpp
std::string _cmdv_base = ::basename(_cmdv[0]);
auto        _has_lib_suffix = _cmdv_base.length() > 3 && ...
```
What: libc `::basename` of the target command, then classified as lib/exe (G16).

**`source/bin/rocprof-sys-instrument/rocprof-sys-instrument.cpp:1218-1220`**
```cpp
auto _is_local = (path::realpath(cmdv0) ==
                  TIMEMORY_JOIN('/', get_cwd(), ::basename(cmdv0.c_str())));
auto _cmd      = std::string{ ::basename(cmdv0.c_str()) };
```
What: libc `::basename` of the target used both to test cwd-locality and to form the default output name.

**`source/bin/rocprof-sys-instrument/rocprof-sys-instrument.cpp:2841-2850`**
```cpp
auto _base_name = std::string_view{ filepath::basename(_name) };
if(_base_name.find("lib") == 0 || _base_name.find(".so") != std::string::npos ||
   _base_name.find(".a") != std::string::npos)
    std::reverse(_combine_paths.begin(), _combine_paths.end());
```
What: Basename for the library-vs-binary search-order heuristic (classification in G16).

---

# G3 — Get stem / strip extension

**`source/bin/rocprof-sys-instrument/info.hpp:152-154`**
```cpp
auto        _pos = _iname.find_last_of('.');
std::string _ext = {};
if(_pos != std::string::npos) _ext = _iname.substr(_pos + 1);
```
What: Extracts the file extension (suffix after last `.`) to select the XML vs JSON deserializer.

*(See also the several `config.cpp` "strip known extension" sites classified under G16, and the `find_last_of('.')` splits.)*

---

# G4 — Resolve real/absolute path (`realpath`)

**`source/lib/common/path.hpp:283-307`** *(implementation)*
```cpp
std::string
realpath(const std::string& _relpath, std::string* _resolved)
{
    ...
    char        _buffer[MaxLen] = { '\0' };
    const char* _result         = _buffer;
    if(::realpath(_relpath.c_str(), _buffer) == nullptr)
        _result = _relpath.data();     // fallback-to-input on failure
    ...
}
```
What: The in-tree `realpath` — POSIX `::realpath` into a `PATH_MAX` buffer, **returning the original string on failure** (identical semantics to `tim::filepath::realpath`).

**`source/lib/binary/link_map.cpp:50-52`, `155-159`**
```cpp
return filepath::realpath(_link_map->l_name, nullptr, false);   // :52
...
std::string link_file::real() const { return filepath::realpath(name, nullptr, false); } // :158
```
What: Canonicalize a linker-reported / stored library path.

**`source/lib/binary/dwarf_entry.cpp:102-104`**
```cpp
const auto* _file = dwarf_linesrc(_line, nullptr, nullptr);
if(!_file) _file = dwarf_diename(_die);
itr.file = filepath::realpath(_file, nullptr, false);
```
What: Canonicalize a DWARF source filename.

**`source/lib/binary/symbol.cpp:59-64`, `239-246`**
```cpp
_data.emplace_back(inlined_symbol{ _line, filepath::realpath(_file, nullptr, false), _func }); // :63
...
file = filepath::realpath(file, nullptr, false);  // :245
```
What: Canonicalize BFD-reported inliner / symbol source file paths.

**`source/lib/binary/analysis.cpp:137-141`, `147-156`, `208-213`**
```cpp
auto _path = filepath::realpath(_v.pathname, nullptr, false);       // :139  (procfs map path)
auto _filename = filepath::realpath(itr, nullptr, false);           // :149  (input file, keyed into a std::set)
auto _real_v = filepath::realpath(_v);                              // :209  (default/warning form — the ONLY such site)
```
What: Canonicalize procfs map paths and input files; the `:209` site is the only one using the warning-enabled default overload and is compared `_real_v == _v` (relies on fallback-to-input).

**`source/lib/core/config.cpp:1368-1375`, `2129-2135`**
```cpp
get_exe_realpath() = filepath::realpath(_exe, nullptr, false);      // :1372
...
return filepath::realpath(_cmd_line.front(), nullptr, false);       // :2132 (lazy static exe realpath)
```
What: Resolve the executable's real path (cached), from the command line.

**`source/lib/rocprof-sys/library/causal/data.cpp:810-815`**
```cpp
auto _location = (_dl_info.location)
    ? filepath::realpath(std::string{ _dl_info.location.name }, nullptr, false)
    : std::string{};
```
What: Canonicalize a dl_info location name (debug path).

**`source/lib/core/argparse.cpp:111-120`**
```cpp
_data.env.dl_libpath   = path::realpath(path::get_internal_libpath("librocprof-sys-dl.so").c_str());
_data.env.omni_libpath = path::realpath(path::get_internal_libpath("librocprof-sys.so").c_str());
auto _libexecpath = path::realpath(path::get_internal_script_path());
auto _rootpath    = path::realpath(path::get_rocprofsys_root());
```
What: Canonicalize the discovered install-layout paths (lib/script/root) before exporting them as env vars.

**`source/bin/rocprof-sys-causal/impl.cpp:205-213`**
```cpp
update_env(_env, "LD_PRELOAD",
           join(":", LIBPTHREAD_SO,
                path::realpath(path::get_internal_libpath("librocprof-sys-dl.so"))),
           true);
```
What: Canonicalize the internal dl-lib path for `LD_PRELOAD`.

**`source/bin/common/tool_runner.cpp:329-330`**
```cpp
auto libexec_path = path::realpath(path::get_internal_script_path());
if(!libexec_path.empty()) data.env.set(env_vars::SCRIPT_PATH, libexec_path);
```
What: Canonicalize the internal libexec/script path.

**`source/bin/common/preset_registry.cpp:170-197`**
```cpp
auto resolved  = common::path::realpath(name_or_path);          // is_path branch
...
auto resolved  = common::path::realpath(filepath);             // name branch (dir + name + ".json")
auto canon_dir = common::path::realpath(m_directory);
if(resolved.empty() || canon_dir.empty() ||
   resolved.compare(0, canon_dir.size(), canon_dir) != 0)      // containment check
```
What: Canonicalize preset path + preset dir and enforce directory containment (path-traversal guard).

**`source/bin/rocprof-sys-instrument/details.cpp:735`, `761-768`, `883-885`, `1089-1094`**
```cpp
return tim::filepath::realpath(_link_map->l_name, nullptr, false);  // :735 (loaded lib path)
return tim::filepath::realpath(_buffer, nullptr, false);            // :768 (RTLD_DI_ORIGIN dir)
auto _module_real = tim::filepath::realpath(_module_name, nullptr, false); // :885
auto _real_name   = tim::filepath::realpath(itr->fullName(), nullptr, false); // :1092
```
What: Canonicalize loaded-lib paths, linker origin dir, and Symtab module paths for internal-lib matching.

**`source/bin/rocprof-sys-instrument/internal_libs.cpp:46`, `126`, `157`, `410-411`, `422`, `433`**
```cpp
return filepath::realpath("/proc/self/exe", nullptr, false);       // :46 get_exe_realpath
return filepath::realpath(_link_map->l_name, nullptr, false);      // :126
_chain.emplace(filepath::realpath(_next->l_name, nullptr, false)); // :157 (into a std::set)
...(dirname(dirname(realpath("/proc/self/exe"))))                  // :411
_libs.emplace_back(filepath::realpath(_libpath, nullptr, false));  // :422
auto _fpath = filepath::realpath(itr, nullptr, false);             // :433 (map key)
```
What: Canonicalize exe, loaded-lib chain entries (set members), and discovered lib paths (map keys).

**`source/bin/rocprof-sys-instrument/module_function.cpp:420-422`**
```cpp
auto _realpath = [](const std::string& _v) {
    return tim::filepath::realpath(_v, nullptr, false);
};
```
What: Local realpath helper for module-path canonicalization.

**`source/bin/rocprof-sys-instrument/rocprof-sys-instrument.cpp:314-318`, `444-450`, `656-657`, `1170`, `1218-1220`, `1663-1673`, `2892`, `2900`**
```cpp
auto _omni_exe_path = path::realpath(get_absolute_exe_filepath(argv[0]));   // :314
auto resolved_mutname = path::realpath(get_absolute_filepath(mutname));     // :446
keys.at(0) = path::realpath(get_absolute_filepath(keys.at(0)));             // :656
_rocprofsys_exe_path = tim::dirname(path::realpath("/proc/self/exe"));      // :1170
_is_local = (path::realpath(cmdv0) == JOIN('/', get_cwd(), basename(cmdv0)))// :1218
_libname = path::realpath(get_absolute_lib_filepath(_libname));            // :1670 (before loadLibrary)
_name = path::realpath(_name); ... S_ISREG(...)                            // :2892 is_file
_name = path::realpath(_name); ... S_ISDIR(...)                            // :2900 is_directory
```
What: Canonicalize the instrument exe, the mutatee, `--command` first token, `/proc/self/exe`, cmd for cwd-locality, each instrumentation lib before loading, and inside the local `is_file`/`is_directory` helpers.

---

# G5 — Read symlink target (`readlink`)

**`source/lib/common/path.hpp:255-281`** *(implementation)*
```cpp
std::string
readlink(const std::string& _path)
{
    if(!is_link(_path)) return _path;
    char _buffer[PATH_MAX];
    ... = ::readlink(_path.c_str(), _buffer, _buffer_len);
    if(_buffer_len < 0 || _buffer_len == (MaxLen)) { auto* rp = ::realpath(...); ... }
    ...
}
```
What: The in-tree `readlink` — libc `::readlink`, falling back to `::realpath`, returning input if not a link.

**`source/lib/rocprof-sys-dl/dl.cpp:1244-1252`**
```cpp
if(_exe.empty())
    _exe = tim::filepath::readlink(join('/', "/proc", getpid(), "exe"));
...
rocprofsys_push_trace(basename(_exe.c_str()));
```
What: Resolve `/proc/<pid>/exe` to the real executable path (preloaded-library site — the only `tim::filepath::readlink` use).

**`source/lib/common/tests/test_path.cpp:138-155`** *(tests)*
```cpp
TEST_F(PathTest, Readlink_SymbolicLink) {
    std::string link_path = create_symlink(target, "readlink_link");
    EXPECT_EQ(readlink(link_path), target);
}
```
What: Unit tests pinning `readlink` behavior on links, non-links, missing paths.

---

# G6 — Check **file** exists (regular/symlink; false-for-dirs)

**`source/bin/rocprof-sys-avail/common.cpp:410-417`** *(implementation)*
```cpp
bool
file_exists(const std::string& _fname)
{
    struct stat _buffer;
    if(stat(_fname.c_str(), &_buffer) == 0)
        return (S_ISREG(_buffer.st_mode) != 0 || S_ISLNK(_buffer.st_mode) != 0);
    return false;
}
```
What: A local `file_exists` — regular-file/symlink only (false for dirs); mirrors `tim::filepath::exists`.

**`source/lib/binary/analysis.cpp:138-140`, `149-155`**
```cpp
return (filepath::exists(_path) && _satisfies_binary_filter(_path));   // procfs map path
...
if(filepath::exists(_filename) && _satisfies_binary_filter(_filename) && ...)  // input file
```
What: Existence gate before including a mapping / parsing a file.

**`source/lib/core/argparse.cpp:135-137`**
```cpp
auto _libdir = filepath::dirname(_data.env.dl_libpath);
if(filepath::exists(_libdir))     // BUG: _libdir is a directory → false-for-dirs makes guard ~always fail
    update_env(_data, "LD_LIBRARY_PATH", _libdir, update_mode::APPEND);
```
What: Meant to check a *directory* but uses the file-exists semantics → latent bug (LD_LIBRARY_PATH rarely appended).

**`source/lib/rocprof-sys/library/kokkosp.cpp:290-298`** and **`source/lib/rocprof-sys/library.cpp:1209-1217`**
```cpp
if(!_path.empty() && _path.at(0) != '[' && rocprofsys::filepath::exists(_path))
    _libs.emplace(_path);
```
What: Keep procfs map pathnames that exist as regular files (skip `[stack]` etc.).

**`source/lib/core/dynamic_library.cpp:30-32`, `50-55`, `78-79`**
```cpp
if(_path.find(_name) != std::string::npos && filepath::exists(_path)) return _path;  // :31
auto _v = fmt::format("{}/{}", itr, _name); if(filepath::exists(_v)) return _v;       // :50
if(_env_val.find('/') == 0 && filepath::exists(_env_val)) filename = _env_val;        // :78
```
What: Existence checks while resolving a dynamic library across maps / candidate dirs / env override.

**`source/lib/core/config.cpp:1400-1401`, `3298-3303`, `3423-3427`, `1315` (procfs)**
```cpp
if(expanded_filename.ends_with(".json") && filepath::exists(expanded_filename) && ...) // :1400
if(!filepath::exists(filename)) { auto _ofs = std::ofstream{}; filepath::open(_ofs, filename); } // :3298
if(filepath::exists(filename)) { ... ::remove(filename.c_str()); }                     // :3423
```
What: Existence gates for config-file validation and temp-file create/remove.

**`source/lib/core/trace_cache/discovery.cpp:152-156`**, **`source/lib/core/perfetto.cpp:290-292`**
```cpp
if(!tim::filepath::exists(_script_path)) { LOG_WARNING("Merge script not found: {}", _script_path); return; }
```
What: Check the merge-output shell script exists before invoking it.

**`source/bin/rocprof-sys-avail/generate_config.cpp:246-249`**
```cpp
if(file_exists(_fname)) { ... }   // before overwriting an output file
```
What: File-exists guard before overwriting a generated config file.

**`source/bin/rocprof-sys-instrument/internal_libs.cpp:170-179`** *(raw stat)*
```cpp
auto _path_exists = [](const std::string& _filename) {
    struct stat dummy;
    return (_filename.empty()) ? false : (stat(_filename.c_str(), &dummy) == 0);
};
```
What: Raw `::stat` existence test (any type) for candidate library search-path dirs.

**`source/bin/rocprof-sys-instrument/rocprof-sys-instrument.cpp:2792-2807`, `2819-2820`, `2889-2895`** and **`rocprof-sys-instrument.hpp:137-147`**
```cpp
_exists = exists(pitr) && is_file(pitr);            // instrument.cpp:2801 (exists + regular-file)
...
bool is_file(std::string _name) { ...; return (stat(...) == 0 && S_ISREG(...)); } // :2889
...
if(!tim::filepath::exists(std::string{ _cmd })) { verbprintf(0, "Warning! ... not found"); } // hpp:139
```
What: File/regular-file existence during search-path resolution; the `.hpp:139` site is inside `if(_cmd.empty())` — a latent guard bug (`exists` unreachable for non-empty cmd).

**`source/lib/core/trace_cache/post_processor.cpp:33-40`** and **`tests/test_discovery.cpp:34-38`**
```cpp
if(::stat(path.c_str(), &st) != 0) return 0; return st.st_size;   // post_processor (exists+size)
return ::access(path.c_str(), F_OK) == 0;                          // test helper
```
What: Raw stat-for-size / `access(F_OK)` existence checks.

---

# G7 — Check **directory** exists / `is_directory`

**`source/lib/core/config.cpp:3013-3018`**
```cpp
auto ensure_dir = [](std::string path) {
    if(!path.empty() && !tim::filepath::direxists(path))
        tim::filepath::makedir(path);
    return path;
};
```
What: Directory-exists check before creating the unified-memory output dir.

**`source/lib/core/rocpd/data_storage/database.cpp:62-66`**
```cpp
if(!tim::filepath::direxists(_db_dirname))
    tim::filepath::makedir(_db_dirname);
```
What: Directory-exists check before creating the db parent dir (collapsible to a single `create_directories`).

**`source/lib/common/environment.hpp:563-569`**
```cpp
std::string torch_libdir = result + "/lib";
if(!::tim::filepath::direxists(torch_libdir))
{ LOG_WARNING("torch lib directory does not exist: {}", torch_libdir); return {}; }
```
What: Verify the torch `lib` directory exists (preloaded/common-layer site — **the 3rd `direxists`, missed by earlier surveys**).

**`source/bin/rocprof-sys-instrument/rocprof-sys-instrument.cpp:2799`, `2897-2903`**
```cpp
if(!is_directory(itr) || is_file(itr)) itr = filepath::dirname(itr);   // :2799
...
bool is_directory(std::string _name) { ...; return (stat(...) == 0 && S_ISDIR(...)); } // :2897
```
What: `is_directory` (raw stat + `S_ISDIR`) used to normalize search-path entries.

**`source/lib/core/trace_cache/discovery.cpp:33-50`** and **`tests/test_discovery.cpp:40-54`**, **`bin/common/preset_registry.cpp:230-246`**
```cpp
std::unique_ptr<DIR,...> dir(opendir(path.c_str()), dir_deleter);
if(!dir) throw std::runtime_error(...);
while((entry = readdir(dir.get())) != nullptr) { ... }
```
What: `opendir`/`readdir` directory *enumeration* (implies "is a directory"): scan the trace-cache tmp dir and the preset dir.

---

# G8 — Generic exists (dir **or** file: `common::path::exists`)

**`source/lib/common/path.hpp:165-173`** *(implementation)*
```cpp
bool
exists(const std::string& _fname)
{
    struct stat _buffer;
    if(lstat(_fname.c_str(), &_buffer) == 0)
        return (S_ISDIR(_buffer.st_mode) != 0 || S_ISREG(_buffer.st_mode) != 0 ||
                S_ISLNK(_buffer.st_mode) != 0);
    return false;
}
```
What: The in-tree `exists` — **true for dir, regular file, OR symlink** (differs from `tim::filepath::exists`). This is the "false friend."

**`source/bin/rocprof-sys-causal/impl.cpp:397-400`**
```cpp
if(!_dir.empty()) _config_folder = std::move(_dir);
if(!filepath::exists(_config_folder)) filepath::makedir(_config_folder);
```
What: exists-then-mkdir on a *directory* — via the false-for-dirs `filepath::exists`, so `makedir` always runs (latent bug / redundant guard).

**`source/bin/common/preset_registry.cpp:24-49`**
```cpp
if(common::path::exists(dir)) return dir;                     // env PRESET_DIR
...
auto candidate = common::join('/', root, "share", "rocprofiler-systems", "presets");
if(common::path::exists(candidate)) return candidate;
```
What: Existence checks (dir-or-file) while locating the preset directory.

**`source/bin/rocprof-sys-instrument/internal_libs.cpp:420`, `427-428`**
```cpp
if(filepath::exists(_libpath)) _libs.emplace_back(filepath::realpath(_libpath,...)); // :420
...
filter_sort_unique(_libs, [](const auto& itr){ return itr.empty() || !filepath::exists(itr); }); // :427
```
What: Existence gate for discovered rocprof-sys libs; filter out non-existent entries.

**`source/bin/rocprof-sys-instrument/internal_libs.cpp:532`, `549`**
```cpp
auto _path = join('/', itr, _lib_v);
if(filepath::exists(_path)) return ...;    // find_library / find_libraries
```
What: Existence checks while searching lib directories.

**`source/bin/rocprof-sys-instrument/rocprof-sys-instrument.cpp:293-296`, `314-318`, `2819-2820`, `2874`, `2883-2887`, `2936-2939`**
```cpp
if(!_omni_root.empty() && exists(_omni_root))        // :295  (local exists() → filepath::exists(absolute(name)))
if(!exists(_omni_exe_path)) ...                       // :315
bool exists(const std::string& name) { return filepath::exists(absolute(name)); } // :2883 (definition)
if(_orig_name == lib_name && !exists(lib_name) && ...) // :2874
if(!exists(_dyn_api_rt_abs)) ...                       // :2936
```
What: The file's local `exists()` helper (canonicalize → `filepath::exists`) used pervasively during path/lib resolution.

**`source/lib/core/trace_cache/tests/test_discovery.cpp:29-32`** and **`source/lib/common/tests/test_path.cpp:88-119`**
```cpp
if(const char* env = std::getenv("TMPDIR"); ...) return env; return "/tmp";   // temp root
TEST_F(PathTest, Exists_ExistingFile) { EXPECT_TRUE(exists(create_file(...))); }
```
What: Tests pinning generic-exists behavior (files, dirs, symlinks, broken links, empty).

---

# G9 — Is symlink / path-type classification

**`source/lib/common/path.hpp:151-163`** *(implementation)*
```cpp
path_type::path_type(const std::string& _fname)
{
    struct stat _buffer;
    if(lstat(_fname.c_str(), &_buffer) == 0)
    {
        if(S_ISDIR(...)) m_type = directory;
        else if(S_ISREG(...)) m_type = regular;
        else if(S_ISLNK(...)) m_type = link;
    }
}
```
What: `path_type` classifier (dir / regular / link / unknown) via `lstat`.

**`source/lib/common/path.hpp:247-253`** *(implementation)*
```cpp
bool
is_link(const std::string& _path)
{
    struct stat _buffer;
    if(lstat(_path.c_str(), &_buffer) == 0) return (S_ISLNK(_buffer.st_mode) != 0);
    return false;
}
```
What: The in-tree `is_link` — `lstat` + `S_ISLNK`.

**`source/lib/common/tests/test_path.cpp:121-136`, `227-253`** and **`48-53`**
```cpp
TEST_F(PathTest, IsLink_SymbolicLink) { EXPECT_TRUE(is_link(create_symlink(...))); }
TEST_F(PathTest, PathType_SymbolicLink) { path_type pt(link_path); EXPECT_TRUE(pt.exists()); }
std::string create_symlink(...) { symlink(target.c_str(), link_path.c_str()); ... }  // libc symlink()
```
What: Tests for `is_link` / `path_type`; the fixture uses libc `symlink()` to create links.

---

# G10 — Check is text/binary file

**`source/bin/rocprof-sys-instrument/details.cpp:588-617`** and **`source/lib/common/path.hpp:309-338`** *(two near-identical implementations)*
```cpp
bool is_text_file(const std::string& filename) {
    std::ifstream _file{ filename, std::ios::in | std::ios::binary };
    if(!_file.is_open()) { ...; return false; }
    while(_file.read(buffer, sizeof(buffer)))
        for(char itr : buffer) if(itr == '\0') return false;   // NUL byte ⇒ binary
    ...
    return true;
}
```
What: Home-grown text-vs-binary classifier (scans for NUL bytes). **Duplicated** in `details.cpp` and `common/path.hpp`.

**`source/bin/rocprof-sys-instrument/rocprof-sys-instrument.hpp:167-176`, `237-243`**
```cpp
if(is_text_file(_name)) errprintf(-127, "'%s' is a text file...");    // rewrite branch
...
if(is_text_file(_cmdv[0])) errprintf(-1, "'%s' is a text file...");   // process-create branch
```
What: Reject text-file inputs before opening a target binary for instrumentation.

**`source/lib/common/tests/test_path.cpp:204-225`**
```cpp
TEST_F(PathTest, IsTextFile_BinaryFile) { ...; EXPECT_FALSE(is_text_file(file_path)); }
```
What: Tests pinning `is_text_file` (text/binary/empty).

---

# G11 — Create directory tree (`makedir` / `mkdir`)

**`source/lib/core/config.cpp:3013-3019`**, **`source/lib/core/rocpd/data_storage/database.cpp:62-66`**, **`source/bin/rocprof-sys-causal/impl.cpp:399`**
```cpp
tim::filepath::makedir(path);            // config UMP output dir
tim::filepath::makedir(_db_dirname);     // db parent dir
filepath::makedir(_config_folder);       // causal config output dir
```
What: `makedir` (recursive `mkdir -p`, umask mode, EEXIST-tolerant). All three discard the return value; two redundantly pre-guard with `exists`/`direxists`.

**`source/bin/rocprof-sys-instrument/rocprof-sys-instrument.cpp:2501-2510`**
```cpp
auto outdir = outf.substr(0, outf.find_last_of('/'));
tim::makedir(outdir);          // ensure output dir before writeFile
bool success = app_binary->writeFile(outfile.c_str());
```
What: Create the instrumented-binary output directory via the `tim::makedir` façade.

**`source/lib/core/trace_cache/tests/test_discovery.cpp:59-69`**, **`source/lib/common/tests/test_path.cpp:22-31` (`mkdtemp`), `55-60` (`mkdir`), `360-373` (nested `mkdir`)**
```cpp
if(::mkdir(m_dir.c_str(), S_IRWXU) != 0 && errno != EEXIST) FAIL() << ...;
char* dir = mkdtemp(tmpl);
mkdir(path.c_str(), 0755);
```
What: Test fixtures create temp/working directories via raw libc `mkdir`/`mkdtemp`.

---

# G12 — Open OUTPUT stream / `fopen` **with auto-mkdir** (the shim)

The `tim::filepath::open(stream, path, args...)` shim splits `path` into dir+base,
`makedir`s the dir, falls back to `./<base>` on mkdir failure, then opens the stream.
`fopen` is the C-stream variant. All 15 sites below rely on the auto-mkdir behavior.

**`source/bin/rocprof-sys-instrument/info.hpp:59-71`, `90-99`, `118-127`** — `.txt`/`.xml`/`.json` report outputs.
```cpp
std::ofstream ofs{};
if(!tim::filepath::open(ofs, _oname)) _handle_error();
```

**`source/bin/rocprof-sys-instrument/rocprof-sys-instrument.cpp:1256-1264`** — instrumentation log file.
```cpp
if(!filepath::open(*log_ofs, logfile))
    throw std::runtime_error(JOIN(" ", "Error opening log output file", logfile));
```

**`source/bin/rocprof-sys-avail/generate_config.cpp:267-278`** — generated config file.
```cpp
if(filepath::open(_ofs, _fname)) { ... } else { throw std::runtime_error(...); }
```

**`source/lib/rocprof-sys/library/coverage.cpp:211-216`, `259-262`** — coverage `.txt`/`.json`.
```cpp
std::ofstream ofs{};
if(tim::filepath::open(ofs, _fname)) { ... }
```

**`source/lib/rocprof-sys/library/causal/data.cpp:698-712`** — causal symbol-info outputs.
```cpp
auto _ofs = std::ofstream{};
if(tim::filepath::open(_ofs, ofname)) { save_line_info_impl(_ofs, ...); save_maps_info_impl(_ofs); }
```

**`source/lib/rocprof-sys/library/causal/experiment.cpp:543-549`, `577-580`** — experiments `.json` / `.coz`.
```cpp
std::ofstream ofs{};
if(tim::filepath::open(ofs, _fname)) { ... }
```

**`source/lib/core/perfetto.cpp:252-257`** and **`source/lib/core/trace_cache/perfetto_processor.cpp:643-648`** — perfetto trace (forward `std::ios::out|binary`).
```cpp
std::ofstream ofs{};
if(!filepath::open(ofs, _filename, std::ios::out | std::ios::binary)) { ... }
```

**`source/lib/core/config.cpp:3300-3303` (open)** and **`3332-3342` (fopen)** — temp-file create.
```cpp
auto _ofs = std::ofstream{}; filepath::open(_ofs, filename);   // return ignored
...
file = filepath::fopen(filename, _mode);                       // the ONLY fopen shim site
```

**`source/python/libpyrocprofsys.cpp:818-826`** — coverage save (`.json`).
```cpp
std::ofstream ofs{};
if(tim::filepath::open(ofs, _name)) { ... }
```

---

# G13 — Open INPUT stream (shim or direct, incl. procfs/sysfs)

**`source/lib/rocprof-sys/library/causal/experiment.cpp:672-681`** — experiments input via shim.
```cpp
auto ifs = std::ifstream{};
if(tim::filepath::open(ifs, _fname)) { ... }
```
What: The one ifstream use of the `filepath::open` shim.

**`source/lib/core/config.cpp:323-328`, `453-458`, `1315-1318`, `1417-1424`** — config files & procfs.
```cpp
auto input = std::ifstream{ filepath };                                 // :325 config pre-scan
std::ifstream ifs{ json_path };                                         // :455 json root check
std::ifstream _fparanoid{ "/proc/sys/kernel/perf_event_paranoid" };    // :1315
std::ifstream _in{ expanded_filename };                                 // :1417 echo for debug
```
What: Direct input streams for config files and a procfs knob.

**`source/lib/rocprof-sys/library.cpp:349-364`, `489-494`** — procfs/sysfs.
```cpp
fcmdline << "/proc/" << _pid << "/cmdline"; auto ifs = std::ifstream{ fcmdline.str().c_str() };  // read_command_line
std::ifstream _fenforcing{ "/sys/fs/selinux/enforce" };                                          // SELinux mode
```
What: Read process cmdline and SELinux enforce flag.

**`source/lib/rocprof-sys/library/causal/data.cpp:346-354`** — `/proc/<pid>/maps`.
```cpp
auto _maps_file = fmt::format("/proc/{}/maps", process::get_id());
auto _ifs       = std::ifstream{ _maps_file };
```

**`source/lib/rocprof-sys/library/causal/experiment.cpp:559-566`** — existing `.coz` for append.
```cpp
std::ifstream ifs{ _fname }; if(ifs) { ... }
```

**`source/bin/common/preset_registry.cpp:130-134`** — preset JSON.
```cpp
std::ifstream ifs{ filepath }; if(!ifs.is_open()) return std::nullopt;
```

**`source/python/libpyrocprofsys.cpp:774-781`** — coverage JSON input.
```cpp
std::ifstream ifs{ _inp }; if(ifs) { ... }
```

**`source/lib/core/perfetto.cpp:173-188`** and **`source/lib/core/trace_cache/perfetto_processor.cpp:577-595`** — perfetto temp file via `::fopen("rb")`.
```cpp
FILE* _fdata = ::fopen(_tmp_file->filename.c_str(), "rb"); ... ::fread(...); ::fclose(_fdata);
```

---

# G14 — Direct `std::ofstream`/`ifstream`/`fopen` (no shim, no auto-mkdir)

**`source/bin/rocprof-sys-instrument/info.hpp:163-183`, `185-193`** — XML/JSON instrumentation-info input.
```cpp
std::ifstream ifs{ _iname }; if(!ifs) _handle_error();
```

**`source/lib/core/trace_cache/buffer_storage.cpp:42-52`** — buffered-storage output.
```cpp
m_ofs = std::ofstream{ m_filepath, std::ios::binary | std::ios::out };
if(!m_ofs.good()) throw std::runtime_error(...);
```

**`source/lib/core/trace_cache/unified_memory_processor.cpp:153-159`, `168-174`** — UMP txt/json output.
```cpp
std::ofstream txt_file(txt_path); if(!txt_file.is_open()) LOG_ERROR(...);
std::ofstream json_file(json_path); ...
```

**`source/lib/core/trace_cache/metadata_registry.cpp:844-855`, `869-881`** — metadata json out/in.
```cpp
std::ofstream file(filepath); ... file << json_string;
std::ifstream file(filepath); ... file >> json;
```

**`source/lib/core/trace_cache/storage_parser.hpp:52-59`** — buffered-storage input.
```cpp
std::ifstream ifs(m_filename, std::ios::binary); if(!ifs.good()) throw std::runtime_error(...);
```

**`source/lib/core/node_info.cpp:17-24`, `mproc.cpp:27-33`, `cpu.cpp:22-31`** — fixed/procfs paths.
```cpp
auto ifs = std::ifstream{ "/etc/machine-id" };
std::ifstream _ifs{ fmt::format("/proc/{}/task/{}/children", _ppid, _ppid) };
std::ifstream cpuinfo_file("/proc/cpuinfo");
```

**`source/bin/rocprof-sys-causal/impl.cpp:784-793`** — generated causal config files.
```cpp
fname << _config_folder << "/causal-" << std::setw(nwidth) << i << ".cfg";
std::ofstream _ofs{ fname.str() };
```

**`source/lib/core/trace_cache/discovery.cpp:100-111`** — delete cache files (`std::remove`).
```cpp
if(std::remove(fname->c_str()) == 0) LOG_DEBUG("Removed file: {}", *fname);
else if(errno != ENOENT) LOG_WARNING(...);
```

**Tests:** `test_path.cpp:40-46` (`ofstream` create), `204-225` (binary write); `test_discovery.cpp:74-77` (`ofstream{}.put`), `40-54` (`unlink`/`rmdir`).

---

# G15 — Build / join path components

Dominant idioms: `common::join('/', ...)`, `TIMEMORY_JOIN('/', ...)`/`JOIN`, and
`fmt::format("{}/{}", ...)`. Representative sites:

- **`common/path.hpp`**: `find_path` (`join('/', itr, _path)`, `join('/', dirname(itr), sitr, _path)`), `get_internal_libpath/script_path/libdir` (`join('/', root, "lib"/"lib64"/...)`), `get_rocprofsys_root` (`join('/', _exe_dir, "..")`).
- **`common/setup.hpp:74-88`**: `join('/', _omnilib_path, ::basename(_omnilib))` + prepend to search paths.
- **`bin/common/preset_registry.cpp:24-49`, `170-197`, `256-257`**: `join('/', root, "share", ..., "presets")`, `join('/', dir, name + ".json")`, `join('/', m_directory, filename)`.
- **`lib/core/dynamic_library.cpp:50-56`**: `fmt::format("{}/{}", itr, _name)` and `"{}/{}/{}"` (dir + suffix + name).
- **`lib/core/config.cpp`**: many `fmt::format("{}/{}", pwd/cwd, result)` "make absolute" joins (`:2623`, `:2907`, `:2940`, `:2998`, `:3033`), subdirectory build (`:3477`), trailing-slash dir normalization (`:2930-2938`).
- **`lib/core/trace_cache/*`**: `trace_cache::tmp_directory + filename` (discovery), `get_output_absolute_path(...)` (unified memory), `fmt::format("{}/{}", ...)` (test).
- **`lib/core/mproc.cpp:27`**: `fmt::format("/proc/{}/task/{}/children", ...)`.
- **`lib/rocprof-sys/library/causal/data.cpp:346`**: `fmt::format("/proc/{}/maps", ...)`.
- **`lib/core/perfetto.cpp:282-287`, `trace_cache/discovery.cpp:147-150`**: `fmt::format("{}/{}", _script_dir, script)`.
- **`bin/rocprof-sys-causal/impl.cpp:205-213`**: `join(":", LIBPTHREAD_SO, realpath(get_internal_libpath(...)))`.
- **`bin/rocprof-sys-instrument/internal_libs.cpp:185-196`, `419`, `531`, `548`**: `join('/', ditr, "lib")`, `join('/', base, libdir, name)`, `join('/', itr, _lib_v)`.
- **`bin/rocprof-sys-instrument/rocprof-sys-instrument.cpp`**: `JOIN('/', _omni_root, "bin"/"lib"/...)` (`:298-309`), `JOIN('/', _omni_lib_path, "rocprofiler-systems", ...)` (`:320-330`), `JOIN('/', get_cwd(), basename)` (`:1219`), default-output joins (`:1224-1236`), search candidates `JOIN('/', itr, _name / basename)` (`:2803-2805`), `absolute()` join (`:2786`), canonicalize reassembly (`:2773-2777`).
- **`bin/rocprof-sys-avail/generate_config.cpp:194-202`**: reassemble output dir from split components with `TIMEMORY_JOIN('/', ...)`.

---

# G16 — Path-string classification (extension / prefix checks)

**Extension checks (`.so` / `.a` / `.json` / …):**
- `rocprof-sys-instrument.cpp:1177-1181` — `.so.` / `.so` (len-3) / `.a` (len-2) suffix + `lib` prefix on the target basename.
- `rocprof-sys-instrument.cpp:1221-1236` — no-`.`→exe; `lib`/`.so`/`.a`→library (output-name layout).
- `rocprof-sys-instrument.cpp:1693-1704` — has `.so`/`.a`? else append `.so`/`.a`.
- `rocprof-sys-instrument.cpp:2848-2850`, `2874-2878` — `lib`/`.so`/`.a` on basename (search-order + `.so` retry).
- `rocprof-sys-instrument.cpp:343-352` — regex over link-map names incl. `\.(so|a)`.
- `internal_libs.cpp:264-295` — hard-coded `.so`-suffixed internal-lib name lists.
- `config.cpp:1400`, `3572-3578` — `.ends_with(".json")`; strip trailing `.txt/.json/.xml`.
- `generate_config.cpp:188-216` — detect `.cfg/.txt/.json/.xml` by suffix; split dir components.
- `kokkosp.cpp:286-302`, `library.cpp:885-897`, `dl.cpp:101-124`, `1189-1196`, `1270-1281` — `librocprof-sys*.so` substring checks in `LD_PRELOAD` / `KOKKOS_TOOLS_LIBS` / link map.
- `preset_registry.cpp:155-170`, `230-246` — `.json` suffix (+ reject `..` traversal).
- `info.hpp:152-154` (also G3) — extension extraction.
- `module_function.cpp:438-443`, `496-531` — regexes: source-tree layout, `.s/.S`, `../sysdeps/`, `/build/`, `lib(elf)…`.

**Prefix / absolute checks (leading `/`, `./`, `../`):**
- `dynamic_library.cpp:26-32` (`find('/')==0`), `76-94` (env-val/filename absolute checks).
- `config.cpp:2595-2605`, `2900-2905`, `2955-2976`, `3473-3487` — `find_last_of('/')`/`rfind('/')` dir splits, `find_last_of('.')` ext split, leading-`/` absolute tests.
- `config.cpp` "make absolute" leading-`/` tests at `:2623`, `:2907`, `:2940`, `:2998`, `:3033`; trailing-slash normalization `:2930-2938`.
- `rocprof-sys-instrument.cpp:2751-2787` — `canonicalize`/`absolute`: `./`/`../` prefixes, leading `/`, manual `.`/`..` resolution.
- `rocprof-sys-instrument.cpp:2504-2515` — output name contains `/` / begins `/`.
- `libpyrocprofsys.cpp:456-467` — basename slice + module-path prefix match (`strncmp`).
- `preset_registry.cpp:155-170` — `/` present ⇒ "is a path".

---

# G17 — Search for file across directories (`find_path` / resolvers)

**`source/lib/common/path.hpp:187-237`** *(implementation)* — `find_path`: return input if absolute+exists; else try each `:`-delimited dir (and `lib/lib64/../lib/../lib64` under non-lib dirs), joining with `/` and testing `exists`. Plus `get_default_lib_search_paths` (`:175-185`) building the `:`-list from PATH/LD_LIBRARY_PATH/LIBRARY_PATH/PWD/`.`.

**Call sites:** `dl.cpp:200-204` (omni/dl/user libs), `setup.hpp:90-91` (omni + omni-dl), `test_path.cpp:306-325` (tests).

**`source/lib/core/dynamic_library.cpp:48-59`** — search candidate dirs (env/hints + suffixes), first `exists` wins.

**`source/bin/rocprof-sys-instrument/internal_libs.cpp:201-248`** — parse `ldconfig -p` output for library dirs; `344-372` — resolve internal libs via `find_library`/`find_libraries`; `523-552` — `find_library`/`find_libraries` (link-map first, then join+`exists` over search paths).

**`source/bin/rocprof-sys-instrument/rocprof-sys-instrument.cpp:2792-2817`** — `get_absolute_filepath` core resolver (join dir+name/basename, first exists+is_file); `2861-2881` — `get_absolute_exe_filepath` / `get_absolute_lib_filepath` (with `.so` retry); `1663-1673`, `2026-2035` — resolve instrumentation libs before load.

---

# G18 — Install-layout discovery (`get_rocprofsys_root` / `get_internal_*`)

**`source/lib/common/path.hpp:408-440`** *(implementations)*
```cpp
get_rocprofsys_root()      // realpath("/proc/self/exe") → dirname → join('/', dir, "..")
get_internal_libpath(lib)  // {root}/{lib,lib64}/lib  (first exists)
get_internal_script_path() // {root}/libexec/rocprofiler-systems
get_internal_libdir()      // {root}/lib
```
What: The canonical install-layout resolvers, all built on `realpath`/`dirname`/`join`/`exists`.

**Call sites:** `argparse.cpp:111-120`, `impl.cpp:205-213`, `tool_runner.cpp:329-330`, `preset_registry.cpp:24-49`, `setup.hpp` (via `get_origin`), `internal_libs.cpp:410-425` (exe-relative probe of `{lib,lib64}/librocprof-sys-*.so`), `rocprof-sys-instrument.cpp:182-193` (`get_internal_libdir` seeding), `293-334` (ROOT-relative bin/lib/libexec derivation), `2922-2939` (`find_dyn_api_rt`), `link_map.cpp:84-90` (`config::get_exe_realpath`), and the `test_path.cpp:256-292` tests.

---

# G19 — Dynamic-linker introspection yielding paths

Pattern: `dlopen` + `dlinfo(RTLD_DI_LINKMAP)` to walk the `link_map` chain
(`l_name`/`l_next`), or `dlinfo(RTLD_DI_ORIGIN)` for a library's origin dir.
These yield filesystem paths that then flow into `realpath`/`dirname`/search logic.

**Implementations:** `common/path.hpp:340-406` (`get_link_map` ×2, `get_origin`),
`dl.cpp:1152-1182` (`get_link_map`), `internal_libs.cpp:103-160` (`get_linked_path`,
`get_link_map`), `details.cpp:663-682`, `729-737`, `761-768` (loaded-path / link-map /
origin), `link_map.cpp:30-58`, `79-96` (`get_linked_path`, chain walk).

**Call sites feeding paths onward:** `analysis.cpp:221-225` (exclude ranges),
`rocprof-sys-instrument.cpp:341-356` (add loaded-lib dirs to search paths),
`dl.cpp:1189-1196`, `1270-1281` (mode detection by `.so` name),
`setup.hpp:75-76` (`get_origin` for omnilib dirs),
`dynamic_library.cpp:28-33` (procfs `get_maps` — related module introspection).

---

## Cross-cutting observations for the redesign

1. **`dirname`/`basename` are reimplemented many ways**: `common::path::dirname`,
   `tim::filepath::(dir|base)name`, `tool_runner::basename_of`, libc `::basename`,
   and ad-hoc `find_last_of('/')`/`substr` in config.cpp, library.cpp,
   libpyrocprofsys.cpp, instrument.cpp. A single `parent_path`/`filename` primitive
   subsumes all.
2. **`realpath` is uniform** — every non-test site is `realpath(p, nullptr, false)`
   (silent) except `analysis.cpp:209`. The existing `common::path::realpath`
   (POSIX + fallback-to-input) already matches `tim::filepath::realpath` exactly.
3. **Two `exists` semantics** (G6 vs G8) are the main correctness trap and the
   source of ≥3 latent bugs (`argparse.cpp:136`, `impl.cpp:399`,
   `instrument.hpp:139`). A redesign should offer explicit
   `file_exists` / `dir_exists` / `any_exists` names.
4. **The auto-mkdir `open`/`fopen` shim (G12)** is copy-pasted 3× in timemory and
   is load-bearing (dir creation + `./<base>` fallback). One templated
   `open(stream, path, args...)` replaces it.
5. **`is_text_file` (G10) is duplicated** in `details.cpp` and `common/path.hpp`.
6. **Path building (G15)** mixes `join`, `JOIN`/`TIMEMORY_JOIN`, and
   `fmt::format("{}/{}")` — a single `path / a / b` join (or keeping `common::join`)
   would unify it.
7. **Path classification (G16)** — ~30 brittle `find(x)==0` / `find(x)==len-n`
   idioms are prime candidates for C++20 `starts_with`/`ends_with` (contains checks
   stay `find(x)!=npos` — `std::string::contains` is C++23).
8. **Directory enumeration (G7)** uses raw `opendir`/`readdir` in 3 places;
   `std::filesystem::directory_iterator` would replace them (production impact only
   in `discovery.cpp` + `preset_registry.cpp`).
9. **`std::filesystem` today**: production = none; tests = `test_common::fs`
   (ghc/std/experimental fallback shim). The `dl`/`common` header-only layer that
   the preloaded library consumes is deliberately syscall-based.

## Category → file index (for the migration)

- **`tim::filepath` removal touches**: instrument (`details/internal_libs/module_function/rocprof-sys-instrument[.cpp/.hpp]/info.hpp`), core (`config/argparse/common.hpp/database/discovery`), binary (`analysis/link_map/dwarf_entry/symbol`), runtime (`coverage/causal:data,experiment/kokkosp/library`), dl (`dl.cpp` readlink), common (`environment.hpp`), avail (`generate_config`), causal bin (`impl`), python (`libpyrocprofsys`).
- **`common::path` is the timemory-free target** (definitions in `path.hpp`).
- **Raw-syscall consolidation candidates**: `internal_libs.cpp` (stat), `rocprof-sys-instrument.cpp` (`is_file`/`is_directory`/`get_cwd`/`canonicalize`), `trace_cache/*` (opendir/stat/remove/ofstream), `node_info/mproc/cpu` (procfs ifstream), `perfetto*/config` (`::fopen`).
