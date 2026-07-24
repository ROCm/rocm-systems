# rocjitsu stylist

`stylist` applies the locally enforceable subset of the rocjitsu C++ style
guide. It uses only Python's standard library and the repository's pinned
`clang-format`.

The normal mode checks:

- non-POD `struct` declarations (inheritance, access restrictions, constructors,
  destructors, operators, and methods),
- CamelCase types, snake_case methods/data members, non-public member suffixes,
  and `kCamelCase` global constants,
- class access-section uniqueness and `public` / `protected` / `private` order,
- module top-level namespaces,
- forbidden `printf`, `fprintf`, and `std::cerr` logging,
- unprefixed macro definitions.

It also fixes formatting, `class` template parameters, canonical header guards,
decorative separator-only comments, and owned C++ uses of C standard headers.

From any working directory, pass files, directories, or both:

```bash
emulation/rocjitsu/scripts/stylist.py path/to/file.cpp path/to/directory
```

With no paths, the command processes all supported C and C++ files under
`emulation/rocjitsu`. Build and tool cache directories are skipped.

Use `--check` in read-only environments. It reports every file that would
change or introduces a semantic violation, and returns 1 when work is needed:

```bash
emulation/rocjitsu/scripts/stylist.py --check
```

`--strict` additionally requires `@brief` and `@details` documentation for
owned public header types and record methods. `--all-violations` displays both
existing and new findings. Existing findings are counted in
`stylist_baseline.json`, so
normal and strict checks reject new debt without making unrelated changes fail.
Refresh that baseline only after reviewing the full report:

```bash
emulation/rocjitsu/scripts/stylist.py --strict --all-violations --check
emulation/rocjitsu/scripts/stylist.py --update-baseline
```

The amdisa generator uses `--format-only`: generated directories are formatted
deterministically but semantic checks remain the responsibility of the
generator and generated-code tests.

Fix mode returns 0 when every selected file conforms after the command,
including when files were changed. Missing paths, unavailable tools, and
formatter or UTF-8 decoding failures return 2. `--jobs` controls parallelism
from 1 to 256 workers and `--verbose` reports successful work.

The stylist intentionally does not enforce rules that require semantic or
cross-file analysis. Generated files, vendored `external_headers`, C sources,
and C-compatible public API headers are exempt from semantic checks. Rules such
as whether `auto` is obvious, comment prose quality, exception safety, hot-path
throwing, and namespace necessity still require human or compiler-level review.
The complete policy remains in `docs/style.md`.
