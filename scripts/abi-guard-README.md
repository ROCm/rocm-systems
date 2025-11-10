# ABI Guard (libabigail) — Quick Start

This guide gets you from zero to a working ABI policy check using **`abi-guard.py`** and libabigail’s `abidw/abidiff`.
You’ll generate a config, record your current ABI, compare against a baseline, and enforce SemVer bumps automatically.
In general, you need a `VERSION` file and a `versioning.yml` file.

---

## Prerequisites

* **Linux** with your project checked out and buildable
* **libabigail** tools on `PATH`:

  * `abidw`, `abidiff` (e.g., `sudo apt install libabigail-tools` or build from source)
* **Python 3.8+** and packages:

  * `PyYAML`, `Jinja2`
  * Optional but recommended: `Ninja`, `CMake`
* Your project should have a **`VERSION`** file (plain text `MAJOR.MINOR.PATCH`)

The Python requirements are located in [abi-guard-requirements.txt](abi-guard-requirements.txt).

---

## Install (local)

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install PyYAML Jinja2
# (this script is standalone; you can run it directly)
chmod +x abi-guard.py
```

---

## 1) Generate a starter `versioning.yml`

The config tells ABI Guard where to find headers, sources, built libraries, and your `VERSION` file.

```bash
./abi-guard.py generate \
  --project-name mylib \
  --include-dirs include/mylib \
  --source-dirs  src \
  --library-names mylib \
  -d . \
  -o versioning.yml
```

What you get (excerpt):

```yaml
versioning:
  name: mylib
  build-directory: build
  source-tree:
    version-file: VERSION
    working-directory: .
    require-working-directory-exists: true
    headers:
      recursive-include:
        - include/mylib/**
    sources:
      recursive-include:
        - src/**
  build-tree:
    working-directory: .
    abi-check:
      recursive-include:
        - libmylib*.so*
  install-tree:
    working-directory: ../..
    version-file: share/mylib/VERSION
    headers:
      - include/mylib/**/*.h
      - include/mylib/**/*.hpp
    abi-check:
      recursive-include:
        - libmylib*.so*
```

> **Tip:**
>
> * **build mode** (`-m build`) uses `source-tree` and `build-tree` (good when you have uninstalled `.so` files under `build/`).
> * **install mode** (`-m install`) uses `install-tree` (good if you compare two installed prefixes or artifacts).

---

## 2) Query the config (sanity check)

```bash
./abi-guard.py query -i versioning.yml -m build all
./abi-guard.py query -i versioning.yml -m build headers
```

---

## 3) Manage your `VERSION` file (SemVer helper)

Show current version:

```bash
./abi-guard.py version -i versioning.yml -m build --echo
# prints: VERSION_PATH: X.Y.Z[-build]~hash
```

Bump with policy:

```bash
# major/minor/patch bump helpers
./abi-guard.py version -i versioning.yml -m build --bump-major
./abi-guard.py version -i versioning.yml -m build --bump-minor
./abi-guard.py version -i versioning.yml -m build --bump-patch

# or set directly
./abi-guard.py version -i versioning.yml -m build --set 1.4.0
```

> When writing the `VERSION`, the script recomputes a **content hash** of your declared *headers + sources*.
> It strips comments (`C/C++`, `Python`, `CMake`) and whitespace before hashing, to avoid noisy diffs.

---

## 4) Generate ABI XML snapshots (optional, cacheable)

If you want to pre-generate the libabigail XML dumps (handy for CI caching):

```bash
# Build your project first so libs exist under build/
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j

# Generate ABI XML for the current (head) libs
./abi-guard.py abi-generate -i versioning.yml -m build \
  --abidw-args "--headers-only" \
  -d abi-guard/head-xml
```

This writes `*.abi` files (one per library) into `abi-guard/head-xml/`.

---

## 5) Compare Baseline vs Head & Enforce SemVer

You need **two configs**:

* **baseline** (e.g., `base/versioning.yml`) pointing at an older source + build or installation
* **head** (e.g., `head/versioning.yml`) pointing at the current source + build or installation

Run the check:

```bash
./abi-guard.py check \
  -m build \
  -i build-head/versioning.yml \
  -b build-base/versioning.yml \
  -d abi-guard/reports
```

Optional: reuse pre-generated XML:

```bash
./abi-guard.py check \
  -i versioning.yml -m build \
  -b versioning-baseline.yml \
  --head-abi-xml abi-guard/head-xml \
  --base-abi-xml abi-guard/base-xml \
  -d abi-guard/reports
```

**Outputs:**

* Per-library `abidiff` reports in `abi-guard/reports/*.txt`
* Non-zero exit on policy failures (great for CI)

**SemVer policy enforced by `check`:**

* **Incompatible ABI breaks** → require **major** version increment.
* **Compatible API changes** (added/changed/deleted public symbols; added/removed libs) → require **minor** increment (or major).
* **No public ABI changes** but headers/sources changed → require **patch** increment.
* **No changes** → version must remain identical.

If the policy is violated, the script prints a clear `::error title=ABI / Version policy:: …` and exits `1`.

---

## 6) Quick content hash (debug/visibility)

```bash
./abi-guard.py hash -i versioning.yml -m build
# prints md5 of headers+sources (comments/whitespace stripped)
```

---

## Logging & Verbosity

Use environment or flags:

```bash
# env
export ROCM_ABI_GUARD_LOG_LEVEL=info

# or CLI
./abi-guard.py ... --log-level debug
./abi-guard.py ... --log-file abi-guard/run.log
```

---

## Typical Local Workflow

```bash
# 1) Create config once
./abi-guard.py generate --project-name mylib --include-dirs include/mylib --source-dirs src --library-names mylib -o versioning.yml

# 2) Build baseline and capture its XML
git checkout v1.2.3
cmake -S . -B build && cmake --build build -j
./abi-guard.py abi-generate -i versioning.yml -m build -d abi-guard/base-xml
cp versioning.yml versioning-baseline.yml

# 3) Move to head, build, capture head XML
git checkout main
cmake -S . -B build && cmake --build build -j
./abi-guard.py abi-generate -i versioning.yml -m build -d abi-guard/head-xml

# 4) Compare & enforce SemVer
./abi-guard.py check -i versioning.yml -m build -b versioning-baseline.yml \
  --head-abi-xml abi-guard/head-xml --base-abi-xml abi-guard/base-xml \
  -d abi-guard/reports
```

---

## Troubleshooting

* **“No ABI libraries found”**
  Ensure your `abi-check.recursive-include` patterns match built `.so` names (e.g., `libmylib*.so*`) and the build has completed.

* **“VERSION must be X.Y.Z”**
  The `VERSION` file should contain `MAJOR.MINOR.PATCH` on the first line. When written by the tool, it also includes a `# hash:` comment.

* **Headers not influencing ABI dumps**
  Pass header hints to `abidw` via `--abidw-args` or define headers in the config. The tool already forwards them as `--header-file` arguments.

* **Comparing versioned SONAMEs**
  The tool normalizes common `.so.X[.Y[.Z]]` suffixes so the same library across versions is matched by basename.

---

## Help

Every subcommand supports `-h/--help`. Examples:

```bash
./abi-guard.py --help
./abi-guard.py check --help
./abi-guard.py generate --help
```

You’re set! Point `versioning.yml` at your headers/libs, build both sides, and let `abi-guard.py` keep your SemVer honest.
