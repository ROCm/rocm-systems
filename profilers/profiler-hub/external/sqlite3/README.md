# Bundled SQLite3 amalgamation

`profiler-hub` links a private, symbol-sealed copy of SQLite3. This directory
holds the pre-generated **amalgamation** it is built from.

`cmake/sqlite3.cmake` compiles `sqlite3.c` from here directly. Configure does
**no** network access, **no** `git clone`, and does **not** need `tclsh`.

| | |
|---|---|
| Upstream release | **SQLite 3.45.3** (2024-04-15) |
| `SQLITE_SOURCE_ID` | `2024-04-15 13:34:05 8653b758870e6ef0c98d46b3ace27849054af85da891eb121e9aaa537f1e8355` |
| Upstream git tag | [`version-3.45.3`](https://github.com/sqlite/sqlite/tree/version-3.45.3) (`b74eb00e2cb05d9749859e6fbe77d229ad1dc1e1`) |
| Source archive | <https://www.sqlite.org/2024/sqlite-amalgamation-3450300.zip> |
| Archive SHA256 | `ea170e73e447703e8359308ca2e4366a3ae0c4304a8665896f068c736781c651` |
| `sqlite3.c` SHA256 | `9ca336fbcbff9f1d78b4f45b6a19583fcc097192310dd2f5f6cd43b9a33d7d69` (9 027 389 bytes) |
| `sqlite3.h` SHA256 | `882ad3c0448d0324fb3a6b1a85333a9173d539ac669c9972ae1f03722ff86282` (641 889 bytes) |
| Licence | Public domain — <https://www.sqlite.org/copyright.html> |

Only `sqlite3.c` and `sqlite3.h` are kept. The archive also ships `shell.c` and
`sqlite3ext.h`; `profiler-hub` uses neither, and the amalgamation is
self-contained without them.

The same three values are duplicated in `cmake/sqlite3.cmake`
(`SQLITE3_AMALGAMATION_VERSION`, `SQLITE3_AMALGAMATION_SHA256_C`,
`SQLITE3_AMALGAMATION_SHA256_H`), which verifies both files at configure time.
**They must be updated in the same commit as the sources.**

______________________________________________________________________

## Why the amalgamation is stored rather than generated

The previous implementation cloned `sqlite/sqlite` at configure time and ran
`./configure --disable-tcl && make sqlite3.c`. That make target invokes
`tool/mksqlite3c.tcl`, so it hard-requires `tclsh` — `--disable-tcl` only drops
the Tcl *bindings*. Tcl is not a documented prerequisite of this project, of
ROCm, or of TheRock, so the build failed on any lean Ubuntu/Debian image
(ROCm/rocm-systems#10059). It also meant every configure did a ~2000-file clone.

Storing the pre-generated amalgamation removes the Tcl dependency, the clone
and the configure-time network round trip in one step.

______________________________________________________________________

## Updating to a newer SQLite release

Work through the whole checklist; steps 2 and 3 are the ones that actually
establish provenance.

### 1. Fetch the published amalgamation

Pick the release and its `sqlite.org` year directory from
<https://www.sqlite.org/chronology.html>. The archive name encodes the version
as `(major)(minor:2)(patch:2)00` — 3.45.3 is `3450300`, 3.50.1 is `3500100`.

```bash
VERSION=3.45.3
YEAR=2024
NUM=3450300

curl -fLO "https://www.sqlite.org/${YEAR}/sqlite-amalgamation-${NUM}.zip"
sha256sum "sqlite-amalgamation-${NUM}.zip"
unzip -q "sqlite-amalgamation-${NUM}.zip"
sha256sum "sqlite-amalgamation-${NUM}"/sqlite3.[ch]
```

Cross-check the archive hash against the SHA3-256 published on
<https://www.sqlite.org/download.html> for the current release, or against a
second independent download, before trusting it.

### 2. Verify it against what `make sqlite3.c` generates

This is the check that proves the stored file really is the amalgamation the
old build produced. It needs `tclsh` — that is fine, because this runs **once,
on a maintainer's machine**, not in anyone's build.

```bash
docker run --rm -v "$PWD:/out" ubuntu:24.04 bash -lc '
  set -e
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
      git build-essential ca-certificates autoconf automake tcl
  git clone -q --depth 1 --branch version-'"${VERSION}"' \
      https://github.com/sqlite/sqlite.git /sq
  cd /sq && ./configure --disable-tcl >/dev/null && make sqlite3.c >/dev/null
  cp /sq/sqlite3.c /sq/sqlite3.h /out/
'
```

Then diff the generated pair against the published pair.

- `sqlite3.h` must be **byte-identical**.
- `sqlite3.c` is expected to differ by exactly three lines, and by nothing
  else. The GitHub mirror of SQLite rewrites `manifest.uuid`, so
  `mksqlite3c.tcl` annotates the generated file's header comment:

  ```diff
  21c21,23
  < ** 8653b758870e6ef0c98d46b3ace27849054a.
  ---
  > ** 8653b758870e6ef0c98d46b3ace27849054a with changes in files:
  > **
  > **    manifest.uuid
  ```

  This is a comment inside the leading banner. `SQLITE_VERSION`,
  `SQLITE_SOURCE_ID` and every line of code are identical, so the compiled
  object is unchanged. The published archive is the cleaner of the two, which
  is why it — not the git-generated file — is what gets stored.

If `diff` reports anything beyond those three lines, stop and investigate
before committing.

### 3. Confirm the compile definitions still apply

`cmake/sqlite3.cmake` builds with `SQLITE_OMIT_DEPRECATED`,
`SQLITE_OMIT_PROGRESS_CALLBACK`, `SQLITE_OMIT_SHARED_CACHE`,
`SQLITE_LIKE_DOESNT_MATCH_BLOBS`, `SQLITE_DEFAULT_MEMSTATUS=0`,
`SQLITE_THREADSAFE=1` and `SQLITE_DEFAULT_WAL_SYNCHRONOUS=1`. Check the release
notes for any change to these options, and check that
`source/data_storage/backends/sqlite_api_policy.hpp` still matches the C API.

### 4. Replace the files and update the metadata

```bash
cp "sqlite-amalgamation-${NUM}"/sqlite3.c "sqlite-amalgamation-${NUM}"/sqlite3.h \
   profilers/profiler-hub/external/sqlite3/
```

Update, in the same commit:

- the table at the top of this file (version, source id, tag, URL, all three
  SHA256s, byte sizes);
- `SQLITE3_AMALGAMATION_VERSION`, `SQLITE3_AMALGAMATION_SHA256_C` and
  `SQLITE3_AMALGAMATION_SHA256_H` in `profilers/profiler-hub/cmake/sqlite3.cmake`;
- `profilers/profiler-hub/CHANGELOG.md`.

A stale checksum is a hard configure error, so a half-done bump cannot ship
silently.

### 5. Build and test

```bash
cmake -S profilers/profiler-hub -B /tmp/phb -GNinja
cmake --build /tmp/phb --parallel
ctest --test-dir /tmp/phb --output-on-failure
```

`profilers/profiler-hub/external/` is excluded from `clang-format`, `gersemi`,
`clang-tidy` and the repository `pre-commit` hooks, so the upstream sources are
never reformatted. Do not remove those exclusions, and do not hand-edit
`sqlite3.c` or `sqlite3.h` — any local change must instead be expressed as a
`SQLITE_*` compile definition in `cmake/sqlite3.cmake`.

______________________________________________________________________

## If these files are moved to DVC

The repository already uses [DVC](https://dvc.org) for large artifacts — see
the *Large File Storage* section of the top-level `CONTRIBUTING.md`, and
`shared/amdgpu-windows-interop/`, whose `.lib`/`.a` blobs are stored that way.
Keeping the amalgamation in git is a deliberate choice, because unlike those
blobs it must be readable by build paths that have no DVC context at all:

- `projects/rocprofiler-systems/cmake/ProfilerHub.cmake` obtains profiler-hub by
  doing `git init` + a `--filter=blob:none` sparse fetch of
  `profilers/profiler-hub` into the build tree. That throwaway checkout has no
  top-level `.dvc/` directory and no remote configuration, so `dvc pull` cannot
  run in it at all.
- The `profiler-hub-*.yml` workflows sparse-check out `profilers/profiler-hub`
  and `.github/workflows` only — again, no `.dvc/`.
- TheRock's `build_tools/fetch_sources.py` defaults `--dvc-projects` to
  `["rocm-libraries", "rocm-systems"]` on Windows but only `["rocm-libraries"]`
  on Linux, so `rocm-systems` DVC artifacts are not pulled for Linux builds.

Should those constraints change and a move to DVC become worthwhile, the CMake
side already supports it: `cmake/sqlite3.cmake` detects a `sqlite3.c.dvc`
pointer next to a missing `sqlite3.c` and fails with a `dvc pull` instruction
instead of a generic "not found". The migration is then:

```bash
pip install 'dvc[s3]'

cd profilers/profiler-hub/external/sqlite3
git rm --cached sqlite3.c sqlite3.h          # keep the files on disk
dvc add sqlite3.c sqlite3.h                  # writes *.dvc + .gitignore entries
dvc push                                     # uploads to s3://therock-dvc/rocm-systems

cd -
git add profilers/profiler-hub/external/sqlite3/sqlite3.c.dvc \
        profilers/profiler-hub/external/sqlite3/sqlite3.h.dvc \
        profilers/profiler-hub/external/sqlite3/.gitignore
```

`dvc push` needs write credentials for the S3 remote; the bucket allows
anonymous **read** only (`allow_anonymous_login = true` in `.dvc/config`), and
an unauthenticated push fails with `PermissionError: Access Denied`. Ask a
project lead for credentials, as `CONTRIBUTING.md` describes.

Every consumer would then also need a `dvc pull` step: the five building
`profiler-hub-*.yml` workflows and the `rocprofiler-systems-*` workflows would
need `.dvc` added to their sparse-checkout sets, `ProfilerHub.cmake` would need
to stop using a DVC-less sparse fetch, and TheRock's Linux `--dvc-projects`
default would need `rocm-systems` added. Treat that as the actual cost of the
move.
