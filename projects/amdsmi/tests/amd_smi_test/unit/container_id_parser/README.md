<!--
SPDX-License-Identifier: MIT
Copyright (c) Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
-->

# Container-ID parser test suite

Tests for `amd::smi::ExtractContainerId`
(`include/amd_smi/impl/amd_smi_container_id_parser.h`), the parser used by
`src/amd_smi/fdinfo.cc` to pull a container identifier from
`/proc/<pid>/cgroup` and populate `amdsmi_proc_info_t.container_name`.

Every test links to the shipped implementation through the thin wrappers in
`container_id_test_util.h`, so the suite can never drift from production
behavior. The cases are table-driven; each row carries a `desc` string.

## Files

### Shared infrastructure

| File | Purpose |
|---|---|
| `container_id_test_util.h` | `ExtractIdInto` / `ExtractIdString` wrappers over the production parser |
| `fixtures.h` | Shared test inputs (e.g. 64-char SHA-256 hex) |
| `guarded_buffer.h` | Canary-protected buffer for overflow detection |

### Positive and adversarial coverage

| File | Focus | Primary CWE / CVE |
|---|---|---|
| `test_container_id_real_world.cc` | Happy-path for Docker, LXC, cgroup v2, K8s formats | — |
| `test_container_id_log_injection.cc` | Newline / ANSI / CR / BEL / BS / Tab | CWE-117, CVE-2017-8816, CVE-2021-23385 |
| `test_container_id_homoglyph.cc` | BiDi override, zero-width, UTF-8 BOM, Cyrillic | CWE-1007, CVE-2021-42574 (Trojan Source) |
| `test_container_id_nul_smuggling.cc` | Embedded / leading NUL bytes | CWE-626, CVE-2006-7243 |
| `test_container_id_prefix_confusion.cc` | Unanchored `find()` false positives | CWE-20, CVE-2020-8617, CVE-2019-14271 |
| `test_container_id_shell_metachars.cc` | `;`, `` ` ``, `$`, `|`, `&`, `>`, quotes, space | CWE-78, CVE-2014-6271, CVE-2018-6789 |
| `test_container_id_dos.cc` | Oversize input, bounded iteration, timing | CWE-400, CVE-2022-0185 |
| `test_container_id_edge_cases.cc` | npos arithmetic, empty input, multi-marker | CWE-190, CVE-2017-1000253 |
| `test_container_id_buffer_bounds.cc` | Canary-protected overflow detection | CWE-120, CWE-787 |

### Fuzzing and performance

| File | Purpose |
|---|---|
| `fuzz_container_id_parser.cc` | Invariant fuzz harness (GTest sweep by default; libFuzzer when compiled with `-DCONTAINER_ID_PARSER_FUZZER_MAIN`) |
| `bench_container_id.cc` | Disabled-by-default microbenchmark for the production parser |

## Build and run

These files are globbed into the `amdsmitst` binary via the parent
`tests/amd_smi_test/CMakeLists.txt`; build with `-DBUILD_TESTS=ON`:

```sh
ctest --test-dir build -R ContainerIdParser
# or, directly:
./build/.../amdsmitst --gtest_filter='ContainerIdParser*'
```

Run the disabled microbenchmark:

```sh
./build/.../amdsmitst --gtest_also_run_disabled_tests --gtest_filter='*Bench*'
```

The parser is header-only, so the suite can also be compiled on its own during
development (the amdsmi `include/` tree must be on the include path):

```sh
c++ -std=c++17 -O2 -Wall -Wextra -I. -I../../../../include *.cc \
    -lgtest -lgtest_main -lpthread -o container_id_test && ./container_id_test
```

**Coverage-guided fuzzing (optional, requires clang):** the same invariants
that `fuzz_container_id_parser.cc` asserts as GTest cases can be driven by
libFuzzer for deeper exploration:

```sh
clang++ -std=c++17 -g -O1 -fsanitize=fuzzer,address,undefined \
    -DCONTAINER_ID_PARSER_FUZZER_MAIN -I. -I../../../../include \
    fuzz_container_id_parser.cc -o container_id_fuzzer
./container_id_fuzzer -max_total_time=60
```

The harness checks, for arbitrary input: no out-of-bounds write (canary),
bounded length, NUL termination, and `[a-zA-Z0-9_-]` output charset.

## Threat model summary

The parser is defended by three layers:

1. **Anchored type_name match** — the container-type string (`docker`, `lxc`)
   must be preceded by `/` (or start-of-line) and followed by `/` or `-`.
   This prevents substring false positives like `/not-docker-evil/…`.

2. **Charset whitelist** — only `[a-zA-Z0-9_-]` is accepted. This is the
   union of Docker SHA-256 hex and LXC container names. Every control byte,
   every non-ASCII byte, every shell metacharacter, every path separator
   is rejected at the byte level. No delimiter trust; no allowlist gap.

3. **Bounded length** — capped at `AMDSMI_MAX_CONTAINER_ID_LENGTH` (64),
   matching Docker's `fullLen = 64`
   ([moby/moby stringid.go](https://github.com/moby/moby/blob/2200f277f9f576886e90ca75929a2bb892b9ef23/client/pkg/stringid/stringid.go#L14-L15)).
   Constant-time worst case in the consume loop.

## Adding a new attack class

1. Create `test_container_id_<cwe_or_name>.cc` in this directory.
2. At the top, document the threat, its CWE link, related public CVEs, and how
   the parser defends against it.
3. Add a table-driven `TEST(ContainerIdParser_<Category>, …)` whose case table
   holds one `{input, expected, desc}` row per adversarial input.
4. The parent `file(GLOB_RECURSE …)` picks up new files automatically.
5. Update the table in this README.
