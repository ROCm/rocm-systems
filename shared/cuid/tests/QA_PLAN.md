# CUID QA

Scope: `shared/cuid` (library) and the CUID API, bindings and CLI in
`projects/amdsmi`. The kernel's only CUID test is a KUnit known-answer test in
amdgpu.

## Test wrappers and hardware gaps

CTest discovers three wrappers: `cuid_unprivileged_tests`,
`cuid_privileged_tests` and `cuid_regression_tests`. The last runs all
`cuidtst.*` cases; some skip without root. Read skip reasons on each run: a
green CTest result alone does not establish privileged or hardware coverage.

Partition and VF unit tests use fixtures; they do not establish CPX/DPX or
guest behavior on hardware. The MI350X release check below does that for
partitions; VF guests are not covered.

## Lookup configurations

* **Tier 1:** the driver publishes CUIDs. With a node key set, amd-smi reports
  the derived value verbatim with source `DRIVER`, for root and ordinary users
  alike. Before a key is set `cuid_derived` fails with `ENODATA`, which leaves a
  whole GPU temporary and a partition without a CUID.
* **Tier 2:** the driver publishes nothing. A whole GPU gets a temporary CUID
  with source `LIBRARY`; a partition gets none.

CPU, NIC, NPU and Platform CUIDs are derived with the node key for root when
amdgpu holds one (read from any `cuid_seed`) and are temporary otherwise.

| Axis | Cases to distinguish |
|---|---|
| Privilege | Root and ordinary user; primary identity is privileged |
| Driver support | CUID attributes present and absent |
| Node key | None (`cuid_seed` and `cuid_derived` fail with `ENODATA`), set by an administrator, no amdgpu |
| Firmware identity | Adopted system UUID and constructed Platform identity |
| Hardware | Whole GPU, no readable serial, spatial partition, SR-IOV VF |

A constructed primary encodes component identity. An adopted firmware primary
is opaque and need not be UUIDv8. Canonical derived CUIDs depend on the node
key; auxiliary values use the machine-id application key (K_app) and do not
change on node re-key. Do not decode adopted UUIDs as constructed payloads or
expect every derived value to move after provisioning.

## Builds

Run from the repository root. Use an unused build directory; an installed
`/opt/rocm` package can shadow in-tree work.

```sh
W=$(mktemp -d)
cmake -S shared/cuid -B "$W/build-cuid" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON \
  -DBUILD_EXAMPLES=ON \
  -DCMAKE_INSTALL_PREFIX="$W/install-cuid"
cmake --build "$W/build-cuid" --parallel 8

cmake -S projects/amdsmi -B "$W/build-amdsmi" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_CUID=ON -DBUILD_TESTS=ON \
  -DBUILD_SHARED_LIBS=ON -DENABLE_ESMI_LIB=OFF \
  -DBUILD_CLI=OFF -DBUILD_WRAPPER=OFF \
  -DCMAKE_DISABLE_FIND_PACKAGE_amdcuid=ON \
  -DCMAKE_INSTALL_PREFIX="$W/install-amdsmi"
cmake --build "$W/build-amdsmi" --parallel 8
```

Disabling package discovery forces the in-tree CUID build. A CLI/package build
needs its corresponding options.

Two additional build configurations need separate checks:

* **Installed CUID:** install the library component to a private prefix, then
  configure amd-smi with `AMDCUID_HINT_DIR` pointing there and without disabling
  package discovery. Confirm the configure log selected that package.
* **CUID disabled:** configure with `BUILD_CUID=OFF`. Confirm the same exported
  amd-smi ABI and `NOT_SUPPORTED` from the CUID entry points.

`BUILD_CUID=AUTO` uses an installed package when found, otherwise the sibling
source tree. `ON` fails if neither is available; `AUTO` warns and disables CUID.
`AMDSMI_CUID_TEST_SYSFS_OVERRIDE=ON` enables an amd-smi provenance test seam;
keep it out of shipped builds. It does not relocate libamdcuid's hardware reads.

## Running tests safely

Set `AMDCUID_VECTORS_PATH` to this checkout's
`shared/cuid/tests/vectors/cuid_vectors.txt` when comparing separately built
binaries.

```sh
export AMDCUID_VECTORS_PATH="$PWD/shared/cuid/tests/vectors/cuid_vectors.txt"
"$W/build-cuid/tests/amdcuid_test"
ctest --test-dir "$W/build-cuid" --output-on-failure -V
"$W/build-amdsmi/tests/amd_smi_test/amdsmitst" \
  --gtest_filter='*Cuid*:*CUID*:*cuid*'
python3 -m pytest projects/amdsmi/tests/python/unit/gpu/test_cli_cuid_*.py -q
```

**Setting the key re-keys the host.** A `cuid_seed` write replaces the key
amdgpu holds and every derived CUID on the host. On the host, only
`cuidtstPrivileged.HMAC` sets a key and amd-smi's
`GpuUnit.CuidSetSeedRefreshFailureWithTheKeyAlreadyInPlace` writes the key in
use back, both only with `AMDCUID_TEST_ALLOW_SET_KEY=1`; run them only on a
host whose key may change.
`cuid_gpu_paths` also sets keys, but only into its fixture `/sys` inside a
private mount namespace.

`AMDCUID_BUILD_LIFECYCLE_TESTS=ON` adds `cuid_gpu_paths`, which runs GPU
discovery against a synthetic `/sys` in a private mount namespace as root. Its
build directory must be outside `/tmp`, which the namespace replaces. Run it
with `sudo ctest --test-dir <build> -R cuid_gpu_paths --output-on-failure`.
Besides GPU path resolution it covers node-key loading from `cuid_seed` (a
key, `ENODATA`, a short read, `EPERM` for root without `CAP_SYS_ADMIN`, and a
`cuid_seed` on a device bound to another driver, which is ignored),
`amdcuid_set_hash_key()` through `cuid_seed`, its `UNSUPPORTED` without
amdgpu and its `FILE_ERROR` when reading the new key back fails, no copy of
the key left in a process after it drops root, after fingerprinting it, or in
a child forked during a call or an HMAC, no key held where key storage cannot be protected from fork
and core dumps while temporary CUIDs still work there, `cuidtstPrivileged.HMAC` setting a key only with
`AMDCUID_TEST_ALLOW_SET_KEY=1`, `ENODATA` from `cuid_derived` on whole GPUs
and partitions, a non-AMD display device, and a whole GPU before and after the
driver publishes `cuid_derived`. The shim stands for `ENODATA` with an empty
`cuid_seed` or `cuid_derived`, and fails a `cuid_seed` read with `EPERM`
without `CAP_SYS_ADMIN`, as amdgpu does. A fixture that
opts in also discovers a CPU (from the host's `/proc/cpuinfo`), a NIC and the
Platform against synthetic SMBIOS and PCI data.

For tier 2 use a host without CUID support, or a separately authorized
maintenance run with `amdgpu.cuid=0` set at module load. Verify attributes are
absent. Loading/unloading a GPU driver is not a prerequisite
for the fixture tests and must not be done on a host with a pending partition
change without a hardware-specific plan.

## Assertions and gaps

* `ConformanceVectors` consumes all 39 rows and records
  `conformance_vectors=39` in GTest XML. Run
  `python3 shared/cuid/tests/vectors/cuid_vectors.py --check` too. That checks
  the local table against its generator. This table is the source of truth;
  the kernel's KUnit test asserts a subset of its rows, and nothing checks that
  subset against it automatically. Table SHA-256:
  `b339274b8be91a546c474391d72c41e695f4b522c24ca6e0a56b141e377198c4`.
* UnitID is zero for a whole component. A GPU partition's is its XCC range,
  `(xcc_count << 6) | first_xcc`, published by the driver; an SR-IOV VF's is
  its one-based index. Values above `0x1fff` are refused, not masked. The suite checks unchanged output on refusal
  and propagation through GPU reconstruction.
* Compare every driver-published GPU derived CUID against both libamdcuid and
  amd-smi. A partition must report its own driver value or no value, never its
  parent's, except that amd-smi reports the whole GPU's for the one partition
  of a GPU in SPX that has none. The MI350X release check below covers this on
  hardware; SR-IOV VF guests have not been run.
* Handle tests require success for discovered devices and reject absent ones.
  `ColdLookupEnvironment` starts before the test fixtures enumerate, but later
  calls in that environment share the manager warmed by the first lookup; it
  does not independently cold-start each API.
* CLI checks: seed metadata once per invocation; derived/primary/type/auxiliary/
  source fields per GPU; primary only when explicitly requested and permitted;
  consistent JSON/CSV field names. Python seed tests simulate provisioning and
  do not establish real driver behavior.
* The library reads `cuid_seed` end to end only from a fixture in
  `cuid_gpu_paths`. Only the MI350X release check reads a key amdgpu holds.
* GPU discovery lists AMD GPUs (Vendor ID `0x1002`) only. `cuid_gpu_paths`
  checks this with a synthetic `0x1a03` display device; on a host with a BMC
  display device, `amdcuid_get_all_handles()` must not return it either.

## Release check on Instinct MI350X

Run the in-memory key kernel and the library and amd-smi from the same tree,
on a host whose key may change. Record the host's state first and restore it
at the end: reload amdgpu, or reboot, to drop the test key.

1. Kernel: after amdgpu loads, `cuid_seed` and every `cuid_derived` fail with
   `ENODATA`, and every GPU and partition publishes its other `cuid_*`
   attributes with the documented modes. `amd-smi set --cuid-seed` with a
   test key: every `cuid_derived` recomputes from the key, and `cuid_seed`
   returns it on every GPU. A 31-octet write fails with `EINVAL`.
2. Cycle one GPU through SPX, DPX, QPX and CPX with the key set: every
   partition derives with it and the partition CUIDs are pairwise distinct.
3. Unbind and rebind one GPU: the key and its derived CUIDs are unchanged.
   Reload amdgpu, then reboot: each time `cuid_seed` fails with `ENODATA`
   again.
4. The library suite, `cuid_gpu_paths` as root, the amd-smi CUID gtests and
   Python tests, and `amd-smi static --cuid`, `amd-smi list` and
   `amd-smi node --cuid` as root and as an unprivileged user, before and
   after the key is set.

## Packaging checks still required for release

Check artifacts from each intended channel: amd-smi deb/rpm, its manylinux
wheel, and TheRock packages/wheels. A successful configure or exported CUID
symbol alone does not establish compiled-in support.

* Confirm configure selected in-tree CUID or the intended installed package.
* Load the artifact's library, not a system copy, and check CUID entry points.
* For the in-tree build, confirm CUID objects are absorbed without installing a
  separate amdcuid header, archive or CMake package.
* Link a consumer of `amd_smi_static` through `find_package(amd_smi CONFIG)`.
  An installed-CUID build must resolve its exported amdcuid/Threads dependencies;
  an in-tree build must not require an external amdcuid package.
* Compare exported symbols with `BUILD_CUID=OFF` and keep the test-only sysfs
  override disabled in release artifacts.

Report actual pass/skip/fail counts, selected build paths and untested matrix
cells.
