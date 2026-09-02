# CUID QA plan — rocm-systems side

Scope: `projects/cuid` (libamdcuid, `amdcuid_tool`) and `projects/amdsmi`
(the CUID C API, the Python binding, the `amd-smi` CLI), as they stand on
`users/dgalants/cuid-aux-vectors` (PR 10818) over
`users/dgalants/sha256-shared` (PR 10719).

Out of scope here: the `amdgpu` sysfs interface itself and its selftests.
They live in the kernel tree and are covered by
`openspec/changes/add-cuid-kernel-interface`. The kernel is present in this
plan only as an *input* — the values it publishes are something this side has
to agree with.

Everything in "Findings" was observed on real hardware while writing this
plan; each says how to reproduce it.

---

## 0. Tiers

Internal shorthand, used throughout this plan and in `openspec/README.md`.

* **Tier 1** — the `amdgpu` driver publishes the CUID attributes. The kernel is
  the producer; the library and `amd-smi` report its value verbatim.
* **Tier 2** — the driver publishes nothing and `libamdcuid` is the only
  producer. It reads the record store, and computes where the store is empty.

**Tier 2 is what ships first and is what to test first.** CUID reaches
customers inside AMD SMI a release before the kernel series lands, so every
node is tier 2 for that release and stays tier 2 while it runs an older kernel.

The reason this matters for test design and not just for vocabulary: on a
tier-1 node the driver's value stands in front of every code path tier 2
depends on, so a tier-1 result says nothing about tier 2. Four defects have now
been found that were invisible in tier 1 — §5.2, §5.9, and the two tier-2 ones
in §5.11.

## 1. What a CUID is, in the two sentences QA needs

A CUID is a 128-bit UUID naming one component. The **primary** encodes the
component's real identity (serial, IDs, type) and is root-only because the
serial is in it; the **derived** (secondary) is an HMAC of the primary under a
node-wide seed, is safe to hand to unprivileged software, and changes if and
only if the seed changes.

Three producers must agree on the value for one component: the `amdgpu`
driver, `libamdcuid`, and `amd-smi` reporting through the library.
**Disagreement between producers is the defect class this whole feature exists
to prevent, and is the thing QA should be hunting.**

---

## 2. Environment matrix

A run is characterised by five axes. Most defects found so far were only
visible in a particular cell, so pick cells deliberately rather than testing
one machine hard.

| Axis | Values | Why it changes behaviour |
|---|---|---|
| Privilege | root / unprivileged | The primary CUID is root-gated at the source, and resolving a single device makes the library discover it, which reads PCIe configuration space. Unprivileged callers take a different code path. |
| Kernel CUID | `cuid_primary`/`cuid_secondary`/`cuid_seed` present / absent | Present means the driver is a producer and its value must match the library's. Absent means the library is the only producer. **Absent is the shipping order**: CUID reaches customers in AMD SMI first and in the kernel a release later, so absent is the majority case at launch, not the exotic one. |
| Record store | `/var/lib/amdcuid` populated / absent | Absent is a node nobody has run `amdcuid_tool --generate-cuid` on, i.e. a fresh install. With the driver *also* silent this is the only configuration in which the library has to derive rather than read, and it is where both of the defects in §5.2 and §5.9 lived. |
| Node seed | provisioned / not | Unprovisioned uses the public `AMD-CUID-DEFAULT-SEED-v1`, so every derived CUID is reproducible by anyone. Provisioned changes every derived value on the node. |
| Firmware identity | SMBIOS system UUID present / absent | Present means the Platform CUID is **adopted** — a firmware UUID verbatim, version 1/3/4, with no CUID payload to decode. Absent means it is constructed, v8, decodable. Three tests failed on this axis alone (§5.1). |
| Hardware | discrete GPU with a PCIe DSN / GPU without / partitioned (MI300X, MI350X in CPX) / SR-IOV VF / BMC display controller | Selects primary-source stage, and the partitioned and VF cases have unresolved specification questions (§6). |

**The cell to test first** is kernel CUID *absent* and record store *absent*,
because it is what a customer gets in the release AMD SMI ships CUID in, and
because it is the only cell where the library derives rather than reads. Every
other cell has the driver or the store standing in front of the derivation, so
a defect underneath is invisible. Two were.

Minimum useful set for a first QA pass:

0. **A node with no kernel CUID support and no `/var/lib/amdcuid`.** Root and
   unprivileged. See §4.7 for how to produce one without waiting for a
   pre-CUID kernel build.
2. The same node after `amd-smi set --cuid-seed`.
3. A node with **no** kernel CUID support (stock `amdgpu`), to exercise the
   library-only path.
4. A partitioned MI300X/MI350X in CPX, which is the only way to cover
   `CuidIsNotReportedForASecondaryPartition`.
5. A build with `-DBUILD_CUID=OFF`, to confirm the ABI is unchanged and every
   entry point returns `AMDSMI_STATUS_NOT_SUPPORTED`.

---

## 3. Building what is under test

The two projects are separate CMake projects, and there are **two ways
amd-smi gets CUID**. Test both; they produce different artifacts.

* **In-tree (the default from a monorepo checkout, and what every packaging
  channel now uses).** `find_package(amdcuid CONFIG)` finds nothing,
  amd-smi builds `projects/cuid` from the sibling directory and absorbs the
  objects. Nothing named amdcuid is installed, and `libamd_smi_static.a` is
  self-contained.
* **Against an installed package.** `libamdcuid` was built and installed first,
  and is on `CMAKE_PREFIX_PATH` or `AMDCUID_HINT_DIR`. It wins over the
  sibling. `amd_smi_static` then names `amdcuid::amdcuid` in its exported
  interface and `amd_smi-config.cmake` resolves it with `find_dependency`.

Which one a build took is in the configure output; check it rather than assume.

```bash
SRC=$PWD                      # rocm-systems checkout
W=/var/tmp/cuidqa

# 1. libamdcuid + its tests
cmake -S $SRC/projects/cuid -B $W/build-cuid \
      -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON \
      -DCMAKE_INSTALL_PREFIX=$W/stage-cuid
cmake --build $W/build-cuid -j"$(nproc)"
cmake --install $W/build-cuid

# 2. amd-smi against it. BUILD_CUID=ON, not AUTO: AUTO silently ships a
#    library without CUID if the package is not found, which is exactly the
#    thing QA must not let through.
cmake -S $SRC/projects/amdsmi -B $W/build-amdsmi \
      -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON \
      -DBUILD_CUID=ON -DAMDCUID_HINT_DIR=$W/stage-cuid \
      -DAMDSMI_CUID_TEST_SYSFS_OVERRIDE=ON \
      -DCMAKE_INSTALL_PREFIX=$W/stage-amdsmi
cmake --build $W/build-amdsmi -j"$(nproc)"
cmake --install $W/build-amdsmi

# 3. The in-tree build, which is what the packaging channels do. Same command
#    with nowhere to find an installed package.
cmake -S $SRC/projects/amdsmi -B $W/build-intree \
      -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON \
      -DBUILD_CUID=ON -DAMDCUID_HINT_DIR=/nonexistent \
      -DCMAKE_INSTALL_PREFIX=$W/stage-intree
cmake --build $W/build-intree -j"$(nproc)"
cmake --install $W/build-intree
```

`AMDSMI_CUID_TEST_SYSFS_OVERRIDE=ON` is required for
`CuidSourceIsDriverWhenTheAttributeIsPublished` to run at all; it makes the
library honour `AMDSMI_CUID_SYSFS_ROOT`, which lets an unprivileged user forge
the `source` field, so **it must never appear in a shipped build**. Confirm
its absence from any release configuration as part of sign-off.

Configure must print exactly one of:

```
-- CUID support: enabled (amdcuid 2.0.0 at .../lib/cmake/amdcuid)
-- CUID support: enabled (built from .../projects/cuid, absorbed into the library)
-- CUID support: disabled (BUILD_CUID=OFF)
CMake Warning ... CUID support: disabled -- BUILD_CUID=AUTO and no amdcuid ...
```

A packaging job that prints the third line has produced an artifact without
CUID. Treat it as a build failure.

---

## 4. The suites, what they cover, and how to run them

### 4.1 `amdcuid_test` — the library

```bash
# Unprivileged. Redirects its record store to a temporary directory
# automatically (see the constructor in tests/test_common.cc).
$W/build-cuid/tests/amdcuid_test

# Privileged. Root deliberately uses the REAL /etc/amdcuid and
# /var/lib/amdcuid, because the privileged tests exist to exercise the real
# path. Isolate them, or the run rewrites the node's key store:
sudo unshare -m bash -c '
  mount -t tmpfs none /etc/amdcuid &&
  mount -t tmpfs none /var/lib/amdcuid &&
  '"$W"'/build-cuid/tests/amdcuid_test'
```

**The isolation is not optional and it is not complete.** It covers the two
directories the library writes. It does **not** cover sysfs: a privileged run
that provisions a seed pushes that seed into every GPU's `cuid_seed`, and the
kernel offers no way to un-provision one. See §5.5.

Expected on a two-GPU node with kernel CUID present, SMBIOS UUID present:

| | passed | skipped | failed |
|---|---|---|---|
| root | 50 | 1 (`GimDeviceEnumeration`, no GIM node) | 0 |
| unprivileged | 40 | 11 (the whole `cuidtstPrivileged` suite) | 0 |

CTest registers the two halves separately:

```bash
ctest --test-dir $W/build-cuid -R cuid_unprivileged_tests
ctest --test-dir $W/build-cuid -R cuid_privileged_tests
```

Note that `cuid_privileged_tests` **passes as an ordinary user** by skipping
every case in it. A green CTest run says nothing about the privileged half
unless it was run as root. Read the skip count, not the exit status.

### 4.2 `amdsmitst --gtest_filter='*Cuid*'` — the amd-smi C API

```bash
$W/build-amdsmi/tests/amd_smi_test/amdsmitst --gtest_filter='*Cuid*'
sudo unshare -m bash -c 'mount -t tmpfs none /etc/amdcuid &&
  mount -t tmpfs none /var/lib/amdcuid &&
  '"$W"'/build-amdsmi/tests/amd_smi_test/amdsmitst --gtest_filter="*Cuid*"'
```

Expected on the same node: root 11 passed / 4 skipped, unprivileged 9 passed /
6 skipped, no failures.

The skips are informative and should be read every time:

| Skip | What it means |
|---|---|
| `CuidEntryPointsNotSupportedWithoutTheLibrary` | This build has CUID. Run the `BUILD_CUID=OFF` configuration to cover it. |
| `CuidIsNotReportedForASecondaryPartition` | No partitioned GPU. Needs an MI300X/MI350X in CPX. **Currently uncovered anywhere.** |
| `CuidSourceIsDriverWhenTheAttributeIsPublished` | Either every device publishes `cuid_secondary`, or the build lacks `AMDSMI_CUID_TEST_SYSFS_OVERRIDE`. Needs a node *without* kernel CUID to be meaningful. |
| `CuidDerivedValuesDoNotCollideAcrossProcessors` | Fewer than two GPUs answered. **This is the collision check** — do not sign off a run where it skipped. |
| `CuidDriverPublishedValueIsUsedVerbatim` | amd-smi reported no CUID at all. Expected unprivileged today (§5.4); a skip as root is a failure to investigate. |

### 4.3 `test_cli_cuid_seed.py` — the CLI seed surface

Pure Python, no hardware, no root:

```bash
cd $SRC/projects/amdsmi && python3 -m pytest tests/python/unit/gpu/test_cli_cuid_seed.py -q
```

### 4.4 End to end through the CLI

The check that matters most, because it is what a customer runs:

```bash
sudo LD_LIBRARY_PATH=$W/stage-amdsmi/lib \
     PYTHONPATH=$W/stage-amdsmi/share/amd_smi \
     python3 $W/stage-amdsmi/libexec/amdsmi_cli/amdsmi_cli.py static --cuid
```

Assert all of:

1. The seed block (`SEED_PROVISIONED`, `SEED_FINGERPRINT`) appears **once**,
   before the per-GPU blocks, not inside each of them.
2. One CUID block per GPU, each with `DERIVED_CUID`, `PRIMARY_CUID`,
   `COMPONENT_TYPE`, `AUXILIARY`, `SOURCE`.
3. `PRIMARY_CUID` is `N/A (not requested)` without `--cuid-primary`, and a
   v8 UUID with it under root, and empty with it as an ordinary user.
4. `--cuid-primary` alone implies `--cuid`.
5. **Cross-check against the kernel.** For each GPU:
   ```bash
   cat /sys/bus/pci/devices/<bdf>/cuid_secondary
   ```
   must equal that GPU's `DERIVED_CUID`. This is the producer-agreement check,
   it is one command, and it catches the entire class of defect the feature is
   about. Do it on every run.
6. `--json` and `--csv` carry the same seven field names as the screen output:
   `derived_cuid`, `primary_cuid`, `component_type`, `auxiliary`, `source`,
   `seed_provisioned`, `seed_fingerprint`.

### 4.5 The library CLI

```bash
sudo $W/stage-cuid/bin/amdcuid_tool --list --show-primary
```

Cross-check its GPU entries against `amd-smi static --cuid`: same BDF, same
derived CUID. Note that `amdcuid_tool` also reports the Platform, the CPU and
the NICs, which `amd-smi` never will — that is expected, not a discrepancy.

### 4.6 Conformance vectors

`cuid_vectors.txt` is shared byte-for-byte with the kernel tree. The library
asserts every vector in `cuidtstUnprivileged.ConformanceVectors`, and the
build fails if the table has drifted from its generator:

```bash
python3 $SRC/projects/cuid/tests/vectors/cuid_vectors.py --check
```

Be precise about what that buys: it compares the table against the generator
*sitting beside it*. Nothing in this repository notices if the kernel tree's
copy of the generator has diverged. Comparing the two trees is
`check_vectors_drift.py`, which runs on the kernel side only.

### 4.7 Producing a driver with no CUID support

The cell that matters most needs a driver that publishes no CUID attributes.
Three ways, cheapest first.

**Boot a kernel from before the series.** A stock distro kernel has no CUID.
Correct, and costs a reboot.

**Rebuild the module without the registration call.** No reboot, and it keeps
everything else about the running kernel identical, so a difference in results
is a difference in CUID and nothing else. In the kernel tree that
`/lib/modules/$(uname -r)/build` points at:

```bash
# One line: do not register the attributes.
sed -i 's|ret = cuid_sysfs_init(&adev->cuid, adev->dev, &ident);|ret = 0; (void)\&ident;|' \
    drivers/gpu/drm/amd/amdgpu/amdgpu_device.c
make -j"$(nproc)" modules            # incremental on a built tree

# Strip debug info and BTF, or the running kernel rejects the module with
# "failed to validate module [amdgpu] BTF: -22".
cp drivers/gpu/drm/amd/amdgpu/amdgpu.ko /var/tmp/amdgpu-nocuid.ko
strip --strip-debug /var/tmp/amdgpu-nocuid.ko
objcopy --remove-section=.BTF --remove-section=.BTF_ids /var/tmp/amdgpu-nocuid.ko

M=/lib/modules/$(uname -r)/kernel/drivers/gpu/drm/amd/amdgpu/amdgpu.ko
sudo cp "$M" /var/tmp/amdgpu-with-cuid.ko          # keep the original
sudo modprobe -r amdgpu
sudo cp /var/tmp/amdgpu-nocuid.ko "$M" && sudo depmod -a && sudo modprobe amdgpu
git checkout drivers/gpu/drm/amd/amdgpu/amdgpu_device.c
```

`insmod` on the built `.ko` directly does not work: unloading amdgpu takes its
dependency modules with it and `insmod` resolves none of them. Install into
`/lib/modules` and use `modprobe`.

Confirm, and this is the check that says the cell is really the cell:

```bash
ls /sys/bus/pci/devices/<bdf>/ | grep -c '^cuid_'    # must be 0
```

Restore by copying `amdgpu-with-cuid.ko` back, `depmod -a`, `modprobe amdgpu`,
and confirming the count is 4 again.

**Fabricate the sysfs root.** `AMDSMI_CUID_SYSFS_ROOT`, in a build configured
with `-DAMDSMI_CUID_TEST_SYSFS_OVERRIDE=ON`, relocates the root *amd-smi*
consults. It covers amd-smi's `source` field and nothing else: `libamdcuid`
reads the real `/sys` regardless, so this does not produce the cell above.

And empty the store, or the driver is not the only thing standing in front of
the derivation:

```bash
sudo rm -rf /var/lib/amdcuid
```

### 4.8 Packaging

See §7. This is the part most likely to be wrong and least likely to be
tested, because every defect in it is silent.

---

## 5. Findings from this pass

Reproduced on a two-W6800 node with the CUID kernel patches loaded, SMBIOS
UUID present, seed initially unprovisioned.

### 5.1 Three privileged tests failed on any machine with a firmware UUID — fixed

`ReverseSerialNumber`, `ReverseVendorId` and `ReverseDeviceType` decoded
VendorID, serial and Component Type straight out of the Platform's primary
CUID. Where firmware supplies a system UUID that primary *is* that UUID,
version 1/3/4, with no CUID payload, so the decode returned whatever the
firmware bytes contained — Component Type `0xa` ("Rack") for a Platform.

3 of 10 privileged tests failed on correct data. Fixed by checking the version
nibble before reading any field, which is the rule `cuid_util.h` already
states.

### 5.2 `amd-smi` reported no CUID for any GPU — fixed, and the highest-severity item here

`amdsmi_get_gpu_cuid_info()` returned `AMDSMI_STATUS_NOT_SUPPORTED` for both
GPUs, as root, on a node where the kernel was publishing `cuid_secondary` for
each of them. `amd-smi static --cuid` therefore printed nothing.

Cause: `discover_device_by_path()` resolved every path through
`real_dev_path_from_fd()`, which reads `st_rdev` — zero for a directory — so
every *sysfs* path failed. `bdf_to_device_path()` returns sysfs, and amd-smi
resolves each GPU by BDF and makes no other call first.

Reproduce on an unfixed tree:

```c
amdcuid_id_t h;
amdcuid_get_handle_by_bdf("0000:63:00.0", AMDCUID_DEVICE_TYPE_GPU, &h);
/* DEVICE_NOT_FOUND -- but SUCCESS if amdcuid_get_all_handles() ran first */
```

That "if" is why it survived: every existing test calls
`TestBase::SetUp()` first, which enumerates, and the lookups then answer out
of the device-manager cache without reaching discovery.

### 5.3 The four handle-lookup tests could not fail — fixed

They named a BDF and a device path literally (`0000:03:00.0`,
`/dev/dri/renderD128`) and accepted `DEVICE_NOT_FOUND` as a pass:

```c
if (status == AMDCUID_STATUS_SUCCESS) { ...checks... }
else { EXPECT_EQ(status, AMDCUID_STATUS_DEVICE_NOT_FOUND); }
```

`DEVICE_NOT_FOUND` is the failure mode of the code under test. They were green
throughout 5.2. They now enumerate the machine, require success for a device
that is present, assert the identifier matches enumeration, and include a
negative case. A new `ColdLookupEnvironment` records what each lookup answers
before the first test runs — the only moment the library is cold — and
`cuidtstPrivileged.ColdHandleLookup` asserts those.

**QA lesson worth carrying:** treat any `if (ok) {...} else { expect(error); }`
shape in a test as untested code.

### 5.4 An unprivileged `amd-smi` still reports nothing — open

The kernel publishes `cuid_secondary` world-readable. `libamdcuid` resolves a
single device by discovering it, discovery reads PCIe configuration space, and
an unprivileged process is routed to the daemon instead — which is not running
on a normal node. So an ordinary user gets nothing for a GPU whose derived
CUID is sitting in sysfs a few bytes away.

The staged lookup is specified as driver → store → library. The driver stage
is consulted for the `source` *label* but never as a *value* source when the
library cannot answer.

Suggested fix: where `cuid_handle_for()` fails, read
`/sys/bus/pci/devices/<bdf>/cuid_secondary`, report it with
`AMDSMI_CUID_SOURCE_DRIVER`, and derive `auxiliary` from bit 117 of the value.
That strictly widens the set of cases that answer and changes no answer that
already works. It needs a spec delta against
`changes/integrate-cuid-into-amdsmi`, and it interacts with 5.5 — so it should
land with a mismatch diagnostic, not without one.

Until then, document that `amd-smi static --cuid` requires root.

### 5.5 A provisioned kernel seed cannot be un-provisioned — open

`cuid_seed_store()` accepts exactly 32 bytes and sets `seed_len`; there is no
encoding for "back to the public default", and `cuid_seed_show()` returns the
24-byte default only while `seed_len` is zero. So:

* a seed provisioned by mistake can be replaced but not withdrawn, other than
  by unbinding or reloading the driver;
* and a driver reload *does* withdraw it, silently, while the library's key
  store still holds the secret and `amd-smi` still reports
  `SEED_PROVISIONED: True` with its fingerprint.

Both directions were observed on this node. Any privileged test run that
provisions a seed leaves the machine changed in a way `unshare -m` does not
contain, because the seed goes to sysfs.

**QA procedure:** before any provisioning test, record
`cat /sys/bus/pci/devices/<bdf>/cuid_secondary` for every GPU; afterwards,
either restore by re-provisioning the original key or reload `amdgpu`; and
state in the report which was done.

### 5.6 Discovery scope is unstated — open, low severity

`libamdcuid` enumerates every DRM card node as a GPU regardless of vendor. On
this node that included the ASPEED BMC display controller (`1a03:2000`, driver
`ast`), which got a GPU CUID of its own. Nothing in the library filters by
vendor and nothing in the specification says whether it should.

Not obviously wrong — CUID has component types for storage, memory and generic
PCIe, so a whole-platform inventory may be the intent — but it means
`amdcuid_tool --list` and `amd-smi static --cuid` report different device sets
on the same machine, and QA should expect that rather than file it.

It also has a testable consequence: that device has no unprivileged
fingerprint source, so it appears in a root enumeration and not in an
unprivileged one.

### 5.7 CI could not fail on CUID, twice over — fixed

Two independent reasons the defects in 5.1 to 5.3 reached `develop` with CI
green:

* `cuid-workflow.yml`'s nine-distro `build-and-test` job was
  `continue-on-error: true`. A build failure or a test failure on any distro,
  or on all nine, reported green. That is not a gate. It is now advisory on
  eight and required on Ubuntu22 — every distro still runs and still uploads
  its log; one of them can now block a merge. Made conditional rather than
  removed because eight of these are self-hosted GPU containers whose
  flakiness is a separate problem, and requiring all nine would swap "no gate"
  for "a gate nobody can keep green".
* No amd-smi workflow triggered on `projects/cuid/**`. Now that libamdcuid is
  compiled into `libamd_smi`, a change to the library changes the amd-smi
  binary, the wheel and every package, and none of them were being built.
  `amdsmi-build.yml` and `amdsmi-manylinux-build.yml` now trigger on
  `projects/cuid/**` and `shared/sha256/**`.

TheRock's matrix had a third variant of the same hole: `projects/cuid` mapped
to a project group that does not exist, and the lookup skips an unknown group
silently, so a change to the library selected no build options and no tests.

**Two things follow for QA.** A green CI run on this feature has, until now,
meant less than it looks like; and the state of these gates is worth checking
as part of sign-off, not assumed.

### 5.9 With no kernel CUID and no record store, amd-smi reported nothing — fixed, and the one QA should care about most

Measured on the same two W6800s with the driver rebuilt without
`cuid_sysfs_init()` and `/var/lib/amdcuid` removed — i.e. the configuration
this feature ships into first:

* `amdsmi_get_gpu_cuid_info()` returned `AMDSMI_STATUS_API_FAILED` for every
  GPU, as root.
* `amd-smi static --cuid` printed `N/A` for all five fields of every GPU.
* `amdcuid_tool --list` on the same machine, same moment, computed the CUIDs
  without trouble.

Cause: `amdcuid_get_handle_by_bdf()` derives with the node key and returns that
value as the handle; `CuidDeviceManager::add_device()` then indexed the device
by deriving *again with no key*. That call has no HMAC key to use, fails, and
the device lands in the device list but not in the CUID index that
`lookup_by_handle()` searches. So the handle was correct and unusable: every
`amdcuid_query_device_property()` on it answered `DEVICE_NOT_FOUND`, and
amd-smi turns a failed auxiliary-flag query into a failed snapshot by design.

Why nobody saw it: where the driver publishes `cuid_secondary`, and where the
record store already holds an entry, *nothing is derived* — both calls return
the same recorded value and agree by accident. Only the fresh, kernel-less node
derives, and that is the node no one had tested.

**This is the single most important thing to carry into the QA matrix.** Two
separate defects (this and §5.2) lived behind the driver and the store. Any
test run on a node that has either is testing the reading path, not the
computing one.

`cuidtstPrivileged.ColdHandleLookup` now queries a property on the handle the
cold by-BDF lookup returned, at the moment it is returned. Confirmed to fail
against the old code naming both BDFs, and to pass against the new.

### 5.11 Tier 2 answered almost nobody — fixed

Measured on a driver built without `cuid_sysfs_init()`:

| | before | after |
|---|---|---|
| root, no record store | worked, recorded nothing | works, records what it computed |
| unprivileged, no store | `NOT_SUPPORTED` | `NOT_SUPPORTED`, and correctly — deriving reads PCIe configuration space |
| unprivileged, store present | `NO_PERM` | full answer |
| `source` | `UNKNOWN` for every device | `STORE`, or `LIBRARY` before anything is recorded |

So before this, an ordinary user could not read a CUID on a tier-2 node at all,
and root recomputed from configuration space on every call. Two causes:

* `is_temporary_cuid()` went through the CAP_SYS_ADMIN-gated primary, although
  the auxiliary marker is bit 117 of the derived value as well — which is what
  that bit is for. amd-smi fails the whole snapshot when the flag cannot be
  established, deliberately, so this one failure took the entire block with it.
* The by-name lookups had their own single-device discovery that neither
  indexed the device nor wrote the store. They now fall back to the same
  enumeration everything else uses, keeping by-name discovery only for what
  enumeration does not reach.

Nothing else populates the store any more: `libamdcuid` ships inside AMD SMI
and has no package of its own left to run a post-install step. **So on a tier-2
node, one privileged call is the precondition for every unprivileged one.** Put
`sudo amd-smi static --cuid` — or `amdcuid_tool --generate-cuid` where the tool
is available — in the setup for any unprivileged tier-2 case, and check that it
is not needed twice.

### 5.12 Packaging — see §7

Three defects, all silent, all fixed in this branch.

---

## 6. Questions the specification has not answered

These are not test failures; they are places where two documents or two layers
disagree and QA cannot decide which is right. They are recorded in
`openspec/CONFLICTS.md`.

* **O1 — UnitID for an SR-IOV Virtual Function.** The baseline blesses a
  non-zero UnitID for a VF, a delta forbids anything positional, the library
  assigns the 1-based VF index, and the kernel hardcodes 0. So a VF's
  driver-published primary and the library's computed primary for the same VF
  **do not agree**, by construction. Any VF testing will hit this. Do not file
  it as a bug; it needs a decision.
* **O3 — a driver reload changes every derived CUID on the node, silently.**
  One `modprobe -r amdgpu && modprobe amdgpu` moved both GPUs' derived CUIDs on
  the test node while `amd-smi` went on reporting `SEED_PROVISIONED: True` with
  the provisioned seed's fingerprint. The kernel seed is per-device and lost on
  reload; the library's key store is not, and nothing compares them. A driver
  package update does exactly this. **Expect it during testing and record the
  values rather than filing it** — the numbers are in `CONFLICTS.md`.
* **UnitID overflow.** Both producers do `(unit_id >> 8) & 0x1F` on a 16-bit
  value, so a UnitID of 8192 packs identically to 0 — a partition colliding
  with its whole device, silently. Unreachable while the kernel hardcodes 0.
* **Partitioned GPUs.** `amdsmi_get_gpu_cuid_info()` returns `NOT_SUPPORTED`
  for any partition other than partition 0, because the kernel hardcodes the
  partition field to zero and reporting the physical device's value would give
  every partition the same identity. That is a deliberate choice, not a gap,
  but it means **a partitioned MI300X reports no CUID for most of its
  devices** and QA should confirm that is the intended customer-visible
  behaviour before release.

---

## 7. Packaging verification

CUID reaches a customer only if it survives the packaging path, and every
failure in that path is silent: the library still exports all four
`amdsmi_*cuid*` symbols, they just return `NOT_SUPPORTED`.

### 7.1 The one-command check

For any built or installed `libamd_smi.so`:

```bash
python3 - <<'PY'
import ctypes
lib = ctypes.CDLL("libamd_smi.so")
lib.amdsmi_init(0)
info = (ctypes.c_ubyte * 512)()
print("cuid built in:", lib.amdsmi_get_cuid_seed_info(ctypes.byref(info)) != 2)
PY
```

`AMDSMI_STATUS_NOT_SUPPORTED` is 2. A build with CUID answers 0, or 10
(`NO_PERM`) for an unprivileged caller on a node whose key store exists; a
build without it answers 2 unconditionally, at any privilege. Run this against
the artifact from every channel. It was run against all five listed in §7.4.

### 7.2 The channels, and what to check in each

Every channel now gets CUID the same way — amd-smi builds `projects/cuid` from
the sibling in the same checkout — so there is nothing per-channel to arrange.
What is left is confirming that it happened.

| Channel | Built by | What to verify |
|---|---|---|
| `amd-smi-lib` deb/rpm | amd-smi's own CMake + CPack | configure said "enabled"; §7.1 against the installed `.so`; the package contains **no** `libamdcuid.a`, `amd_cuid.h` or `lib/cmake/amdcuid` |
| `amdsmi` PyPI wheel | `tools/build_wheel.py` in a manylinux container | §7.1 against the bundled `libamd_smi_python.so`. The container has no `/opt/rocm`, so this is the case the in-tree build exists for. |
| `rocm-sdk-core` (TheRock) | TheRock subproject | §7.1 against the wheel's `.so`. TheRock passes only `CMAKE_VERBOSE_MAKEFILE` and `BUILD_TESTS` to amd-smi and registers no amdcuid subproject, artifact or feature — which is fine now, and is why it was not fine before. |
| `amdrocm-amdsmi` (TheRock) | same artifact | as above, plus: the `dev` component must not have acquired an `amdcuid` header or CMake directory |

Two things worth checking once per release rather than per channel:

* `nm libamd_smi_static.a | grep -c ' T amdcuid'` is non-zero on an in-tree
  build (the objects are in the archive) and the archive links with no
  `amdcuid` package anywhere on the prefix path.
* On a build that *did* find an installed package, `amd_smi-config.cmake`
  contains a live `find_dependency(amdcuid ...)`; on an in-tree build it must
  not, because there is nothing to find.

### 7.3 What was wrong, and is now fixed on this branch

0. **Nothing built libamdcuid in any packaging path.** The deb/rpm build is one
   CMake project and builds no sibling; the wheel is built in a manylinux
   container with no `/opt/rocm`; TheRock registers no amdcuid subproject,
   artifact or feature anywhere in its topology. So three of four channels
   shipped a library whose CUID entry points all answered `NOT_SUPPORTED`,
   silently. amd-smi now builds the sibling and absorbs the objects, which
   needs no change in any channel. On top of that, TheRock's CI matrix mapped
   `projects/cuid` to a project group that does not exist, so a change to the
   library selected no build options and no tests at all.
1. **libamdcuid was not findable.** It installs under `${ROCM_DIR}/core`;
   amd-smi's prefix is `${ROCM_DIR}`. Config mode has no search template with
   an arbitrary intermediate directory, so a prefix of `/opt/rocm` never
   reaches `/opt/rocm/core/lib/cmake/amdcuid`. Verified directly. It worked
   only by accident, through the `/opt/rocm/lib -> /opt/rocm/core-X.Y/lib`
   symlink that a *different* package creates. `AMDCUID_HINT_DIR` now names
   the real prefix.
2. **`option()` froze the answer.** A build directory configured before
   libamdcuid was installed kept CUID off for its whole life, silently.
   `BUILD_CUID` is now `AUTO`/`ON`/`OFF` and the found-ness is recomputed
   every configure.
3. **The dependency leaked into the export unresolved.** `amd_smi_static`
   links `amdcuid::amdcuid` PRIVATE; a private link dependency of a *static*
   library is not absorbed, so CMake wrote
   `$<LINK_ONLY:amdcuid::amdcuid>` into `INTERFACE_LINK_LIBRARIES` and
   `amd_smi-config.cmake` never called `find_dependency`. Any consumer linking
   `amd_smi_static` died at generate time. `amd_smi_static` is built whenever
   `BUILD_TESTS=ON`, which the nine-distro packaging job sets, and the
   standalone example build that runs on each of those distros resolves
   amd-smi through exactly that `find_package`.

### 7.4 What was actually produced and checked

Not a plan; this was run. Node: Ubuntu 24.04, GCC 13, two W6800s, CUID kernel
patches loaded.

| Artifact | How it was built | CUID present | libamdcuid file shipped | `amdcuid_*` symbols exported |
|---|---|---|---|---|
| `amd-smi-lib_27.1.0_amd64.deb` | `cpack -G DEB` on the in-tree build | yes | none | none |
| `amd-smi-lib-27.1.0.x86_64.rpm` | `cpack -G RPM`, same build | yes | none | none |
| `amdsmi-27.1.0+…-py3-none-linux_x86_64.whl` | `tools/build_wheel.py --no-repair` | yes | none | none |
| `stage-intree/lib/libamd_smi.so` | `cmake` with `AMDCUID_HINT_DIR=/nonexistent` | yes | none | none |
| `stage-nocuid2/lib/libamd_smi.so` | `-DBUILD_CUID=OFF` | no, by request | none | none |

The wheel is the one that matters most, because `tools/build_wheel.py` passes
**no CUID option at all** — it sets `BUILD_TESTS`, `ENABLE_ESMI_LIB`,
`BUILD_PYTHON_WHEEL`, `AMDSMI_WHEEL_RELEASE`, `CMAKE_BUILD_TYPE` and
`Python3_EXECUTABLE`, and nothing else. The wheel came out with CUID compiled
in anyway, which is the whole point of building the sibling: no channel has to
learn about a second project.

Also checked:

* **The ABI does not move.** `nm -D --defined-only` on the CUID and the
  `BUILD_CUID=OFF` library produce byte-identical sorted symbol lists. In the
  `OFF` build `amdsmi_get_cuid_seed_info()` and `amdsmi_set_cuid_seed()` both
  return 2 (`NOT_SUPPORTED`) through `ctypes`.
* **`libamd_smi_static.a` is self-contained.** A consumer doing nothing but
  `find_package(amd_smi CONFIG REQUIRED)` and
  `target_link_libraries(app PRIVATE amd_smi_static)` configures *and links*
  against a prefix that contains no amdcuid at all.
* **Both compilers.** `projects/cuid` builds clean with GCC 13, Clang 20 and
  ROCm's AMD clang 23 — the last is the one TheRock compiles subprojects with.
* **The degenerate case is loud.** A sparse checkout of `projects/amdsmi`
  alone, which is the documented contributor workflow, has no sibling: `AUTO`
  warns and names both the package search and `AMDCUID_SOURCE_DIR`, and
  `BUILD_CUID=ON` fails configure naming the same two.

### 7.5 Regression test for 7.3, worth adding to CI

```bash
cmake -S <consumer> -B <build> -DCMAKE_PREFIX_PATH=$W/stage-amdsmi
# where <consumer> is three lines:
#   find_package(amd_smi CONFIG REQUIRED)
#   add_executable(app main.cc)
#   target_link_libraries(app PRIVATE amd_smi_static)
```

It must configure with no `find_package` calls of its own. Before the fix it
failed with `The link interface of target "amd_smi_static" contains:
amdcuid::amdcuid but the target was not found`.

### 7.6 Symbol hygiene — checked, clean

```bash
nm -D --defined-only libamd_smi.so.<ver> | grep -i cuid
```

returns exactly the four public entry points, all in `AMDSMI_1`. libamdcuid's
symbols are hidden by the version script. The static archive is *not* filtered
— `libamd_smi_static.a` carries undefined `amdcuid_*` references — which is
the reason §7.3.3 matters and is expected, not a defect.

---

## 8. Entry and exit criteria

**Entry.** A build that printed `CUID support: enabled` and named where the
library came from. The kernel CUID attributes present on at least one node in
the matrix and absent on at least one, and at least one node with an empty
`/var/lib/amdcuid`.

**Exit.** All of:

* `amdcuid_test` green as root and unprivileged, on every node in §2's
  minimum set, with the skip list reviewed rather than the exit status.
* `amdsmitst --gtest_filter='*Cuid*'` green both ways, with
  `CuidDerivedValuesDoNotCollideAcrossProcessors` **not** skipped on at least
  one node.
* `test_cli_cuid_seed.py` green.
* For every GPU on every node **whose driver publishes CUIDs**:
  `amd-smi static --cuid`'s `DERIVED_CUID` equals
  `/sys/bus/pci/devices/<bdf>/cuid_secondary`.
* On a node with **neither** kernel CUID nor a record store: every GPU reports
  a derived CUID, a component type and an auxiliary flag; `SOURCE` is
  `UNKNOWN`, which is correct — amd-smi can only positively identify the driver
  stage; and two consecutive invocations report identical values. Then run
  `amdcuid_tool --generate-cuid --set-key <the node key>` and confirm the
  values did not move: creating the record store must record what was already
  being computed, not mint something new.
* **The reload check, recorded rather than passed or failed.** On a node whose
  driver does publish CUIDs, note every GPU's `DERIVED_CUID`, run
  `modprobe -r amdgpu && modprobe amdgpu`, and note them again. They will
  change, and the seed fields will not — that is O3 (§6), not a regression, and
  the point of doing it is to have the numbers in the report so the decision
  gets made against evidence. Measured values are in `openspec/CONFLICTS.md`.
* §7.1 answers "built in: True" for the artifact of every packaging channel
  that is supposed to carry CUID.
* A `-DBUILD_CUID=OFF` build in which all four entry points return
  `NOT_SUPPORTED` and the exported symbol set is unchanged.
* Provisioning tested and the node's seed state restored, with the report
  saying which method was used.
* §6's open questions unchanged, or closed by a decision — not by a test.
