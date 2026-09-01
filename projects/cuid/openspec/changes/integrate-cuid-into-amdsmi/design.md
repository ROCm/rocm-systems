## Context

`amd-smi` is the layer an operator touches. Everything below it exists so one
number names one component, and none of it reaches a fleet inventory unless
`amd-smi` reports that number.

- `amdsmi_get_gpu_device_uuid()` is the identity call every tool uses. It builds
  a UUID from `rsmi_dev_unique_id_get()`, the PCI device ID and the KFD partition
  index. It is not a CUID: different inputs, layout and framing.
- `amdsmi_get_gpu_device_cuid()` is compiled only when `BUILD_CUID=ON` (default
  `OFF`), returns a single string, and tells the caller nothing about it.
- `libamdcuid` already exposes the primary, the derived value, the component type
  and the auxiliary marker through `amdcuid_query_device_property()`, and seed
  provisioning through `amdcuid_set_hash_key()`. Nothing in it needs to change.
- The two projects are separate deliverables with separate release cadences and
  separate CI, and must stay that way.

## Goals / Non-Goals

**Goals:** make the CUID reportable and provisionable through the tool operators
already run, with enough context attached that a recorded value can be trusted;
keep the dependency one-directional, optional, and invisible to a build without
`libamdcuid`.

**Non-Goals:** merging the projects or vendoring the library; changing any value
the library or driver produces, since this change reports rather than computes;
retiring `amdsmi_get_gpu_device_uuid()`; CPU, NIC and platform CUIDs, which do
not fit the per-GPU handle model.

## Decisions

### A1: supersede `amdsmi_get_gpu_device_uuid()`, do not replace it

*Rejected:* making the existing UUID call return the CUID. It silently changes
the value of a published ABI: the ROCm runtime's agent enumeration, container
device plugins, Kubernetes device advertisement and anything that recorded the
value would read the same symbol and get a different answer for the same card,
with nothing in the type, length or format to show it.

The two values also mean different things. The existing UUID incorporates the
current partition index, so it changes on repartition; a CUID moves the partition
index into UnitID and changes nothing else. Both callers exist, so both calls
remain, and the CLI prints both.

### A2: one struct, not four getters

`amdsmi_get_gpu_cuid_info()` fills a single `amdsmi_cuid_info_t`. The four values
must be consistent with each other: a caller that fetches the derived CUID and
then separately asks whether it is auxiliary can be answered across a seed re-key
or a device rescan and record a value with the wrong provenance.

The primary is `CAP_SYS_ADMIN`-gated at the source, so it is returned as an empty
string to an unprivileged caller rather than failing the whole call.

### A3: report which stage answered

`amdsmi_cuid_info_t` carries the source: driver, daemon/store, or locally
computed. The auxiliary bit says the identity was synthesised, but not the
converse: a locally computed *canonical* primary is a real serial read by an
unprivileged path, and is worth less than the same value published by the driver,
because only one of the two is authoritative.

*Rejected:* exposing only the auxiliary bit. The caller cannot determine
provenance; nothing in the value says where it came from.

### A4: the seed is write-only through `amd-smi`

`amdsmi_set_cuid_seed()` provisions 32 octets. `amdsmi_get_cuid_seed_info()`
reports whether a seed is provisioned and a non-reversible fingerprint,
`SHA-256(seed)` truncated to eight octets, which answers "do two nodes carry the
same seed" with no way back to the value.

The kernel's `cuid_seed` is readable so a provisioning daemon can verify its own
write, but it is `0600`, `CAP_SYS_ADMIN`-gated, and one device's copy. `amd-smi`
is run with `sudo` casually and its output gets pasted into bug reports.

### A5: `BUILD_CUID` defaults to on when the library is found

A feature that must be asked for at build time is absent from every distributed
package. The option stays so a build can force it off. Detection is
`find_package(amdcuid CONFIG QUIET)`, defaulting `ON` when the package is found;
otherwise the calls compile and return `AMDSMI_STATUS_NOT_SUPPORTED`.

*Rejected:* hard-requiring `libamdcuid`. `amd-smi` would stop building on any
tree that has not built the CUID project, including downstream integrations that
consume `amd-smi` alone.

### A6: the CLI prints the derived CUID under `--cuid`

`amd-smi static --cuid` shows the derived CUID, the component type, the auxiliary
flag, the source and the node seed's state. The primary appears only under
`--cuid-primary`, and only where the caller could read it; otherwise the field
reads `N/A (requires root)`, because the primary embeds a raw serial number and
`amd-smi static` output ends up in public bug reports.

The seed's state rides in the same block rather than a command of its own: it is
the context the derived CUID beside it only means anything in.

### A7: a CMake package, and a self-contained archive

`amd-smi` consumes `libamdcuid` through `find_package(amdcuid CONFIG)` and the
imported target `amdcuid::amdcuid`.

*Rejected:* `find_library()` plus a separately located header directory. The two
halves are found independently and can disagree, giving a build that compiles
against one ABI and links against another with nothing to say so.

A static library's link dependencies travel into the export set even when
declared `PRIVATE`. `rocm-sha256` is a build-tree helper that is never installed,
so exporting a link to it produces a package whose imported target names a target
the consumer does not have, and CMake reports this only at install time. The
objects are folded into the archive instead.

## Risks / Trade-offs

- **Two identity calls invite the wrong one being used.** Mitigated by
  documentation ordering and by the CLI, which labels the legacy value as such.
- **A struct in a published header is an ABI commitment.** Mitigated by a
  reserved tail sized for the fields already anticipated, following the pattern
  the neighbouring `amdsmi_*_info_t` structs use.
- **Turning `BUILD_CUID` on by default changes what packages link.** It links a
  static library with no external dependencies, so the added surface is object
  code, not a runtime dependency.
- **`amd-smi` gains a way to invalidate every derived CUID on a node.** That is
  what provisioning is, and it is gated on privilege. Traceability is only in
  place on the kernel side, which logs the association at bind and on every
  re-key; neither `libamdcuid` nor `amd-smi` records it, so a derived CUID issued
  by the library before a re-key is not traceable through this path. Tracked as
  an outstanding gap against `cuid/key-constants`.

## Migration Plan

1. **Header and API**, compiling to `NOT_SUPPORTED` stubs when the library is
   absent.
2. **Rewrite `amdsmi_get_gpu_device_cuid()` as a wrapper**, so there is one
   lookup path from the first commit that has two.
3. **CMake default**, once the calls degrade cleanly.
4. **CLI and Python**, the deliverable an operator sees.
5. **Tests**, in `amd-smi`'s own suite, running with or without CUID support and
   with or without a GPU. A fake sysfs root was the original plan and is not what
   landed: nothing in `amdsmi_get_gpu_cuid_info()`'s path takes an injectable
   root, so the driver-sourced case is exercised only on hardware.

Rollback: every step is independently revertible; the API additions are additive
and the CMake default is one line.

## Open Questions

- Whether the node-scoped identities, the platform CUID and the CPU CUID, are
  reported by `amd-smi` at all, given its per-processor object model.
- Whether seed provisioning should also drive the kernel's per-device
  `cuid_seed`, or only the library's node-wide key file. The kernel's copy is
  per-device and non-persistent, so "the node is provisioned" means different
  things. Scoped out until the kernel's node-wide seed work lands.
