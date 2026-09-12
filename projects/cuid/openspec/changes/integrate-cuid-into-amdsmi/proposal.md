## Why

A CUID that nothing reports is a format, not an identifier. The kernel publishes
one and the library computes one, and `amd-smi`, the tool every operator runs,
prints neither. What it prints instead is `amdsmi_get_gpu_device_uuid()`: a value
built from the SMU unique ID, the PCI device ID and the KFD partition index,
framed as a UUID. That names the right card, but it is not a CUID, not what the
driver publishes, and not what a fleet database keyed on CUIDs will contain.

`amdsmi_get_gpu_device_cuid()` exists behind an off-by-default `BUILD_CUID` and
returns one string with no way to tell what it is. It cannot distinguish a
driver-published identifier from one the library synthesised out of
`/etc/machine-id`, cannot report the primary, and there is no way to provision
the seed every derived CUID depends on, so every machine in the fleet reports a
value produced by a public placeholder key.

This change makes `amd-smi` a first-class CUID consumer and the supported way to
provision the node seed. The dependency stays one-directional and optional, and
`libamdcuid` remains buildable and testable on its own.

## What Changes

- **Add `amdsmi_get_gpu_cuid_info()`**, returning the primary and derived CUIDs,
  the component type, whether the value is auxiliary, and which source answered:
  driver, daemon or local computation. The auxiliary flag and the source are the
  point, because a consumer that cannot tell a driver-published CUID from a
  synthesised one will record both in the same column.
- **Add `amdsmi_set_cuid_seed()` and `amdsmi_get_cuid_seed_info()`** for the
  node-wide 32-octet seed, with the seed itself never readable back through the
  API. Provisioning is what turns a derived CUID from a public, reproducible
  value into a fleet-unique one, and today there is no supported way to do it.
- **Keep `amdsmi_get_gpu_device_uuid()` exactly as it is** and document it as
  superseded rather than removing it. **Not a breaking change**, deliberately:
  see the design note. Its value is not a CUID and never was, and the ROCm
  runtime and several tools read it today.
- **Reduce `amdsmi_get_gpu_device_cuid()` to a wrapper** over the derived string
  from the new call, so there is one implementation and one lookup order.
- **Build `BUILD_CUID` by default where `libamdcuid` is present**, and degrade to
  `AMDSMI_STATUS_NOT_SUPPORTED` where it is not. An identifier nobody's build
  turns on is an identifier nobody has.
- **Report CUIDs in `amd-smi static`** and provision the seed through
  `amd-smi set`, in both the CLI and the Python interface.

## Capabilities

### New Capabilities

- `cuid/amdsmi-identity-api`: what `amd-smi` reports for a component's identity,
  and what a caller can tell about it.
- `cuid/amdsmi-seed-provisioning`: provisioning the node seed through `amd-smi`,
  and what may and may not be read back.
- `cuid/amdsmi-cli`: the command-line and Python surface.

### Modified Capabilities

_None. `cuid/library-consumer` describes the library's obligations as a consumer
of the kernel; this change describes `amd-smi`'s obligations as a consumer of the
library, which is a separate layer with a separate ABI._

## Impact

- **`projects/amdsmi`**: three new API entry points, one existing one rewritten
  as a wrapper, the CMake option's default, the CLI's `static` and `set`
  subcommands, and the Python interface. The goamdsmi and Rust shims are **not**
  updated: they are generated from the header and will pick the new entry points
  up when next regenerated, so until then a Rust or Go consumer has the
  single-string call and nothing else.
- **`projects/cuid`**: one new entry point, `amdcuid_get_key_info()`, so the seed
  fingerprint is computed inside the library and the seed never crosses an ABI
  boundary. The library also gains a packaging obligation: it must export a CMake
  package rather than leaving `amd-smi` to locate a `.a` and a header
  separately.
- **Consumers**: anything reading `amdsmi_get_gpu_device_uuid()` is unaffected.
  Anything reading `amdsmi_get_gpu_device_cuid()` keeps working and gets the same
  string.
- **Operators**: `amd-smi static` grows a CUID block; provisioning a fleet seed
  becomes a documented command rather than a file to write by hand.
