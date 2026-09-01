## Context

See `proposal.md`. The design-relevant constraints:

- **Two implementations already exist and must agree.** The kernel driver and the
  ROCm CUID library have diverged three times: mirrored UUIDv8 octet order, an
  off-by-one component type enumeration, and independently dropping payload bits
  120:121. Cross-layer verification on two W6800s now shows byte-identical
  values.
- **The published specification is internally inconsistent.** S1 v84's primary
  table gives `112:117` to UnitID part 2 while its derived table claims 117 as
  the Auxiliary Value Identifier; the derived table allocates 45 bit positions
  for 46 hash bits while the prose says 110.
- **The auxiliary/fallback construction is not implementable as published**, and
  is user-mode-only work.
- **Sysfs is a stable ABI.** Anything uncertain is left out rather than shipped.
- **The code is MIT-licensed** but the derivation needs
  `hmac_sha256_usingrawkey()`, which is `EXPORT_SYMBOL_GPL`.

## Goals / Non-Goals

**Goals:** one implementation of the packing, framing and derivation with no
driver-specific types in it, so AINIC and NPU can adopt it unchanged; a
resolution of every ambiguity that affects an emitted byte; an infallible read
path; a privilege split enforced per-read, not only by file mode.

**Non-Goals:** seed persistence, node-wide seed scope, IOCTL access,
per-partition UnitID and the auxiliary construction, all deferred; a standalone
`amd_cuid.ko`, though the core is written to be extractable; the userspace
library, tracked separately.

## Decisions

### D1: the core is driver-independent, called through an input descriptor

The caller fills a `struct cuid_ident` (PCI device, serial, vendor/device/
revision IDs, UnitID, component type) and embeds a `struct cuid` in its own
device structure; the core owns the `device_attribute`s and recovers the struct
from `container_of()`. Three AMD drivers are expected to publish CUIDs, so
passing an `amdgpu_device *` would have to be undone before the second caller.

*Rejected:* ops-vtable callbacks. The identity is fully known at bind time and
never changes.

### D2: bit 117 is permanently the Auxiliary Value Identifier; UnitID part 2 is `112:116`

The kernel packs `112:116` for UnitID and reserves 117 for the auxiliary marker
in **both** the primary and the derived layout, so the derived hash slice is
`hash[64:108]`, 45 bits, not 46.

*Rejected:* returning bit 117 to UnitID and typing the auxiliary CUID as UUIDv5.
RFC 9562 defines v5 as SHA-1 over `namespace ‖ name`, so a v5-typed
HMAC-SHA-256 value is not a conforming v5 UUID; it would branch every consumer on
the version nibble, make the same device present different versions by privilege
and by host, and invalidate every derived CUID emitted so far.

*Consequence for the specification:* S1's primary table must become `112:116`,
its derived slice `hash[64:108]`, and its "110 bit" prose 109.

### D3: UUIDv8 framing inserts six bits rather than overwriting

The payload is emitted in order from octet 0, with the version nibble and the two
variant bits shifted in, displacing everything after them. Payload bits 122:127
fall off the end and are always zero, so the transform is exactly invertible.

S1's representation table pins the orientation: version at 48:51 and variant at
64:65 are the RFC 9562 positions only under MSB-first numbering over the rendered
octets, and that table places "ID value part 1, LSB of ID Value" at bits 0:47.
The least significant payload octet therefore leads. The kernel originally
scattered payload bit 0 to the last octet, mirroring the library's value, and was
corrected.

*Rejected:* overwriting the six bits, which is what a naive `uuid` helper does.
It discards two component-type bits and makes the value non-invertible.

### D4: remove the in-kernel auxiliary/fallback path entirely

`cuid_aux_serial()`, its DMI/SMBIOS lookup, the `aux` argument threaded through
the packing routine and the `cuid_temporary` attribute all go. The library's
auxiliary algorithm is defined over `/etc/machine-id`, which a kernel driver
cannot read, so a kernel-side temporary CUID would disagree by construction.
Removing the path also deletes the kernel's only dependency on DMI.

### D5: compute at bind, cache, serve from cache

Primary and derived values are computed in the init path and stored; the show
handlers only format; the derived value is recomputed synchronously inside the
seed write. This puts the only fallible work, attribute group creation, in the
probe path where failure can be reported. Enabled by the synchronous
`hmac_sha256_usingrawkey()`: no allocation, no request objects, no error return.

### D6: mutex around the seed, and only the seed

`seed_lock` guards the seed bytes, its length, and the cached derived UUID.
Primary is immutable after init. The secondary show handler takes the lock,
copies 16 bytes out, drops it, and formats outside, so a reader concurrent with a
re-key sees the old or the new value, never a torn one.

### D7: capability checked on every read, not just the file mode

`cuid_primary` and `cuid_seed` are `0400`/`0600` *and* call
`capable(CAP_SYS_ADMIN)` in their handlers. File mode is checked at open: a
descriptor can be opened under one set of credentials and read under another, and
a container or relaxed sysfs mount can make the path reachable.

### D8: the seed round-trips as raw bytes

Read and write use the same encoding: raw bytes, no newline, no hex, no trailing
NUL. A write is accepted only at exactly 32 bytes (see D7 in
`pin-cuid-cross-layer-contract`). Read returns whichever seed is in use; the
built-in default is shorter than 32 bytes, so a read before provisioning returns
fewer bytes than a write would accept.

*Trade-off:* `cat cuid_seed` prints binary. Acceptable for a root-only secret,
and documented.

### D9: the association record is a `dev_info()` log line

S1 requires the derived-to-primary association to be tracked "with a timestamp in
a log". The driver emits one `dev_info()` naming both values at bind and on every
re-key.

## Risks / Trade-offs

- **The specification may be revised against these resolutions (D2).** Every
  resolution is written back as a concrete edit with replacement text. The
  direction chosen changes no already-emitted value; the alternative invalidates
  all of them.
- **The two placeholder constants are not final (T1).** `CUID_DEFAULT_SEED` is
  byte-identical between kernel and library, so the layers agree whatever the
  final value is.
- **The layers can silently diverge again.** Mitigated by the shared vectors,
  bound into both trees' CI.
- **The GPL-only HMAC export against MIT-licensed files.** Tolerable while the
  object is linked into `amdgpu`; must be resolved before the core becomes a
  standalone module.
- **The seed is lost on module reload.** Documented in the ABI file as a
  requirement to re-provision. A world-readable efivar is not an acceptable store
  for a secret.
- **Cross-instance correlation under virtualisation.** `cuid_secondary` is `0444`
  in a guest, so two guests on the same hardware can determine they share a
  component. Answered by the `cuid` module parameter; see tasks 8.1.

## Migration Plan

There is nothing deployed to migrate; the out-of-tree branch that loses
`cuid_temporary` has no consumers. Deployment is a normal driver update: the
attributes appear on bind, the derived value is immediately readable from the
default seed, and a provisioning daemon writes the real secret when one exists.
Rollback is removing the group; nothing persists across it. Before posting, the
userspace library work and the shared vectors land first, so there is a consumer.

## Open Questions

- **Exact bytes of the canonical default seed (T1).** Not blocking: both layers
  ship the same placeholder, overridden in deployment.
- **When does the library start reading these files (T3)?** Owned by the library
  schedule.
