## Why

Every remaining CUID question is open for the same reason: the specification
describes the format in prose and tables, and never once states a value. Three
defects have shipped because of it. Two implementations rendered the same payload
as mirror images; both independently dropped payload bits 120:121, so an NPU
collided with a Platform; and S1's derived table is off by one against its own
prose.

`add-cuid-kernel-interface` fixes the wire format on the kernel side. It cannot
fix the rest: the two constants that key every derived value, the auxiliary
construction the userspace library owns, and the absence of any artifact that
makes a divergence between the two layers fail rather than go unnoticed. This
change supplies those as literal bytes and executable vectors, disposing of the
questions in `CUID-FOLLOWUP.md`, `CUID-SIGNOFF.md` and `CUID-SIGNOFF-TONY.md`
that are still marked blocking.

Where an answer was requested and did not arrive, this change makes the decision
and notes what it would cost to reverse, which for every item below is nothing,
because no conforming value changes.

## What Changes

- **Pin the canonical fallback seed** as the ASCII bytes
  `AMD-CUID-DEFAULT-SEED-v1` (24 octets, no NUL, **not** padded). This is what
  the kernel and the library already ship, byte for byte, so it changes no
  emitted value. Answers `CUID-FOLLOWUP.md` Q2(a) / `CUID-SIGNOFF-TONY.md` T1.
- **Pin the temporary/auxiliary fixed key** as the ASCII bytes
  `AMD-CUID-TEMP-KEY-v1` (20 octets, no NUL, **not** padded), used in the same
  operand order as every other derivation (key = the fixed key, message = the
  16 auxiliary primary octets). Answers Q2(b) / T2.
  **BREAKING** against both current implementations: the kernel zero-pads this
  key to 32 octets, and the library instead inverts the operands and uses a
  fixed application UUID as the message. Both change. The kernel's copy is being
  deleted outright, so in practice only the library moves.
- **Require a provisioned seed to be exactly 32 octets.** S1 and S3 both say
  256 bits; the library already enforces it and the kernel accepts anything up
  to 32. **BREAKING** for the kernel's `cuid_seed` write path, which must reject
  a short write with `-EINVAL` rather than silently keying on it.
- **Define the auxiliary CUID** as a normal UUIDv8 primary with bit 117 set,
  built from a repaired version of S1's 256-bit input structure. The Format
  field becomes `0:15` and the Machine ID `16:143`. S1's ranges already sum to
  256; what is wrong is that they give Format 17 bits and Machine ID 127, so
  neither is a whole number of octets and the Machine ID cannot hold the
  128-bit `/etc/machine-id` its own description names. Answers `CUID-SIGNOFF.md`
  §2.1 and voids §2.2 to §2.4: under uniform v8 there is no namespace, so the
  namespace string, the namespace form and the HMAC operand order cease to be
  questions.
- **Define the per-component-type serial sources**, including the Platform CUID
  as the SMBIOS system UUID used verbatim with a serial-number else-branch.
  **BREAKING** for the library, which collapses the 16-byte system UUID to a
  64-bit fingerprint and discards half of it.
- **Publish normative conformance vectors**: inputs, 16-octet payload hex, full
  HMAC digest and final UUID string for the primary, the derived value under two
  keys, all six interesting component types, a split UnitID, and the auxiliary
  pair. Answers `CUID-SIGNOFF.md` §4 item 10, the item every other defect traces
  back to.
- **Require the library to consume the kernel interface** and to stop
  recomputing what the kernel already publishes, with the staged fallback order
  S2 and S3 already specify. Answers T3's "when, not whether".
- **Renumber the library's device-type enumeration onto the wire values**
  (`PLATFORM = 0x0` through `NPU = 0x4`, sentinel moved to `0xFF`). **BREAKING** for a
  published API: the library currently emits `0x3` for a GPU, which a conforming
  reader decodes as a NIC. Answers T8.

## Capabilities

### New Capabilities

- `cuid/key-constants`: the two key values as literal bytes, the length and
  scope rules for a provisioned seed, and what re-keying means.
- `cuid/auxiliary-identifier`: the user-mode auxiliary CUID. The repaired
  256-bit input structure, the synthesised serial, bit 117, and the derivation.
- `cuid/component-sources`: which serial each component type contributes to
  bits 0:63, including the Platform CUID's verbatim SMBIOS UUID.
- `cuid/conformance-vectors`: the normative worked examples, and the obligation
  on every producer to reproduce them.
- `cuid/library-consumer`: the userspace library's lookup order, its device-type
  enumeration, and its prohibition on independent recomputation.

### Modified Capabilities

_None. `openspec/specs/` records the published specification as a baseline, and
every repair to it is owned by `amend-published-cuid-spec`.
`cuid/identifier-format` and `cuid/sysfs-interface` exist only as deltas in the
unarchived `add-cuid-kernel-interface`, which this change layers on top of rather
than edits. The one requirement here that touches the kernel, the 32-octet seed
length, is called out in that change's tasks instead._

## Impact

- **Specification pages**: S1 needs the derived slice renumbered to
  `hash[64:108]`, the prose dropped from 110 to 109, the primary table corrected
  to `112:116`, and the fallback section's fields repaired to `0:15` / `16:143`.
  S3 needs its device-type enumeration renumbered, and S4 should be retired in
  favour of S1. Tracked in full in `amend-published-cuid-spec`.
- **Userspace library** (`projects/cuid`): the device-type enumeration, the
  component-type high-bit packing, the last octet of the UUIDv8 framing, the
  temporary-CUID operand order and key, the Platform fingerprint collapse, and
  the kernel-consumption path. This is the bulk of the work.
- **Kernel** (`amdgpu_cuid.c`): the seed length check. Everything else here is
  either already implemented or being deleted.
- **Consumers**: amd-smi, rocprofiler-sdk, the debugger and anything that has
  recorded a CUID from the library. Every library-emitted GPU CUID changes when
  the enumeration is renumbered, which is the point: the current values claim to
  be NICs.
- **CI**: the conformance vectors have to run in both trees, or this change buys
  a one-time correction rather than a durable one.
