## Context

Two independent producers exist: the kernel driver (`amdgpu_cuid.c`) and the
userspace library (`projects/cuid`). Each computes the whole format from scratch,
nothing forces them to agree, and they have been verified to agree exactly once,
by hand, on two W6800s. The kernel is already conforming for the parts this
change touches; the library is not.

Four specification pages disagree with each other. S1 is normative and its
derived table is off by one against its own prose; S3 mandates bit 117 and
publishes a device-type enumeration offset by one from S1's; S4 is empty and is
the page S1's own reply cites as the authority for the bit tables.

The constraint that shapes everything below: **no conforming value may change
unless it is currently wrong.** The alternative is invalidating identifiers
already recorded in the field.

This change is authoritative for any value the layers must agree on. Where
another change in this directory states one of those values, this one governs.

## Goals / Non-Goals

**Goals:** express the format as values, not prose, and make those values
executable; make a future divergence between the two layers a build failure.

**Non-Goals:** anything the kernel already does correctly (the only kernel-side
requirement here is the seed length check); the kernel's auxiliary path, deleted
by `add-cuid-kernel-interface`; seed persistence, node-wide seed plumbing, the
IOCTL and the hypervisor opt-out; editing the specification pages, which this
change only says what to say.

## Decisions

### D1: uniform UUIDv8 with bit 117, not UUIDv5 for the auxiliary

*Rejected:* auxiliary CUIDs typed as UUIDv5, with the derived slot widening to 46
bits. RFC 9562 defines v5 as SHA-1 over `namespace ‖ name`, so a v5-typed
HMAC-SHA-256 value is not a conforming v5 UUID; it would split every consumer
onto two parse paths, make a device's UUID version depend on the caller's
privilege, and invalidate every derived CUID emitted so far. Choosing v8 changes
no emitted value.

*Consequences:* the auxiliary marker is bit 117; the derived slot stays 45 bits;
S1's derived slice renumbers to `hash[64:108]` and its prose drops 110 to 109;
S1's primary table becomes `112:116`; the namespace string, namespace form and
HMAC operand-order questions cease to exist; and the temporary fixed key becomes
necessary, because without a namespace nothing else separates the two
derivations. The UUIDv5 reference may remain in S1 as a note on the derivation
*style*, but must not describe the version nibble.

### D2: constants are ASCII, unpadded

Both keys are fixed at the bytes the two layers already ship, with no NUL and no
padding.

*Rejected:* padding both to 32 octets, as an earlier kernel draft did.
HMAC-SHA-256 pads a short key to its own **64**-octet block internally, so
padding to 32 first is a different key for no cryptographic gain. Neither key is
a secret.

### D3: one derivation function, one operand order

Every derivation is `HMAC-SHA256(key, message = the 16 primary payload octets)`.
Only the key varies: the seed for a canonical derived CUID, the temporary fixed
key for an auxiliary one.

*Rejected:* the library's current path, keyed with the primary payload and passed
a fixed application UUID as the message. With a public key HMAC is a keyed hash
whose preimage resistance protects the message either way, so the swap bought a
second code path and a real defect: the derived auxiliary CUID read bit 117 out
of the fixed constant and was never marked auxiliary.

### D4: fixed-width binary input structure for the auxiliary serial

S1's 256-bit table is kept with two boundaries repaired so the widths sum to 256
(`0:15` Format, `16:143` Machine ID). The serial is the first 8 octets of the
unkeyed SHA-256 of that structure, little-endian.

*Rejected:* the library's string concatenation, a BDF string filtered to
hexadecimal digits joined to the machine-id string. The filter erases separators,
so `0000:65:00.0` and `0000:65:0:00.0` reduce to the same input and the CPU
variant's `"socket:"` prefix degenerates to the constant `cce`.

*Also rejected:* the kernel's auxiliary serial. The auxiliary path is user-mode
only and the kernel's copy is being deleted.

### D5: vectors are a shared artifact, not a transcription

The vector suite lives in one place, both trees consume it, and a drift check
fails the build.

*Rejected:* writing the same expectations into each producer's own suite. That is
what exists now, and it is how the bug got confirmed against itself: a
reverse-lookup test decoded the component type from the same wrong bit positions
the packer wrote it to, and passed for months.

### D6: Platform CUID is the SMBIOS UUID, untouched

*Rejected:* the library's fold of the system UUID to a 64-bit fingerprint packed
through the normal layout. S1 says twice to use the value directly, and the fold
discards half of an identifier the firmware has already made unique while making
the result depend on which producer folded it.

*Cross-layer consequence:* the literal reading leaves the Platform CUID with no
component type and no framing of ours. That is the specification's intent, and it
was confirmed rather than assumed, so a consumer must not expect to decode a
Platform CUID's payload fields.

### D7: 32 octets for a provisioned seed, and the default is exempt

S1 and S3 both specify a 256-bit shared secret. The kernel formerly accepted any
write no longer than 32 bytes; it has since moved to exact-32 (`amdgpu_cuid.c`,
`cuid_seed_store()`: `if (count != sizeof(cuid->seed)) return -EINVAL;`).

The canonical fallback seed is 24 octets and stays 24. It is a built-in
placeholder, not a provisioned secret, so the length rule does not apply, and
lengthening it would change every unprovisioned derived value for nothing.

*Trade-off accepted:* the two lengths look inconsistent side by side. The
alternative is changing values or weakening the rule that catches a corrupt key
file.

### D8: S4 is retired, not restored

S4's body is "Duplicated from S1" plus a stray character, and it is the page S1's
reply links to as the authority for the bit tables.

*Rejected:* restoring S4 by copying S1's tables into it. A duplicated normative
table is how S1 and S3 got out of step on the device-type enumeration.

### D10: the PCIe Device Serial Number is little-endian, unswapped

Config space is little-endian and the DSN capability's first serial dword is the
low half, so the eight octets at `dsn_cap_offset + 4` are a 64-bit little-endian
value that payload bits 0:63 carry verbatim. The kernel already did this via
`pci_get_dsn()`; the library byte-swapped.

*Rejected:* keeping the library's swap. The swapped value is neither the number
the capability holds nor the number any other tool prints for the same card.
Every DSN-sourced library CUID changes once, and nothing that was correct
changes.

### D11: a NIC's permanent MAC address needs an orientation, not permission

A NIC with no driver serial, no DSN and no vendor-specific capability falls back
to its permanent MAC address, octet 0 at payload bits 0:7. S1 already permits the
source in two places; neither says which end goes at bit 0, and an unstated
orientation is exactly how D10 happened.

*Rejected:* dropping the fallback and letting such a NIC take an auxiliary CUID.
A burned-in MAC is a genuine per-device serial; an auxiliary CUID is explicitly
not one.

### D12: the driver stage belongs to the bus, not to amdgpu

The staged lookup's first stage is attempted for every component with a PCI
routing ID, not only for the GPU whose driver publishes the attributes today.

*Rejected:* generalising when a second producer appears. The failure mode of
deferring it is silent: the day an NPU driver publishes a CUID, the library keeps
computing its own and the two disagree with nothing to say so. A component with
no routing ID falls through at the first check.

### D13: the serial-only Platform branch carries no vendor

The else-branch of the Platform CUID, SHA-256 of the SMBIOS system serial when
there is no system UUID, packs UnitID, Revision, Device and Vendor as zero, as
the specification says. The library was packing an SMBIOS-derived vendor
ID into bits 96:111, which is a property of whoever assembled the machine, not of
the platform.

## Risks / Trade-offs

- **Renumbering the library's device-type enumeration is a published-API break,
  and every library-emitted GPU CUID changes.** The current values misname the
  hardware: a GPU ships as type `3`, which a conforming reader decodes as a NIC.
  Land it before the library has consumers that have recorded CUIDs.
- **The two constants are fixed without the answer that was asked for.** Both are
  fixed at the values already shipping in both layers, so the decision is a no-op
  against the current state, and the vectors make a later replacement mechanical.
- **The auxiliary construction is specified in more detail than S1 states.**
  Unavoidable: S1's version cannot be implemented as written. The two changed
  boundaries are called out so a spec author can apply exactly them.
- **The vector suite pins placeholder constants into a normative artifact.** D-1
  uses the canonical seed and will change if that constant does; D-2 uses
  `00..1f`, pinning the fold, framing and operand order independently.
- **A drift check across two repositories is a coupling that can rot.** It is the
  mechanism already in use for the SHA-256 sources in the library's CI.

## Migration Plan

1. **Library, format-affecting, one commit**: device-type renumbering,
   component-type high-bit packing, the framing's last octet, the temporary-key
   operand order and key, and the Platform passthrough. These must land together,
   because changing the packing without the framing collides an NPU with a
   Platform.
2. **Vectors into both trees**, with the drift check.
3. **Kernel, one line**: the seed length check.
4. **Library, kernel consumption**: the staged lookup. This is the durable fix.
5. **Specification edits**, independent of the code.

Rollback: steps 2, 3 and 4 are independently revertible. Step 1 is not, in the
sense that reverting it restores identifiers that name the wrong device class.

## Open Questions

- Where the shared vector artifact physically lives. The drift mechanism is the
  same whichever tree hosts it.
- Whether the CPU Format's auxiliary structure should carry the APIC or package
  identifier in the reserved bits `220:255`. Additive within a reserved field.
