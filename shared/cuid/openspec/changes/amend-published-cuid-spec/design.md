## Context

`openspec/specs/cuid/` records the published specification at version 84. This
change is the delta between that and what the kernel driver and the userspace
library had to agree on to emit the same value for the same card. Both are
verified against the shared conformance vectors and against two W6800s, so the
question here is not what the format should be, but what the page has to say so
the next implementation reaches the same answer without three defects on the way.

Where a decision here overlaps `pin-cuid-cross-layer-contract`, that change
governs the value; this one governs only the wording of the page.

## Goals / Non-Goals

**Goals:** remove every contradiction the page has with itself, so a conforming
implementation is possible at all; state every value an implementer is currently
forced to invent.

**Non-Goals:** changing any emitted value; redesigning the format, since the
two-level model, the UUIDv8 framing, the HMAC derivation and the component-type
enumeration are kept as published; the virtualization and out-of-band sections,
which are internally consistent.

## Decisions

### E1: UnitID yields bit 117, not the Auxiliary Value Identifier

The Primary and Derived tables both claim bit 117. UnitID gives it up, becoming
13 bits and capping a partition index at 8191, which no AMD component approaches.

*Rejected:* keeping `112:117` for UnitID and moving the auxiliary marker into the
derived layout's Reserved field at `118:121`. That range is Component Type in the
primary layout, so the marker would sit at a different bit in the two layouts and
could not be carried across the derivation by a single mask. A derived value
could then not be identified as auxiliary without reference to its primary.

A producer that kept the bit for UnitID would also set the auxiliary marker on
any device whose UnitID exceeds 8191, silently relabelling a canonical identifier
as synthesised.

### E2: the slot width wins over the label and the prose

The derived hash slot is described three ways: 45 bits by position, 46 by label,
110 by prose. The position is authoritative because it is the only one consistent
with the rest of the table: 64 + 45 + 1 + 4 + 8 = 122 exactly. Both
implementations already carry 109 bits.

### E3: uniform UUIDv8, and the auxiliary path loses its namespace

*Rejected:* keeping auxiliary CUIDs as UUIDv5 and widening the derived slot to 46
bits. A v5-typed HMAC-SHA-256 value is not a conforming v5 UUID, so a validating
parser may reject it; it would split every consumer's parser in two; and it would
make a device's UUID version depend on the caller's privilege and the host it was
enumerated on. Both packages were acceptable to the page's author; this one is
what the kernel and S3 already implement.

*Consequence:* the `amd.com` namespace string and the namespace form cease to
exist as questions rather than being answered.

### E4: state the constants, even though they are placeholders

The canonical seed's and the temporary key's exact bytes were requested and did
not arrive. Both are placeholders in both layers, and byte-identical across them.
Writing down what already ships changes no emitted value; not writing it down
means the third implementation invents a third value, and an unprovisioned
machine reports different derived CUIDs depending on which layer is asked.

### E5: orientation is part of a field definition

The page gives the Device Serial Number's capability ID and width but not its
byte order, and gives the NIC MAC fallback but not its orientation. Both gaps
produced divergence. **A field definition that does not fix the byte order is not
a field definition.**

## Risks / Trade-offs

- **Narrowing UnitID is technically breaking.** Only for a producer that packed a
  UnitID above 8191, which none does.
- **Publishing placeholder constants makes them harder to change later.**
  Mitigated by stating that the fallback seed is public and a placeholder, and by
  splitting the derived vectors so one uses a reproducible test key. If the
  constant moves, exactly one vector moves with it.
- **The page and the code can drift again.** The conformance vectors are the
  mitigation, and the reason task 4.1 exists.
- **Retiring S4 loses history.** S4 is empty. What it cost was being cited by
  S1's own reply as the authority for the bit tables.

## Open Questions

- Whether the specification should describe a hypervisor opt-out at all. amdgpu
  implements one as a module parameter, on the reasoning that a GIM-level disable
  leaves bare metal and non-GIM hypervisors with no lever.
- Whether the CPU auxiliary structure should carry the APIC or package identifier
  in the reserved bits `220:255`. Additive within a reserved field.
- Where the shared conformance-vector artifact should physically live.
