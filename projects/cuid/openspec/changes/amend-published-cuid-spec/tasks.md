No code. Every item is an edit to a Confluence page, and every one is already
implemented in both trees, which is what makes them safe. The exact replacement
text for each S1 item is in `specs/cuid/` beside this file.

## 1. S1: field tables

- [ ] 1.1 Primary table: narrow UnitID part 2 from `112:117` to `112:116`, and
      add a row naming bit 117 the Auxiliary Value Identifier. This is the edit
      that stops bit 117 having two owners.
- [ ] 1.2 Derived table: relabel the `72:116` slot from `hash[64:109]` to
      `hash[64:108]`, so the label matches the 45-bit width the table already
      gives it.
- [ ] 1.3 Derived prose: correct "110 bit LSB of the resulting hash" to 109, and
      "12 bits reserved" to 13.
- [ ] 1.4 State that UnitID is 13 bits total, split `64:71` / `112:116`.

## 2. S1: the fallback section

- [ ] 2.1 Correct the auxiliary input structure to Format `0:15` and Machine ID
      `16:143`, so both fields are whole octets.
- [ ] 2.2 Correct "32bit MachineID" to 128 bits, matching the `/etc/machine-id`
      the same row names.
- [ ] 2.3 Withdraw UUIDv5. Keep the UUIDv5 reference only as a note on the
      derivation *style*, stated explicitly not to describe the version nibble.
- [ ] 2.4 Delete the `amd.com` namespace string and the namespace form, which
      have no meaning once the auxiliary value is a normal UUIDv8.
- [ ] 2.5 State that a CPU's auxiliary Routing ID field is zero.
- [ ] 2.6 Widen the auxiliary Component Type row from `2/3/4` to the full
      on-wire numbering.

## 3. S1: values the specification leaves to the implementer

- [ ] 3.1 State the canonical fallback seed as the 24 ASCII octets
      `AMD-CUID-DEFAULT-SEED-v1`, unpadded and not NUL-terminated.
- [ ] 3.2 State the auxiliary fixed key as the 20 ASCII octets
      `AMD-CUID-TEMP-KEY-v1`, on the same terms, and pin the operand order.
- [ ] 3.3 State that a provisioned salt is exactly 32 octets and that any other
      length is refused rather than adopted.
- [ ] 3.4 State that the PCIe Device Serial Number is used in
      configuration-space order, little-endian and unswapped, read from
      `dsn_cap_offset + 4`.
- [ ] 3.5 State the NIC MAC fallback's orientation, and that a zero-valued
      source is absent rather than an identity.

## 4. S1: worked examples

- [ ] 4.1 Paste the conformance vectors from
      `pin-cuid-cross-layer-contract/specs/cuid/conformance-vectors/spec.md` in
      as normative text: the primary vectors, all six component types, the two
      derived vectors and the auxiliary pair.
- [ ] 4.2 State the obligation that every producer reproduces them, and that the
      artifact is shared rather than transcribed per producer.

## 5. S3 and S4

- [ ] 5.1 S3: renumber the device-type enumeration onto the on-wire values
      (`PLATFORM = 0x0` through `NPU = 0x4`) and note the sentinel move to
      `0xFF`.
- [ ] 5.2 S3: add NPU, which the page does not currently list.
- [ ] 5.3 S3: remove or correct the example call to a query that is not in the
      enumeration.
- [ ] 5.4 S3: repoint the fallback reference at S1.
- [ ] 5.5 Retire S4. It is empty, and it is the page S1's own reply cites as the
      authority for the bit tables. A duplicate is how the two got out of step.

## 6. Not edits to the page

- [ ] 6.1 Get an owner for the S3 enumeration change. It is a published API and
      the library change depends on it.
- [ ] 6.2 Decide whether the specification should say anything about the
      hypervisor opt-out. `amdgpu` implements one as a module parameter
      (`add-cuid-kernel-interface` D-8.1); the page currently describes no
      opt-out at any level.
