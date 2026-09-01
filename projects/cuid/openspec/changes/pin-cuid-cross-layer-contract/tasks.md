Ordering matters. Group 1 must land before group 2: the four library format
fixes are only jointly correct, because the component-type packing and the
framing's last octet collide an NPU with a Platform if either moves alone.

## 1. The shared vector artifact

- [x] 1.1 Write the generator script that produces every vector in
      `specs/cuid/conformance-vectors/spec.md` from its stated inputs: primary
      packing, UUIDv8 framing, the derived fold, the auxiliary input structure
      and the auxiliary serial.
- [x] 1.2 Assert in the generator that the framing discards exactly payload bits
      122:127, by probing all 128 single-bit payloads against the all-zero
      framing, and that the framing round-trips over at least a few thousand
      random 122-bit payloads.
- [x] 1.3 Assert that every vector's UUID string reproduces from its payload,
      and that every HMAC digest reproduces under `openssl dgst -sha256 -mac
      HMAC`, so the artifact is checkable without the generator.
- [x] 1.4 Emit the vectors as a machine-readable table both trees can parse, and
      as the markdown table the specification pages need pasted into them.
- [x] 1.5 Regenerate and confirm P-1 and P-2 still equal the two
      hardware-verified W6800 values.

## 2. Library: the four joint format fixes

These land as one commit. Any subset is wrong.

- [x] 2.1 Renumber `amdcuid_device_type_t` onto the on-wire values
      (`PLATFORM = 0x0`, `CPU = 0x1`, `GPU = 0x2`, `NIC = 0x3`, `NPU = 0x4`) and
      move the sentinel to `0xFF`.
- [x] 2.2 Pack the component type's high two bits at payload bits 120:121
      (`raw[15]` bits 0:1) instead of 126:127.
- [x] 2.3 Change the framing's last octet to carry payload bits 120:121 rather
      than 126:127, and change the inverse to match.
- [x] 2.4 Take the enumeration type, not an integer, in the primary-generation
      interface so a raw value cannot be passed.
- [x] 2.5 Fix the reverse-lookup test, which decodes the component type from the
      same bit positions the packer wrote it to and so confirms the bug against
      itself.
- [x] 2.6 Update the one existing framing known-answer test, whose expected
      value encodes the old last-octet convention.

## 3. Wire the vectors into both trees

- [x] 3.1 Add the vector suite to the library's tests as real assertions, not a
      transcription.
- [x] 3.2 Add the vector suite to the kernel side. A userspace harness over the
      same packing, framing and derive functions is sufficient.
- [x] 3.3 Add the drift check that fails the build when a local copy differs
      from the shared artifact, modelled on the existing SHA-256 drift check.
- [x] 3.4 Retire or regenerate the stale reference artifacts that still carry
      the withdrawn mirrored framing, so nothing in the tree contradicts the
      vectors.

## 4. Library: keys and the auxiliary construction

- [x] 4.1 Replace the temporary-CUID construction: key is the 20-octet
      `AMD-CUID-TEMP-KEY-v1`, message is the 16 auxiliary primary octets. Remove
      the fixed application UUID and the inverted operands.
- [x] 4.2 Confirm bit 117 now survives into the derived auxiliary value, and add
      the A-2 vector assertion that catches it if it stops.
- [x] 4.3 Replace the string-concatenation auxiliary serial with the 32-octet
      fixed-width input structure, and delete the hexadecimal-digit filter.
- [x] 4.4 Populate the structure's Machine ID from the 16 decoded octets of
      `/etc/machine-id`. Where no machine identity is available the library
      declines to emit an auxiliary identifier rather than packing zero, since
      zero would collide across two identically configured hosts.
- [x] 4.5 Fix the CPU auxiliary seed so the library and its test agree on what
      the input is; they currently do not.
- [x] 4.6 Fix the on-disk round-trip, which stores five hash bits in the octet
      holding bit 117 and reads back six.
- [x] 4.7 Implement the documented query that reports whether a CUID is
      auxiliary, or remove the documentation claiming it exists.

## 5. Library: Platform and identity sources

- [x] 5.1 Use the SMBIOS system UUID verbatim as the Platform CUID; delete the
      64-bit fingerprint collapse.
- [x] 5.2 Implement the serial-number else-branch through the normal layout with
      Component Type `0x0`.
- [x] 5.3 Keep the all-`0x00`/all-`0xFF` sentinel rejection and confirm it routes
      to the else-branch rather than to an auxiliary CUID.
- [x] 5.4 Confirm the PCI configuration-space reads land as specified: no
      byte-swap on vendor and device, a single octet for the revision.

## 6. Library: kernel consumption

- [x] 6.1 Read the driver-published primary and derived CUIDs where the
      interface is present, and return them unchanged.
- [x] 6.2 Fall through to the daemon or configuration store, then to an
      auxiliary CUID, in that order.
- [x] 6.3 Stop recomputing a primary for any component whose driver published
      one, including where the library has the inputs.
- [x] 6.4 Add a test that a driver-published value and the library-reported
      value for the same component are the same value, not merely equal.

## 7. Library: key handling

- [x] 7.1 Confirm a key file of any size other than 32 octets is refused as
      corruption and does not fall back to the canonical seed.
- [x] 7.2 Confirm an absent key file uses the 24-octet canonical seed, and that
      the resulting derived CUID equals the driver's on an unprovisioned machine.
- [x] 7.3 Confirm a null or unset key reports an error rather than proceeding.
- [x] 7.4 Confirm the provisioning call propagates failure to its caller.
- [x] 7.5 Confirm any digest other than SHA-256 is rejected at selection time.

## 8. Kernel

- [x] 8.1 Reject a `cuid_seed` write whose length is not exactly 32 octets with
      `-EINVAL`, leaving the effective seed unchanged.
- [x] 8.2 Document the exact length in the ABI file, and document the canonical
      fallback seed as 24 octets, unpadded, not NUL-terminated.
- [x] 8.3 Confirm no kernel path sets bit 117 after the auxiliary path is
      removed by `add-cuid-kernel-interface`.

## 9. Specification edits

_Superseded. These are edits to Confluence pages, now specified in full, with
replacement text, in `amend-published-cuid-spec`. The entries below stay as the
pointer; do not work them from this file._

- [ ] 9.1 S1 derived table: renumber the slice to `hash[64:108]` and correct the
      prose from 110 to 109.
- [ ] 9.2 S1 primary table: correct UnitID part 2 to `112:116`, so it stops
      claiming bit 117 that the derived table also claims.
- [ ] 9.3 S1 fallback section: correct the input-structure ranges to `0-15`
      Format and `16-143` Machine ID, and correct the "32bit MachineID" prose to
      128 bits.
- [ ] 9.4 S1: keep the UUIDv5 reference only as a note on the derivation style,
      stated explicitly not to describe the version nibble.
- [ ] 9.5 S1: paste in the worked example from task 1.4 as normative.
- [ ] 9.6 S3: renumber the device-type enumeration to the on-wire values and
      note the sentinel move; add NPU, which S3 does not currently list; remove
      or correct the example call to a query that is not in the enumeration.
- [ ] 9.7 Retire S4 and repoint S3's fallback reference at S1.

## 10. Cross-layer verification

_Done, against the two W6800s at `0000:03:00.0` and `0000:63:00.0` with the CUID
driver loaded. `cuid_primary` reads `d4abaad3-9b34-8c50-9800-028dcc084200` and
`ffeb5272-7771-88c8-b800-028dcc084200` (P-1, P-2); `cuid_secondary` reads
`61ffe99a-b3e0-8e16-a802-4b1d515d5438` (D-1), which
`cuidtstUnprivileged.ConformanceVectors` computes independently from the P-1
inputs. Writing the 32-octet key `00..1f` to `cuid_seed` produced
`73488f9e-ea52-86ce-8401-2627fa41b068` (D-2), and restoring the canonical seed
restored D-1._

- [x] 10.1 Re-run the two-W6800 comparison: driver and library must emit
      byte-identical primary CUIDs unprovisioned.
- [x] 10.2 Provision a 32-octet seed and confirm byte-identical derived CUIDs.
- [x] 10.3 Confirm an unprovisioned machine produces identical derived CUIDs
      from both layers under the canonical seed alone.
- [x] 10.4 Confirm every vector in the specification passes in both trees' CI,
      not just locally.

## 11. Cross-layer defects found while wiring the two layers together

_Found by reading the two layers side by side after the staged lookup landed.
Each is a value-changing divergence that no test in either tree could have
caught, because each layer was self-consistent. D10-D13 in `design.md`._

- [x] 11.1 Remove the library's DSN byte-swap and read the capability's eight
      octets as the little-endian 64-bit value the kernel reads (D10).
- [x] 11.2 Delete `PciUtil::le64_to_be64` outright, so the swap cannot be
      reintroduced by a caller that assumes it is needed.
- [x] 11.3 Decode the NIC MAC fallback explicitly little-endian rather than by
      `memcpy` over a host-order integer, and pin the orientation in the
      specification (D11).
- [x] 11.4 Move the driver stage from `CuidGpu` to `CuidDevice` so the NIC and
      the NPU consult it too, and delete the GPU-specific override (D12).
- [x] 11.5 Zero the Vendor, Device, Revision and UnitID fields in the
      serial-only Platform branch (D13).
