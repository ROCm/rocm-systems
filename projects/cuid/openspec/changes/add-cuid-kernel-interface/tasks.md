Work already landed on `dgalants/cuid-refactor` is checked, with the commit that
did it. Everything unchecked is outstanding.

## 1. CUID core: packing and framing

- [x] 1.1 Add `amdgpu_cuid.{c,h}` with `struct cuid_ident`, `struct cuid` and
      the `cuid_sysfs_init()` / `cuid_sysfs_fini()` entry points; no
      driver-specific types in either file (D1). `04b1c6231e64`
- [x] 1.2 Pack the 122-bit primary payload LSB-first into 16 octets per the
      field table, with UnitID split `64:71` / `112:116` and the component type
      spanning the octet-14/15 boundary at bits 118:121. `41978841be6d`
- [x] 1.3 Define the component type constants using the specification's on-wire
      numbering (Platform 0 to Other 0xF), so the constant can be written to the
      wire without translation. `41978841be6d`
- [x] 1.4 Render the payload as an RFC 9562 UUIDv8 by *inserting* the version
      nibble at 48:51 and the variant bits at 64:65, payload LSB first (D3).
      `549b13784104`
- [x] 1.5 Carry payload bits 120:121, the component type's high bits, through
      the framing into the last octet instead of dropping them. `5570bfc9357f`
- [x] 1.6 Confirm no path sets bit 117, and that the constant is documented as
      reserved for user-mode auxiliary values. `cuid_pack()` never sets it and
      `cuid_derive()` only copies it from the primary.

## 2. Derivation and seed

- [x] 2.1 Derive the secondary payload as
      `HMAC-SHA256(key = seed, message = the 16 primary octets)` using the
      synchronous `hmac_sha256_usingrawkey()`, so the path cannot fail (D5).
      `41978841be6d`
- [x] 2.2 Fold the digest into the derived layout: `hash[0:63]` to `0:63`,
      reserved `64:71`, `hash[64:108]` to `72:116` (45 bits), bit 117 carried
      from the primary (D2). `41978841be6d`
- [x] 2.3 Zero the digest with `memzero_explicit()` after each derivation.
      `41978841be6d`
- [x] 2.4 Guard the seed, its length and the cached derived UUID with
      `seed_lock`; keep the critical section to a copy (D6). `41978841be6d`
- [x] 2.5 Re-confirm `CUID_DEFAULT_SEED` is byte-identical to the library's
      `kDefaultSeed` (`"AMD-CUID-DEFAULT-SEED-v1"`, 24 bytes, no NUL), and add a
      comment naming it a placeholder pending T1. An unprovisioned `cuid_seed`
      reads back 24 bytes and the derived value equals D-1.

## 3. Remove the in-kernel auxiliary path (D4)

- [x] 3.1 Delete `cuid_aux_serial()` and the `<linux/dmi.h>` include it is the
      only user of.
- [x] 3.2 Delete `cuid_init_temporary()`, `CUID_TEMP_KEY`, the
      `temporary_uuid` field, `attr_temporary` and `cuid_temporary_show()`.
- [x] 3.3 Drop the `aux` parameter from the packing routine and the
      auxiliary-serial fallback from `cuid_sysfs_init()`; a device with no
      device-specific serial and no PCIe DSN must not get a synthesised one.
- [x] 3.4 Implement the no-serial behaviour: either skip registering the group,
      or register with the serial field zero. Do not emit an Auxiliary Value
      Identifier either way.
- [x] 3.5 Shrink the attribute array to four entries (three attributes plus the
      NULL terminator) and re-check the group teardown.
- [x] 3.6 Confirm the object no longer references any DMI symbol.

## 4. Sysfs attributes

- [x] 4.1 Create the three attributes as one group so they appear atomically,
      and gate teardown on a `registered` flag so a failed init does not try to
      remove them. `41978841be6d`
- [x] 4.2 `cuid_primary`: mode `0400`, `capable(CAP_SYS_ADMIN)` checked in the
      handler, `-EPERM` otherwise, emitted as `%pUb` with a newline (D7).
      `41978841be6d`
- [x] 4.3 `cuid_secondary`: mode `0444`, no privilege check, copy under the lock
      and format outside it. `41978841be6d`
- [x] 4.4 `cuid_seed`: mode `0600`, `CAP_SYS_ADMIN` on both show and store, raw
      bytes both directions, `-EINVAL` for any length other than 32 bytes,
      recompute the derived value before the write returns (D8). `41978841be6d`
- [x] 4.5 Log the primary-to-secondary association with `dev_info()` at bind and
      on every re-key (D9). `41978841be6d`
- [x] 4.6 Zero the seed on teardown and on the init failure path.
      `41978841be6d`

## 5. amdgpu integration

- [x] 5.1 Add `amdgpu_cuid.o` to the Makefile, embed `struct cuid` in
      `struct amdgpu_device`, and call init/fini from the device
      setup and teardown paths with component type GPU. `41978841be6d`
- [x] 5.2 Re-verify the fini path after task 3: unbind, module unload, and a
      probe that fails after CUID init, each without a leak or a stale kobject
      reference. No `kobject`, `refcount`, `WARNING`, `BUG:` or
      `sysfs: cannot create` line appears in `dmesg` across any of it.

## 6. Documentation

- [x] 6.1 Add `Documentation/ABI/testing/sysfs-driver-amdgpu-cuid` describing
      all attributes, their permissions, the placeholder default seed and the
      non-persistence of the provisioned seed. `41978841be6d`
- [x] 6.2 Remove the `cuid_temporary` entry and every reference to the auxiliary
      serial from the ABI file; state explicitly that a kernel-produced CUID
      never has the Auxiliary Value Identifier set.
- [x] 6.3 Reword the `cuid_primary` entry so it no longer describes a
      synthesised-serial fallback.

## 7. Verification

_Done. `cuid_kat` covers the format with no hardware; `cuid_sysfs_test.sh`
covers the interface and needs root and a publishing device. Both are in
`tools/testing/selftests/amdgpu`. Runtime numbers are from two Radeon Pro W6800s
at `0000:03:00.0` and `0000:63:00.0`._

- [x] 7.1 Regenerate the known-answer vectors for the final format and check the
      round-trip (payload to UUIDv8 to payload) for each.
- [x] 7.2 Re-run cross-layer verification on two GPUs. `cuid_primary` reads P-1
      and P-2, `cuid_secondary` reads D-1, and writing the 32-octet key `00..1f`
      gives D-2: the same four values the library computes independently in
      `cuidtstUnprivileged.ConformanceVectors`.
- [x] 7.3 Add a vector exercising a component type above 3 (NPU), so a regression
      on payload bits 120:121 fails a test. `T-NPU` in `cuid_vectors.txt`, plus a
      generator assertion that all sixteen types render distinctly.
- [x] 7.4 Test the permission matrix. Modes are 0400/0444/0600; an unprivileged
      read of `cuid_primary` is refused and of `cuid_secondary` allowed; writes
      of 1, 16, 24, 31, 33 and 64 bytes are all rejected with the derived value
      unchanged; 32 bytes is accepted and round-trips.
- [x] 7.5 Confirm a read of `cuid_secondary` concurrent with a seed write always
      returns a well-formed UUID from one seed or the other. 3840 reads against
      400 concurrent re-keys; none torn.
- [x] 7.6 Confirm the seed reverts to the default across a module reload and that
      no seed material appears in the log. `cuid_seed` reads 24 bytes again and
      the derived value is back to D-1; the provisioned seed's hex never appears
      in `dmesg`, and the association does, once per re-key.
- [x] 7.7 `checkpatch --strict` clean, `W=1` build clean, no new sparse warnings,
      all on `amdgpu_cuid.c`; `shellcheck` clean on the sysfs test.
- [x] 7.8 A zero-length write to `cuid_seed` cannot be rejected, because sysfs
      completes it without calling the store handler. The test asserts instead
      that the seed in use is unchanged, and the ABI file states the nuance.

## 8. Blocking on decisions outside this change

- [x] 8.1 The hypervisor opt-out (PSO, S1 comment 454362121): amdgpu gets its
      own `cuid` module parameter, default 1. A GIM-level disable only covers
      hosts running GIM, leaving bare metal and non-GIM hypervisors with no way
      to decline. Verified: `cuid=0` creates none of the three attributes and the
      GPU still enumerates; `cuid=1` restores them, derived value unchanged.
- [x] 8.2 Send the specification edits this change depends on: S1 primary table
      `112:117` to `112:116`, derived slice `hash[64:109]` to `hash[64:108]`,
      prose 110 to 109, and bit 117 confirmed as the auxiliary marker under
      uniform UUIDv8 (D2). Specified as `amend-published-cuid-spec` §1.
- [x] 8.3 Get an owner for the S3 device-type enumeration correction (T8).
      Specified as `amend-published-cuid-spec` §5 and §6.1.
- [x] 8.4 Ask for a worked example in S1: inputs, 122-bit payload hex, final
      UUID, as normative text. Specified as `amend-published-cuid-spec` §4.

## 9. Follow-ups, explicitly not in this change

- [x] 9.1 Shared cross-layer KAT wired into both trees' CI. `cuid_kat` checks
      `cuid_vectors.txt` against functions lifted verbatim from
      `amdgpu_cuid.c`, the library asserts it in
      `cuidtstUnprivileged.ConformanceVectors`, and the drift check runs in
      `cuid-workflow.yml`.
- [ ] 9.2 Node-wide seed scope (T4) and one keyed context re-deriving every
      registered device on write.
- [ ] 9.3 Seed persistence in a UEFI variable, once GUID, name and attributes
      are agreed with AINIC and NPU (T5).
- [ ] 9.4 `AMDGPU_INFO_PRIMARY_CUID` / `AMDGPU_INFO_SECONDARY_CUID` IOCTLs (T7).
- [ ] 9.5 Per-partition / XCP `unit_id`, and the SMBIOS/ACPI short-circuit for
      the Platform and CPU component types.
- [ ] 9.6 Extract the core into a standalone `amd_cuid.ko`, resolving MIT versus
      the GPL-only `hmac_sha256_usingrawkey` export at that point.
