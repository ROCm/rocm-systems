## 1. Public API

- [x] 1.1 Add `amdsmi_cuid_info_t` with the primary string, the derived string,
      the component type, the auxiliary flag, the source enum, and a reserved
      tail sized like the neighbouring `amdsmi_*_info_t` structs.
- [x] 1.2 Add `amdsmi_cuid_source_t` (`DRIVER`, `STORE`, `LIBRARY`, `UNKNOWN`)
      and `amdsmi_cuid_component_type_t` on the on-wire values.
- [x] 1.3 Add `amdsmi_get_gpu_cuid_info()`.
- [x] 1.4 Add `amdsmi_set_cuid_seed()` taking exactly 32 octets, privileged.
- [x] 1.5 Add `amdsmi_get_cuid_seed_info()` returning provisioned/default and the
      8-octet fingerprint, and nothing else.
- [x] 1.6 Document the legacy device UUID call as superseded, stating that it is
      not a CUID and that it changes on repartition.

## 2. Implementation

- [x] 2.1 Implement `amdsmi_get_gpu_cuid_info()` over
      `amdcuid_get_handle_by_bdf()` plus `amdcuid_query_device_property()` for
      the primary, the derived value, the device type and the temporary flag.
- [x] 2.2 Return an empty primary rather than failing when the primary query
      reports a permission error.
- [x] 2.3 Report the stage that answered via `AMDCUID_QUERY_SOURCE`. Record the
      source during derivation; a later lookup could observe different state.
      Retain the sysfs-existence fallback for older libraries without the query.
- [x] 2.4 Rewrite `amdsmi_get_gpu_device_cuid()` as a wrapper over 2.1.
- [x] 2.5 Implement `amdsmi_set_cuid_seed()` over `amdcuid_set_hash_key()`,
      propagating the store failure.
- [x] 2.6 Implement the fingerprint as the first 8 octets of unkeyed SHA-256 of
      the seed in use, computed inside the CUID library so the seed does not
      cross the ABI.
- [x] 2.7 Stub every entry point to `AMDSMI_STATUS_NOT_SUPPORTED` when built
      without the library, so the ABI is the same either way.

## 3. Build

- [x] 3.1 Support `BUILD_CUID=AUTO`, `ON` and `OFF`. Prefer the installed
      package, then the sibling `shared/cuid` source. Recompute availability on
      configure instead of caching the first result. If neither is available,
      `ON` fails and `AUTO` warns and disables CUID. Preserve legacy boolean
      spellings for explicit requests.
- [x] 3.2 Confirm the static library links with no new external dependency, and
      that the exported target does not reference the uninstalled `rocm-sha256`
      (A7).
- [x] 3.3 Confirm a tree with no `libamdcuid` still builds and links.
- [x] 3.4 Search `${ROCM_DIR}/core` for the `amdcuid` package through the
      overridable `AMDCUID_HINT_DIR`. CMake's default config search does not
      traverse the extra `core` prefix.
- [x] 3.5 Call `find_dependency(amdcuid)` from the installed
      `amd_smi-config.cmake`, gated on the build having linked it.
      `amd_smi_static` links `amdcuid::amdcuid` PRIVATE, and a private link
      dependency of a static library remains in its exported link interface.
      Resolve `Threads::Threads` there too. The in-tree build absorbs CUID
      objects instead and needs no external amdcuid package.

## 4. CLI and Python

- [x] 4.1 Add the derived CUID, component type, auxiliary flag and source to
      `amd-smi static`.
- [x] 4.2 Mark the fields unsupported rather than omitting them when there is no
      value.
- [x] 4.3 Gate the primary behind an explicit request and a privilege check.
- [x] 4.4 Add the seed provisioning command, reading from a file or stdin only.
- [x] 4.5 Report the seed's provisioned state and fingerprint. Folded into the
      `--cuid` block of `amd-smi static` rather than given its own command: it is
      the context the derived CUID printed beside it only means anything in.
- [x] 4.6 Carry every field through JSON and CSV output under stable names.
- [x] 4.7 Expose the same calls through the Python interface.

## 5. Tests

Cases live in `tests/amd_smi_test/unit/gpu/cuid_info_test.cc` and
`tests/python/unit/gpu/test_cli_cuid_seed.py`. Current release counts and
hardware limitations are in `shared/cuid/tests/QA_PLAN.md`. CLI provisioning
tests stub the library call; real privileged CUID tests require seed restoration.

- [x] 5.1 Snapshot call against a fake sysfs root: a driver-published value is
      returned verbatim and reported as driver-sourced
      (`CuidSourceIsDriverWhenTheAttributeIsPublished`). The driver-sourced half
      fabricates `cuid_derived` for each GPU's BDF under a temporary root and
      points `AMDSMI_CUID_SYSFS_ROOT` at it, and also asserts the negative. That
      override is compiled out unless the build sets
      `-DAMDSMI_CUID_TEST_SYSFS_OVERRIDE=ON`, because a variable that relocates
      the sysfs root lets any unprivileged user forge
      `AMDSMI_CUID_SOURCE_DRIVER`. The verbatim half compares the returned
      derived CUID against the real attribute, and skips where it is not real.
- [x] 5.2 Unprivileged path does not fail and returns a populated derived value
      (`CuidSnapshotIsSelfConsistent`). A non-root run asserts the primary *is*
      empty, a root run that it is populated and well-formed. The weaker "empty
      or well-formed" form would have passed a snapshot that handed an
      unprivileged process the serial-bearing primary.
- [x] 5.3 Auxiliary flag agrees with payload bit 117 of the returned derived
      CUID (`CuidSnapshotIsSelfConsistent`), and the decoder is pinned against
      the decoder's thirteen embedded vectors
      (`CuidAuxiliaryBitDecoderMatchesConformanceVectors`). The assertion read
      bit 7 of rendered octet 14, a VendorID bit, where the framing puts payload
      bit 117 in bit 7 of rendered octet **15**; against the vectors the old
      expression is wrong on A-1, A-2 and D-2.
- [x] 5.4 The single-string call and the snapshot call return the same string
      (`CuidSingleStringCallMatchesSnapshot`).
- [x] 5.5 A 16-octet and a 64-octet seed are both refused, with no state change
      (`TestCuidSeedLengthIsEnforced`, `TestCuidSeedLengthEnforcedInTheBinding`).
      Tested at the two layers that enforce it, not at the C entry point, which
      takes a fixed-width array and has no length to check. Both layers assert
      the refused seed never reached the library, and a 32-octet control is
      included, without which a command that refused everything would pass.
- [x] 5.6 The fingerprint of the canonical fallback seed is stable and matches
      the value the library computes
      (`CuidSeedFingerprintMatchesTheCanonicalFallbackSeed`). The expected value,
      `be8937fba7ed4e6f`, is the first 8 octets of the unkeyed SHA-256 of the
      24-octet public seed `AMD-CUID-DEFAULT-SEED-v1`, pinned as a literal rather
      than recomputed, so a producer that fingerprinted the wrong bytes fails
      instead of agreeing with itself. Both branches are asserted: an
      unprovisioned node must report exactly that value, a provisioned one must
      not.
- [x] 5.7 No seed octet appears in any output stream after provisioning
      (`TestCuidSeedNeverReachesOutput`). Drives the real CLI command with a
      distinctive 32-octet seed and the library call stubbed out, then asserts
      that neither the whole seed nor any 8-octet window of it appears, as raw
      bytes or hex, in stdout, stderr, or anything published to the logger, which
      is the complete input to every renderer. Provisioning is simulated, so this
      covers what `amd-smi` emits, not what the kernel or `libamdcuid` might log
      during a real re-key.
- [x] 5.8 A build without `libamdcuid` returns not-supported from every entry
      point (`CuidEntryPointsNotSupportedWithoutTheLibrary`). The cases could not
      tell a CUID-less build from a device the library has nothing to say about,
      so they skipped on `NOT_SUPPORTED` and asserted nothing.
      `tests/amd_smi_test/CMakeLists.txt` now passes `BUILD_CUID` to the test
      target, making the distinction available at compile time.
