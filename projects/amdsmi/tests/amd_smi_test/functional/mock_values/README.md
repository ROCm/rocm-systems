# Mock CPER fixtures

CPER (Common Platform Error Record) fixtures used by `functional/mock_cper.cc`
to exercise the CPER read path (`amdsmi_get_gpu_cper_entries_by_path`) without
GPU hardware.

## Provenance and scrubbing

These are **real GFX950 CPER captures** (RAS error-injection records,
`amdgpuras`), **sanitized** for a public repo by `sanitize_cper.py`. The raw
captures are intentionally not committed.

Kept (non-identifying, structural / public-spec data):

- CPER header: signature, revision, `signature_end`, `sec_cnt`,
  `error_severity`, valid bits, `record_length`
- `platform_id` `0x1002:0x75A8` (PCI vendor:device), `creator_id` `amdgpu`,
  `notify_type` GUID (CMC/MCE spec GUID), `record_id` index
- the section descriptor (section-type GUID, FRU text, section severity)

Scrubbed:

- `timestamp` -> fixed synthetic `2026-01-01T00:00:00` (removes the capture-time
  fingerprint, makes fixtures deterministic)
- the section data body `[section_offset, record_length)` -> zeroed. This is the
  injected RAS error payload (physical addresses, syndrome, register dumps,
  AFIDs) and is the only machine-specific content in the record. Zeroing also
  prevents any stray `CPER` signature in the raw payload from being mistaken for
  a second record when the fixtures are concatenated.

The read path under test only counts records and filters by severity, so the
zeroed body does not affect the tests. The `serial_number` was already `N/A` and
the FRU id already a generic `OAM0` in the captures.

## Files

| File | Records | Severity |
|------|---------|----------|
| `cper_corrected.cper` | 5 | non-fatal corrected (`AMDSMI_CPER_SEV_NON_FATAL_CORRECTED`, 2) |
| `cper_uncorrected.cper` | 1 | non-fatal uncorrected (`AMDSMI_CPER_SEV_NON_FATAL_UNCORRECTED`, 0) |
| `cper_fatal.cper` | 1 | fatal (`AMDSMI_CPER_SEV_FATAL`, 1) |
| `cper_mixed.cper` | 7 | 5 corrected + 1 uncorrected + 1 fatal |

## Regenerating

```
python3 sanitize_cper.py <raw_capture_dir>
```

where `<raw_capture_dir>` holds the raw `corrected-1..5.cper`,
`uncorrected-6.cper`, and `fatal-7.cper` captures. `sanitize_cper.py` documents
exactly which bytes are kept and scrubbed.
