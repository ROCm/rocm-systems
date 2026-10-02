## REMOVED Requirements

### Requirement: Canonical fallback seed

**Reason**: No fixed default key; see design.md.
**Migration**: Vectors keyed with `AMD-CUID-DEFAULT-SEED-v1` remain as test-key
vectors; no producer uses the constant.

### Requirement: Temporary and auxiliary fixed key

**Reason**: Auxiliary CUIDs are keyed with a machine-id-derived application key.
**Migration**: See `auxiliary-identifier`.
