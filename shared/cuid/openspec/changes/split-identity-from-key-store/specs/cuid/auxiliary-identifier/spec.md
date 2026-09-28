## MODIFIED Requirements

### Requirement: Auxiliary derivation carries bit 117

An auxiliary CUID SHALL be keyed with an application key
`K_app = HMAC-SHA256(key = the 16-octet machine-id, message = "AMD-CUID-TEMP-v2")`.
Its serial SHALL be the first 8 octets of `HMAC-SHA256(K_app, S)`, where `S` is
the 32-octet auxiliary input structure with its machine-id field zero-filled, and
its derived value SHALL be `HMAC-SHA256(K_app, raw_primary)` framed as UUIDv8
with bit 117 set. Without a machine-id no auxiliary CUID SHALL be produced.
This construction is provisional until the temporary-CUID specification is final.

#### Scenario: Image-baked machine-id

- **WHEN** two containers from one image share a baked-in machine-id
- **THEN** they produce identical auxiliary CUIDs, which the specification
  documents as a limitation of auxiliary identifiers
