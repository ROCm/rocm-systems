## Purpose

The two key values every derived CUID depends on, as literal octets rather than
descriptions, and the length, scope and lifecycle rules for an
administrator-provisioned seed. A producer that gets any of these wrong emits
values that look correct and name nothing.

## ADDED Requirements

### Requirement: Canonical fallback seed

Until an administrator provisions a secret, a producer of a derived CUID SHALL
key the derivation with the canonical fallback seed, which is exactly the
24-octet ASCII string:

```
41 4D 44 2D 43 55 49 44 2D 44 45 46 41 55 4C 54 2D 53 45 45 44 2D 76 31
"A  M  D  -  C  U  I  D  -  D  E  F  A  U  L  T  -  S  E  E  D  -  v  1"
```

The seed SHALL NOT include a terminating NUL octet and SHALL NOT be padded to
any other length. HMAC-SHA256 pads a key to its own 64-octet block size
internally; padding to 32 octets is not an HMAC operation but a different key,
and would change every value already emitted for no cryptographic benefit.

The fallback seed is public. A derived CUID produced under it is stable and
reproducible but SHALL NOT be treated as fleet-unique, and a producer SHALL
document it as a placeholder.

#### Scenario: An unprovisioned kernel and an unprovisioned library agree

- **WHEN** the kernel driver and the userspace library are asked for the derived
  CUID of the same component on a machine with no provisioned seed
- **THEN** both key HMAC-SHA256 with the same 24 octets
- **AND** both emit the same value

#### Scenario: The seed is not NUL-terminated or padded

- **WHEN** a producer's key material is inspected
- **THEN** it is 24 octets long
- **AND** its last octet is `0x31`, not `0x00`

### Requirement: Temporary and auxiliary fixed key

An auxiliary CUID's derived value SHALL be keyed with the temporary fixed key,
which is exactly the 20-octet ASCII string:

```
41 4D 44 2D 43 55 49 44 2D 54 45 4D 50 2D 4B 45 59 2D 76 31
"A  M  D  -  C  U  I  D  -  T  E  M  P  -  K  E  Y  -  v  1"
```

No NUL octet, no padding, for the reason given above.

The key is public by construction: an auxiliary CUID is built from
non-privileged information precisely so that unprivileged and out-of-band
consumers can reproduce it, which they cannot do with a secret.

The derivation SHALL use the same operand order as every other CUID derivation:
`HMAC-SHA256(key = the temporary fixed key, message = the 16 auxiliary primary
octets)`. A producer SHALL NOT exchange the operands, and SHALL NOT substitute a
fixed constant for the message.

#### Scenario: Operand order matches the primary derivation

- **WHEN** an auxiliary derived CUID is computed
- **THEN** the key is the 20-octet fixed key and the message is the auxiliary
  primary payload
- **AND** a consumer can verify it with `openssl dgst -sha256 -mac HMAC -macopt
  key:AMD-CUID-TEMP-KEY-v1` over those 16 octets

*One producer currently keys with the primary payload and passes a fixed
application UUID as the message. With a public key HMAC is a keyed hash whose
preimage resistance protects the message either way, so exchanging the operands
buys nothing and costs a second derivation function.*

### Requirement: Provisioned seed length

A provisioned seed SHALL be exactly 32 octets. A producer SHALL reject any
attempt to provision a seed of any other length, and SHALL continue to use the
previously effective seed rather than adopting a truncated or padded one.

A wrong-sized seed is corruption, not absence: silently accepting it changes
every derived CUID on the machine, and the change is undetectable from the
values themselves.

The canonical fallback seed is exempt from this rule. It is a built-in
placeholder, not a provisioned secret, and its 24 octets are already fixed above.

#### Scenario: A short seed is refused

- **WHEN** an administrator provisions a 16-octet seed
- **THEN** the producer reports an error
- **AND** the derived CUID is unchanged from before the attempt

#### Scenario: A 32-octet seed is accepted

- **WHEN** an administrator provisions 32 octets of secret material
- **THEN** the derived CUID for every component is recomputed under it

### Requirement: Seed scope

The seed SHALL be node-wide: one seed shared by every component and every
participating producer on a node, not one per device.

A per-device seed makes the derived CUIDs of two components on the same node
incomparable, requires provisioning once per component, and cannot be moved to a
node-wide scope later without changing every value.

#### Scenario: One provisioning covers every component

- **WHEN** an administrator provisions a seed once on a node
- **THEN** every component on that node whose derived CUID is keyed with the seed
  reports one keyed with it
- **AND** an auxiliary component is excepted, its derived CUID being keyed with
  the fixed public temporary key rather than with the seed

#### Scenario: Two components on one node share a seed

- **WHEN** two GPUs on the same node are queried
- **THEN** their derived CUIDs differ only because their primary payloads differ

### Requirement: Re-keying is an administrative invalidation

Given a persistent seed, a component's primary and derived CUIDs SHALL be
persistent by construction. Changing the seed SHALL therefore be understood as
an administrative invalidation of every seed-derived CUID previously handed out,
not as a routine operation. Derived values keyed with the fixed public temporary
key `AMD-CUID-TEMP-KEY-v1`, which is the auxiliary case, are unaffected by
definition: the seed is not an input to them.

A producer SHALL record the association between the primary and the derived
identifier whenever that association changes, so that a derived CUID recorded
before a re-key can still be traced to the component it named.

#### Scenario: Re-keying changes every seed-derived value

- **WHEN** the seed is replaced
- **THEN** every derived CUID on the node that was keyed with the seed changes
- **AND** the primary CUIDs are unchanged
- **AND** a derived CUID keyed with the fixed public temporary key
  `AMD-CUID-TEMP-KEY-v1`, which is every auxiliary component's, is unchanged,
  because the seed is not one of its inputs

#### Scenario: The association survives the re-key

- **WHEN** the seed is replaced
- **THEN** the producer records both the primary and the new derived identifier
  with a timestamp
