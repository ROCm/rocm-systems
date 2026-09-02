# Conflict register

Every place the published specification contradicts itself, contradicts a
standard it cites, or leaves a value an implementer must invent.

`specs/cuid/` records each one where it occurs, marked
`Recorded contradiction` / `Recorded defect` / `Recorded gap`. Every one is
resolved in `changes/amend-published-cuid-spec/`.

**Recording is not correcting.** `specs/cuid/` is the published text as
published: at each marked site it still states the broken rule, and the marker
only says that the rule is broken and points elsewhere. The corrected rule lives
solely in `changes/amend-published-cuid-spec/`, so a marker is an instruction to
go and get the resolution, not a claim that the surrounding text has been fixed.

`openspec/check_conflict_register.py` checks the *bookkeeping* of that mapping:
that every labelled marker has a row here, that every row names a
non-placeholder resolution, and that a short list of named constants is stated
consistently across the change dirs. It does not read the specs for meaning and
does not check that a resolution was implemented. A clean run is not a claim
that the corpus is consistent.

Source: "Persistent platform component identification for SW tools", version 84.

| ID | Conflict | Recorded in | Resolution |
|---|---|---|---|
| C1 | A firmware-supplied UUID is "used directly" as the PrimaryID, but every CUID must be a UUIDv8. An SMBIOS system UUID is version 1, 3 or 4, never 8. | `primary-identifier` | Two constructions, distinguished by the version nibble: **constructed** (payload, always v8, decodable) and **adopted** (firmware UUID verbatim, opaque). Consumers must not reject on the nibble. |
| C2 | The HMAC message is "the primary ID 122bit wide value". 122 bits is not a whole number of octets, so this does not name a byte string. | `derived-identifier` | The message is the sixteen packed octets, LSB first, padding bits present and zero. |
| C3 | The auxiliary serial must be "64bit wide" and the input structure is 256 bits; the reduction is never stated. | `auxiliary-fallback` | First 8 octets of unkeyed SHA-256 over the structure, little-endian. |
| C4 | The auxiliary input is described twice and incompatibly: an unordered field list with an `amd.com` namespace, and a fixed-width 256-bit structure. | `auxiliary-fallback` | The structure is the sole input. No namespace, no field list. |
| C5 | A DerivedID must be "constant for the lifetime of the lesser-privileged SW context", and is also a function of a salt that may change. | `identifier-model` | Constant **while the salt is unchanged**; re-keying is an administrative invalidation, and the association is recorded. |
| C6 | Scalability requires no collision "anywhere"; the fallback section says auxiliary inputs are "not guaranteed to be unique across cluster nodes". | `identifier-model` | The guarantee is scoped to canonical CUIDs. Auxiliary values are marked by bit 117 so a consumer can tell before relying on one. |
| C7 | The collision bound of 1/(2^(122-1)) assumes 122 varying bits; the derived payload carries at most 109. | `derived-identifier` | Bound restated over the 109 hash bits. No value changes. |
| C8 | The auxiliary Component Type field admits only GPU, NIC and NPU, while the primary layout defines sixteen types. | `auxiliary-fallback` | Full on-wire enumeration; the fallback is not a property of component type. |
| C9 | The CPU auxiliary table renames bits `200:215` from VendorID to FamilyID and splits Family/Model, disagreeing with both the PCIe structure and the primary layout. | `auxiliary-fallback` | One structure, one meaning per field. Device holds Family+Model combined, Vendor holds the vendor. |
| C10 | Bit 117 is UnitID part 2 in the primary table and the Auxiliary Value Identifier in the derived table. | `primary-identifier`, `derived-identifier` | UnitID narrows to `112:116`, 13 bits total. Bit 117 is the auxiliary marker in both layouts. |
| C11 | The derived hash slot is 45 bits wide, labelled `hash[64:109]` (46), inside prose saying 110. | `derived-identifier` | `hash[64:108]`, 45 bits, 109 hash bits total. The slot width was already right. |
| C12 | Auxiliary CUIDs are specified as UUIDv5 with an `amd.com` namespace, using HMAC-SHA-256 in place of SHA-1, which is not a conforming UUIDv5. | `auxiliary-fallback` | Withdrawn. Uniform UUIDv8 distinguished solely by bit 117. |
| C13 | The auxiliary input structure gives Format 17 bits and Machine ID 127; neither is a whole number of octets and the Machine ID cannot hold `/etc/machine-id`. | `auxiliary-fallback` | Format `0:15`, Machine ID `16:143`. |
| C14 | The fallback prose calls the Linux Machine ID a "32bit MachineID". `/etc/machine-id` is 128 bits. | `auxiliary-fallback` | Corrected to 128 bits. |
| C15 | The CPU auxiliary structure retains a PCIe Routing ID field for a component with no Bus/Device/Function. | `auxiliary-fallback` | Zero. |
| C16 | A constant seed key is permitted for auxiliary derivation but never given. | `auxiliary-fallback` | `AMD-CUID-TEMP-KEY-v1`, 20 ASCII octets, unpadded, key not message. |
| C17 | No canonical fallback seed is named, so an unprovisioned machine's derived CUID is undefined. | n/a | `AMD-CUID-DEFAULT-SEED-v1`, 24 ASCII octets, unpadded. |
| C18 | The PCIe Device Serial Number's byte order is never stated. | `component-discovery` | Configuration-space order, little-endian, unswapped, from `dsn_cap_offset + 4`. |
| C19 | The NIC MAC fallback is permitted but its orientation is never stated. | `component-discovery` | Octet 0 at payload bits 0:7; an all-zero address is absent. |

C10, C12, C13 and C18 each produced a wrong value in shipped code. C1 produced
three: a Platform reported as an NPU, a firmware identity reported as
synthesised, and two platforms differing only in version bits deriving the same
secondary CUID.

## Open: needs a human decision, not registered above

Places where this corpus and the shipped code disagree and nobody has yet
decided which is right. They are deliberately not given a `Cn` row, because a
`Cn` row means "recorded, and resolved over there".
`check_conflict_register.py` does not look at this section.

### O1: UnitID for an SR-IOV Virtual Function

Three positions, all currently in the tree:

| Position | Where | What it says |
|---|---|---|
| Baseline blesses a non-zero UnitID for a VF | `specs/cuid/primary-identifier/spec.md:76-80`, scenario "A subdivided function" | "**WHEN** a driver names a spatial partition or a Virtual Function of a physical device / **THEN** it assigns a non-zero UnitID rooted in the parent device definition" |
| The delta forbids it | `changes/pin-cuid-cross-layer-contract/specs/cuid/component-sources/spec.md:158-163`, "UnitID identifies a sub-unit, not a location" | UnitID "SHALL NOT carry a bus address, an **enumeration index**, or any other property of where the component is" |
| The shipped library does it | `lib/src/cuid_gpu.cc:200-204` | `info.header.fields.gpu.unit_id = CuidUtilities::get_gpu_vf_id(device_path);`, "For VFs, unit_id is the 1-based VF index" |

A fourth data point: the kernel hardcodes UnitID to 0, so a VF's
driver-published primary and the library's computed primary for the same VF do
not agree.

Why it matters:

* The delta **silently narrows the baseline**. It carries no "MODIFIED" or
  "REMOVED" marker against the baseline scenario it contradicts, so a reader of
  either document alone will not notice the other.
* A 1-based VF index is exactly an enumeration index. On the delta's reading the
  shipped library is non-conforming; on the baseline's reading it is doing the
  required thing and the delta is wrong to have narrowed.
* Deciding it changes values. If the delta wins, every VF's primary CUID (and
  therefore its derived CUID) changes, on nodes that have already recorded them.
  If the baseline wins, the delta's requirement has to be amended and the
  kernel's hardcoded 0 becomes the divergence instead.

The code was deliberately not changed. Resolving this needs someone who can say
what a VF's UnitID is *for*, and who can accept the value churn on whichever
side loses.

**A way through, for whoever decides.** The two positions are not actually
about the same field. What the delta forbids is a value that says *where a
component is*; what the baseline wants is a value that says *which sub-unit of
a parent this is*. A VF index is both, which is why the two readings collide.
Separating them resolves it without either document being simply wrong:

* UnitID names a sub-unit **within a parent identity**, and is meaningful only
  beside the parent's serial, which the primary already carries. A VF's index
  is not a bus address; it is which of the parent's functions this is. On that
  reading the baseline governs, the delta's prohibition is about bus addresses
  and enumeration order across *devices*, and the delta needs one sentence
  saying so rather than a reversal.
* The kernel's hardcoded 0 then becomes the divergence, and it is the cheap
  side to change: no value that has ever been recorded changes, because the
  kernel does not currently publish a VF primary that anyone could have
  recorded.

Choosing the other way -- delta governs, library changes -- moves every VF's
primary and derived CUID on every node that has already recorded them. That is
the expensive direction, and it should not be taken by default just because the
delta is the newer document.

Whichever is chosen, the losing document needs an explicit `MODIFIED` marker.
The reason this sat undetected is that the delta narrowed a baseline `SHALL`
with no marker at all.

### O2: an unprivileged `amd-smi` reports no CUID, with the value in sysfs — resolved for tier 2, open for tier 1

`amdsmi_get_gpu_cuid_info()` resolves a GPU through
`amdcuid_get_handle_by_bdf()`. Resolving a single device makes the library
discover it; discovery reads PCIe configuration space; an unprivileged process
is routed to the daemon instead, which is not running on a normal node. So an
ordinary user gets `AMDSMI_STATUS_NOT_SUPPORTED` for a GPU whose derived CUID
the kernel is publishing world-readable in
`/sys/bus/pci/devices/<bdf>/cuid_secondary`.

The staged lookup is specified driver → store → library, and
`amdsmi-identity-api` reports a `source` of `driver`. But the driver stage is
read for the *label* only: `cuid_source_for()` tests whether the attribute
exists and nothing reads its contents. So the one stage that needs no privilege
is the one that cannot answer.

Resolution: where the library cannot answer, read `cuid_secondary`, report it
with `AMDSMI_CUID_SOURCE_DRIVER`, and take `auxiliary` from bit 117 of that
value. It strictly widens the set of cases that answer and changes no answer
that already works. It is not free of judgement, because of O3: after a driver
reload the kernel's value is keyed with the public default while the library
holds the provisioned secret, so the fallback must arrive together with a
mismatch diagnostic rather than on its own.

**Tier 2 is done.** On a node whose driver publishes nothing, an unprivileged
caller now gets a full answer once the record store exists, because the
auxiliary flag no longer needs the privileged primary -- bit 117 is in the
derived value, which is what that bit is for -- and because the first
privileged lookup records what it computed. Before, an ordinary user got
NO_PERM with the store present and NOT_SUPPORTED without it, i.e. nothing, ever.

**Tier 1 is still open**, and is the narrower case it always was: on a node
whose driver *does* publish, an unprivileged caller with no record store still
gets nothing from amd-smi, while `cuid_secondary` sits in sysfs at 0444 a few
bytes away. The resolution above stands for that case -- read the attribute
when the library cannot answer, report `AMDSMI_CUID_SOURCE_DRIVER`, take
`auxiliary` from bit 117 -- and still wants the mismatch diagnostic O3 argues
for, because after a driver reload the kernel's value is keyed with the public
default while the library holds the provisioned secret.

Needs a delta against `integrate-cuid-into-amdsmi` for the tier-1 half.

### O3: a provisioned kernel seed cannot be withdrawn, and withdraws itself

`cuid_seed_store()` accepts exactly 32 octets and sets `seed_len`;
`cuid_seed_show()` returns the 24-octet public default only while `seed_len` is
zero. There is no encoding for "back to the default", so a seed provisioned by
mistake can be replaced but not withdrawn, other than by unbinding or reloading
the driver.

And a driver reload withdraws it silently. Measured on a two-W6800 node, with
no configuration change and no administrator action between the two readings:

```
                       0000:03:00.0                          0000:63:00.0
before   4caa5df4-ea63-875f-8401-114ed88da870   16311585-f661-83f2-8003-78f23fb00c60
  modprobe -r amdgpu && modprobe amdgpu
after    3c6f0f16-d99c-8013-a801-e8214acd0848   61ffe99a-b3e0-8e16-a802-4b1d515d5438
```

Across both readings `amd-smi static --cuid` reported `SEED_PROVISIONED: True`
and the same fingerprint `61372d27e9764ead`, and reported the new values
without qualification. So every persistent identifier on the node changed, the
one field that exists to tell an operator the seed state said nothing had, and
`amd-smi` faithfully published the public-seed values as though they were the
provisioned ones.

A driver reload is routine: a driver package update does it, and so does
`amdgpu-install`. A fleet inventory keyed on these identifiers loses every GPU
on the node each time, with nothing in any output to attribute it.

The library's key store is untouched throughout, which is why the fingerprint
stays right while the values do not: the two are read from different places and
nothing compares them.

`key-constants` calls the seed node-wide; `sysfs-interface` makes it per-device
and lost on reload. Those two cannot both be true, and the second is what the
kernel does. Either the seed's scope is restated as per-device with a named
mechanism that re-publishes it after a reload -- `publish_seed_to_drivers()`
exists and runs at provisioning time only, and appears in no spec -- or the
kernel gains a way to say "unprovisioned" and something re-asserts the node
seed on device add.

Not registered as `Cn` because it is not a defect in the published
specification; it is two of this corpus's own documents disagreeing about a
value that has already shipped in both trees.

The cheapest thing that would make it visible rather than silent, short of
deciding the scope question: have `amd-smi` compare the fingerprint of the seed
the library holds against the seed the driver is using, and say so when they
differ. The driver already exposes `cuid_seed` to a `CAP_SYS_ADMIN` reader, so
the comparison needs no new kernel interface. That is a diagnostic, not a fix,
and it should not be mistaken for one -- but it converts a silent identity
change into a reported one, which is the difference between an inventory that
is wrong and an inventory that knows it is.

### O4: what the library is supposed to enumerate is unstated

`libamdcuid` enumerates every DRM card node as a GPU regardless of vendor. On a
node with two W6800s it also minted a GPU CUID for the ASPEED BMC display
controller (`1a03:2000`, driver `ast`). Nothing filters by vendor and nothing in
`component-discovery` says whether anything should.

It may well be intended -- the Component Type enumeration has storage, memory
and generic PCIe in it, which only makes sense for a whole-platform inventory --
but it is not written down, and it has a visible consequence: `amdcuid_tool
--list` and `amd-smi static --cuid` report different device sets on the same
machine, and that device appears in a root enumeration and not in an
unprivileged one, because it has no unprivileged fingerprint source.

Resolution needs one sentence in `component-discovery` saying whether discovery
is AMD components or platform components. Either answer is defensible; the
absence is not.
