# Go API coverage

The current package is a read-only GPU foundation. Full applicable public C API
coverage is the delivery target; inventory coverage is not implementation or
consumer-readiness coverage.

The [API inventory](https://github.com/ROCm/rocm-systems/blob/develop/projects/amdsmi/tests/go/api_coverage.json)
records every public function in `amdsmi.h`, including conditional CPU/ESMI
declarations. Each entry stores its normalized C signature, binding status, and
reason. Bound entries identify a public Go export and, when needed, an internal
helper. Deferred entries are implementation gaps, not permanent backend exceptions.

## Current scope

| Area | Status |
| --- | --- |
| GPU lifecycle, discovery, selected identity/telemetry, current partition metadata, ECC/RAS | Implemented foundation; see the [API reference](amdsmi-go-api.md) for field availability |
| PCIe information, power-management status, link metrics, nearest GPU topology, SOC P-state policies | Implemented read-only queries; native units and partial availability are preserved |
| Partition setters, supported-profile enumeration, and process-idle queries | Priority follow-up pending consumer requirements and handle/restart contracts |
| Remaining GPU queries and controls | Deferred for incremental coverage with native behavior and ownership tests |
| CPU, NIC, events, counters, remaining fabric and node APIs | Deferred for scoped lifecycle, ownership and consumer review |
| Unified BM/Host loading and package architecture | Not implemented; requires joint design and migration agreement |

The legacy Go shim remains unchanged. The foundation does not replace DRA/DCM
partition configuration or change the Exporter/GPU Agent integration.

## Coverage check

Run from the AMD SMI project root:

```bash
python3 -B tests/go/test_api_coverage.py
python3 -B -m unittest discover -s tests/go -p 'test_*.py' -v
```

The check fails on new, removed, changed, duplicate, or unparsed declarations;
unexplained inventory entries; stale Go mappings; and changes to transitive C
types or macros used by the binding. C parameter names and whitespace do not
change a signature. Conditional declarations are inventoried without dropping
disabled branches. Unsupported declaration syntax fails rather than being skipped.
Preprocessor directives and their positions are also recorded conservatively;
conditional or packing changes require inventory review even when no Go code changes.

The checked-in type fingerprints detect source drift, not binary ABI compatibility
with an arbitrary installed library. Updating a fingerprint requires reviewing the
layout, Go conversion, unavailable values, fixture and API documentation. A new
field may be explicitly deferred instead of exposed, but must not be silently lost.

Inventory updates are manual and reviewed; CI never refreshes its own expected
data. A new binding also needs behavioral tests and the agreed Host contract.
The inventory accounts for 262 C declarations: 37 bound and 225 deferred. The
separate 37-call scope check and 14-enum completeness check constrain the
implemented slice; complete accounting does not imply complete implementation.

## Partition delivery requirements

Before adding setters, agree the consumer API list and migration from deprecated
compute/memory calls to the Host-compatible profile/memory-mode APIs. Define
permission failures, unsupported modes, busy devices, partial failure, restart
requirements and handle invalidation. Supported-profile enumeration can write an
XCP selector on BM and must not be treated as an unrestricted read-only query.

Disruptive tests require approved hardware and recovery procedures. Final evidence
must identify the tested revision and preserve existing Host consumers.