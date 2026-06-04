# RocFuzz AFL++ Integration

RocFuzz is the planned AFL++ integration for rocjitsu device-code
instrumentation. This first slice only wires the fuzzer directory and a small
host-side AFL++ smoke test so reviewers can validate the AFL toolchain before
any ROCm runtime or DBI pieces are introduced.

The later preload and device-coverage pieces build on this scaffold.
