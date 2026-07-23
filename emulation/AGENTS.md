# Emulation agent guide

These instructions apply to changes under `emulation/`. Read the nearest
component documentation before editing code.

## Authoritative guidance

- For rocjitsu C++, Python, and CMake, `rocjitsu/docs/style.md` is the final
  authority. If another document or nearby code disagrees with it, follow the
  style guide and call out the discrepancy.
- Read `rocjitsu/CONTRIBUTING.md` before changing rocjitsu architecture,
  generated ISA code, or tests.
- Read `mirage/docs/architecture.md` and `mirage/docs/building.md` before
  changing Mirage. Preserve crate boundaries and established Rust patterns.
- For dashboard work, also follow `mirage/dashboard/web/AGENTS.md`.

Prefer extending existing libraries over adding dependencies or duplicate
helpers. Subject to the rocjitsu style guide, keep documentation proportional
to the user-visible contract and non-obvious design decisions; do not add
comments that merely narrate code.

## Review and maintenance skills

Portable Agent Skills are cataloged in `.agents/skills/README.md`:

- `emulation-review` reviews a GitHub pull request, a branch, local changes, or
  named files.
- `rocjitsu-kernel-parity` compares rocjitsu behavior with Linux AMDGPU/KFD.
- `emulation-rebase` safely rebases a branch when some of its commits or
  equivalent changes have already landed.

Use these workflows when the request matches. They are read-only unless the
user explicitly asks for fixes or history changes. A review never posts to
GitHub on its own.

## GitHub write approval

Never push commits, branches, tags, rebased history, or other content to
GitHub—and never create or modify pull requests, issues, comments, reviews,
labels, releases, or other remote state—without the user's explicit approval
for that specific write. A request to review, edit, commit, rebase, prepare a
pull request, or follow a skill is not approval to publish. Prepare and validate
changes locally, show what would be written, and ask immediately before the
GitHub write. Approval for one write does not authorize later writes.

## Local reference material

On Linux development systems, agents may assume these optional sources exist:

- `~/linux` contains a Linux source checkout for public AMDGPU/KFD behavior.
- `~/reference/public/shader-programming-guides` contains public AMD shader
  programming guides, ISA manuals, and architecture documentation. Download
  public documents from the
  [AMD GPU architecture programming documentation](https://gpuopen.com/amd-gpu-architecture-programming-documentation/)
  page when local copies are useful.
- `~/reference/confidential` contains confidential reference PDFs.

Keep public and confidential documents in their respective subdirectories.
Confidential references and their contents may be consulted but must never be
quoted, named, linked, copied, committed, summarized, or exposed through paths,
metadata, screenshots, logs, prompts, generated artifacts, or internal
terminology. Do not upload them to external services or include their contents
in issues, pull requests, reviews, commits, tests, or chat. Findings must stand
on code in this repository, public sources, or reproducible tests. Treat
uncertain publication status as confidential.

## Validation baseline

Choose the smallest tests that exercise the changed behavior, then expand when
risk warrants it. Record every command run and distinguish failures from tests
that were unavailable.

Emulation builds and runs on Linux. The commands below use a POSIX shell; a
non-Linux host should run them in the project's Linux development environment
rather than inventing an unsupported native workflow.

### rocjitsu

From `emulation/rocjitsu`:

```sh
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
cd lib/python && python -m pytest amdisa/tests/ -x
```

Run relevant sanitizer or clang-tidy configurations for concurrency, memory,
ABI, or lifetime-sensitive changes. ISA changes must update the amdisa
generator, regenerate every affected ISA with multi-ISA mode, and include all
resulting generated files; never hand-edit generated output.

### Mirage

From `emulation/mirage`:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
```

For dashboard changes, from `emulation/mirage/dashboard/web`:

```sh
npm run lint
npm run build
npm test
npm run test:e2e
```

Rust formatting and lint behavior are defined by `cargo fmt`, `cargo clippy`,
and established code in the owning crate. Dashboard formatting and lint
behavior are defined by its TypeScript and ESLint configuration; Rete.js work
also follows `mirage/dashboard/web/AGENTS.md`.

Mirage changes that affect emulator discovery, profiles, session lifecycle,
environment wiring, FFI, or rocjitsu integration need an integration test in
addition to unit tests. Integration tests live in `mirage/tests/`; use the
existing lifecycle, daemon, container, or matrix E2E suite that owns the
behavior. Do not omit Mirage validation merely because most of a change is in
rocjitsu.

## Review quality bar

- Read complete changed files and the owning, calling, cleanup, and test paths.
- Prioritize correctness, races, deadlocks, ownership, ABI compatibility,
  generated-code integrity, and cross-component behavior over cosmetic issues.
- Require a concrete mechanism and evidence for every finding. State
  uncertainty instead of converting suspicion into a defect.
- Report findings by severity with exact file and line, impact, evidence, and a
  minimal suggested fix or test. Deduplicate overlapping findings.
- Robust code includes focused negative, boundary, failure, cleanup, and
  concurrency tests where applicable. More tests are not automatically better;
  each test should protect a meaningful behavior.
