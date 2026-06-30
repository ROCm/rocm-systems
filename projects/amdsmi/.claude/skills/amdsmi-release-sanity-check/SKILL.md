---
name: amdsmi-release-sanity-check
description: "Use when auditing amd-smi for a release or release candidate, sanity-checking everything that changed since the last ROCm/TheRock release, verifying a version bump, or before merging/cutting a release branch. Catches API cascade gaps (header/wrapper present but Python interface/__init__ missing), semantic breaks that survive a clean rebase/merge, stale-image wrapper regeneration, and silent CHANGELOG/doc drift."
---

# Release Sanity Check — amd-smi

Audit everything that changed since the last release for the failure modes that
pass CI and a clean `git merge` but still ship broken. Clean merge ≠ correct.

**Core principle:** Diff against the *release boundary*, not against `develop`'s
tip. A textually clean rebase can be semantically broken.

## When to Use

- Cutting or reviewing a release / RC branch, or a version bump
- "Sanity check what changed since the last release"
- After a rebase that touched header, wrapper, or Python interface files

**Don't use when:** reviewing a single small PR (use `amdsmi-changelog-automation`
+ the `project-layout` rule). This skill is release-scope.

## 1. Establish the Release Boundary

Find the last amd-smi shipped via TheRock
(`https://github.com/ROCm/TheRock/releases`) — note its `develop` SHA. The audit
set is `git log <release-sha>..origin/develop -- .` from `projects/amdsmi`.
Confirm `AMDSMI_LIB_VERSION_*` in `include/amd_smi/amdsmi.h` moved if behavior
changed.

This SHA scopes the *feature / changelog* audit. For an **ABI** question
specifically (is a symbol rename or removal a real break?), the authoritative
freeze point is the last `amdsmi_pkg_ver-*` package tag, not the TheRock SHA —
see the architecture agent's *ABI Release Baseline*.

## 2. The Sanity Sweep

| Check | Command / method | Red flag |
|-------|------------------|----------|
| **Cascade gap** | For each new `amdsmi_*` in the header, grep all 5 layers (header → `amd_smi.cc` → `amdsmi_wrapper.py` → `amdsmi_interface.py` → `__init__.py`) | Present in header/wrapper, **absent** in `amdsmi_interface.py`/`__init__.py` → unreachable from `from amdsmi import ...` |
| **Semantic rename break** | Static-resolve every `from .amdsmi_interface import X` in `__init__.py` against functions actually defined there; grep docs/changelog for the old name | Export/doc references a name a rebase renamed elsewhere — zero git conflict, runtime `ImportError` |
| **Wrapper drift** | Regenerate with `tools/update_wrapper.sh` (Python) using the **canonical pinned Docker image**, never a stale local one | Diff vs committed wrapper ≠ ∅. Cross-check that Python and Rust wrappers agree with the header before trusting either |
| **CHANGELOG conventions** | `grep -nP '^- \*\*.*\*\*[^ ]*$' CHANGELOG.md` in the unreleased section | Missing two-trailing-space Sphinx hard break; entry in wrong section (see `amdsmi-changelog-automation`) |
| **Doc vs code** | New public API or CLI flag without a docs/reference entry | User-facing change, zero docs |
| **Version bump** | Behavior changed but `AMDSMI_LIB_VERSION_*` unchanged | Stale version |

> **Wrapper trap:** `update_wrapper.sh` tags its image by Dockerfile hash. A
> stale local image silently emits spurious bindings; the pinned canonical image
> reproduces the committed wrapper byte-for-byte. Non-empty regen diff → suspect
> the image before the header.

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Diffing against `develop` tip, not the release SHA | Anchor on TheRock's released SHA |
| Trusting a conflict-free rebase | Resolve every export/doc name against the post-rebase interface |
| Committing a regen from a stale image | Use the pinned image; reproduce the committed wrapper first |
| "It imports for me" as proof | Installed package may differ from the tree — static-check the source |

---

*Validation status: grounded in an observed release miss (a cascade gap plus a
rename break that passed a clean merge). A formal RED/GREEN pressure test per the
`writing-skills` skill is still pending.*
