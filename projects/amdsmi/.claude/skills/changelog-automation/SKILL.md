---
name: changelog-automation
description: "Check and generate changelog entries for amd-smi. Use when: reviewing PRs for changelog updates, generating release notes, checking CHANGELOG.md compliance."
---

# Changelog Automation — amd-smi

Verifies changelog entries exist for meaningful changes and helps generate them.

## Changelog Location

`CHANGELOG.md` in the workspace root.

## When a Changelog Entry is Required

| Change Type | Required? | Section |
|------------|-----------|---------|
| New public API function | ✅ Yes | Added |
| Bug fix | ✅ Yes | Fixed |
| Breaking API change | ✅ Yes | Changed (+ migration note) |
| Performance improvement | ✅ Yes | Changed |
| New CLI flag/subcommand | ✅ Yes | Added |
| Build system fix | ⚠️ If user-visible | Fixed |
| Internal refactor (no behavior change) | ❌ No | — |
| Test-only changes | ❌ No | — |
| Documentation-only changes | ❌ No | — |
| Style/formatting-only changes | ❌ No | — |

## Entry Format

Follow [Keep a Changelog](https://keepachangelog.com/) format:

```markdown
## [Unreleased]

### Added
- New `amdsmi_get_gpu_<feature>()` API for querying <feature>

### Fixed
- Fixed `amd-smi metric` crash when NIC device not present

### Changed
- `amdsmi_get_gpu_temperature()` now returns temperature in millidegrees (breaking change)
```

### Rules
- One bullet per logical change (not per file)
- Start with the affected component: API name, CLI subcommand, or module
- For breaking changes: include migration guidance
- Use past tense for fixes ("Fixed"), present tense for additions ("New")
- Reference the JIRA/issue if available: `[SWDEV-XXXXXX]`

## Review Checklist

When reviewing a PR, check:

- [ ] `CHANGELOG.md` updated if change is user-visible
- [ ] Entry is in the correct section (Added/Fixed/Changed/Removed)
- [ ] Entry describes the **impact**, not the implementation
- [ ] Breaking changes have migration notes
- [ ] Entry is under `## [Unreleased]` (not a version number)

## Severity

| Finding | Severity |
|---------|----------|
| Missing changelog for new public API | ⚠️ IMPORTANT |
| Missing changelog for breaking change | ❌ BLOCKING |
| Missing changelog for bug fix | ⚠️ IMPORTANT |
| Changelog entry in wrong section | 💡 SUGGESTION |
| Missing changelog for internal-only change | Not a finding |
