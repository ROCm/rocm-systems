---
name: amdsmi-changelog-automation
description: "Check and generate changelog entries for amd-smi. Use when: reviewing PRs for changelog updates, generating release notes, checking CHANGELOG.md compliance."
---

# Changelog Automation — amd-smi

Verifies changelog entries exist for meaningful changes and helps generate them.

## Changelog Location

`CHANGELOG.md` in the workspace root. Entries are grouped under
`## amd_smi_lib for ROCm <MAJOR>.<MINOR>.<PATCH>` headings, **not** a
`## [Unreleased]` heading. The topmost heading is the release currently being
accumulated.

## Which Release Does an Entry Belong To?

**Never assume the topmost heading is correct for a new entry.** A ROCm release
is cut at a specific `rocm-systems` commit (the "release point"). Anything merged
**after** that commit belongs to the **next** release, even if the current top
heading still names the released version.

The release point is not a tag in `rocm-systems` — it is the `rocm-systems`
submodule commit that [ROCm/TheRock](https://github.com/ROCm/TheRock/releases)
pins for that release. Resolve it like this:

```bash
# 1. Find the rocm-systems submodule commit pinned by the TheRock release tag.
#    (therock-<ver> tags map 1:1 to ROCm releases.)
PIN=$(gh api repos/ROCm/TheRock/git/trees/therock-7.14 \
  | python3 -c "import sys,json;print(next(t['sha'] for t in json.load(sys.stdin)['tree'] if t['path']=='rocm-systems'))")

# 2. Make sure the pin is fetched locally.
git fetch origin "$PIN"

# 3. A commit belongs to 7.14 iff it is an ancestor of the pin; otherwise it
#    belongs to the next release (7.15).
git merge-base --is-ancestor <commit> "$PIN" && echo "7.14" || echo "7.15+"
```

To audit an entire section, blame its lines and classify each contributing
commit against the pin:

```bash
git blame -L <start>,<end> -s CHANGELOG.md | awk '{print $1}' | sort -u \
  | while read h; do h=${h#^};
      git merge-base --is-ancestor "$h" "$PIN" 2>/dev/null \
        || echo "AFTER-PIN (move to next release): $h"; done
```

Any `AFTER-PIN` commit's entry must move to a new `## amd_smi_lib for ROCm
<next-version>` section at the top of the file. Keep the same section headings
(Added / Changed / Removed / Resolved Issues) in the new release block.

## When a Changelog Entry is Required

The repo uses these section headings under each release (matching the ROCm
release-notes taxonomy): **Added**, **Changed**, **Removed**, **Optimized**,
**Resolved Issues**, **Upcoming Changes**, **Known Issues**. There is no `Fixed`
section — bug fixes go under **Resolved Issues**, deprecations under **Upcoming
Changes**.

| Change Type | Required? | Section |
|------------|-----------|---------|
| New public API function | ✅ Yes | Added |
| New CLI flag/subcommand | ✅ Yes | Added |
| Bug fix | ✅ Yes | Resolved Issues |
| Build system fix | ⚠️ If user-visible | Resolved Issues |
| Breaking API change | ✅ Yes | Changed (+ migration note) |
| Behavior/output change (non-fix) | ✅ Yes | Changed |
| Performance / tooling improvement | ✅ Yes | Optimized |
| Deprecation (still functional) | ✅ Yes | Upcoming Changes |
| Internal refactor (no behavior change) | ❌ No | — |
| Test-only changes | ❌ No | — |
| Documentation-only changes | ❌ No | — |
| Style/formatting-only changes | ❌ No | — |

## Entry Format

Entries follow [Keep a Changelog](https://keepachangelog.com/) conventions but
under ROCm-version headings, not `## [Unreleased]`:

```markdown
## amd_smi_lib for ROCm <MAJOR>.<MINOR>.<PATCH>

### Added
- **New `amdsmi_get_gpu_<feature>()` API for querying <feature>**.··

### Changed
- **`amdsmi_get_gpu_temperature()` now returns millidegrees (breaking change)**.··

### Optimized
- **Improved `amd-smi metric` refresh path**.··

### Resolved Issues
- **Fixed `amd-smi metric` crash when NIC device not present**.··

### Upcoming Changes
- **`amdsmi_get_gpu_vram_vendor()` is deprecated in favor of `amdsmi_get_gpu_vram_info()`**.··
```

### Rules
- One bullet per logical change (not per file)
- Start with the affected component: API name, CLI subcommand, or module
- For breaking changes: include migration guidance
- Bug fixes go under **Resolved Issues**, not a `Fixed` section (this repo has none)
- Reference the JIRA/issue only in the PR `JIRA ID` section, never in the entry text
- **Bolded headline bullets must end with two trailing spaces** (Markdown hard line break) so Sphinx renders the headline and its sub-bullets on separate lines. Example:
  ```markdown
  - **Fixed `amd-smi static` hang on gfx1153**.··
    - Added 60-second timeout to `amdsmi_init()`.
  ```
  (`··` = two literal trailing spaces.) Without them, Sphinx collapses the headline into the first sub-bullet. Verify with `grep -nP '^- \*\*.*\*\*[^ ]*$' CHANGELOG.md` — any match is missing the trailing whitespace.

## Review Checklist

When reviewing a PR, check:

- [ ] `CHANGELOG.md` updated if change is user-visible
- [ ] Entry is in the correct section (Added/Changed/Removed/Resolved Issues)
- [ ] Entry describes the **impact**, not the implementation
- [ ] Breaking changes have migration notes
- [ ] Entry is under the correct release heading — verify against the TheRock release pin, not just the topmost heading (see "Which Release Does an Entry Belong To?")
- [ ] Bolded headline bullets end with two trailing spaces (Sphinx hard break)

## Severity

| Finding | Severity |
|---------|----------|
| Missing changelog for new public API | ⚠️ IMPORTANT |
| Missing changelog for breaking change | ❌ BLOCKING |
| Missing changelog for bug fix | ⚠️ IMPORTANT |
| Changelog entry in wrong section | 💡 SUGGESTION |
| Changelog entry under the wrong release (merged after the release pin) | ⚠️ IMPORTANT |
| Missing changelog for internal-only change | Not a finding |
