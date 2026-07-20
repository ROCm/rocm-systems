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
