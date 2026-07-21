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

**A release's changelog is the diff between two consecutive ROCm release
points.** Everything merged after release `N`'s point and up to release `N+1`'s
point belongs to `N+1` — regardless of which heading currently sits at the top
of the file. Never assume the topmost heading is correct for a new entry.

A release point is not a tag in `rocm-systems`; it is the `rocm-systems`
submodule commit that [ROCm/TheRock](https://github.com/ROCm/TheRock/releases)
pins for that release (the `therock-<ver>` tags map 1:1 to ROCm releases).

Resolve the two bracketing pins dynamically — do not hard-code versions:

```bash
# PREV = the last shipped release, CURR = the one being accumulated.
# List tags with: gh release list --repo ROCm/TheRock
PREV_TAG=therock-<prev>     # e.g. the newest tag that is already released
CURR_TAG=therock-<curr>     # the next tag, if it exists yet

pin() {  # resolve the rocm-systems submodule commit pinned by a TheRock tag
  gh api "repos/ROCm/TheRock/git/trees/$1" \
    | python3 -c "import sys,json;print(next(t['sha'] for t in json.load(sys.stdin)['tree'] if t['path']=='rocm-systems'))"
}

PREV_PIN=$(pin "$PREV_TAG")
git fetch origin "$PREV_PIN"
# If CURR_TAG is not tagged yet, the current release is still open: use HEAD
# (origin/develop) as its upper bound instead of a pin.
CURR_PIN=$(pin "$CURR_TAG" 2>/dev/null || echo origin/develop)
git fetch origin "$CURR_PIN" 2>/dev/null || true
```

The set of commits that belong to the release being accumulated is exactly the
range `PREV_PIN..CURR_PIN`:

```bash
git log --oneline "$PREV_PIN..$CURR_PIN"     # everything that must be documented
```

## Verifying an Entry Lands in the Right Release

Once the bracketing pins are known, classify any commit with
`git merge-base --is-ancestor` — the same check the PR-history walk uses:

```bash
# Ancestor of PREV_PIN  -> already shipped, belongs to a PRIOR release
# Otherwise             -> belongs to the release currently being accumulated
git merge-base --is-ancestor <commit> "$PREV_PIN" && echo "prior release" || echo "current release"
```

To audit an entire changelog section, blame its lines and flag any contributing
commit that is *not* newer than the previous release point (i.e. was already
shipped and is filed under the wrong heading):

```bash
git blame -L <start>,<end> -s CHANGELOG.md | awk '{print $1}' | sort -u \
  | while read h; do h=${h#^};
      git merge-base --is-ancestor "$h" "$PREV_PIN" 2>/dev/null \
        && echo "ALREADY-SHIPPED (belongs to a prior release): $h"; done
```

Any flagged commit's entry must move to the correct
`## amd_smi_lib for ROCm <version>` section (creating a new top section when the
entry actually belongs to the next, not-yet-tagged release). Keep the same
section headings in whichever release block the entry moves to.

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
