---
name: amdsmi-changelog-automation
description: "Check and generate changelog entries for amd-smi. Use when: reviewing PRs for changelog updates, generating release notes, checking CHANGELOG.md compliance."
---

# Changelog Automation — amd-smi

## Location & Headings

`CHANGELOG.md` at the amd-smi project root. Entries group under
`## amd_smi_lib for ROCm <MAJOR>.<MINOR>.<PATCH>` headings (no `## [Unreleased]`).
Allowed subsection headings — anything else is a violation (e.g. `### Fixed`,
which this repo does not use): **Added, Changed, Removed, Optimized, Resolved
Issues, Upcoming Changes, Known Issues.** Bug fixes go under **Resolved Issues**,
deprecations under **Upcoming Changes**.

## Which Release Does an Entry Belong To?

**An entry belongs to release `V` if its authoring commit is in the range
`pin(V-1)..pin(V)`** — the diff between two consecutive ROCm release points, not
the heading it currently sits under.

A "pin" is the `rocm-systems` commit that
[ROCm/TheRock](https://github.com/ROCm/TheRock/releases) records for a release.
TheRock tags drop the patch component: section `7.14.0` → tag `therock-7.14`
(confirm exact names with `gh release list --repo ROCm/TheRock`; `V-1` is the tag
immediately below `V`).

```bash
pin() {  # rocm-systems commit pinned by a TheRock tag; origin/develop if the tag is absent
  gh api "repos/ROCm/TheRock/git/trees/$1" 2>/dev/null \
    | python3 -c "import sys,json;d=json.load(sys.stdin);print(next(t['sha'] for t in d.get('tree',[]) if t['path']=='rocm-systems'))" 2>/dev/null \
    || echo origin/develop
}
```

A tagged release is **frozen at its pin**: `pin(therock-<V>)` is the exact tree
that shipped as ROCm `<V>`.

**Pins are not on `develop`.** TheRock records a commit from a `release/therock-*`
branch, so `git merge-base --is-ancestor <develop-commit> <pin>` is false for
nearly every commit on `develop` and flags the whole section. Convert each pin to
its develop-lineage bound first:

```bash
bound() { git merge-base "$(pin "$1")" origin/develop; }  # where the release branched off develop
```

## Auditing a Section

Bound the section by **its own version**: `LOW=bound(V-1)`, `HIGH=bound(V)` (use
`HEAD` when `V` is not tagged yet). Checking only one bound misses entries merged
after `V` shipped.

```bash
git fetch origin >/dev/null 2>&1
LOW=$(bound therock-<MAJOR.MINOR of V-1>)
HIGH=$(bound therock-<MAJOR.MINOR of V>)   # HEAD if V is untagged
read START END < <(awk '/^## amd_smi_lib for ROCm/{n++; if(n==1)s=NR; else if(n==2){print s, NR-1; exit}}' CHANGELOG.md)

# placement screen — flag commits outside bound(V-1)..bound(V)
# grep drops blank lines: whoever last reflowed the section owns them and they name no entry
git blame -L "$START,$END" -s CHANGELOG.md | grep -P '\)\s*\S' | awk '{print $1}' | sed 's/^\^//' | sort -u \
  | while read h; do
      git merge-base --is-ancestor "$h" "$LOW"  2>/dev/null && { echo "TOO OLD (earlier release): $h"; continue; }
      git merge-base --is-ancestor "$h" "$HIGH" 2>/dev/null || echo "TOO NEW (later release): $h"
    done

# taxonomy — flag disallowed section headings
sed -n "${START},${END}p" CHANGELOG.md | grep -nE '^### ' \
  | grep -vE '### (Added|Changed|Removed|Optimized|Resolved Issues|Upcoming Changes|Known Issues)'

# format — bold headline bullets must end with two trailing spaces; JIRA IDs never in entry text
sed -n "${START},${END}p" CHANGELOG.md | grep -nP '^- \*\*.*\*\*[^ ]*$'         # any match = a headline bullet missing its two trailing spaces
sed -n "${START},${END}p" CHANGELOG.md | grep -nE 'ROCM-[0-9]+|SWDEV-[0-9]+'    # JIRA belongs only in the PR JIRA ID section
```

**Blame is a screen, not a verdict.** Confirm every placement flag against the
shipped tree before reporting it:

```bash
git grep -l "<symbol named in the entry>" "$(pin therock-<V>)" -- projects/amdsmi | grep -v CHANGELOG
```

Present at `pin(V)` → the entry belongs in `V`, whatever blame said. Absent → a
real misfile; find the earliest pin where it is present. This settles the two
cases blame gets wrong:

| Blame artifact | Why it misleads |
|----------------|-----------------|
| Cherry-pick onto the release branch | Work shipped in `V` but reached `develop` after `bound(V)`, so blame reads TOO NEW |
| Moved or reworded line | Blame names the commit that relocated the entry, not the one that wrote it — use `git log -S '<entry text>' -- CHANGELOG.md` for the original |

- **TOO OLD** — commit shipped in an earlier release; the entry is a misfile/duplicate.
- **TOO NEW** — commit merged after `bound(V)`; confirm by content presence, then
  move the entry to the next release's section (create a new top
  `## amd_smi_lib for ROCm <next>` block if none exists).
- **Entry type vs heading** — no command catches this: read each entry's wording
  against the change-type table below. A valid heading can still hold a wrong-type
  entry (e.g. a deprecation filed under `### Changed` instead of Upcoming Changes).

## Section for Each Change Type

| Change | Section | Entry needed? |
|--------|---------|---------------|
| New public API / new CLI flag or subcommand | Added | Yes |
| Bug fix (incl. user-visible build fix) | Resolved Issues | Yes |
| Breaking API change | Changed (+ migration note) | Yes |
| Other behavior/output change | Changed | Yes |
| Performance / tooling improvement | Optimized | Yes |
| Deprecation (still functional) | Upcoming Changes | Yes |
| Internal refactor / test-only / docs-only / style-only | — | No |

## Entry Rules

- One bullet per logical change, starting with the affected component (API name, CLI subcommand, or module).
- Breaking changes include migration guidance.
- JIRA/issue refs only in the PR `JIRA ID` section, never in entry text.
- Bold headline bullets must end with **two trailing spaces** (`··`) so Sphinx
  keeps the headline and its sub-bullets on separate lines:
  ```markdown
  - **Fixed `amd-smi static` hang on gfx1153**.··
    - Added 60-second timeout to `amdsmi_init()`.
  ```
