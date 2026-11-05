# Changelog fragments

Add one file here for each pull request that needs a changelog entry. Do not edit
`../CHANGELOG.md`. Many pull requests edit that file, and the edits conflict.

## File name

`<name>.<section>.md`, for example `14316-direct-reducescatter.resolved.md`. The
`<name>` part must be unique. It does not set the order of the entries.

`<section>` is one of:

| Suffix | Section in `CHANGELOG.md` |
|---|---|
| `added` | Added |
| `changed` | Changed |
| `removed` | Removed |
| `resolved` | Resolved issues |
| `known` | Known issues |

Use only these suffixes, in lowercase, and put the file directly in this directory.
`CHANGELOG.md` also has older headings, such as Fixed and Optimized, but no suffix
selects them. The `RCCL env-var changelog` check fails a pull request that adds any
other file here.

## Content

Write the entry as it must appear in `CHANGELOG.md`: one bullet that starts with `* `.
A bullet can span more than one line. Name every environment variable that the pull
request adds, removes or changes. The `RCCL env-var changelog` check reads this
directory, and it accepts a fragment instead of a `CHANGELOG.md` edit.

## Release

At release time, run `python3 .github/scripts/rccl_assemble_changelog.py` from the
repository root. The script appends each fragment to the newest release section of
`CHANGELOG.md` and deletes the fragment files. It applies fragments in merge order, so
run it in a clone with full history.
