---
description: Create or update AMD SMI documentation
allowed-tools: Read, Write, Edit, Glob, Grep, Bash(git diff*), Bash(git log*)
argument-hint: "<what to document> — e.g., 'update Python API reference for amdsmi_get_gpu_id'
---

Create or update AMD SMI documentation using the `amdsmi-update-docs` subagent.

## Arguments

- `$ARGUMENTS` describes what to create or update.

## Process

### 1. Gather context

Read the relevant existing files before dispatching:
- If updating an existing page, read it first.
- If adding a new page, read a peer page in the same directory for structure reference.
- If documenting a code change, run:

```bash
git diff main..HEAD -- docs/ include/ amdsmi_cli/ py-interface/
```

### 2. Identify scope

Determine:
- **File path(s)** — which file(s) to create or modify
- **Diátaxis category** — `install/`, `how-to/`, `reference/`, `conceptual/`
- **Toc wiring needed** — new pages require entries in `docs/sphinx/_toc.yml.in` and `docs/index.md`
- **Breathe directives** — C++ API content should use `.. doxygen*::` directives, not hand-written signatures

### 3. Dispatch the `amdsmi-update-docs` skill

Use the `amdsmi-update-docs` skill with the task description and all gathered context.

### 4. Confirm and report

After the skill completes:
- Confirm files exist at expected paths
- Confirm toc wiring is present for new pages
- Report the docs update summary
- Make sure all internal cross-references are correct
