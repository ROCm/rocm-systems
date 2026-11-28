# General guidelines:
- An LLD describes how to build a design from an approved HLD. Link the HLD at the top. Do not repeat its problem statement or requirements.
- Unlike an HLD, an LLD names the modules, classes, files, and command-line options that it changes.
- The PR plan is the most important part. A reviewer should be able to read the entry for one PR and know what its diff will contain.
- Show data structures, schemas, and interfaces as short code blocks. Show data flow as a diagram.
- Follow the [writing style rules](../../CONTRIBUTING.md#writing-style).

# Template
## Motivation
- Link to the HLD.
- Which phases of the HLD this LLD covers. Which phases are left for a later LLD.

## Data flow
- How data moves through the components that this design touches, from input to output.

## Data models and interfaces
- New or changed classes, database tables, file formats, and function signatures.
- For each one, which component writes it and which component reads it.

## Changes per layer
- One subsection per layer that changes, for example profile mode, analyze mode, the native tool, or packaging.
- What changes in that layer, and why it changes this way.

## Failure modes
- What can go wrong at run time, and what the user sees then: an error, a warning, or a fallback.

## Testing
- The unit tests and integration tests that cover the change, and what each one checks.

## PR plan
- One subsection per phase from the "Implementation phases" section of the HLD.
- Under each phase, list its PRs in merge order. For each PR, give:
    - The title, in the `<type>(rocprofiler-compute): ` form.
    - What it changes, in one to three bullets.
    - The PRs that it depends on.
    - How it is tested.
- Each PR merges on its own and leaves `rocprofiler-compute-develop` working.
- Keep each PR small enough to review in one sitting. Put refactors and file moves in their own PR, before the PR that changes behavior.
- If a feature is not usable until a later PR, say how the earlier PRs keep it hidden from users, for example behind an experimental option.

```
### Phase 1: <phase name from the HLD>
1. refactor(rocprofiler-compute): <title>
    - Changes: <what it changes>
    - Depends on: none
    - Tests: <how it is tested>
2. feat(rocprofiler-compute): <title>
    - Changes: <what it changes>
    - Depends on: PR 1
    - Tests: <how it is tested>
```

## Open questions
- Decisions left for review, and choices that a PR review may change.
