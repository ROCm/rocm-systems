# RCCL default-behavior changes

Running log of every change to RCCL's **user-visible default behavior**. Downstream consumers
(Meta in particular) read this file between releases to see behavior deltas ahead of the formal
release notes, so an entry here is how a default change gets communicated.

A CI check (`rccl-default-behavior-gate`) inspects each PR's diff and blocks the PR until every
environment variable it detected has an entry below. See [Workflow](#workflow).

## What belongs here

Add an entry when your PR:

* adds a new environment variable (`NCCL_PARAM` / `RCCL_PARAM` / `RCCL_PARAM_NCCL_ALIAS` /
  `DEFINE_NCCL_PARAM`, or a raw `getenv` / `ncclGetEnv`);
* changes the default value of an existing environment variable;
* makes an existing environment variable take effect on a code path where it previously had
  none.

Behavior changes that are *not* env-var-driven (a new algorithm selected by default, a changed
threshold baked into a heuristic) are welcome here too — the CI check will not ask for them,
but the reader wants them.

## Format

Each release section carries two tables.

### New environment variables

A variable that did not exist before needs to be *documented*, not just announced: a reader has
to be able to tell whether they may rely on it, what it does, and what they can set it to.

| Variable | Supported | Description | Accepted values | Default | Reason for Change |
|---|---|---|---|---|---|
| `NCCL_FOO` | Supported | Selects the X path for AllReduce on MI355 | `0` (off), `1` (on) | `0` | Gates the new X path until it is validated on MI355 |
| `RCCL_BAZ_DEBUG` | Not supported | Dumps per-channel scheduling decisions to stderr | `0` (off), `1` (on) | `0` | Debug aid for scheduler bring-up; not for production use |

* **Supported** — `Supported` or `Not supported`. Say plainly whether users may rely on the
  knob. Use `Not supported` for debug, experimental and internal-only tunables, so nobody
  builds on something that can vanish. (`Yes` / `No` are accepted as synonyms.)
* **Description** — what the variable controls, in one sentence.
* **Accepted values** — the values the variable understands and what each means, including
  units and any sentinel such as `-1` for automatic.
* **Default** — the value in code, and what it means if that is not obvious.
* **Reason for Change** — free text.

CI requires Supported, Description and Accepted values to be filled in; a pasted row left with
its `_italic placeholders_` still fails.

### Changed defaults and other behavior changes

| Change | Reason for Change |
|---|---|
| `NCCL_BAR` default changed `1` -> `2` | Doubles the chunk count; +8% busbw on large all-reduce |
| `NCCL_QUX` now consulted on the NET path | Was previously read only on the P2P path |

The **Change** cell must name the environment variable, and the old and new default where
applicable: CI matches an entry to the variable it detected by looking for the name in this
cell, and checks that a default change mentions the new value, so that a row written for an
earlier release does not silently satisfy a fresh change to the same variable.

CI only reads tables under the `## Unreleased` and `## RCCL <version>` headings, so the example
rows above do not count as entries.

## Workflow

1. Open your PR as usual.
2. If `rccl-default-behavior-gate` fails, read the job summary. It names every variable it
   detected, the `file:line` that triggered it, which condition fired, the old and new default
   for a default change, what is still missing, and a pre-filled table row.
3. Paste the row into the right table under `## Unreleased` and fill in the italicised cells.

### False positives

The check fires on *any* added line that references an environment variable, including an
ordinary refactor that moves a `ncclParamFoo()` call without changing anything. To dismiss
those, add a line to the **PR description**:

```
BEHAVIOR-CHANGE-EXEMPT: NCCL_FOO, NCCL_BAR — moved the call site; no default or code-path change
```

The exemption is per-variable and the reason is mandatory. It is echoed into the CI job summary
so the override is recorded alongside the run. The failure output gives you this line
pre-filled with every reference-only detection, so the common case is a single paste.

Exempting a *newly added* variable or a *default change* is possible but is flagged in the job
summary — those are behavior changes by definition and almost always want a real entry.

## At release

The `## Unreleased` section is folded into the matching `CHANGELOG.md` release section and then
moved down here under its own `## RCCL <version>` heading, so the log stays complete.

---

## Unreleased

### New environment variables

| Variable | Supported | Description | Accepted values | Default | Reason for Change |
|---|---|---|---|---|---|

### Changed defaults and other behavior changes

| Change | Reason for Change |
|---|---|
