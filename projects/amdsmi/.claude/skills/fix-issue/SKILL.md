---
name: fix-issue
description: Fix a defect from a Jira ticket, OR implement a feature from a Confluence/wiki design page, end-to-end, for any git repository — reproduces/builds/tests entirely inside Docker (never touches the host beyond running `docker`), dispatches every heavy-context step to a subagent so the main conversation never auto-compacts, runs a multi-dimension code review loop (max 3 iterations by default, overridable), independently re-verifies every build/test/review claim with a fresh agent, runs a dedicated completeness/correctness audit against the raw ticket/spec before staging, and stages a branch/worktree for user review (never commits or pushes without explicit approval)
---

Fix a defect (from a Jira ticket) or implement a feature (from a Confluence/wiki
design page) end-to-end, against a reviewed, staged branch — dispatching every
heavy-context step to a subagent so the main conversation never auto-compacts.

**Everything reproduces, builds, and tests inside Docker.** This workflow never
installs anything system-wide on the host, never needs sudo, and never mocks
host-level state (sysfs, drivers, hardware). If a defect or feature genuinely
requires real hardware/devices Docker can't emulate, that is a stop-and-ask
condition (see Step 3) — not something to work around with host-level mocking.

## Arguments

Invoked with: `<jira-ticket-url | confluence-or-wiki-page-url> [repo-path-or-url] [max-review-iterations]`

- First argument (required) — detect which kind it is:
  - **Jira ticket** → `MODE = defect`. A URL containing `/browse/<KEY>` on any
    `*.atlassian.net` site, e.g. `https://yoursite.atlassian.net/browse/ABC-123`.
    Extract the issue key (`ABC-123`) via regex `[A-Z][A-Z0-9_]+-\d+` on the URL.
    A bare issue key typed directly (no URL) is also valid.
  - **Confluence/wiki design page** → `MODE = feature`. A URL matching
    `/wiki/spaces/.+/pages/(\d+)/` (Confluence Cloud) or an internal wiki page
    URL the user pastes directly. Extract `PAGE_ID` (the numeric segment) if
    present; otherwise use the page title/URL as given.
  - If neither pattern matches, stop and ask the user which it is.
  - `PAGE_URL` (feature mode only) = the original page URL argument verbatim —
    reused as-is later (Step 12), no re-fetch needed.
- `[repo-path-or-url]` (optional): a local path or git URL for the target repo,
  if the user already knows it. If omitted, Step 2 figures it out.
- `[max-review-iterations]` (optional): a positive integer overriding the
  Step 8 review→fix loop's default cap of 3. Parse it as this argument only if
  it's a bare integer (to avoid misreading it as a malformed repo path); if
  `[repo-path-or-url]` was omitted and this looks like a path/URL instead of a
  number, treat it as `[repo-path-or-url]`, not this. Clamp to a maximum of 3
  — a deeper loop risks subagent nesting limits. If the user passes a value >
  3, use 3 and tell them: `"max-review-iterations capped at 3."` Values of 1
  or 2 are honored as given. Sets `MAX_ITERATIONS` for Step 8. Announce the
  effective value once detected.

Do NOT run this skill's steps directly inline — every heavy-context step below
is its own subagent dispatch. Never paste raw ticket/page JSON, full diffs,
build logs, or review findings tables into the main conversation; agents
return small structured summaries or a path to a handoff doc instead.

## Process

### Step 1 — Fetch and analyze the ticket or design page

Dispatch **Agent A — Ticket/Design Analysis** (foreground; everything below
is gated on its output; nesting depth 1/3). Use the variant matching `MODE`.

Both variants fetch via the **slai-atlassian** skill, not raw Atlassian MCP
tools — it is the maintained replacement for any Atlassian MCP server and
works against whatever site the user's `~/.atlassian/config.json` (or
`ATLASSIAN_*` env vars) points at, so nothing here hardcodes a company/site
domain. If that skill isn't installed in the dispatched agent's environment,
have it install/locate it first (see the slai-atlassian SKILL.md's "Locating
the CLI" section) rather than falling back to a raw MCP tool guess.

**Variant: MODE = defect**
```
TASK: Fetch and analyze Jira ticket {TICKET_KEY} for a defect-fix workflow.

Use the slai-atlassian skill's CLI: `atl jira-issue {TICKET_KEY} --comments`
(run `atl check` first if you haven't confirmed Atlassian access yet this
session).

Return EXACTLY this structure, nothing else:
TITLE: <verbatim ticket title>
SUMMARY: <2-3 sentence paraphrase of the defect>
LABELS: <comma-separated>
COMPONENT: <component/platform/environment mentioned in the ticket>
REPRO_STEPS: <numbered repro steps from the description>
EXPECTED_VS_ACTUAL: <what should happen vs what actually happens>
CHECKLIST: <numbered list, one line per DISTINCT, atomic ask in the ticket —
  split compound asks apart; if the ticket says "fix/also fix/additionally
  address A, B, C, D, E" that is 5 items, never one. Each item must be
  independently checkable against the diff later (a concrete behavior/
  output/field, not a vague theme). If the ticket really is one atomic
  defect, this is a 1-item list — don't invent items that aren't there.>
ENVIRONMENT_SIGNALS: <any specifics the ticket mentions about OS/distro,
  language/runtime version, container vs bare metal, hardware, etc. — these
  drive the Docker image choice in Step 3 — or "none">
STATUS: DONE | BLOCKED

Nesting depth: 1/3
```

**Variant: MODE = feature**
```
TASK: Fetch and analyze design page "{PAGE_ID or URL}" for a feature-dev workflow.

Use the slai-atlassian skill's CLI: `atl conf-page {PAGE_ID}` (or
`atl conf-page --title "..." --space ...` if only a title/space is known).
If the page body is short and mostly links (an index/stub page), also run
`atl conf-children {PAGE_ID} --descendants` one level deep and read the
actual content pages it points to — don't over-fetch beyond that one level.

Return EXACTLY this structure, nothing else:
TITLE: <verbatim page title>
SUMMARY: <2-3 sentence paraphrase of what the feature is and why>
REQUIREMENTS: <bullet list of concrete, testable requirements extracted from the design>
SCOPE_NOTES: <anything the doc explicitly marks out-of-scope, or "none stated">
COMPONENT: <component/platform/environment the design targets>
CHECKLIST: <numbered list, one line per DISTINCT, atomic requirement — this
  is REQUIREMENTS re-expressed as atomic, independently checkable items
  (one behavior/field/output each), not a re-paraphrase. Split any
  requirement bullet that bundles more than one concrete ask.>
ENVIRONMENT_SIGNALS: <any specifics the design mentions about OS/distro,
  language/runtime version, container vs bare metal, hardware, etc. — these
  drive the Docker image choice in Step 3 — or "none">
STATUS: DONE | BLOCKED

Nesting depth: 1/3
```

Downstream steps read `MODE` to reinterpret framing (defect vs. feature) but
the mechanics (repo detection, Docker setup, reproduce, implement, build,
review loop, verification, staging) are identical for both — noted inline
only where wording actually differs.

### Step 2 — Identify the target repository

Deterministic, inline — no subagent needed.

- If `[repo-path-or-url]` was given, use it directly: resolve it to a local
  clone path (clone it fresh into a scratch directory if it's a URL and no
  local copy exists — confirm with the user before cloning, since that's a
  one-time setup action).
- Otherwise, mine `{TITLE}` / `{SUMMARY}` / `{LABELS}` / `{COMPONENT}` for a
  repo name or link. If the ticket/design itself links a repository (a GitHub/
  GitLab URL, or an explicit "repo:" field), use that.
- If nothing points at a specific repo, don't just stall — offer a concrete
  default to anchor the question, then confirm it with the user rather than
  silently assuming it. `DEFAULT_REPO = https://github.com/ROCm/rocm-systems`
  is a reasonable out-of-the-box default; swap in whatever repo you actually
  work in most. Ask: *"This ticket/design doesn't name a specific
  repository — should I default to {DEFAULT_REPO}, or is this for a
  different repo (a path to an existing local clone, or a git URL to clone
  fresh)?"* Never guess silently either way — always get an explicit yes on
  the default, or a different repo, before cloning/using anything.

Resolve to `REPO_PATH` (a local clone path) and confirm it's a git repo
(`git -C {REPO_PATH} rev-parse --show-toplevel`).

### Step 3 — Set up the Docker environment

This skill assumes plain Docker only — no internal/company-specific build
skill is available to lean on. Before dispatching anything, confirm Docker
itself is usable, inline:

- `docker version` — if this fails to reach the daemon:
  - Linux: the daemon usually needs starting —
    `sudo systemctl start docker` (or `sudo service docker start` on
    non-systemd distros). If it also needs to survive a reboot,
    `sudo systemctl enable docker` — but only do that if the user asks, it's
    outside this workflow's scope.
  - macOS/Windows: Docker Desktop needs to be running — ask the user to
    start the Docker Desktop app, since this workflow has no way to launch a
    GUI app on their behalf.
- A `permission denied` talking to `/var/run/docker.sock` on Linux means the
  current user isn't in the `docker` group yet — either run subsequent
  commands with `sudo docker ...`, or ask the user to add themselves
  (`sudo usermod -aG docker $USER`, then re-login) if they'd rather not
  prefix every command with `sudo`.
- Confirm with `docker info` once the daemon responds, then proceed.

**If Docker genuinely isn't available** (not installed, no permission to
install it, or the user says not to bother) — don't treat that as a hard
stop. Fall back to reproducing via the repo's own unit tests, using
dummy/fixture files to stand in for whatever real condition the ticket/
design depends on (a recorded config, a canned input file, a fake data
fixture, a stubbed dependency the test suite already knows how to swap in)
instead of a full OS/container environment. Most repos' test suites already
support exactly this kind of dummy-file-driven test case — look for an
existing `fixtures/`, `testdata/`, `test/data/` (or similar) directory and
existing tests that load a file from it as the pattern to follow, rather
than inventing a new mocking mechanism. This fallback runs directly on the
host (no container), so it's a lower bar of isolation than Docker — note
that tradeoff to the user once, don't silently treat it as equivalent.

Dispatch **Agent B0 — Docker Environment Setup** (foreground; nesting depth
1/3). This replaces any host-level mocking entirely: every build, test, and
repro command for the rest of this workflow runs inside a container, never
on the host. (If the Docker-unavailable fallback above applies, skip this
dispatch and go straight to Step 4's fallback variant instead.)

```
TASK: Determine and build the Docker environment to reproduce, build, and
test inside, for {REPO_PATH}.

1. Look for an environment definition the repo ALREADY has, in this order,
   and reuse it verbatim if found — don't invent a parallel one:
   - A `Dockerfile` / `docker-compose.yml` / `.devcontainer/devcontainer.json`
     at the repo root or under `docker/`, `.devcontainer/`.
   - A CI workflow (`.github/workflows/*.yml`, `.gitlab-ci.yml`,
     `Jenkinsfile`, etc.) that names a container image for build/test — reuse
     that same image.
2. If none exists, infer a minimal image from the repo's own build tooling
   (read README/CONTRIBUTING/CLAUDE.md/AGENTS.md first): pick a base image
   matching the primary language/toolchain and {ENVIRONMENT_SIGNALS} from
   Step 1 (OS/distro/runtime version called out in the ticket/design, if
   any), then install only the packages that repo's own build instructions
   say are required. Write this as a throwaway Dockerfile under a scratch
   directory (e.g. ${TMPDIR:-/tmp}/fix-issue-{slug}/docker/) — never commit
   it to {REPO_PATH} unless the user later asks for it as part of the fix.
3. `docker build` the image. Mount {REPO_PATH} read-write into the container
   for all later steps (`docker run -v {REPO_PATH}:/workspace -w /workspace
   ...`), so edits made by later agents on the host are immediately visible
   inside the container and vice versa — never copy the repo into the image.
4. If reproducing requires simulating a specific condition (a particular
   OS/distro, a missing/present config file, an env var, a particular
   language/runtime version) — do it via the image's base tag, `docker run
   -e KEY=VALUE`, or a file written into the mounted repo tree. Never via
   host sudo, host package installs, or host-level device/driver mocking.
5. **Emulating a sysfs-style device interface.** This matters most when a
   kernel driver has only defined the interface shape (which files/paths
   exist) but not the real data yet — e.g. you're building the userspace
   side of a feature against a spec while the driver work lands in
   parallel, or the real hardware generating that interface isn't available
   to you. Don't try to fake this with host-level overlay/mount tricks —
   Docker's own bind-mount isolation already does the job: prepare a plain
   directory tree mirroring the real paths/files/values the interface would
   expose (hand-authored for a not-yet-real driver, or captured once from
   any real machine that does have it), then bind-mount it over the
   container's own path, e.g. `docker run -v <snapshot-dir>/sys:/sys:ro
   ...` for a Linux sysfs-based interface. The code under test reads through
   the normal filesystem path and never knows it's not talking to a real
   device. If the code also issues ioctls against a device node (not just
   plain file reads), a bind-mounted file tree alone can't fake those — that
   needs an `LD_PRELOAD` shim intercepting the specific ioctl calls,
   recording real responses once (if any real device exists anywhere) and
   replaying them later.
6. If the defect/feature genuinely requires something even this can't
   emulate (true real-time hardware behavior, a kernel feature not present
   in the container's kernel) — STOP and report BLOCKED with exactly what's
   missing; do not attempt a host-level workaround. The user may be able to
   supply `--device`/`--gpus`/`--privileged` flags if their Docker setup
   supports it, but that's their call, not a default.

EXPECTED OUTPUT:
STATUS: READY | BLOCKED
IMAGE_TAG: <the built image tag>
DOCKERFILE_PATH: <path to the Dockerfile used, whether reused from the repo
  or newly written to scratch>
RUN_TEMPLATE: <the exact `docker run ...` invocation template later steps
  should reuse, with {CMD} as a placeholder for the actual command>
NOTES: <anything Docker-specific later steps need to know — e.g. "tests need
  --init for signal handling" or "mounted at /workspace, not /repo">
Nesting depth: 1/3
```

If `BLOCKED` for a reason OTHER than "Docker isn't available" (e.g. the
image itself won't build): stop the whole workflow and report to the user —
do not guess and continue to reproduction with a broken or incomplete
environment. If Docker isn't available at all, don't dispatch this agent —
use Step 4's fallback variant instead (`DOCKER_AVAILABLE = false`).

Every later step that needs to run a command inside the environment does so
via `{RUN_TEMPLATE}` with `{CMD}` substituted — never a bare host command for
anything build/test/repro-related. (Fallback mode has no `{RUN_TEMPLATE}` —
later steps run the bare host test command instead, per Step 4's fallback
variant.)

### Step 4 — Reproduce the defect / capture the baseline

**Docker-available path** — dispatch **Agent B — Repro** (foreground;
nesting depth 1/3).

```
TASK: (defect mode) Reproduce the defect described in ticket {TICKET_KEY}
inside the Docker environment from Step 3. (feature mode) Capture the
CURRENT behavior relevant to the feature described in the design page,
inside the same environment, before implementing it.

CONTEXT: {SUMMARY} / {REPRO_STEPS or REQUIREMENTS} / {EXPECTED_VS_ACTUAL or
SCOPE_NOTES} / {COMPONENT} / Docker image {IMAGE_TAG}, run template
{RUN_TEMPLATE}.

Use `{RUN_TEMPLATE}` for every command. Do not run build/test/repro commands
directly on the host.

CONSTRAINTS: no env-var hacks or test-only branches in production code to
fake the condition under test — if the condition needs faking, do it at the
Docker layer (Step 3's mechanisms), not inside application code.

EXPECTED OUTPUT:
STATUS: REPRODUCED | NOT_REPRODUCED | BLOCKED   (feature mode: BASELINE_CAPTURED | BLOCKED)
REPRO_COMMAND: <exact `docker run ...` command(s), verbatim>
REPRO_OUTPUT_BEFORE: <the actual bad output/behavior observed>
  (feature mode: the CURRENT behavior before the feature is implemented)
Nesting depth: 1/3
```

**No-Docker fallback path** (`DOCKER_AVAILABLE = false` from Step 3) —
dispatch the same **Agent B — Repro** role, host-native variant instead:

```
TASK: (defect mode) Reproduce the defect described in ticket {TICKET_KEY} by
adding (or running, if one already exists) a unit test that uses a
dummy/fixture file to stand in for the real condition — no Docker, no
container, run directly against the repo's own test runner on the host.
(feature mode) Same, but capture the CURRENT behavior as a test that will
later flip from failing to passing once the feature lands.

CONTEXT: {SUMMARY} / {REPRO_STEPS or REQUIREMENTS} / {EXPECTED_VS_ACTUAL or
SCOPE_NOTES} / {COMPONENT}.

1. Find the repo's existing fixture/testdata convention (e.g. `fixtures/`,
   `testdata/`, `test/data/`) and an existing test that loads a file from it
   — mirror that pattern exactly rather than inventing a new one.
2. Add a dummy file under that same directory representing the exact
   condition from {REPRO_STEPS}/{EXPECTED_VS_ACTUAL} (or {REQUIREMENTS}) —
   e.g. a config/response/data file with the specific field/value combo that
   triggers the defect, or that the feature needs to handle.
3. Add (or identify) the unit test that exercises that dummy file through
   the code path in question, and run it with the repo's own test runner.

CONSTRAINTS: the dummy file and test belong in the repo's normal test/
fixture tree, following its existing naming conventions — not a scratch
directory, since this fixture is part of the reproduction and will likely
ship with the fix. No env-var hacks or production-code test-only branches;
the dummy file is the only "fake" here.

EXPECTED OUTPUT:
STATUS: REPRODUCED | NOT_REPRODUCED | BLOCKED   (feature mode: BASELINE_CAPTURED | BLOCKED)
REPRO_COMMAND: <exact host test-runner command(s), verbatim>
REPRO_OUTPUT_BEFORE: <the actual failing test output observed>
  (feature mode: the CURRENT (failing/missing) behavior before the feature is implemented)
FIXTURE_FILES: <path(s) to the dummy/fixture file(s) added, under the repo's
  own test tree>
Nesting depth: 1/3
```

If `NOT_REPRODUCED` (defect mode): STOP the whole workflow and report to the
user — do not guess and continue to a fix. If `BLOCKED`, report exactly what
blocked it and ask the user how to proceed.

### Step 5 — Detect whether cwd is already this ticket/feature's worktree or branch

Deterministic, inline — no subagent needed.

Purpose: avoid always creating a brand-new branch/worktree when the user
already set up a folder for this work before invoking this skill.

- If the current directory (or `{REPO_PATH}`) is already a git checkout on a
  branch that looks purpose-made for this ticket/feature (not the repo's
  default branch), reuse it in place — confirm with `git remote -v` /
  `git rev-parse --show-toplevel` that it's actually a checkout of
  `{REPO_PATH}`'s repo before editing.
- Otherwise: create a plain feature branch off that repo's default branch
  (ask the user which branch is the integration branch if it isn't obviously
  `main`/`master`/`develop`), named something like
  `fix/{TICKET_KEY-or-slug}` (defect) or `feature/{slug-from-design-title}`
  (feature). If the repo's own CLAUDE.md/CONTRIBUTING documents a specific
  worktree or branch-naming convention, follow that instead.

Announce which path was taken and the resulting `{BRANCH}` /
`{WORKTREE_PATH}` (same as `{REPO_PATH}` unless a separate worktree was made).

### Step 6 — Author the fix / implement the feature

Only proceed if Step 4 returned `STATUS: REPRODUCED` (defect mode) or
`STATUS: BASELINE_CAPTURED` (feature mode).

**Decide single-agent vs. split-by-layer first.** Default to the
single-agent path below — it's the right call for nearly all defect fixes
and small/localized features. Only use the split path when this is
**feature mode** AND `{REQUIREMENTS}` clearly spans more than one
independent layer (e.g. a new/changed core API **and** its own CLI/UI
exposure **and** dedicated test coverage the design calls out) — large
enough that one agent authoring all layers in sequence would burn a lot of
context re-deriving each layer's conventions. Do not split a defect fix or a
feature that's really just one layer with no separate surface — that just
adds coordination overhead for no benefit.

#### Single-agent path (default)

Dispatch **Agent C — Fix/Feature Author** (foreground; nesting depth 1/3):

```
TASK: (defect mode) Fix the defect in ticket {TICKET_KEY}, confirmed
reproduced in Step 4. (feature mode) Implement the feature described in
design page {TITLE}, per {REQUIREMENTS}, against the baseline captured in
Step 4.

CONTEXT: {TITLE} / {SUMMARY} / {EXPECTED_VS_ACTUAL or REQUIREMENTS +
SCOPE_NOTES} / {CHECKLIST} (from Step 1) / repro/baseline command +
before-output from Step 4: {REPRO_COMMAND}, {REPRO_OUTPUT_BEFORE} /
{BRANCH} at {WORKTREE_PATH} (from Step 5) / Docker image {IMAGE_TAG}, run
template {RUN_TEMPLATE} (from Step 3).

FIRST — re-read the raw source yourself: {CHECKLIST} and {SUMMARY} are a
starting index, not a replacement for the original text — re-fetch and read
it directly (`atl jira-issue {TICKET_KEY}` in defect mode, `atl conf-page
{PAGE_ID}` in feature mode) before implementing, so nothing Step 1
compressed or dropped gets silently carried forward. Treat {CHECKLIST} as
the minimum set of items to address, not the ceiling — if your own read of
the raw source turns up an item Step 1 missed, address that too and note it
in RATIONALE.

SECOND — `cd` into {WORKTREE_PATH}; confirm you're on {BRANCH} before
editing anything.

FOLLOW (read before editing, whichever exist in this repo): its own
README/CONTRIBUTING/CLAUDE.md/AGENTS.md for conventions, coding style, and
any "do not hand-edit this generated file" / "put X here, not there" rules.
Match existing patterns in the surrounding code rather than introducing a
new style. Minimal, surgical diff — touch only what the defect/feature
requires.

CONSTRAINTS: do not commit yet (staging happens later). Do not touch files
outside what {CHECKLIST} (defect mode) or {REQUIREMENTS}/{SCOPE_NOTES}
(feature mode) requires.

EXPECTED OUTPUT:
STATUS: DONE | BLOCKED | NEEDS_CONTEXT
FILES CHANGED: <path — one-line summary, per file>
RATIONALE: <1-2 sentences: root cause + fix approach, or feature design + implementation approach>
CHECKLIST_COVERAGE: <one line per CHECKLIST item — "item N: file:line" if
  addressed, or "item N: not yet addressed" if not. Self-reported — Step 11
  independently re-derives and audits this, so don't round up.>

Write a handoff doc to ${TMPDIR:-/tmp}/fix-issue-handoff-$(date +%Y%m%d-%H%M%S).md
summarizing the above, referencing the diff by path (`git diff`) rather than
pasting it — later steps read this doc instead of re-deriving context.

Nesting depth: 1/3
```

#### Split-by-layer path (large multi-layer features only)

1. Dispatch **Agent C-core — Core Author** (foreground; nesting depth 1/3)
   first, alone. Same TASK/CONTEXT/FOLLOW/CONSTRAINTS as the single-agent
   prompt above, except:
   - Scope it to the core logic/API surface only — not the CLI/UI exposure
     layer, not tests. Those are the other agents' job.
   - EXPECTED OUTPUT adds a `CONTRACT` field: the exact new/changed
     function signatures, types, and behaviors now exposed — this is the
     frozen interface the other agents below must code against. Do not let
     this drift once written; if a downstream agent finds the contract
     insufficient, it reports `BLOCKED` (see step 2) rather than editing
     core files itself.
   - Handoff doc name: `fix-issue-handoff-core-$(date +%Y%m%d-%H%M%S).md`.
   - Do not proceed to step 2 unless this agent returns `STATUS: DONE`.

2. Only after Agent C-core returns `STATUS: DONE`, dispatch **Agent C-ui**
   and **Agent C-tests** together — one message, two Agent tool calls, so
   they run concurrently (each foreground; nesting depth 1/3). Both work on
   the SAME `{BRANCH}`/`{WORKTREE_PATH}` Agent C-core used and both read
   Agent C-core's handoff doc + `CONTRACT` field — this is safe without a
   merge step because each is restricted to a disjoint file set:
   - **Agent C-ui** — implement the CLI/UI exposure of the new/changed
     behavior strictly per `CONTRACT`, in that layer's files only. Must NOT
     touch the core logic files.
   - **Agent C-tests** — add/update tests for the new/changed behavior
     strictly per `CONTRACT`, in the repo's test directory only. Must NOT
     touch core or UI/CLI source files.
   - Same EXPECTED OUTPUT shape as the single-agent path, each with its own
     handoff doc (`fix-issue-handoff-ui-*.md` / `fix-issue-handoff-tests-*.md`).
   - If either reports `STATUS: BLOCKED` because `CONTRACT` doesn't cover
     something it needs, or reports `FILES CHANGED` outside its assigned
     layer, stop and surface both reports to the user rather than letting it
     improvise past its lane — re-run Agent C-core with the missing piece
     instead.

3. Step 7 then reads all three handoff docs (core + ui + tests) instead of a
   single one — pass `{HANDOFF_PATH}` through as a list.

### Step 7 — Build and verify the fix

Dispatch **Agent D — Build & Verify** (foreground; nesting depth 1/3),
reading Step 6's handoff doc(s) rather than re-deriving the diff. (If Step 3
used the no-Docker fallback, substitute the repo's own host test-runner
command everywhere `{RUN_TEMPLATE}` appears below — same idea, just without
a container.)

```
TASK: Build and smoke-test the change in {WORKTREE_PATH}, inside the Docker
environment from Step 3 (image {IMAGE_TAG}, run template {RUN_TEMPLATE}) —
or, in no-Docker fallback mode, directly via the repo's host test runner.

CONTEXT: read the handoff doc(s) at {HANDOFF_PATH} for what changed and why.

Rebuild and run the repo's own build command via `{RUN_TEMPLATE}` (never a
bare host command, UNLESS this is the no-Docker fallback, in which case the
host command IS the correct one to use). Then smoke-test the touched area
the same way.

You MUST actually execute every command yourself and report the real
observed output — never state a result you did not run.

EXPECTED OUTPUT:
STATUS: BUILD_OK | BUILD_FAIL
COMMANDS_RUN: <every `docker run ...` command you executed, verbatim, one
  per line, in order>
BUILD_LOG_TAIL: <last ~30 lines of REAL captured build output, if failed>
SMOKE_TEST_RESULT: <pass/fail>
RAW_OUTPUT_TAIL: <last ~20-30 lines of the REAL captured smoke-test output,
  verbatim — not a paraphrase>
Nesting depth: 1/3
```

If `BUILD_FAIL`: STOP and report — do not proceed to review with a broken
build.

**Independent verification (Agent D') — mandatory, do not skip:**

Immediately dispatch a SEPARATE, fresh agent to independently confirm Agent
D's claim before trusting it (foreground; nesting depth 1/3):

```
TASK: Independently verify Agent D's build/smoke-test claim for
{TICKET_KEY or TITLE} at {WORKTREE_PATH}. Do NOT trust the report below as
fact — re-run the commands yourself inside the same Docker environment and
compare your own actual output to what was claimed.

CLAIMED COMMANDS_RUN: {Agent D's COMMANDS_RUN}
CLAIMED STATUS: {Agent D's STATUS}
CLAIMED RAW_OUTPUT_TAIL: {Agent D's RAW_OUTPUT_TAIL}

Re-execute each claimed command via `{RUN_TEMPLATE}`. Do not just re-read
Agent D's handoff doc as if it were proof; actually run the build and smoke
test again.

EXPECTED OUTPUT:
VERIFIED: MATCH | MISMATCH | COULD_NOT_VERIFY
DETAILS: <diff between claimed and actual output, or "confirmed identical
  (modulo timestamps/counters/PIDs)">
Nesting depth: 1/3
```

If `VERIFIED: MISMATCH` or `COULD_NOT_VERIFY`: STOP the whole workflow. Show
both Agent D's and Agent D''s reports to the user side by side and ask how to
proceed (retry / investigate manually / abort) — never silently proceed on
an unconfirmed build/test claim.

### Step 8 — Run the code review

This step takes a `REVIEW_MODE` param, set by the Step 9 loop:
`comprehensive` (iteration 1) or `scoped` (iteration > 1). Two dimension
groups matter for `scoped` mode:

- **HOLISTIC** — `skeptic`, `spec`, `architecture`. These catch
  system-level regressions a narrow delta check would miss, so they always
  re-run full-context, comprehensive, against the whole diff vs. the base
  branch, every iteration. This is the safety net that preserves "catch big
  things before last minute."
- **MECHANICAL** — `style`, `tests`, `docs`, `build`, `security`,
  `performance`. Findings here are almost always localized to specific
  lines, so on `scoped` iterations these run against the delta only (the fix
  commit(s)/changes since the last review), seeded with that dimension's own
  prior findings to confirm instead of rediscover.

If the target repo has its own project-specific review agent/checklist
(e.g. a `.github/agents/*.md` review spec, a documented review checklist in
CONTRIBUTING), use it in place of the generic 9-dimension set below —
otherwise use the generic set.

**If `REVIEW_MODE == comprehensive`:**

Dispatch **Agent E — Review** (foreground; nesting depth 1/3; the Step 9
loop depends on its verdict):

```
TASK: Review the branch at {WORKTREE_PATH} for {TICKET_KEY or "feature:
{TITLE}"}.

1. `git branch --show-current` / `git log --oneline {BASE_BRANCH}..HEAD` /
   `git diff --stat {BASE_BRANCH}..HEAD`
2. `git diff {BASE_BRANCH}..HEAD`
3. Review the diff across these dimensions (dispatch one subagent per
   dimension, comprehensive mode, all of them): style, tests, docs,
   architecture, security, performance, build, skeptic (actively try to
   poke holes in the fix/feature), spec (does it actually match
   {CHECKLIST}/{REQUIREMENTS}?). Follow with a brief rebuttal round where
   each dimension gets to see the others' findings and retract/confirm.
4. Report using a standard findings table: severity markers ❌ BLOCKING /
   ⚠️ IMPORTANT / 💡 SUGGESTION / 📋 FUTURE WORK, status ✅ APPROVED /
   ⚠️ CHANGES REQUESTED / 🚫 REJECTED.

Write the FULL findings table to
${TMPDIR:-/tmp}/fix-issue-handoff-review-$(date +%Y%m%d-%H%M%S).md.

EXPECTED OUTPUT (small — do not paste the full table here):
STATUS: ✅ APPROVED | ⚠️ CHANGES REQUESTED | 🚫 REJECTED
BLOCKING_COUNT: <N>
REVIEWED_HEAD_SHA: <output of `git rev-parse HEAD` at review time>
REVIEW_HANDOFF_PATH: <path to the findings doc>
Nesting depth: 1/3
```

**If `REVIEW_MODE == scoped`:**

Dispatch **Agent E_{iteration} — Scoped Review**, still one foreground agent
(nesting depth 1/3), doing both groups itself:

```
TASK: Re-review the branch at {WORKTREE_PATH} after a fix round, for
{TICKET_KEY or "feature: {TITLE}"}. This is iteration {iteration} of the
review/fix loop — do not repeat a full from-scratch review; you have prior
findings and a small delta to check.

CONTEXT YOU ARE GIVEN (do not re-derive):
- Prior findings doc: {REVIEW_HANDOFF_PATH}
- Previously reviewed commit: {PREV_REVIEWED_HEAD_SHA}

1. DELTA_DIFF = `git diff {PREV_REVIEWED_HEAD_SHA}..HEAD` — this is just
   the fix agent's changes from the last iteration.
2. HOLISTIC pass (always full-context, catches regressions the delta view
   can't): re-review `skeptic`, `spec`, `architecture` against the FULL
   diff vs. {BASE_BRANCH}, same as a comprehensive run — but additionally
   give each of the 3 dimensions a one-paragraph "already cleared" pointer:
   which subsystems/files this dimension scrutinized in the prior iteration
   with no finding, so it can concentrate its exploration on DELTA_DIFF's
   blast radius rather than re-deriving a full opinion on already-cleared
   code from zero. This does not narrow WHAT it's allowed to flag (still
   full diff, still independent, still free to raise anything anywhere) —
   it only biases where it spends effort first.
3. MECHANICAL pass (delta-scoped): for each of style, tests, docs, build,
   security, performance that had an UNRESOLVED finding as of the PREVIOUS
   iteration (still open in {REVIEW_HANDOFF_PATH} — not "ever had a finding
   historically"; once a dimension comes back clean for one full iteration,
   drop it from all subsequent iterations, since the holistic pass covers
   regressions), re-review that dimension with:
     - only DELTA_DIFF (not the full diff)
     - that dimension's own prior findings, extracted from
       {REVIEW_HANDOFF_PATH}
     - instruction: "confirm each listed finding is resolved in this delta;
       separately flag any NEW issue in this delta within your dimension's
       scope; do not re-review code untouched by the delta"
   Skip a mechanical dimension entirely if it has zero unresolved findings
   going into this iteration.
4. Merge holistic + mechanical results into one findings table, same
   severity markers and status as the comprehensive template.

Write the FULL merged findings table to
${TMPDIR:-/tmp}/fix-issue-handoff-review-$(date +%Y%m%d-%H%M%S).md.

EXPECTED OUTPUT (small — do not paste the full table here):
STATUS: ✅ APPROVED | ⚠️ CHANGES REQUESTED | 🚫 REJECTED
BLOCKING_COUNT: <N>
REVIEWED_HEAD_SHA: <output of `git rev-parse HEAD` at review time>
REVIEW_HANDOFF_PATH: <path to the findings doc>
Nesting depth: 1/3
```

If a BLOCKING finding survives 2 consecutive scoped iterations without
resolving (the mechanical dimension keeps reporting it unfixed), treat that
as a signal the fix isn't working — fall back to a full `comprehensive`
review on the next iteration instead of a 3rd scoped pass, so a stuck fix
gets fresh eyes on the whole diff rather than another narrow delta check.

Nesting-depth accounting: this skill's own dispatches (Agents A–H, including
the independent-verification agents D' and G', Step 11's audit agent and its
loop's fix agents, each loop iteration's fix-agent and review-agent, and
Step 6's split-path Agent C-core/C-ui/C-tests) are always `1/3` — C-ui and
C-tests run concurrently with each other but are still flat,
sequential-from-the-skill's-perspective dispatches, not nested inside one
another. Both the comprehensive review agent's internal per-dimension
dispatch and the scoped review agent's internal holistic+mechanical
dispatches are `2/3`. Nothing in this workflow should need `3/3`.

### Step 9 — Review → fix loop (max iterations, default 3, hard cap 3)

```
iteration = 1
MAX_ITERATIONS = min({max-review-iterations arg, else 3}, 3)   # 3 is both
    # the default and the hard ceiling. A lower override (1 or 2) is
    # honored; anything above 3 is clamped to 3 (see Arguments section).
PREV_REVIEWED_HEAD_SHA = null
force_comprehensive_next = false

loop:
    if iteration > MAX_ITERATIONS:
        STOP. Present to the user:
          "After {MAX_ITERATIONS} review/fix iterations, {BLOCKING_COUNT}
           BLOCKING finding(s) remain:
           {list of findings, one line each}
           Full findings: {REVIEW_HANDOFF_PATH}
           How would you like to proceed?
           (A) I'll fix them myself — pause here
           (B) Try one more iteration anyway
           (C) Accept as-is and proceed to verification despite BLOCKING findings
           (D) Discard this attempt"
        WAIT for the user's reply. Never silently continue past
        MAX_ITERATIONS, and never silently give up.

    REVIEW_MODE = comprehensive if (iteration == 1 or force_comprehensive_next) else scoped
    run Step 8 with REVIEW_MODE (passing REVIEW_HANDOFF_PATH and
    PREV_REVIEWED_HEAD_SHA from the previous iteration when REVIEW_MODE
    == scoped)
    PREV_REVIEWED_HEAD_SHA = REVIEWED_HEAD_SHA   # for next iteration's delta

    if BLOCKING_COUNT == 0:
        break   # proceed to Step 10

    force_comprehensive_next = (a BLOCKING finding repeats, unresolved,
                                 from the previous iteration's findings —
                                 see the fallback rule in Step 8)

    dispatch a FRESH Agent F_{iteration} — Fix Reviewer Findings
    (foreground, nesting depth 1/3), distinct from both Agent C and Agent E:
        TASK: Address the {BLOCKING_COUNT} ❌ BLOCKING findings from the
        review at {WORKTREE_PATH}.
        CONTEXT: read the review handoff doc at {REVIEW_HANDOFF_PATH} — do
        not ask for the findings to be re-explained, do not re-derive the
        diff from scratch.
        CONSTRAINTS: fix ONLY the BLOCKING findings (⚠️/💡/📋 are
        optional — mention if trivially bundled, otherwise leave for the
        user). Same repo conventions as Step 6. Do not touch files outside
        what's needed for these findings.
        EXPECTED OUTPUT: same handoff-doc contract as Agent C
        (STATUS / FILES CHANGED / handoff doc path)
        Nesting depth: 1/3

    iteration += 1
```

### Step 10 — Full verification

Once BLOCKING findings are cleared (or the user overrode in Step 9),
dispatch **Agent G — Verification** (foreground; nesting depth 1/3). (Same
no-Docker-fallback substitution as Step 7 applies throughout: swap
`{RUN_TEMPLATE}`/"Docker environment" for the host test runner if Step 3 was
skipped.)

```
TASK: Fully verify the fix for {TICKET_KEY or TITLE} at {WORKTREE_PATH},
inside the Docker environment from Step 3 — or, in no-Docker fallback mode,
via the repo's host test runner.

1. Build verification: rebuild clean (no new warnings) via `{RUN_TEMPLATE}`.
2. Run the repo's own test suite(s) the same way.
3. Behavioral verification: re-run the EXACT REPRO_COMMAND from Step 4
   (inside the same Docker environment, or on the host in fallback mode) and
   show:
   BEFORE (Step 4's REPRO_OUTPUT_BEFORE)
   AFTER (fresh output now)
   — confirm the defect no longer reproduces, not just "no crash."
4. Regression check: no existing functionality broken (test pass/fail
   deltas vs. a baseline from the default branch, if time permits).

You MUST actually execute every command yourself and report the real
observed output — never state a result you did not run.

EXPECTED OUTPUT:
STATUS: VERIFIED | FAILED
COMMANDS_RUN: <every command you executed, verbatim, one per line, in
  order — build, each test-suite invocation, and the behavioral repro re-run>
TEST_RESULTS: <N passed, M failed, K skipped, per suite>
BEFORE_AFTER: <the before/after repro output pair, verbatim>
RAW_OUTPUT_TAIL: <last ~20-30 lines of REAL captured test-suite output,
  verbatim — not a paraphrase>
Nesting depth: 1/3
```

If `STATUS: FAILED` — STOP, report exactly which check failed, do not
proceed to Step 11.

**Independent verification (Agent G') — mandatory, do not skip:**

Immediately dispatch a SEPARATE, fresh agent to independently confirm Agent
G's claim before trusting it (foreground; nesting depth 1/3):

```
TASK: Independently verify Agent G's full-verification claim for
{TICKET_KEY or TITLE} at {WORKTREE_PATH}. Do NOT trust the report below as
fact — re-run the commands yourself inside the same Docker environment and
compare your own actual output to what was claimed.

CLAIMED COMMANDS_RUN: {Agent G's COMMANDS_RUN}
CLAIMED RESULT: {Agent G's STATUS, TEST_RESULTS, BEFORE_AFTER, RAW_OUTPUT_TAIL}

Re-execute each claimed command via `{RUN_TEMPLATE}`. Do not just re-read
Agent G's report as if it were proof; actually run the test suites and the
before/after repro again yourself.

EXPECTED OUTPUT:
VERIFIED: MATCH | MISMATCH | COULD_NOT_VERIFY
DETAILS: <diff between claimed and actual output, or "confirmed identical
  (modulo timestamps/counters/PIDs)">
Nesting depth: 1/3
```

If `VERIFIED: MISMATCH` or `COULD_NOT_VERIFY`: STOP the whole workflow. Show
both Agent G's and Agent G''s reports to the user side by side and ask how to
proceed (retry / investigate manually / abort) — never silently proceed on
an unconfirmed verification claim, and do not proceed to Step 11.

### Step 11 — Completeness & correctness audit (mandatory, do not skip)

This closes a gap the review loop (Step 8/9) doesn't fully cover: that
loop's `spec` dimension re-fetches the real spec and checks conformance, but
only `BLOCKING` findings force another iteration — a missed or mismatched
requirement reported as `⚠️ IMPORTANT`, or still open when the loop hits its
iteration cap and the user picks "accept as-is," would otherwise ship
silently incomplete/incorrect. This step is an unconditional gate,
independent of that severity triage.

Dispatch **Agent H — Completeness & Correctness Audit** (foreground; nesting
depth 1/3), a fresh agent that did not author the fix and did not run the
code review — it must re-derive its judgment from the raw source, not trust
any prior paraphrase or self-report:

```
TASK: Independently audit completeness and correctness of the (defect mode)
fix for {TICKET_KEY} / (feature mode) feature "{TITLE}" at {WORKTREE_PATH},
against the ORIGINAL ticket/design. Do NOT trust Step 1's paraphrase or
Agent C's CHECKLIST_COVERAGE claim as fact — re-fetch the raw source
yourself and re-derive each judgment from it independently.

CONTEXT:
- CHECKLIST (from Step 1, verbatim — treat as a floor, not a ceiling):
  {CHECKLIST}
- Re-fetch fresh, do not reuse Step 1's cached copy: (defect mode)
  `atl jira-issue {TICKET_KEY}` (feature mode) `atl conf-page {PAGE_ID}`
- Full diff: `git -C {WORKTREE_PATH} diff {BASE_BRANCH}..HEAD`

FOR EACH CHECKLIST ITEM, determine independently:
- ADDRESSED: Yes / No / Partial — point to the exact diff hunk (file:line)
  that addresses it. No hunk found = No, even if some other change looks
  like it should plausibly cover it — do not give credit for a vibe.
- CORRECT: Yes / No / Unclear — does the change match what the raw
  spec/ticket text literally says (exact field name/threshold/behavior/
  output format), not just "something plausible in the same area." Quote
  the matching spec/ticket text and the matching code side by side as
  evidence.
Also check, across the WHOLE diff (not item by item):
- EXTRA_SCOPE: diff content that maps to no checklist item at all —
  possible scope creep, or a sign the checklist itself under-counted
  requirements when it was first written in Step 1.

Write the full item table to
${TMPDIR:-/tmp}/fix-issue-handoff-audit-$(date +%Y%m%d-%H%M%S).md.

EXPECTED OUTPUT (small — do not paste the full table here):
STATUS: COMPLETE | INCOMPLETE | MISMATCHED
ITEM_TABLE_SUMMARY: <N of M items ADDRESSED+CORRECT>
MISSING_OR_WRONG: <item #s with a one-line reason each, or "none">
EXTRA_SCOPE: <one-line list, or "none">
AUDIT_HANDOFF_PATH: <path to the full table>
Nesting depth: 1/3
```

**Bounded fix → reaudit loop** (same shape as Step 9, capped at 2 rounds
total so it can't run away):

```
audit_round = 1
loop:
    run Agent H
    if STATUS == COMPLETE:
        break   # proceed to Step 12

    if audit_round >= 2:
        STOP. Present to the user:
          "After {audit_round} completeness/correctness audit round(s),
           these checklist items are still missing or wrong:
           {MISSING_OR_WRONG, one line each}
           Full audit: {AUDIT_HANDOFF_PATH}
           How would you like to proceed?
           (A) I'll fix them myself — pause here
           (B) Try one more round anyway
           (C) Accept as-is and proceed to staging despite gaps
           (D) Discard this attempt"
        WAIT for the user's reply. Never silently continue past 2 rounds,
        and never silently give up.

    dispatch a FRESH fix agent (foreground, nesting depth 1/3), distinct
    from Agent C, E, and F, scoped ONLY to {MISSING_OR_WRONG}:
        TASK: Address ONLY the checklist items listed as missing/wrong in
        {AUDIT_HANDOFF_PATH} for {TICKET_KEY or TITLE} at {WORKTREE_PATH}.
        CONTEXT: read the audit handoff doc — do not ask for the gaps to
        be re-explained, do not touch items already marked
        ADDRESSED+CORRECT.
        Same repo conventions as Step 6.
        EXPECTED OUTPUT: same handoff-doc contract as Agent C.

    re-run a lighter pass of Step 10 (rebuild + smoke test + behavioral
    repro re-run only — not the full test-suite independent-verification
    pair again, since this is a targeted follow-up, not a from-scratch
    verification) to confirm the new changes build and don't regress
    anything already confirmed working.

    audit_round += 1
```

If `STATUS == MISMATCHED` (not just incomplete, but an addressed item is
actively wrong), treat it the same as `INCOMPLETE` for this loop — it still
counts toward `MISSING_OR_WRONG` and still blocks Step 12.

### Step 12 — Stage the final code for user review (no commit/push by default)

Inline, not a subagent — this is just git bookkeeping already summarized by
prior handoff docs.

1. Confirm the branch/worktree from Step 5 is still correctly checked out.
2. Run `git status --short` and `git diff --stat` in the worktree to confirm
   only the intended files changed.
3. **Confirm with the user before committing.** Show the exact commit
   message, e.g.:
   ```
   type: imperative summary

   - Verb-first bullets, one per logical change

   (no ticket/issue reference in the body — that goes in the PR description
   later, when the user opens a PR themselves)
   ```
4. On confirmation: `git commit` (no `--no-verify`, no `--amend`, and run
   whatever pre-commit/lint hooks the repo has configured first).
5. Do NOT merge to the default branch, do NOT push, do NOT open a PR. Never
   do those without a separate, explicit, later ask from the user. (feature
   mode) When the user does eventually open a PR themselves, suggest they
   link the design page ({PAGE_URL}) in the PR description.

Final message to the user:
```
(defect mode) Fix for {TICKET_KEY} is complete and committed locally.
(feature mode) Feature "{TITLE}" (design: {PAGE_URL}) is implemented and
committed locally.

Branch:   {BRANCH} (from {BASE_BRANCH})
Commit:   {SHA} — "{commit subject}"

Verification: {TEST_RESULTS summary}
(defect mode) Defect repro: confirmed fixed (before/after above)
(feature mode) Feature behavior: confirmed matches {REQUIREMENTS} (before/after above)
Review: {N} iteration(s), 0 BLOCKING findings remaining (or: user-overridden with M still open)
Independent verification: build/test claims re-confirmed by a separate agent (D', G') — MATCH
Completeness/correctness audit: {N}/{N} checklist items confirmed ADDRESSED+CORRECT
  (or: user-overridden with M still open) — full table: {AUDIT_HANDOFF_PATH}

Everything above ran inside Docker only — nothing was installed or mocked on
your host. Nothing has been pushed. Review the diff yourself
(`git -C {WORKTREE_PATH} diff {BASE_BRANCH}..HEAD`), then push to your own
branch/fork and open a PR when you're ready. I will not push or open a PR
unless you explicitly ask me to.
```

### Step 13 — If the user later explicitly asks to open a pull request

Only triggered by a separate, explicit follow-up ask — never part of the
default flow above.

- Look for the repo's own `.github/pull_request_template.md` (or
  `PULL_REQUEST_TEMPLATE.md`) and fill it in verbatim, section by section;
  only fall back to a freeform Motivation/Technical Details/Test Plan body
  if the repo has no template file at all.
- Always populate whatever section references the originating ticket/issue
  with the real key/link — never leave it blank or as the template's
  placeholder comment:
  - defect mode: reference `{TICKET_KEY}` (e.g. `Closes ABC-123` or
    `JIRA: ABC-123`, whichever form the repo's template/CI bot expects).
  - feature mode: if the design page itself names/links a ticket, use that
    key the same way. If none exists, don't fabricate one — link
    `{PAGE_URL}` in the Motivation/description section instead.
- Confirm the exact PR title, body, and target branch with the user before
  running `gh pr create` (or the repo's equivalent) — this is a shared/
  visible action, separate from the earlier "never commit/push without
  approval" gate.
