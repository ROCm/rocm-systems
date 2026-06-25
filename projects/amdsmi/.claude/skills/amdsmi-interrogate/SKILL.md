---
name: amdsmi-interrogate
description: "Use whenever amd-smi work begins from a design — a Confluence Feature Design, a Jira/SWDEV ticket, a driver hand-off, or a user description. Reconciles the design against what the code actually does, questions it adversarially, and produces an approved spec. This is the single front door to any feature or behavior change."
---

# Interrogate — amd-smi

The front door for any feature or behavior change. A design arrives from somewhere
upstream; your job is to **turn it into an implementable, contradiction-free spec**
before a line of code is written. amd-smi work starts from a design, and this skill
handles it — including the rare case where the design must be generated from scratch.

**Announce at start:** "I'm using the `amdsmi-interrogate` skill."

**The governing principle:** *the design is a hypothesis; the code is ground truth.*
amd-smi Feature Designs routinely drift from what actually shipped. Never trust a
design's claim about current behavior — verify it against the source, and treat every
gap between the two as something to resolve, not assume.

<HARD-GATE>
No source edits, no wrapper/CLI bindings, no implementation skill until a written
spec exists and the user has explicitly approved it. Even a "trivial" change gets a
spec — three sentences is fine, but it gets written and approved.
</HARD-GATE>

## Knowledge Lives Here (agent-owned, not public docs)

- **Glossary:** `.claude/context/CONTEXT.md` — canonical domain terms. The only
  agent-owned file checked into the repo.
- **Specs:** ephemeral. Write to `${TMPDIR:-/tmp}/amdsmi-agent-specs/YYYY-MM-DD-<topic>-design.md`.
  Specs are scratch artifacts for the current session's hand-off chain; they are
  deliberately **not** committed and disappear with the temp dir.

## 1. Source the Design

Pull the upstream material before asking the user anything you could read yourself:

- **Confluence AMDSMI space** — via the `confluence-atlassian-cloud` MCP server when
  available. Designs live under the AMD SMI space, typically the
  **AMD SMI Feature Designs** page tree
  (`amd.atlassian.net/wiki/spaces/AMDSMI/`). Treat these as the upstream intent —
  and as possibly stale.
- **Jira / SWDEV ticket** — via `jira-atlassian-amd-hub` when available.
- **Driver hand-off or user prose** — whatever was pasted.

If the MCP servers aren't reachable, ask the user to paste the page/ticket rather than
guessing its contents. Record the Confluence page and/or ticket id in the spec for
traceability. Read `CONTEXT.md` first for any pinned terms the design uses.

If there is genuinely **no upstream design**, switch to generate mode: walk the user
through the problem one question at a time, propose 2–3 approaches with a recommendation,
and converge on a design — *then* interrogate it with the rest of this skill.

## 2. Establish Ground Truth

Before interrogating, read the actual code in every cascade layer the design touches
(`project-layout` rule maps them: header → `amd_smi.cc` → wrapper → interface → CLI →
docs). Write down where the design and the code **already disagree** — wrong return
codes, renamed fields, a flag the design assumes but the code doesn't have. These
drift points are your sharpest interrogation material.

## 3. Interrogate (one question at a time)

Adversarial by design. Ask **one** question, wait for the answer, then follow the
dependency it exposes. Give your own recommended answer each time. If the code can
answer it, read the code instead of asking. Attack along these axes:

- **Drift** — "The design says this returns a partial value; `amd_smi.cc` returns
  `AMDSMI_STATUS_INVAL`. Which is the intended behavior?"
- **Necessity / scope** — Is each piece required by the ticket, or gold-plating?
  What's the smallest version that satisfies the upstream need?
- **Terminology** — Pin overloaded words to one meaning, checked against `CONTEXT.md`
  ("device" → BDF, processor handle, or socket?). Flag glossary conflicts.
- **Cascade reach** — Does the value actually have to surface at the CLI / JSON layer,
  or stop deeper? Which layers genuinely change?
- **Edge cases** — Invent concrete scenarios (partitioned GPU, missing sysfs node,
  multi-socket) that force precision.

If the request spans multiple independent subsystems, stop and decompose — each gets
its own intake → spec → plan.

## 4. Pin the amd-smi Specifics

The spec is not done until every row is answered:

| Concern | Question |
|---------|----------|
| **Cascade impact** | Which layers change: header → `amd_smi.cc` → wrapper → interface → CLI → docs |
| **Backward compat** | Public C API is a contract — additions are safe, signature changes break ABI |
| **Error path** | New `amdsmi_status_t`? Mapping to `AmdSmiException`? |
| **Test plan** | Which suites: C++ GTest, Python unit/integration, CLI |
| **Hardware dependency** | Requires GPU? Specific ASIC? Partition mode? Test environment? |
| **Build / packaging** | New CMake option? New install file? RPM + DEB both? |
| **Changelog** | Entry for `CHANGELOG.md` under the next version |

## 5. Record What You Learned

- **`CONTEXT.md`** — add a resolved term the moment it's pinned; don't batch.
- **Spec** — write to `${TMPDIR:-/tmp}/amdsmi-agent-specs/YYYY-MM-DD-<topic>-design.md`,
  including the upstream source link and any design-vs-code drift you reconciled.
  `mkdir -p` the directory first. Self-review (placeholders, contradictions, scope,
  ambiguity, cascade completeness), fix inline, then ask the user to approve.
  The spec is session-scoped scratch; do not commit it.

## 6. Hand Off

After the user approves the spec, invoke `writing-plans`. Invoke no other skill from
here, and write no production code.

## Knowledge-Extraction Mode

When there's no design and the user just wants to build out `CONTEXT.md`, run steps 2
and 3 only: interrogate a subsystem, cross-check against the code (and the AMDSMI
Confluence space for upstream intent), and record resolved terms as you go.

## Red Flags — STOP

- Trusting the design's description of current behavior without reading the code
- "Too simple to need a spec"
- Jumping to "what file do I edit"
- Bundling multiple questions into one message
- Proposing one approach (in generate mode) without alternatives
- Skipping the user approval gate
