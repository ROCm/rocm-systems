---
name: pressure-testing-skills
description: "Use when validating or hardening a SKILL.md, prompt, rule, or any agent-followed document — when you need proof it actually works, not just that it reads well. Runs a fresh subagent that follows the doc literally on a known-answer fixture, captures friction, and iterates to determinism before minimizing. Use after writing-skills produces a draft, or whenever a skill 'looks right' but hasn't been proven under an agent."
---

# Pressure-Testing Skills — amd-smi

Empirically prove a skill (or prompt/rule/checklist) works by making a fresh
subagent follow it **literally** on a fixture whose answers you already know, then
iterate on the failures until the run is deterministic. Reading a skill tells you
it's plausible. Only a literal run by a naive agent tells you it's correct.

**Iron Law: A SKILL IS NOT DONE UNTIL A FRESH SUBAGENT, FOLLOWING IT VERBATIM ON
A KNOWN-ANSWER FIXTURE, CATCHES EVERY PLANTED ISSUE WITH ZERO GUESSING AND NO
BLOCKING FRICTION — WITHOUT INVENTING STEPS THE SKILL DIDN'T GIVE.**

If the agent got the right answer by adding its own check, the skill failed —
that check belongs in the skill. If it "ran clean" but missed a planted issue,
the skill failed. If it had to guess which value to use, the skill failed.

**REQUIRED BACKGROUND:** you must understand `writing-skills` (structure, CSO,
token budget) and `dispatching-parallel-agents` (how to run subagents) first.

## Why Literal + Naive + Known-Answer

The whole method rests on three constraints — drop any one and the test lies:

| Constraint | Drop it and… |
|-----------|--------------|
| **Literal** — agent runs the documented commands verbatim, invents nothing | A clever agent papers over gaps with its own reasoning; you ship a skill only experts can follow |
| **Naive** — fresh subagent, no prior context, no domain memory | Your own knowledge leaks in and the skill looks clearer than it is |
| **Known-answer** — fixture with a pre-written answer key | You can't tell "caught everything" from "got lucky / ran clean" |

## The Loop

```
0. Build the fixture + answer key (the hardest and most important step)
1. Snapshot the skill to a stable temp path  (vN)
2. Dispatch a FRESH, correctly-tooled subagent — follow vN verbatim, invent nothing
3. Collect: issues-caught-vs-answer-key  AND  friction log
4. Fix the ROOT CAUSE of each miss / friction  → snapshot v(N+1)
5. Repeat 2–4 until GREEN (stop condition below)
6. Minimize: cut redundancy, re-test after EACH cut, revert any regression
```

### 0. Fixture + answer key

You cannot grade a test with no answer key. Before iterating, write down every
issue the fixture contains and the exact evidence that should surface it.

- **Best fixture:** a real artifact with known defects (e.g. a `CHANGELOG.md`
  on `develop` you already audited by hand). Real fixtures expose real ambiguity.
- **Answer key:** a list like "F1: entry X is misfiled → commit `abc` merged after
  pin `def`; F2: `### Fixed` is a disallowed heading; F3: deprecation under the
  wrong section." Grade every run against it.
- A skill with no fixture cannot be pressure-tested. If you can't build one, the
  skill is probably too vague to be useful — fix that first.
- **Prefer the LIVE state; historical replays carry time-anchored traps.** If the
  skill resolves any bound from "now" (current tags, `origin/develop`, HEAD), a
  checked-out *past* state gives meaningless results — the bound is anchored to the
  present, not the fixture's era. In one run, auditing a pre-release commit bounded
  a section by a pin that didn't exist yet, and the untagged-release fallback
  (`origin/develop`) pointed at *today's* tip, so the "too new" check could never
  fire and the audit ran clean while proving nothing. Use the live artifact as the
  fixture, or freeze the time-relative inputs too — otherwise you test a mirage.
- **Grade coverage against what the real task touches, not what the skill looks
  at.** If the fixture's defects span the whole file but the skill only inspects
  the top section, a literal run reports "clean" and is structurally blind to most
  of the answer key. A scope mismatch between skill and task is itself a finding.

### 1. Snapshot to a stable path

Copy the skill under test to a temp path (`/tmp/<skill>-test/SKILL-vN.md`) and
point the subagent at ONLY that path. Never point it at the file you're editing
or an installed copy — a stale `.claude/skills/` copy will silently override your
work. Tell the agent explicitly: "ignore any installed copy; use only this path."

### 2. Dispatch a correctly-tooled, fresh subagent

- **Tools must match what the skill's commands need.** A skill full of `git`/`gh`
  shell commands tested by a no-terminal agent proves nothing — it will only catch
  surface issues (headings, wording) and give false confidence on the real logic.
  Use a terminal-capable general subagent for shell-based skills.
- **Fresh context every iteration** so prior runs don't inflate apparent clarity.
- Give it the fixture setup, the snapshot path, and the literal-follow rules.

### 3. Collect two distinct signals

Grade every run on both — they fail independently:

| Signal | Question | A gap here means |
|--------|----------|------------------|
| **Coverage** | Did it catch every answer-key issue, using only documented commands? | The skill is missing a check → add it |
| **Determinism** | Did it have to guess, interpret, or invent anything? | The skill is ambiguous → spell it out |

"Ran clean with no errors" is **not** success. A command can exit 0, leak a
swallowed traceback, and still produce a plausible-but-wrong bound. Ask for the
friction log: every command that failed, was ambiguous, leaked an error, or forced
a guess — quoted with its **exact command and actual output**. The friction log,
not the verdict, is what drives the next iteration.

**Absence of output is not proof.** A clean pass and a silently-broken run look
identical — both print nothing. A check that greps a section and finds no problems,
and a check that ran on *zero* input (wrong path, empty range, swallowed error),
are indistinguishable from the outside. Make the skill emit a positive count
("N items checked, 0 flagged"), and have the test agent confirm the check ran on
real input, not an empty set, before trusting a clean result.

### 4. Fix root causes, not symptoms

Each friction item points at a real defect. Fix the cause:

| Real example from a changelog-skill loop | Root-cause fix |
|------------------------------------------|----------------|
| Audit loop checked only the lower bound → false "all clear" | Check both bounds of the range |
| `<V>`=`7.14.0` built tag `therock-7.14.0` (404) → silent `origin/develop` fallback | Document that tags drop the patch component |
| `2>/dev/null` on `gh` didn't cover the `python3` in the pipe → leaked traceback | Move error handling to cover the whole pipe |
| Agent doubted a correct verdict and nearly overrode it | Add the *why* so a skeptic trusts it (persuasion, not just rules) |
| A clean placement audit and one that ran on zero commits looked identical | Emit a positive count ("N checked, 0 flagged") so a real pass is distinguishable |
| Skill inspected only the top section; the task spanned the whole file | Match the skill's scope to the task, or make the scope an explicit choice |
| Every rule was prose; the agent had to hand-derive a regex for each check | Give an enforced rule a copy-paste command/regex, not just a description |
| A shortcut grep covered 5 of the 7 layers the doc listed, but read as complete | Make the shortcut cover every item the doc enumerates, or name what it skips |
| "When X, do Y" with X ("changes to the public API") undefined → agent over-applied it | Define the trigger precisely; give the boundary cases (function vs enum vs comment) |
| Fixture body was truncated with `...`, so whole body-rule classes couldn't be graded | Give the agent the COMPLETE artifact, never a paraphrase or excerpt |

Silent-failure traps (`|| fallback`, `2>/dev/null`) are the most dangerous:
they turn a wrong answer into a confident one. Make failures loud or document them.

### 5. Stop condition (loop for REAL improvements, optimally)

Stop when a literal run **catches every answer-key issue, with zero guessing, no
blocking friction, and no invented steps.** Not before — "ran clean" is a trap.
Not after — once it's deterministic and complete, more iterations just gold-plate.
Each iteration must fix a concrete failure a subagent actually hit; if you can't
name the failure, stop.

### 6. Minimize last, and guard against regressions

Only after GREEN, cut for size — and **re-test after every cut**, because a cut or
a "cleaner" rewrite can regress:

- Cut redundant tables, checklists, and restated conventions.
- Self-verify cheap mechanical changes locally (run the new regex/command on the
  fixture) before spending a subagent.
- **A proven heuristic beats theoretical completeness.** In one loop a "more
  correct" regex added false positives on the real fixture; the terse original had
  zero. Reverting was the fix. Validate every change against the fixture; revert
  anything that regresses, even if it looks smarter.

## Test-Agent Prompt Template

Reuse this skeleton for every iteration (fill the bracketed parts):

```
Follow this skill LITERALLY and run its commands verbatim. Do NOT use prior
knowledge of [domain]. Do NOT add checks the skill doesn't instruct.

Skill under test (ONLY source of truth): /tmp/<skill>-test/SKILL-vN.md
Ignore any copy at .claude/skills/... — use only the path above.

Setup: [worktree / fixture commands]
Task:  [what to audit/produce using only the skill's documented procedure]

Report back:
1. Judgment calls — did you have to guess or interpret ANYTHING? (KEY QUESTION)
2. Every issue found: exact text, responsible commit/line, which skill step
   flagged it, the fix.
3. Friction log — every command that failed, leaked an error, produced wrong
   output, or was ambiguous. Quote exact command + actual output. Say so
   explicitly if it ran clean with zero guessing. (MOST IMPORTANT)
4. Verbatim key outputs.
5. Exact commands run, trimmed. Then clean up: [worktree remove].
Read-only. Do NOT edit the fixture or the skill.
```

The "KEY QUESTION" and "MOST IMPORTANT" labels matter — without them agents
report a tidy verdict and bury the friction that you actually need.

## Common Mistakes

| Mistake | Why it breaks the test |
|---------|------------------------|
| Testing a shell-heavy skill with a no-terminal agent | Only surface issues surface; core logic untested → false pass |
| No fixture / no answer key | Can't distinguish "caught everything" from "ran clean" |
| Pointing the agent at the file you're editing or an installed copy | Stale copy silently overrides; you test the wrong text |
| Accepting the verdict, skipping the friction log | You miss the ambiguities that make the next reader fail |
| Treating "exit 0 / no errors" as success | Swallowed failures produce confident wrong answers |
| Letting the agent invent its own checks and calling it a pass | The check isn't in the skill — the next agent won't have it |
| Minimizing before it's correct, or not re-testing each cut | You shrink a broken skill, or a cut regresses silently |
| Looping forever / gold-plating after GREEN | Diminishing returns; stop when deterministic and complete |
| Using a historical/checked-out-past fixture for a skill with "now"-anchored bounds | Time-relative inputs resolve to the present → clean-but-meaningless run |
| Trusting empty output as a pass | Clean pass and silently-broken run are indistinguishable without a positive count |
| Grading only what the skill inspects, not what the task touches | Scope mismatch hides most of the answer key behind a "clean" verdict |
| Feeding the agent a truncated or paraphrased fixture | Rules that need the full artifact (body wording, trailing lines) can't be graded |
| Assuming a stated rule is enforceable without a command | Prose-only rules force the agent to invent a check; two agents check differently |
| Trusting a shortcut check that lists fewer items than the doc | It reads as complete while silently skipping layers the doc enumerated |
