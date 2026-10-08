# High-level design

Follow **[`AGENTS.md`](../../AGENTS.md)** and the full redirect chain it references.

This skill helps the author write an HLD. Grill the whole design tree first.
Write `docs/design/hld-<topic>.md` once, after the author accepts the summary
with no changes.

## What binds

[`docs/design/hld-template.md`](../../docs/design/hld-template.md) owns the
section headings and the writing rules. Do not paste those headings into the
HLD, and do not keep a second outline here.

The template says to avoid producing an HLD with an AI tool. This skill
replaces that one line, and only after the author has confirmed every
decision. Say that to the author before the first question. Until that
confirmation, do not write the file.

[`hld-torch-trace-collector.md`](../../docs/design/hld-torch-trace-collector.md)
shows tone and length. It does not supply headings. Its `## Implementation`
stub is not a valid phase section. Headings come from the template.

## Entry

If the author has not named a topic and a problem, ask for those two things
and wait.

If a `docs/design/hld-*.md` for this topic already exists and the author has
not chosen a mode, ask whether to review that file or replace it. Review
follows the Review section. Replace follows the interview, then the write.

Before the first question, say the answering rules: the author reacts to a
recommendation, may reject any of them, and no file is written until they
accept a summary with no changes.

Record each decision the author has already stated. Do not ask it again.
Recommend only where they have not chosen.

Propose the file slug, show `docs/design/hld-<topic>.md`, and wait for the
author to accept the path. Use lowercase words separated by hyphens.

List the branches below and start with System Context. That is the root.

## Branches

Ask in this order. A later branch waits until the earlier one is answered
or explicitly deferred.

1. System Context
2. Problem statement
3. Requirements
4. Design
5. Implementation phases
6. Validation, security and debuggability
7. Open questions

These prompts are the closed set. Ask one only when the branch applies.
To skip one, name it and the reason in that round. The skip counts only
after the author agrees. The file still gets that heading, with one line
on why it does not apply. Do not invent a body for it.

- What is in scope, and what is explicitly out.
- Why the problem matters, and what breaks if it stays unsolved.
- Why this approach, and what the alternatives are. What is the worst case.
  How do you roll it back.
- What happens on failure, on a missing dependency, and at a load the author
  names in their own units (profile size, kernel count, or concurrent runs).
  If they have no number, defer "no load target".
- What you will test, and what you will log, metric, or trace.
- What data the change touches, who can invoke it, and what must not leak.
- What you are deliberately leaving open.

System Context names only components the design changes or depends on, and
facts a later decision uses. When the author names more than one functional
requirement, ask for their order and record it.

A vertical slice is one change a user can see in a named workflow: profile,
analyze, or the report. A phase that never shows up in one of those is not
a slice.

In Design, use the template's word "alternatives". Say "rejected" only for
an option the author discarded. A still-open option goes under Open questions.

## Decision log

After each settled answer, append one line to a decision log. Repeat the log
at the start of the next round so a long chat cannot drop it. The log is not
the HLD. Do not write a section when its questions end.

Each line is `author-stated` or `accepted recommendation`, plus the author's
reason. Copy an agent recommendation into the log only when the author
adopts that reason.

## Rounds

The frontier is every closed prompt for the current branch that is not yet
answered or deferred. Ask those questions in one round, at most five. Leave
the rest for the next round. Number questions continuously across rounds.

Independent questions inside one section share a round. A question that
depends on another question still open in this round waits.

```
❓ **Q1** - **<title>**: <question>

➡️ <recommendation from the author's words or a verified fact, and one line why>

---

❓ **Q2** - **<title>**: <question>

➡️ <trade-off, with no winner, when nothing verified supports a choice>
```

Look up facts in the repo before asking. Do not ask the author for anything
you can look up. Do not show a recommendation until the lookup it depends on
has finished. If a lookup fails, finds nothing, or contradicts the author,
say what you checked, park the conflict as an open question, and ask the
rest of the frontier. If a finished lookup overturns a recommendation already
shown, ask that question again.

A round stays open until every question in it is answered or explicitly
deferred. Only an explicit answer settles a question. A skipped question
returns on the next frontier.

The author reacts to a recommendation. They are not filling a blank.

If the answer is vague or "I don't know", restate the trade-off once and
ask them to decide or to defer. A second vague reply does not settle it.
Ask them to defer. An explicit deferral moves the item to Open questions
and off the frontier. Then continue.

A decision is weak when it has no reason, no alternative, or no failure
behavior. State that objection once. If the author keeps the choice, log it
with their reason and move on.

If the author tells you to write the file or to start implementation while
any prompt is still open, name the open branches and ask the next questions.
Refuse implementation in one sentence and stay in the interview.

## Summary and write

The frontier is empty when every closed prompt is answered or deferred, and
nothing is left assumed. In the summary, list:

1. Each decision, labeled `author-stated` or `accepted recommendation`, with
   the author's reason.
2. Every assumption you used. The author accepts or corrects that list.
3. What is out of scope, including agreed skips.
4. Open questions: unverified facts, conflicts, and explicit deferrals.

Ask whether that summary is right.

Any correction, including "yes, except ...", is not acceptance. Do not write.
Reopen only the decisions the author named, drop recommendations that
depended on them, run another round, summarize again, and require a new
yes with no changes.

After that yes, write `docs/design/hld-<topic>.md` once. If the file exists,
stop and ask whether to review it or replace it. Do not overwrite on the
same yes that accepted the summary.

The write uses every template heading, in template order. Confirmed
decisions and their alternatives go in Design. Unverified facts and
deferrals go under Open questions only. A number that is not in the code,
the docs, or the author's answer is an open question. Do not invent it.

Do not resume the interview between sections.

After the write, ask the author to accept the file or name one section to
change. Change only that section, from the confirmed decisions.

## Review

Review runs only when the author asks to review an existing HLD, or chooses
review at entry.

Grill the gaps with the same question format: unresolved decisions, missing
alternatives, and claims with no why. The ban on writing during the
interview does not apply here. Rewrite a section only when the author asks
for that section, even if other gaps stay open. Stop when they say to stop,
or when that edit is done.

## Worked session

Shape only. Not a real design, and not facts about the tree.

Branches: System Context (start), Problem statement, Requirements, Design,
Implementation phases, Validation, Open questions.

❓ **Q1** - **Scope**: Should this HLD cover only the analyze error when
`--path` is not the directory that holds the profile config?

➡️ Limit it to analyze. Profile already reports its own path failure, so
pulling profile in adds a second workflow with no new decision.

Deferred, not asked: which component formats the error. That waits until
the problem statement says what is wrong today.

Author: "1 yes."

Later, on the failure-behavior prompt, the author says "I don't know."
Restate that prompt once. The author still will not choose. Defer it to
Open questions and continue. Log: `accepted recommendation` — scope is
analyze only. Author's reason: profile already has its own error.

Summary, after the other prompts are answered or deferred:

- `accepted recommendation` — scope is the analyze path error. Author's
  reason: profile already reports its own.
- Assumption: the author meant the directory that contains the profile
  config. Author must accept or correct this.
- Out of scope: profile mode.
- Open: who formats the error. No load target.

"Yes, except also cover profile" is not acceptance. Reopen scope, summarize
again. Write only after a later yes with no changes.

## Done when

- The HLD file does not exist before the author accepts a summary with no
  changes.
- Every template heading is present. A skipped branch is one line on why it
  does not apply.
- Each decision in the file is in the decision log, with the author's reason.
- Numbers and unverified claims appear only under Open questions.
- A partial yes did not create or overwrite the file.
