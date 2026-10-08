# High-level design

Follow **[`AGENTS.md`](../../AGENTS.md)** and the full redirect chain it references.

The template is **[`docs/design/hld-template.md`](../../docs/design/hld-template.md)**.
Follow that file. Do not invent a second outline, and do not paste the template
into the HLD.

## Who writes it

The author owns every decision. The agent grills, looks up facts, and writes
the document only after the author confirms a shared understanding.

Do not invent requirements, measurements, customer feedback, competitor
behavior, or a decision the author did not make. Anything unverified is an
open question.

## Grill

Interview before writing. The output of the interview is a resolved design
tree, not a draft.

Map the HLD as a design tree. Every decision branches into the decisions that
hang off it. The branches follow the template, and later branches wait on
earlier ones:

1. System Context, Problem statement, and Requirements.
2. Design decisions, each with the alternatives rejected.
3. Implementation phases, validation, and what stays open.

The frontier is every decision whose prerequisites are already settled: the
questions you can ask now without guessing at an answer you have not heard.
Ask the whole frontier in one round. Number each question and give a
recommended answer. Then wait. A question that depends on another question
still open in this round belongs to a later round.

```
❓ **Q1** - **<title>**: <question>

➡️ <recommended answer, and one line why>

---

❓ **Q2** - **<title>**: <question>

➡️ <recommended answer, and one line why>
```

Before the first round, list the branches you will walk and say which one
you start with. That is the branch the others depend on.

Facts are yours. Read the code and the docs. Do not ask the author for
anything you can look up. A lookup still running blocks only the questions
that depend on it. Ask the rest of the frontier now.

Decisions are the author's. They are reacting to a recommendation, not
filling a blank. If the answer is vague or "I don't know", give the
trade-off and the recommendation again, and ask what it looks like in
practice. Do not treat that as settled.

Work these in where they apply. Skip a branch that this design does not have.

- What is in scope, and what is explicitly out.
- Why the problem matters, and what breaks if it stays unsolved.
- Why this approach, and not the alternative. What is the worst case. How
  do you roll it back.
- What happens at the edges: failure, a missing dependency, much more load.
- What you will test, and what you will log, metric, or trace.
- What you are deliberately leaving open.

Be a direct technical partner. Find the weak decision. Short sentences. No
filler, and no praise before the next question. Do not write the HLD, and
do not start implementation, during the interview.

When the frontier is empty and nothing is silently assumed:

1. Summarize each decision and why.
2. List what is out of scope and what stays open.
3. Ask whether that shared understanding is right.

Write `docs/design/hld-<topic>.md` only after the author says it is. Use
their decisions and the facts you verified. Leave the rest under Open
questions.

## Order

System Context, Problem statement, and Requirements come before Design. The
design is only as good as those three. Do not draft Design until they say what
is in scope, which problem it solves, and why that problem matters.

## Where it goes

Write it to `docs/design/hld-<topic>.md`.
[`hld-torch-trace-collector.md`](../../docs/design/hld-torch-trace-collector.md)
is a short example of the shape. Match that shape. Do not copy its content.

## Writing rules

- Say why: why it is a problem, why this decision, why this requirement.
- Be short. Cut filler and new jargon. Keep each decision in one place.
- Use a diagram when a flow or a boundary is hard to see in prose.
- Support a claim with a measurement, an estimate, or a stable reference such
  as a CLI flag, a public behavior, or a doc. Do not cite file paths, line
  numbers, variable names, or class names. They go stale.
- Use lists, tables, and short paragraphs to make a point. Skip formatting
  that only adds noise.

## Sections

Use the template's sections, in this order:

1. **System Context.** Purpose of the part in scope, the class of problem it
   solves, and what it does not cover. Main components and the technical
   details that matter. Surrounding components. Customer requests, competitor
   solutions, assumptions, and constraints, when they change the design.
2. **Problem statement.** The problems this design solves, why they matter,
   and their impact.
3. **Requirements.** Functional requirements, prioritized when the list is
   long. Non-functional requirements. Guidelines the design must follow, each
   with a reason.
4. **Design.** The proposed design. Each architectural decision says why it
   was chosen and which alternatives were rejected.
5. **Implementation phases.** Vertical slices. Each phase crosses the stack
   and delivers something a user can use.
6. **Validation, security and debuggability.** Which unit, functional, and
   integration tests are required, and the validation strategy. Logging,
   metrics, alerting, or tracing that reduces maintenance cost.
7. **Open questions.** Known unknowns, deferred decisions, and trade-offs.

## Review

When asked to review an HLD, do not rewrite it. Grill the gaps: unresolved
decisions, missing alternatives, and claims with no why. Rewrite a section
only when the author asks, and keep their decisions.
