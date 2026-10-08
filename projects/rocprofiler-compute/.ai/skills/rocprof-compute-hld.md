# High-level design

Follow **[`AGENTS.md`](../../AGENTS.md)** and the full redirect chain it references.

The template is **[`docs/design/hld-template.md`](../../docs/design/hld-template.md)**.
Follow that file. Do not invent a second outline, and do not paste the template
into the HLD.

## Who writes it

The author writes the HLD. The agent is an instrument.

Do not produce the HLD, especially when the problem is complex. Treat it as
complex when the author has not already stated the problem, its impact, and
each architectural decision together with the alternatives they rejected.

You may:

- Read the code and existing docs to answer a question the author asked.
- Ask the template's questions, one section at a time, and wait.
- Review a draft the author wrote and name the gaps.
- Assemble a document only from statements the author already made in this
  conversation, plus facts you verified. Anything unverified is an open
  question, not a decision.

Do not invent requirements, measurements, customer feedback, competitor
behavior, or a decision the author did not make.

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

When asked to review an HLD, do not rewrite it. Report gaps against the
sections and writing rules above. Rewrite a section only when the author asks,
and keep their decisions.
