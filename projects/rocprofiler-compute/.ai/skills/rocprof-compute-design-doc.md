# Write a design document

Follow **[`AGENTS.md`](../../AGENTS.md)** and the full redirect chain it references.

## When to use this

Use this to write a design document under `docs/design/`. A high-level design (HLD)
states the problem and the chosen design. A low-level design (LLD) says how to build
that design and splits the work into PRs.

## Steps

1. **Ask which document to write.**
   - Ask the user: "Do you want to write an HLD or an LLD?"
   - For an LLD, ask for the HLD that it builds on. If there is none, ask whether to
     write the HLD first.

2. **Read the template and the writing rules.**
   - HLD: [`docs/design/hld-template.md`](../../docs/design/hld-template.md)
   - LLD: [`docs/design/lld-template.md`](../../docs/design/lld-template.md)
   - Read [`writing-style.md`](../rules/writing-style.md) and the section it points to.
   - Read one existing document of the same kind in `docs/design/` to see the level of
     detail.

3. **Ask for the technical requirements.**
   - HLD: ask for the problem, why it matters, the functional and non-functional
     requirements, and any constraints.
   - LLD: ask which HLD phases it covers, and for any requirements that the HLD leaves
     open.

4. **Research.**
   - Read the code and the docs that the design touches, and the related documents in
     `docs/design/`.
   - HLD: find out how the system works today and which designs could meet the
     requirements. Look at how other tools solve the same problem when that helps.
   - LLD: find the modules, data models, and interfaces that change, and a way to split
     the work into PRs.
   - Do not invent requirements, measurements, or decisions.

5. **Present the findings and align with the user.**
   - Present in the chat, not in a file: the requirements as you understand them, how the
     system works today, the options and their trade-offs, a recommendation, and the open
     questions. For an LLD, also present an outline of the PR plan.
   - Apply the user's feedback. Research again where the feedback needs it, and present
     the updated proposal.
   - Repeat until the user approves the proposal. Do not write the design document
     before that.

6. **Write the draft.**
   - Save it as `docs/design/hld-<slug>.md` or `docs/design/lld-<slug>.md`, where the
     slug names the feature, for example `hld-torch-trace-collector.md`.
   - Follow the section order of the template. Replace the guidance bullets of the
     template with the content of the design.
   - If a section does not apply, keep its heading and say so in one line.

7. **Check the draft.**
   - Every sentence follows the writing rules.
   - Each decision says why, and names the alternative that it was chosen over.
   - Claims about speed or size come with a measurement or an estimate.
   - An HLD names no files, classes, or functions, because those names change fast.
   - In an LLD, each PR in the PR plan says what it changes, what it depends on, and how
     it is tested.

8. **Show the draft to the user.** Apply their feedback. Commit only when the user asks.
