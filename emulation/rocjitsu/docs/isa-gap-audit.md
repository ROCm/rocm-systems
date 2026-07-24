# ISA Gap Audit Guide

Use this agent workflow when auditing gaps between AMDGPU ISA handbooks, the
machine-readable ISA XML, and rocjitsu's generated decoder and execution
support. The goal is to find semantic information that is missing from
structured inputs or missing from rocjitsu.

## Inputs

- ISA handbook prose for the target architecture (for example, a PDF or
  equivalent Markdown).
- The matching checked-in MR ISA XML under
  `shared/machine-readable-isa/isa/`.
- Rocjitsu generated and handwritten ISA support under
  `emulation/rocjitsu/lib/`.

## Audit workflow

1. Compare handbook prose to MR ISA XML, one small section at a time. Record
   fields, flags, semantic tables, register bits, selector behavior, instruction
   notes, and hardware-state-dependent behavior that are present in the handbook
   but not inferable from XML.

2. Compare rocjitsu behavior to the handbook, using the handbook-vs-XML notes as a
   starting point. Check decoders, operand classes, generated execute bodies,
   helper functions, tests, and handwritten architecture hooks. This pass can
   expose both XML gaps and rocjitsu-specific modeling gaps.

3. Format, condense, sort by priority, and remove obvious false positives.

As of Codex GPT-5.5, auditing all architectures takes approximately one day.

## Codex agent prompt example

The following prompt is a reusable starting point for an agent-assisted audit:

### Step 1

```text
/goal Audit rocjitsu ISA decode and generated components for gaps against ISA
handbook prose.

For each target architecture, compare each small handbook section against the
matching MR ISA XML. Record information that appears in the handbook but is not
inferable from XML, such as missing flags, undocumented encoding bits,
descriptor or mode bits, semantic tables, selector exceptions, and
hardware-state-dependent behavior.

Then audit the corresponding rocjitsu decoder, generated execute code, helper
functions, operand handling, and tests against the handbook. Work chapter by
chapter or instruction family by instruction family. Prioritize careful
cross-references over speed. Subagents may be used when the section boundaries
are clean.

## Deliverables

Produce two markdown reports per architecture:

- `<arch>-handbook-vs-xml.md`: handbook information not inferable from XML.
- `<arch>-rocjitsu-gaps.md`: gaps between rocjitsu behavior and the handbook.

Split each report by ISA chapter, instruction family, or another stable section
that makes follow-up patches easy to scope. Each finding should include the
handbook location, XML location or absence, rocjitsu file or test references when
applicable, and a short note about confidence or validation.

## Architecture Order

1. RDNA4
2. CDNA4
3. CDNA3
4. RDNA3
```

### Step 2

This step should be done with a different agent. The sample below will not
remove any flagged incorrect findings on its own.

```text
/goal: Audit and Condense SOURCE_REPORT

For each finding in: SOURCE_REPORT

1. Verify the finding against provided XML and PDF/MD.
   After verification, select one primary DISPOSITION from this controlled
   vocabulary:
   - **Confirmed.** A valid, in-scope XML/handbook gap or conflict.
   - **Scope caveat.** Real handbook-only information that is internal-only,
     hardware-only, compiler guidance, performance guidance, host/runtime ABI,
     microarchitecture, or otherwise outside ordinary instruction-XML scope.
   - **Qualified.** The sources differ, but authority, legality, naming, product
     scope, or intended behavior is unresolved.
   - **Incorrect.** The source pair does not support the original claim.

2. Rewrite each entry into exactly this shape:

   ### GFXNNN-XML-NNN: Title
   - Verdict: **DISPOSITION.** Concise evidence-backed conclusion.
   - Source: Exact handbook section/table/page and XML records.
   - Fails: Concrete consequence, or “No demonstrated gap.”
   - Need: Required correction, clarification, or “None.”

3. Run a completion audit proving:
   - all original IDs are represented once;
   - every entry has one Verdict, Source, Fails, and Need field;
   - citations point to the correct handbook pages;
   - concrete inventory claims match the exact archive member;
   - unsupported and unclear claims are visibly classified;
   - whitespace validation passes.

## Deliverables

Write the result to:

CONDENSED
```
