# ISA Gap Audit Guide

Use this agent workflow when auditing gaps between AMDGPU ISA manuals, the
machine-readable ISA XML, and rocjitsu's generated decoder and execution
support. The goal is to find semantic information that is missing from
structured inputs or missing from rocjitsu.

## Inputs

- ISA manual prose for the target architecture. (IE pdf or equivalent markdown)
- The matching checked-in MR ISA XML under
  `shared/machine-readable-isa/isa/`.
- Rocjitsu generated and handwritten ISA support under
  `emulation/rocjitsu/lib/`.

## Audit Workflow

1. Compare manual prose to MR ISA XML, one small section at a time.
   Record fields, flags, semantic tables, register bits, selector behavior,
   instruction notes, and hardware-state-dependent behavior that are present in
   the manual but not inferable from XML.

2. Compare rocjitsu behavior to the manual, using the manual-vs-XML notes as a
   starting point.
   Check decoders, operand classes, generated execute bodies, helper functions,
   tests, and handwritten architecture hooks. This pass can expose both XML gaps
   and rocjitsu-specific modeling gaps.

This will take around a day for all archs, as of codex gpt-5.5

## CODEX Agent Prompt

The following prompt is a reusable starting point for an agent-assisted audit:

```text
/goal Audit rocjitsu ISA decode and generated components for gaps against ISA
manual prose.

For each target architecture, compare each small manual section against the
matching MR ISA XML. Record information that appears in the manual but is not
inferable from XML, such as missing flags, undocumented encoding bits,
descriptor or mode bits, semantic tables, selector exceptions, and
hardware-state-dependent behavior.

Then audit the corresponding rocjitsu decoder, generated execute code, helper
functions, operand handling, and tests against the manual. Work chapter by
chapter or instruction family by instruction family. Prioritize careful
cross-references over speed. Subagents may be used when the section boundaries
are clean.

## Deliverables

Produce two markdown reports per architecture:

- `<arch>-manual-vs-xml.md`: manual information not inferable from XML.
- `<arch>-rocjitsu-gaps.md`: gaps between rocjitsu behavior and the manual.

Split each report by ISA chapter, instruction family, or another stable section
that makes follow-up patches easy to scope. Each finding should include the
manual location, XML location or absence, rocjitsu file or test references when
applicable, and a short note about confidence or validation.

## Suggested Architecture Order

1. RDNA4
2. CDNA4
3. CDNA3
4. RDNA3
5. Other architectures
```
