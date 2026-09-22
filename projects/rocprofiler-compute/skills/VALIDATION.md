# Agent Skill release validation

This file records release-level validation. A skill is not release-validated
until both trigger evaluation and representative GPU workflows pass.

## ROCm 10.2 / rocprofiler-compute 3.10

| Check | Profile skill | Analyze skill |
|---|---|---|
| Eval JSON parses | Pass | Pass |
| Skill frontmatter | Pass | Pass |
| Helper script syntax | Pass | Pass |
| Recommended workflow on supported GPU | Pending | Pending |
| Named fallback on supported GPU | Pending | Pending |
| PyTorch operator trace workflow | Pending | Pending |
| Generic user-authored ROCTx boundary documented | Pass | Pass |
| rocpd CSV/database report workflow | N/A | Pending |
| Tested build | Pending | Pending |
| Tested GPU/firmware | Pending | Pending |

Status: **not release-validated**. Complete the pending rows on a supported
ROCm 10.2 system before marking the skills ready for release.

## Required evidence

For each release, record:

- exact ROCm and rocprofiler-compute versions;
- GPU architecture/model, firmware reported by the runtime, and Linux
  distribution;
- trigger-evaluation results;
- commands and outcomes for the recommended and fallback workflows;
- PyTorch operator attribution result;
- generated output formats and workload validation result;
- known failures, unsupported configurations, and linked issues.

Add a new section for each release. Do not overwrite prior results.
