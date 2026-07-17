# Examples

## What the user says

**Archive triage (read-only, default)**

> Decode and triage my HRR archive at `/data/crash/capture.hrr/pid-1842`

**Existing replay log**

> Analyze this HRR replay log: `replay.log` — archive is `capture.hrr/pid-1842`

**Capture explainer only**

> What's in this HRR capture? `capture.hrr/pid-1842`

The user does not mention scripts, `HRR_PLAYBACK`, or GPU numbers.

## What the agent does

1. Resolves `pid-1842` (or largest `events.bin` if only root given)
2. Finds `hrr-playback` for `--info` (PATH → `$ROCM_PATH/bin` → user path)
3. Runs `decode_finding.sh --archive ...` (and `--log` if provided)
4. **Prints the finding summary in the chat reply** (required — user must not need to open a file)
5. Adds **capture explainer** (events.bin, blobs, completeness, version)
6. Does **not** launch full GPU replay unless the user explicitly asks

## Example chat reply (what the user sees)

```markdown
## HRR finding summary

- **Outcome**: MAF
- **Fault class**: `read_only_page_fault`
- **Kernel**: `Cijk_..._MT128x192x128_..._SK3_...`
- **Fault address**: `0x7f8a1c400000`
- **Failing call**: n/a (GPU fault during kernel execution)
- **Archive**: 13,209,290 events, Complete: NO

### Capture explainer
Crash-truncated v4 archive from pid-138: `events.bin` holds the HIP trace, `blobs/` holds
host payloads. No clean shutdown trailer — expected after a GPU memory fault. Reader recovered
all complete events; replay/analysis can proceed.
```

## Example structured output (markdown file)

```markdown
## Summary
- **Outcome**: MAF
- **Fault class**: `read_only_page_fault`
- **Kernel**: `Cijk_..._MT128x192x128_..._SK3_...`

## Fault details
- **Fault address**: `0x7f8a1c400000`
- **Failing event seq**: 13118764
- **Failing API**: n/a (GPU fault after launch)

## Archive / capture
- **Events**: 15230441
- **Complete**: NO
```

## If `hrr-playback` is not in a standard location

User answers: *"It's in `/home/me/rocm-build/hrr-playback`"*

Agent sets `HRR_PLAYBACK=/home/me/rocm-build/hrr-playback` for that session and re-runs `decode_finding.sh`.

## When the user asks for full replay

That is outside the default read-only path. Point them to [README.md](../../README.md) § Replay on GPU, or run `hrr-playback <archive>` separately, then feed the resulting log back into this skill.
