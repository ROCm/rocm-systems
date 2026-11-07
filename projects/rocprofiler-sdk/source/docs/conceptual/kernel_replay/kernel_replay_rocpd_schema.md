# Kernel replay in the rocpd schema

Kernel replay executes one dispatch once per counter group. Every pass reports the same
`dispatch_id`. rocpd stores every pass's counter readings as `rocpd_pmc_event` rows on that
dispatch's event. `counters_collection` used to `SUM` those rows, so a sanity counter repeated in
three groups was reported three times too high.

```bash
rocprofv3 --pmc SQ_WAVES,SQ_INSTS_VALU,GRBM_COUNT \
          --pmc SQ_WAVES,SQ_INSTS_VALU,GRBM_GUI_ACTIVE \
          --pmc SQ_WAVES,SQ_INSTS_VALU,SQ_INSTS_SALU \
          --replay-mode kernel --kernel-replay-beta-enabled -- ./app
```

On a kernel that launches 512 waves, each pass reads `SQ_WAVES` as 512. The view used to report
1536. Identical readings now report that value once, so this reports 512. Readings that are not
identical, such as hardware instances of one counter, are still summed. A counter collected once
is unchanged. A run that never replayed is unchanged when each counter name is collected once.

The three readings stay in `rocpd_pmc_event`. A query against that table still lists 512, 512, and
512. For a deterministic counter those readings are the same number, so the view reports
that number instead of adding the copies.

`counters_collection` stays one row per dispatch per counter. That is the file most people read.
Keeping a row per pass, with its own timestamps, would be a larger schema change, and it is not
what this view is for. One default cannot serve both.
