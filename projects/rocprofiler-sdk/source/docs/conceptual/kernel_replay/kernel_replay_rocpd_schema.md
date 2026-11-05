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
1536. It now reports `AVG`, which is 512. A counter collected in only one group is unchanged,
because the average of one value is that value. A run that never replayed is unchanged for the
same reason.

The three readings stay in `rocpd_pmc_event`. A query against that table still lists 512, 512, and
512. For a deterministic counter those readings are the same number, so the average is exactly
what each pass collected.

`counters_collection` stays one row per dispatch per counter. That is the file most people read.
Keeping a row per pass, with its own timestamps, would be a larger schema change, and it is not
what this view is for. One default cannot serve both.
