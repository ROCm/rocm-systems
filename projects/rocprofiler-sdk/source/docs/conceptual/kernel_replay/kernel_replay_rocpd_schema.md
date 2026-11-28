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
1536. It now sums the hardware instances of one pass, then averages over the passes that collected
the counter, so this reports 512. A counter collected on only some passes is averaged over those
passes. A pass with no row for the counter, such as a pass dedicated to another service, is not in
the denominator. A run that never replayed leaves every row untagged, so the view still sums
hardware instances.

The individual readings stay in `rocpd_pmc_event`. A row from a pass after the first carries
`replay_pass` in `extdata`, and a row without it belongs to pass 0. The `pmc_events` view reports
that index. This is rocpd schema 3.0.6 and later.

`counters_collection` stays one row per dispatch per counter. That is the file most people read.
Keeping a row per pass, with its own timestamps, would be a larger schema change, and it is not
what this view is for. One default cannot serve both.
