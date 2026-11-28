* Grouped collectives on a WarpSpeed communicator with a large channel pool (for example 224 channels) were launched
  as one kernel per collective, because each collective was packed over the whole pool and the second one no longer
  fit the kernel arguments. Each collective of a grouped WarpSpeed plan now gets its own range of whole workgroups,
  sized by its traffic, so up to three collectives share one launch.
