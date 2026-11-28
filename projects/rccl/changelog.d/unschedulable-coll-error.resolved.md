* Fixed a hang in group launch when a collective needs more work batches than the kernel arguments can hold. The launch now fails with `ncclInvalidUsage` and a warning.
